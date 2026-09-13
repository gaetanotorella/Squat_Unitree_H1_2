from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


def command_gate_threshold(env, command_name: str, threshold: float = 0.1):
    """Soglia "comando ~ zero" resa proporzionale alla scala di profondita' dello squat.

    Diverse parti del MDP decidono "l'operatore ha chiesto di stare fermo?" confrontando la norma
    del comando con 0.1: il clock del passo (obs ``gait_phase`` e reward ``feet_gait``) e la
    reward di doppio appoggio. Quella soglia e' tarata sulla stazione eretta, dove il comando ha
    ampiezza piena. Con lo scaling per profondita' il comando si rimpicciolisce ma la soglia no,
    e si apre una banda in cui il robot riceve un target di velocita' non nullo mentre il clock
    del passo e' congelato e il doppio appoggio lo paga per restare piantato: obiettivi in
    conflitto, che la policy risolve non muovendosi. Misurato in sim2sim (2026-09-10): a 0.74 m
    guadagno 0.09 indietro, 0.13-0.23 laterale, 0.35-0.40 in imbardata, con la velocita' del
    ginocchio a 0.015 rad/s contro 0.045 in piedi -- non fa il passo, oscilla sul posto.

    Ritorna un tensore (N,) se il termine espone ``speed_scale``, altrimenti il float invariato
    (cosi' i task senza scaling e le vecchie config si comportano esattamente come prima).
    """
    term = env.command_manager.get_term(command_name)
    scale = getattr(term, "speed_scale", None)
    if scale is None:
        return threshold
    floor = getattr(term.cfg, "threshold_min_scale", 0.2)
    return threshold * scale.squeeze(-1).clamp(min=floor)


class UniformLevelVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command + explicit TURN-IN-PLACE sampling.

    WHY (measured on the 2026-09-04 run): the base class samples ``lin_vel_x``, ``lin_vel_y`` and
    ``ang_vel_z`` INDEPENDENTLY, and standing envs zero all three. The scenario that matters for
    the operator -- "no linear command, significant yaw", i.e. turning on the spot -- therefore
    occurs in only ~2% of samples, purely by chance. That is why the yaw gain stayed low at deploy
    and why ``lateral_drift_on_turn`` measured completely inert (reward stuck at -0.000) despite
    being a well-formed term: the situation it penalises was almost never sampled.

    ``rel_turn_in_place_envs`` forces a fraction of the NON-standing envs to have exactly zero
    linear command and a yaw command of at least half the current range, sign uniform. The
    magnitude scales with ``cfg.ranges.ang_vel_z``, so it follows the curriculum instead of
    fighting it.
    """

    cfg: UniformLevelVelocityCommandCfg

    def __init__(self, cfg: UniformLevelVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # Comando GREZZO, prima della scala per profondita'. Serve perche' _update_command gira a
        # ogni step mentre vel_command_b viene scritto solo al resample: applicare la scala in place
        # la COMPONEREBBE, e il comando svanirebbe in una frazione di secondo (a 0.785 il fattore e'
        # 0.5, quindi 0.5^10 = 1e-3 dopo 0.2 s) in modo del tutto silenzioso.
        self._raw_vel_command = torch.zeros_like(self.vel_command_b)

        # Fattore di scala per profondita', esposto PUBBLICAMENTE: le soglie "comando ~ zero"
        # sparse nel MDP (gait_phase, feet_gait, double_support) devono scalare insieme al
        # comando, altrimenti a mezzo squat il comando finisce sotto una soglia pensata per la
        # stazione eretta e il robot riceve un target di velocita' con il clock del passo
        # congelato. Vale 1.0 quando lo scaling e' disattivo.
        self.speed_scale = torch.ones(self.num_envs, 1, device=self.device)

        # Accumulatori di deriva, azzerati a ogni resample del comando. Misurano l'errore
        # INTEGRATO, non istantaneo: la deriva misurata in sim2sim (0.7-2.9 deg/s di imbardata
        # non comandata, 0.02 m/s di strisciamento laterale) e' piu' PICCOLA dell'oscillazione
        # sana dentro il passo (2-7 deg/s fra le due semifasi), quindi una penalita' sul rate
        # punirebbe soprattutto l'andatura normale. Sull'integrale il rapporto si inverte:
        # l'oscillazione per-passo si media a zero, la deriva no.
        self.heading_err = torch.zeros(self.num_envs, device=self.device)
        self.path_err = torch.zeros(self.num_envs, 2, device=self.device)
        # tempo TRASCORSO in accumulo: i reward dividono per questo e penalizzano l'errore MEDIO,
        # non quello accumulato. Vedi heading_hold/path_hold per il perche'.
        self.drift_time = torch.zeros(self.num_envs, device=self.device)
        self._drift_steps = torch.zeros(self.num_envs, device=self.device)
        # non ci appoggiamo a self.robot della classe base: il nome dell'attributo e' cambiato fra
        # le versioni di IsaacLab, mentre scene[cfg.asset_name] e' stabile.
        self._asset = env.scene[cfg.asset_name]

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        if self.cfg.rel_turn_in_place_envs <= 0.0:
            self._post_resample(env_ids)
            return

        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if env_ids.numel() == 0:
            return

        # non sovrapporre agli env fermi: quelli vengono azzerati a ogni step da _update_command
        pick = torch.rand(env_ids.numel(), device=self.device) <= self.cfg.rel_turn_in_place_envs
        pick &= ~self.is_standing_env[env_ids]
        ids = env_ids[pick]
        if ids.numel() == 0:
            self._post_resample(env_ids)
            return

        lo, hi = self.cfg.ranges.ang_vel_z
        amp = max(abs(lo), abs(hi))
        mag = torch.empty(ids.numel(), device=self.device).uniform_(0.5 * amp, amp)
        sign = torch.where(torch.rand(ids.numel(), device=self.device) < 0.5, -1.0, 1.0)

        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = mag * sign
        self._post_resample(env_ids)

    def _post_resample(self, env_ids):
        self._raw_vel_command[env_ids] = self.vel_command_b[env_ids]
        # la deriva si misura su una finestra di comando costante -> riparte da zero
        self.heading_err[env_ids] = 0.0
        self.path_err[env_ids] = 0.0
        self.drift_time[env_ids] = 0.0
        self._drift_steps[env_ids] = 0.0

    def _update_command(self):
        scaling = (self.cfg.speed_scale_zero_height is not None
                   and self.cfg.height_command_name in self._env.command_manager.active_terms)
        if scaling:
            # ripristina il grezzo PRIMA di super(), cosi' l'azzeramento degli env fermi resta valido
            self.vel_command_b[:] = self._raw_vel_command

        super()._update_command()

        if not scaling:
            # niente scaling (task senza comando di altezza): la scala resta 1.0, ma gli
            # accumulatori di deriva devono girare comunque.
            self._accumulate_drift()
            return

        # P8: l'ampiezza del comando scala con la profondita' dello squat.
        # Misurato in sim2sim (sim_2026-09-07_23-16-17, protocollo completo a tre quote):
        #     quota   vx avanti  vx indietro   vy     wz
        #     0.98      1.06        0.80      0.79   0.69
        #     0.78      0.82        0.69       --    0.58
        #     0.60      0.39        0.18      0.03   0.05
        # A 0.60 m la locomozione non esiste (hip -96 deg, knee +118, busto +24, appena 5.8 deg di
        # dorsiflessione residua): chiedere velocita' li' spende gradiente su un compito impossibile,
        # e al deploy manda il robot nella sua condizione meno stabile (sway 0.057, roll -4.9 deg).
        # Scalare invece di troncare mantiene la levetta proporzionata: a fondo squat non risponde.
        #
        # ROUND 7: la curva e' ora NON LINEARE (esponente < 1). Con la rampa lineare il comando a
        # meta' altezza diventava troppo piccolo per essere eseguibile: a 0.74 m il fattore era
        # 0.364, quindi vy e vx-indietro a fondo corsa valevano 0.109 -- appena sopra la soglia
        # 0.1 del clock del passo, e sotto 0.713 m *sempre* sotto. Misurato in sim2sim
        # (sim_2026-09-10_18-08-39) a 0.74 m: vx avanti 0.79 di guadagno (comando 0.36, sopra
        # soglia) contro 0.09 indietro, 0.13-0.23 laterale e 0.35-0.40 in imbardata. Con
        # esponente 0.5 il fattore a 0.74 passa a 0.60 (vy a fondo corsa 0.18, wz 0.30), mentre
        # a 0.62 resta esattamente 0 e a 0.95 esattamente 1: si alza l'autorita' solo nella
        # fascia intermedia, che e' quella che vogliamo mobile.
        h = self._env.command_manager.get_command(self.cfg.height_command_name)[:, 0]
        h0, h1 = self.cfg.speed_scale_zero_height, self.cfg.speed_scale_full_height
        scale = ((h - h0) / (h1 - h0)).clamp(0.0, 1.0)
        if self.cfg.speed_scale_exponent != 1.0:
            scale = scale.pow(self.cfg.speed_scale_exponent)
        self.speed_scale[:] = scale.unsqueeze(1)
        self.vel_command_b *= self.speed_scale

        self._accumulate_drift()

    def _accumulate_drift(self):
        """Integra l'errore di imbardata e lo scostamento laterale dalla traiettoria comandata.

        Chiamato a ogni step, azzerato a ogni resample. ``heading_hold`` e ``path_hold`` leggono
        questi due buffer.

        Lo scostamento e' PERPENDICOLARE al comando lineare: la componente LUNGO il comando e'
        il normale ritardo di inseguimento (guadagno 0.87 -> 13% di errore, che integrato su 10 s
        farebbe 0.4 m) e penalizzarla trasformerebbe questo termine in un track_lin_vel molto piu'
        duro, non in un termine di rettilineita'. A comando lineare nullo (giro sul posto, oppure
        env fermo) la direzione e' indefinita e si accumula l'intero vettore: e' esattamente il
        cerchio di raggio ~10 cm misurato durante il comando di yaw puro.
        """
        dt = self._env.step_dt
        grace = self.cfg.drift_grace_steps
        self._drift_steps += 1.0
        active = (self._drift_steps > grace).float()

        self.drift_time += dt * active

        wz = self._asset.data.root_ang_vel_b[:, 2]
        self.heading_err += (wz - self.vel_command_b[:, 2]) * dt * active

        cmd_xy = self.vel_command_b[:, :2]
        err = self._asset.data.root_lin_vel_b[:, :2] - cmd_xy
        n = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
        u = cmd_xy / n.clamp(min=1e-6)
        u = torch.where(n > 1e-6, u, torch.zeros_like(u))
        err_perp = err - (err * u).sum(dim=1, keepdim=True) * u
        self.path_err += err_perp * dt * active.unsqueeze(1)


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Hard limits the velocity curriculum may not exceed."""

    rel_turn_in_place_envs: float = 0.0
    """Fraction of the non-standing envs forced to turn on the spot (zero linear command, yaw at
    50-100% of the current range). 0.0 reproduces the base-class behaviour."""

    height_command_name: str = "base_height"
    """Name of the base-height command term used by the depth-dependent speed scaling."""

    speed_scale_zero_height: float | None = None
    """Base height [m] at which the velocity command is scaled to ZERO. ``None`` disables the
    scaling entirely (and nothing is written to deploy.yaml, so old policies keep working).

    MUST be kept in sync with the deploy: ``export_deploy_cfg`` writes these two numbers into
    ``deploy.yaml`` under ``commands.base_velocity.speed_scale``, and the C++ controller applies
    the identical formula in BOTH ``velocity_commands`` and ``gait_phase``. Scaling only one side
    would create exactly the distribution mismatch this is meant to remove."""

    speed_scale_full_height: float = 0.95
    """Base height [m] above which the velocity command keeps its full range."""

    speed_scale_exponent: float = 1.0
    """Exponent of the depth-speed curve: ``scale = clamp((h-h0)/(h1-h0))**exponent``.

    1.0 = linear ramp (round 6). 0.5 = square root: same endpoints, but much more command
    authority in the mid range, where the linear ramp pushed the lateral/backward command under
    the 0.1 gait threshold and the robot stopped moving altogether. Exported to ``deploy.yaml``
    and applied IDENTICALLY by the C++ controller."""

    threshold_min_scale: float = 0.2
    """Floor applied to ``speed_scale`` when it multiplies the "command is ~zero" thresholds.

    Those thresholds must shrink with the command, otherwise a squatted robot is given a target
    it is never allowed to gait for. But they must NOT reach zero: at full squat the scale is
    exactly 0, and a zero threshold would make ``cmd_norm < thr`` false, switching OFF the
    double-support reward precisely where standing perfectly still is what we want. With the
    floor the effective threshold at full squat is 0.1*0.2 = 0.02, and the command there is
    identically 0, so the gait stays frozen and double support stays rewarded."""

    drift_grace_steps: float = 25.0
    """Policy steps after a resample during which the drift accumulators stay frozen (25 = 0.5 s).

    Without it every term would carry an unavoidable offset: reaching a freshly sampled yaw
    command takes ~0.5 s, which alone integrates to ~0.12 rad even for a perfect policy."""
