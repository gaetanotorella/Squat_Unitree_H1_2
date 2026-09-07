from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


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

    def _update_command(self):
        scaling = (self.cfg.speed_scale_zero_height is not None
                   and self.cfg.height_command_name in self._env.command_manager.active_terms)
        if scaling:
            # ripristina il grezzo PRIMA di super(), cosi' l'azzeramento degli env fermi resta valido
            self.vel_command_b[:] = self._raw_vel_command

        super()._update_command()

        if not scaling:
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
        h = self._env.command_manager.get_command(self.cfg.height_command_name)[:, 0]
        h0, h1 = self.cfg.speed_scale_zero_height, self.cfg.speed_scale_full_height
        scale = ((h - h0) / (h1 - h0)).clamp(0.0, 1.0)
        self.vel_command_b *= scale.unsqueeze(1)


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
