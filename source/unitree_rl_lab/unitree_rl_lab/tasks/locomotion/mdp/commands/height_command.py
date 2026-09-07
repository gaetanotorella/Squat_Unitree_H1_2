from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class UniformHeightCommand(CommandTerm):
    """Command term that samples a target base height (absolute, meters).

    The commanded height is sampled uniformly in ``cfg.ranges.height``. A curriculum
    (see ``mdp.curriculums.height_cmd_levels``) can lower the minimum over training so the
    robot first masters standing/walking near the nominal height, then learns to squat.

    If ``cfg.only_when_standing`` is True, envs that are NOT standing (i.e. have a non-zero
    ``base_velocity`` command) keep the nominal height, so the squat is learned at ~zero
    velocity (Phase A). Set it False to learn squat-while-walking (Phase B).
    """

    cfg: UniformHeightCommandCfg

    def __init__(self, cfg: UniformHeightCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        # current commanded height per env, initialised at the nominal (standing) height
        self.height_command = torch.full((self.num_envs, 1), cfg.nominal_height, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"UniformHeightCommand(range={self.cfg.ranges.height}, nominal={self.cfg.nominal_height})"

    @property
    def command(self) -> torch.Tensor:
        # shape (num_envs, 1): absolute target base height in meters
        return self.height_command

    def _resample_command(self, env_ids: Sequence[int]):
        lo, hi = self.cfg.ranges.height
        h = torch.empty(len(env_ids), device=self.device).uniform_(lo, hi).unsqueeze(1)
        if self.cfg.only_when_standing:
            # squat only where the velocity command is ~zero; walking envs stay at nominal height
            vel_term = self._env.command_manager.get_term(self.cfg.velocity_command_name)
            standing = vel_term.is_standing_env[env_ids].unsqueeze(1)
            gets_squat = standing
            if self.cfg.rel_squat_while_walking > 0.0:
                # Fase B PARZIALE: una frazione degli env in cammino riceve comunque una profondita'
                walking_squat = (torch.rand_like(h) < self.cfg.rel_squat_while_walking) & ~standing
                if self.cfg.min_height_while_walking is not None:
                    # Pavimento di profondita' SOLO per chi cammina; gli env fermi restano liberi.
                    # Rimappa affine invece di troncare: un clamp accumulerebbe TUTTI i campioni
                    # sotto il pavimento su un unico valore (col range [0.60,0.98] e pavimento 0.70
                    # sarebbe un picco del 26% esattamente a 0.70), qui invece la distribuzione
                    # resta uniforme su [pavimento, hi].
                    lo_w = max(lo, self.cfg.min_height_while_walking)
                    if hi > lo:
                        h_walk = lo_w + (h - lo) * ((hi - lo_w) / (hi - lo))
                    else:
                        h_walk = torch.full_like(h, lo_w)
                    h = torch.where(walking_squat, h_walk, h)
                gets_squat = gets_squat | walking_squat
            h = torch.where(gets_squat, h, torch.full_like(h, self.cfg.nominal_height))
        self.height_command[env_ids] = h

    def _update_command(self):
        # height command is held constant between resamples
        pass

    def _update_metrics(self):
        asset = self._env.scene[self.cfg.asset_name]
        base_h = asset.data.root_pos_w[:, 2:3]
        self.metrics["error_height"] = torch.abs(self.height_command - base_h).squeeze(1)

    # --- debug vis (no-op skeleton) ---
    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass


@configclass
class UniformHeightCommandCfg(CommandTermCfg):
    """Configuration for :class:`UniformHeightCommand`."""

    class_type: type = UniformHeightCommand

    asset_name: str = MISSING
    """Name of the robot asset whose base height is commanded."""

    velocity_command_name: str = "base_velocity"
    """Name of the velocity command term, used to read ``is_standing_env`` when
    ``only_when_standing`` is True."""

    nominal_height: float = 0.98
    """Standing (upright) base height in meters. Used as the default / walking height."""

    only_when_standing: bool = True
    """If True, only standing envs (zero velocity command) receive a squat depth; walking
    envs keep the nominal height. Set False for squat-while-walking (Phase B)."""

    rel_squat_while_walking: float = 0.0
    """Fraction of the WALKING envs that also receive a squat depth, on top of the standing ones
    (partial Phase B). Only meaningful when ``only_when_standing`` is True.

    Why a fraction and not the all-or-nothing switch: with ``only_when_standing=False`` (round 2,
    every walking env squatting) locomotion degraded measurably -- terrain_levels 4.78 -> 2.61,
    track_lin_vel_xy 0.835 -> 0.733, vx gain 0.68 -> 0.58. With True, walking-while-squatting is
    out of distribution by construction and the sim2sim gains collapsed to 0.09-0.28 at half squat.
    A fraction keeps the skill in distribution without letting it dominate the batch."""

    min_height_while_walking: float | None = None
    """Depth floor applied ONLY to the walking envs selected by ``rel_squat_while_walking``.
    ``None`` disables it; standing envs are never affected and keep the full curriculum range.

    Why (measured on ``sim_2026-09-07_14-23-00``, full command protocol at three heights): command
    gains collapse with depth and hit zero at the bottom --

        height   vx fwd   vx bwd    vy      wz
        0.98      1.02     0.60    0.63    0.73
        0.76      0.55     0.16    0.13    0.47
        0.60      0.28    -0.03   -0.01    0.13

    and the sway triples there. It is NOT a torque or range limit -- in single support the knee
    would use 54% of its limit and the ankle 2% -- but at 0.60 m the gait is simply gone (knee
    oscillation 0.014 vs 0.123 standing). Letting a third of the walking envs sample depths where
    locomotion currently yields nothing spends gradient on a task that fails anyway, instead of on
    0.76-0.85 where the recovery is realistic. Lower the floor in a later round, once walking at
    half squat is solid."""

    @configclass
    class Ranges:
        height: tuple[float, float] = MISSING  # (min, max) sampled target height [m]

    ranges: Ranges = MISSING
    """Current sampling range for the target height (min is lowered by the curriculum)."""

    limit_ranges: Ranges = MISSING
    """Hard limits the curriculum may not exceed (e.g. deepest allowed squat)."""
