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

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)

        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device).reshape(-1)
        if env_ids.numel() == 0:
            return

        # non sovrapporre agli env fermi: quelli vengono azzerati a ogni step da _update_command
        pick = torch.rand(env_ids.numel(), device=self.device) <= self.cfg.rel_turn_in_place_envs
        pick &= ~self.is_standing_env[env_ids]
        ids = env_ids[pick]
        if ids.numel() == 0:
            return

        lo, hi = self.cfg.ranges.ang_vel_z
        amp = max(abs(lo), abs(hi))
        mag = torch.empty(ids.numel(), device=self.device).uniform_(0.5 * amp, amp)
        sign = torch.where(torch.rand(ids.numel(), device=self.device) < 0.5, -1.0, 1.0)

        self.vel_command_b[ids, 0] = 0.0
        self.vel_command_b[ids, 1] = 0.0
        self.vel_command_b[ids, 2] = mag * sign


@configclass
class UniformLevelVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = UniformLevelVelocityCommand

    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    """Hard limits the velocity curriculum may not exceed."""

    rel_turn_in_place_envs: float = 0.0
    """Fraction of the non-standing envs forced to turn on the spot (zero linear command, yaw at
    50-100% of the current range). 0.0 reproduces the base-class behaviour."""
