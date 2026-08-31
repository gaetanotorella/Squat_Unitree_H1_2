from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

        # Yaw (turning) expansion, gated INDEPENDENTLY on its own tracking reward. Without this the
        # ang_vel_z command stays frozen at its initial (tiny) range for the whole training while the
        # deploy joystick maps the full limit range -> turning is out-of-distribution and ignored.
        try:
            ang_reward = (
                torch.mean(env.reward_manager._episode_sums["track_ang_vel_z"][env_ids])
                / env.max_episode_length_s
            )
            ang_weight = env.reward_manager.get_term_cfg("track_ang_vel_z").weight
            if ang_reward > ang_weight * 0.8:
                delta_command = torch.tensor([-0.1, 0.1], device=env.device)
                ranges.ang_vel_z = torch.clamp(
                    torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                    limit_ranges.ang_vel_z[0],
                    limit_ranges.ang_vel_z[1],
                ).tolist()
        except KeyError:
            pass  # task without a track_ang_vel_z reward: skip yaw expansion

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def height_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    height_command_name: str = "base_height",
    lin_vel_reward_name: str = "track_lin_vel_xy",
    height_reward_name: str = "track_base_height",
    step: float = 0.05,
    lin_vel_thresh_ratio: float = 0.8,
    height_thresh_ratio: float = 0.8,
) -> torch.Tensor:
    """Lower the minimum commanded squat height, gated on competence.

    The minimum of ``ranges.height`` is decreased by ``step`` (toward ``limit_ranges.height[0]``)
    only when BOTH locomotion tracking (``track_lin_vel_xy``) AND current height tracking
    (``track_base_height``) are good. This realises "learn to walk/stand first, then squat":
    while the robot is still learning to walk, the squat depth does not increase.

    Returns the current minimum commanded height (for logging).
    """
    cmd_term = env.command_manager.get_term(height_command_name)
    ranges = cmd_term.cfg.ranges
    limit_ranges = cmd_term.cfg.limit_ranges

    rm = env.reward_manager
    lin_reward = torch.mean(rm._episode_sums[lin_vel_reward_name][env_ids]) / env.max_episode_length_s
    height_reward = torch.mean(rm._episode_sums[height_reward_name][env_ids]) / env.max_episode_length_s
    lin_weight = rm.get_term_cfg(lin_vel_reward_name).weight
    height_weight = rm.get_term_cfg(height_reward_name).weight

    if env.common_step_counter % env.max_episode_length == 0:
        walk_ok = lin_reward > lin_weight * lin_vel_thresh_ratio
        height_ok = height_reward > height_weight * height_thresh_ratio
        if walk_ok and height_ok:
            new_min = max(ranges.height[0] - step, limit_ranges.height[0])
            ranges.height = (new_min, ranges.height[1])

    return torch.tensor(ranges.height[0], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)
