from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gait_phase(
    env: ManagerBasedRLEnv, period: float, command_name: str | None = None, threshold: float = 0.1
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    global_phase = (env.episode_length_buf * env.step_dt) % period / period

    phase = torch.zeros(env.num_envs, 2, device=env.device)
    phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
    phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)

    # Opzione C: congela il clock a (0,0) quando il comando e' ~0 (stessa soglia della gait reward).
    # Elimina il dondolio 1.25 Hz a comando-zero. Deve essere replicato IDENTICO nel deploy C++
    # (sul robot reale "fermo" = comando joystick sotto soglia).
    if command_name is not None:
        cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1, keepdim=True)
        phase = phase * (cmd_norm > threshold)
    return phase
