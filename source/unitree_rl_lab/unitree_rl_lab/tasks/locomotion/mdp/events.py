from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

"""
Domain randomization events.
"""


def randomize_body_com(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_range: dict[str, tuple[float, float]],
):
    """Randomize the CENTER OF MASS offset of selected bodies (persistent within an episode).

    WHY THIS EXISTS (measured on the real robot, 2026-09-04): with the same policy, the H1_2
    stood with its CoM ~3-6 cm FURTHER BACK than the model predicts (estimated from the steady
    ankle torque, tau = M*g*d). That ate more than half of the backward stability margin
    (10.7 cm in sim -> 4.1 cm real, with the heel edge at -8.6 cm) and made the robot tip onto
    its heels as soon as a squat was commanded.

    Randomizing the MASS alone does NOT cover this: ``torso_link`` has its CoM at x = +0.0005 m,
    so adding -1..+3 kg there moves the whole-body CoM horizontally by only 0.08 cm (measured) --
    40-80x too little. What matters is randomizing WHERE the mass is, not how much.

    Use ``mode="reset"``: the offset must be resampled every episode so the policy has to INFER
    it from the observation history rather than see one fixed value per environment.

    Args:
        com_range: per-axis offset ranges, e.g. ``{"x": (-0.06, 0.06), "y": (-0.03, 0.03)}``.
            Offsets are applied on top of the asset's DEFAULT CoM (cached on first call), so they
            never accumulate across resets. Missing axes default to no offset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    view = asset.root_physx_view

    # cache the untouched CoMs once, so repeated resets offset from the default and never drift
    cache_key = f"_default_coms_{asset_cfg.name}"
    if not hasattr(env, cache_key):
        setattr(env, cache_key, view.get_coms().clone())
    default_coms = getattr(env, cache_key)

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    env_ids = env_ids.to(default_coms.device)

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, slice):
        body_ids = torch.arange(asset.num_bodies, device=default_coms.device)
    else:
        body_ids = torch.as_tensor(body_ids, device=default_coms.device)

    coms = default_coms.clone()
    for axis, idx in (("x", 0), ("y", 1), ("z", 2)):
        lo, hi = com_range.get(axis, (0.0, 0.0))
        if lo == 0.0 and hi == 0.0:
            continue
        # one offset per env, shared by all selected bodies (a rigid shift of the payload)
        offsets = torch.empty(len(env_ids), device=default_coms.device).uniform_(lo, hi)
        coms[env_ids[:, None], body_ids[None, :], idx] += offsets[:, None]

    view.set_coms(coms, env_ids)
