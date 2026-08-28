"""Squat task for H1_2: walk + stand + squat-in-place (velocity command ~0).

Built on top of the velocity task (``velocity_env_cfg``). It ADDS a base-height command,
a height-tracking reward and squat-specific stability terms, and a height curriculum that
only deepens the squat once locomotion is mastered ("learn to walk/stand first, then squat").

Design intent / sim2real notes:
- The ACTOR only gains one new observation, ``base_height_command`` (the operator's target,
  reproducible on the real robot). It infers its current height from ``joint_pos_rel`` (FK-like).
- The CRITIC additionally sees the TRUE base height (privileged, training-only).
- Rewards may use the true base height (training-only), so height tracking is exact.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .velocity_env_cfg import (
    CommandsCfg,
    CurriculumCfg,
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
)

NOMINAL_HEIGHT = 0.98
DEEPEST_SQUAT = 0.55


@configclass
class SquatCommandsCfg(CommandsCfg):
    """Velocity commands (inherited) + a base-height command for the squat."""

    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        nominal_height=NOMINAL_HEIGHT,
        only_when_standing=True,  # squat only at ~zero velocity (Phase A)
        velocity_command_name="base_velocity",
        debug_vis=False,
        # start with NO squat (pure standing/walking); curriculum lowers the minimum.
        ranges=mdp.UniformHeightCommandCfg.Ranges(height=(NOMINAL_HEIGHT, NOMINAL_HEIGHT)),
        limit_ranges=mdp.UniformHeightCommandCfg.Ranges(height=(DEEPEST_SQUAT, NOMINAL_HEIGHT)),
    )


@configclass
class SquatObservationsCfg(ObservationsCfg):
    """Adds the height command to the actor, and the true height to the (privileged) critic."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # operator target height, centered on nominal (reproducible on the real robot)
        base_height_command = ObsTerm(
            func=mdp.base_height_command_rel,
            params={"command_name": "base_height", "nominal_height": NOMINAL_HEIGHT},
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        base_height_command = ObsTerm(
            func=mdp.base_height_command_rel,
            params={"command_name": "base_height", "nominal_height": NOMINAL_HEIGHT},
        )
        # TRUE base height -- privileged, critic only, never on the real robot
        base_height = ObsTerm(func=mdp.base_height_rel, params={"nominal_height": NOMINAL_HEIGHT})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class SquatRewardsCfg(RewardsCfg):
    """Velocity rewards (inherited) + squat task/stability terms."""

    # -- squat task
    track_base_height = RewTerm(
        func=mdp.track_base_height_exp,
        weight=3.0,
        params={"command_name": "base_height", "std": 0.1},
    )
    # penalize vertical bouncing ONLY once near the target height (does not fight the transition)
    settle_height = RewTerm(
        func=mdp.lin_vel_z_gated,
        weight=-2.0,
        params={"command_name": "base_height", "height_tol": 0.05},
    )
    # -- squat stability
    squat_symmetry = RewTerm(  # L/R symmetry on the squat DOF (anti-asymmetric-squat)
        func=mdp.joint_mirror,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["left_hip_pitch_joint", "right_hip_pitch_joint"],
                ["left_knee_joint", "right_knee_joint"],
                ["left_ankle_pitch_joint", "right_ankle_pitch_joint"],
            ],
        },
    )
    double_support = RewTerm(  # both feet planted when standing/squatting
        func=mdp.feet_contact_without_cmd,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
            "command_name": "base_velocity",
        },
    )
    feet_too_near = RewTerm(  # keep feet apart (no collapse) while squatting
        func=mdp.feet_too_near,
        weight=-2.0,
        params={"threshold": 0.2, "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*")},
    )

    def __post_init__(self):
        # fixed-target height penalty -> replaced by track_base_height (variable target)
        self.base_height = None
        # ungated vertical-velocity penalty would fight the squat transition -> replaced by settle_height
        self.base_linear_velocity = None


@configclass
class SquatCurriculumCfg(CurriculumCfg):
    """Velocity/terrain curricula (inherited) + squat-depth curriculum."""

    height_cmd_levels = CurrTerm(func=mdp.height_cmd_levels)


@configclass
class SquatEnvCfg(RobotEnvCfg):
    """Configuration for the H1_2 walk+stand+squat environment."""

    observations: SquatObservationsCfg = SquatObservationsCfg()
    commands: SquatCommandsCfg = SquatCommandsCfg()
    rewards: SquatRewardsCfg = SquatRewardsCfg()
    curriculum: SquatCurriculumCfg = SquatCurriculumCfg()


@configclass
class SquatPlayEnvCfg(SquatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        # at play time expose the full command ranges (walk + full squat depth)
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.commands.base_height.ranges = self.commands.base_height.limit_ranges
