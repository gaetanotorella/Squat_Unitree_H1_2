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
# 0.60 (era 0.55): misurato in sim2sim che con i piedi piatti il robot NON puo' scendere sotto
# ~0.59 m -- l'ankle_pitch va a fondo corsa (limite HW [-0.897, +0.524] rad, soft limit di training
# 0.9x -> -0.826). Sotto 0.62 il tracking saturava (+3 cm di errore) e la policy comandava target
# oltre il limite fisico (-1.00 rad), sprecando autorita' di controllo contro il fine corsa.
DEEPEST_SQUAT = 0.60


@configclass
class SquatCommandsCfg(CommandsCfg):
    """Velocity commands (inherited) + a base-height command for the squat."""

    base_height = mdp.UniformHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        nominal_height=NOMINAL_HEIGHT,
        # FASE B: False -> lo squat viene campionato anche sugli env in cammino. In deploy l'operatore
        # puo' tenere RT mentre cammina (nel run 2026-09-03 e' successo nel 12% dei campioni) e con
        # True quel regime era FUORI DISTRIBUZIONE. Il curriculum di altezza resta il freno di
        # sicurezza: approfondisce solo se track_lin_vel E track_base_height sono entrambi buoni.
        only_when_standing=False,
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
        # std 0.07 (era 0.1): piu' gradiente sotto i 5 cm -> raggiunge la profondita' comandata
        # (prima ~4 cm di undershoot alla massima discesa).
        params={"command_name": "base_height", "std": 0.07},
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
        # -1.0 (era -0.5): riduce l'asimmetria L/R ~5 deg (hip_pitch/ankle_pitch) che rendeva il passo
        # irregolare e causava la deriva lenta di yaw. Applicata sempre (non gated).
        weight=-1.0,
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
    # -- anti-barcollio: smorza il ciclo-limite di caviglia da fermo (vive nell'AZIONE, non nella
    # posizione misurata -> stand_still non lo vede). Gated su is_standing_env: non tocca il passo; la
    # discesa squat (lenta) ha action-rate piccola -> non la ostacola. Peso DA TUNARE dal basso.
    ankle_action_smoothness = RewTerm(
        func=mdp.joint_action_rate_standing,
        # -0.04 (era -0.02): a -0.02 il ciclo-limite si e' dimezzato (sway wy 0.030 -> 0.016) senza
        # alcun costo di stabilita' -> c'e' margine per stringere ancora.
        weight=-0.04,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"]),
        },
    )
    # -- simmetria L/R sui giunti ROLL (antisimmetrici: L = -R). Attacca l'asimmetria di risposta
    # laterale/rotazionale misurata in sim (vy verso sinistra al 13% del comandato vs 73% a destra).
    # Peso piu' basso di squat_symmetry: i roll servono anche all'equilibrio, non vanno congelati.
    roll_symmetry = RewTerm(
        func=mdp.joint_mirror_antisym,
        weight=-0.3,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "mirror_joints": [
                ["left_hip_roll_joint", "right_hip_roll_joint"],
                ["left_ankle_roll_joint", "right_ankle_roll_joint"],
            ],
        },
    )
    # -- anti-sbandamento in rotazione: girare sul posto era la manovra piu' destabilizzante
    # (roll -9.1 deg e vy spuria fino a 0.45 m/s con wz=-0.5 e comando lineare nullo).
    turn_lateral_drift = RewTerm(
        func=mdp.lateral_drift_on_turn,
        weight=-0.5,
        params={"command_name": "base_velocity", "lin_cmd_thresh": 0.1, "ang_cmd_thresh": 0.2},
    )

    def __post_init__(self):
        # fixed-target height penalty -> replaced by track_base_height (variable target)
        self.base_height = None
        # ungated vertical-velocity penalty would fight the squat transition -> replaced by settle_height
        self.base_linear_velocity = None
        # stand_still penalizza |q - q_default| sugli is_standing_env, ma quegli env ricevono anche un
        # target di squat (posa che DEVE scostarsi dalla default) -> frena lo squat, e NON tocca il
        # ciclo-limite (che vive nell'azione). Tenuto basso a -0.1: standing-tall e' gia' buono
        # (sway 0.016, err altezza 0.29 cm); l'anti-barcollio e' ankle_action_smoothness.
        self.stand_still_zero_cmd.weight = -0.1


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
