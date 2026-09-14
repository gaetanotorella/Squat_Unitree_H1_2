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
        # Fase B riattivata PARZIALMENTE al round 5. Con only_when_standing=False (round 2, tutti
        # gli env in cammino che squattano) la locomozione era peggiorata in modo misurabile:
        # terrain_levels 4.78 -> 2.61, track_lin_vel 0.835 -> 0.733, guadagno vx 0.68 -> 0.58.
        # Il flag resta True e la Fase B passa dalla frazione qui sotto.
        only_when_standing=True,
        # P4: Fase B PARZIALE. Il 35% degli env in cammino riceve comunque una profondita' di squat,
        # cosi' camminare accovacciati torna IN distribuzione senza dominare il batch. E' la
        # modifica con l'evidenza piu' netta di tutte (confronto a sei policy sullo stesso
        # protocollo, comando pieno a 0.75 m): da R4 (senza) a R5 (con) vx avanti 0.50 -> 0.94,
        # vx indietro 0.03 -> 0.16, vy 0.02 -> 0.11, wz 0.16 -> 0.27.
        # ROUND 9: torna a 0.35 dal 0.50 del round 8 (quel round e' collassato per altre ragioni e
        # il 0.50 non e' separabile dal collasso; 0.35 e' il valore misurato).
        rel_squat_while_walking=0.35,
        # ...ma senza scendere sotto 0.70 mentre cammina: a 0.60 i guadagni misurati sono
        # -0.03/-0.01/+0.13 e lo sway triplica, quindi quegli env spenderebbero gradiente su un
        # compito che al momento non riesce. 0.60 resta pienamente disponibile da fermo.
        # NB: questo pavimento e' anche il CUTOFF del deploy (export_deploy_cfg lo scrive in
        # deploy.yaml come cutoff_height): sotto questa quota il controller azzera il comando di
        # velocita', cosi' il deploy non chiede mai una manovra che il training non ha mai visto.
        min_height_while_walking=0.70,
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
        # -1.0 (era -0.5): riduce l'asimmetria L/R ~5 deg (hip_pitch/ankle_pitch).
        # ORA GATED sugli env fermi: (L-R)^2 senza gate penalizza l'andatura, che e' in
        # CONTRO-FASE per costruzione. Misurato -0.023/-0.031 nel run 2026-09-04, cioe' termine
        # attivamente violato dal passo (~4% di zavorra su track_lin_vel_xy). Il gate conserva
        # l'intento -- uno squat simmetrico -- ed elimina il conflitto col cammino.
        # NB: roll_symmetry NON e' gated: vale -0.003, quindi e' gia' soddisfatto e non ostacola.
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
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
    # -- lateral_drift_on_turn: NON in config. Provato nel round 7 a -2.0: 0.003% della reward,
    # inerte, con il raggio del giro sul posto peggiorato (0.04 -> 0.13 m). Resta in rewards.py.

    # -- margine di stabilita': tiene il baricentro sopra le caviglie quando si sta in piedi.
    # E' il termine che mancava e che sul robot reale ha fatto la differenza (CoM 4.5 cm dietro,
    # a 4.1 cm dal bordo del tallone). Taratura: in sim la coppia da fermo e' ~14 N·m sui due
    # ankle_pitch; con L1 e peso -7e-4 la penalita' nominale e' ~0.010, in linea con gli altri
    # termini di regolarizzazione (action_rate ~0.05, ankle_action_smoothness ~0.01).
    # P0: ora ATTIVO A TUTTE LE PROFONDITA' (prima era gated su "vicino alla quota nominale").
    # Il test sim2sim col baricentro spostato di -3.5 cm (2026-09-06) ha misurato che dove questo
    # termine e' acceso la policy compensa il 69% dello sbilanciamento, dove e' spento il 43% --
    # e il 43% non e' dovuto al fine corsa della caviglia (restano 4.9 gradi di margine) ma alla
    # mancanza di incentivo. Il gate ora e' "assestato alla quota COMANDATA" (|h - h_cmd| < tol),
    # che esclude il transitorio di discesa/risalita ma non la profondita'.
    ankle_torque_margin = RewTerm(
        func=mdp.ankle_torque_standing,
        weight=-7e-4,
        params={
            "command_name": "base_velocity",
            "height_command_name": "base_height",
            "height_tol": 0.05,
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_pitch_joint"]),
        },
    )

    # -- orientamento del busto ASIMMETRICO, sostituisce flat_orientation_l2 (stesso peso -5.0).
    # A quota nominale e' IDENTICO al termine che sostituisce (banda morta = 0), quindi la postura
    # in camminata eretta non cambia; scendendo apre progressivamente l'inclinazione IN AVANTI
    # fino a ~20 deg a fondo squat, mentre l'inclinazione ALL'INDIETRO resta penalizzata sempre.
    # Motivo: il poligono d'appoggio e' asimmetrico (punta +17.3 cm, tallone -8.6 cm dalla
    # caviglia), quindi cadere all'indietro costa il doppio, e il lean in avanti E' la manovra di
    # recupero quando il CoM sta dietro -- il modo di fallire misurato sul robot reale.
    # NB: NON e' il collo di bottiglia del "bacino dietro le caviglie" (misurato: la policy chiede
    # gia' +37.8 deg di lean a 0.73 m e ne realizza -1.6, il limite e' il cedimento del PD di
    # caviglia). E' allineamento della reward alla geometria, a costo e rischio nulli.
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_asym,
        weight=-5.0,
        params={
            "height_command_name": "base_height",
            "nominal_height": NOMINAL_HEIGHT,
            "deepest_height": DEEPEST_SQUAT,
            # P6: 0.42 (~25 deg), era 0.35 (~20.5). Il round 4 ha misurato la policy a +20.78 gradi
            # a fondo squat, cioe' ESATTAMENTE sul recinto: la banda morta e' diventata il vincolo
            # attivo. E a 0.60 m la caviglia ha solo 5.8 gradi di dorsiflessione residua, quindi il
            # busto e' l'unico grado di liberta' rimasto per ricentrare il baricentro piu' in basso.
            "forward_deadband": 0.42,
        },
    )

    def __post_init__(self):
        # fixed-target height penalty -> replaced by track_base_height (variable target)
        self.base_height = None
        # simmetrico avanti/indietro -> sostituito da flat_orientation (asimmetrico, depth-gated)
        self.flat_orientation_l2 = None
        # ungated vertical-velocity penalty would fight the squat transition -> replaced by settle_height
        self.base_linear_velocity = None
        # stand_still_zero_cmd a -0.1: TENUTO, e ora si sa perche'. Penalizza |q - q_default| su
        # tutti i 27 giunti degli is_standing_env, che sono anche gli env che squattano: e' una
        # tassa proporzionale alla profondita' (-0.18/step a 0.98, -0.75 a 0.64). Non frena lo
        # squat (l'equilibrio col gradiente di track_base_height e' un offset di 1.3 mm) e non vede
        # il ciclo-limite della caviglia (che vive nell'azione). Nel round 8 e' stato RIMOSSO per
        # questo, e il risultato ha dimostrato il suo vero ruolo: e' l'ANCORA DELLA POSTURA in
        # piedi. Senza, la policy ha portato il baricentro sull'asse della caviglia (pitch -4.5 deg,
        # coppia di caviglia da +7.8 a -0.8 N.m), la caviglia scarica e' entrata in un ciclo-limite
        # a 1.5 Hz (azione std x44, coppia +-16 N.m, sway da fermo x36). Con la caviglia precaricata
        # dalla postura nominale il chatter non parte. A -0.1 il costo sullo squat e' trascurabile.
        self.stand_still_zero_cmd.weight = -0.1


@configclass
class SquatCurriculumCfg(CurriculumCfg):
    """Velocity curriculum (inherited) + squat-depth curriculum. Terrain curriculum disabled."""

    # ROUND 9: il curriculum del terreno e' un NO-OP e viene spento. L'unico sotto-terreno in
    # config e' `flat`, quindi le 9x21 tessere sono identiche e promuovere o retrocedere un env non
    # cambia nulla di fisico -- `Curriculum/terrain_levels` misurava solo se gli env percorrevano
    # la distanza comandata, non la robustezza al terreno. Il __post_init__ della base mette
    # terrain_generator.curriculum = False quando questo termine e' None. Il terreno resta lo stesso.
    terrain_levels = None
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
