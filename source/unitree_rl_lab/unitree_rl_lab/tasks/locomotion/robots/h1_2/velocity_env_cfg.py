import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.assets.robots.unitree import UNITREE_H1_2_CFG as ROBOT_CFG
from unitree_rl_lab.tasks.locomotion import mdp

COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=9,
    num_cols=21,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
    },
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=COBBLESTONE_ROAD_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=COBBLESTONE_ROAD_CFG.num_rows - 1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # Incertezza di modello sulla massa dei singoli link: il MJCF pesa 67.0 kg contro i ~70 kg
    # reali. "startup" perche' serve solo COPERTURA statistica (e' incertezza di modello, non
    # cambia nel tempo): 4096 env = 4096 campioni, sufficienti.
    randomize_link_masses = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.85, 1.15),
            "operation": "scale",
        },
    )

    # Il training usa un PD IDEALE: sul robot vero la rigidezza effettiva differisce. "startup"
    # per non interferire con lo stato interno del modello di attuatore.
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    # *** IL FIX PRINCIPALE PER IL ROBOT REALE ***
    # Sul robot vero il baricentro e' risultato 3-6 cm PIU' INDIETRO di quanto il modello preveda
    # (stimato dalla coppia di caviglia a regime, tau = M*g*d), consumando piu' della meta' del
    # margine di ribaltamento all'indietro (10.7 cm in sim -> 4.1 cm reale, tallone a -8.6 cm):
    # da li' l'inclinazione all'indietro e la caduta appena comandato lo squat.
    # "reset" (NON startup): il valore deve cambiare ad ogni episodio, altrimenti la history
    # (history_length=5) non ha nulla da inferire e la policy diventa solo conservativa invece
    # che adattiva. Copertura: 4096 env x ~120 episodi ~= 500k campioni contro i 4096 di startup.
    # TUTTI i link, non solo pelvis+torso. La funzione applica lo STESSO offset a tutti i corpi
    # selezionati, quindi e' una TRASLAZIONE RIGIDA della distribuzione di massa e lo spostamento
    # del CoM globale e' scalato dalla frazione di massa selezionata:
    #   pelvis+torso = 23.77/67.0 = 35.5%  ->  +-8 cm impostati davano solo +-2.8 cm EFFETTIVI,
    #   cioe' meno del disallineamento reale misurato: il round 3 addestrava fuori distribuzione.
    #   tutti i link = 100%  ->  rapporto 1:1, il numero qui sotto e' quello vero.
    # Vale anche con randomize_link_masses attivo: sum(m_i*s_i*D)/sum(m_i*s_i) = D per ogni s_i.
    # RANGE x = +-0.06 (non +-0.08): l'ERRORE DI MODELLO misurato e' ~3-4 cm (stessa policy, da
    # fermo: il CoM si assesta a +0.6/+1.3 cm in sim e a -2.4/-3.0 cm sul reale) -> fattore di
    # sicurezza 1.5-2x. Con +-8 cm effettivi, sommati alla posa di squat profondo, molti episodi
    # sarebbero irrecuperabili (il margine al tallone e' solo 8.6 cm) -> collasso suicida.
    randomize_base_com = EventTerm(
        func=mdp.randomize_body_com,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.06, 0.06), "y": (-0.02, 0.02), "z": (-0.02, 0.02)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformLevelVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        # P3: il giro sul posto (comando lineare nullo + yaw significativo) capitava per caso nel
        # ~2% dei campioni, perche' vx/vy/wz sono estratti in modo indipendente. Con 0.15 diventa
        # uno scenario allenato esplicitamente -- ed e' anche la condizione che rende finalmente
        # non-inerte turn_lateral_drift, se lo riattiveremo.
        rel_turn_in_place_envs=0.15,
        # P8: ampiezza del comando proporzionale alla profondita'. 1.0 sopra 0.95 m, 0 sotto 0.62.
        # Vale 0.50 a 0.785 e 0.24 a 0.70, in linea con cio' che il robot dimostra di saper fare.
        # Completa P7: quello impediva agli env in cammino di scendere sotto 0.70, questo impedisce
        # di chiedere a 0.70 la stessa velocita' che si chiede a 0.98. Applicato IDENTICO nel
        # controller C++ (velocity_commands e gait_phase) tramite deploy.yaml.
        speed_scale_zero_height=0.62,
        speed_scale_full_height=0.95,
        # ROUND 7: curva NON LINEARE. Con l'esponente 1.0 il fattore a 0.74 m valeva 0.364, quindi
        # vy e vx-indietro a fondo corsa scendevano a 0.109 -- appena sopra la soglia 0.1 del clock
        # del passo, e sotto 0.713 m sempre sotto. Misurato in sim2sim (2026-09-10) a 0.74 m:
        # guadagno 0.79 in avanti (comando 0.36, sopra soglia grazie al range asimmetrico
        # [-0.3, 1.0]) contro 0.09 indietro, 0.13-0.23 laterale, 0.35-0.40 in imbardata, con la
        # velocita' del ginocchio a 0.015 rad/s contro 0.045 in piedi: non fa il passo.
        # Con 0.5: 0.74 -> 0.60 (vy pieno 0.18, wz 0.30), 0.85 -> 0.84, 0.70 -> 0.49.
        # Estremi invariati: 0.62 -> 0 esatto, 0.95 -> 1 esatto.
        speed_scale_exponent=0.5,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
        limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 1.0), lin_vel_y=(-0.3, 0.3), ang_vel_z=(-0.5, 0.5)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    JointPositionAction = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8, "command_name": "base_velocity"})

        def __post_init__(self):
            # 5 frame (100 ms) di storia. NECESSARIA perche' la randomizzazione del baricentro
            # abbia effetto: uno sbilanciamento del CoM e' un parametro PERSISTENTE e non
            # osservabile in un singolo frame. Senza memoria la policy puo' solo essere
            # conservativa; con la storia puo' dedurre "sono inclinato in modo persistente
            # nonostante stia correggendo -> il mio baricentro e' spostato" e compensare.
            # Il deploy C++ la supporta gia' (chiave use_gym_history in observation_manager.h).
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        joint_effort = ObsTerm(func=mdp.joint_effort, scale=0.01)
        last_action = ObsTerm(func=mdp.last_action)
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": 0.8, "command_name": "base_velocity"})

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=1.0, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )
    # -- P1: TRACKING A DUE SCALE --------------------------------------------------------------
    # I due termini sopra hanno std=0.5 contro comandi di +-0.3 m/s laterali e +-0.5 rad/s: la
    # campana e' quasi piatta su tutto il range, quindi restare fermi rende quasi quanto inseguire.
    # Reward MEDIA in palio (differenza fra inseguimento perfetto e immobilita'), misurata:
    #     vx [-0.3,1.0] -> 0.455    wz [-0.5,0.5] -> 0.253    vy [-0.3,0.3] -> 0.108
    # e i guadagni misurati in sim2sim seguono lo stesso ordine: vx 0.77, wz 0.73, vy 0.63.
    # Per vy la reward in palio (0.108) e' addirittura INFERIORE al costo del solo
    # joint_deviation_hips (0.120): muoversi di lato era un affare in perdita.
    #
    # Questi due termini AGGIUNTIVI hanno campana stretta e portano il montepremi a
    #     vx 0.792 (1.7x)    wz 0.533 (2.1x)    vy 0.272 (2.5x)
    # senza toccare quelli originali -- necessario perche' i curriculum leggono i termini PER NOME
    # (`get_term_cfg("track_lin_vel_xy")`) e girano gia' al limite del loro gate:
    #     track_lin_vel_xy 0.841 vs soglia 0.800   |   track_ang_vel_z 0.609 vs soglia 0.550
    # Stringere lo std di quelli esistenti avrebbe chiuso entrambi i gate facendo collassare il
    # range dei comandi. Peso 0.5 e non 1.0: aggiunge ~0.8 al bilancio positivo (oggi 6.48) e
    # sposta gia' abbastanza l'equilibrio verso la locomozione a scapito di track_base_height (2.94).
    track_lin_vel_xy_fine = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z_fine = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.25},
    )

    # -- ROUND 7: RETTILINEITA' -----------------------------------------------------------------
    # Misurato in sim2sim (sim_2026-09-10_18-08-39): su percorsi di 2.3-3.0 m il robot esce dalla
    # retta di 15-52 cm. Due cause indipendenti, entrambe INVISIBILI ai termini di tracking sopra:
    #
    #   A) deriva di imbardata non comandata, omega ~= +0.72 - 4.45*vx + 6.52*vy [deg/s]
    #   B) creep costante nel body frame di (+0.02 avanti, -0.02 a destra) m/s, presente in ogni
    #      modalita' di cammino e assente da fermo -> durante un comando di yaw puro il bacino
    #      percorre un cerchio di raggio |v|/omega ~ 10 cm invece di girare sul posto.
    #
    # Con std 0.5/0.25 una deriva di 2 deg/s mette in palio 0.0146/step e un creep di 0.02 m/s ne
    # mette 0.0048, su ~115 di reward totale: sono sotto la risoluzione del reward, non e' una
    # questione di peso. E stringere lo std sul RATE sarebbe controproducente, perche' le due
    # semifasi del passo differiscono gia' di 2-7 deg/s: si punirebbe l'andatura sana. Per questo
    # i due termini qui sotto lavorano sull'errore INTEGRATO sulla finestra di comando (10 s),
    # dove l'oscillazione per-passo si media a zero e resta solo la deriva.
    #
    # ROUND 8: entrambi riformulati sull'errore MEDIO (err/tempo) invece che accumulato, e
    # path_hold portato da 0.3 a 1.0. Misurato sul confronto a cinque policy con lo stesso
    # protocollo (sim_logs/automatic, 2026-09-13):
    #   - heading_hold ha funzionato A META': accoppiamenti collassati (vy +6.21 -> -1.07,
    #     vx -1.54 -> -0.17) ma |omega| solo da 1.70+-0.18 a 1.23+-0.22 deg/s. Colpa della
    #     saturazione dell'integrale: a 1.7 deg/s su 9.5 s l'errore e' 0.28 rad contro std 0.25.
    #   - path_hold e' stato CONTROPRODUCENTE: creep laterale da 0.0165+-0.0014 (round 6, senza
    #     il termine) a 0.0309+-0.0033 m/s, 4 sigma di peggioramento. E in addestramento il suo
    #     valore e' CALATO (0.183 -> 0.168): a peso 0.3 la policy lo ha semplicemente barattato.
    # La media non satura, non dipende dalla lunghezza della finestra, e gli std sono ora in
    # unita' fisiche dirette (rad/s e m/s) tarate sui valori misurati.
    # heading_hold e' a DUE SCALE, come track_lin_vel/track_ang_vel dopo P1, e per la stessa
    # ragione. Invertendo il valore del termine nel round 7 (0.1933 a iterazione 1000, peso 0.5)
    # la deriva implicita a inizio addestramento e' 3.46 deg/s: a std 0.02 il reward li' vale
    # 0.000 con gradiente 0.0, cioe' una zona morta che sprecherebbe meta' del round. La campana
    # larga tiene vivo il gradiente da 3.5 deg/s in giu' (0.298 con gradiente 10.9), quella
    # stretta da' la precisione sotto 1.5 deg/s dove la larga e' ormai piatta.
    #   deriva   larga(0.06)   stretta(0.02)   totale in palio
    #   3.46       0.298          0.000            0.089
    #   1.70       0.783          0.111            0.291
    #   1.23       0.880          0.316            0.422
    #   0.50       0.979          0.827            0.708
    heading_hold = RewTerm(
        func=mdp.heading_hold,
        weight=0.3,
        params={"command_name": "base_velocity", "std": 0.06},
    )
    heading_hold_fine = RewTerm(
        func=mdp.heading_hold,
        weight=0.5,
        params={"command_name": "base_velocity", "std": 0.02},
    )
    # path_hold resta a scala singola: invertendo il suo valore nel round 7 il creep implicito e'
    # 0.0292 m/s a iterazione 1000 e 0.0324 a fine (contro 0.0309+-0.0033 misurati in deploy --
    # buona verifica incrociata), e a std 0.03 quel livello vale 0.34 con gradiente 23.7. Nessuna
    # zona morta. Da notare che il creep e' PEGGIORATO durante l'addestramento (0.0292 -> 0.0324):
    # a peso 0.3 il termine veniva barattato, da cui il passaggio a 1.0.
    path_hold = RewTerm(
        func=mdp.path_hold,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.03},
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    # P5 ANNULLATA: tornata a -0.05. Al round 5 l'avevo abbassata a -0.035 pensando fosse una
    # penalita' sul MOVIMENTO. Misurato: non lo e'. Azioni consecutive sono campionate in modo
    # indipendente, quindi il rumore di esplorazione contribuisce da solo 2*sigma^2*n_giunti --
    #     round 4: sigma=0.342 -> contributo 0.316 contro 0.323 misurati = 98%
    #     round 5: sigma=0.453 -> contributo 0.387 contro 0.410 misurati = 95%
    # cioe' action_rate misura quasi solo il rumore della policy: e' un regolarizzatore di entropia
    # implicito. Abbassarne il peso ha reso l'esplorazione piu' economica e sigma e' salito del 32%
    # (l'equilibrio previsto sigma* = sqrt(entropy_coef/(4w)) da' +20%, il resto lo spiegano
    # joint_vel e joint_acc). Conseguenze misurate sulla policy ESPORTATA, quindi deterministica:
    # action_rate grezza da fermo +137%, sway wy da fermo +79%, |wy| massima +68%, gait -23%.
    # Se serve piu' liberta' di movimento la leva onesta e' entropy_coef, non questa.
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    # FASE 2 (anti-barcollio, per FINETUNE da baseline): penalizza deviazione da default SOLO sugli
    # env veri-fermi (is_standing_env), non su cmd<0.1 -> non combatte il tracking, niente deadlock.
    stand_still_zero_cmd = RewTerm(
        func=mdp.stand_still_zero_cmd,
        weight=-0.5,
        params={"command_name": "base_velocity"},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_shoulder_pitch.*",
                    ".*_shoulder_roll.*",
                    ".*_shoulder_yaw.*",
                    ".*_elbow.*",
                    ".*_wrist.*",
                ],
            )
        },
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["torso_joint"])},
    )
    # -- P2: hip_roll e hip_yaw separati ---------------------------------------------------------
    # Erano un unico termine a -1.0 su entrambi, cioe' una penalita' su ESATTAMENTE i giunti che
    # servono a eseguire i comandi: hip_roll per camminare di lato, hip_yaw per girare. Costava
    # 0.120 alla convergenza, piu' dell'intera reward media in palio per il canale vy (0.108).
    # hip_roll resta a -1.0: e' la guardia contro la spaccata (vedi V7.2), e comunque il lato
    # laterale ha gia' il rinforzo di roll_symmetry e feet_too_near.
    # hip_yaw scende a -0.25: deviare di imbardata serve SOLO quando il giro e' comandato, quindi
    # a -1.0 era una tassa diretta sul comando wz.
    joint_deviation_hip_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_roll_joint"])},
    )
    # ROUND 8: -0.25 -> -0.15. hip_yaw e' il giunto con cui la policy sterza, ed e' esattamente
    # cio' che heading_hold le chiede di fare: nel round 7 la deviazione di hip_yaw e' salita del
    # 68% (0.0223 -> 0.0375) proprio mentre heading_hold veniva ottimizzato (0.193 -> 0.328). I due
    # termini si remano contro. Stessa logica gia' scritta nel docstring di joint_mirror_antisym:
    # "Do NOT apply to hip_yaw: turning legitimately needs L/R yaw asymmetry".
    joint_deviation_hip_yaw = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.98})

    # -- feet
    gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.5,
        params={
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )
    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        },
    )
    feet_contact_forces = RewTerm(
        func=mdp.contact_forces,
        weight=-0.0002,
        params={
            "threshold": 500,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ankle_roll.*"),
        },
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    "torso_link",
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    lin_vel_cmd_levels = CurrTerm(mdp.lin_vel_cmd_levels)


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # --- Robot Cfg ---
        self.gait_f = 2.0

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 2
        self.scene.terrain.terrain_generator.num_cols = 10
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
