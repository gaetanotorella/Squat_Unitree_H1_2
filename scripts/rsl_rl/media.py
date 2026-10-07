# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Immagini e video della policy per la tesi e la presentazione.

Carica la policy come play.py ma NON esporta nulla: exported/policy.onnx e params/deploy.yaml del run
(tracciati in git e letti dal controller del robot reale) non vengono toccati, quindi non serve una
copia _media del run. Le modifiche di ogni preset si applicano alla configurazione di play IN MEMORIA:
squat_env_cfg.py resta com'e'.

Uso (dalla radice del repo, ambiente conda di Isaac Lab attivo):

    python scripts/rsl_rl/media.py --preset S1 --headless
    python scripts/rsl_rl/media.py --preset V3 --headless --preview     # un fotogramma, per l'inquadratura
    scripts/rsl_rl/media_all.sh                                         # tutti i preset in media_tesi/media_N

Preset (i file finiscono in media_tesi/, con accanto un .json che dice come sono stati fatti):

    S1   32 robot su griglia, camera obliqua dall'alto, 5 istanti x 2 versioni (frecce / pulito), 4K
    V1   4096 robot, camera alta e lontana, 12 s
    V2   32 robot, stessa camera di S1, 18 s (un solo episodio: niente reset simultaneo a 20 s)
    V3   squat da fermo, sequenza di quote fissa ogni 4 s, camera laterale, 30 s
    V4a  camminata accovacciata a 0.75 m, vx 0.5, camera laterale
    V4b  come V4a ma vy 0.3, camera frontale
    V4c  come V4a ma rotazione sul posto wz 0.5, camera laterale
    V5a  spinte da fermo a 0.98 m ogni 3 s, camera laterale
    V5b  spinte da fermo a 0.70 m ogni 3 s, camera laterale

Cosa cambia rispetto a play.py, per tutti i preset:
  * terreno piano con i robot su una griglia (env_spacing): col terreno generato i robot nascono al
    centro delle tessere da 8 m, piu' robot nella stessa tessera, uno dentro l'altro;
  * seed fisso (--seed), cosi' due lanci danno la stessa scena e si puo' ritoccare solo la camera;
  * i video si scrivono un passo = un fotogramma (50 fps nel tempo simulato), qualunque sia la
    velocita' del PC.
Per i preset a un robot (V3-V5), inoltre:
  * orientamento e posizione al reset fissi (yaw 0): il robot avanza verso +x, di lato alla camera;
  * episodio lungo quanto il video, altrimenti a 20 s il robot viene riportato al punto di partenza;
  * curriculum spenti, randomizzazioni fisiche spente (--dr on per riaccenderle): robot nominale;
  * camera che segue solo x/y del robot, ad altezza fissa (non sale e scende con lo squat).

Aspetto (--look): 'tesi' (default) = robot nero satinato, pavimento chiaro con la griglia bianca del
piano di Isaac (colore del fondo calibrato in automatico su --ground_rgb), sole con ombre nette dietro il soggetto a destra (come la foto di riferimento), cielo
uniforme chiaro; 'isaac' = scena di default di Isaac Lab. Luce e robot si regolano con --sun_*,
--sky_intensity, --robot_*; --fstop accende la profondita' di campo.

Le opzioni --cam_* e --focal_length ritoccano l'inquadratura senza toccare il codice; --ground sceglie
il pavimento (grid = default di Isaac Lab, grey = grigio uniforme, marble = piastrelle del training);
--arrow_scale ingrandisce le frecce.

Resa (dal primo giro di test, 2026-10-07):
  * prima di iniziare si renderizza finche' l'immagine smette di cambiare: materiali e cielo si
    caricano in background, e il primo fotogramma dei video era la scena non ancora caricata;
  * il renderer media ogni fotogramma con i precedenti: gli scatti fermi usano --settle render, e la
    versione pulita di S1 --settle_toggle render dopo aver spento le frecce (con 16 ne restava un
    fantasma al ~10%); lo script conta i pixel nei colori delle frecce e avvisa se ne restano;
  * camere dal lato del sole (vedi LUCE): la prima camera laterale era in controluce.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

FINAL_RUN = "2026-09-13_21-27-39"

# camere: "grid" = fissa, obliqua sul centro della griglia; "follow" = segue x/y del robot 0.
# azim: direzione IN CUI sta la camera rispetto al bersaglio (0 = +x, 90 = +y).
# LUCE: la scena ha solo la cupola con il cielo HDR; dalle ombre dei primi test il sole sta dal lato
# +y / -x (azimut ~110-135 deg). Le camere stanno quindi da quel lato, col sole alle spalle: con la
# camera laterale a -y (primo test) il robot era in controluce, una sagoma scura contro il cielo.
# I robot dei preset singoli guardano verso -x (yaw = pi): con la camera laterale a +y avanzano
# comunque verso DESTRA nell'immagine.
# frac (solo "grid"): dove sta il bersaglio, come frazione della distanza centro -> bordo della griglia
# dal lato della camera. 0 = centro; V1 a 0.8 guarda vicino all'angolo: i robot in primo piano si
# riconoscono e la griglia continua fino all'orizzonte (a 130 m dal centro erano puntini).
CAM_GRID_32 = dict(mode="grid", dist=16.0, elev=35.0, azim=135.0, look_z=0.5, frac=0.0)
CAM_GRID_4096 = dict(mode="grid", dist=30.0, elev=22.0, azim=135.0, look_z=0.8, frac=0.8)
CAM_SIDE = dict(mode="follow", dist=3.8, azim=90.0, eye_z=1.4, look_z=0.8)
CAM_FRONT = dict(mode="follow", dist=3.8, azim=180.0, eye_z=1.4, look_z=0.8)

# Gruppi di modifiche alla configurazione: (percorso puntato, valore). Un percorso che non esiste
# ferma lo script con un errore: un nome sbagliato non deve passare in silenzio.
STANDING = [
    ("commands.base_velocity.rel_standing_envs", 1.0),
    ("commands.base_velocity.debug_vis", False),
]
NO_CURRICULUM = [("curriculum.lin_vel_cmd_levels", None), ("curriculum.height_cmd_levels", None)]
YAW_MINUS_X = 3.141592653589793  # robot rivolto verso -x (vedi LUCE sopra)
FIXED_RESET = [("events.reset_base.params.pose_range",
                {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (YAW_MINUS_X, YAW_MINUS_X)})]
NO_DR = [
    ("events.physics_material", None),
    ("events.add_base_mass", None),
    ("events.randomize_link_masses", None),
    ("events.randomize_actuator_gains", None),
    ("events.randomize_base_com", None),
]


def crouch_walk(vx, vy, wz):
    return [
        ("commands.base_velocity.rel_standing_envs", 0.0),
        ("commands.base_velocity.rel_turn_in_place_envs", 0.0),
        ("commands.base_velocity.ranges.lin_vel_x", (vx, vx)),
        ("commands.base_velocity.ranges.lin_vel_y", (vy, vy)),
        ("commands.base_velocity.ranges.ang_vel_z", (wz, wz)),
        ("commands.base_height.only_when_standing", False),
        ("commands.base_height.ranges.height", (0.75, 0.75)),
        ("events.push_robot", None),
    ]


# HEADING_HOLD. La policy insegue la VELOCITA' di imbardata, non la direzione: un errore di velocita'
# si integra in un errore di direzione che nessuno corregge. Camminando di lato a mezzo squat c'e' anche
# un accoppiamento laterale -> imbardata. Misurato in MuJoCo sulla policy finale (sim_2026-09-14_01-09-57,
# 0.74 m): vy +0.3 comandato -> 0.14 m/s e +0.13 rad/s di imbardata non comandata, +54 deg in 7 s;
# deriva ~ 0.8 - 4.5*vx + 11.2*vy deg/s. Nel primo V4b (media_1) il robot girava in tondo su ~1 m.
# Il rimedio e' quello dell'operatore col joystick: correggere la rotta. Isaac Lab lo ha gia' nel
# generatore di comandi (heading_command): wz = clip(k * errore di direzione, ang_vel_z). La policy vede
# solo un comando di imbardata piccolo e dentro il range di training (+-0.5 rad/s).
def heading_hold(k):
    return [
        ("commands.base_velocity.heading_command", True),
        ("commands.base_velocity.rel_heading_envs", 1.0),
        ("commands.base_velocity.heading_control_stiffness", k),
        ("commands.base_velocity.ranges.heading", (YAW_MINUS_X, YAW_MINUS_X)),  # la direzione del reset
        ("commands.base_velocity.ranges.ang_vel_z", (-0.5, 0.5)),  # limite della correzione = range di training
    ]


SINGLE = dict(num_envs=1, preroll=1.0, dr=False)
PRESETS = {
    "S1": dict(kind="stills", name="S1_envs", num_envs=32, spacing=4.0, res=(3840, 2160), preroll=0.0,
               # presto: a 8 s (primo test) chi cammina ha gia' lasciato la griglia e i robot si incrociano
               dr=True, cam=CAM_GRID_32, arrows="both", times=(2.5, 3.0, 3.5, 4.0, 5.0), mods=[]),
    "V1": dict(kind="video", name="V1_scala", num_envs=4096, spacing=2.5, duration=12.0, preroll=1.0,
               dr=True, cam=CAM_GRID_4096, arrows="off", mods=[]),
    "V2": dict(kind="video", name="V2_misti", num_envs=32, spacing=4.0, duration=18.0, preroll=1.0,
               dr=True, cam=CAM_GRID_32, arrows="on", mods=[]),
    "V3": dict(kind="video", name="V3_squat", duration=30.0, cam=CAM_SIDE, arrows="off", **SINGLE,
               heights="0.98,0.60,0.98,0.80,0.65,0.90,0.70,0.98", segment=4.0,
               mods=STANDING + [("commands.base_height.ranges.height", (0.60, 0.98)),
                                ("commands.base_height.resampling_time_range", (4.0, 4.0)),
                                ("events.push_robot", None)]),
    # hold: controllo di rotta (vedi HEADING_HOLD). Solo dove la direzione deve restare fissa; V4c ruota.
    "V4a": dict(kind="video", name="V4a_crouch_avanti", duration=30.0, cam=CAM_SIDE, arrows="on", hold=True, **SINGLE,
                mods=crouch_walk(0.5, 0.0, 0.0)),
    "V4b": dict(kind="video", name="V4b_crouch_laterale", duration=30.0, cam=CAM_FRONT, arrows="on", hold=True, **SINGLE,
                mods=crouch_walk(0.0, 0.3, 0.0)),
    "V4c": dict(kind="video", name="V4c_crouch_rotazione", duration=30.0, cam=CAM_SIDE, arrows="on", **SINGLE,
                mods=crouch_walk(0.0, 0.0, 0.5)),
    # le spinte le da' lo script (evento spento): stesso effetto di push_by_setting_velocity, ma con
    # tempi e ampiezze noti e registrati nel .json
    "V5a": dict(kind="video", name="V5a_spinte_098", duration=30.0, cam=CAM_SIDE, arrows="off", **SINGLE,
                push_interval=3.0, mods=STANDING + [("commands.base_height.ranges.height", (0.98, 0.98)),
                                                    ("events.push_robot", None)]),
    "V5b": dict(kind="video", name="V5b_spinte_070", duration=30.0, cam=CAM_SIDE, arrows="off", **SINGLE,
                push_interval=3.0, mods=STANDING + [("commands.base_height.ranges.height", (0.70, 0.70)),
                                                    ("events.push_robot", None)]),
}

parser = argparse.ArgumentParser(description="Immagini e video della policy (nessun export).")
parser.add_argument("--task", type=str, default="Unitree-H1_2-Squat")
parser.add_argument("--preset", type=str, required=True, choices=list(PRESETS))
parser.add_argument("--out", type=str, default=None, help="Cartella di uscita (default: <repo>/media_tesi).")
parser.add_argument("--overwrite", action="store_true", help="Sovrascrive i file esistenti invece di numerarli.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--spacing", type=float, default=None, help="Distanza fra i robot sulla griglia [m].")
parser.add_argument("--duration", type=float, default=None, help="Durata del video [s].")
parser.add_argument("--resolution", type=int, nargs=2, default=None, metavar=("W", "H"))
parser.add_argument("--arrows", choices=("on", "off"), default=None, help="Frecce dei comandi di velocita'.")
parser.add_argument("--dr", choices=("on", "off"), default=None, help="Randomizzazioni fisiche del training.")
parser.add_argument("--times", type=float, nargs="+", default=None, help="S1: istanti degli scatti [s].")
parser.add_argument("--heights", type=str, default=None, help="V3: quote separate da virgole, oppure 'random'.")
parser.add_argument("--push", choices=("train", "alternate"), default="train",
                    help="V5: 'train' = x,y uniformi in +-push_speed come in training; 'alternate' = "
                         "+-push_speed lungo x a segni alterni (la direzione che si vede di lato).")
parser.add_argument("--push_speed", type=float, default=0.5, help="V5: ampiezza delle spinte [m/s].")
parser.add_argument("--heading_hold", choices=("on", "off"), default=None,
                    help="Controllo di rotta nel generatore di comandi (default: on per V4a e V4b). 'off' mostra la "
                         "policy a comando di imbardata nullo, con la sua deriva.")
parser.add_argument("--heading_stiffness", type=float, default=4.0,
                    help="Guadagno del controllo di rotta [rad/s per rad]. Simulato con i valori misurati a mezzo "
                         "squat (risposta in imbardata 0.42, deriva 0.13 rad/s, ritardo 0.1-0.2 s): con 4.0 errore "
                         "a regime 3-5 deg e comando massimo 0.35 rad/s; col default di Isaac Lab (0.5) 35 deg; "
                         "da 10 in su il comando satura e oscilla. Senza controllo, 223 deg in 30 s.")
parser.add_argument("--cam_dist", type=float, default=None)
parser.add_argument("--cam_elev", type=float, default=None, help="Camere 'grid': elevazione [deg].")
parser.add_argument("--cam_azim", type=float, default=None, help="Direzione della camera [deg], 0 = +x.")
parser.add_argument("--cam_eye_z", type=float, default=None, help="Camere 'follow': altezza della camera [m].")
parser.add_argument("--cam_look_z", type=float, default=None, help="Altezza del punto inquadrato [m].")
parser.add_argument("--cam_smooth", type=float, default=0.4, help="Camere 'follow': costante di tempo [s].")
parser.add_argument("--focal_length", type=float, default=None, help="Focale della camera [mm] (default Kit 18.1).")
parser.add_argument("--look", choices=("tesi", "isaac"), default="tesi",
                    help="'tesi' = robot nero, pavimento chiaro, sole con ombre nette e cielo uniforme; "
                         "'isaac' = scena di default (cielo HDR, robot grigi).")
parser.add_argument("--ground", choices=("lightgrid", "grid", "grey", "marble"), default=None,
                    help="Pavimento: 'lightgrid' = la griglia bianca del piano di default su fondo chiaro "
                         "calibrato al colore --ground_rgb; 'grid' = piano di default (nero a griglia); 'grey' = "
                         "grigio opaco uniforme; 'marble' = materiale del terreno di training (sul piano appare "
                         "uniforme). Default: lightgrid con --look tesi, grid con --look isaac.")
parser.add_argument("--ground_rgb", type=float, nargs=3, default=(206.0, 204.0, 203.0), metavar=("R", "G", "B"),
                    help="lightgrid: colore del fondo come deve uscire nell'immagine (sRGB 0-255). Default: il "
                         "pavimento misurato nelle anteprime del 2026-10-07 (mediana su V3, S1, V1).")
parser.add_argument("--robot_color", type=float, default=0.02, help="--look tesi: albedo del robot (lineare, 0-1).")
parser.add_argument("--robot_roughness", type=float, default=0.45)
parser.add_argument("--robot_metallic", type=float, default=0.2)
parser.add_argument("--sun_elev", type=float, default=35.0, help="--look tesi: altezza del sole [deg].")
parser.add_argument("--sun_rel_azim", type=float, default=135.0,
                    help="--look tesi: direzione del sole rispetto alla camera [deg]; 135 = dietro il soggetto, "
                         "a destra nell'immagine (ombre verso la camera, come nella foto di riferimento); "
                         "45 = alle spalle della camera, a destra.")
parser.add_argument("--sun_intensity", type=float, default=3000.0)
parser.add_argument("--sun_angle", type=float, default=0.53, help="Diametro apparente del sole [deg]: ombre nette.")
parser.add_argument("--sky_intensity", type=float, default=800.0, help="--look tesi: luce diffusa del cielo uniforme.")
parser.add_argument("--fstop", type=float, default=0.0,
                    help="Profondita' di campo: f-stop della camera (0 = spenta), fuoco sul bersaglio.")
parser.add_argument("--ground_shade", type=float, default=0.35, help="--ground grey: grigio del pavimento (0-1).")
parser.add_argument("--arrow_scale", type=float, default=None, help="Spessore/lunghezza delle frecce (Isaac Lab: 0.5).")
parser.add_argument("--settle", type=int, default=32, help="Render ripetuti sullo stesso stato per uno scatto fermo.")
parser.add_argument("--settle_toggle", type=int, default=128,
                    help="Render dopo aver spento le frecce: il renderer media i fotogrammi e con 16 restava "
                         "un fantasma delle frecce al ~10%%.")
parser.add_argument("--warmup", type=int, default=300, help="Massimo di render iniziali per caricare materiali e cielo.")
parser.add_argument("--preview", action="store_true", help="Salva un solo fotogramma ed esce.")
parser.add_argument("--preview_at", type=float, default=3.0, help="Istante del fotogramma di --preview [s].")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # serve il render offscreen anche con --headless

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import copy
import datetime
import gymnasium as gym
import json
import math
import numpy as np
import os
import shutil
import subprocess
import torch
from importlib.metadata import version
from pathlib import Path

from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------------
# configurazione
# --------------------------------------------------------------------------------------------------
def set_path(cfg, path: str, value, log: list):
    """cfg.a.b.c = value, attraverso attributi e dizionari. Errore se un pezzo del percorso manca."""
    parts = path.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = obj[p] if isinstance(obj, dict) else getattr(obj, p)
    last = parts[-1]
    if isinstance(obj, dict):
        if last not in obj:
            raise KeyError(f"config: '{path}' non esiste")
        obj[last] = value
    else:
        if not hasattr(obj, last):
            raise AttributeError(f"config: '{path}' non esiste")
        setattr(obj, last, value)
    log.append(f"{path} = {value!r}")


def build_env_cfg(preset: dict, args, log: list):
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs or preset["num_envs"],
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args.seed

    # In play `ranges` e' lo STESSO oggetto di `limit_ranges` (SquatPlayEnvCfg li assegna l'uno all'altro):
    # una modifica a ranges cambierebbe anche limit_ranges. Qui non si esporta nulla, ma le separiamo
    # lo stesso, cosi' i preset toccano solo il range di campionamento.
    for term in ("base_velocity", "base_height"):
        t = getattr(env_cfg.commands, term, None)
        if t is not None:
            t.ranges = copy.deepcopy(t.ranges)

    # terreno piano + griglia: col terreno generato le origini sono i centri delle tessere da 8 m
    set_path(env_cfg, "scene.terrain.terrain_type", "plane", log)
    set_path(env_cfg, "scene.terrain.terrain_generator", None, log)
    set_path(env_cfg, "scene.env_spacing", args.spacing or preset.get("spacing", 4.0), log)
    if getattr(env_cfg.curriculum, "terrain_levels", None) is not None:  # richiede il terreno generato
        set_path(env_cfg, "curriculum.terrain_levels", None, log)
    # il ray caster non entra in nessuna osservazione o reward: via, cosi' non dipende dal tipo di terreno
    if getattr(env_cfg.scene, "height_scanner", None) is not None:
        set_path(env_cfg, "scene.height_scanner", None, log)

    single = preset["num_envs"] == 1
    if single:
        for path, value in NO_CURRICULUM + FIXED_RESET:
            set_path(env_cfg, path, value, log)
    for path, value in preset["mods"]:
        set_path(env_cfg, path, value, log)

    hold = (args.heading_hold or ("on" if preset.get("hold") else "off")) == "on"
    if hold:
        for path, value in heading_hold(args.heading_stiffness):
            set_path(env_cfg, path, value, log)

    dr = (args.dr or ("on" if preset["dr"] else "off")) == "on"
    if not dr:
        for path, value in NO_DR:
            set_path(env_cfg, path, value, log)

    arrows = args.arrows or preset["arrows"]
    if arrows in ("on", "off"):
        set_path(env_cfg, "commands.base_velocity.debug_vis", arrows == "on", log)
    else:  # "both": accese, si spengono solo per lo scatto pulito
        set_path(env_cfg, "commands.base_velocity.debug_vis", True, log)
    if args.arrow_scale:
        s = float(args.arrow_scale)
        for vis in ("goal_vel_visualizer_cfg", "current_vel_visualizer_cfg"):
            getattr(env_cfg.commands.base_velocity, vis).markers["arrow"].scale = (s, s, s)
        log.append(f"frecce: scala {s:g}")

    # episodio piu' lungo del video: niente reset (teletrasporto) a meta' ripresa
    if preset["kind"] == "video":
        need = preset["preroll"] + (args.duration or preset["duration"]) + 2.0
        if env_cfg.episode_length_s < need:
            set_path(env_cfg, "episode_length_s", need, log)

    if args.look == "tesi":
        # cielo uniforme chiaro al posto dell'HDR: fa da luce diffusa e da sfondo neutro; la luce
        # direzionale (il sole) si aggiunge dopo la creazione della scena, in add_sun()
        import isaaclab.sim as sim_utils

        set_path(env_cfg, "scene.sky_light.spawn", sim_utils.DomeLightCfg(
            intensity=args.sky_intensity, color=(0.86, 0.88, 0.92), texture_file=None), log)

    res = tuple(args.resolution) if args.resolution else preset.get("res", (1920, 1080))
    set_path(env_cfg, "viewer.resolution", res, log)
    set_path(env_cfg, "viewer.origin_type", "world", log)  # la camera la muove lo script
    return env_cfg, dr, arrows


# --------------------------------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------------------------------
class Camera:
    def __init__(self, base_env, cam: dict, args):
        self.sim = base_env.sim
        self.robot = base_env.scene["robot"]
        self.p = dict(cam)
        for k in ("dist", "elev", "azim", "eye_z", "look_z"):
            v = getattr(args, f"cam_{k}")
            if v is not None:
                self.p[k] = v
        self.alpha = base_env.step_dt / max(args.cam_smooth + base_env.step_dt, 1e-6)
        self.xy = None
        self.eye = self.look = None
        self.usd_cam = None
        if args.fstop > 0:  # profondita' di campo: f-stop fisso, fuoco aggiornato sul bersaglio
            import omni.usd
            from pxr import UsdGeom

            prim = omni.usd.get_context().get_stage().GetPrimAtPath(base_env.cfg.viewer.cam_prim_path)
            self.usd_cam = UsdGeom.Camera(prim)
            self.usd_cam.GetFStopAttr().Set(float(args.fstop))
        if self.p["mode"] == "grid":
            o = base_env.scene.env_origins.cpu().numpy()
            c = o.mean(axis=0)
            e, a = math.radians(self.p["elev"]), math.radians(self.p["azim"])
            u = np.array([math.cos(a), math.sin(a)])
            reach = float(((o[:, :2] - c[:2]) @ u).max())  # dal centro al bordo, verso la camera
            t = c[:2] + self.p.get("frac", 0.0) * reach * u
            self.look = np.array([t[0], t[1], self.p["look_z"]])
            self.eye = self.look + self.p["dist"] * np.array(
                [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        self.update()

    def update(self):
        if self.p["mode"] == "follow":
            xy = self.robot.data.root_pos_w[0, :2].cpu().numpy()
            self.xy = xy if self.xy is None else self.xy + self.alpha * (xy - self.xy)
            a = math.radians(self.p["azim"])
            self.look = np.array([self.xy[0], self.xy[1], self.p["look_z"]])
            self.eye = np.array([self.xy[0] + self.p["dist"] * math.cos(a),
                                 self.xy[1] + self.p["dist"] * math.sin(a), self.p["eye_z"]])
        self.sim.set_camera_view(tuple(float(v) for v in self.eye), tuple(float(v) for v in self.look))
        if self.usd_cam is not None:
            self.usd_cam.GetFocusDistanceAttr().Set(float(np.linalg.norm(self.eye - self.look)))

    def describe(self):
        return {**self.p, "eye": [round(float(v), 3) for v in self.eye], "lookat": [round(float(v), 3) for v in self.look]}


def srgb_to_lin(c):
    c = np.asarray(c, dtype=float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def set_input(prim, name: str, type_name, value):
    """Imposta un input di shader; se il prim e' dentro un'istanza lo scrive nel layer di origine."""
    from pxr import Sdf

    if not prim.IsInstanceProxy():
        attr = prim.GetAttribute(name)
        if not attr:
            attr = prim.CreateAttribute(name, type_name)
        attr.Set(value)
        return
    spec = prim.GetPrimStack()[0]
    a = spec.attributes[name] if name in spec.attributes else Sdf.AttributeSpec(spec, name, type_name)
    a.default = value


def find_grid_shader(ground_root: str):
    """Shader 'theGrid' del piano di default di Isaac (Environments/Grid/default_environment.usd):
    OmniPBR proiettato in coordinate mondo. Le linee bianche sono EMISSIVE (maschera WireframeBlur,
    intensita' 1000); il fondo e' una texture blu scuro moltiplicata per diffuse_tint, che Isaac Lab
    mette a nero. Per un fondo chiaro si toglie la texture e si usa diffuse_color_constant."""
    import omni.usd
    from pxr import Usd

    root = omni.usd.get_context().get_stage().GetPrimAtPath(ground_root)
    if not root.IsValid():
        return None
    for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Shader" and p.GetParent().GetName() == "theGrid":
            return p
    return None


def apply_ground(kind: str, env_cfg, shade: float):
    """Materiale del pavimento sopra il piano di default. Il piano di Isaac Lab ('grid') e' nero con
    una griglia bianca a 1 m: molto contrastato, e in V1 da lontano le linee sfarfallano.
    Restituisce (descrizione, shader della griglia da calibrare o None)."""
    if kind == "grid":
        return "piano di default (nero a griglia)", None
    if kind == "lightgrid":
        from pxr import Gf, Sdf

        shader = find_grid_shader(env_cfg.scene.terrain.prim_path)
        if shader is None:
            print("[media] ATTENZIONE: shader 'theGrid' non trovato, resta il pavimento di default")
            return "piano di default (shader della griglia non trovato)", None
        set_input(shader, "inputs:diffuse_texture", Sdf.ValueTypeNames.Asset, Sdf.AssetPath(""))
        set_input(shader, "inputs:diffuse_tint", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(1.0, 1.0, 1.0))
        set_input(shader, "inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(0.5, 0.5, 0.5))
        return "griglia bianca su fondo chiaro (colore calibrato)", shader
    import isaaclab.sim as sim_utils

    mat_path = "/World/Looks/media_ground"
    if kind == "grey":
        mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(shade, shade, shade), roughness=0.9, metallic=0.0)
        desc = f"grigio uniforme {shade:g}"
    else:
        mat = env_cfg.scene.terrain.visual_material  # lo stesso MDL del terreno di training
        desc = "piastrelle del training (MDL)"
    try:
        mat.func(mat_path, mat)
        sim_utils.bind_visual_material(env_cfg.scene.terrain.prim_path, mat_path)
    except Exception as e:  # meglio un'immagine col pavimento di default che un lancio perso
        print(f"[media] ATTENZIONE: pavimento '{kind}' non applicato ({e}); resta quello di default")
        return f"piano di default (fallito '{kind}': {e})", None
    return desc, None


def calibrate_floor(base, cam, shader, target_srgb, settle, iters: int = 6) -> dict:
    """Regola il colore del fondo finche' NELL'IMMAGINE esce target_srgb (entro 1 livello su 255).

    Il colore renderizzato dipende da luce, esposizione e riflesso del materiale, quindi non si puo'
    impostare a priori: si guarda un tratto di pavimento vuoto (camera spostata di 500 m, stesso
    orientamento dell'inquadratura vera), si misura la mediana di una fascia centrale -- le linee della
    griglia sono sottili e non la spostano -- e si corregge in spazio lineare, dove il diffuso e'
    proporzionale al colore del materiale."""
    from pxr import Gf, Sdf

    target = np.asarray(target_srgb, dtype=float)
    target_lin = srgb_to_lin(target / 255.0)
    c = target_lin.copy()  # primo tentativo: albedo = colore voluto
    off = np.array([500.0, 500.0, 0.0])
    base.sim.set_camera_view(tuple(float(v) for v in cam.eye + off), tuple(float(v) for v in cam.look + off))
    steps, meas = [], target
    for _ in range(iters):
        set_input(shader, "inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, c)))
        img = settle(48)
        h, w = img.shape[:2]
        band = img[int(0.60 * h):int(0.95 * h), int(0.2 * w):int(0.8 * w)].reshape(-1, 3).astype(float)
        meas = np.median(band, axis=0)
        steps.append({"albedo": [round(float(v), 4) for v in c], "misurato_srgb": [round(float(v), 1) for v in meas]})
        if np.abs(meas - target).max() <= 1.0:
            break
        c = np.clip(c * target_lin / np.maximum(srgb_to_lin(meas / 255.0), 1e-4), 0.0, 1.0)
    cam.update()  # di nuovo sull'inquadratura vera
    ok = bool(np.abs(meas - target).max() <= 1.0)
    print(f"[media] pavimento: voluto sRGB {tuple(int(v) for v in target)}, ottenuto {tuple(int(round(v)) for v in meas)} "
          f"in {len(steps)} passi{'' if ok else '  <-- NON raggiunto (albedo al limite?)'}")
    return {"voluto_srgb": [float(v) for v in target], "ottenuto_srgb": [round(float(v), 1) for v in meas],
            "albedo_finale": steps[-1]["albedo"], "passi": steps, "raggiunto": ok}


def add_sun(cam_azim: float, args) -> dict:
    """Luce direzionale (il sole) con ombre nette. Una DistantLight illumina lungo il proprio -Z: la si
    ruota in modo che +Z punti verso il sole. L'azimut e' relativo alla camera, cosi' la luce resta la
    stessa rispetto all'inquadratura in ogni preset."""
    import isaaclab.sim as sim_utils

    az, el = math.radians(cam_azim + args.sun_rel_azim), math.radians(args.sun_elev)
    s = np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])  # verso il sole
    axis = np.cross([0.0, 0.0, 1.0], s)
    n = np.linalg.norm(axis)
    theta = math.acos(float(np.clip(s[2], -1.0, 1.0)))
    q = (1.0, 0.0, 0.0, 0.0) if n < 1e-9 else (math.cos(theta / 2), *(axis / n * math.sin(theta / 2)))
    cfg = sim_utils.DistantLightCfg(intensity=args.sun_intensity, angle=args.sun_angle, color=(1.0, 0.95, 0.88))
    cfg.func("/World/media_sun", cfg, orientation=tuple(float(v) for v in q))
    return {"azimut_assoluto_deg": round((cam_azim + args.sun_rel_azim) % 360, 1), "elevazione_deg": args.sun_elev,
            "intensita": args.sun_intensity, "angolo_deg": args.sun_angle, "cielo": args.sky_intensity}


def recolor_robot(robot_root: str, color: float, roughness: float, metallic: float) -> int:
    """Rende il robot nero modificando i materiali CONDIVISI dal modello convertito dall'URDF.

    I robot sono instanziati (una sola geometria per tutti gli env: 550k triangoli a robot, senza
    instancing 4096 robot sarebbero 2.2 miliardi), e un materiale legato dall'esterno potrebbe non
    arrivare dentro le istanze. Qui si cambia direttamente lo shader nel layer da cui tutte le istanze
    lo leggono, solo in memoria: il file convertito su disco non viene salvato.
    Restituisce quanti shader/mesh sono stati modificati (0 = niente trovato, il robot resta com'e')."""
    import omni.usd
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(robot_root)
    if not root.IsValid():
        print(f"[media] ATTENZIONE: {robot_root} non esiste, robot non ricolorato")
        return 0
    rgb = Gf.Vec3f(color, color, color)
    shaders, meshes = {}, {}
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() == "Shader":
            shaders[str(prim.GetPath())] = prim
        elif prim.IsA(UsdGeom.Gprim):
            meshes[str(prim.GetPath())] = prim
            mat = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if mat:
                for p in Usd.PrimRange(mat.GetPrim(), Usd.TraverseInstanceProxies()):
                    if p.GetTypeName() == "Shader":
                        shaders[str(p.GetPath())] = p

    def put(spec, name, tname, value):
        a = spec.attributes[name] if name in spec.attributes else Sdf.AttributeSpec(spec, name, tname)
        a.default = value

    done, seen = 0, set()
    for prim in shaders.values():
        stack = prim.GetPrimStack()
        if not stack or (stack[0].layer.identifier, stack[0].path) in seen:
            continue  # stesso shader del prototipo condiviso: basta una volta
        spec = stack[0]
        seen.add((spec.layer.identifier, spec.path))
        sid = UsdShade.Shader(prim).GetIdAttr().Get()
        src = str(prim.GetAttribute("info:mdl:sourceAsset").Get() or "")
        if sid == "UsdPreviewSurface":
            put(spec, "inputs:diffuseColor", Sdf.ValueTypeNames.Color3f, rgb)
            put(spec, "inputs:roughness", Sdf.ValueTypeNames.Float, roughness)
            put(spec, "inputs:metallic", Sdf.ValueTypeNames.Float, metallic)
        elif "OmniPBR" in src or "OmniSurface" in src or src.endswith(".mdl"):
            put(spec, "inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, rgb)
            put(spec, "inputs:reflection_roughness_constant", Sdf.ValueTypeNames.Float, roughness)
            put(spec, "inputs:metallic_constant", Sdf.ValueTypeNames.Float, metallic)
            if "inputs:diffuse_texture" in spec.attributes:
                spec.attributes["inputs:diffuse_texture"].default = Sdf.AssetPath("")
        else:
            continue
        done += 1
    for prim in meshes.values():  # mesh senza materiale: il renderer usa displayColor
        stack = prim.GetPrimStack()
        if stack and "primvars:displayColor" in stack[0].attributes:
            stack[0].attributes["primvars:displayColor"].default = Vt.Vec3fArray([rgb])
            done += 1
    print(f"[media] robot ricolorato: {done} shader/mesh modificati ({len(meshes)} mesh nel robot 0)")
    if done == 0:
        print("[media] ATTENZIONE: nessun materiale trovato nel robot: resta del colore di default")
    return done


def arrow_pixels(img: np.ndarray) -> int:
    """Pixel nei colori delle frecce (blu e verde saturi): nello scatto pulito devono essere ~0."""
    r, g, b = (img[..., k].astype(np.int16) for k in range(3))
    blue = (b > 120) & (b > r + 60) & (b > g + 60)
    green = (g > 120) & (g > r + 60) & (g > b + 60)
    return int((blue | green).sum())


def set_focal_length(cam_prim_path: str, f: float):
    import omni.usd
    from pxr import UsdGeom

    prim = omni.usd.get_context().get_stage().GetPrimAtPath(cam_prim_path)
    UsdGeom.Camera(prim).GetFocalLengthAttr().Set(float(f))


# --------------------------------------------------------------------------------------------------
# uscita
# --------------------------------------------------------------------------------------------------
def out_path(folder: Path, name: str, ext: str, overwrite: bool) -> Path:
    p = folder / f"{name}{ext}"
    k = 2
    while p.exists() and not overwrite:
        p = folder / f"{name}_v{k}{ext}"
        k += 1
    return p


def save_png(frame: np.ndarray, path: Path):
    from PIL import Image

    Image.fromarray(np.ascontiguousarray(frame)).save(path)
    print(f"[media] salvata {path}")


class Mp4Writer:
    """Fotogrammi RGB -> ffmpeg (H.264, yuv420p, crf 16). Senza ffmpeg salva i PNG e stampa il comando."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        self.path, self.n, self.proc, self.frames_dir = path, 0, None, None
        exe = shutil.which("ffmpeg")
        if exe is None:
            try:
                import imageio_ffmpeg

                exe = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                exe = None
        self.fps = fps
        if exe:
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                   "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "16",
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path)]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.frames_dir = path.with_suffix("")
            self.frames_dir.mkdir(parents=True, exist_ok=True)
            print(f"[media] ffmpeg non trovato: salvo i fotogrammi in {self.frames_dir}")

    def write(self, frame: np.ndarray):
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame[:, :, :3], dtype=np.uint8).tobytes())
        else:
            from PIL import Image

            Image.fromarray(np.ascontiguousarray(frame[:, :, :3])).save(self.frames_dir / f"{self.n:05d}.png")
        self.n += 1

    def close(self):
        if self.proc:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError(f"ffmpeg ha restituito {self.proc.returncode} per {self.path}")
            print(f"[media] salvato {self.path} ({self.n} fotogrammi, {self.n / self.fps:.1f} s)")
        else:
            print(f"[media] per il video: ffmpeg -framerate {self.fps:g} -i {self.frames_dir}/%05d.png "
                  f"-c:v libx264 -crf 16 -pix_fmt yuv420p {self.path}")


def git_commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


# --------------------------------------------------------------------------------------------------
def main():
    preset = PRESETS[args_cli.preset]
    log: list[str] = []
    env_cfg, dr, arrows = build_env_cfg(preset, args_cli, log)

    # V3: sequenza di quote. La prima va nel range PRIMA di creare l'env, perche' il reset iniziale
    # campiona subito il comando; le successive si impostano nel loop appena prima di ogni resample.
    heights = None
    if args_cli.preset == "V3":
        spec = args_cli.heights or preset["heights"]
        if spec.strip().lower() != "random":
            heights = [float(h) for h in spec.split(",")]
            env_cfg.commands.base_height.ranges.height = (heights[0], heights[0])
            log.append(f"sequenza di quote ogni {preset['segment']:g} s: {heights}")

    # policy: di default il run finale, ancorato ($) perche' --load_run e' una regex e prenderebbe
    # anche eventuali copie con un suffisso
    if args_cli.load_run is None and args_cli.checkpoint is None:
        args_cli.load_run = f"^{FINAL_RUN}$"
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = REPO / "logs" / "rsl_rl" / agent_cfg.experiment_name
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(str(log_root), agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[media] policy: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    dt = base.step_dt
    fps = 1.0 / dt

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=base.device)

    if args_cli.focal_length:
        set_focal_length(env_cfg.viewer.cam_prim_path, args_cli.focal_length)
    if args_cli.ground is None:
        args_cli.ground = "lightgrid" if args_cli.look == "tesi" else "grid"
    ground, grid_shader = apply_ground(args_cli.ground, env_cfg, args_cli.ground_shade)
    sun, recolored = None, None
    if args_cli.look == "tesi":
        cam_azim = args_cli.cam_azim if args_cli.cam_azim is not None else preset["cam"]["azim"]
        sun = add_sun(cam_azim, args_cli)
        robot_root = env_cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        recolored = recolor_robot(robot_root.replace("env_.*", "env_0"), args_cli.robot_color,
                                  args_cli.robot_roughness, args_cli.robot_metallic)
    cam = Camera(base, preset["cam"], args_cli)
    vel_term = base.command_manager.get_term("base_velocity")
    robot = base.scene["robot"]

    def settle(n: int):
        """n render sullo stesso stato fisico (render NON avanza la fisica). Servono perche' il renderer
        media ogni fotogramma con i precedenti e l'annotatore restituisce l'immagine con un fotogramma di
        ritardo: subito dopo un cambiamento (frecce spente, salto di 1 s) l'immagine e' un misto."""
        for _ in range(max(n - 1, 0)):
            base.sim.render()
        return base.render()

    def warmup(max_renders: int) -> int:
        """Render finche' l'immagine smette di cambiare: materiali, cielo e texture si caricano in
        background. Nel primo test il fotogramma 0 dei video era la scena NON caricata (fondo grigio
        chiaro, niente pavimento ne' ombre). Restituisce quanti render sono serviti."""
        means = []
        for k in range(max_renders):
            means.append(float(base.render().mean()))
            if k >= 30 and max(means[-10:]) - min(means[-10:]) < 0.2:
                return k + 1
        print(f"[media] ATTENZIONE: dopo {max_renders} render l'immagine cambia ancora (scena non caricata?)")
        return max_renders

    out_dir = Path(args_cli.out) if args_cli.out else REPO / "media_tesi"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    n_warm = warmup(args_cli.warmup)
    print(f"[media] scena pronta dopo {n_warm} render")
    floor_cal = None
    if grid_shader is not None:  # a scena caricata, prima del primo fotogramma
        floor_cal = calibrate_floor(base, cam, grid_shader, args_cli.ground_rgb, settle)
        settle(args_cli.settle)  # il renderer riparte dall'inquadratura vera

    preroll = int(round(preset["preroll"] / dt))
    if args_cli.preview:
        total, kind = int(round(args_cli.preview_at / dt)), "preview"
    elif preset["kind"] == "stills":
        times = args_cli.times or list(preset["times"])
        shots = {int(round(t / dt)): k for k, t in enumerate(times, start=1)}
        total, kind = max(shots) + 1, "stills"
    else:
        duration = args_cli.duration or preset["duration"]
        total, kind = preroll + int(round(duration / dt)), "video"

    seg = int(round(preset.get("segment", 4.0) / dt))
    push_every = int(round(preset["push_interval"] / dt)) if "push_interval" in preset else None
    pushes, files, residual = [], [], []
    tele = []  # V3-V5: comando e stato misurato del robot a ogni fotogramma (-> <video>_telemetria.csv)
    h_term = base.command_manager.get_term("base_height")
    rng = np.random.default_rng(args_cli.seed)
    writer = None
    if kind == "video":
        name = preset["name"]
        mp4 = out_path(out_dir, name, ".mp4", args_cli.overwrite)
        w, h = env_cfg.viewer.resolution
        writer = Mp4Writer(mp4, w, h, fps)
        files.append(mp4.name)

    print(f"[media] {args_cli.preset}: {kind}, {total} passi ({total * dt:.1f} s), dt {dt:g} s")
    for i in range(total):
        with torch.inference_mode():
            if heights is not None:
                k = min((i + 1) // seg, len(heights) - 1)
                h_k = heights[k]
                base.command_manager.get_term("base_height").cfg.ranges.height = (h_k, h_k)
            if push_every and i >= preroll and (i - preroll) % push_every == push_every - 1:
                if args_cli.push == "alternate":
                    dv = np.array([args_cli.push_speed * (1 if len(pushes) % 2 == 0 else -1), 0.0])
                else:
                    dv = rng.uniform(-args_cli.push_speed, args_cli.push_speed, size=2)
                vel = robot.data.root_vel_w.clone()
                vel[:, 0] += float(dv[0])
                vel[:, 1] += float(dv[1])
                robot.write_root_velocity_to_sim(vel)
                pushes.append({"t": round((i - preroll + 1) * dt, 3), "dvx": round(float(dv[0]), 3),
                               "dvy": round(float(dv[1]), 3), "|dv|": round(float(np.hypot(*dv)), 3)})
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        cam.update()

        if kind == "video":
            if i < preroll:
                base.sim.render()  # renderer al passo con la scena anche prima di registrare
            else:
                if i == preroll:
                    settle(8)  # scartati: niente fotogramma vecchio o mezzo mediato in apertura
                writer.write(base.render())
                if base.num_envs == 1:
                    c = vel_term.command[0].tolist()
                    v = robot.data.root_lin_vel_b[0].tolist()
                    w = robot.data.root_ang_vel_b[0].tolist()
                    pos = robot.data.root_pos_w[0].tolist()
                    tele.append([round((i - preroll + 1) * dt, 3), *c, float(h_term.command[0, 0]), v[0], v[1], w[2],
                                 math.degrees(float(robot.data.heading_w[0])), *pos])
        elif kind == "stills" and i in shots:
            k = shots[i]
            on = out_path(out_dir, f"{preset['name']}_frecce_{k}", ".png", args_cli.overwrite)
            img_on = settle(args_cli.settle)
            save_png(img_on, on)
            vel_term.set_debug_vis(False)
            off = out_path(out_dir, f"{preset['name']}_pulito_{k}", ".png", args_cli.overwrite)
            img_off = settle(args_cli.settle_toggle)
            save_png(img_off, off)
            vel_term.set_debug_vis(True)
            res_on, res_off = arrow_pixels(img_on), arrow_pixels(img_off)
            residual.append({"scatto": k, "t": round((i + 1) * dt, 2), "pixel_frecce": res_on, "pixel_residui_pulito": res_off})
            flag = "  <-- frecce ancora visibili: aumenta --settle_toggle" if res_off > 0.02 * max(res_on, 1) else ""
            print(f"[media] scatto {k}: pixel delle frecce {res_on} -> nello scatto pulito {res_off}{flag}")
            files += [on.name, off.name]

    if kind == "preview":
        suffix = "" if args_cli.ground == "grid" else f"_{args_cli.ground}"  # anteprime confrontabili
        p = out_dir / f"_preview_{args_cli.preset}{suffix}.png"
        save_png(settle(args_cli.settle), p)
        files.append(p.name)
    tele_summary = None
    if tele:
        cols = ["t", "cmd_vx", "cmd_vy", "cmd_wz", "cmd_h", "vx_b", "vy_b", "wz_b", "yaw_deg", "x", "y", "z"]
        T = np.array(tele)
        csv_path = out_dir / f"{files[0].rsplit('.', 1)[0]}_telemetria.csv"
        np.savetxt(csv_path, T, delimiter=",", header=",".join(cols), comments="", fmt="%.5f")
        k = {c: j for j, c in enumerate(cols)}
        yaw = np.unwrap(np.radians(T[:, k["yaw_deg"]]))
        mean = lambda c: round(float(T[:, k[c]].mean()), 3)
        tele_summary = {
            "csv": csv_path.name,
            "comando_medio": {"vx": mean("cmd_vx"), "vy": mean("cmd_vy"), "wz": mean("cmd_wz"), "h": mean("cmd_h")},
            "misurato_medio": {"vx": mean("vx_b"), "vy": mean("vy_b"), "wz": mean("wz_b"), "z_base": mean("z")},
            "rotazione_totale_deg": round(float(np.degrees(yaw[-1] - yaw[0])), 1),
            "spostamento_netto_m": round(float(np.hypot(T[-1, k["x"]] - T[0, k["x"]], T[-1, k["y"]] - T[0, k["y"]])), 2),
            "percorso_m": round(float(np.hypot(np.diff(T[:, k["x"]]), np.diff(T[:, k["y"]])).sum()), 2),
        }
        print(f"[media] telemetria: comando {tele_summary['comando_medio']}, misurato {tele_summary['misurato_medio']}, "
              f"rotazione {tele_summary['rotazione_totale_deg']} deg, spostamento netto {tele_summary['spostamento_netto_m']} m")
        files.append(csv_path.name)
    if writer:
        writer.close()

    meta = {
        "preset": args_cli.preset, "file": files, "creato": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task, "checkpoint": str(resume_path), "commit": git_commit(), "seed": args_cli.seed,
        "num_envs": base.num_envs, "risoluzione": list(env_cfg.viewer.resolution), "fps": fps,
        "preroll_s": preset["preroll"] if kind == "video" else 0.0, "durata_s": (total - preroll) * dt if kind == "video" else None,
        "randomizzazioni": "training" if dr else "spente (robot nominale)", "frecce": arrows,
        "aspetto": args_cli.look, "sole": sun, "shader_robot_modificati": recolored,
        "robot": None if args_cli.look != "tesi" else {"albedo": args_cli.robot_color, "roughness": args_cli.robot_roughness,
                                                      "metallic": args_cli.robot_metallic},
        "fstop": args_cli.fstop or None,
        "camera": cam.describe(), "focale_mm": args_cli.focal_length, "pavimento": ground,
        "calibrazione_pavimento": floor_cal,
        "scala_frecce": args_cli.arrow_scale, "rendering_mode": getattr(args_cli, "rendering_mode", None),
        "render_di_riscaldamento": n_warm, "modifiche_config": log,
        "spinte": {"modo": args_cli.push, "ogni_s": preset["push_interval"], "eventi": pushes} if push_every else None,
        "controllo_frecce": residual or None,
        "controllo_di_rotta": {"k": args_cli.heading_stiffness}
        if (args_cli.heading_hold or ("on" if preset.get("hold") else "off")) == "on" else None,
        "telemetria": tele_summary,
    }
    if kind == "stills":
        meta_path = out_path(out_dir, preset["name"], ".json", args_cli.overwrite)
    else:  # stesso nome del video (gia' numerato se esisteva) o dell'anteprima
        meta_path = out_dir / f"{files[0].rsplit('.', 1)[0]}.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[media] dettagli in {meta_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
