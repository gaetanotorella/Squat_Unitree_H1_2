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

Le opzioni --cam_* e --focal_length ritoccano l'inquadratura senza toccare il codice.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

FINAL_RUN = "2026-09-13_21-27-39"

# camere: "grid" = fissa, obliqua sul centro della griglia; "follow" = segue x/y del robot 0.
# azim: direzione IN CUI sta la camera rispetto al bersaglio (0 = +x davanti al robot, -90 = -y, cioe'
# alla destra del robot: si vede il fianco destro e il robot avanza verso destra nell'immagine).
CAM_GRID_32 = dict(mode="grid", dist=16.0, elev=35.0, azim=225.0, look_z=0.5)
CAM_GRID_4096 = dict(mode="grid", dist=130.0, elev=30.0, azim=225.0, look_z=0.0)
CAM_SIDE = dict(mode="follow", dist=3.5, azim=-90.0, eye_z=1.0, look_z=0.85)
CAM_FRONT = dict(mode="follow", dist=3.5, azim=0.0, eye_z=1.0, look_z=0.85)

# Gruppi di modifiche alla configurazione: (percorso puntato, valore). Un percorso che non esiste
# ferma lo script con un errore: un nome sbagliato non deve passare in silenzio.
STANDING = [
    ("commands.base_velocity.rel_standing_envs", 1.0),
    ("commands.base_velocity.debug_vis", False),
]
NO_CURRICULUM = [("curriculum.lin_vel_cmd_levels", None), ("curriculum.height_cmd_levels", None)]
FIXED_RESET = [("events.reset_base.params.pose_range", {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)})]
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


SINGLE = dict(num_envs=1, preroll=1.0, dr=False)
PRESETS = {
    "S1": dict(kind="stills", name="S1_envs", num_envs=32, spacing=4.0, res=(3840, 2160), preroll=0.0,
               dr=True, cam=CAM_GRID_32, arrows="both", times=(3.0, 4.0, 5.0, 6.5, 8.0), mods=[]),
    "V1": dict(kind="video", name="V1_scala", num_envs=4096, spacing=2.5, duration=12.0, preroll=1.0,
               dr=True, cam=CAM_GRID_4096, arrows="off", mods=[]),
    "V2": dict(kind="video", name="V2_misti", num_envs=32, spacing=4.0, duration=18.0, preroll=1.0,
               dr=True, cam=CAM_GRID_32, arrows="on", mods=[]),
    "V3": dict(kind="video", name="V3_squat", duration=30.0, cam=CAM_SIDE, arrows="off", **SINGLE,
               heights="0.98,0.60,0.98,0.80,0.65,0.90,0.70,0.98", segment=4.0,
               mods=STANDING + [("commands.base_height.ranges.height", (0.60, 0.98)),
                                ("commands.base_height.resampling_time_range", (4.0, 4.0)),
                                ("events.push_robot", None)]),
    "V4a": dict(kind="video", name="V4a_crouch_avanti", duration=30.0, cam=CAM_SIDE, arrows="on", **SINGLE,
                mods=crouch_walk(0.5, 0.0, 0.0)),
    "V4b": dict(kind="video", name="V4b_crouch_laterale", duration=30.0, cam=CAM_FRONT, arrows="on", **SINGLE,
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
parser.add_argument("--cam_dist", type=float, default=None)
parser.add_argument("--cam_elev", type=float, default=None, help="Camere 'grid': elevazione [deg].")
parser.add_argument("--cam_azim", type=float, default=None, help="Direzione della camera [deg], 0 = +x.")
parser.add_argument("--cam_eye_z", type=float, default=None, help="Camere 'follow': altezza della camera [m].")
parser.add_argument("--cam_look_z", type=float, default=None, help="Altezza del punto inquadrato [m].")
parser.add_argument("--cam_smooth", type=float, default=0.4, help="Camere 'follow': costante di tempo [s].")
parser.add_argument("--focal_length", type=float, default=None, help="Focale della camera [mm] (default Kit 18.1).")
parser.add_argument("--settle", type=int, default=16, help="Render ripetuti sullo stesso stato per un'immagine ferma.")
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

    dr = (args.dr or ("on" if preset["dr"] else "off")) == "on"
    if not dr:
        for path, value in NO_DR:
            set_path(env_cfg, path, value, log)

    arrows = args.arrows or preset["arrows"]
    if arrows in ("on", "off"):
        set_path(env_cfg, "commands.base_velocity.debug_vis", arrows == "on", log)
    else:  # "both": accese, si spengono solo per lo scatto pulito
        set_path(env_cfg, "commands.base_velocity.debug_vis", True, log)

    # episodio piu' lungo del video: niente reset (teletrasporto) a meta' ripresa
    if preset["kind"] == "video":
        need = preset["preroll"] + (args.duration or preset["duration"]) + 2.0
        if env_cfg.episode_length_s < need:
            set_path(env_cfg, "episode_length_s", need, log)

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
        if self.p["mode"] == "grid":
            c = base_env.scene.env_origins.mean(dim=0).cpu().numpy()
            e, a = math.radians(self.p["elev"]), math.radians(self.p["azim"])
            self.look = np.array([c[0], c[1], self.p["look_z"]])
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

    def describe(self):
        return {**self.p, "eye": [round(float(v), 3) for v in self.eye], "lookat": [round(float(v), 3) for v in self.look]}


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
    cam = Camera(base, preset["cam"], args_cli)
    vel_term = base.command_manager.get_term("base_velocity")
    robot = base.scene["robot"]

    def settle():
        """Piu' render sullo stesso stato fisico: l'annotatore restituisce l'immagine con un ritardo
        e l'accumulo temporale del renderer converge. Serve per gli scatti fermi e dopo le frecce."""
        for _ in range(max(args_cli.settle - 1, 0)):
            base.sim.render()
        return base.render()

    out_dir = Path(args_cli.out) if args_cli.out else REPO / "media_tesi"
    out_dir.mkdir(parents=True, exist_ok=True)

    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()

    settle()  # riscaldamento del renderer prima del primo passo

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
    pushes, files = [], []
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

        if kind == "video" and i >= preroll:
            writer.write(base.render())
        elif kind == "stills" and i in shots:
            k = shots[i]
            on = out_path(out_dir, f"{preset['name']}_frecce_{k}", ".png", args_cli.overwrite)
            save_png(settle(), on)
            vel_term.set_debug_vis(False)
            off = out_path(out_dir, f"{preset['name']}_pulito_{k}", ".png", args_cli.overwrite)
            save_png(settle(), off)
            vel_term.set_debug_vis(True)
            files += [on.name, off.name]

    if kind == "preview":
        p = out_dir / f"_preview_{args_cli.preset}.png"
        save_png(settle(), p)
        files.append(p.name)
    if writer:
        writer.close()

    meta = {
        "preset": args_cli.preset, "file": files, "creato": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": args_cli.task, "checkpoint": str(resume_path), "commit": git_commit(), "seed": args_cli.seed,
        "num_envs": base.num_envs, "risoluzione": list(env_cfg.viewer.resolution), "fps": fps,
        "preroll_s": preset["preroll"] if kind == "video" else 0.0, "durata_s": (total - preroll) * dt if kind == "video" else None,
        "randomizzazioni": "training" if dr else "spente (robot nominale)", "frecce": arrows,
        "camera": cam.describe(), "focale_mm": args_cli.focal_length, "modifiche_config": log,
        "spinte": {"modo": args_cli.push, "ogni_s": preset["push_interval"], "eventi": pushes} if push_every else None,
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
