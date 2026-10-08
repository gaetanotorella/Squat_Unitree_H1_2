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
    S4   un robot fermo in piedi a 0.98 m, senza frecce, con l'inquadratura della foto del robot reale
         (misurata in MuJoCo), 4:3
    V1   4096 robot, camera alta e lontana, 12 s
    V2   32 robot, stessa camera di S1, 18 s (un solo episodio: niente reset simultaneo a 20 s)
    V3   squat da fermo, sequenza di quote fissa ogni 4 s, camera laterale, 30 s
    V4a  camminata accovacciata a 0.75 m, vx 0.5, camera laterale
    V4b  come V4a ma vy 0.3, camera frontale
    V4c  come V4a ma rotazione sul posto wz 0.5, camera laterale
    V5a  spinte da fermo a 0.98 m ogni 3 s, camera laterale
    V5b  spinte da fermo a 0.70 m ogni 3 s, camera laterale
    V6a  4096 robot, la camera gira attorno al centro (distanza 16 m, elevazione 35 deg), 24 s
    V6b  come V6a con distanza 20 m ed elevazione 28 deg
    V7a  4096 robot fermi a quote casuali, vertical reveal: indietro di 35 m a livello dei robot, poi su fino
         alla vista zenitale da 150 m, 30 s
    V7b  4096 robot fermi a quote casuali, trench run: in volo lungo il corridoio del bordo, poi su verso
         l'angolo opposto, 20 s
    V7c  4096 robot, spirale: raggio da 5 a 180 m ed elevazione da 10 a 60 deg in un giro e un quarto, 30 s
    (V7: vedi TRAIETTORIE; --traj chiave=valore cambia un parametro del percorso, --crowd il comportamento
    dei robot, --fstop accende la profondita' di campo, spenta di default)

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
CAM_GRID_32_2 = dict(mode="grid", dist=10.0, elev=30.0, azim=135.0, look_z=0.5, frac=0.0)
CAM_GRID_32_3 = dict(mode="grid", dist=30.0, elev=22.0, azim=135.0, look_z=0.8, frac=0.0)

CAM_GRID_4096 = dict(mode="grid", dist=30.0, elev=22.0, azim=135.0, look_z=0.8, frac=0.8)
# ORBITA: la camera gira su una circonferenza attorno al centro della griglia e punta sempre al centro
# (dist = distanza camera-centro, elev = elevazione, azim = punto di partenza). period = secondi per un
# giro completo, positivo = antiorario visto dall'alto. Il sole resta fermo nel mondo (orientato rispetto
# all'azimut di PARTENZA): durante il giro la luce passa da tre quarti a controluce e ritorno.
CAM_ORBIT_16 = dict(mode="orbit", dist=16.0, elev=35.0, azim=135.0, look_z=0.5, period=24.0)
CAM_ORBIT_20 = dict(mode="orbit", dist=20.0, elev=28.0, azim=135.0, look_z=0.5, period=24.0)
CAM_SIDE = dict(mode="follow", dist=3.8, azim=90.0, eye_z=1.4, look_z=0.8)
CAM_FRONT = dict(mode="follow", dist=3.8, azim=180.0, eye_z=1.4, look_z=0.8)
# SUL ROBOT ("robot"): camera agganciata alla posa del robot 0, con le convenzioni del visualizzatore di MuJoCo,
# per rifare in Isaac Lab un'inquadratura presa li' (ATTENZIONE: azim ed elev qui NON sono quelli delle altre
# camere):
#   azim = direzione IN CUI GUARDA la camera, rispetto alla direzione in cui guarda il robot [deg];
#   elev = beccheggio della vista, negativo = verso il basso [deg]; dist = distanza camera-bersaglio [m];
#   bersaglio = base del robot + look_fwd in avanti + look_left a sinistra (riferimento del robot), alla quota
#   assoluta look_z; vfov = campo visivo verticale [deg] (MuJoCo: vis.global.fovy).
# Occhio = bersaglio - dist * (cos elev cos az, cos elev sin az, sin elev), come mjv_updateCamera.
# S4 (2026-10-08): l'inquadratura della foto del robot reale, ricostruita in MuJoCo. La camera sta davanti al
# robot, ruotata di ~29 deg verso la sua sinistra, a 1.05 m di quota.
CAM_S4 = dict(mode="robot", dist=3.047, elev=-2.46, azim=209.16, look_fwd=0.389, look_left=0.371, look_z=0.919,
              vfov=37.28)

# TRAIETTORIE ("traj"): la camera segue un percorso nel tempo. Coordinate in metri RISPETTO AL CENTRO della
# griglia, scritte per 4096 robot a 2.5 m (64 x 64, lato 160 m: bordo a +-80 m; le file di robot stanno a
# +-1.25, +-3.75, ...). Il percorso dipende solo da u = t / durata: --duration lo accelera o lo rallenta
# senza cambiarne la forma. Ogni parametro si cambia con --traj chiave=valore (es. --traj z_top=250).
# azim serve solo a orientare il sole (vedi LUCE): la direzione da cui la camera guarda la scena per la
# maggior parte del percorso.
# FUOCO: profondita' di campo spenta di default in tutti i V7 (decisione del 2026-10-08, dopo le anteprime);
# con --fstop > 0 il fuoco parte da focus_near [m] e arriva a focus_far (1e4 = infinito) tra le frazioni
# focus_from e focus_to della durata, interpolato in diottrie (1/d): lo sfocato di un oggetto a distanza d
# e' proporzionale a |1/d - 1/d_fuoco|. Misurato sulle anteprime del V7b: ~130 px per diottria a f/11, ~65 a
# f/22, ~45 a f/32 (1920 px di larghezza). La finestra va chiusa PRIMA che i robot in vista siano tutti
# lontani, altrimenti l'immagine e' tutta sfocata.
TRAJ_DEFAULTS = {
    # VERTICAL REVEAL: parte da (x0, y0, z0) puntata al centro, carrellata all'indietro di `back` m nella
    # prima frazione `dolly` (lungo il corridoio x = 0, fra due file); poi, nella frazione `rise`, sale fino
    # a z_top tornando sopra il centro, col beccheggio che arriva alla vista zenitale. La salita parte e
    # arriva dolce; il beccheggio segue la quota in scala logaritmica, cioe' quanto si rimpicciolisce la
    # scena: ~50 deg a 17 m, quasi zenitale gia' a 75 m, poi la camera sale ancora guardando giu'.
    # back = 35: 5 m della prima versione + 30 m chiesti dopo le anteprime.
    "reveal": dict(x0=0.0, y0=-5.0, z0=1.5, look_z=0.9, back=35.0, z_top=150.0, dolly=0.4, rise=0.5,
                   focus_near=5.0, focus_far=1.0e4, focus_from=0.0, focus_to=0.4),
    # TRENCH RUN: nel corridoio fra due file di robot a quota z0, da y0 a y_lift a velocita' costante nella
    # frazione `run`, guardando lungo il corridoio (ahead m avanti, drop m piu' in basso: con 2.7 la vista
    # scende di 6 deg e l'orizzonte sta nel terzo superiore). Da y_lift, nella frazione `lift`: frenata dolce
    # in avanti (velocita' continua), salita a z_top e pan verso l'angolo (corner_x, corner_y, corner_z);
    # aim_down [deg] abbassa la vista finale sotto l'angolo (6.2 = angolo nel terzo superiore).
    # Seconda versione (2026-10-08): corridoio lungo il bordo sinistro (x = -77.5, fra le due file esterne)
    # e salita a y = -40 invece che al centro: in volo la camera guarda in diagonale attraverso tutta la
    # griglia verso l'angolo opposto. Robot inquadrati a fine salita (modello della camera): ~2000 contro i
    # ~470 della prima versione (corridoio centrale, salita a y = 0). x = il corridoio voluto: si usa quello
    # fra due file piu' vicino.
    "trench": dict(x=0.0, y0=-80.0, y_lift=-40.0, z0=1.5, ahead=25.0, drop=2.7, run=0.3, lift=0.55, z_top=30.0,
                   corner_x=80.0, corner_y=80.0, corner_z=0.5, aim_down=6.2, focus_near=5.0, focus_far=1.0e4,
                   focus_from=0.3, focus_to=0.45),
    # EXPANSIVE SPIRAL: orbita attorno a (cx, cy, look_z); raggio da r0 a r1 ed elevazione da elev0 a elev1
    # crescono linearmente con l'azimut, che va da azim0 ad azim0 + turn [deg]. L'azimut parte e si ferma
    # con dolcezza. Al centro di una griglia pari non c'e' un robot: --traj cx=1.25 cy=1.25 ne mette uno
    # al centro dell'inquadratura iniziale.
    "spiral": dict(cx=0.0, cy=0.0, look_z=0.8, r0=5.0, r1=180.0, elev0=10.0, elev1=60.0, azim0=0.0, turn=450.0,
                   focus_near=5.0, focus_far=1.0e4, focus_from=0.0, focus_to=0.35),
}
CAM_REVEAL = dict(mode="traj", traj="reveal", azim=270.0)
CAM_TRENCH = dict(mode="traj", traj="trench", azim=270.0)
CAM_SPIRAL = dict(mode="traj", traj="spiral", azim=0.0)

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
# FOLLA (--crowd, preset con piu' robot): al reset sempre sul punto esatto della griglia (in training
# +-0.5 m: robot vicini nascono compenetrati e le file non sono allineate), orientamento casuale.
# 'misti' = comandi del training: camminano, girano, si accovacciano (in 20 s si spostano di metri);
# 'squat' = fermi sul posto a quote casuali e senza spinte: file dritte e corridoi liberi.
LATTICE_RESET = [("events.reset_base.params.pose_range", {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (-3.14, 3.14)})]
CROWD = {"misti": LATTICE_RESET, "squat": LATTICE_RESET + STANDING + [("events.push_robot", None)]}


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
               dr=True, cam=CAM_GRID_32, arrows="both", times=(2.5, 3.0, 3.5, 4.0, 5.0), mods=[]),
    "S2": dict(kind="stills", name="S2_envs", num_envs=128, spacing=2.5, res=(3840, 2160), preroll=0.0,
               dr=True, cam=CAM_GRID_32_2, arrows="both", times=(2.5, 3.0, 3.5, 4.0, 5.0), mods=[]),
    # S4: robot nominale fermo in piedi (quota comandata 0.98 m, niente spinte ne' frecce) con l'inquadratura
    # della foto reale (CAM_S4). 4:3 come la foto di riferimento (1300 x 975), al doppio. Sole alle spalle
    # della camera a destra, come la luce del render MuJoCo: robot illuminato davanti, ombre dietro a sinistra.
    "S4": dict(kind="stills", name="S4_in_piedi", num_envs=1, res=(2600, 1950), preroll=0.0, dr=False,
               cam=CAM_S4, arrows="off", times=(5.0,), sun_rel_azim=45.0,
               mods=STANDING + [("commands.base_height.ranges.height", (0.98, 0.98)), ("events.push_robot", None)]),
    "S3": dict(kind="stills", name="S3_envs", num_envs=4096, spacing=2.5, res=(3840, 2160), preroll=0.0,
               dr=True, cam=CAM_GRID_32_3, arrows="both", times=(2.5, 3.0, 3.5, 4.0, 5.0), mods=[]),

    "V1": dict(kind="video", name="V1_scala", num_envs=4096, spacing=2.5, duration=12.0, preroll=1.0,
               dr=True, cam=CAM_GRID_4096, arrows="off", mods=[]),
    # V6: un giro completo di camera attorno ai 4096 env, sempre puntata verso il centro
    "V6a": dict(kind="video", name="V6a_orbita_d16_e35", num_envs=4096, spacing=2.5, duration=24.0, preroll=1.0,
                dr=True, cam=CAM_ORBIT_16, arrows="off", mods=[]),
    "V6b": dict(kind="video", name="V6b_orbita_d20_e28", num_envs=4096, spacing=2.5, duration=24.0, preroll=1.0,
                dr=True, cam=CAM_ORBIT_20, arrows="off", mods=[]),
    # V7: traiettorie di camera temporizzate (vedi TRAIETTORIE). Robot fermi dove la camera passa a 1.5 m fra
    # le file: V7a (35 m di carrellata) e V7b. Con i comandi del training i robot camminano per metri e la
    # camera finirebbe dentro qualcuno: a posizioni ormai mescolate, ~3 robot in media nella striscia di
    # 0.6 x 35 m che la camera del V7a attraversa.
    "V7a": dict(kind="video", name="V7a_vertical_reveal", num_envs=4096, spacing=2.5, duration=30.0, preroll=1.0,
                dr=True, cam=CAM_REVEAL, arrows="off", crowd="misti", mods=[]),
    "V7b": dict(kind="video", name="V7b_trench_run", num_envs=4096, spacing=2.5, duration=20.0, preroll=1.0,
                dr=True, cam=CAM_TRENCH, arrows="off", crowd="misti", mods=[]),
    "V7c": dict(kind="video", name="V7c_spirale", num_envs=4096, spacing=2.5, duration=30.0, preroll=1.0,
                dr=True, cam=CAM_SPIRAL, arrows="off", crowd="misti", mods=[]),
    "V2":dict(kind="video", name="V2_misti", num_envs=32, spacing=4.0, duration=18.0, preroll=1.0,
               dr=True, cam=CAM_GRID_32, arrows="on", mods=[]),
    "V2a": dict(kind="video", name="V2a_misti", num_envs=512, spacing=2.0, duration=18.0, preroll=1.0,
                dr=True, cam=CAM_GRID_32, arrows="off", mods=[]),

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
parser.add_argument("--cam_elev", type=float, default=None,
                    help="Camere 'grid'/'orbit': elevazione [deg]; 'robot' (S4): beccheggio MuJoCo, negativo = verso il basso.")
parser.add_argument("--cam_azim", type=float, default=None,
                    help="Direzione della camera [deg], 0 = +x; 'robot' (S4): direzione di vista MuJoCo rispetto al robot.")
parser.add_argument("--cam_eye_z", type=float, default=None, help="Camere 'follow': altezza della camera [m].")
parser.add_argument("--cam_look_z", type=float, default=None, help="Altezza del punto inquadrato [m].")
parser.add_argument("--cam_smooth", type=float, default=0.4, help="Camere 'follow': costante di tempo [s].")
parser.add_argument("--orbit_period", type=float, default=None,
                    help="Camere 'orbit': secondi per un giro completo (negativo = orario). Default del preset: 24 s, "
                         "un giro esatto nella durata del video.")
parser.add_argument("--traj", type=str, nargs="+", default=None, metavar="CHIAVE=VALORE",
                    help="Camere 'traj' (V7): cambia parametri del percorso, es. --traj z_top=250 back=8. Chiavi e "
                         "default in TRAJ_DEFAULTS; una chiave sbagliata ferma lo script con l'elenco di quelle valide.")
parser.add_argument("--crowd", choices=("misti", "squat"), default=None,
                    help="Preset con piu' robot: 'misti' = comandi del training, 'squat' = fermi a quote casuali; in "
                         "entrambi i casi i robot partono sui punti esatti della griglia. Default: quello del preset "
                         "(V7), altrimenti come in training.")
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
parser.add_argument("--sun_rel_azim", type=float, default=None,
                    help="--look tesi: direzione del sole rispetto alla camera [deg]; 135 = dietro il soggetto, "
                         "a destra nell'immagine (ombre verso la camera, come nella foto di riferimento); "
                         "45 = alle spalle della camera, a destra. Default: 135, per S4 45.")
parser.add_argument("--sun_intensity", type=float, default=3000.0)
parser.add_argument("--sun_angle", type=float, default=0.53, help="Diametro apparente del sole [deg]: ombre nette.")
parser.add_argument("--sky_intensity", type=float, default=800.0, help="--look tesi: luce diffusa del cielo uniforme.")
parser.add_argument("--fstop", type=float, default=None,
                    help="Profondita' di campo: f-stop della camera (0 = spenta; piu' basso = piu' sfocato). Fuoco "
                         "sul bersaglio, o animato come in TRAIETTORIE per i V7. Default: 0 (spenta) per tutti i preset.")
parser.add_argument("--ground_shade", type=float, default=0.35, help="--ground grey: grigio del pavimento (0-1).")
parser.add_argument("--arrow_scale", type=float, default=None, help="Spessore/lunghezza delle frecce (Isaac Lab: 0.5).")
parser.add_argument("--settle", type=int, default=32, help="Render ripetuti sullo stesso stato per uno scatto fermo.")
parser.add_argument("--settle_toggle", type=int, default=128,
                    help="Render dopo aver spento le frecce: il renderer media i fotogrammi e con 16 restava "
                         "un fantasma delle frecce al ~10%%.")
parser.add_argument("--warmup", type=int, default=300, help="Massimo di render iniziali per caricare materiali e cielo.")
parser.add_argument("--preview", action="store_true", help="Salva solo i fotogrammi di --preview_at ed esce.")
parser.add_argument("--preview_at", type=float, nargs="+", default=[2.0],
                    help="Istanti dei fotogrammi di --preview, nel tempo del video (dopo il preroll) [s]: lo stesso "
                         "fotogramma che il video avrebbe a quell'istante. Con piu' istanti i file finiscono in _t<s>.")
parser.add_argument("--preview_fstop", type=float, nargs="+", default=None,
                    help="Con --preview: rende ogni fotogramma con questi f-stop (0 = senza sfocatura), file _f<N>. "
                         "Stessa scena e stesso fuoco: serve a scegliere --fstop senza rilanciare la simulazione.")
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
    crowd = None if single else (args.crowd or preset.get("crowd"))
    if crowd:
        for path, value in CROWD[crowd]:
            set_path(env_cfg, path, value, log)

    hold =(args.heading_hold or ("on" if preset.get("hold") else "off")) == "on"
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
    return env_cfg, dr, arrows, crowd


# --------------------------------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------------------------------
PITCH_MAX = math.radians(89.5)  # "zenitale": a 90 esatti la vista coincide con l'asse su e il look-at non e' definito
FOCUS_INF = 1.0e4  # [m] fuoco "all'infinito"


def _smooth(x):
    """smoothstep su [0, 1]: velocita' nulla agli estremi."""
    x = min(max(x, 0.0), 1.0)
    return x * x * (3.0 - 2.0 * x)


def _smoother(x):
    """smootherstep su [0, 1]: anche l'accelerazione e' nulla agli estremi."""
    x = min(max(x, 0.0), 1.0)
    return x * x * x * (x * (6.0 * x - 15.0) + 10.0)


def _view_dir(yaw, pitch):
    """Versore di vista da imbardata e beccheggio [rad]; beccheggio positivo = verso il basso."""
    return np.array([math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch)])


def _angles(v):
    """(imbardata, beccheggio verso il basso) di un vettore."""
    return math.atan2(v[1], v[0]), math.atan2(-v[2], math.hypot(v[0], v[1]))


def traj_pose(kind: str, P: dict, u: float):
    """(occhio, bersaglio) della traiettoria `kind` alla frazione u in [0, 1] della durata, rispetto al centro
    della griglia. Funzione pura (solo numpy): si prova senza Isaac Sim."""
    u = min(max(u, 0.0), 1.0)
    if kind == "reveal":
        center = np.array([0.0, 0.0, P["look_z"]])
        y_back = P["y0"] - P["back"]
        if u <= P["dolly"]:
            s = _smooth(u / P["dolly"])
            return np.array([P["x0"], P["y0"] + (y_back - P["y0"]) * s, P["z0"]]), center
        # salita: imbardata quella con cui la carrellata guardava il centro, beccheggio da li' allo zenit
        s = (u - P["dolly"]) / P["rise"]
        start = np.array([P["x0"], y_back, P["z0"]])
        yaw, pitch0 = _angles(center - start)
        k = _smooth(s)  # frazione della salita in metri
        z = P["z0"] + (P["z_top"] - P["z0"]) * k
        lz = math.log(z / P["z0"]) / math.log(P["z_top"] / P["z0"]) if P["z_top"] > P["z0"] else 1.0
        pitch = pitch0 + (PITCH_MAX - pitch0) * _smooth(lz)  # segue la quota in scala logaritmica
        eye = np.array([*(start[:2] * (1.0 - k)), z])  # torna sopra il centro mentre sale (40 m su 150)
        return eye, eye + _view_dir(yaw, pitch) * max(float(np.linalg.norm(center - eye)), 1.0)
    if kind == "trench":
        yaw0, pitch0 = math.pi / 2, math.atan2(P["drop"], P["ahead"])  # lungo +y, appena verso il basso
        if u <= P["run"]:
            eye = np.array([P["x"], P["y0"] + (P["y_lift"] - P["y0"]) * u / P["run"], P["z0"]])
            return eye, eye + _view_dir(yaw0, pitch0) * P["ahead"]
        # da y_lift: y = y_lift + dy (1 - (1 - s)^3), con dy scelto perche' la velocita' iniziale sia quella della corsa
        s = min((u - P["run"]) / P["lift"], 1.0)
        dy = (P["y_lift"] - P["y0"]) * P["lift"] / (3.0 * P["run"])
        eye = np.array([P["x"], P["y_lift"] + dy * (1.0 - (1.0 - s) ** 3),
                        P["z0"] + (P["z_top"] - P["z0"]) * _smooth(s)])
        to_c = np.array([P["corner_x"], P["corner_y"], P["corner_z"]]) - eye
        yaw_c, pitch_c = _angles(to_c)
        pitch_c += math.radians(P.get("aim_down", 0.0))
        w = _smooth(s)
        yaw = yaw0 + w * ((yaw_c - yaw0 + math.pi) % (2.0 * math.pi) - math.pi)  # pan dal corridoio all'angolo
        pitch = pitch0 + w * (pitch_c - pitch0)
        return eye, eye + _view_dir(yaw, pitch) * (P["ahead"] + w * (float(np.linalg.norm(to_c)) - P["ahead"]))
    if kind == "spiral":
        e = _smooth(u)  # raggio, elevazione e azimut avanzano insieme: lineari fra loro, dolci nel tempo
        r = P["r0"] + (P["r1"] - P["r0"]) * e
        el = math.radians(P["elev0"] + (P["elev1"] - P["elev0"]) * e)
        a = math.radians(P["azim0"] + P["turn"] * e)
        look = np.array([P["cx"], P["cy"], P["look_z"]])
        return look + r * np.array([math.cos(el) * math.cos(a), math.cos(el) * math.sin(a), math.sin(el)]), look
    raise ValueError(f"traiettoria sconosciuta: {kind}")


def traj_focus(P: dict, u: float) -> float:
    """Distanza di fuoco [m]: focus_near fino a focus_from, focus_far da focus_to, in mezzo lineare in 1/d."""
    w = _smooth((u - P["focus_from"]) / max(P["focus_to"] - P["focus_from"], 1e-6))
    inv = (1.0 - w) / P["focus_near"] + w / P.get("focus_far", FOCUS_INF)
    return FOCUS_INF if inv <= 1.0 / FOCUS_INF else 1.0 / inv


def corridor_x(xs, x_req: float) -> float:
    """Il corridoio (meta' fra due file adiacenti di robot) piu' vicino a x_req; xs = x delle file."""
    xs = np.unique(np.round(np.asarray(xs, dtype=float), 3))
    if len(xs) < 2:
        return float(x_req)
    mids = 0.5 * (xs[:-1] + xs[1:])
    return float(mids[np.argmin(np.abs(mids - x_req))])


def set_cam_attr(cam_prim_path: str, name: str, value: float, quiet: bool = False) -> dict:
    """Scrive un attributo della camera del viewport (fStop, focusDistance, focalLength) e lo rilegge.

    /OmniverseKit_Persp sta nel session layer di Kit, che vince sul layer radice, dove scrive un Set()
    normale: un attributo che Kit ha gia' scritto li' resta quello di Kit. Nel primo test (2026-10-08, V7b)
    l'f-stop funzionava ma il fuoco no: sfocati solo i robot vicini, orizzonte nitido, cioe' fuoco ancora
    lontano. Ipotesi: Kit crea la camera con un focusDistance gia' scritto (400) nel session layer; il valore
    'prima' stampato qui lo conferma. Qui si scrive nel session layer e si rilegge."""
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    attr = stage.GetPrimAtPath(cam_prim_path).GetAttribute(name)
    before = attr.Get()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        attr.Set(float(value))
    got = attr.Get()
    ok = got is not None and abs(float(got) - value) <= 1e-4 * max(1.0, abs(value))
    if not ok or not quiet:
        print(f"[media] camera: {name} {before} -> {value:g} (riletto {got})"
              + ("" if ok else "  <-- NON applicato: il renderer usa ancora il valore vecchio"))
    return {"prima": None if before is None else float(before), "scritto": float(value),
            "riletto": None if got is None else float(got), "ok": ok}


class Camera:
    def __init__(self, base_env, cam: dict, args, duration: float | None = None):
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
        self.focus = None
        self.cam_path = base_env.cfg.viewer.cam_prim_path
        self.fstop = args.fstop if args.fstop is not None else self.p.get("fstop", 0.0)
        # profondita' di campo: f-stop fisso, fuoco aggiornato a ogni fotogramma (anche solo per --preview_fstop)
        self.dof = {} if (self.fstop > 0 or args.preview_fstop) else None
        if self.fstop > 0:
            self.dof["fStop"] = set_cam_attr(self.cam_path, "fStop", self.fstop)
        if args.orbit_period is not None:
            self.p["period"] = args.orbit_period
        self.t = 0.0
        if self.p["mode"] == "traj":
            self.P = dict(TRAJ_DEFAULTS[self.p["traj"]])
            for kv in args.traj or []:
                k, _, v = kv.partition("=")
                if k not in self.P:
                    raise SystemExit(f"[media] --traj: '{k}' non e' un parametro di '{self.p['traj']}' "
                                     f"(validi: {', '.join(self.P)})")
                self.P[k] = float(v)
            self.T = float(duration or 1.0)
            o = base_env.scene.env_origins.cpu().numpy()
            self.c = np.array([*o[:, :2].mean(axis=0), 0.0])
            xs = np.unique(np.round(o[:, 0] - self.c[0], 3))
            side = float(xs[-1] - xs[0]) + (float(np.diff(xs).min()) if len(xs) > 1 else 0.0)
            if abs(side - 160.0) > 10.0:
                print(f"[media] ATTENZIONE: le traiettorie sono scritte per una griglia di 160 m (4096 robot a 2.5 m), "
                      f"questa e' di {side:.0f} m")
            if self.p["traj"] == "trench":
                x_req = self.P["x"]
                self.P["x"] = corridor_x(xs, x_req)
                print(f"[media] trench: corridoio a x = {self.P['x']:+.2f} m (chiesto {x_req:+.2f})")
        elif args.traj:
            print("[media] --traj ignorato: il preset non ha una camera 'traj'")
        self.fov = None
        if self.p.get("vfov") and not args.focal_length:
            self.fov = set_vfov(self.cam_path, self.p["vfov"], base_env.cfg.viewer.resolution)
        self.robot_pose = None
        if self.p["mode"] == "orbit":
            c = base_env.scene.env_origins.cpu().numpy().mean(axis=0)
            self.look = np.array([c[0], c[1], self.p["look_z"]])  # il centro della circonferenza
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

    def update(self, t: float | None = None):
        """t = tempo dall'inizio del video [s]; serve solo all'orbita (None = l'ultimo ricevuto)."""
        if t is not None:
            self.t = t
        if self.p["mode"] == "traj":
            u = self.t / self.T
            eye, look = traj_pose(self.p["traj"], self.P, u)
            self.eye, self.look = self.c + eye, self.c + look
            self.focus = traj_focus(self.P, u)
        if self.p["mode"] == "orbit":
            e = math.radians(self.p["elev"])
            a = math.radians(self.p["azim"] + 360.0 * self.t / self.p["period"])
            self.eye = self.look + self.p["dist"] * np.array(
                [math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        if self.p["mode"] == "follow":
            xy = self.robot.data.root_pos_w[0, :2].cpu().numpy()
            self.xy = xy if self.xy is None else self.xy + self.alpha * (xy - self.xy)
            a = math.radians(self.p["azim"])
            self.look = np.array([self.xy[0], self.xy[1], self.p["look_z"]])
            self.eye = np.array([self.xy[0] + self.p["dist"] * math.cos(a),
                                 self.xy[1] + self.p["dist"] * math.sin(a), self.p["eye_z"]])
        if self.p["mode"] == "robot":  # convenzioni MuJoCo, vedi CAM_S4
            pos = self.robot.data.root_pos_w[0].cpu().numpy()
            yaw = float(self.robot.data.heading_w[0])
            c, s = math.cos(yaw), math.sin(yaw)
            fw, lf = self.p["look_fwd"], self.p["look_left"]
            self.look = np.array([pos[0] + c * fw - s * lf, pos[1] + s * fw + c * lf, self.p["look_z"]])
            a, e = yaw + math.radians(self.p["azim"]), math.radians(self.p["elev"])
            self.eye = self.look - self.p["dist"] * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a),
                                                               math.sin(e)])
            self.robot_pose = {"base": [round(float(v), 4) for v in pos], "yaw_deg": round(math.degrees(yaw), 2)}
        if self.p["mode"] != "traj":
            self.focus = float(np.linalg.norm(self.eye - self.look))  # fuoco sul bersaglio
        self.sim.set_camera_view(tuple(float(v) for v in self.eye), tuple(float(v) for v in self.look))
        if self.dof is not None:
            r = set_cam_attr(self.cam_path, "focusDistance", self.focus, quiet="focusDistance" in self.dof)
            self.dof.setdefault("focusDistance", r)  # il primo: dice che valore c'era prima

    def calib_view(self):
        """Inquadratura su cui calibrare il colore del pavimento: quella della camera, tranne per le
        traiettorie, che partono anche radenti al suolo (dove il riflesso schiarisce il pavimento): li' una
        vista obliqua come V6 (16 m, 35 deg), cosi' il grigio e' lo stesso degli altri video."""
        if self.p["mode"] != "traj":
            return self.eye, self.look
        look = self.c + np.array([0.0, 0.0, 0.5])
        e, a = math.radians(35.0), math.radians(self.p["azim"])
        return look + 16.0 * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]), look

    def row(self):
        """t, occhio, bersaglio e fuoco: una riga del <video>_camera.csv."""
        return [round(self.t, 4), *map(float, self.eye), *map(float, self.look), float(self.focus)]

    def describe(self):
        d = {**self.p, "fstop": self.fstop or None, "attributi_camera": self.dof, "eye": [round(float(v), 3) for v in self.eye],
             "lookat": [round(float(v), 3) for v in self.look]}
        if self.p["mode"] == "traj":
            d.update(parametri=self.P, durata_s=self.T, centro=[round(float(v), 3) for v in self.c])
        if self.fov:
            d["campo_visivo"] = self.fov
        if self.robot_pose:
            d["robot_allo_scatto"] = self.robot_pose
        return d


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


def calibrate_floor(base, cam, shader, target_srgb, settle, iters: int = 8) -> dict:
    """Regola il colore del fondo finche' NELL'IMMAGINE esce target_srgb (entro 1 livello su 255).

    Il colore renderizzato dipende da luce, esposizione e riflesso del materiale, quindi non si puo'
    impostare a priori: si guarda un tratto di pavimento vuoto (camera spostata di 500 m, stesso
    orientamento dell'inquadratura vera), si misura la mediana di una fascia centrale -- le linee della
    griglia sono sottili e non la spostano -- e si corregge.

    Correzione: secante in scala log-log (misurato ~ k * albedo^g, g stimato dagli ultimi due passi). La
    correzione proporzionale (g = 1) convergeva troppo piano: nel V7b del 2026-10-08 240 -> 233 -> 225 ->
    219 -> 214 -> 212 contro 206, perche' sul pavimento chiaro l'immagine dipende poco dall'albedo (riflesso
    del cielo, tone mapping: g ~ 0.3).
    Il renderer ha memoria: dopo 48 render resta ~30% dell'immagine precedente. Con passi grandi la secante
    oscillava (secondo test: 240 233 179 203 212 213 148 199). Quindi: si misura finche' due misure di fila
    differiscono meno di 0.3 livelli, ogni passo cambia l'albedo al massimo di x0.7-x1.4 e g sta in
    [0.15, 1.5]. Simulato con la stessa memoria: entro 1 livello in 5-6 passi."""
    from pxr import Gf, Sdf

    target = np.asarray(target_srgb, dtype=float)
    target_lin = srgb_to_lin(target / 255.0)
    c = target_lin.copy()  # primo tentativo: albedo = colore voluto
    off = np.array([500.0, 500.0, 0.0])
    eye, look = cam.calib_view()
    base.sim.set_camera_view(tuple(float(v) for v in eye + off), tuple(float(v) for v in look + off))
    steps, meas, prev = [], target, None
    def measure(n):
        img = settle(n)
        h, w = img.shape[:2]
        return np.median(img[int(0.60 * h):int(0.95 * h), int(0.2 * w):int(0.8 * w)].reshape(-1, 3).astype(float), axis=0)

    for _ in range(iters):
        set_input(shader, "inputs:diffuse_color_constant", Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*map(float, c)))
        meas, n = measure(32), 32
        for _ in range(12):  # finche' l'immagine non smette di cambiare
            m2 = measure(16)
            n += 16
            stable = np.abs(m2 - meas).max() < 0.3
            meas = m2
            if stable:
                break
        steps.append({"albedo": [round(float(v), 4) for v in c], "misurato_srgb": [round(float(v), 1) for v in meas],
                      "render": n})
        if np.abs(meas - target).max() <= 1.0:
            break
        lc, lm = np.log(np.maximum(c, 1e-4)), np.log(np.maximum(srgb_to_lin(meas / 255.0), 1e-4))
        g = np.ones(3)
        if prev is not None:
            dc = lc - prev[0]
            g = np.where(np.abs(dc) > 1e-6, (lm - prev[1]) / np.where(np.abs(dc) > 1e-6, dc, 1.0), 1.0)
            g = np.clip(g, 0.15, 1.5)
        prev = (lc, lm)
        step = np.clip((np.log(target_lin) - lm) / g, math.log(0.7), math.log(1.4))
        c = np.clip(np.exp(lc + step), 1e-3, 1.0)
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
    set_cam_attr(cam_prim_path, "focalLength", f)  # stesso problema del fuoco: va nel session layer


def set_vfov(cam_prim_path: str, vfov_deg: float, resolution) -> dict:
    """Campo visivo verticale come in MuJoCo (fovy). Kit tiene fissa l'apertura orizzontale (20.955) e ricava
    il resto dal formato dell'immagine: si sceglie la focale che da' vfov con l'apertura verticale = orizzontale
    * H / W, e si scrive anche quella, cosi' il risultato non dipende da quale delle due il renderer usa."""
    import omni.usd

    ha = float(omni.usd.get_context().get_stage().GetPrimAtPath(cam_prim_path).GetAttribute("horizontalAperture").Get())
    w, h = resolution
    va = ha * h / w
    f = va / (2.0 * math.tan(math.radians(vfov_deg) / 2.0))
    set_cam_attr(cam_prim_path, "verticalAperture", va)
    set_cam_attr(cam_prim_path, "focalLength", f)
    hfov = 2.0 * math.degrees(math.atan(ha / (2.0 * f)))
    return {"vfov_deg": vfov_deg, "hfov_deg": round(hfov, 2), "focale": round(f, 3), "apertura_o": ha, "apertura_v": round(va, 3)}


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
    env_cfg, dr, arrows, crowd = build_env_cfg(preset, args_cli, log)

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
        if preset["cam"]["mode"] == "robot":  # azim = direzione di VISTA rispetto al robot, girato di YAW_MINUS_X
            cam_azim = math.degrees(YAW_MINUS_X) + cam_azim + 180.0
        if args_cli.sun_rel_azim is None:
            args_cli.sun_rel_azim = preset.get("sun_rel_azim", 135.0)
        sun = add_sun(cam_azim, args_cli)
        robot_root = env_cfg.scene.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_0")
        recolored = recolor_robot(robot_root.replace("env_.*", "env_0"), args_cli.robot_color,
                                  args_cli.robot_roughness, args_cli.robot_metallic)
    cam = Camera(base, preset["cam"], args_cli, duration=args_cli.duration or preset.get("duration"))
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
        # istanti nel tempo del video: il fotogramma al passo i e' quello del video a (i - preroll + 1) * dt
        previews = {max(preroll + int(round(t / dt)) - 1, 0): t for t in args_cli.preview_at}
        total, kind = max(previews) + 1, "preview"
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
    track = []  # V7: posa e fuoco della camera a ogni fotogramma (-> <video>_camera.csv)
    h_term = base.command_manager.get_term("base_height")
    rng = np.random.default_rng(args_cli.seed)
    writer = None
    if kind == "video":
        name = preset["name"]
        if preset["cam"]["mode"] == "orbit":  # il nome dice distanza ed elevazione davvero usate
            d = args_cli.cam_dist if args_cli.cam_dist is not None else preset["cam"]["dist"]
            el = args_cli.cam_elev if args_cli.cam_elev is not None else preset["cam"]["elev"]
            name = f"{name.split('_orbita')[0]}_orbita_d{d:g}_e{el:g}"
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
        cam.update(t=(i - preroll + 1) * dt)

        if kind == "video":
            if i < preroll:
                base.sim.render()  # renderer al passo con la scena anche prima di registrare
            else:
                if i == preroll:
                    settle(8)  # scartati: niente fotogramma vecchio o mezzo mediato in apertura
                writer.write(base.render())
                if cam.p["mode"] == "traj":
                    track.append(cam.row())
                if base.num_envs == 1:
                    c = vel_term.command[0].tolist()
                    v = robot.data.root_lin_vel_b[0].tolist()
                    w = robot.data.root_ang_vel_b[0].tolist()
                    pos = robot.data.root_pos_w[0].tolist()
                    tele.append([round((i - preroll + 1) * dt, 3), *c, float(h_term.command[0, 0]), v[0], v[1], w[2],
                                 math.degrees(float(robot.data.heading_w[0])), *pos])
        elif kind == "stills" and i in shots and arrows == "off":  # S4: una sola versione, senza frecce
            k = shots[i]
            p = out_path(out_dir, preset["name"] if len(shots) == 1 else f"{preset['name']}_{k}", ".png", args_cli.overwrite)
            save_png(settle(args_cli.settle), p)
            files.append(p.name)
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
        elif kind == "preview" and i in previews:
            suffix = "" if args_cli.ground == "grid" else f"_{args_cli.ground}"  # anteprime confrontabili
            tag = f"_t{previews[i]:g}" if len(previews) > 1 else ""
            for fs in args_cli.preview_fstop or [None]:
                ftag = ""
                if fs is not None:
                    set_cam_attr(cam.cam_path, "fStop", fs, quiet=True)
                    ftag = f"_f{fs:g}"
                p = out_dir / f"_preview_{args_cli.preset}{suffix}{tag}{ftag}.png"
                save_png(settle(args_cli.settle), p)
                files.append(p.name)
            if args_cli.preview_fstop:
                set_cam_attr(cam.cam_path, "fStop", cam.fstop, quiet=True)
            if cam.p["mode"] == "traj":
                print(f"[media]   t {previews[i]:g} s: camera {np.round(cam.eye, 2).tolist()} -> "
                      f"{np.round(cam.look, 2).tolist()}, fuoco {cam.focus:.1f} m")

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
    if track:
        cam_csv = out_dir / f"{files[0].rsplit('.', 1)[0]}_camera.csv"
        np.savetxt(cam_csv, np.array(track), delimiter=",", comments="", fmt="%.4f",
                   header="t,eye_x,eye_y,eye_z,look_x,look_y,look_z,fuoco_m")
        files.append(cam_csv.name)
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
        "fstop": cam.fstop or None, "folla": crowd,
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
