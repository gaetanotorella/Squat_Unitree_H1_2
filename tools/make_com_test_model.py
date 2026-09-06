#!/usr/bin/env python3
"""Genera un modello MuJoCo dell'H1_2 con il BARICENTRO SPOSTATO, per il test sim2sim.

PERCHE'.  Il sim2sim gira sul modello nominale, dove il baricentro e' gia' al posto giusto:
conferma che la policy ha imparato una posa migliore, ma NON puo' dire se ha imparato ad
ADATTARSI a un CoM diverso -- che e' esattamente lo scopo della randomizzazione del CoM
(``randomize_base_com``, rapporto 1:1 su tutti i link) introdotta nel round 4.

Sul robot reale il baricentro e' risultato ~3-4 cm PIU' INDIETRO di quanto il modello preveda
(stimato dalla coppia di caviglia a regime, tau = M*g*d). Questo script inietta quell'errore
nel modello di simulazione, cosi' il test si puo' fare a terra invece che appesi alla gru.

COSA FA.  Somma lo stesso offset alla ``pos`` di OGNI tag ``<inertial>``, cioe' al baricentro
LOCALE di ogni corpo. E' la riproduzione fedele di cio' che ``randomize_body_com`` fa in
addestramento, dove PhysX ``set_coms()`` scrive anch'esso il CoM nel frame locale del corpo.
Massa e tensore d'inerzia non vengono toccati: in MuJoCo ``diaginertia`` e' gia' espresso
attorno al baricentro, quindi spostare il baricentro equivale a traslare rigidamente la
distribuzione di massa -- che e' proprio il modello di errore che vogliamo.

CALIBRAZIONE.  Un offset uguale nei frame LOCALI non da' uno spostamento identico nel frame
del mondo, perche' i corpi (coscia, tibia, piede, braccia) sono ruotati rispetto alla base.
Lo script misura quindi il guadagno reale k = (spostamento nel mondo)/(offset locale) alla posa
di riferimento e riscala l'offset, in modo che lo spostamento richiesto sia quello EFFETTIVO.

USO
    python3 tools/make_com_test_model.py                  # -3.5 cm all'indietro (default)
    python3 tools/make_com_test_model.py --shift -0.05 --tag com_back50
    python3 tools/make_com_test_model.py --shift -0.035 --axis y --tag com_left35

Poi in ``unitree_mujoco/simulate_python/config.py``:
    ROBOT_SCENE = "../unitree_robots/h1_2/scene_com_test.xml"
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import mujoco
import numpy as np

ROBOT_DIR = Path("/home/gaetano/unitree_ws/unitree_mujoco/unitree_robots/h1_2")
SRC_MODEL = ROBOT_DIR / "h1_2_handless.xml"
SRC_SCENE = ROBOT_DIR / "scene.xml"

# posa di riferimento per la calibrazione: la default del training (27 giunti handless)
JN = ["left_hip_yaw", "right_hip_yaw", "torso", "left_hip_pitch", "right_hip_pitch",
      "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_roll", "right_hip_roll",
      "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee",
      "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle_pitch", "right_ankle_pitch",
      "left_elbow", "right_elbow", "left_ankle_roll", "right_ankle_roll",
      "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch",
      "left_wrist_yaw", "right_wrist_yaw"]
DEFV = np.array([0, 0, 0, -.3, -.3, .35, .35, 0, 0, .25, -.25, .5, .5, -.1, -.1,
                 -.2, -.2, .65, .65, 0, 0, 0, 0, 0, 0, 0, 0.])

INERTIAL_POS = re.compile(r'(<inertial\b[^>]*?\bpos=")([^"]+)(")')


def com_and_ankle(model_path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """CoM del corpo intero, centro caviglie e massa totale alla posa di riferimento."""
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")] for n in JN]
    d.qpos[:] = 0
    d.qpos[3] = 1.0
    d.qpos[qadr] = DEFV
    mujoco.mj_kinematics(m, d)
    mujoco.mj_comPos(m, d)
    mass = m.body_mass.sum()
    com = np.sum(m.body_mass[:, None] * d.xipos, axis=0) / mass
    ank = np.mean([d.xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{s}_ankle_pitch_link")]
                   for s in ("left", "right")], axis=0)
    return com, ank, mass


def write_model(dst: Path, offset: np.ndarray, header: str) -> int:
    text = SRC_MODEL.read_text()
    n = 0

    def sub(mo: re.Match) -> str:
        nonlocal n
        n += 1
        p = np.array([float(v) for v in mo.group(2).split()]) + offset
        return f"{mo.group(1)}{p[0]:.6g} {p[1]:.6g} {p[2]:.6g}{mo.group(3)}"

    out = INERTIAL_POS.sub(sub, text)
    # il commento va DOPO la dichiarazione <mujoco ...>, non prima (XML: un solo root)
    out = out.replace("<mujoco model=\"h1_2\">", "<mujoco model=\"h1_2\">\n" + header, 1)
    dst.write_text(out)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shift", type=float, default=-0.035,
                    help="spostamento EFFETTIVO del CoM globale, in metri (negativo = indietro). Default -0.035")
    ap.add_argument("--axis", choices=["x", "y", "z"], default="x")
    ap.add_argument("--tag", default="com_test", help="suffisso dei file generati. Default 'com_test'")
    args = ap.parse_args()

    axis_i = {"x": 0, "y": 1, "z": 2}[args.axis]
    dst_model = ROBOT_DIR / f"h1_2_handless_{args.tag}.xml"
    dst_scene = ROBOT_DIR / f"scene_{args.tag}.xml"
    for p in (dst_model, dst_scene):
        if p.exists():
            print(f"  [i] sovrascrivo {p.name}")

    com0, ank0, mass = com_and_ankle(SRC_MODEL)
    print(f"MODELLO ORIGINALE ({SRC_MODEL.name})  -- INTATTO, non viene toccato")
    print(f"  massa totale        {mass:.2f} kg")
    print(f"  CoM - caviglia (x)  {(com0[0] - ank0[0]) * 100:+.2f} cm")

    # --- calibrazione: quanto vale, nel mondo, un offset unitario nei frame locali? ---
    probe = 0.01
    off = np.zeros(3)
    off[axis_i] = probe
    tmp = ROBOT_DIR / f".calib_{args.tag}.xml"
    write_model(tmp, off, "  <!-- calibrazione -->")
    com1, _, _ = com_and_ankle(tmp)
    k = (com1[axis_i] - com0[axis_i]) / probe
    tmp.unlink()

    off = np.zeros(3)
    off[axis_i] = args.shift / k
    header = (f"  <!-- MODELLO DI TEST: baricentro spostato di {args.shift * 100:+.1f} cm lungo {args.axis}\n"
              f"       (offset {off[axis_i] * 100:+.2f} cm applicato al CoM LOCALE di ogni corpo, guadagno k={k:.4f}).\n"
              f"       Generato da tools/make_com_test_model.py -- NON modificare a mano.\n"
              f"       Serve a verificare se la policy si ADATTA a un baricentro diverso da quello\n"
              f"       nominale, riproducendo a terra l'errore di modello misurato sul robot reale. -->")
    n = write_model(dst_model, off, header)

    com2, ank2, mass2 = com_and_ankle(dst_model)
    real = (com2[axis_i] - com0[axis_i]) * 100

    print(f"\nCALIBRAZIONE")
    print(f"  guadagno locale->mondo   k = {k:.4f}  (i corpi ruotati non contribuiscono 1:1)")
    print(f"  offset applicato ai {n} corpi: {off[axis_i] * 100:+.2f} cm nel frame locale")

    print(f"\nMODELLO GENERATO ({dst_model.name})")
    print(f"  massa totale        {mass2:.2f} kg   (invariata: {abs(mass2 - mass) < 1e-9})")
    print(f"  CoM - caviglia (x)  {(com2[0] - ank2[0]) * 100:+.2f} cm   "
          f"(era {(com0[0] - ank0[0]) * 100:+.2f})")
    print(f"  spostamento OTTENUTO {real:+.2f} cm   richiesto {args.shift * 100:+.2f} cm   "
          f"errore {abs(real - args.shift * 100):.3f} cm")
    print(f"\n  NB: i due 'CoM - caviglia' qui sopra sono alla POSA DI RIFERIMENTO (la default del")
    print(f"      training), non alla posa che la policy sceglie davvero. Cio' che conta e' lo")
    print(f"      SPOSTAMENTO ({real:+.2f} cm): va sommato al CoM misurato nel run sul modello nominale.")
    print(f"      Coppia di caviglia aggiuntiva che ne deriva: "
          f"{mass2 * 9.81 * abs(args.shift):.1f} N*m (= M*g*{abs(args.shift):.3f}).")

    # --- scena che include il modello modificato ---
    scene = SRC_SCENE.read_text().replace(
        '<include file="h1_2_handless.xml" />',
        f'<include file="{dst_model.name}" />  <!-- MODELLO CON CoM SPOSTATO -->')
    dst_scene.write_text(scene)
    print(f"\nSCENA GENERATA ({dst_scene.name})")
    print(f"  per usarla, in unitree_mujoco/simulate_python/config.py:")
    print(f'      ROBOT_SCENE = "../unitree_robots/h1_2/{dst_scene.name}"')
    print(f"  per tornare al modello nominale, rimetti:")
    print(f'      ROBOT_SCENE = "../unitree_robots/h1_2/scene.xml"')


if __name__ == "__main__":
    main()
