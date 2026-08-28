#!/usr/bin/env python3
"""Analisi quantitativa di un run sim2sim (MuJoCo C++ + controller di deploy).

CSV letti da sim_logs/sim_<ts>/ (ognuno sul PROPRIO asse temporale):
  - ground_truth.csv : stato VERO della base (sim_t, 100 Hz)  -> posa xyz + orientazione
  - obs.csv          : osservazioni in input al controller (ep_t, 50 Hz)
  - commands.csv     : comandi q_target inviati al robot (ep_t)
  - errors.csv       : errore di tracking di giunto q_target-q_misurato (ep_t)

I giunti sono in ordine-policy; vengono divisi in GAMBE (hip/knee/ankle/torso)
e BRACCIA (shoulder/elbow/wrist). NB: obs e ground_truth NON vengono allineati
tra loro qui (il wall_t del run vecchio era a bassa precisione); ogni grafico usa
il tempo del proprio file.

Uso:
    python3 analyze_sim.py [cartella_run]     # default: ultima sim_* in ~/sim_logs (o ~/unitree_ws/sim_logs)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- ordine-policy dei 27 giunti H1_2 (i = indice in joint_pos_rel/q_target/q_err) ---
JOINT_NAMES = [
    "left_hip_yaw", "right_hip_yaw", "torso", "left_hip_pitch", "right_hip_pitch",
    "left_shoulder_pitch", "right_shoulder_pitch", "left_hip_roll", "right_hip_roll",
    "left_shoulder_roll", "right_shoulder_roll", "left_knee", "right_knee",
    "left_shoulder_yaw", "right_shoulder_yaw", "left_ankle_pitch", "right_ankle_pitch",
    "left_elbow", "right_elbow", "left_ankle_roll", "right_ankle_roll",
    "left_wrist_roll", "right_wrist_roll", "left_wrist_pitch", "right_wrist_pitch",
    "left_wrist_yaw", "right_wrist_yaw",
]
DEFAULT_JOINT_POS = np.array([
    0.0, 0.0, 0.0, -0.3, -0.3, 0.35, 0.35, 0.0, 0.0, 0.25, -0.25, 0.5, 0.5,
    -0.1, -0.1, -0.2, -0.2, 0.65, 0.65, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
])
ARM_KEYS = ("shoulder", "elbow", "wrist")


def is_arm(name):
    return any(k in name for k in ARM_KEYS)


def joint_type(name):
    """togli left_/right_ per accoppiare L/R nello stesso subplot."""
    return name.replace("left_", "").replace("right_", "")


def side(name):
    return "L" if name.startswith("left_") else ("R" if name.startswith("right_") else "")


def groups(indices):
    """dict ordinato tipo -> [(idx, side)] preservando l'ordine di comparsa."""
    g = {}
    for i in indices:
        g.setdefault(joint_type(JOINT_NAMES[i]), []).append((i, side(JOINT_NAMES[i])))
    return g


def latest_run():
    for base in [os.path.expanduser("~/sim_logs"), os.path.expanduser("~/unitree_ws/sim_logs")]:
        runs = sorted(glob.glob(os.path.join(base, "sim_*")))
        if runs:
            return runs[-1]
    sys.exit("Nessun run trovato in ~/sim_logs o ~/unitree_ws/sim_logs")


def quat_to_rpy(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def jmat(df, suffix, fb_prefix, fb_off):
    """Matrice (N,27) in ordine-policy: prova colonne <giunto>_<suffix> (nuove),
    altrimenti fallback posizionale <fb_prefix><fb_off+i> (CSV vecchi)."""
    named = [f"{n}_{suffix}" for n in JOINT_NAMES]
    if all(c in df.columns for c in named):
        return df[named].to_numpy()
    return df[[f"{fb_prefix}{fb_off + i}" for i in range(27)]].to_numpy()


def vecc(df, named, fb):
    """Colonne nominate (nuove) o fallback posizionale (vecchie)."""
    cols = named if all(c in df.columns for c in named) else fb
    return df[cols].to_numpy()


# famiglie per riga (torso escluso: e' l'unico giunto di tronco)
FAMILIES = {"legs": ["hip", "knee", "ankle"], "arms": ["shoulder", "elbow", "wrist"]}


def dof_label(name, family):
    s = name.replace("left_", "").replace("right_", "").replace(family, "").strip("_")
    return s if s else family


def plot_part(t, mat, part, title, ylabel, path):
    """Griglia 3 righe (famiglie) x 2 colonne (SX/DX). mat: (N,27)."""
    fams = FAMILIES[part]
    sides = [("left", "SX"), ("right", "DX")]
    fig, axes = plt.subplots(len(fams), 2, figsize=(11, 8.5), sharex=True, squeeze=False)
    for r, fam in enumerate(fams):
        for c, (pref, slab) in enumerate(sides):
            ax = axes[r][c]
            for i in range(27):
                nm = JOINT_NAMES[i]
                if nm.startswith(pref) and fam in nm:
                    ax.plot(t, mat[:, i], lw=1.0, label=dof_label(nm, fam))
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            ax.legend(fontsize=8, loc="upper right")
            if r == 0:
                ax.set_title(slab, fontsize=12, fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{fam}\n{ylabel}", fontsize=9)
    fig.suptitle(title, fontsize=13)
    fig.supxlabel("t [s]", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    run = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    print(f"== Run: {run}")
    outdir = os.path.join(run, "plots")
    os.makedirs(outdir, exist_ok=True)

    obs = pd.read_csv(os.path.join(run, "obs.csv"))
    cmd = pd.read_csv(os.path.join(run, "commands.csv"))
    err = pd.read_csv(os.path.join(run, "errors.csv"))
    gt = pd.read_csv(os.path.join(run, "ground_truth.csv"))

    leg_idx = [i for i in range(27) if not is_arm(JOINT_NAMES[i])]
    arm_idx = [i for i in range(27) if is_arm(JOINT_NAMES[i])]

    # ---- posizione VERA dei giunti = joint_pos_rel (obs 9:36) + default
    t_obs = obs["ep_t"].to_numpy()
    jpos_rel = jmat(obs, "pos", "obs_", 9)  # joint_pos_rel (nuovo: <giunto>_pos)
    jpos = jpos_rel + DEFAULT_JOINT_POS  # broadcasting su 27 colonne
    plot_part(t_obs, jpos, "legs", "Posizione giunti - GAMBE", "q [rad]",
              os.path.join(outdir, "01_joint_pos_legs.png"))
    plot_part(t_obs, jpos, "arms", "Posizione giunti - BRACCIA", "q [rad]",
              os.path.join(outdir, "02_joint_pos_arms.png"))

    # ---- errore di posizione (q_target - q_misurato) da errors.csv
    t_err = err["ep_t"].to_numpy()
    qerr = jmat(err, "q_err", "q_err_", 0)
    plot_part(t_err, qerr, "legs", "Errore posizione giunti - GAMBE", "err [rad]",
              os.path.join(outdir, "03_joint_err_legs.png"))
    plot_part(t_err, qerr, "arms", "Errore posizione giunti - BRACCIA", "err [rad]",
              os.path.join(outdir, "04_joint_err_arms.png"))

    # ---- comandi della policy q_target da commands.csv
    t_cmd = cmd["ep_t"].to_numpy()
    qtgt = jmat(cmd, "q_target", "q_target_", 0)
    plot_part(t_cmd, qtgt, "legs", "Comandi policy q_target - GAMBE", "q_target [rad]",
              os.path.join(outdir, "06_cmd_legs.png"))
    plot_part(t_cmd, qtgt, "arms", "Comandi policy q_target - BRACCIA", "q_target [rad]",
              os.path.join(outdir, "07_cmd_arms.png"))

    # ---- posa base: xyz + orientazione (ground_truth)
    t_gt = gt["sim_t"].to_numpy()
    q = gt[["qw", "qx", "qy", "qz"]].to_numpy()
    roll, pitch, yaw = quat_to_rpy(q)
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(t_gt, gt["base_x"], label="x")
    ax[0].plot(t_gt, gt["base_y"], label="y")
    ax[0].plot(t_gt, gt["base_z"], label="z")
    ax[0].set_ylabel("posizione [m]"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("Posizione XYZ della base")
    ax[1].plot(t_gt, np.degrees(roll), label="roll")
    ax[1].plot(t_gt, np.degrees(pitch), label="pitch")
    ax[1].plot(t_gt, np.degrees(yaw), label="yaw")
    ax[1].set_ylabel("orientazione [deg]"); ax[1].set_xlabel("t [s]"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "05_base_pose.png"), dpi=110); plt.close(fig)

    # ---- tutte le altre osservazioni
    fig, ax = plt.subplots(3, 2, figsize=(13, 9))
    ax = ax.ravel()
    bav = vecc(obs, ["base_ang_vel_0", "base_ang_vel_1", "base_ang_vel_2"], ["obs_0", "obs_1", "obs_2"])
    pg = vecc(obs, ["projected_gravity_0", "projected_gravity_1", "projected_gravity_2"], ["obs_3", "obs_4", "obs_5"])
    vc = vecc(obs, ["velocity_commands_0", "velocity_commands_1", "velocity_commands_2"], ["obs_6", "obs_7", "obs_8"])
    gp = vecc(obs, ["gait_phase_0", "gait_phase_1"], ["obs_90", "obs_91"])
    jv = jmat(obs, "vel", "obs_", 36)  # joint_vel_rel
    la = jmat(obs, "act", "obs_", 63)  # last_action
    for k, lab in zip(range(3), ["wx", "wy", "wz"]):
        ax[0].plot(t_obs, bav[:, k], label=lab)
    ax[0].set_title("base_ang_vel (scalata 0.2)"); ax[0].legend(fontsize=7)
    for k, lab in zip(range(3), ["gx", "gy", "gz"]):
        ax[1].plot(t_obs, pg[:, k], label=lab)
    ax[1].set_title("projected_gravity"); ax[1].legend(fontsize=7)
    for k, lab in zip(range(3), ["vx", "vy", "wz"]):
        ax[2].plot(t_obs, vc[:, k], label=lab)
    ax[2].set_title("velocity_commands"); ax[2].legend(fontsize=7)
    ax[3].plot(t_obs, gp); ax[3].set_title("gait_phase (sin, cos)")
    ax[4].plot(t_obs, jv, lw=0.6); ax[4].set_title("joint_vel_rel (27, scalata 0.05)")
    ax[5].plot(t_obs, la, lw=0.6); ax[5].set_title("last_action (27)")
    for a in ax:
        a.grid(alpha=0.3); a.tick_params(labelsize=7); a.set_xlabel("t [s]", fontsize=8)
    fig.suptitle("Altre osservazioni", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "08_other_obs.png"), dpi=110); plt.close(fig)

    # ---- metriche a schermo
    print("\n== Stabilita' (ground_truth)")
    print(f"  altezza base media = {gt['base_z'].mean():.3f} m (std {gt['base_z'].std():.4f})")
    print(f"  |roll| max = {np.max(np.abs(np.degrees(roll))):.1f} deg  |pitch| max = {np.max(np.abs(np.degrees(pitch))):.1f} deg")
    print(f"  deriva xy = {np.hypot(gt['base_x'].iloc[-1]-gt['base_x'].iloc[0], gt['base_y'].iloc[-1]-gt['base_y'].iloc[0]):.2f} m")
    print("== Tracking di giunto (q_target - q_misurato)")
    print(f"  GAMBE : |err| medio={np.abs(qerr[:, leg_idx]).mean():.4f}  max={np.abs(qerr[:, leg_idx]).max():.4f} rad")
    print(f"  BRACCIA: |err| medio={np.abs(qerr[:, arm_idx]).mean():.4f}  max={np.abs(qerr[:, arm_idx]).max():.4f} rad")
    print(f"\n== {len([f for f in os.listdir(outdir) if f.endswith('.png')])} plot salvati in {outdir}")


if __name__ == "__main__":
    main()
