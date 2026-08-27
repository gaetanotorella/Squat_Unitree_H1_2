#!/usr/bin/env python3
"""Analisi quantitativa di un run sim2sim (MuJoCo C++ + controller di deploy).

Legge i 4 CSV prodotti in unitree_ws/sim_logs/sim_<ts>/:
  - ground_truth.csv : stato VERO della base dal simulatore (posa + velocita')
  - obs.csv          : tutte le osservazioni in input al controller
  - commands.csv     : comandi inviati al robot (q_target, kp, kd)
  - errors.csv       : errore di tracking di giunto (q_target - q_misurato)

Allinea controller <-> ground_truth tramite wall_t (orologio di sistema).
Genera i plot in una sottocartella plots/ e stampa le metriche a schermo.

Uso:
    python3 analyze_sim.py [cartella_run]     # default: ultima sim_* in ~/unitree_ws/sim_logs
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------- utils
def latest_run():
    base = os.path.join(os.path.expanduser("~"), "unitree_ws/sim_logs")
    runs = sorted(glob.glob(os.path.join(base, "sim_*")))
    if not runs:
        sys.exit(f"Nessun run trovato in {base}")
    return runs[-1]


def quat_to_rpy(q):
    """q: (N,4) w,x,y,z -> roll,pitch,yaw (N,)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def world_vel_to_base(vw, q):
    """Ruota la velocita' lineare da world a base-frame. vw:(N,3) q:(N,4) w,x,y,z."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # matrice di rotazione R (body->world), poi v_body = R^T v_world
    out = np.zeros_like(vw)
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - w * z); R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y); R[:, 2, 1] = 2 * (y * z + w * x); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    for i in range(len(q)):
        out[i] = R[i].T @ vw[i]
    return out


def rmse(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.any() else float("nan")


# ----------------------------------------------------------------------------- load
def main():
    run = sys.argv[1] if len(sys.argv) > 1 else latest_run()
    run = os.path.abspath(run)
    print(f"== Run: {run}")
    outdir = os.path.join(run, "plots")
    os.makedirs(outdir, exist_ok=True)

    gt = pd.read_csv(os.path.join(run, "ground_truth.csv"))
    obs = pd.read_csv(os.path.join(run, "obs.csv"))
    cmd = pd.read_csv(os.path.join(run, "commands.csv"))
    err = pd.read_csv(os.path.join(run, "errors.csv"))

    # allinea ground_truth ai timestamp del controller (nearest su wall_t)
    gt = gt.sort_values("wall_t")
    obs = obs.sort_values("wall_t")
    m = pd.merge_asof(obs, gt, on="wall_t", direction="nearest", suffixes=("", "_gt"))
    t = m["ep_t"].to_numpy()

    # velocita' vera in base-frame
    vw = m[["vx", "vy", "vz"]].to_numpy()
    q = m[["qw", "qx", "qy", "qz"]].to_numpy()
    vb = world_vel_to_base(vw, q)
    act_vx, act_vy = vb[:, 0], vb[:, 1]
    act_wz = m["wz"].to_numpy()  # ang. vel body-frame ~ yaw rate (robot eretto)

    # comandi (obs velocity_commands, scala 1.0)
    cmd_vx = m["velocity_commands_0"].to_numpy()
    cmd_vy = m["velocity_commands_1"].to_numpy()
    cmd_wz = m["velocity_commands_2"].to_numpy()

    roll, pitch, _ = quat_to_rpy(q)
    base_z = m["base_z"].to_numpy()

    # ---- metriche
    print("\n== Metriche di tracking (comando vs reale, base-frame)")
    print(f"  RMSE vx  = {rmse(cmd_vx, act_vx):.4f} m/s")
    print(f"  RMSE vy  = {rmse(cmd_vy, act_vy):.4f} m/s")
    print(f"  RMSE wz  = {rmse(cmd_wz, act_wz):.4f} rad/s")
    print("== Stabilita'")
    print(f"  altezza base media = {np.nanmean(base_z):.3f} m  (std {np.nanstd(base_z):.4f})")
    print(f"  |roll| max = {np.nanmax(np.abs(roll)):.3f} rad   |pitch| max = {np.nanmax(np.abs(pitch)):.3f} rad")
    qerr_cols = [c for c in err.columns if c.startswith("q_err_")]
    qerr = err[qerr_cols].to_numpy()
    print("== Tracking di giunto (q_target - q_misurato)")
    print(f"  |err| medio = {np.nanmean(np.abs(qerr)):.4f} rad   |err| max = {np.nanmax(np.abs(qerr)):.4f} rad")

    # ---- plot 1: comando vs reale
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for a, c, r, lab in zip(ax, [cmd_vx, cmd_vy, cmd_wz], [act_vx, act_vy, act_wz],
                            ["v_x [m/s]", "v_y [m/s]", "w_z [rad/s]"]):
        a.plot(t, c, "k--", lw=1.5, label="comando")
        a.plot(t, r, "b-", lw=1.2, label="reale")
        a.set_ylabel(lab); a.grid(alpha=0.3); a.legend(loc="upper right")
    ax[0].set_title("Tracking del comando di velocita' (base-frame)")
    ax[-1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "01_velocity_tracking.png"), dpi=110); plt.close(fig)

    # ---- plot 2: stabilita' (altezza + orientazione)
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(t, base_z, "g-"); ax[0].set_ylabel("altezza base [m]"); ax[0].grid(alpha=0.3)
    ax[0].set_title("Stabilita' posturale")
    ax[1].plot(t, np.degrees(roll), label="roll"); ax[1].plot(t, np.degrees(pitch), label="pitch")
    ax[1].set_ylabel("[deg]"); ax[1].set_xlabel("t [s]"); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "02_stability.png"), dpi=110); plt.close(fig)

    # ---- plot 3: errore di tracking velocita'
    fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for a, c, r, lab in zip(ax, [cmd_vx, cmd_vy, cmd_wz], [act_vx, act_vy, act_wz],
                            ["err v_x", "err v_y", "err w_z"]):
        a.plot(t, c - r, "r-", lw=1.0); a.axhline(0, color="k", lw=0.6)
        a.set_ylabel(lab); a.grid(alpha=0.3)
    ax[0].set_title("Errore di tracking (comando - reale)")
    ax[-1].set_xlabel("t [s]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "03_tracking_error.png"), dpi=110); plt.close(fig)

    # ---- plot 4: errore di tracking di giunto
    te = err["ep_t"].to_numpy()
    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(te, np.max(np.abs(qerr), axis=1), "m-", label="max |err| giunti")
    ax[0].plot(te, np.mean(np.abs(qerr), axis=1), "c-", label="media |err| giunti")
    ax[0].set_ylabel("[rad]"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[0].set_title("Errore di tracking di giunto (PD)")
    im = ax[1].imshow(np.abs(qerr).T, aspect="auto", origin="lower",
                      extent=[te[0], te[-1], 0, qerr.shape[1]], cmap="viridis")
    ax[1].set_ylabel("giunto #"); ax[1].set_xlabel("t [s]")
    fig.colorbar(im, ax=ax[1], label="|err| [rad]")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "04_joint_tracking_error.png"), dpi=110); plt.close(fig)

    # ---- plot 5: traiettoria x-y
    fig, a = plt.subplots(figsize=(7, 7))
    a.plot(m["base_x"], m["base_y"], "b-"); a.plot(m["base_x"].iloc[0], m["base_y"].iloc[0], "go", label="start")
    a.plot(m["base_x"].iloc[-1], m["base_y"].iloc[-1], "rs", label="end")
    a.set_xlabel("x [m]"); a.set_ylabel("y [m]"); a.axis("equal"); a.grid(alpha=0.3); a.legend()
    a.set_title("Traiettoria della base")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "05_trajectory.png"), dpi=110); plt.close(fig)

    print(f"\n== Plot salvati in {outdir}")


if __name__ == "__main__":
    main()
