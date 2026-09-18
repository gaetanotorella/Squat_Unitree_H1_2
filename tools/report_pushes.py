#!/usr/bin/env python3
"""Risposta del robot alle spinte MANUALI date da tastiera nella finestra MuJoCo.

Le spinte le applica il simulatore (unitree_mujoco/simulate/src/push_disturbance.h, preset in
simulate/disturbances.yaml) durante una normale sessione sim2sim col joystick fisico, e le logga in
sim_logs/sim_<ts>/disturbance_events.csv. Questo script le legge insieme a ground_truth.csv e
obs.csv e per ogni spinta misura: variazione di velocita' del baricentro (contro quella attesa
dall'impulso: verifica che la forza sia arrivata davvero), picco di pitch/roll, spostamento massimo
e finale, rotazione di imbardata, tempo di assestamento, caduta. Scrive pushes.json.

    ./report_pushes.py ~/unitree_ws/sim_logs/sim_<ts>

Non fa parte del banco automatico (run_protocol.py): e' uno strumento per il test manuale.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

PRE = 0.5      # [s] finestra prima della spinta per la baseline
POST = 4.0     # [s] finestra di osservazione dopo la spinta
V_SETTLE = 0.05    # [m/s]
TILT_SETTLE = 1.0  # [deg] rispetto alla baseline
HOLD = 0.5         # [s] per dichiarare assestato
FALL_Z = 0.50      # [m]
FALL_TILT = 30.0   # [deg]


def main() -> int:
    run = Path(sys.argv[1]).resolve()
    ev = pd.read_csv(run / "disturbance_events.csv")
    gt = pd.read_csv(run / "ground_truth.csv")
    obs = pd.read_csv(run / "obs.csv")
    t = gt.sim_t.to_numpy()
    q = gt[["qw", "qx", "qy", "qz"]].to_numpy()
    # roll/pitch dal quaternione della base (negativo pitch = avanti, come nelle altre analisi)
    pitch = -np.degrees(np.arcsin(np.clip(2 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1, 1)))
    roll = np.degrees(np.arctan2(2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))
    v = gt[["vx", "vy"]].to_numpy(); z = gt.base_z.to_numpy(); xy = gt[["base_x", "base_y"]].to_numpy()
    yaw = np.degrees(np.unwrap(np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))))
    # quota comandata sull'asse del ground truth
    osim = np.interp(obs.wall_t.to_numpy(), gt.wall_t.to_numpy(), t)
    hcmd = np.interp(t, osim, obs.base_height_command_0.to_numpy() + 0.98)

    starts = ev[ev.event == "start"].reset_index(drop=True)
    rows = []
    for _, e in starts.iterrows():
        t0 = float(e.sim_t); dur = float(e.dur_planned)
        F = np.array([e.fx, e.fy, e.fz]); T = np.array([e.tx, e.ty, e.tz])
        imp = np.linalg.norm(F) * max(dur, 0.005)
        pre = (t >= t0 - PRE) & (t < t0); post = (t >= t0) & (t < t0 + POST)
        v0 = v[pre].mean(0); p0 = pitch[pre].mean(); r0 = roll[pre].mean(); xy0 = xy[pre].mean(0)
        dv = np.linalg.norm(v[post] - v0, axis=1)
        dp = pitch[post] - p0; dr = roll[post] - r0
        disp = np.linalg.norm(xy[post] - xy0, axis=1)
        dyaw = yaw[post] - yaw[pre].mean()
        fell = bool((z[post] < FALL_Z).any() or (np.abs(dp) > FALL_TILT).any() or (np.abs(dr) > FALL_TILT).any())
        # assestamento: primo istante da cui, per HOLD s, |dv|<V_SETTLE e |tilt|<TILT_SETTLE
        tp = t[post]; ok = (dv < V_SETTLE) & (np.abs(dp) < TILT_SETTLE) & (np.abs(dr) < TILT_SETTLE)
        settle = None
        n_hold = int(HOLD / np.median(np.diff(tp)))
        for i in range(len(ok) - n_hold):
            if ok[i:i + n_hold].all(): settle = float(tp[i] - t0); break
        rows.append({
            "t": round(t0, 2), "body": e.body, "h_cmd": round(float(hcmd[pre].mean()), 3),
            "force": [float(x) for x in F], "torque": [float(x) for x in T], "dur": dur, "impulse_Ns": round(float(imp), 1),
            "dv_peak": round(float(dv.max()), 3), "dv_expected": round(float(imp / 79.5), 3),
            "dpitch_peak": round(float(dp[np.argmax(np.abs(dp))]), 2), "droll_peak": round(float(dr[np.argmax(np.abs(dr))]), 2),
            "disp_max": round(float(disp.max()), 3), "disp_end": round(float(disp[-1]), 3),
            "dyaw_end": round(float(dyaw[-1]), 2),
            "settle_s": None if settle is None else round(settle, 2), "fell": fell,
        })
    (run / "pushes.json").write_text(json.dumps(rows, indent=2))
    print(f"{'t':>6}{'h':>6}{'corpo':>12}{'F/T':>22}{'N.s':>6}{'dv_pk':>7}{'(att)':>7}{'dpitch':>8}{'droll':>7}{'disp':>7}{'fine':>7}{'dyaw':>7}{'assest':>8}{'caduto':>8}")
    for r in rows:
        ft = f"F{tuple(int(x) for x in r['force'])}" if any(r["force"]) else f"T{tuple(int(x) for x in r['torque'])}"
        st = "—" if r["settle_s"] is None else "%.2f" % r["settle_s"]
        print(f"{r['t']:6.1f}{r['h_cmd']:6.2f}{r['body']:>12}{ft:>22}{r['impulse_Ns']:6.0f}{r['dv_peak']:7.3f}{r['dv_expected']:7.3f}"
              f"{r['dpitch_peak']:+8.2f}{r['droll_peak']:+7.2f}{r['disp_max']:7.3f}{r['disp_end']:7.3f}{r['dyaw_end']:+7.1f}"
              f"{st:>8}{('SI' if r['fell'] else 'no'):>8}")
    print(f"\nscritto {run / 'pushes.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
