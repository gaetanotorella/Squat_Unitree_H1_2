#!/usr/bin/env python3
"""Analisi quantitativa di un run del controller di deploy (sim2sim MuJoCo o robot reale).

Due tipi di run, entrambi sotto sim_logs/:
  - sim_<ts>/  simulazione MuJoCo  -> include ground_truth.csv
  - exp_<ts>/  test sul ROBOT REALE -> NON include ground_truth.csv

CSV letti (ognuno sul PROPRIO asse temporale):
  - obs.csv          : osservazioni in input al controller (ep_t, 50 Hz)   [sempre]
  - commands.csv     : comandi q_target inviati al robot (ep_t)            [sempre]
  - errors.csv       : errore di tracking di giunto q_target-q_misurato    [sempre]
  - ground_truth.csv : stato VERO della base (sim_t, 100 Hz)               [SOLO simulazione]

Sul robot reale posizione assoluta della base, quaternione e velocita' lineare NON sono
misurabili: i plot/metriche che li usano vengono omessi automaticamente. Al loro posto
roll/pitch sono stimati da projected_gravity (IMU), che e' gia' un'osservazione della policy.

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
        runs = sorted(glob.glob(os.path.join(base, "sim_*")) + glob.glob(os.path.join(base, "exp_*")),
                      key=lambda p: os.path.basename(p).split("_", 1)[-1])
        if runs:
            return runs[-1]
    sys.exit("Nessun run trovato in ~/sim_logs o ~/unitree_ws/sim_logs")


MJCF_CANDIDATES = [
    os.environ.get("H1_2_MJCF", ""),
    os.path.expanduser("~/unitree_ws/unitree_mujoco/unitree_robots/h1_2/h1_2_handless.xml"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "../../unitree_mujoco/unitree_robots/h1_2/h1_2_handless.xml"),
]


def estimate_base_height_fk(jpos, roll, pitch):
    """Stima l'altezza della base da CINEMATICA DIRETTA -- usabile sul ROBOT REALE.

    Ingredienti: angoli dei giunti (encoder) + orientazione IMU. Nient'altro. Nessun sensore di
    posizione assoluta, quindi e' riproducibile a bordo esattamente come lo e' per la policy, che
    inferisce la propria altezza dai joint_pos allo stesso modo.

    Metodo: si mette il modello MuJoCo nella posa misurata, con la base all'origine ruotata secondo
    roll/pitch dell'IMU (lo YAW e' irrilevante: non cambia una distanza verticale), si esegue la FK
    e si prende il punto piu' basso dei due piedi. Assumendo che quel punto poggi a terra, l'altezza
    della base e' la sua quota cambiata di segno. Prende il minimo sui DUE piedi -> usa
    automaticamente il piede in appoggio (quello sollevato darebbe una stima piu' bassa).

    Accuratezza misurata contro il ground_truth di sim_2026-09-03 (5486 campioni):
    bias +0.33 cm, std 0.23 cm, corr 0.9995 (fermo: std 0.07-0.12 cm; cammino: 0.26 cm).

    IPOTESI: suolo piatto e almeno un piede in appoggio. Su terreno inclinato/irregolare degrada.
    Ritorna None se mujoco o il modello non sono disponibili.
    """
    try:
        import mujoco
    except ImportError:
        print("  [altezza FK] mujoco non installato -> stima non disponibile")
        return None
    path = next((p for p in MJCF_CANDIDATES if p and os.path.exists(p)), None)
    if path is None:
        print("  [altezza FK] MJCF h1_2 non trovato (impostare $H1_2_MJCF) -> stima non disponibile")
        return None

    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    qadr = [model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n + "_joint")]
            for n in JOINT_NAMES]
    feet = []
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_ankle_roll_link")
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_contype[g] != 0:
                mid = model.geom_dataid[g]
                a, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                feet.append((g, model.mesh_vert[a:a + n].astype(np.float64)))

    out = np.empty(len(jpos))
    for i in range(len(jpos)):
        cr, sr = np.cos(roll[i] / 2), np.sin(roll[i] / 2)
        cp, sp = np.cos(pitch[i] / 2), np.sin(pitch[i] / 2)
        data.qpos[:] = 0
        data.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]  # (w,x,y,z), yaw=0
        data.qpos[qadr] = jpos[i]
        mujoco.mj_kinematics(model, data)
        lowest = min((V @ data.geom_xmat[g].reshape(3, 3)[2] + data.geom_xpos[g][2]).min()
                     for g, V in feet)
        out[i] = -lowest
    return out


def tilt_from_gravity(obs):
    """Roll/pitch [rad] da projected_gravity (IMU) -- disponibile anche sul ROBOT REALE.

    projected_gravity e' gia' un'osservazione della policy, quindi non richiede nulla in piu' a
    bordo. Formule validate contro il ground_truth di una simulazione: RMS 0.06 deg (roll) e
    0.03 deg (pitch). Lo YAW non e' ricavabile dal vettore gravita' (e' non osservabile).
    """
    pg = vecc(obs, ["projected_gravity_0", "projected_gravity_1", "projected_gravity_2"],
              ["obs_3", "obs_4", "obs_5"])
    gx, gy, gz = pg[:, 0], pg[:, 1], pg[:, 2]
    roll = np.arctan2(-gy, -gz)
    pitch = np.arctan2(gx, np.sqrt(gy**2 + gz**2))
    return roll, pitch


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

    # ground_truth.csv lo scrive SOLO il simulatore MuJoCo: posizione assoluta della base,
    # quaternione e velocita' lineare NON sono misurabili sul robot reale. Nei run exp_* (robot
    # reale) il file non esiste -> si mostrano solo i dati realmente ottenibili a bordo.
    gt_path = os.path.join(run, "ground_truth.csv")
    gt = pd.read_csv(gt_path) if os.path.exists(gt_path) else None
    real_robot = gt is None
    print(f"== Modalita': {'ROBOT REALE (nessun ground truth)' if real_robot else 'simulazione MuJoCo'}")

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

    # ---- posa base
    if real_robot:
        # Robot reale: posizione assoluta e yaw NON sono misurabili. Roll/pitch invece si ricavano
        # da projected_gravity (IMU), che e' gia' un'osservazione della policy. Validato contro il
        # ground truth in simulazione: RMS 0.06 deg (roll) e 0.03 deg (pitch).
        roll_i, pitch_i = tilt_from_gravity(obs)
        fig, ax = plt.subplots(1, 1, figsize=(11, 4.5))
        ax.plot(t_obs, np.degrees(roll_i), label="roll (da IMU)")
        ax.plot(t_obs, np.degrees(pitch_i), label="pitch (da IMU)")
        ax.axhline(0, ls="--", lw=0.8, color="gray")
        ax.set_ylabel("inclinazione [deg]"); ax.set_xlabel("t [s]"); ax.grid(alpha=0.3); ax.legend()
        ax.set_title("Inclinazione della base da projected_gravity (IMU)\n"
                     "yaw e posizione XYZ non sono osservabili a bordo")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "05_base_tilt_imu.png"), dpi=110); plt.close(fig)
    else:
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

    # ---- SQUAT: altezza comandata vs altezza vera + comando altezza
    # base_height_command (obs) = target - nominale (0 = in piedi, <0 = squat). L'altezza VERA
    # e' in ground_truth.base_z ma su un altro clock -> la allineo a obs via wall_t condiviso.
    if "base_height_command_0" in obs.columns:
        NOMINAL_H = 0.98
        bh_cmd = obs["base_height_command_0"].to_numpy()
        target_h = NOMINAL_H + bh_cmd
        squatting = bh_cmd < -0.03
        roll_e, pitch_e = tilt_from_gravity(obs)
        h_fk = estimate_base_height_fk(jpos, roll_e, pitch_e)  # None se mujoco/MJCF mancanti
        if real_robot:
            # L'altezza VERA non e' misurabile a bordo, ma e' STIMABILE per cinematica diretta
            # (encoder + IMU): stessa informazione da cui la policy inferisce la propria altezza.
            fig, axs = plt.subplots(2 if h_fk is not None else 1, 1, figsize=(11, 7 if h_fk is not None else 4.5),
                                    sharex=True, squeeze=False)
            a0 = axs[0][0]
            a0.plot(t_obs, target_h, label="altezza comandata (target)", lw=1.6, color="tab:orange")
            if h_fk is not None:
                a0.plot(t_obs, h_fk, label="altezza STIMATA (FK: encoder + IMU)", lw=1.2, color="tab:green")
            a0.axhline(NOMINAL_H, ls="--", lw=0.8, color="gray", label="nominale 0.98")
            a0.set_ylabel("altezza [m]"); a0.grid(alpha=0.3); a0.legend(fontsize=9)
            a0.set_title("Squat: altezza comandata vs stimata (nessuna misura diretta a bordo)")
            if h_fk is not None:
                herr = h_fk - target_h
                a1 = axs[1][0]
                a1.plot(t_obs, herr * 100, lw=1.0, color="tab:red")
                a1.axhline(0, ls="--", lw=0.8, color="gray")
                a1.set_ylabel("errore stimato [cm]"); a1.grid(alpha=0.3)
                if squatting.any():
                    a1.set_title(f"|err| stimato in squat: medio={np.abs(herr[squatting]).mean()*100:.2f}cm  "
                                 f"max={np.abs(herr[squatting]).max()*100:.2f}cm   "
                                 f"(stima FK: bias +0.3cm, std 0.2cm vs ground truth in simulazione)")
            axs[-1][0].set_xlabel("t [s]")
        else:
            base_z_on_obs = np.interp(obs["wall_t"].to_numpy(), gt["wall_t"].to_numpy(), gt["base_z"].to_numpy())
            herr = base_z_on_obs - target_h
            fig, axs2 = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
            axs2[0].plot(t_obs, target_h, label="altezza comandata (target)", lw=1.6, color="tab:orange")
            axs2[0].plot(t_obs, base_z_on_obs, label="altezza vera base_z (gt)", lw=1.2, color="tab:blue")
            if h_fk is not None:
                # controprova: la stessa stima FK usata sul robot reale, per vederne l'accuratezza
                axs2[0].plot(t_obs, h_fk, label="stima FK (come sul robot reale)", lw=1.0,
                             ls="--", color="tab:green", alpha=0.8)
            axs2[0].axhline(NOMINAL_H, ls="--", lw=0.8, color="gray", label="nominale 0.98")
            axs2[0].set_ylabel("altezza [m]"); axs2[0].grid(alpha=0.3); axs2[0].legend(fontsize=9)
            axs2[0].set_title("Squat: altezza comandata vs vera")
            axs2[1].plot(t_obs, herr * 100, lw=1.0, color="tab:red")
            axs2[1].axhline(0, ls="--", lw=0.8, color="gray")
            axs2[1].set_ylabel("errore altezza [cm]"); axs2[1].set_xlabel("t [s]"); axs2[1].grid(alpha=0.3)
            if squatting.any():
                axs2[1].set_title(f"|err| in squat: medio={np.abs(herr[squatting]).mean()*100:.2f}cm  max={np.abs(herr[squatting]).max()*100:.2f}cm")
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "09_squat_height.png"), dpi=110); plt.close(fig)

    # ---- tutte le altre osservazioni
    has_squat = "base_height_command_0" in obs.columns
    nrows = 4 if has_squat else 3
    fig, ax = plt.subplots(nrows, 2, figsize=(13, 3 * nrows))
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
    if has_squat:
        bhc = obs["base_height_command_0"].to_numpy()
        ax[6].plot(t_obs, bhc, color="tab:orange"); ax[6].axhline(0, ls="--", lw=0.8, color="gray")
        ax[6].set_title("base_height_command (target - nominale) [m]")
        # riquadro 8: comando altezza assoluto per leggibilita'
        ax[7].plot(t_obs, 0.98 + bhc, color="tab:orange")
        ax[7].axhline(0.98, ls="--", lw=0.8, color="gray")
        ax[7].set_title("altezza comandata assoluta [m]")
    for a in ax:
        a.grid(alpha=0.3); a.tick_params(labelsize=7); a.set_xlabel("t [s]", fontsize=8)
    fig.suptitle("Altre osservazioni", fontsize=12)
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "08_other_obs.png"), dpi=110); plt.close(fig)

    # ---- metriche a schermo
    if real_robot:
        roll_i, pitch_i = tilt_from_gravity(obs)
        print("\n== Stabilita' (da IMU: projected_gravity)")
        print(f"  |roll| max = {np.max(np.abs(np.degrees(roll_i))):.1f} deg  |pitch| max = {np.max(np.abs(np.degrees(pitch_i))):.1f} deg")
        print(f"  roll std = {np.degrees(roll_i.std()):.2f} deg   pitch std = {np.degrees(pitch_i.std()):.2f} deg")
        print("  (altezza base, deriva xy e yaw non sono misurabili sul robot reale)")
    else:
        print("\n== Stabilita' (ground_truth)")
        print(f"  altezza base media = {gt['base_z'].mean():.3f} m (std {gt['base_z'].std():.4f})")
        print(f"  |roll| max = {np.max(np.abs(np.degrees(roll))):.1f} deg  |pitch| max = {np.max(np.abs(np.degrees(pitch))):.1f} deg")
        print(f"  deriva xy = {np.hypot(gt['base_x'].iloc[-1]-gt['base_x'].iloc[0], gt['base_y'].iloc[-1]-gt['base_y'].iloc[0]):.2f} m")
    print("== Tracking di giunto (q_target - q_misurato)")
    print(f"  GAMBE : |err| medio={np.abs(qerr[:, leg_idx]).mean():.4f}  max={np.abs(qerr[:, leg_idx]).max():.4f} rad")
    print(f"  BRACCIA: |err| medio={np.abs(qerr[:, arm_idx]).mean():.4f}  max={np.abs(qerr[:, arm_idx]).max():.4f} rad")

    # ---- SQUAT + command tracking (obs allineate a gt via wall_t)
    if "base_height_command_0" in obs.columns:
        bh_cmd = obs["base_height_command_0"].to_numpy()
        vcmd = vecc(obs, ["velocity_commands_0", "velocity_commands_1", "velocity_commands_2"],
                    ["obs_6", "obs_7", "obs_8"])
        walking = np.linalg.norm(vcmd, axis=1) > 0.1
        squatting = bh_cmd < -0.03
        standing = (~walking) & (~squatting)
        target_h = 0.98 + bh_cmd
        print("== Squat / altezza")
        print(f"  min altezza comandata = {target_h.min():.3f} m  |  fasi: cammino {100*walking.mean():.0f}% / squat {100*squatting.mean():.0f}% / fermo-eretto {100*standing.mean():.0f}%")
        if real_robot:
            roll_m, pitch_m = tilt_from_gravity(obs)
            h_est = estimate_base_height_fk(jpos, roll_m, pitch_m)
            if h_est is None:
                print("  (altezza effettiva non disponibile: nessuna misura diretta e stima FK non calcolabile)")
            else:
                herr_e = h_est - target_h
                print("  altezza STIMATA per cinematica diretta (encoder + IMU; nessuna misura diretta a bordo)")
                print(f"  [accuratezza della stima, misurata contro il ground truth in simulazione: bias +0.33 cm, std 0.23 cm]")
                for lbl, mk in [("eretto", standing), ("squat", squatting)]:
                    if mk.any():
                        print(f"  {lbl}: |err altezza| medio={np.abs(herr_e[mk]).mean()*100:.2f}cm  max={np.abs(herr_e[mk]).max()*100:.2f}cm")
        else:
            base_z_on_obs = np.interp(obs["wall_t"].to_numpy(), gt["wall_t"].to_numpy(), gt["base_z"].to_numpy())
            herr = base_z_on_obs - target_h
            for lbl, m in [("eretto", standing), ("squat", squatting)]:
                if m.any():
                    print(f"  {lbl}: |err altezza| medio={np.abs(herr[m]).mean()*100:.2f}cm  max={np.abs(herr[m]).max()*100:.2f}cm")
        # sway per fase (base_ang_vel wx,wy)
        bav = vecc(obs, ["base_ang_vel_0", "base_ang_vel_1", "base_ang_vel_2"], ["obs_0", "obs_1", "obs_2"])
        print("== Sway base_ang_vel [rad/s] per fase")
        for lbl, m in [("cammino", walking), ("eretto", standing), ("squat", squatting)]:
            if m.sum() > 20:
                print(f"  {lbl}: wx std={bav[m,0].std():.3f}  wy std={bav[m,1].std():.3f}  |wxy| max={np.hypot(bav[m,0],bav[m,1]).max():.3f}")

    print(f"\n== {len([f for f in os.listdir(outdir) if f.endswith('.png')])} plot salvati in {outdir}")


if __name__ == "__main__":
    main()
