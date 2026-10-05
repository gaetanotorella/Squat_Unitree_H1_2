#!/usr/bin/env python3
"""Riproduce in MuJoCo un esperimento fatto sul robot reale, con gli STESSI comandi.

    ./replay_experiment.py ~/unitree_ws/sim_logs/FINALE/exp_2026-09-22_14-45-16
    ./replay_experiment.py ~/unitree_ws/sim_logs/FINALE/exp_*          # tutti

Per ogni esperimento produce una cartella `sim_logs/sim_<ts>_replay_<exp>/` con gli stessi file di
ogni sim2sim (obs.csv, commands.csv, errors.csv, ground_truth.csv, plots/) piu':
    replay.yaml          la timeline del joystick ricostruita dall'esperimento
    source_run_info.md   il run_info.md dell'esperimento reale
    replay_fidelity.md   quanto i comandi visti dalla policy in sim coincidono con quelli del reale

METODO. Non si forza nulla dentro il controller: in simulazione gira lo STESSO controller del robot
(stesso codice, stessa policy, stessi kp/kd, stesso deploy.yaml) e cambia solo la sorgente del
joystick, che invece del radiocomando e' il joystick scriptato del simulatore. La timeline del
joystick si ricostruisce da obs.csv dell'esperimento, che registra ESATTAMENTE i comandi visti dalla
policy a ogni passo:

  - velocita' (velocity_commands): nel controller passano da un filtro EMA (x <- 0.97x + 0.03u a
    1 kHz) prima di essere registrate. Rimandarle tali e quali le filtrerebbe due volte (~33 ms di
    ritardo); qui il filtro viene PRE-COMPENSATO: per ogni passo k si manda u_k tale che, dopo i 20
    aggiornamenti del filtro fra un passo e l'altro, l'uscita sia esattamente il valore registrato:
        u_k = (y_k - 0.97^20 * y_{k-1}) / (1 - 0.97^20)
    e il filtro lato simulatore viene disattivato (stick_smoothing: false).
  - quota (base_height_command): il controller la integra di 1 mm per passo mentre RT (scende) o B
    (sale) e' premuto. La timeline riporta la quota registrata passo per passo, e il joystick
    scriptato tiene RT/B esattamente per quei passi. RT ha a sua volta un filtro EMA lato controller
    (~23 ms per superare la soglia), B no: le voci di discesa sono anticipate di conseguenza.
  - tempi: il controller scandisce i passi sull'orologio di PARETE, la simulazione gira a rtf
    ~0.9995. Il replay va percio' sull'orologio di parete a partire dall'ingresso in RL
    (clock: wall_after_rl), altrimenti su 270 s i comandi slitterebbero di ~7 passi.
  - avvio: FixStand, poi la corda elastica viene rilasciata PRIMA di entrare in RL (sul reale la
    fune era lasca: la policy parte con il robot interamente sui piedi), poi RL.

LIMITE. Si riproducono i COMANDI dell'operatore, non le perturbazioni esterne: spinte a mano,
contatti con la fune o con l'operatore non sono misurati e non compaiono nel replay.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
WS = HERE.parents[1]
POLICIES = WS / "Squat_Unitree_H1_2" / "logs" / "rsl_rl" / "unitree_h1_2_squat"

# --- costanti del controller e del joystick scriptato (devono combaciare) ---------------------------
STEP = 0.02                 # step_dt della policy
EMA_A = 0.97                # Axis::smooth = 0.03 -> x <- 0.97 x + 0.03 u, a 1 kHz (ciclo FSM)
EMA_N = 20                  # aggiornamenti del filtro fra due passi di policy
RT_DELAY = 0.023            # passi del filtro EMA perche' RT superi 0.5 lato controller: ln(.5)/ln(.97) ms
MARGIN = 0.010              # margine di temporizzazione delle rampe (meta' passo)
CTRL_DELAY = 0.003          # ingresso in RL e primo passo: qualche ms dopo il fronte di X
MODIFIER_LEAD = 0.15        # = kModifierLead in scripted_joystick.h (fronte di X dopo RB)

T_FIXSTAND = 2.0            # LT + su (rampa FixStand di 3 s)
T_RL = 8.0                  # RB + X
# Rilascio della corda 0.3 s PRIMA dell'ingresso in RL. Sul reale la fune era lasca: la policy parte
# con il robot interamente sui piedi. In sim pero' la posa FixStand senza corda NON e' stabile:
# misurato rilasciando a 6.5 s, il robot si assesta di 9 mm nei primi 0.5 s (il carico che arriva sui
# piedi, come sul reale) e poi crolla (0.74 m a 8.0 s, a terra a 8.2 s). 0.3 s prima dell'ingresso in
# RL il robot e' gia' interamente caricato e ancora fermo in piedi.
BAND_LEAD = 0.30
T_BAND = T_RL + MODIFIER_LEAD - BAND_LEAD


def build_timeline(exp: Path, nominal: float, rate: float) -> dict:
    obs = pd.read_csv(exp / "obs.csv")
    y = obs[["velocity_commands_0", "velocity_commands_1", "velocity_commands_2"]].to_numpy(float)
    h = obs["base_height_command_0"].to_numpy(float) + nominal
    n = len(obs)

    # pre-compensazione del filtro EMA del controller
    a = EMA_A ** EMA_N
    yprev = np.vstack([np.zeros((1, 3)), y[:-1]])
    u = (y - a * yprev) / (1.0 - a)
    u[np.abs(u) < 1e-6] = 0.0

    # quota: passo per passo, il segno della variazione decide RT (scende) o B (sale)
    dh = np.diff(np.concatenate([[nominal], h]))
    anchor = T_RL + MODIFIER_LEAD + CTRL_DELAY        # istante del passo k = 0
    tl = [{"t": T_FIXSTAND, "do": "fixstand"}, {"t": T_RL, "do": "rl"},
          {"t": round(T_RL + 0.5, 4), "vx": 0.0, "vy": 0.0, "wz": 0.0, "height": nominal}]

    last = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    events = []   # (t, dict)
    for k in range(n):
        s_k = anchor + STEP * k
        # levette: il valore pre-compensato deve agire nei 20 ms che precedono il passo k
        cur = {"vx": float(u[k, 0]), "vy": float(u[k, 1]), "wz": float(u[k, 2])}
        if any(abs(cur[c] - last[c]) > 1e-6 for c in cur):
            events.append((s_k - STEP, {c: round(v, 6) for c, v in cur.items()}))
            last = cur
        # quota: la voce porta il target del passo k; anticipo diverso per RT (filtro) e B
        if abs(dh[k]) > 1e-7:
            lead = (RT_DELAY + MARGIN) if dh[k] < 0 else MARGIN
            events.append((s_k - lead, {"height": round(float(h[k]), 5)}))

    # la timeline deve essere ordinata; voci con lo stesso istante si fondono
    events.sort(key=lambda e: e[0])
    merged = []
    for t, d in events:
        t = round(t, 4)
        if merged and abs(merged[-1]["t"] - t) < 1e-9:
            merged[-1].update(d)
        else:
            merged.append({"t": t, **d})
    tl += merged

    stop_at = anchor + STEP * n + 2.0
    return {
        "name": f"replay_{exp.name}",
        "replay_of": str(exp),
        "nominal_height": nominal,
        "deepest_height": 0.60,
        "descend_rate": rate,
        "rise_rate": rate,
        "band_release_at": T_BAND,
        "overhold": 0.2,
        "stick_smoothing": False,
        "clock": "wall_after_rl",
        "stop_at": round(stop_at, 2),
        "timeline": tl,
    }


def fidelity(real: Path, sim: Path, nominal: float) -> str:
    r = pd.read_csv(real / "obs.csv"); s = pd.read_csv(sim / "obs.csv")
    cols = ["velocity_commands_0", "velocity_commands_1", "velocity_commands_2"]
    yr = r[cols].to_numpy(float); ys = s[cols].to_numpy(float)
    hr = r.base_height_command_0.to_numpy(float) + nominal; hs = s.base_height_command_0.to_numpy(float) + nominal
    n = min(len(r), len(s))
    if n < 50:
        return (f"# Fedelta' del replay — {real.name}\n\n**REPLAY FALLITO**: la simulazione ha registrato "
                f"solo {len(s)} passi di policy contro {len(r)} del reale (robot caduto o controller "
                f"uscito da RL). Controllare ground_truth.csv.\n")

    def err_at(lag):
        a, b = (slice(0, n - lag), slice(lag, n)) if lag >= 0 else (slice(-lag, n), slice(0, n + lag))
        return yr[a] - ys[b], hr[a] - hs[b]

    lags = range(-5, 6)
    cost = {L: np.sqrt(np.mean(err_at(L)[0] ** 2)) + np.sqrt(np.mean(err_at(L)[1] ** 2)) for L in lags}
    # con comandi costanti (es. solo stazione eretta) ogni sfasamento da' lo stesso errore: non misurabile
    if max(cost.values()) - min(cost.values()) < 1e-9:
        lag_txt = "non misurabile (comandi costanti)"
    else:
        lag_txt = f"**{min(cost, key=cost.get)} passi** (0 = allineato)"
    ev, eh = err_at(0)
    active = np.linalg.norm(yr[:n], axis=1) > 0.02
    L = []
    L.append(f"# Fedelta' del replay — {real.name}\n")
    L.append(f"- passi reali {len(r)}, passi in sim {len(s)} (confronto sui primi {n})")
    L.append(f"- sfasamento che minimizza l'errore: {lag_txt}\n")
    L.append("| comando | RMS errore | max errore | RMS durante i comandi | ampiezza reale |")
    L.append("|---|---|---|---|---|")
    for i, c in enumerate(["vx [m/s]", "vy [m/s]", "wz [rad/s]"]):
        e = ev[:, i]
        ea = e[active] if active.any() else np.array([0.0])
        L.append(f"| {c} | {np.sqrt(np.mean(e**2)):.4f} | {np.abs(e).max():.4f} | {np.sqrt(np.mean(ea**2)):.4f} | {np.abs(yr[:n, i]).max():.3f} |")
    L.append(f"| quota [m] | {np.sqrt(np.mean(eh**2)):.4f} | {np.abs(eh).max():.4f} | — | {hr.min():.3f}–{hr.max():.3f} |")
    # stato iniziale all'ingresso in RL: quanto la sim parte dalla stessa condizione del reale
    pg = ["projected_gravity_0", "projected_gravity_1", "projected_gravity_2"]
    def tilt(df):
        g = df[pg].to_numpy(float)[:25]
        return np.degrees(np.arctan2(-g[:, 0], -g[:, 2])).mean(), np.degrees(np.arctan2(g[:, 1], -g[:, 2])).mean()
    jp = [c for c in r.columns if c.endswith("_pos") and "hm" not in c]
    dj = np.abs(r[jp].to_numpy(float)[0] - s[jp].to_numpy(float)[0])
    (pr, rr), (ps, rs) = tilt(r), tilt(s)
    L.append("")
    L.append("**Stato all'ingresso in RL** (primi 0.5 s): "
             f"pitch reale {pr:+.2f}° / sim {ps:+.2f}°, roll reale {rr:+.2f}° / sim {rs:+.2f}°; "
             f"posizioni dei giunti al primo passo: scarto medio {np.degrees(dj.mean()):.2f}°, massimo {np.degrees(dj.max()):.2f}° "
             f"({jp[int(dj.argmax())].replace('_pos', '')}).")
    L.append("")
    L.append("Il confronto e' passo per passo (ep_t identico nei due log): ogni riga di obs.csv in sim")
    L.append("e' il comando che la policy ha visto nello stesso passo dall'ingresso in RL del reale.")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", help="cartelle exp_<ts> del robot reale")
    ap.add_argument("--policy", default=None, help="run di training (default: quello scritto in run_info.md)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--python", default=str(WS / "venv_sim" / "bin" / "python"))
    args = ap.parse_args()

    rc_all = 0
    for e in args.experiments:
        exp = Path(e).resolve()
        if not (exp / "obs.csv").exists():
            print(f"SALTO {exp}: manca obs.csv"); rc_all = 1; continue

        info = (exp / "run_info.md").read_text() if (exp / "run_info.md").exists() else ""
        m = re.search(r"\*\*policy_name\*\*:\s*(\S+)", info)
        policy = Path(args.policy).resolve() if args.policy else (POLICIES / m.group(1) if m else None)
        if policy is None or not (policy / "exported" / "policy.onnx").exists():
            print(f"SALTO {exp.name}: policy non trovata ({policy})"); rc_all = 1; continue

        dep = yaml.safe_load((policy / "params" / "deploy.yaml").read_text())
        nominal = float(dep["observations"]["base_height_command"]["params"]["nominal_height"])
        # rampa del controller: misurata dall'esperimento stesso (1 mm/passo -> 0.05 m/s)
        h = pd.read_csv(exp / "obs.csv").base_height_command_0.to_numpy(float)
        steps = np.abs(np.diff(h)); steps = steps[steps > 1e-7]
        rate = float(np.median(steps) / STEP) if len(steps) else 0.05
        rate = round(rate, 4)

        proto = build_timeline(exp, nominal, rate)
        tmp = Path("/tmp") / f"replay_{exp.name}.yaml"
        tmp.write_text(yaml.safe_dump(proto, sort_keys=False, default_flow_style=None, width=200))
        print(f"\n=== {exp.name}: {len(proto['timeline'])} voci, rampa {rate} m/s, durata {proto['stop_at']:.0f} s, policy {policy.name}")

        cmd = [args.python, str(HERE / "run_protocol.py"), "--protocol", str(tmp), "--policy", str(policy),
               "--label", f"replay_{exp.name}", "--no-metrics", "--meta", f"replay_of={exp}"]
        if args.no_plots:
            cmd.append("--no-plots")
        out = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write("\n".join(l for l in out.stdout.splitlines() if not l.startswith("RUN_DIR=")) + "\n")
        mm = re.search(r"^RUN_DIR=(.+)$", out.stdout, re.M)
        if out.returncode != 0 or not mm:
            print(out.stderr[-2000:]); print(f"ERRORE nel replay di {exp.name}"); rc_all = 1; continue
        run = Path(mm.group(1))
        dest = run.with_name(f"{run.name}_replay_{exp.name}")
        run.rename(dest)
        shutil.copy(tmp, dest / "replay.yaml")
        if (exp / "run_info.md").exists():
            shutil.copy(exp / "run_info.md", dest / "source_run_info.md")
        rep = fidelity(exp, dest, nominal)
        (dest / "replay_fidelity.md").write_text(rep)
        print(rep)
        print(f"cartella: {dest}")
    return rc_all


if __name__ == "__main__":
    sys.exit(main())
