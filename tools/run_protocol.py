#!/usr/bin/env python3
"""Esegue un protocollo di test sim2sim in modo automatico e ripetibile.

Sostituisce la sequenza manuale (avvia controller, avvia MuJoCo, LT+su, RB+X, tasto 9, joystick,
chiudi, analyze_sim.py) con un'unica invocazione. Il joystick e' riprodotto DENTRO il simulatore
contro il tempo di simulazione (vedi unitree_mujoco/simulate/src/scripted_joystick.h), perche' il
controller legge il joystick da lowstate->joystick, che pubblica il simulatore: non c'e' modo di
iniettarlo da fuori via DDS.

    ./run_protocol.py                                  # protocollo standard, policy piu' recente
    ./run_protocol.py --protocol protocols/standard_v1.yaml --label round7
    ./run_protocol.py --policy .../2026-09-08_00-25-08 --no-plots

Alla fine produce, dentro la cartella sim_<timestamp>/ del run:
    protocol.yaml   copia esatta del protocollo usato (il run e' autocontenuto)
    run_meta.json   policy, commit, fattore di real-time, esito dei controlli
    metrics.json    le metriche fisse, per compare_runs.py
    report.md       le stesse metriche in forma leggibile
    plots/          i grafici di analyze_sim.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

WS = Path(__file__).resolve().parents[2]           # ~/unitree_ws
REPO = Path(__file__).resolve().parents[1]         # ~/unitree_ws/Squat_Unitree_H1_2
SIM_DIR = WS / "unitree_mujoco" / "simulate"
SIM_BIN = SIM_DIR / "build" / "unitree_mujoco"
CTRL_BIN = REPO / "deploy" / "robots" / "h1_2_squat" / "build" / "h1_2_ctrl_squat"
LOGS = WS / "sim_logs"
POLICIES = REPO / "logs" / "rsl_rl" / "unitree_h1_2_squat"


def newest(pattern_dir: Path, pattern: str) -> Path | None:
    items = sorted(pattern_dir.glob(pattern))
    return items[-1] if items else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", default=str(Path(__file__).parent / "protocols" / "standard_v1.yaml"))
    ap.add_argument("--policy", default=None, help="cartella del run di training (default: la piu' recente)")
    ap.add_argument("--label", default=None, help="etichetta del run, finisce in run_meta.json")
    ap.add_argument("--no-plots", action="store_true", help="salta analyze_sim.py")
    ap.add_argument("--no-metrics", action="store_true",
                    help="salta report_metrics.py (per i replay: le metriche a segmenti del protocollo non hanno senso)")
    ap.add_argument("--meta", action="append", default=[], metavar="CHIAVE=VALORE",
                    help="campi extra da scrivere in run_meta.json (ripetibile)")
    ap.add_argument("--python", default=str(WS / "venv_sim" / "bin" / "python"))
    ap.add_argument("--keep-config", action="store_true", help="conserva i config temporanei generati")
    args = ap.parse_args()

    protocol = Path(args.protocol).resolve()
    for path, what in [(SIM_BIN, "simulatore"), (CTRL_BIN, "controller"), (protocol, "protocollo")]:
        if not path.exists():
            print(f"ERRORE: {what} non trovato: {path}")
            return 1

    proto = yaml.safe_load(protocol.read_text())
    stop_at = float(proto["stop_at"])

    policy = Path(args.policy).resolve() if args.policy else newest(POLICIES, "*")
    if policy is None or not policy.exists():
        print(f"ERRORE: nessuna policy trovata in {POLICIES}")
        return 1

    # --- fermo di quota: il protocollo e il controller devono vederlo nello stesso punto ---------
    # Il controller ferma R2 a cutoff_height + 1 cm (dal deploy.yaml della policy); il joystick
    # scriptato lo modella con `stop_height`. Se divergono, le quote sotto il fermo arrivano sbagliate
    # senza nessun errore visibile. Un protocollo SENZA `stop_height` tiene A premuto con RT e non
    # dipende dal fermo: va bene con qualunque policy.
    if "stop_height" in proto:
        dep = yaml.safe_load((policy / "params" / "deploy.yaml").read_text())
        cutoff = (dep.get("commands", {}).get("base_velocity") or {}).get("cutoff_height")
        if cutoff is None or abs(float(cutoff) + 0.01 - float(proto["stop_height"])) > 1e-6:
            print(f"ERRORE: stop_height {proto['stop_height']} nel protocollo, ma la policy ha "
                  f"cutoff_height {cutoff} (fermo del controller = cutoff_height + 0.01)")
            return 1

    # --- config USA-E-GETTA, generati qui e passati con --config ------------------------------
    # I config di produzione NON vengono mai toccati. Quello del controller in particolare e' lo
    # STESSO file usato con `--network <eth>` sul robot reale: se un test interrotto lo lasciasse
    # patchato, il giro successivo sul robot vero caricherebbe la policy del test. Per questo sia
    # il simulatore sia il controller accettano `--config <file>` (default invariato: il file
    # accanto al binario). I due percorsi, manuale e automatico, sono cosi' del tutto separati.
    tmp = Path(tempfile.mkdtemp(prefix="h1_2_protocol_"))

    sim_cfg = yaml.safe_load((SIM_DIR / "config.yaml").read_text())
    sim_cfg["use_joystick"] = 1
    sim_cfg["joystick_type"] = "script"
    sim_cfg["joystick_script"] = str(protocol)
    sim_cfg["enable_elastic_band"] = 1   # serve durante FixStand; il rilascio e' nel protocollo
    sim_cfg["disturbance_script"] = ""   # nelle prove automatiche nessuna spinta da tastiera, mai
    sim_cfg_path = tmp / "sim_config.yaml"
    sim_cfg_path.write_text(yaml.safe_dump(sim_cfg, sort_keys=False))

    # `robot_scene` e' relativo e il controller/simulatore lo risolvono contro la cartella del
    # BINARIO, non contro quella del config: resta valido anche da qui.
    ctrl_src = REPO / "deploy" / "robots" / "h1_2_squat" / "config" / "config.yaml"
    ctrl_cfg = yaml.safe_load(ctrl_src.read_text())
    # Policy sotto test fissata esplicitamente: il controller altrimenti prende "la piu' recente",
    # che cambia sotto i piedi fra un test e l'altro e falsa i confronti senza segnalarlo.
    ctrl_cfg["FSM"]["Squat"]["policy_dir"] = str(policy)
    ctrl_cfg_path = tmp / "ctrl_config.yaml"
    ctrl_cfg_path.write_text(yaml.safe_dump(ctrl_cfg, sort_keys=False))

    before = {p.name for p in LOGS.glob("sim_*")}
    procs: list[tuple[str, subprocess.Popen]] = []
    # Uscita dei due processi su FILE, poi copiata nel run. Con una PIPE mai letta, superati i 64 kB
    # del buffer il processo si blocca sulla scrittura: il controller ora stampa la quota durante
    # ogni rampa, e con il suo thread di policy fermo il robot in sim resterebbe senza comandi.
    # Cartella separata da `tmp`, che il finally cancella prima che si conosca la cartella del run.
    out_dir = Path(tempfile.mkdtemp(prefix="h1_2_protocol_out_"))
    ctrl_log = open(out_dir / "controller.log", "w")
    sim_log = open(out_dir / "simulator.log", "w")
    t0 = time.time()
    try:
        print(f"protocollo : {protocol.name}  (stop_at {stop_at:.0f} s)")
        print(f"policy     : {policy.name}")
        print("avvio controller...")
        ctrl = subprocess.Popen([str(CTRL_BIN), "--network", "lo", "--config", str(ctrl_cfg_path)],
                                cwd=CTRL_BIN.parent,
                                stdout=ctrl_log, stderr=subprocess.STDOUT, text=True)
        procs.append(("controller", ctrl))
        time.sleep(2.0)
        if ctrl.poll() is not None:
            ctrl_log.flush()
            print("ERRORE: il controller e' uscito subito:\n" + (out_dir / "controller.log").read_text())
            return 1

        print("avvio simulatore...")
        sim = subprocess.Popen([str(SIM_BIN), "--config", str(sim_cfg_path)], cwd=SIM_DIR,
                               stdout=sim_log, stderr=subprocess.STDOUT, text=True)
        procs.append(("simulatore", sim))
        t_sim = time.time()   # il protocollo parte da qui, non da quando e' partito il controller

        # il protocollo gira sul tempo di simulazione: se MuJoCo rallenta, aspettiamo di piu'
        budget = stop_at * 1.6 + 30.0
        print(f"in esecuzione (fino a {budget:.0f} s di attesa)...")
        deadline = time.time() + budget
        while time.time() < deadline:
            if sim.poll() is not None or ctrl.poll() is not None:
                print("ATTENZIONE: un processo e' terminato da solo")
                break
            time.sleep(1.0)
            if time.time() - t_sim > stop_at + 8.0:
                break
        wall = time.time() - t0
    finally:
        for name, p in reversed(procs):
            if p.poll() is None:
                p.send_signal(signal.SIGINT)
        time.sleep(1.5)
        for name, p in reversed(procs):
            if p.poll() is None:
                p.terminate()
        time.sleep(1.0)
        for name, p in reversed(procs):
            if p.poll() is None:
                p.kill()
        ctrl_log.close()
        sim_log.close()
        if not args.keep_config:
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            print(f"config temporanei conservati in {tmp}")

    # --- individua la cartella del run -------------------------------------------------------
    after = {p.name for p in LOGS.glob("sim_*")}
    created = sorted(after - before)
    if not created:
        print("ERRORE: nessuna cartella sim_* creata; il simulatore non ha loggato nulla")
        print(f"uscita di controller e simulatore in {out_dir}")
        return 1
    run_dir = LOGS / created[-1]
    print(f"run        : {run_dir}")

    shutil.copy(protocol, run_dir / "protocol.yaml")
    for f in ("controller.log", "simulator.log"):
        shutil.move(str(out_dir / f), str(run_dir / f))
    shutil.rmtree(out_dir, ignore_errors=True)

    def git(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=REPO, text=True).strip()
        except Exception:
            return None

    meta = {
        "protocol": proto.get("name", protocol.stem),
        "protocol_file": protocol.name,
        "policy": policy.name,
        "policy_path": str(policy),
        "label": args.label,
        "wall_seconds": round(wall, 1),
        "stop_at": stop_at,
        "git_commit": git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
    }
    for kv in args.meta:
        k, _, v = kv.partition("=")
        meta[k] = v
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    # --- metriche + report -------------------------------------------------------------------
    if not args.no_metrics:
        rc = subprocess.call([args.python, str(Path(__file__).parent / "report_metrics.py"), str(run_dir)])
        if rc != 0:
            print("ATTENZIONE: report_metrics.py ha fallito")

    if not args.no_plots:
        subprocess.call([args.python, str(Path(__file__).parent / "analyze_sim.py"), str(run_dir)])

    print(f"\nfatto. report: {run_dir / 'report.md'}")
    print(f"RUN_DIR={run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
