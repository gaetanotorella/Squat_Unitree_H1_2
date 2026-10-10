# Controller in Docker sul PC di bordo del robot

Immagine minimale (**169 MB**) con il solo `h1_2_ctrl_squat`, le librerie che gli servono e la
config. **La policy non è dentro l'immagine**: è un volume, si cambia senza ricostruire. Stessa
immagine per sim2sim (`--network lo`) e robot reale (`--network eth0`).

Validata il 2026-09-16 **sul robot vero** (H1_2, PC2: i7-1255U x86_64, Ubuntu 20.04.6, Docker 24):
connessione DDS, FSM Passive → FixStand → Passive, log sul volume. E su questo PC in sim2sim
(container + `unitree_mujoco` sull'host, protocollo `smoke`, 37 s di policy).

## Il robot

| | |
|---|---|
| SSH | `ssh unitree@192.168.123.162` |
| PC di bordo | `unitree-h1-pc2`, i7-1255U → **x86_64** (niente cross-build) |
| OS | Ubuntu 20.04.6, gcc 9 |
| Docker | 24.0.7, utente `unitree` già nel gruppo `docker` |
| interfaccia robot | **`eth0`** (192.168.123.162/24) → `--network eth0` |
| immagini che DEVONO restare | `unitree_sdk2:latest` (non nostra) |

**Prima di ogni lancio il robot va messo in DEBUG MODE.** Altrimenti il controller interno di PC1
pubblica `lowcmd` e il nostro stampa `[critical] The other process is using the lowcmd channel`
— e poi va avanti lo stesso (il codice avvisa, non esce): due controller sui motori. Se vedi quel
messaggio, `docker stop h1_2_ctrl`, metti il debug mode, rilancia.

## Uso quotidiano — sul ROBOT

```bash
cd ~/unitree_ws
./run.sh 2026-09-13_21-27-39 eth0     # avvio (staccato: sopravvive alla chiusura dell'SSH)
docker logs -f h1_2_ctrl              # log, dall'inizio, riapribile quante volte vuoi
docker stop h1_2_ctrl                 # chiusura (prima: robot in Passive con L2+B)
docker ps                             # gira?  (docker ps -a: anche quelli usciti)
```

**`Ctrl-C` dentro `docker logs -f` chiude solo la finestra dei log, NON il controller.** È voluto:
il container gira con `-d` proprio perché la caduta dell'SSH non uccida il processo che comanda i
motori (quello che succederebbe con il binario lanciato a mano in una shell).

Sequenza attesa nel log:

```
Connected to robot.
Policy directory: .../2026-09-13_21-27-39
FSM: Start Passive
```

poi dal radiocomando originale (arriva dal firmware via `LowState`, identico a prima):
`L2 + su` → FixStand (rampa 3 s) · `R1 + X` → Squat, compare `[deploy_logger] ... exp_<ts>` ·
`R2` tenuto scende e si ferma da solo a 0.71 m (quota minima di camminata, `cutoff_height` + 1 cm),
`A + R2` tenuti scende oltre fino a 0.60, `B` sale · `L2 + B` → Passive. La quota comandata compare nel
log (`Squat: height ...`), con un avviso quando si scende sotto 0.70 e i comandi di velocità si spengono.

Un `FixStand → Passive` non voluto ha due sole cause: `L2 + B` premuto, oppure `lowstate` fermo
per > 1000 ms (fallback di sicurezza registrato in `FSMState`). Il log oggi non le distingue.

## Recupero dei log — dal TUO PC

I log finiscono in `~/unitree_ws/sim_logs/exp_<ts>/` **sul robot** (volume montato).

```bash
rsync -av unitree@192.168.123.162:~/unitree_ws/sim_logs/exp_* ~/unitree_ws/sim_logs/
```

Poi `analyze_sim.py sim_logs/exp_<ts>` come sempre.

## Cosa serve ricostruire, e cosa no

| cosa modifichi | ricostruire l'immagine? | cosa fare |
|---|---|---|
| codice C++ (`deploy/include/`, `robots/h1_2_squat/src`, `main.cpp`) | **sì** | build + save/load |
| `robots/h1_2_squat/config/config.yaml` (FSM, kp/kd, posa FixStand) | **sì**, è copiato nell'immagine | idem |
| policy (`exported/policy.onnx`, `params/deploy.yaml`) | **no**, è un volume | rsync + `./run.sh <run> eth0` |

### Ciclo per una modifica al codice o al config

```bash
# dal tuo PC
cd ~/unitree_ws/Squat_Unitree_H1_2/deploy/docker
./build.sh                                                      # ~40 s con la cache
docker save h1_2_ctrl_squat:x86_64 | ssh unitree@192.168.123.162 docker load
# sul robot
docker stop h1_2_ctrl 2>/dev/null; ./run.sh 2026-09-13_21-27-39 eth0
```

`docker load` sostituisce l'immagine con lo stesso tag; la vecchia resta come `<none>` e si toglie
con `docker image prune -f` (non tocca `unitree_sdk2:latest`, che è taggata). Il trasferimento è
di tutti i 169 MB anche per una riga: ~10 s sulla 192.168.123.

### Nuova policy

```bash
# dal tuo PC
RUN=<run>
ssh unitree@192.168.123.162 "mkdir -p ~/unitree_ws/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN"
rsync -av Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN/exported \
          Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN/params \
          unitree@192.168.123.162:~/unitree_ws/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN/
# sul robot
./run.sh $RUN eth0
```

Il controller risolve la policy come `../../../logs/rsl_rl/unitree_h1_2_squat/<run>` relativo a
`robots/h1_2_squat/` e legge solo `exported/policy.onnx` + `params/deploy.yaml`: per questo il
layout sul robot rispecchia quello del PC di sviluppo.

Per provare un `config.yaml` diverso senza ricostruire: aggiungi a `run.sh` un
`-v /percorso/config.yaml:/root/unitree_ws/Squat_Unitree_H1_2/deploy/robots/h1_2_squat/config/config.yaml:ro`.

## Come è fatta

- **`Dockerfile`** multi-stage: lo stadio `builder` (Ubuntu 20.04, i `-dev`) compila `unitree_sdk2`
  (il nostro fork, con `dds_wrapper` e CycloneDDS prebuilt per x86_64 e aarch64) e il controller;
  lo stadio finale copia solo binario, `libddsc`/`libddscxx`/`libonnxruntime`, config.
  Lo smoke test `--help` in fase di build garantisce che il binario trovi le librerie.
- **`build.sh [linux/arm64]`**: copia l'SDK nel contesto (`_ctx/`, ignorato da git), verifica che il
  tarball onnxruntime per l'architettura ci sia in `deploy/thirdparty/`, costruisce
  `h1_2_ctrl_squat:<arch>`. Il `.dockerignore` alla radice del repo (quello upstream, con le nostre
  righe in coda) tiene fuori `logs/` e le build native: contesto ~110 MB.
- **`run.sh <run> <iface>`**: `--network host` (**obbligatorio**: CycloneDDS scopre il robot via
  multicast su 192.168.123.x, la rete bridge non la vede), `-d`, policy montata in sola lettura,
  `sim_logs` montata in scrittura, tag scelto da `uname -m`.

## Cosa ha scoperto il collaudo (e che sul robot ti avrebbe fermato)

- **gcc 9 compila in C++14 di default**; `std::clamp` e `std::filesystem` sono C++17. Il
  `CMakeLists` del controller ora fissa `CMAKE_CXX_STANDARD 17` e linka `stdc++fs`. Vale anche per
  la compilazione nativa sul robot.
- Il `CMakeLists` sceglie il tarball onnxruntime da `CMAKE_SYSTEM_PROCESSOR`: lo stesso albero
  compila su x86_64 e aarch64.
- `docker stop` termina il container con **exit 137** (SIGKILL dopo i 10 s di grazia): il
  controller non gestisce SIGTERM. Per questo la chiusura corretta è **prima** `L2 + B` (Passive)
  e **poi** `docker stop` — mai il contrario con il robot in RL.

## Se serve il PC di bordo aarch64 (non è il nostro caso)

`./build.sh linux/arm64` via buildx + QEMU (qualche minuto), con
`onnxruntime-linux-aarch64-1.22.0` scompattato in `deploy/thirdparty/`
(github.com/microsoft/onnxruntime/releases). Oppure `./build.sh` direttamente sul robot.
