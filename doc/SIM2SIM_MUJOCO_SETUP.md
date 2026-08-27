# Sim2Sim MuJoCo — Setup & Run su un nuovo PC

Guida per far girare la **simulazione di deploy** (controller C++ `h1_2_ctrl` + `unitree_mujoco`)
dell'H1-2 su una macchina **Ubuntu 22.04 pulita**.

> ⚠️ Questa guida copre **solo** il path di deploy/sim2sim (MuJoCo).
> **NON** serve IsaacSim / IsaacLab / `env_isaaclab` (quelli servono solo per addestrare).
> Il modello addestrato è già versionato nella repo (`policy.onnx` + `deploy.yaml`).

> 🖥️ **Setup a due PC**: il training (IsaacSim/IsaacLab) gira su un PC; la **simulazione sim2sim gira su un ALTRO PC** (questa guida è per quel PC-simulazione). Trasferisci il repo `Squat_Unitree_H1_2` (con `policy.onnx` + `deploy.yaml`) e il fork `unitree_mujoco` sul PC-simulazione.

> Il simulatore ha **due frontend**: `simulate_python` (Python, semplice) e `simulate` (**C++**, timing deterministico, meno jitter di latenza → più vicino al robot reale). Vedi la sezione **10. Simulatore C++** in fondo.

---

## 0. Componenti e layout

Sim2sim = due processi che comunicano via **DDS** (CycloneDDS):

- **`unitree_mujoco`** (simulatore Python) — pubblica lo stato del robot e riceve i comandi motore.
- **`h1_2_ctrl`** (controller C++ di deploy, in `Squat_Unitree_H1_2`) — carica `policy.onnx`, calcola le azioni.

Layout consigliato (tutto sibling in una workspace):

```
~/unitree_ws/
├── Squat_Unitree_H1_2/     # deploy C++ + policy addestrata (repo privata)
├── unitree_mujoco/         # simulatore (tua repo privata, fork modificato per h1_2)
├── unitree_sdk2/           # SDK C++ Unitree (ufficiale)
└── unitree_sdk2_python/    # binding Python DDS (ufficiale)
```

```bash
export WS=~/unitree_ws
mkdir -p $WS && cd $WS
```

---

## 1. Dipendenze di sistema (apt)

```bash
sudo apt update
sudo apt install -y git git-lfs cmake g++ build-essential \
    libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
git lfs install
```

---

## 2. Clona le due repo private

```bash
cd $WS
git clone https://github.com/gaetanotorella/Squat_Unitree_H1_2.git
git clone https://github.com/gaetanotorella/unitree_mujoco.git
# scarica gli asset LFS (mesh, ecc.)
cd Squat_Unitree_H1_2 && git lfs pull && cd ..
cd unitree_mujoco     && git lfs pull && cd ..
```

Il modello addestrato è già dentro `Squat_Unitree_H1_2` (tracciato via .gitignore):
`logs/rsl_rl/unitree_h1_2_velocity/<run>/exported/policy.onnx` + `params/deploy.yaml`.

---

## 3. unitree_sdk2 (SDK C++ + CycloneDDS)

> Fornisce header + libreria + **CycloneDDS** (`ddsc`/`ddscxx`) in `/usr/local`.
> ⚠️ Un'installazione parziale (solo la `.a`, senza header/DDS) fa fallire il build del controller:
> assicurati di fare l'install completo qui sotto.

```bash
cd $WS
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
sudo ldconfig                      # IMPORTANTE: registra /usr/local/lib nel linker
```

Verifica:
```bash
ls -d /usr/local/include/unitree /usr/local/include/ddscxx
ls /usr/local/lib/libddsc.so /usr/local/lib/libddscxx.so /usr/local/lib/libunitree_sdk2.a
```
(`/usr/local/include/iceoryx/...` può mancare: CycloneDDS è buildato senza shared-memory, non serve.)

---

## 4. onnxruntime (runtime ONNX per il controller)

Il controller usa l'onnxruntime **bundled** in `deploy/thirdparty/onnxruntime-linux-x64-1.22.0/`.
⚠️ Gli header sono nella repo ma le **librerie `.so` sono in `.gitignore`** → dopo il clone **mancano**.
Vanno recuperate (una volta sola):

```bash
cd /tmp
wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz
tar xzf onnxruntime-linux-x64-1.22.0.tgz
cp onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so* \
   $WS/Squat_Unitree_H1_2/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib/
```
(In alternativa copia la cartella `lib/` da un PC dove già funziona.)

Crea il symlink non versionato che il CMakeLists si aspetta:
```bash
cd $WS/Squat_Unitree_H1_2/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib
ln -sf libonnxruntime.so.1.22.0 libonnxruntime.so
```

---

## 5. Compila il controller di deploy

```bash
cd $WS/Squat_Unitree_H1_2/deploy/robots/h1_2
mkdir -p build && cd build
cmake .. && make
```
Produce l'eseguibile `h1_2_ctrl`.

---

## 6. Ambiente Python per il simulatore

`unitree_mujoco/simulate_python` gira in Python. Sul PC di sviluppo viveva in `env_isaaclab`;
per un PC **solo-sim** basta un ambiente leggero, Python ≥ 3.10. **Non serve conda** — usa un `venv`:

```bash
sudo apt install -y python3-venv python3-pip
python3 -m venv $WS/venv_sim
source $WS/venv_sim/bin/activate
pip install --upgrade pip
pip install mujoco numpy pygame        # pygame serve per il joystick
```

<details><summary>In alternativa, se preferisci conda</summary>

```bash
conda create -n unitree_sim python=3.11 -y
conda activate unitree_sim
pip install mujoco numpy pygame
```
</details>

Installa i binding Python DDS di Unitree (`unitree_sdk2py`):
```bash
cd $WS
git clone https://github.com/unitreerobotics/unitree_sdk2_python.git
cd unitree_sdk2_python
pip install -e .
```

---

## 7. Configurazione

**Simulatore** — `unitree_mujoco/simulate_python/config.py` (già impostato nella tua repo):
```python
ROBOT        = "h1_2"
DOMAIN_ID    = 0
INTERFACE    = "lo"          # loopback: sim e controller sullo stesso PC
USE_JOYSTICK = 1            # 1 se usi un gamepad (F710); 0 altrimenti
ENABLE_ELASTIC_BAND = True   # molla che sostiene il torso durante lo stand-up
```

**Controller** — `deploy/robots/h1_2/config/config.yaml`: `policy_dir` punta a
`logs/rsl_rl/unitree_h1_2_velocity` e prende automaticamente il run più recente
(che deve contenere `params/deploy.yaml` + `exported/policy.onnx`).

**Gamepad (opzionale ma consigliato)**: Logitech F710 con interruttore su **X** (XInput),
dongle collegato. Verifica: `ls /dev/input/js*` → `/dev/input/js0`.

---

## 8. Avvio della simulazione

**Terminale 1 — simulatore:**
```bash
source $WS/venv_sim/bin/activate     # (o: conda activate unitree_sim)
cd $WS/unitree_mujoco/simulate_python
python unitree_mujoco.py
```

**Terminale 2 — controller:**
```bash
cd $WS/Squat_Unitree_H1_2/deploy/robots/h1_2/build
export LD_LIBRARY_PATH=$WS/Squat_Unitree_H1_2/deploy/thirdparty/onnxruntime-linux-x64-1.22.0/lib:/usr/local/lib:$LD_LIBRARY_PATH
./h1_2_ctrl --network lo
```
Deve stampare `Connected rt/lowstate` → `Connected to robot.` e caricare la `policy_dir` giusta.

**Sequenza FSM (gamepad F710):**
1. **LT + ↑ (d-pad)** → `FixStand` (posa eretta, sostenuto dalla molla)
2. **RB + X** → `Velocity` (parte la policy addestrata)
3. Nella **finestra MuJoCo**: premi **`8`** più volte per allungare la molla e scaricare il peso
   sui piedi (rilascio graduale, dopo aver attivato la policy); **`9`** stacca del tutto la molla.
4. **Levetta sinistra** = avanti/laterale, **levetta destra** = rotazione.

---

## 9. Troubleshooting (problemi già incontrati)

| Sintomo | Causa | Soluzione |
|---|---|---|
| `error while loading shared libraries: libddsc.so.0` | `/usr/local/lib` non nel linker | `sudo ldconfig` (o `LD_LIBRARY_PATH=/usr/local/lib`) |
| `libonnxruntime.so.1.22.0: cannot open ...` | onnxruntime non nel path | esporta `LD_LIBRARY_PATH` col path di `thirdparty/.../lib` (vedi Passo 8) |
| Il controller resta su `Waiting for connection rt/lowstate` | interfaccia DDS diversa tra sim e controller | avvia il controller con **`--network lo`** (il sim usa `INTERFACE="lo"`) |
| Build del controller fallisce (header `unitree/...` mancanti) | SDK2 installato solo parzialmente | rifai il Passo 3 completo + `sudo ldconfig` |
| Non ho un gamepad | il FSM è guidato dal joystick | usa un gamepad USB (anche generico, `joystick_type="xbox"`), oppure emula il `WirelessController` via DDS |
| `selected interface "lo" is not multicast-capable` | normale su loopback | ignora: DDS usa unicast, funziona lo stesso |

---

## Riepilogo dipendenze

| Componente | Fonte | Serve per |
|---|---|---|
| `Squat_Unitree_H1_2` | repo privata | deploy C++ + policy |
| `unitree_mujoco` | repo privata (fork) | simulatore |
| `unitree_sdk2` | ufficiale (build+install) | SDK C++ + CycloneDDS |
| `unitree_sdk2_python` | ufficiale (pip -e) | binding Python per il sim |
| onnxruntime 1.22.0 | release Microsoft | runtime ONNX (`.so` non in repo) |
| mujoco, pygame | pip | simulatore Python + joystick |

**Non serve** su un PC solo-sim: IsaacSim, IsaacLab, `unitree_ros`, l'ambiente `env_isaaclab`.

---

## 10. Simulatore C++ (alternativa a simulate_python)

Il frontend C++ (`unitree_mujoco/simulate`) ha timing **deterministico** (niente GIL/thread-jitter del Python) → latenza più pulita, più vicina al robot reale. Il suo bridge sceglie automaticamente il tipo robot dal numero di attuatori (`m->nu > 20` → G1Bridge, IDL `unitree_hg`), quindi l'**h1_2 (27 giunti) funziona senza modifiche al codice**.

**Build** (sul PC-simulazione; usa la stessa versione MuJoCo del Python, qui 3.7.0):
```bash
sudo apt install -y libglfw3-dev          # yaml-cpp/boost/unitree_sdk2 già presenti
cd ~/unitree_ws
wget https://github.com/google-deepmind/mujoco/releases/download/3.7.0/mujoco-3.7.0-linux-x86_64.tar.gz
tar xzf mujoco-3.7.0-linux-x86_64.tar.gz
ln -s ~/unitree_ws/mujoco-3.7.0 ~/unitree_ws/unitree_mujoco/simulate/mujoco  # CMake si aspetta simulate/mujoco/{include,lib,simulate}
cd ~/unitree_ws/unitree_mujoco/simulate && mkdir -p build && cd build
cmake .. && make                          # -> eseguibile ./unitree_mujoco
```
(Se il release MuJoCo non contiene `simulate/`, i sorgenti del viewer vanno presi dal sorgente MuJoCo — vedi README ufficiale unitree_mujoco.)

**Config**: il C++ legge `simulate/config.yaml` (NON `config.py`). Imposta:
```yaml
robot: "h1_2"
robot_scene: "scene.xml"
domain_id: 0
interface: "lo"
use_joystick: 1
enable_elastic_band: 1
```

**Run**: `./unitree_mujoco` (Term.1) + `./h1_2_ctrl --network lo` (Term.2), stessa sequenza FSM (LT+Up → RB+X → rilascio molla). Tasti elastic band nella finestra: `8`/`7`/`9`.

