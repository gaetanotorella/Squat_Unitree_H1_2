# Training sul server del lab con lo stack della tesi

Immagine Docker con **Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + rsl-rl-lib 5.0.1**, lo stack con cui sono stati
addestrati tutti i run della tesi (codice al tag `isaaclab-2.3.2`). Sul server Isaac è disponibile solo come
Isaac Lab 3.0.0-rc1, e il porting a 3.0 ha cambiamenti silenziosi (il più grave: `deploy.yaml` con kp/kd = 0)
e renderebbe i run nuovi non confrontabili con i vecchi. Immagine personale **concordata con gli admin, da
cancellare a fine tesi** (sezione [Pulizia](#pulizia)).

Regole rispettate: nessun sudo; container cancellato a ogni uscita (`--rm`); cache in una cartella propria,
separata da `~/docker/isaac-sim` del comando `isaaclab` del lab (i due possono girare insieme); il container
gira come utente 1234 come previsto da Isaac Sim 5.1, e i file che scrive restano tuoi grazie alle ACL.

| file | cosa fa |
|---|---|
| `Dockerfile` | l'immagine (ricalca `docker/Dockerfile.base` di Isaac Lab v2.3.2) |
| `build.sh` | la costruisce, una volta sola |
| `setup_host.sh` | crea le cartelle delle cache e mette le ACL (idempotente) |
| `il232` | il comando di lancio, l'equivalente di `isaaclab` |

## Prima volta, sul server

```bash
# 1. il repo (privato): con una deploy key in sola lettura, oppure rsync dal PC
mkdir -p ~/projects && cd ~/projects
git clone git@github.com:gaetanotorella/Squat_Unitree_H1_2.git

# 2. di unitree_ros serve solo l'URDF dell'H1_2: clone parziale
git clone --filter=blob:none --sparse https://github.com/unitreerobotics/unitree_ros.git
git -C unitree_ros sparse-checkout set robots/h1_2_description

# 3. cartelle e permessi, poi l'immagine (~30-60 min la prima volta, ~40 GB)
D=~/projects/Squat_Unitree_H1_2/docker/isaaclab-2.3.2
$D/setup_host.sh
$D/build.sh
# il232 nel PATH (un alias non basterebbe: tmux lancia i comandi senza leggere ~/.bashrc)
mkdir -p ~/.local/bin && ln -sf $D/il232 ~/.local/bin/il232   # poi rientrare in SSH se ~/.local/bin non era nel PATH
```

Il primo avvio di Isaac Sim compila le shader (10 minuti o più); dal secondo sono in cache.

## Verifica dello stack (prima di fidarsi)

```bash
# versioni
il232 bash -c 'cat /isaac-sim/VERSION; cat /workspace/isaaclab/VERSION; /isaac-sim/python.sh -m pip list 2>/dev/null | grep -iE "^(rsl-rl-lib|torch) "'

# training di prova: deve partire, scrivere in logs/ e i file devono essere cancellabili dall'host
il232 python scripts/rsl_rl/train.py --task Unitree-H1_2-Squat --headless --num_envs 64 --max_iterations 3

# il deploy.yaml del run di prova deve essere IDENTICO a quello della policy finale
# (copiare prima la policy finale dal PC, vedi sotto)
diff logs/rsl_rl/unitree_h1_2_squat/<run di prova>/params/deploy.yaml \
     logs/rsl_rl/unitree_h1_2_squat/2026-09-13_21-27-39/params/deploy.yaml
```

Il driver del server (595) è più recente di quello certificato per Isaac Sim 5.1 (580): se il training di
prova o un render di `media.py` danno errori di GPU, il problema è lì.

## Uso

```bash
cd ~/projects/Squat_Unitree_H1_2 && git pull

# training, in tmux (sopravvive alla chiusura dell'SSH)
nvidia-smi                                   # la GPU è condivisa: guardare prima chi la usa
tmux new -s train 'il232 python scripts/rsl_rl/train.py --task Unitree-H1_2-Squat --headless --max_iterations 5000'
#   staccarsi: Ctrl+B poi D      riattaccarsi: tmux attach -t train

# TensorBoard (dal PC: ssh -L 6006:localhost:6006 PrismaLabWS, poi http://localhost:6006)
il232 tensorboard 6006

# media della tesi (non riscrive i file di deploy, a differenza di play.py)
il232 python scripts/rsl_rl/media.py --preset S4 --headless
```

Il codice è montato, non copiato: dopo un `git pull` non serve ricostruire nulla. L'immagine va ricostruita
solo se cambia il `Dockerfile`.

**`play.py` riscrive `exported/policy.onnx` e `params/deploy.yaml` del run** (i file che legge il robot):
usarlo solo sui run nuovi, mai su quelli già validati.

Run fra PC e server (dal PC):

```bash
S=PrismaLabWS:projects/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat
L=~/unitree_ws/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat
rsync -av $L/2026-09-13_21-27-39 $S/          # PC -> server (es. per il confronto del deploy.yaml)
rsync -av $S/<run> $L/                        # server -> PC (per sim2sim e robot)
```

## Pulizia

A fine tesi, sul server:

```bash
docker ps --filter label=owner=$USER                      # nessun container attivo
docker rmi $USER-isaaclab:2.3.2
docker ps -a --filter ancestor=nvcr.io/nvidia/isaac-sim:5.1.0   # se nessun altro la usa:
docker rmi nvcr.io/nvidia/isaac-sim:5.1.0
rm -rf ~/docker/isaaclab-2.3.2
setfacl -R -b ~/projects/Squat_Unitree_H1_2               # toglie le ACL (o si cancella la cartella)
```

La cache di build di Docker è condivisa fra gli utenti: per liberarla (`docker builder prune`) chiedere
all'admin.
