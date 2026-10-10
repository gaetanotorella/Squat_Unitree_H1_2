#!/usr/bin/env bash
# Prepara sul server le cartelle che il container usa (vedi README.md). Idempotente.
#
# - Cache di Isaac Sim tutte sotto $IL232_DATA (default ~/docker/isaaclab-2.3.2): separate da
#   ~/docker/isaac-sim, che e' del comando `isaaclab` del lab, cosi' i due non si toccano mai.
# - Permessi: il container gira come utente 1234 (Isaac Sim 5.1 e' rootless). La documentazione di Isaac
#   Sim fa `sudo chown -R 1234:1234`; qui si usano le ACL, che chiunque puo' mettere sui propri file:
#   accesso a 1234, e ACL di default (ereditate da cio' che verra' creato) sia per 1234 sia per te, cosi'
#   i file scritti dal container restano modificabili e cancellabili dall'host senza sudo.
set -euo pipefail

DATA="${IL232_DATA:-$HOME/docker/isaaclab-2.3.2}"
REPO="${IL232_REPO:-$HOME/projects/Squat_Unitree_H1_2}"
ME="$(id -u)"

command -v setfacl >/dev/null || { echo "setfacl non trovato: chiedere a un admin (pacchetto acl)"; exit 1; }
[ -d "$REPO/source/unitree_rl_lab" ] || { echo "repo non trovato in $REPO (o imposta IL232_REPO)"; exit 1; }

mkdir -p "$DATA"/cache/main/ov "$DATA"/cache/main/warp "$DATA"/cache/computecache "$DATA"/config \
         "$DATA"/data/documents "$DATA"/data/Kit "$DATA"/logs "$DATA"/pkg
mkdir -p "$REPO/logs"

grant() {
    setfacl -R -m "u:1234:rwX,u:${ME}:rwX" "$1"
    find "$1" -type d -exec setfacl -d -m "u:1234:rwX,u:${ME}:rwX" {} +
    echo "permessi per 1234 + $(id -un): $1"
}
grant "$DATA"
grant "$REPO"
