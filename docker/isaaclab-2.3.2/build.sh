#!/usr/bin/env bash
# Costruisce l'immagine Isaac Sim 5.1.0 + Isaac Lab 2.3.2 + rsl-rl-lib 5.0.1 (vedi README.md).
# Una volta sola: va rifatta solo se cambia il Dockerfile, non quando cambia il codice del repo (che e'
# montato a runtime). Il contesto di build e' questa cartella: Isaac Lab viene clonato dal suo tag.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IL232_IMAGE:-${USER}-isaaclab:2.3.2}"

docker build -t "$IMAGE" "$HERE"
echo
docker image ls "$IMAGE"
echo "immagine: $IMAGE   (il comando di lancio e' $HERE/il232)"
