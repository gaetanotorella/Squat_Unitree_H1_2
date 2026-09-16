#!/usr/bin/env bash
# Costruisce l'immagine del controller.
#   ./build.sh                     -> architettura di questa macchina
#   ./build.sh linux/arm64         -> per un PC di bordo aarch64 (Jetson), via buildx+QEMU
# Tag: h1_2_ctrl_squat:<arch>. La policy non e' nell'immagine (vedi run.sh).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"                 # Squat_Unitree_H1_2/
SDK="$(cd "$ROOT/.." && pwd)/unitree_sdk2"        # ~/unitree_ws/unitree_sdk2
PLATFORM="${1:-}"

[ -d "$SDK" ] || { echo "unitree_sdk2 non trovato in $SDK"; exit 1; }

# il contesto di build deve contenere l'SDK: lo si porta dentro con rsync (senza build/)
mkdir -p "$HERE/_ctx"
rsync -a --delete --exclude build/ --exclude .git/ "$SDK/" "$HERE/_ctx/unitree_sdk2/"

# il tarball onnxruntime per l'architettura richiesta deve gia' essere in deploy/thirdparty/
case "$PLATFORM" in
  linux/arm64|linux/aarch64) NEED=onnxruntime-linux-aarch64-1.22.0; TAG=aarch64 ;;
  ""|linux/amd64)            NEED=onnxruntime-linux-x64-1.22.0;     TAG=x86_64 ;;
  *) echo "piattaforma non gestita: $PLATFORM"; exit 1 ;;
esac
[ -f "$ROOT/deploy/thirdparty/$NEED/lib/libonnxruntime.so" ] || {
  echo "manca $ROOT/deploy/thirdparty/$NEED -- scarica il tarball 1.22.0 da github.com/microsoft/onnxruntime/releases"; exit 1; }

cd "$ROOT"
if [ -n "$PLATFORM" ]; then
  docker buildx build --platform "$PLATFORM" --load -f deploy/docker/Dockerfile -t "h1_2_ctrl_squat:$TAG" .
else
  docker build -f deploy/docker/Dockerfile -t "h1_2_ctrl_squat:$TAG" .
fi
echo
echo "immagine: h1_2_ctrl_squat:$TAG"
echo "per portarla sul robot senza registry:"
echo "  docker save h1_2_ctrl_squat:$TAG | ssh <robot> docker load"
