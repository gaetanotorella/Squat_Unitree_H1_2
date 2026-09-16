#!/usr/bin/env bash
# Lancia il controller nel container, SUL ROBOT (o in sim2sim su questo PC con --network lo).
#   ./run.sh <run_policy> <interfaccia>            es. ./run.sh 2026-09-13_21-27-39 eth0
# - --network host: CycloneDDS scopre il robot via multicast su 192.168.123.x, la rete bridge non la vede.
# - -d: il container sopravvive alla sessione SSH (equivale al tmux). Segui con `docker logs -f`.
# - la policy e i log sono volumi: cambi policy senza ricostruire, i log restano sul robot.
set -euo pipefail
RUN="${1:?run della policy, es. 2026-09-13_21-27-39}"
IFACE="${2:?interfaccia di rete, es. eth0 (robot) o lo (sim2sim)}"
ARCH="$(uname -m)"
WS="${WS:-$HOME/unitree_ws}"
POL="$WS/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN"
[ -f "$POL/exported/policy.onnx" ] || { echo "policy non trovata: $POL/exported/policy.onnx"; exit 1; }
mkdir -p "$WS/sim_logs"

docker rm -f h1_2_ctrl 2>/dev/null || true
docker run -d --name h1_2_ctrl --network host \
  -v "$POL:/root/unitree_ws/Squat_Unitree_H1_2/logs/rsl_rl/unitree_h1_2_squat/$RUN:ro" \
  -v "$WS/sim_logs:/root/unitree_ws/sim_logs" \
  "h1_2_ctrl_squat:$ARCH" --network "$IFACE"
echo "avviato. log:   docker logs -f h1_2_ctrl"
echo "         stop:  docker stop h1_2_ctrl     (SIGTERM -> il controller esce, lowcmd si ferma)"
