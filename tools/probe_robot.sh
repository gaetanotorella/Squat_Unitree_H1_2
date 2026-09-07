#!/usr/bin/env bash
# =============================================================================
# probe_robot.sh -- ricognizione READ-ONLY del PC di bordo dell'H1_2.
#
# Risponde a tre domande:
#   1. che hardware c'e' davvero su questo PC (CPU/RAM/GPU/dischi/rete)
#   2. quali sensori sono collegati e CHI li sta usando (telecamera, LiDAR)
#   3. che processi/servizi Unitree girano e con che configurazione DDS
#
# NON installa nulla, NON scrive nulla, NON tocca il controllo del robot.
#
# USO
#   scp Squat_Unitree_H1_2/tools/probe_robot.sh unitree@192.168.123.164:/tmp/
#   ssh unitree@192.168.123.164 'bash /tmp/probe_robot.sh' | tee /tmp/probe_pc4.txt
# oppure, senza copiare niente:
#   ssh unitree@192.168.123.164 'bash -s' < tools/probe_robot.sh | tee /tmp/probe_pc4.txt
# =============================================================================
set -u

sec() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }
try() { if have "$1"; then "$@" 2>&1; else echo "  [assente: $1]"; fi; }

sec "0. IDENTITA'"
echo "host      : $(hostname)"
echo "kernel    : $(uname -srm)"
echo "distro    : $( (lsb_release -ds 2>/dev/null || cat /etc/os-release 2>/dev/null | sed -n 's/^PRETTY_NAME=//p') | tr -d '"')"
echo "uptime    : $(uptime -p 2>/dev/null)"
echo "utente    : $(whoami)   sudo senza password: $(sudo -n true 2>/dev/null && echo SI || echo no)"

sec "1. CPU"
try lscpu | grep -Ei 'model name|^cpu\(s\)|thread|core|mhz|cache|architecture'
echo "-- estensioni rilevanti per l'inferenza (VNNI/AVX512/AMX) --"
grep -om1 -E 'avx512_vnni|avx_vnni|avx512f|amx_int8|avx2' /proc/cpuinfo | sort -u | tr '\n' ' '; echo
echo "-- carico attuale --"
try uptime

sec "2. MEMORIA E DISCO"
try free -h
echo
try df -h -x tmpfs -x devtmpfs

sec "3. GPU / ACCELERATORI"
echo "-- controller video (PCI) --"
try lspci -nn | grep -Ei 'vga|3d|display'
echo "-- nodi DRM --"
ls -l /dev/dri/ 2>/dev/null || echo "  [nessun /dev/dri]"
echo "-- NVIDIA? --"
if have nvidia-smi; then nvidia-smi 2>&1 | head -12; else echo "  nessuna NVIDIA (atteso: Intel Iris Xe)"; fi
echo "-- Jetson? --"
[ -f /etc/nv_tegra_release ] && cat /etc/nv_tegra_release || echo "  no (non e' un Jetson)"
echo "-- OpenCL / VA-API / OpenVINO --"
have clinfo && clinfo -l 2>&1 | head -5 || echo "  [clinfo assente]"
have vainfo && vainfo 2>&1 | grep -E 'Driver version|VAProfileH264' | head -3 || echo "  [vainfo assente]"
python3 -c "import openvino; print('  OpenVINO', openvino.__version__)" 2>/dev/null || echo "  [OpenVINO non installato]"
python3 -c "import onnxruntime as o; print('  onnxruntime', o.__version__, o.get_available_providers())" 2>/dev/null || echo "  [onnxruntime non installato]"

sec "4. RETE"
try ip -br addr
echo "-- rotte --"
try ip route
echo "-- vicini visti sulla rete interna --"
try ip neigh

sec "5. USB / TELECAMERA"
echo "-- lsusb completo --"
try lsusb
echo "-- Intel RealSense (VID 8086) --"
lsusb 2>/dev/null | grep -i '8086' || echo "  nessun dispositivo Intel su USB"
echo "-- nodi video --"
ls -l /dev/video* /dev/media* 2>/dev/null || echo "  [nessun /dev/video*]"
echo "-- elenco V4L2 --"
try v4l2-ctl --list-devices
echo "-- formati del primo nodo --"
[ -e /dev/video0 ] && try v4l2-ctl -d /dev/video0 --list-formats-ext | head -30
echo "-- CHI TIENE APERTA LA TELECAMERA (fondamentale: se e' occupata non puoi aprirla tu) --"
have fuser && fuser -v /dev/video* 2>&1 || echo "  [fuser assente]"
echo "-- SDK RealSense --"
have rs-enumerate-devices && rs-enumerate-devices -s 2>&1 | head -20 || echo "  [rs-enumerate-devices assente]"
python3 -c "import pyrealsense2 as rs; c=rs.context(); print('  pyrealsense2 ok, device:', [d.get_info(rs.camera_info.name) for d in c.devices])" 2>/dev/null || echo "  [pyrealsense2 non installato]"

sec "6. LIDAR (Livox MID-360)"
echo "-- porte UDP in ascolto (Livox usa 56000-57000) --"
have ss && ss -ulpn 2>/dev/null | head -25 || try netstat -ulpn
echo "-- processi livox/lidar --"
ps aux 2>/dev/null | grep -Ei 'livox|lidar|utlidar' | grep -v grep || echo "  nessuno"

sec "7. PROCESSI E SERVIZI UNITREE"
echo "-- processi con nomi riconoscibili --"
ps -eo pid,ppid,pcpu,pmem,etime,comm,args --sort=-pcpu 2>/dev/null \
  | grep -Ei 'unitree|h1_|dds|ros|camera|realsense|sport|motion|hand|arm' | grep -v grep | head -30
echo "-- servizi systemd attivi non standard --"
have systemctl && systemctl list-units --type=service --state=running --no-pager --no-legend 2>/dev/null \
  | grep -Eiv 'systemd-|dbus|cron|ssh|network|udev|rsyslog|polkit|snapd|accounts|udisks' | head -25
echo "-- top 8 per CPU --"
try top -bn1 | head -16

sec "8. CONFIGURAZIONE DDS"
echo "-- variabili d'ambiente --"
env | grep -Ei 'dds|ros|domain|rmw' || echo "  nessuna variabile DDS/ROS impostata in questa shell"
echo "-- le stesse variabili nei processi che girano DAVVERO (spesso diverse dalla shell ssh) --"
for p in $(pgrep -f 'unitree|sport|motion' 2>/dev/null | head -5); do
  echo "  pid $p ($(cat /proc/$p/comm 2>/dev/null)):"
  tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -Ei 'dds|domain|rmw|ros' | sed 's/^/    /' || echo "    [non leggibile]"
done
echo "-- file di configurazione cyclonedds --"
find / -maxdepth 6 -iname '*cyclonedds*.xml' -o -maxdepth 6 -iname 'cyclonedds.xml' 2>/dev/null | head
echo "-- porte RTPS aperte (dominio 0 = 7400/7401 + unicast 7410+) --"
have ss && ss -ulpn 2>/dev/null | grep -E ':74[0-9][0-9]' || echo "  nessuna porta 74xx (il DDS non gira su questo PC?)"
echo "-- ROS2 installato? --"
ls -d /opt/ros/* 2>/dev/null || echo "  no"

sec "9. SDK / CODICE GIA' PRESENTE A BORDO"
ls -l ~ 2>/dev/null | head -25   # senza dotfile: interessa il codice, non la configurazione
echo "-- installazioni unitree note --"
find / -maxdepth 5 -type d \( -name 'unitree_sdk2*' -o -name 'unitree_ros*' -o -name '*h1_2*' \) 2>/dev/null | head -15
echo "-- moduli python rilevanti --"
python3 -c "
import importlib.util as u
for m in ['cyclonedds','unitree_sdk2py','cv2','pyrealsense2','numpy','open3d','torch','onnxruntime']:
    try: ok = u.find_spec(m) is not None
    except Exception: ok = False
    print(f'  {m:18s} {\"OK\" if ok else \"--\"}')
" 2>/dev/null || echo "  [python3 non utilizzabile]"

sec "FINE"
