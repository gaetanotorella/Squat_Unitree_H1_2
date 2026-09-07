#!/usr/bin/env bash
# =============================================================================
# probe_dds.sh -- elenca i topic CycloneDDS del robot DAL PORTATILE.
#
# Non serve installare niente a bordo: la discovery DDS e' peer-to-peer sulla
# stessa sottorete, quindi basta essere collegati via ethernet. E' PASSIVO:
# legge soltanto i topic builtin di discovery, non pubblica nulla.
#
# USO
#   ./probe_dds.sh                 # interfaccia enp1s0, dominio 0
#   ./probe_dds.sh enp1s0 0
#   ./probe_dds.sh enp1s0 0 peers  # se il multicast non passa (vedi sotto)
# =============================================================================
set -u
IFACE="${1:-enp1s0}"
DOMAIN="${2:-0}"
MODE="${3:-multicast}"
export PATH="$HOME/.local/bin:$PATH"

if [ "$MODE" = "peers" ]; then
  # fallback: discovery unicast verso indirizzi noti, se il multicast e' bloccato
  export CYCLONEDDS_URI="<CycloneDDS><Domain id=\"any\"><General>\
<Interfaces><NetworkInterface name=\"$IFACE\"/></Interfaces>\
<AllowMulticast>false</AllowMulticast></General>\
<Discovery><ParticipantIndex>auto</ParticipantIndex><Peers>\
<Peer address=\"192.168.123.161\"/><Peer address=\"192.168.123.162\"/>\
<Peer address=\"192.168.123.163\"/><Peer address=\"192.168.123.164\"/>\
</Peers></Discovery></Domain></CycloneDDS>"
else
  export CYCLONEDDS_URI="<CycloneDDS><Domain id=\"any\"><General>\
<Interfaces><NetworkInterface name=\"$IFACE\"/></Interfaces></General></Domain></CycloneDDS>"
fi

echo "interfaccia $IFACE | dominio $DOMAIN | discovery $MODE"
ip -br addr show "$IFACE" 2>/dev/null || echo "  !! interfaccia $IFACE inesistente"
echo

echo "================ APPLICAZIONI DDS IN RETE ================"
cyclonedds ps -i "$DOMAIN" -r 10s --suppress-progress-bar 2>&1

echo
echo "================ TOPIC (nome + tipo + QoS) ================"
cyclonedds ls -i "$DOMAIN" -r 10s --suppress-progress-bar 2>&1

echo
echo "================ IDL RICOSTRUITO DEI TOPIC CHIAVE ================"
# Serve a verificare che il TIPO combaci fra le due sponde: un mismatch di IDL
# non produce alcun errore, semplicemente i dati non arrivano mai.
for t in rt/lowstate rt/lowcmd rt/wirelesscontroller; do
  echo "--- $t ---"
  cyclonedds typeof "$t" -i "$DOMAIN" -r 5s --suppress-progress-bar 2>&1 | head -40
done
