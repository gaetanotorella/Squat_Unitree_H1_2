#!/usr/bin/env bash
# Genera tutti i media della tesi (media.py, un preset alla volta) in una cartella nuova:
#   media_tesi/media_1, media_tesi/media_2, ... (il numero successivo al piu' alto gia' presente).
#
# Uso, dalla radice del repo o da dove si vuole, con l'ambiente conda di Isaac Lab attivo:
#   scripts/rsl_rl/media_all.sh                                # tutti i preset
#   scripts/rsl_rl/media_all.sh --rendering_mode quality       # gli argomenti passano a OGNI preset
#   PRESETS="V3 V5a" scripts/rsl_rl/media_all.sh               # solo alcuni preset
#
# Variabili (opzionali):
#   PRESETS     preset da generare, nell'ordine (default: tutti)
#   MEDIA_ROOT  cartella che contiene media_N (default: <repo>/media_tesi)
#   PYTHON      interprete (default: python, quello dell'ambiente conda attivo)
#   TIMEOUT     secondi massimi per preset, 0 = nessuno (default: 3600)
#
# Ogni preset gira in un processo Isaac Sim separato: se uno fallisce gli altri vanno avanti. Nella
# cartella restano i file, un .json per file (come e' stato fatto), log/<preset>.log con l'output
# completo e riepilogo.txt con comando, commit ed esito di ogni preset.

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"
PRESETS="${PRESETS:-S1 V1 V2 V3 V4a V4b V4c V5a V5b}"
MEDIA_ROOT="${MEDIA_ROOT:-$REPO/media_tesi}"
PYTHON="${PYTHON:-python}"
TIMEOUT="${TIMEOUT:-3600}"

if [[ -z "${CONDA_PREFIX:-}" && "$PYTHON" == "python" ]]; then
    echo "[media_all] ATTENZIONE: nessun ambiente conda attivo, 'python' potrebbe non avere Isaac Lab" >&2
fi

# cartella successiva: media_<max+1>
mkdir -p "$MEDIA_ROOT"
n=0
for d in "$MEDIA_ROOT"/media_*/; do
    [[ -d "$d" ]] || continue
    k="$(basename "$d")"; k="${k#media_}"
    [[ "$k" =~ ^[0-9]+$ ]] && (( 10#$k > n )) && n=$((10#$k))
done
OUT="$MEDIA_ROOT/media_$((n + 1))"
mkdir -p "$OUT/log"

SUMMARY="$OUT/riepilogo.txt"
{
    echo "media_all.sh - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "commit:    $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?')$(git -C "$REPO" diff --quiet -- scripts 2>/dev/null || echo ' (con modifiche locali in scripts/)')"
    echo "preset:    $PRESETS"
    echo "argomenti: ${*:-(nessuno)}"
    echo
} > "$SUMMARY"

trap 'echo; echo "[media_all] interrotto: i preset fatti finora sono in $OUT"; echo "INTERROTTO dall utente" >> "$SUMMARY"; exit 130' INT

echo "[media_all] cartella: $OUT"
cd "$REPO" || exit 1
ok=(); ko=()
for p in $PRESETS; do
    log="$OUT/log/$p.log"
    echo "[media_all] === $p (log: log/$p.log)"
    start=$SECONDS
    cmd=("$PYTHON" "$SCRIPT_DIR/media.py" --preset "$p" --headless --out "$OUT" "$@")
    # --foreground: il processo resta nel gruppo del terminale, quindi Ctrl-C arriva anche a Isaac Sim
    (( TIMEOUT > 0 )) && cmd=(timeout --foreground --signal=TERM --kill-after=60 "$TIMEOUT" "${cmd[@]}")
    # a schermo solo le righe dello script; l'output completo di Isaac Sim resta nel log
    PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | tee "$log" | grep --line-buffered '^\[media\]'
    rc=${PIPESTATUS[0]}
    dur=$(( SECONDS - start ))
    # riuscito = media.py e' arrivato a scrivere il .json (Isaac Sim a volte esce con codici strani
    # in chiusura anche quando ha salvato tutto)
    if grep -q '^\[media\] dettagli in' "$log"; then
        ok+=("$p"); esito="ok"
    else
        ko+=("$p"); esito="FALLITO (codice $rc)"
        [[ $rc -eq 124 ]] && esito="FALLITO (oltre ${TIMEOUT} s)"
        echo "[media_all] $p: $esito - ultime righe del log:"; tail -n 15 "$log" | sed 's/^/    /'
    fi
    printf '%-4s  %-28s  %4d:%02d\n' "$p" "$esito" $((dur / 60)) $((dur % 60)) | tee -a "$SUMMARY"
done

{
    echo
    echo "riusciti: ${ok[*]:-nessuno}"
    echo "falliti:  ${ko[*]:-nessuno}"
    echo
    echo "file:"
    (cd "$OUT" && ls -1 *.png *.mp4 2>/dev/null | sed 's/^/  /')
} >> "$SUMMARY"
echo "[media_all] fatto: ${#ok[@]} riusciti, ${#ko[@]} falliti -> $OUT (riepilogo.txt)"
[[ ${#ko[@]} -eq 0 ]]
