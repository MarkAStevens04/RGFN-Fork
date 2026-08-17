#!/bin/bash
# Poll SURROGATE cells and run the campaign as soon as each one's enumeration lands.
#
# WHY A SECOND HARVESTER. autoharvest.sh delegates "is this cell complete?" to
# merge_docking_slices.sh, which is correct for docking cells (they arrive as N disjoint slices that
# must be unioned, and refusing a partial set is the whole point). A surrogate cell has no slices:
# submit_cell.sh writes enum/enum_children.json directly. So the docking harvester would call the
# merge, get "no slices under <dir>", and classify a FINISHED cell as "not ready" on every round,
# forever -- quietly, since that is also what an in-flight cell looks like. Rather than teach the
# docking harvester a second notion of readiness and risk the in-flight docking runs, this is a
# separate loop with the only readiness test a surrogate cell needs: the file exists and is non-empty.
#
# Usage:  MATRIX16_MANIFEST=... MATRIX16_SCRATCH=... MATRIX16_RESULTS=... \
#           setsid nohup bash harvest_surrogate.sh > /dev/null 2>&1 &
# Knobs:  CELLS ("<gen>:<target> ...")  INTERVAL_S (900)  ROUNDS (300)  STATE (dir for log/lock)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1

CELLS=${CELLS:-"rgfn:seh rgfn:drd2 scent:seh scent:drd2 rxnflow:seh rxnflow:drd2 fraggfn:seh fraggfn:drd2"}
INTERVAL_S=${INTERVAL_S:-900}
ROUNDS=${ROUNDS:-300}
STATE=${STATE:-/scratch/markymoo/rgfn_runs}
TAG=${TAG:-surr}
LOG="$STATE/harvest_$TAG.log"
mkdir -p "$STATE"

# flock, not a pgrep scan (which also matches the launching shell's own command line).
exec 9>"$STATE/harvest_$TAG.lock" || exit 1
flock -n 9 || { echo "$(date '+%F %T') another harvest_$TAG holds the lock -- exiting" >> "$LOG"; exit 0; }
say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

source /home/markymoo/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true
say "=== harvest_$TAG start pid=$$ cells='$CELLS' manifest=${MATRIX16_MANIFEST:-default}"

for i in $(seq 1 "$ROUNDS"); do
    all_done=1
    for cell in $CELLS; do
        gen="${cell%%:*}"; tgt="${cell##*:}"
        spec="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$gen" "$tgt" 2>/dev/null)" || { all_done=0; continue; }
        eval "$spec"
        marker="$ENUM_DIR/.harvested_campaign"
        [ -f "$marker" ] && continue
        all_done=0
        enum="$ENUM_DIR/enum_children.json"
        # -s not -f: a zero-byte file means the writer was killed mid-flush, which is NOT ready.
        [ -s "$enum" ] || continue
        say "  [$CELL_TAG] enumeration present -> campaign"
        if bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh "$gen" "$tgt" \
               > "$STATE/harvest_${TAG}_${CELL_TAG}.log" 2>&1; then
            say "  [$CELL_TAG] campaign OK -> $RESULTS_DIR"
            date '+%F %T' > "$marker"
        else
            say "  [$CELL_TAG] campaign FAILED -- see harvest_${TAG}_${CELL_TAG}.log (will retry)"
        fi
    done
    if [ "$all_done" = 1 ]; then say "=== all cells harvested after $i round(s) -- exiting"; exit 0; fi
    date '+%F %T' > "$STATE/harvest_$TAG.heartbeat"
    [ "$i" = "$ROUNDS" ] && break
    sleep "$INTERVAL_S"
done
say "=== harvest_$TAG finished $ROUNDS round(s) -- exiting"
