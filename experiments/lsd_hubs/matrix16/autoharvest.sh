#!/bin/bash
# Poll the in-flight DOCKING cells and, the moment a cell's slices cover every hub, promote it:
# merge -> campaign -> post-hoc gate sweep -> backup. Once per cell, unattended.
#
# WHY THIS EXISTS. A docking cell finishes as 8-12 independent slices spread across two clusters,
# each 8 h of GPU. Without this, the expensive artifacts sit on purge-eligible $SCRATCH as an
# unmerged slice pile until someone runs three commands by hand -- so a weekend of docking yields
# no readable result, and a purge in the meantime costs GPU-days.
#
# WHY POLLING RATHER THAN `--dependency`. SLURM dependencies would be the natural mechanism, but the
# slices are split across Balam and Trillium, which are separate schedulers: a Balam job cannot
# depend on a Trillium job id. What both clusters DO share is the filesystem, so completion is
# detected from the artifacts themselves. That also makes this robust to a requeue, a manual
# resubmission, or a slice that a human re-ran -- none of which a job-id dependency survives.
#
# READINESS IS DELEGATED, NOT REIMPLEMENTED. It simply attempts the STRICT merge, which already
# refuses unless every hub in hubs.csv is present exactly once. A failure means "not ready yet", so
# there is exactly one definition of complete, living in merge_docking_slices.sh. A partial set can
# therefore never be silently promoted to "the cell's enumeration" by this loop.
#
# Usage:  setsid nohup bash experiments/lsd_hubs/matrix16/autoharvest.sh > /dev/null 2>&1 &
# Knobs:  CELLS ("scent:clpp rxnflow:clpp")  INTERVAL_S (1800)  ROUNDS (240 = 5 days)
#         STATE (dir for lock/log/heartbeat) -- REQUIRED if you run MORE THAN ONE instance. The lock is
#         $STATE/autoharvest.lock with no per-instance suffix, so a second instance sharing a STATE
#         stands down immediately ("another autoharvest holds the lock") and harvests nothing while
#         looking alive. One instance per seed tree therefore needs a distinct STATE, e.g.
#         STATE=$SCRATCH/rgfn_runs/ah_seed43 beside the default. (harvest_surrogate.sh keys its lock on
#         TAG and does not have this hazard.)
#         GATES_<target> to override the post-hoc sweep points.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1

CELLS=${CELLS:-"scent:clpp rxnflow:clpp"}
INTERVAL_S=${INTERVAL_S:-1800}
ROUNDS=${ROUNDS:-240}
STATE=${STATE:-/scratch/markymoo/rgfn_runs}
LOG="$STATE/autoharvest.log"
mkdir -p "$STATE"

# Single instance (flock keys on the file -- a pgrep -f scan also matches the launching shell's own
# command line, which is a false positive that silently disables the loop).
exec 9>"$STATE/autoharvest.lock" || exit 1
flock -n 9 || { echo "$(date '+%F %T') another autoharvest holds the lock -- exiting" >> "$LOG"; exit 0; }

say() { printf '%s %s\n' "$(date '+%F %T')" "$*" >> "$LOG"; }

# Post-hoc gate points, in RAW oracle units (records.csv::reward). ClpP brackets its calibrated -8.0
# (Logs/045, AUROC 0.895); 6TD3 brackets its provisional -2.0. Comma lists, not lo:hi:step, because
# these are negative and descending.
GATES_clpp=${GATES_clpp:-"-11,-10.5,-10,-9.5,-9,-8.5,-8"}
GATES_6td3=${GATES_6td3:-"-4,-3.5,-3,-2.5,-2,-1.5,-1"}

say "=== autoharvest start pid=$$ host=$(hostname) cells='$CELLS' rounds=$ROUNDS interval=${INTERVAL_S}s"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate base 2>/dev/null || true

harvest_one() {
    local gen="$1" tgt="$2" tag spec
    spec="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$gen" "$tgt" 2>/dev/null)" || {
        say "  [$gen/$tgt] manifest emit FAILED -- skipping this round"; return 1; }
    eval "$spec"
    tag="$CELL_TAG"

    [ -f "$ENUM_DIR/.harvested" ] && return 0          # already promoted; nothing to do

    # Strict merge IS the readiness test (see header). Quiet on the not-ready path: this runs every
    # 30 min for days, and logging a full refusal each time would bury the events that matter.
    local out
    out="$(bash experiments/lsd_hubs/matrix16/merge_docking_slices.sh "$gen" "$tgt" 2>&1)" || {
        say "  [$tag] not ready: $(echo "$out" | grep -oE '[0-9]+/[0-9]+ hubs' | head -1)"
        return 0
    }
    say "  [$tag] COMPLETE -> merged. $(echo "$out" | grep -oE '[0-9,]+ children' | head -1)"

    if bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh "$gen" "$tgt" \
           > "$STATE/harvest_${tag}_campaign.log" 2>&1; then
        say "  [$tag] campaign OK -> $RESULTS_DIR"
    else
        say "  [$tag] campaign FAILED -- see harvest_${tag}_campaign.log"
        return 1        # no marker: retry next round rather than leave a half-promoted cell
    fi

    # Post-hoc gate sweep. The launch bar was chosen in advance (-8.0 ClpP) with the explicit plan to
    # sweep after the fact; this is that sweep, and it is free -- a re-scoring of child rewards that
    # are already measured, with no enumeration, GPU, or model.
    # NB: gate_curve.py imports glue (-> gin, dgl), so it needs the rgfn env, NOT the base env this
    # loop runs in. run_cell_campaign.sh sources rgfn-smoke-env.sh itself and is therefore fine; a bare
    # `python` here died on ModuleNotFoundError: gin. Run it through a subshell that switches env, so
    # the switch cannot leak into the manifest emits (which need base) on the next round.
    local gv; eval "gv=\${GATES_${tgt}:-}"
    if [ -n "$gv" ]; then
        # `--gates=$gv`, NOT `--gates "$gv"`: every docking gate list starts with a MINUS sign
        # (-11,-10.5,...), and argparse treats a value beginning with '-' as the next option --
        # "error: argument --gates: expected one argument". The equals form is unambiguous.
        if ( source ~/bin/rgfn-smoke-env.sh 2>/dev/null || conda activate rgfn
             python experiments/lsd_hubs/matrix16/gate_curve.py "$gen" "$tgt" --gates="$gv" ) \
               > "$STATE/harvest_${tag}_gates.log" 2>&1; then
            say "  [$tag] gate sweep OK ($gv)"
        else
            say "  [$tag] gate sweep FAILED -- see harvest_${tag}_gates.log (campaign still valid)"
        fi
    fi

    date '+%F %T' > "$ENUM_DIR/.harvested"
    # Back up immediately: the merged enumeration is the expensive, purge-vulnerable artifact and the
    # 6-hourly watchdog might not run for hours. Delta-rsync, so this is seconds.
    bash scripts/backup_scratch_critical.sh > "$STATE/harvest_${tag}_backup.log" 2>&1 \
        && say "  [$tag] backed up" || say "  [$tag] backup returned nonzero -- see harvest_${tag}_backup.log"
    say "  [$tag] HARVESTED"
}

for i in $(seq 1 "$ROUNDS"); do
    for cell in $CELLS; do
        harvest_one "${cell%%:*}" "${cell##*:}" || true
    done
    # Stop early once every named cell is done -- no reason to keep a loop alive for days.
    all_done=1
    for cell in $CELLS; do
        spec="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "${cell%%:*}" "${cell##*:}" 2>/dev/null)" || { all_done=0; continue; }
        eval "$spec"
        [ -f "$ENUM_DIR/.harvested" ] || all_done=0
    done
    if [ "$all_done" = 1 ]; then
        say "=== all cells harvested after $i round(s) -- exiting"
        exit 0
    fi
    date '+%F %T' > "$STATE/autoharvest.heartbeat"
    [ "$i" = "$ROUNDS" ] && break
    sleep "$INTERVAL_S"
done
say "=== autoharvest finished $ROUNDS round(s) -- exiting"
