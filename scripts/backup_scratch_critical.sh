#!/bin/bash
# Back up the IRREPLACEABLE parts of $SCRATCH to $HOME, so a scratch purge cannot cost us a training
# run or block post-hoc analysis.
#
# WHY: /scratch on this cluster is purge-eligible and holds everything expensive we produce, while
# /home is not purged. The repo already versions the small committed results; what is NOT in git and
# NOT reproducible on a deadline is the model weights and the LSD-Flow artifacts derived from them.
#
# Two tiers, most-critical first, so a partial run still saves the things that matter:
#
#   TIER 1 (dealbreaker if lost) — trained model weights + what makes them interpretable:
#     last_gfn.pt        the trained checkpoint. Re-training a 5k-iteration cell is GPU-days.
#     guidance_models.pt SCENT's backward-policy sidecar (entry 024). WITHOUT IT the trained P_B is
#                        unrecoverable even if the checkpoint survives -- the guidance MLPs live in a
#                        plain list that state_dict() skips. Losing this silently downgrades every
#                        flow analysis on that cell.
#     candidates.csv     the training run's own candidate pool; the "did this finish?" evidence and
#                        the best-candidate baseline's input.
#     *.gin/*.yaml/meta  the config actually used, so a cell is re-runnable and citable.
#
#   TIER 2 (regenerable, but hours of GPU each) — the LSD-Flow pipeline outputs:
#     sample/  30k-trajectory flow records + compositions + routes (~30 min/cell)
#     enum/    exhaustive one-step enumeration (~20 min to ~9 h/cell)
#     Skips the biggest re-derivable blobs (see EXCLUDES) to stay inside the /home quota.
#
# Deliberately NOT backed up: SLURM .out/.err logs, core dumps, per-run scratch logdirs, conda envs.
#
# Usage:  bash scripts/backup_scratch_critical.sh [--dry-run] [--tier1-only]
#         DEST=/some/other/path bash scripts/backup_scratch_critical.sh
set -uo pipefail

SRC=${SRC:-/scratch/markymoo/rgfn_runs}
DEST=${DEST:-$HOME/scratch_backup/rgfn_runs}
DRY=""
TIER1_ONLY=0
for a in "$@"; do
    case "$a" in
        --dry-run) DRY="--dry-run" ;;
        --tier1-only) TIER1_ONLY=1 ;;
        *) echo "unknown arg: $a"; exit 2 ;;
    esac
done

command -v rsync >/dev/null || { echo "ERROR: rsync not found"; exit 1; }

echo "=== source: $SRC"
echo "=== dest:   $DEST"
df -h "$(dirname "$DEST")" 2>/dev/null | tail -1
mkdir -p "$DEST" || { echo "ERROR: cannot create $DEST"; exit 1; }

# ---- TIER 1: weights + provenance -----------------------------------------------------------------
# --prune-empty-dirs keeps the tree shallow; the include list is ordered dirs-first so rsync can
# descend (an --include of a file alone never matches inside an excluded dir).
echo
echo "=== TIER 1: checkpoints + SCENT sidecars + candidates + configs ==="
rsync -a --info=stats2 $DRY \
    --prune-empty-dirs \
    --include='*/' \
    --include='last_gfn.pt' \
    --include='guidance_models.pt' \
    --include='candidates.csv' \
    --include='*.gin' \
    --include='*.yaml' \
    --include='meta.json' \
    --include='fragments_*.json' \
    --exclude='*' \
    "$SRC/experiments/" "$DEST/experiments/" || echo "WARNING: tier-1 rsync returned $?"

if [ "$TIER1_ONLY" = 1 ]; then
    echo; echo "=== tier-1 only requested; stopping ==="
    du -sh "$DEST" 2>/dev/null
    exit 0
fi

# ---- TIER 2: LSD-Flow samples + enumerations -------------------------------------------------------
# enumerated_records.csv is the single largest artifact and is fully re-derivable from
# enum_children.json + the checkpoint, so it is excluded to stay inside the /home quota. The scratch
# logdirs (_rxn_scratch_logdir etc.) are per-run junk.
echo
echo "=== TIER 2: LSD-Flow sample/ + enum/ artifacts ==="
# Redundancy excludes, in decreasing order of how much they save:
#  enumerated_records.csv  the largest artifact and fully re-derivable from enum_children.json.
#  matrix16_timing/        the compute-time re-runs RE-ENUMERATE cells we already have, so their
#                          enum_children.json is a duplicate of matrix16/<cell>/enum/. The only unique
#                          output is enum_timings.json, and merge_timings.sh already publishes that
#                          INTO matrix16/<cell>/enum/ -- which this backup covers. Backing up the
#                          slices too would store the same enumeration twice.
#  *smoke*/ *canary*/      throwaway pipeline-validation runs (3-hub enumerations, launcher tests).
#                          Their value was proving the code path works; that is recorded in the logs.
rsync -a --info=stats2 $DRY \
    --exclude='enumerated_records.csv' \
    --exclude='matrix16_timing/' \
    --exclude='*smoke*/' \
    --exclude='*canary*/' \
    --exclude='*_scratch_logdir/' \
    --exclude='run/' \
    --exclude='core.*' \
    "$SRC/lsdflow/" "$DEST/lsdflow/" || echo "WARNING: tier-2 rsync returned $?"

echo
echo "=== done ==="
du -sh "$DEST" 2>/dev/null
df -h "$(dirname "$DEST")" 2>/dev/null | tail -1
echo "Restore is a plain copy back:  rsync -a $DEST/ $SRC/"
