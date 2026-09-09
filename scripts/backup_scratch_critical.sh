#!/bin/bash
# Back up the IRREPLACEABLE parts of $SCRATCH to the def-naeilum PROJECT space, so a scratch purge
# cannot cost us a training run or block post-hoc analysis.
#
# WHY: /scratch on this cluster is purge-eligible and holds everything expensive we produce. The repo
# already versions the small committed results; what is NOT in git and NOT reproducible on a deadline
# is the model weights and the LSD-Flow artifacts derived from them.
#
# WHY NOT $HOME (changed 2026-08-18): /home is a 110 G quota and these backups had grown to 46 G of
# it, pushing it to 89 % and blocking logins. /project has ~1.1 T. The old $HOME copies were migrated
# on 2026-08-18 (copy -> checksum-verify all 227 .pt -> delete). Pointing DEST back at $HOME would
# recreate the outage, which is exactly what this script did before.
#
# /project is mounted on the LOGIN nodes but typically NOT on compute nodes -- run this from a login
# node, as its usage line already assumes. Do not call it from inside a SLURM job.
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
#     trace.csv*         ADDED 2026-09-07, and now as irreplaceable as a checkpoint. Two reasons.
#                        (a) Stage 2 (upsample_to_modes.py) harvests the training trace as a FREE
#                        pool of already-scored molecules -- saturn_clpp/s42 reaches its 500-mode
#                        target from history alone, turning 28 GPU-hours into 0. (b) It is the only
#                        evidence of what budget a checkpoint was trained at, which is what the
#                        whole budget-matched v2 campaign rests on.
#                        The GLOB IS LOAD-BEARING: a re-invoked runner truncates trace.csv to its
#                        59-byte header while the real history survives as trace.csv.1, so backing
#                        up `trace.csv` alone would faithfully preserve a stub. Nine v1 cells are
#                        in exactly that state.
#     timing.json        per-phase wall clock; the compute-frontier exhibit's only source.
#     arm_meta.json      which oracle-call count a v2 checkpoint sits at (see benchmark_v2).
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
DEST=${DEST:-/project/def-naeilum/naeilum/SDL3/RGFN_LSD_MarkStevens/backups/scratch_runs/rgfn_runs}
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

# Free-space floor. /home is a 110 G QUOTA, not a filesystem (`df /home` reports the 63 PB mount and
# is meaningless -- query the quota path itself). This matters because the script runs UNATTENDED from
# cron: filling the quota to zero would not just truncate a backup, it would break conda/login/VS Code
# for every session. So tier 1 (weights -- the actual dealbreaker) always runs, and tier 2 is skipped
# below the floor rather than half-written. Raise MIN_FREE_GB to be more conservative.
MIN_FREE_GB=${MIN_FREE_GB:-5}
free_gb() { df -BG --output=avail "$(dirname "$DEST")" 2>/dev/null | tail -1 | tr -dc '0-9'; }
FREE_BEFORE="$(free_gb)"
echo "=== free on $(dirname "$DEST"): ${FREE_BEFORE:-?} GiB (tier-2 floor ${MIN_FREE_GB} GiB)"
if [ -n "$FREE_BEFORE" ] && [ "$FREE_BEFORE" -lt "$MIN_FREE_GB" ]; then
    echo "WARNING: only ${FREE_BEFORE} GiB free -- below the ${MIN_FREE_GB} GiB floor."
    echo "WARNING: running TIER 1 ONLY (weights). Free space, then re-run for tier 2."
    TIER1_ONLY=1
fi

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
    --include='trace.csv*' \
    --include='timing.json' \
    --include='arm_meta.json' \
    --exclude='*' \
    "$SRC/experiments/" "$DEST/experiments/" || echo "WARNING: tier-1 rsync returned $?"

# ---- TIER 1b: the benchmark_v2 tree ---------------------------------------------------------------
# v2 lives at $SRC/v2/, NOT under experiments/, so the pass above does not see it at all.
#
# NO ALLOWLIST HERE, AND THAT IS THE FIX, NOT AN OVERSIGHT. This pass originally carried an include
# list modelled on tier 1's. Measured against the real tree it would have copied 401 MB of 54 GB --
# 0.74% -- and reported success: `last_gfn.pt` matches FragGFN's 9 files and nothing else, and
# `checkpoint*.pt` matches NOTHING, because the competitors name their weights
# mamba_<N>_agent.ckpt (131), s3gfn_seh-seed<N>_step<N>_model.pt (81), agent_step<N>.chkpt (81),
# final_mamba_agent.ckpt (18) and agent.chkpt (9). 329 of 338 checkpoints missed.
#
# THAT WAS THE THIRD INSTANCE OF ONE FAILURE: a second (or third) definition of "where this
# generator's weights live", written from OUR generators' conventions, in a place that reports
# success either way. The copy step hit it walking checkpoint DIRECTORIES and missing REINVENT's
# agent.chkpt in the run root. A wrong pattern is indistinguishable from a correct one in rsync's
# output, which is what makes it dangerous.
#
# The right fix is not a longer allowlist. **v2 is ALREADY curated** -- the copy step put exactly
# what the plan named there, file by file, with a .copy_manifest.json recording every one.
# Re-filtering an already-filtered tree through a narrower rule IS the bug. Let the copy manifest be
# the definition of what belongs and take the tree wholesale. Retention is not a space question:
# /project holds 949 G free of 1.2 T against a 54 GB payload, and the milestone checkpoints were
# kept deliberately because they are irreplaceable without a retrain.
#
# THE LOSS WAS NOT ONLY WEIGHTS. The same allowlist dropped TANGO's ARM-2 route pickles --
# 9,696 route_*.pkl under routes/tango_seh_s42/arma/full_pool/ -- which are the 3.5 GPU-h artifact
# carried forward specifically so nobody regenerates it. Checking only the checkpoint half of a
# too-narrow filter is how the other half stays missing.
#
# THE ROUTE PDFs ARE KEPT DELIBERATELY, not by omission. Beside every pickle sits a same-basename
# PDF -- syntheseus's graphviz render, emitted five per target because TANGO's config writer never
# overrides num_top_results. Measured: pkl 9,696 files / 0.067 GiB, pdf 9,696 files / 0.809 GiB, so
# the regenerable decoration is 12x the irreplaceable data and 2x everything the old allowlist kept.
# A denylist for `*.pdf` here would be entirely reasonable and is NOT used, for two reasons: 0.8 GiB
# against 949 G free defends no budget, and adding a second differently-shaped filter to the script
# we are fixing BECAUSE of a filter reintroduces the failure class -- a denylist can be wrong in the
# other direction just as silently. If the PDFs ever matter, drop them here knowingly.
#
# PER-CELL, NOT PER-PHASE. Cells run concurrently and reach each stage at different times, so waiting
# for a phase boundary means waiting a long time and probably forgetting. rsync is incremental, so
# calling this after each cell's stage passes verification is cheap and idempotent.
#
# --chmod KEEPS THE DESTINATION WRITABLE. `-a` implies `-p`, so a frozen (read-only) source cell
# would otherwise land read-only here -- and then the NEXT incremental sync could not write into
# those directories at all. Freeze protects the live tree; the backup must stay re-syncable. (An
# earlier version of this comment claimed rsync does not copy the read-only mode. It does.)
V2_SRC=${V2_SRC:-$SRC/v2}
if [ -d "$V2_SRC" ]; then
    echo
    echo "=== TIER 1b: benchmark_v2, WHOLESALE (the tree is already curated) ==="
    rsync -a --info=stats2 $DRY --chmod=Du+w,Fu+w \
        "$V2_SRC/" "$DEST/v2/" || echo "WARNING: tier-1b rsync returned $?"

    # ASSERT THE PAYLOAD LANDED. The failure this replaces was silent: rsync exited 0, printed its
    # stats, and 53 GB was not there. Compare destination bytes against source bytes rather than
    # trusting "files transferred", which reads ~0 on a correct incremental re-run.
    if [ -z "$DRY" ]; then
        v2_src_b=$(du -sb "$V2_SRC" 2>/dev/null | cut -f1)
        v2_dst_b=$(du -sb "$DEST/v2" 2>/dev/null | cut -f1)
        if [ -n "$v2_src_b" ] && [ -n "$v2_dst_b" ] && [ "$v2_src_b" -gt 0 ]; then
            pct=$(( 100 * v2_dst_b / v2_src_b ))
            printf "=== TIER 1b: backup holds %d%% of source (%.1f of %.1f GiB)\n" \
                "$pct" "$(echo "$v2_dst_b/1073741824" | bc -l)" \
                "$(echo "$v2_src_b/1073741824" | bc -l)" 2>/dev/null \
                || echo "=== TIER 1b: backup holds ${pct}% of source"
            if [ "$pct" -lt 95 ]; then
                echo "ERROR: benchmark_v2 backup holds only ${pct}% of the source tree." >&2
                echo "       That is the shape of the 0.74% allowlist bug -- something is filtering" >&2
                echo "       the payload out. Do NOT treat this cell set as backed up." >&2
            fi
        fi
    fi
else
    echo
    echo "=== TIER 1b: no $V2_SRC yet -- skipping benchmark_v2 ==="
fi

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
