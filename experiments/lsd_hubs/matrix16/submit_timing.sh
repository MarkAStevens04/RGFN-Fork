#!/bin/bash
#SBATCH --job-name=m16_timing
#SBATCH --time=04:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths ($HOME is read-only on compute nodes, Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Collect the MEASURED per-component compute time (Logs/039) for a matrix cell whose enumeration
# already completed BEFORE the workers were instrumented.
#
# Why a re-run is unavoidable: the compute-time axis is measured, not modeled (the
# lsdflow-compute-time-measured-not-modeled directive), so the only way to get per-hub
# enumeration/reward-gen/flow-extract seconds for an already-finished cell is to redo the
# enumeration with the timers live. Enumeration is deterministic, so this reproduces the same
# children — we keep only the timings.
#
# Safety: output goes to a SEPARATE scratch tree ($TIMING_ROOT), never the cell's real enum dir, so a
# partial or failed re-run can never damage the validated + committed enum_children.json. Publishing
# is a deliberate second step (merge_timings.sh) that copies ONLY enum_timings.json across.
#
# Slicing: a 200-hub RxnFlow re-run is ~9 h because its per-child retro-P_B dominates (~140 ms/child
# measured). Run it as N disjoint hub slices in parallel and merge with EnumTimings.merge (unions
# per_hub, charges setup_s once — median across slices). FragGFN is ~5 ms/child, so one slice is fine.
#
# Usage:  sbatch [--time=...] submit_timing.sh <generator> <target> [SLICE_IDX] [N_SLICES]
#         SLICE_IDX is 1-based; omit both to time all 200 hubs in one job.
# Knobs:  ENUM_MAX (4000) DEVICE (auto)
set -uo pipefail

GEN=${1:?usage: submit_timing.sh <generator> <target> [slice_idx] [n_slices]}
TGT=${2:?usage: submit_timing.sh <generator> <target> [slice_idx] [n_slices]}
SLICE_IDX=${3:-1}
N_SLICES=${4:-1}
ENUM_MAX=${ENUM_MAX:-4000}
DEVICE=${DEVICE:-auto}

REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "ERROR: cannot cd to repo root '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/matrix16/manifest.py ] || {
    echo "ERROR: not at repo root (no matrix16/manifest.py under $REPO)"; exit 1; }

# Conda must be bootstrapped BEFORE the manifest emit: a bare SLURM shell has no `python` on PATH.
source /home/markymoo/miniconda3/etc/profile.d/conda.sh || { echo "ERROR: no conda.sh"; exit 1; }
conda activate base

eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || {
    echo "ERROR: manifest emit failed for $GEN/$TGT"; exit 1; }
[ "$STATUS" = "ready" ] || { echo "ERROR: cell $CELL_TAG is '$STATUS', not ready"; exit 1; }

conda activate "$CONDA_ENV"
# Hard guard: --export=ALL means a PATH exported in the submitting shell is inherited here, which has
# already caused one cell to silently run in the WRONG env (fraggfn in rgfn -> ModuleNotFoundError;
# worse, SCENT would have imported OUR rgfn instead of its fork). Fail loudly instead.
ENV_PREFIX="/home/markymoo/miniconda3/envs/$CONDA_ENV"
export PATH="$ENV_PREFIX/bin:$PATH"
PY_REAL="$(command -v python || true)"
case "$PY_REAL" in
    "$ENV_PREFIX"/*) : ;;
    *) echo "ERROR: python resolved to '${PY_REAL:-<none>}', not '$ENV_PREFIX/bin/python'."; exit 1 ;;
esac

TIMING_ROOT=${TIMING_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16_timing}
SRC_HUBS="$ENUM_DIR/hubs.csv"
[ -f "$SRC_HUBS" ] || { echo "ERROR: no $SRC_HUBS — cell was never enumerated"; exit 1; }

OUT="$TIMING_ROOT/$CELL_TAG/slice${SLICE_IDX}of${N_SLICES}"
mkdir -p "$OUT"

# Disjoint round-robin hub slice (keeps the header). Round-robin, not contiguous blocks, so each
# slice sees a comparable mix of hub fan-outs and the slices finish at similar times.
python - "$SRC_HUBS" "$OUT/hubs.csv" "$SLICE_IDX" "$N_SLICES" <<'PY'
import csv, sys
src, dst, idx, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
rows = list(csv.DictReader(open(src)))
mine = [r for i, r in enumerate(rows) if i % n == (idx - 1)]
with open(dst, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(mine)
print(f"[timing] slice {idx}/{n}: {len(mine)}/{len(rows)} hubs -> {dst}")
PY

GUIDANCE_ARG=(); [ -n "${GUIDANCE:-}" ] && GUIDANCE_ARG=(--guidance "$GUIDANCE")
SAMPLE_DIR_ARG=(); case "$GENERATOR" in fraggfn) SAMPLE_DIR_ARG=(--sample-dir "$SAMPLE_DIR") ;; esac

echo "=== [$CELL_TAG] TIMING re-enumerate slice $SLICE_IDX/$N_SLICES -> $OUT ==="
python "$WORKER" --mode enumerate \
    --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" "${SAMPLE_DIR_ARG[@]}" \
    --reward-name "$REWARD_NAME" --hubs-file "$OUT/hubs.csv" \
    --enum-max-children "$ENUM_MAX" --device "$DEVICE" \
    --run-dir "$OUT/run" --out-dir "$OUT" \
    || { echo "ERROR: timing enumerate failed"; exit 1; }

echo "=== DONE [$CELL_TAG] slice $SLICE_IDX/$N_SLICES -> $OUT/enum_timings.json ==="
