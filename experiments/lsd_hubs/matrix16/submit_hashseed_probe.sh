#!/bin/bash
#SBATCH --job-name=hashseed_probe
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# Is this generator's sampling sensitive to PYTHONHASHSEED? The question decides whether a
# route-less cell is RECOVERABLE or LOST.
#
# WHY IT MATTERS MORE THAN IT SOUNDS. 23 cell-seeds have an empty routes.json and need a re-sample to
# recover routes. A re-sample is only a RECOVERY if it lands on the cell's original molecules --
# different molecules mean different hubs, which means the existing enumeration is invalid and the
# cell's published numbers move, i.e. a replacement rather than a repair.
#
# The original samples ran with PYTHONHASHSEED UNSET, so CPython used a per-process random hash seed.
# For RGFN that was decisive: --seed alone gave 377 vs 387 routes, because per-process dict/set
# iteration order feeds action-space construction and an identical RNG draw over a differently-ordered
# action list picks a different action. If a generator has that sensitivity, its original sample was
# taken under a condition nobody can reproduce -- so those routes are LOST, not pending.
#
# WHY THIS IS THE RIGHT TEST AND MATCHING THE ORIGINAL IS NOT. Comparing a fresh sample against the
# original confounds two variables at once (we would also be ADDING PYTHONHASHSEED=0, which the
# original lacked), so a mismatch would be uninterpretable. Two FRESH samples at the same --seed and
# DIFFERENT PYTHONHASHSEED isolate the hash variable alone:
#
#   identical  -> not hash-sensitive -> --seed suffices -> the original IS reproducible -> RECOVERABLE
#   differ     -> hash-sensitive     -> the original's condition is gone -> LOST
#
# Small N is legitimate HERE, unlike a reproduction check: sensitivity is a property of the sampler,
# and a single diverging trajectory proves it. (Proving reproduction, by contrast, needs full scale --
# a 400-traj sample recovering 2.4% of production hubs is not evidence of anything.)
#
# Usage:  GEN=rxnflow TGT=seh sbatch -p debug --time=00:40:00 \
#           experiments/lsd_hubs/matrix16/submit_hashseed_probe.sh
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || exit 1

GEN=${GEN:?set GEN=rxnflow|rgfn}
TGT=${TGT:?set TGT=seh|drd2|clpp}
N_TRAJ=${N_TRAJ:-2000}          # enough trajectories that a sensitive sampler must diverge
SEED=${PROBE_SEED:-0}           # SAME seed for both arms -- the hash seed is the only variable
OUT_ROOT=${OUT_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow_hashprobe}

CONDA_SH=/home/markymoo/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"; conda activate base
SPEC="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || exit 1
eval "$SPEC"

export WANDB_MODE=offline PYTHONUNBUFFERED=1
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib TRITON_CACHE_DIR=$SCRATCH/.cache/triton
mkdir -p "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"
module load cuda/11.8.0 2>/dev/null || true
conda activate "$CONDA_ENV"
[ "${CONDA_DEFAULT_ENV:-}" = "$CONDA_ENV" ] || { echo "FATAL: env '$CONDA_DEFAULT_ENV' != '$CONDA_ENV'" >&2; exit 1; }

echo "host=$(hostname)  cell=$CELL_TAG  n_traj=$N_TRAJ  --seed=$SEED (fixed)  writes under $OUT_ROOT"
echo "  NOTE: this NEVER touches the live cell ($SAMPLE_DIR) -- both arms write to $OUT_ROOT"

for HS in 0 12345; do
    OUT="$OUT_ROOT/${CELL_TAG}/hs$HS"
    rm -rf "$OUT"; mkdir -p "$OUT"
    echo "=== arm PYTHONHASHSEED=$HS ==="
    PYTHONHASHSEED=$HS python "$WORKER" --mode sample \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" ${GUIDANCE:+--guidance "$GUIDANCE"} \
        --reward-name "$REWARD_NAME" --n-trajectories "$N_TRAJ" --batch-size 200 \
        --device auto --run-dir "$OUT/run" --out-dir "$OUT" --seed "$SEED" \
        || { echo "ERROR: sample failed at PYTHONHASHSEED=$HS"; exit 1; }
done

echo "=== VERDICT ==="
python - "$OUT_ROOT/$CELL_TAG" "$CELL_TAG" <<'PY'
import csv, sys
from pathlib import Path
root, cell = Path(sys.argv[1]), sys.argv[2]

def rows(hs):
    with open(root / f"hs{hs}" / "records.csv", newline="") as fh:
        r = list(csv.DictReader(fh))
    return [x["child_key"] for x in r], {x["hub_key"] for x in r}

a, ha = rows(0)
b, hb = rows(12345)
print(f"  records      hs0={len(a):,}  hs12345={len(b):,}")
print(f"  ORDER identical: {a == b}")
print(f"  molecule sets identical: {set(a) == set(b)}  (overlap {len(set(a) & set(b)):,})")
print(f"  hub sets identical:      {ha == hb}  (overlap {len(ha & hb):,} of {len(ha):,})")
if a == b:
    print(f"  VERDICT [{cell}]: NOT hash-sensitive — --seed alone reproduces a sample, so the "
          f"original (PYTHONHASHSEED unset) IS reproducible. Routes are RECOVERABLE by re-sample.")
else:
    print(f"  VERDICT [{cell}]: HASH-SENSITIVE — two runs at the same --seed diverge on hash seed "
          f"alone. The original sample's condition cannot be reproduced, so its hubs are LOST; a "
          f"re-sample REPLACES the cell (re-enumerate + re-campaign) rather than repairing it.")
PY
echo "=== DONE $CELL_TAG ==="
