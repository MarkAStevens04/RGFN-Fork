#!/bin/bash
#SBATCH --job-name=cmode_pools
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# CATALOGUE-DISTINCT ("c-mode") POOLS for the tau sweep: 500 molecules that are Tanimoto-< tau from
# every other pool member AND from every purchasable building block (ZINCFrag + our 418).
#
# WHY POOLS AND NOT A POST-FILTER. Filtering an existing 500-molecule pool would change the criterion
# AND shrink the pool, and a smaller pool alone costs SPARROW modes -- the two effects would be
# inseparable. Applying the criterion during construction keeps every rung at a full 500, so the only
# thing that varies is what "distinct" means.
#
# THREE STEPS, IN THIS ORDER, AND THE ORDER IS LOAD-BEARING:
#   1. SAMPLE more from the frozen checkpoint. The block test is a fixed bar (measured: 82% / 32% /
#      7.8% of above-gate molecules clear it at tau 0.5 / 0.4 / 0.3), so the strict rungs need more
#      raw draws -- ~7,600 above-gate molecules to yield 500 c-modes at tau=0.3, against the 1,544 we
#      have. Training is untouched; this is eval-phase scoring and must be reported as such.
#   2. BLOCK-SIM in a CLEAN process. `build_s3gfn_pools.py` imports the diversity module and its heavy
#      transitive deps, and forking a worker pool after that hangs (job bf5n600kr: SIGTERM at 30 min,
#      zero output, zero cache rows). Precomputing here means the builder only ever READS.
#   3. BUILD the three pools. The builder aborts if any candidate is missing from the cache rather
#      than treating it as passing -- an uncached molecule admitted by default could BE a block.
#
# Submit:  SEED=42 sbatch experiments/lsd_hubs/campaign/submit_cmode_pools.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" "$TRITON_CACHE_DIR"

SEED=${SEED:?set SEED}
GATE=${GATE:-5.68}
TAUS=${TAUS:-"0.5 0.4 0.3"}
TARGET_ABOVE=${TARGET_ABOVE:-7600}      # enough for 500 c-modes at the strictest rung
MAXSAMP=${MAXSAMP:-60000}
RUN=$SCRATCH/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed${SEED}
BIG=$SCRATCH/rgfn_runs/cmode/s3gfn_seh_seed${SEED}
CACHE=$SCRATCH/rgfn_runs/lsdflow_sparrow/blocksim/s3gfn_seh_seed${SEED}_cmode.jsonl
POOLROOT=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
mkdir -p "$BIG" "$(dirname "$CACHE")"
echo "host=$(hostname) seed=$SEED gate=$GATE taus='$TAUS'"

source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# --- 1. sample -----------------------------------------------------------------------------------
CANDS=$BIG/fixed_reward/candidates/candidates.csv
if [ ! -s "$CANDS" ]; then
  echo "=== [1/3] enlarging the draw (frozen checkpoint, no training) ==="
  conda run --no-capture-output -n s3gfn python experiments/lsd_hubs/campaign/s3gfn_sample_more.py \
      --run-dir "$RUN" --target-above-gate "$TARGET_ABOVE" --gate "$GATE" \
      --max-samples "$MAXSAMP" --out-dir "$BIG" || { echo "SAMPLE FAILED" >&2; exit 1; }
else
  echo "=== [1/3] candidates already present at $CANDS — skipping ==="
fi
[ -s "$CANDS" ] || { echo "FATAL: no candidates at $CANDS" >&2; exit 1; }
echo "  candidates: $(($(wc -l < "$CANDS") - 1)) rows"

# --- 2. block similarity, clean process ----------------------------------------------------------
echo "=== [2/3] block-similarity cache ==="
source ~/bin/rgfn-smoke-env.sh
python experiments/lsd_hubs/campaign/block_similarity_cache.py \
    --candidates "$CANDS" --gate "$GATE" --higher-is-better true \
    --nproc 32 --out "$CACHE" || { echo "BLOCKSIM FAILED" >&2; exit 1; }

# --- 3. one pool per tau -------------------------------------------------------------------------
RC=0
for TAU in $TAUS; do
  T=${TAU/./}
  echo ""; echo "=== [3/3] c-mode pool tau=$TAU ==="
  python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
      --candidates "$CANDS" --out-root "$POOLROOT" \
      --tag "s3gfn_seh_seed${SEED}_cmode${T}" --sizes 500 --gate "$GATE" \
      --pruned --cutoff "$TAU" --catalogue-distinct --block-sim-cache "$CACHE" \
    || { echo "POOL FAILED tau=$TAU" >&2; RC=1; }
done
echo ""; echo "CMODE-POOLS rc=$RC"
exit "$RC"
