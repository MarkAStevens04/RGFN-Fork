#!/bin/bash
#SBATCH --job-name=cmode_ours
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# Hub-batching under the CATALOGUE-DISTINCT mode definition, swept over tau, to sit beside the
# competitor's pools built the same way. No routing: the block test is structural.
# Needs rgfn-smoke-env (run_campaign imports glue -> dgl); see job 75746.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
SEED=${SEED:-42}; TAUS=${TAUS:-"0.5 0.4 0.3"}; GATE=${GATE:-5.68}
SUF=""; [ "$SEED" != "42" ] && SUF="_seed${SEED}"
S=$SCRATCH/rgfn_runs/lsdflow/matrix16${SUF}/scent_seh
SNAP=$(ls $SCRATCH/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed${SEED}/additional_fragments/fragments_*.json | sort -t_ -k2 -n | tail -1)
CACHE=$SCRATCH/rgfn_runs/lsdflow_sparrow/blocksim/scent_seh_seed${SEED}.jsonl
source ~/bin/rgfn-smoke-env.sh
# Build the cache here if absent, in its OWN process, BEFORE run_campaign imports the diversity
# stack. run_campaign only ever reads it -- forking a pool after that import hangs (job bf5n600kr).
if [ ! -s "$CACHE" ]; then
  echo "=== block-similarity cache (seed $SEED) ==="
  python experiments/lsd_hubs/campaign/block_similarity_cache.py \
      --enum-children "$S/enum/enum_children.json" --gate "$GATE" --higher-is-better true \
      --nproc 16 --out "$CACHE" || { echo "BLOCKSIM FAILED" >&2; exit 1; }
fi
[ -s "$CACHE" ] || { echo "FATAL: no block-sim cache at $CACHE" >&2; exit 1; }
RC=0
for TAU in $TAUS; do
  echo ""; echo "################ ours: catalogue-distinct tau=$TAU ################"
  python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$S/sample" --enum-children "$S/enum/enum_children.json" --snapshot "$SNAP" \
    --reward-threshold "$GATE" --higher-is-better true --similarity "$TAU" \
    --budget-reactions 100 --budget-modes 300 --child-policy free_frag \
    --catalogue-distinct --block-sim-cache "$CACHE" \
    --tag "cmode_ours_s${SEED}_t${TAU}" \
    --out-dir "$SCRATCH/rgfn_runs/lsdflow_sparrow/results/cmode_ours_seed${SEED}_t${TAU}" || RC=1
done
echo "CMODE-OURS rc=$RC"; exit "$RC"
