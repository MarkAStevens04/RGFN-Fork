#!/bin/bash
#SBATCH --job-name=s3gfn_frontier
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# T3.2 — place the S3-GFN sEH pool on the reactions-per-mode frontier (best-candidate; a SMILES model
# has no hubs) via from-scratch AiZynth->SPARROW, overlaying the SCENT hub-batching/best-candidate
# curves (Logs/041) => completes the MVP three-way comparison. Reads the S3-GFN route cache (job
# 71007 pool routed by submit_s3gfn_routes.sh); SPARROW MILP is fast, so the full tau-sweep is cheap.
# Runs in rgfn (imports glue->dgl) and shells to the sparrow env for the MILP.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

POOL=${POOL:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/71007/fixed_reward/candidates/candidates.csv}
CACHE=${CACHE:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/s3gfn_seh/routecache_zinc_uspto.json}
OUT_DIR=${OUT_DIR:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results}
TAG=${TAG:-s3gfn_seh_frontier}
THR=${THR:-7.0}

export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME"
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)  POOL=$POOL"; echo "CACHE=$CACHE"
if [ ! -s "$POOL" ] || [ ! -s "$CACHE" ]; then echo "[frontier] FATAL: pool or cache missing" >&2; exit 1; fi

python experiments/lsd_hubs/campaign/s3gfn_frontier.py \
    --pool-csv "$POOL" --tag "$TAG" --out-dir "$OUT_DIR" \
    --reward-threshold "$THR" \
    --evaluator sparrow --route-source from_scratch --sparrow-cache "$CACHE" \
    --cutoff-min 0.30 --cutoff-max 0.90 --cutoff-step 0.10 --budget-modes 300

echo ""
echo "DONE $TAG -> $OUT_DIR/$TAG  (overlay on results/scent_seh_sparrow_headline for the MVP figure)"
