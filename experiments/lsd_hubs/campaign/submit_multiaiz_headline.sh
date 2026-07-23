#!/bin/bash
#SBATCH --job-name=scent_multiaiz
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# T4.1 — MultiAiZ→SPARROW per-pool pricing of the SCENT sEH reaction-GFN library (Logs/043).
# Each acquisition function (hub_batching free-frag hero + best_candidate) is priced IN ISOLATION by
# MultiAiZ (n_iters cycles, discovers shared intermediates within THAT pool) → SPARROW MILP. Compare
# to the from-scratch AiZynth→SPARROW headline (Logs/041): does smarter discovery lower reactions,
# or does SPARROW's MILP already capture the sharing?
#
# Runs sweep_campaign.py under rgfn (imports glue→dgl); MultiAiZ crosses to the aizynth env + the MILP
# to the sparrow env by subprocess. $HOME is read-only on compute → results to $SCRATCH via --out-dir.
#
# SMOKE (debug node, small library, few cycles):
#   PARTITION=debug TIME=00:30:00 BUDGET_MODES=15 CUTMIN=0.5 CUTMAX=0.5 SNAP_POINTS=1 N_ITERS=3 \
#     TAG=scent_seh_multiaiz_smoke sbatch -p debug experiments/lsd_hubs/campaign/submit_multiaiz_headline.sh
# REAL (compute, coarse cutoffs 0.3/0.5/0.7, paper n_iters=5):
#   TIME=12:00:00 sbatch experiments/lsd_hubs/campaign/submit_multiaiz_headline.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189}
ENUM=${ENUM:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json}
SNAPSHOT=${SNAPSHOT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
OUT_DIR=${OUT_DIR:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results}
TAG=${TAG:-scent_seh_multiaiz}
THR=${THR:-7.0}
BUDGET_MODES=${BUDGET_MODES:-300}
CUTMIN=${CUTMIN:-0.3}; CUTMAX=${CUTMAX:-0.7}; CUTSTEP=${CUTSTEP:-0.2}   # coarse: 0.3/0.5/0.7
SNAP_POINTS=${SNAP_POINTS:-1}          # MultiAiZ is expensive -> price the final library only (1 pt/cutoff)
N_ITERS=${N_ITERS:-5}                   # MultiAiZ cycles (paper value)
MAX_ROUTES=${MAX_ROUTES:-0}            # candidate routes/target fed to SPARROW (0 = all)

export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface PYTHONUNBUFFERED=1
mkdir -p "$OUT_DIR" "$TORCH_HOME"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):$SCRATCH/vina_gpu/boost/lib:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L 2>/dev/null | head -1
echo "TAG=$TAG BUDGET_MODES=$BUDGET_MODES cutoffs=${CUTMIN}..${CUTMAX}/${CUTSTEP} snap_points=$SNAP_POINTS n_iters=$N_ITERS"

python experiments/lsd_hubs/campaign/sweep_campaign.py \
    --analysis-dir "$ANALYSIS" --enum-children "$ENUM" --snapshot "$SNAPSHOT" \
    --reward-threshold "$THR" --tag "$TAG" --out-dir "$OUT_DIR" \
    --evaluator multiaiz --snapshot-schedule geometric --snapshot-points "$SNAP_POINTS" \
    --child-policy free_frag \
    --budget-modes "$BUDGET_MODES" \
    --cutoff-min "$CUTMIN" --cutoff-max "$CUTMAX" --cutoff-step "$CUTSTEP" \
    --multiaiz-n-iters "$N_ITERS" --multiaiz-max-routes "$MAX_ROUTES" \
    --n-hubs 200

echo ""
echo "DONE $TAG -> $OUT_DIR/$TAG"
echo "  MultiAiZ→SPARROW reactions/mode per (strategy, cutoff); compare to results/scent_seh_sparrow_headline (AiZynth→SPARROW)."
echo "Sync back:  cp -r $OUT_DIR/$TAG \$REPO/experiments/lsd_hubs/campaign/results/"
