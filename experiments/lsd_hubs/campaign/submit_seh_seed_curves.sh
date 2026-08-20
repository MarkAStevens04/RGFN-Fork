#!/bin/bash
#SBATCH --job-name=seh_seed_curve
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# OUR OWN side of the sEH competitor comparison, at seeds 43 and 44.
#
# WHY. The R=100 head-to-head in Logs/062 is currently CROSS-SEED and lopsided: the competitor has
# three seeds (BC-SB 42/43/44, BC-Enum-SB 42/43/44) while hub-batching has exactly ONE — seed 42's
# 82 distinct molecules, from `scent_seh_freefrag`. So the headline "~2.5x" compares our n=1 against
# their n=3 and cannot be given an error bar on our side. This run produces the missing two curves so
# the ratio becomes within-seed and both arms are n=3.
#
# WHY THE seed-43/44 CURVES DO NOT ALREADY EXIST. Those seeds were taken through
# `reconcile_t15` (-> `t45_seh_seed{43,44}_k0_reconcile/`), which emits a 300-mode count-once vs
# SPARROW reconciliation — NOT a per-step curve. The reaction-axis readout needs
# `curve_hub_batching.csv`, which only `sweep_campaign.py` writes.
#
# SETTINGS ARE seed 42's, EXACTLY. gate 7.0 (Logs/062's bar, NOT the newer 5.0 headline -- this run
# exists to complete THAT table), tau 0.5, child_policy free_frag (the hero acquisition), prebuild-k
# 0, evaluator count_once. The k0 reconcile runs for these same seeds used gate 7.0 / cutoff 0.5 and
# passed their coverage gate, so the inputs are known-good.
#
# SMOKE FIRST:
#   SEED=43 BUDGET_MODES=15 CUTMIN=0.5 CUTMAX=0.5 TAG=seedcurve_smoke \
#     sbatch -p debug --time=00:30:00 experiments/lsd_hubs/campaign/submit_seh_seed_curves.sh
#
# Usage:  SEED=43 sbatch -p compute --time=06:00:00 \
#           experiments/lsd_hubs/campaign/submit_seh_seed_curves.sh
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

SEED=${SEED:-43}
RUN=${RUN:-/scratch/markymoo/rgfn_runs/lsdflow/t45_seh_seed${SEED}}
ANALYSIS=${ANALYSIS:-$RUN/sample}
ENUM=${ENUM:-$RUN/enum/enum_children.json}
SNAPSHOT=${SNAPSHOT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed${SEED}/additional_fragments/fragments_4000.json}
THR=${THR:-7.0}
TAG=${TAG:-t45_seh_seed${SEED}_freefrag}
OUT_DIR=${OUT_DIR:-/scratch/markymoo/rgfn_runs/lsdflow_sparrow/seed_curves}
BUDGET_MODES=${BUDGET_MODES:-300}
BUDGET_REACTIONS=${BUDGET_REACTIONS:-100}
CUTMIN=${CUTMIN:-0.30}
CUTMAX=${CUTMAX:-0.90}
# prebuild-K is a REAL knob here, not a default to hardcode: the seed-42 gate-7.0 numbers this run is
# compared against exist at BOTH settings (scent_seh_freefrag preK=0 -> 82 modes;
# scent_seh_thr7 preK=20 -> 70), so the comparator has to be selectable or the ratio is not
# like-for-like.
PREBUILD_K=${PREBUILD_K:-0}

# $HOME is read-only on Balam compute; every cache that defaults there must be redirected or the job
# dies late and looks like a science failure.
export PYTHONUNBUFFERED=1 WANDB_MODE=offline
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib TRITON_CACHE_DIR=$SCRATCH/.cache/triton
mkdir -p "$OUT_DIR" "$TORCH_HOME" "$HF_HOME" "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

# sweep_campaign imports glue -> rgfn -> dgl (graphbolt C++), so the CUDA libs are needed even though
# count_once uses no GPU maths (balam-dgl-cuda-module).
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
conda activate rgfn
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/rgfn/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"

for P in "$ANALYSIS/records.csv" "$ENUM" "$SNAPSHOT"; do
    [ -s "$P" ] || { echo "FATAL: missing input $P" >&2; exit 1; }
done
echo "host=$(hostname)  SEED=$SEED  TAG=$TAG  gate=$THR  prebuild_k=$PREBUILD_K"
echo "  ANALYSIS=$ANALYSIS"; echo "  ENUM=$ENUM"; echo "  SNAPSHOT=$SNAPSHOT"

python experiments/lsd_hubs/campaign/sweep_campaign.py \
    --analysis-dir "$ANALYSIS" \
    --enum-children "$ENUM" \
    --snapshot "$SNAPSHOT" \
    --reward-threshold "$THR" \
    --tag "$TAG" \
    --out-dir "$OUT_DIR" \
    --child-policy free_frag \
    --prebuild-k "$PREBUILD_K" \
    --budget-modes "$BUDGET_MODES" \
    --budget-reactions "$BUDGET_REACTIONS" \
    --cutoff-min "$CUTMIN" --cutoff-max "$CUTMAX"
RC=$?
echo "rc=$RC"
[ "$RC" -eq 0 ] && echo "DONE -> $OUT_DIR/$TAG (sync curve_hub_batching.csv back to the repo)"
exit $RC
