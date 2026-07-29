#!/bin/bash
#SBATCH --job-name=tau_sim_surface
#SBATCH --time=01:00:00                  # 221 cells x ~4.5s measured on the login node ~ 17 min; 1h is slack
#SBATCH --partition=debug                # pure CPU, well inside debug's 2h limit
#SBATCH --gpus-per-node=1                # Balam schedules by GPU count; this job never touches the GPU
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# The two-knob cost surface (Logs/051): reactions/mode over (reward bar τ) x (diversity cutoff) at a
# FIXED reaction budget. Off the login node only because the full 17x13 grid runs ~17 min, past the
# 15-minute interactive guideline — there is no GPU, model, or oracle in this job. It re-scores the
# cached enumeration (rewards already computed), so it is exactly `run_campaign.py` run 221 times with
# the pools loaded once.
#
# $HOME is READ-ONLY on Balam compute nodes, so the job writes to $SCRATCH and the four small
# artifacts are copied into the repo from the login node afterwards:
#   cp /scratch/markymoo/rgfn_runs/lsdflow/surface/<TAG>/surface.* \
#      experiments/lsd_hubs/campaign/results/<TAG>/
#
# Submit:   sbatch experiments/lsd_hubs/campaign/submit_tau_similarity_surface.sh
# Override: TAUS=4:8:0.5 SIMS=0.3:0.9:0.1 BUDGET=300 TAG=my_surface sbatch ...

set -uo pipefail

REPO=/home/markymoo/projects/RGFN_Fork/RGFN-Fork
ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189}
ENUM=${ENUM:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
TAUS=${TAUS:-4:8:0.25}
SIMS=${SIMS:-0.3:0.9:0.05}
BUDGET=${BUDGET:-300}
TAG=${TAG:-scent_seh_surface}
OUT=${OUT:-/scratch/markymoo/rgfn_runs/lsdflow/surface/$TAG}   # $HOME is read-only on compute

source ~/bin/rgfn-smoke-env.sh
cd "$REPO" || exit 1

echo "[surface] $(date) grid TAUS=$TAUS SIMS=$SIMS budget=$BUDGET tag=$TAG"
python experiments/lsd_hubs/campaign/tau_similarity_surface.py \
    --analysis-dir "$ANALYSIS" \
    --enum-children "$ENUM" \
    --snapshot "$SNAP" \
    --taus "$TAUS" \
    --similarities "$SIMS" \
    --budget-reactions "$BUDGET" \
    --child-policy free_frag --prebuild-k 20 \
    --tag "$TAG" --out-dir "$OUT"
echo "[surface] $(date) exit=$?"
