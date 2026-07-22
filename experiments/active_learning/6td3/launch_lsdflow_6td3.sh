#!/bin/bash
# Queue the three-arm LSD-Flow oracle-efficiency comparison on the 6TD3 GPU oracle.
#
# One sbatch job per arm (independent runs of the same config/oracle/seed; only the
# acquisition differs) — the [bengio2021gflownet] Fig.7 comparison with the LSD-Flow
# arm added:
#   hub_batching   — LSD-Flow: UCB-rank hubs by z(reward)+lambda*U(h), diversify into modes.
#   best_candidate — top-M sampled terminals under the same hit-bar + diversity filter.
#   random         — uniform policy over the same blocks (the forgiving floor).
#
# Usage:  bash experiments/active_learning/6td3/launch_lsdflow_6td3.sh [SEED ...]
#   default seed 42; pass several to launch a multi-seed campaign, e.g. `... 42 43 44`.
#
# After they finish, build the curve:
#   python -m validation.harness.acquisition_curve \
#     --runs $SCRATCH/rgfn_runs/experiments/active_learning/6td3_lsdflow \
#     --out validation/results/6td3_lsdflow_curve --metric topk_best \
#     --title "6TD3 — LSD-Flow hub-batching vs best-candidate vs random"
set -euo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
SUBMIT=experiments/active_learning/6td3/submit_al_6td3_lsdflow.sh
SEEDS=("${@:-42}")
for SEED in "${SEEDS[@]}"; do
    for ARM in hub_batching best_candidate random; do
        echo "queueing ARM=$ARM SEED=$SEED"
        ARM="$ARM" SEED="$SEED" sbatch "$SUBMIT"
    done
done
echo "queued ${#SEEDS[@]} seed(s) x 3 arms. Watch: squeue -u \$USER"
