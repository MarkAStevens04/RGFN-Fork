#!/bin/bash
#SBATCH --job-name=pilotA
#SBATCH --partition=compute
#SBATCH --exclude=balam008
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# ARM-A VIABILITY PILOT, stage 1 — train ONE reaction-GFN to ~10,000 oracle calls.
#
# WHY THIS SCRIPT EXISTS RATHER THAN REUSING submit_rgfn.sh / submit_baseline.sh.
# `submit_rgfn.sh` hardcodes FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments and derives RUN_NAME from
# (system, seed) with NO override. Pointed at seh/42 it would resolve to the LIVE v1 cell
# rgfn_seh_5k/seed42 and resume from its 5,000-iteration checkpoint. `submit_baseline.sh` does
# honour OUT_ROOT, but only for the generators it handles (rgfn is not one of them). Sharing a run
# dir between a smoke and a real cell has already destroyed a finished cell once on this project, so
# this launcher takes an isolated root and REFUSES to run against the v1 tree.
#
# ARM A IS DEFINED ON ORACLE CALLS, NOT STEPS. The three generators have three different per-step
# call counts (RGFN 100 forward trajectories/iter, SCENT 64/iter, RxnFlow 64), and replay buffers
# make the arithmetic unsettleable. Until `_trace.py` is wired into these runners (agent A), this
# script takes an ITERATION count computed from the per-generator batch and the run reports the
# oracle-call count it ACTUALLY reached, measured from the run's own artifacts. Approximate
# placement is adequate for a viability question ("is there signal at all"); it is NOT adequate for
# the production campaign, which must checkpoint on the trace counter.
#
# Usage:
#   OUT_ROOT=$SCRATCH/rgfn_runs/v2_pilot N_ITERS=157 \
#     sbatch experiments/benchmark_v2/pilot/submit_pilot_train.sh scent seh 42
set -uo pipefail

GEN=${1:?usage: submit_pilot_train.sh <scent|rgfn> <seh|drd2> [seed]}
SYSTEM=${2:?usage: submit_pilot_train.sh <scent|rgfn> <seh|drd2> [seed]}
SEED=${3:-42}
N_ITERS=${N_ITERS:?set N_ITERS (SCENT 157 = 10,048 calls at batch 64; RGFN 100 = 10,000 at 100 traj/iter)}
OUT_ROOT=${OUT_ROOT:?set OUT_ROOT to an isolated tree, NOT the v1 experiments root}

# Repo root: under SLURM the batch script is copied to a spool dir, so BASH_SOURCE is useless.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/matrix16/targets.py ] || { echo "FATAL: not an RGFN-Fork checkout ($REPO)"; exit 1; }

# ---- GUARD: never write into the v1 tree -------------------------------------------------------
V1_ROOT="$SCRATCH/rgfn_runs/experiments"
case "$(readlink -m "$OUT_ROOT")" in
  "$(readlink -m "$V1_ROOT")"|"$(readlink -m "$V1_ROOT")"/*)
    echo "FATAL: OUT_ROOT '$OUT_ROOT' is inside the v1 experiments tree. Refusing." >&2; exit 2 ;;
esac

# `_armA` in the cell name keeps this pilot's artifacts from ever being mistaken for a v1 cell,
# even if someone later points a driver at this root.
RUN_NAME="fixed_reward/${GEN}_${SYSTEM}_armA/seed${SEED}"
RUN_DIR="$OUT_ROOT/$RUN_NAME"
COMPLETE="$RUN_DIR/fixed_reward/candidates/candidates.csv"
if [ -f "$COMPLETE" ]; then
  echo "[pilot] already complete ($COMPLETE); no-op."; exit 0
fi
mkdir -p "$RUN_DIR"

module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh
export PYTHONUNBUFFERED=1
# PYTHONHASHSEED is load-bearing for reproducibility, not hygiene: --seed alone does not pin a
# sample (377 vs 387 routes at the same seed; 730/730 byte-identical with it set).
export PYTHONHASHSEED=0
# $HOME is read-only on compute nodes; anything caching there kills the job minutes in.
export WANDB_MODE=offline WANDB_DIR=$SCRATCH/wandb WANDB_CACHE_DIR=$SCRATCH/.cache/wandb
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache/xdg}
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "host=$(hostname) gen=$GEN system=$SYSTEM seed=$SEED n_iters=$N_ITERS"
echo "repo=$REPO"
echo "run_dir=$RUN_DIR"
nvidia-smi -L || true
T0=$(date +%s)

case "$GEN" in
  scent)
    CFG=validation/configs/scent_${SYSTEM}_fixed_5k.gin
    # --log-recipes: promoted-fragment routes are observable ONLY during training. A run without
    # them can never be repaired, and this pilot needs the promoted-fragment count.
    conda run --no-capture-output -n scent python validation/generators/scent/run_scent_fixed.py \
        --cfg "$CFG" --seed "$SEED" --root-dir "$OUT_ROOT" --run-dir "$RUN_DIR" \
        --log-recipes --n-iterations "$N_ITERS"
    RC=$? ;;
  rgfn)
    case "$SYSTEM" in
      seh)  CFG=configs/glue/fixed_reward_seh_proxy_stdlib_5k.gin ;;
      drd2) CFG=configs/glue/fixed_reward_drd2_stdlib_5k.gin ;;
      *) echo "FATAL: pilot supports surrogate systems only (seh|drd2)"; exit 2 ;;
    esac
    conda activate rgfn
    python scripts/fixed_reward.py \
        --cfg "$CFG" --seed "$SEED" --root-dir "$OUT_ROOT" --run-name "$RUN_NAME" \
        --gin-binding "Trainer.n_iterations=${N_ITERS}"
    RC=$? ;;
  *) echo "FATAL: unknown GEN '$GEN' (want scent|rgfn)"; exit 2 ;;
esac

T1=$(date +%s)
echo "[pilot] $GEN $SYSTEM seed=$SEED exit=$RC train+sample wall=$((T1-T0))s"
[ -f "$COMPLETE" ] && echo "[pilot] COMPLETE: $COMPLETE" || echo "[pilot] INCOMPLETE (no candidates.csv)"
exit $RC
