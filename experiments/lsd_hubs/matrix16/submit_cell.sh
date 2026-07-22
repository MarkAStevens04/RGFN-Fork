#!/bin/bash
#SBATCH --job-name=lsdflow_m16
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths ($HOME is read-only on compute nodes, Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Run ONE matrix cell's GPU pipeline: sample -> pick_hubs -> enumerate, emitting the uniform
# LSD-Flow artifact set (records.csv / compositions.json / routes.json + enum_children.json) that
# the CPU campaign (run_cell_campaign.sh) consumes. All four generators go through the identical
# per-env worker CLI (<gen>_worker.py --mode {sample,enumerate}); the manifest supplies every path.
#
# Usage (SLURM):    sbatch experiments/lsd_hubs/matrix16/submit_cell.sh <generator> <target>
# Usage (direct):   bash   experiments/lsd_hubs/matrix16/submit_cell.sh <generator> <target>
# Smoke override:   N_TRAJ=10000 N_HUBS=50 bash .../submit_cell.sh rgfn seh
# Knobs (env):  N_TRAJ (30000) N_HUBS (200) TOPK (100) SAMPLE_BATCH (200) ENUM_MAX (4000)
#               DEVICE (auto) STAGE (all|sample|enum)
set -uo pipefail

GEN=${1:?usage: submit_cell.sh <generator> <target>}
TGT=${2:?usage: submit_cell.sh <generator> <target>}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # worktree/main repo root
cd "$REPO"

N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
TOPK=${TOPK:-100}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
ENUM_MAX=${ENUM_MAX:-4000}
DEVICE=${DEVICE:-auto}
STAGE=${STAGE:-all}

# Resolve the cell from the manifest (single source of truth) into shell vars.
SPEC="$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || {
    echo "ERROR: manifest emit failed for '$GEN' '$TGT'"; exit 1; }
eval "$SPEC"

echo "=== cell=$CELL_TAG  stage=$STAGE  reward=$REWARD_NAME  gate=${HIGHER_IS_BETTER:+>}$MODE_REWARD_THRESHOLD"
echo "    env=$CONDA_ENV  worker=$WORKER  n_traj=$N_TRAJ  n_hubs=$N_HUBS  enum_max=$ENUM_MAX"
echo "    ckpt=$CHECKPOINT"
[ -f "$CHECKPOINT" ] || { echo "ERROR: checkpoint missing: $CHECKPOINT"; exit 1; }
if [ "$REWARD_TYPE" = "docking" ]; then
    echo "WARNING: '$TGT' is a DOCKING target (deferred) — enumeration would need GPU docking; "
    echo "         this launcher only handles surrogate reward-scoring. Proceeding anyway (sample is fine)."
fi

# Framework caches on $SCRATCH ($HOME read-only on compute nodes); offline W&B.
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export WANDB_MODE=offline PYTHONUNBUFFERED=1
RUN_DIR=$SCRATCH/rgfn_runs/lsdflow/matrix16/$CELL_TAG/run
mkdir -p "$SAMPLE_DIR" "$ENUM_DIR" "$RUN_DIR" "$TORCH_HOME" "$HF_HOME"

CONDA_SH=/home/markymoo/miniconda3/etc/profile.d/conda.sh
module load cuda/11.8.0 2>/dev/null || true    # dgl graphbolt on compute nodes (absent on Trillium)
source "$CONDA_SH"
conda activate "$CONDA_ENV"
# dgl/graphbolt need torch's bundled CUDA libs on LD_LIBRARY_PATH (cluster-agnostic; the
# ~/bin/rgfn-smoke-env.sh trick, applied to whichever env this cell's worker runs in).
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/$CONDA_ENV/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
echo "host=$(hostname)  python=$(which python)"; nvidia-smi -L 2>/dev/null | head -1 || true

# SCENT carries a trained-P_B sidecar (entry 024); pass it when present. Other generators: none.
GUIDANCE_ARG=(); [ -n "${GUIDANCE:-}" ] && GUIDANCE_ARG=(--guidance "$GUIDANCE")

if [ "$STAGE" = all ] || [ "$STAGE" = sample ]; then
    echo "=== [$CELL_TAG] SAMPLE ($N_TRAJ traj) -> $SAMPLE_DIR ==="
    python "$WORKER" --mode sample \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" \
        --reward-name "$REWARD_NAME" --n-trajectories "$N_TRAJ" --batch-size "$SAMPLE_BATCH" \
        --device "$DEVICE" --run-dir "$RUN_DIR" --out-dir "$SAMPLE_DIR" \
        || { echo "ERROR: sample stage failed"; exit 1; }
fi

if [ "$STAGE" = all ] || [ "$STAGE" = enum ]; then
    echo "=== [$CELL_TAG] PICK_HUBS (top-$N_HUBS) -> $ENUM_DIR/hubs.csv ==="
    python experiments/lsd_hubs/campaign/pick_hubs.py \
        --records "$SAMPLE_DIR/records.csv" --out "$ENUM_DIR/hubs.csv" \
        --top-k-candidates "$TOPK" --n-hubs "$N_HUBS" --higher-is-better "$HIGHER_IS_BETTER" \
        || { echo "ERROR: pick_hubs failed"; exit 1; }

    # FragGFN enumerate reloads the hub GRAPH from the sample stage's persisted hub_graphs.pkl
    # (obj_to_graph/SMILES→graph mis-decomposes ~6% of hubs) → point --sample-dir at the sample
    # output. RGFN/SCENT/RxnFlow reconstruct the hub state from SMILES natively (no persistence).
    SAMPLE_DIR_ARG=()
    case "$GENERATOR" in fraggfn) SAMPLE_DIR_ARG=(--sample-dir "$SAMPLE_DIR") ;; esac

    echo "=== [$CELL_TAG] ENUMERATE (<=$ENUM_MAX children/hub) -> $ENUM_DIR/enum_children.json ==="
    python "$WORKER" --mode enumerate \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" "${SAMPLE_DIR_ARG[@]}" \
        --reward-name "$REWARD_NAME" --hubs-file "$ENUM_DIR/hubs.csv" \
        --enum-max-children "$ENUM_MAX" --device "$DEVICE" --run-dir "$RUN_DIR" --out-dir "$ENUM_DIR" \
        || { echo "ERROR: enumerate stage failed"; exit 1; }
fi

echo "=== DONE [$CELL_TAG] -> sample=$SAMPLE_DIR enum=$ENUM_DIR ==="
