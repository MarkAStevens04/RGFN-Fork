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

# Repo root. Under SLURM the batch script is COPIED to a spool dir, so BASH_SOURCE points at
# /var/spool/... not the repo — use SLURM_SUBMIT_DIR (set to where sbatch was invoked; launch
# scripts cd to the repo root first). Fall back to BASH_SOURCE for direct/login runs.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "ERROR: cannot cd to repo root '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/matrix16/manifest.py ] || {
    echo "ERROR: not at repo root (no matrix16/manifest.py under $REPO); check SLURM_SUBMIT_DIR."; exit 1; }

N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
TOPK=${TOPK:-1000}   # top candidates whose parent hubs pick_hubs ranks. CAPS the hub count: only
                     # TOPK distinct parents exist, so TOPK must be >= a few× N_HUBS to reach it
                     # (Logs/031: top-1000 -> 575 distinct -> top-200). TOPK=100 gave only 100 hubs.
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
ENUM_MAX=${ENUM_MAX:-4000}
DEVICE=${DEVICE:-auto}
STAGE=${STAGE:-all}

# Conda must be active BEFORE the manifest emit: a bare SLURM batch shell has NO python on PATH
# until a conda env is activated, yet the emit is what tells us which env the worker needs. conda
# base always has python and manifest.py is stdlib-only, so bootstrap with base for the emit, then
# switch to the cell's env for the worker (below). (Login smokes masked this — they had an env active.)
CONDA_SH=/home/markymoo/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate base

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

module load cuda/11.8.0 2>/dev/null || true    # dgl graphbolt on compute nodes (absent on Trillium)
conda activate "$CONDA_ENV"                     # switch from the bootstrap base env to the cell's env

# HARD GUARD: refuse to run in the wrong conda env. `sbatch --export=ALL` inherits the submitting
# shell's PATH, and any entry pointing at another env's bin can shadow the one we just activated.
# That is not a crash-level bug -- it is worse: scent_worker does `import rgfn`, which in OUR rgfn env
# silently resolves to the WRONG package (our RGFN instead of SCENT's fork) and would produce
# plausible-looking numbers from the wrong model. Prepend defensively, then verify and hard-fail.
ENV_PREFIX="/home/markymoo/miniconda3/envs/$CONDA_ENV"
export PATH="$ENV_PREFIX/bin:$PATH"
PY_REAL="$(command -v python || true)"
case "$PY_REAL" in
    "$ENV_PREFIX"/*) : ;;
    *)
        echo "ERROR: python resolved to '${PY_REAL:-<none>}', not '$ENV_PREFIX/bin/python'."
        echo "       The inherited PATH is shadowing the '$CONDA_ENV' env; refusing to run rather than"
        echo "       silently importing the wrong package. Resubmit from a shell without a conda"
        echo "       env-bin entry on PATH, or use 'sbatch --export=NONE,...'."
        exit 1
        ;;
esac
# dgl/graphbolt need torch's bundled CUDA libs on LD_LIBRARY_PATH (cluster-agnostic; the
# ~/bin/rgfn-smoke-env.sh trick, applied to whichever env this cell's worker runs in).
export LD_LIBRARY_PATH="$(ls -d /home/markymoo/miniconda3/envs/$CONDA_ENV/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
echo "host=$(hostname)  python=$(which python)"; nvidia-smi -L 2>/dev/null | head -1 || true

# SCENT carries a trained-P_B sidecar (entry 024); pass it when present. Other generators: none.
GUIDANCE_ARG=(); [ -n "${GUIDANCE:-}" ] && GUIDANCE_ARG=(--guidance "$GUIDANCE")

# Auto-skip a stage whose output already exists, so re-submitting a cell (after a timeout, a bug fix
# in the LATER stage, or a requeue) never redoes finished GPU work. Sampling 30k trajectories is
# ~20-40 min and enumeration is hours, so the accidental redo is the single most expensive mistake
# available here. Set RESUME=0 to force a clean re-run of both stages.
RESUME=${RESUME:-1}
if [ "$RESUME" = 1 ] && [ "$STAGE" = all ] && [ -s "$SAMPLE_DIR/records.csv" ]; then
    echo "=== [$CELL_TAG] SKIP sample — $SAMPLE_DIR/records.csv exists ($(($(wc -l < "$SAMPLE_DIR/records.csv") - 1)) records). RESUME=0 to redo. ==="
    STAGE=enum
fi

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
