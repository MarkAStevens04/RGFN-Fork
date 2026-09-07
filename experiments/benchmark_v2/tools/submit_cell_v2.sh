#!/bin/bash
#SBATCH --job-name=bm2_cell
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
# Absolute $SCRATCH log paths -- $HOME is READ-ONLY on compute nodes (Logs/012).
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
#
# Run ONE benchmark_v2 reaction-GFN cell: sample -> pick_hubs -> enumerate.
#
# THIS IS NOT A PORT OF matrix16/submit_cell.sh. That file stays exactly as it is -- it is the
# frozen v1 record, jobs still queue against it, and bash resumes a script at a BYTE OFFSET so
# editing one mid-flight runs garbage hours later (it killed job 74318 five hours in, after its work
# had already succeeded). Its hard-won parts are kept here; its two defects are FIXED, not carried:
#
#   1. v1 resolved SAMPLE_DIR from the manifest via `eval`, which CLOBBERED any env override. A
#      caller who set SAMPLE_DIR to somewhere safe was silently ignored and the real cell was
#      overwritten in place. Fixed structurally: every path here is a pure function of (tag, arm),
#      and the single override -- BENCHMARK_V2_SCRATCH -- moves the WHOLE tree at once, so it is
#      impossible to redirect half a run.
#   2. v1's tag was (generator, target) with no seed in it, so a second seed collided in both the
#      scratch and results trees. v2 runs THREE seeds where v1 largely ran one. The seed is now in
#      the tag (scent_seh_s42), so collision is not something to remember to avoid.
#
# Usage (SLURM):  sbatch .../submit_cell_v2.sh scent seh 42
# Usage (direct): bash   .../submit_cell_v2.sh scent seh 42
# Env knobs: ARM (a) N_TRAJ (30000) N_HUBS (200) SAMPLE_BATCH (200) ENUM_MAX (0=uncapped)
#            DEVICE (auto) STAGE (all|sample|enum) RESUME (1) MIN_HUB_DEPTH ("" = unfiltered)
set -uo pipefail

GEN=${1:?usage: submit_cell_v2.sh <generator> <target> <seed>}
TGT=${2:?usage: submit_cell_v2.sh <generator> <target> <seed>}
SEED=${3:?usage: submit_cell_v2.sh <generator> <target> <seed>}

ARM=${ARM:-a}
N_TRAJ=${N_TRAJ:-30000}
N_HUBS=${N_HUBS:-200}
SAMPLE_BATCH=${SAMPLE_BATCH:-200}
# UNCAPPED BY DEFAULT. v1's 4,000-children-per-hub cap silently truncated SCENT (and only SCENT --
# it promotes intermediates into a growing vocabulary, so one scaffold reaches far more children).
# Lifting it gave the affected cells 8-61% more molecules and moved every headline in our favour.
ENUM_MAX=${ENUM_MAX:-0}
DEVICE=${DEVICE:-auto}
STAGE=${STAGE:-all}
RESUME=${RESUME:-1}
MIN_HUB_DEPTH=${MIN_HUB_DEPTH:-}

# Repo root. Under SLURM the batch script is COPIED to a spool dir, so BASH_SOURCE points at
# /var/spool/... -- use SLURM_SUBMIT_DIR (launchers cd to the repo root first).
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "ERROR: cannot cd to repo root '$REPO'"; exit 1; }
TOOLS="$REPO/experiments/benchmark_v2/tools"
[ -f "$TOOLS/manifest.py" ] || { echo "ERROR: not at repo root (no $TOOLS/manifest.py)"; exit 1; }

# Conda must be active BEFORE the manifest emit: a bare SLURM batch shell has NO python on PATH.
# base always has one and manifest.py is stdlib-only, so bootstrap with base, then switch.
CONDA_SH=/home/markymoo/miniconda3/etc/profile.d/conda.sh
source "$CONDA_SH"
conda activate base

SPEC="$(python "$TOOLS/manifest.py" --emit "$GEN" "$TGT" "$SEED" --arm "$ARM")" || {
    echo "ERROR: manifest emit failed for $GEN/$TGT/$SEED arm$ARM"; exit 1; }
eval "$SPEC"

if [ "$ROLE" != "hub_batching" ]; then
    echo "ERROR: $CELL_TAG is role=$ROLE. This driver runs the reaction-GFN pipeline only;"
    echo "       competitors go through upsample -> retrosynthesis -> selection instead."
    exit 2
fi

# ---- the training artifact must exist, be verified, and be frozen ---------------------------------
# Reading an UNVERIFIED checkpoint is how a whole campaign ends up built on a cell that was short of
# budget or missing its trace. The gate is cheap; the re-run it prevents is not.
if [ ! -d "$TRAIN_DIR" ]; then
    echo "ERROR: no training artifact at $TRAIN_DIR (status=$STATUS)"; exit 1
fi
if ! python "$TOOLS/verify_cell.py" --cell "$GEN/$TGT/$SEED" --arm "$ARM" --stage train --no-marker >/dev/null 2>&1; then
    echo "ERROR: $CELL_TAG arm$ARM fails train-stage verification; refusing to build on it."
    python "$TOOLS/verify_cell.py" --cell "$GEN/$TGT/$SEED" --arm "$ARM" --stage train --no-marker 2>&1 | sed -n '/FAIL/p'
    exit 1
fi
if [ "$FROZEN" != true ]; then
    echo "WARNING: $TRAIN_DIR is not frozen. Another process can still overwrite this checkpoint"
    echo "         and its trace mid-run. Freeze it: $TOOLS/freeze_cell.sh $GEN/$TGT/$SEED --arm $ARM"
fi

# Resolve the checkpoint and the config the run actually used, from the cell's own artifacts rather
# than a second hand-maintained table that could disagree with them.
CHECKPOINT="$TRAIN_DIR/$(python -c "import json,sys;print(json.load(open(sys.argv[1])).get('checkpoint','checkpoint.pt'))" "$TRAIN_DIR/arm_meta.json")"
[ -f "$CHECKPOINT" ] || { echo "ERROR: checkpoint missing: $CHECKPOINT"; exit 1; }
CONFIG="$(ls "$TRAIN_DIR"/*.gin "$TRAIN_DIR"/run_config.yaml 2>/dev/null | head -1)"
[ -n "$CONFIG" ] || { echo "ERROR: no config (*.gin / run_config.yaml) recorded in $TRAIN_DIR"; exit 1; }
GUIDANCE_ARG=()
[ -f "$TRAIN_DIR/guidance_models.pt" ] && GUIDANCE_ARG=(--guidance "$TRAIN_DIR/guidance_models.pt")

echo "=== cell=$CELL_TAG arm=$ARM stage=$STAGE  reward=$REWARD_NAME gate=$MODE_REWARD_THRESHOLD"
echo "    env=$CONDA_ENV worker=$WORKER n_traj=$N_TRAJ n_hubs=$N_HUBS enum_max=$ENUM_MAX"
echo "    ckpt=$CHECKPOINT"
echo "    cfg=$CONFIG"

# ---- environment ----------------------------------------------------------------------------------
# $HOME IS READ-ONLY ON COMPUTE NODES. Anything caching there kills the job minutes in, and a login
# smoke can never reproduce it.
export TORCH_HOME=$SCRATCH/.cache/torch HF_HOME=$SCRATCH/.cache/huggingface
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache/xdg}
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"
export WANDB_MODE=offline PYTHONUNBUFFERED=1
# PYTHONHASHSEED IS LOAD-BEARING FOR REPRODUCIBILITY, NOT HYGIENE. --seed alone does NOT reproduce a
# sample: per-process set/dict iteration order feeds action-space construction, so an identical RNG
# draw over a differently-ordered action list picks a different action. Measured: 377 vs 387 routes
# at the same seed; with this set, 730/730 byte-identical.
export PYTHONHASHSEED=0

module load cuda/11.8.0 2>/dev/null || true
conda activate "$CONDA_ENV"

# HARD GUARD: refuse to run in the wrong conda env. `sbatch --export=ALL` inherits the submitting
# shell's PATH, and an entry pointing at another env's bin shadows the one just activated. That is
# worse than a crash: scent_worker does `import rgfn`, which in OUR rgfn env silently resolves to the
# WRONG package (our RGFN, not SCENT's fork) and produces plausible numbers from the wrong model.
ENV_PREFIX="/home/markymoo/miniconda3/envs/$CONDA_ENV"
export PATH="$ENV_PREFIX/bin:$PATH"
PY_REAL="$(command -v python || true)"
case "$PY_REAL" in
    "$ENV_PREFIX"/*) : ;;
    *) echo "ERROR: python resolved to '${PY_REAL:-<none>}', not '$ENV_PREFIX/bin/python'."
       echo "       The inherited PATH is shadowing '$CONDA_ENV'; refusing rather than importing"
       echo "       the wrong package. Resubmit with 'sbatch --export=NONE,...'."
       exit 1 ;;
esac
# dgl/graphbolt need torch's bundled CUDA libs (the ~/bin/rgfn-smoke-env.sh trick, applied to
# whichever env this cell's worker runs in).
export LD_LIBRARY_PATH="$(ls -d "$ENV_PREFIX"/lib/python*/site-packages/nvidia/*/lib 2>/dev/null | paste -sd:):${LD_LIBRARY_PATH:-}"
echo "host=$(hostname)  python=$(which python)"; nvidia-smi -L 2>/dev/null | head -1 || true

mkdir -p "$SAMPLE_DIR" "$ENUM_DIR"

# Auto-skip a finished stage. Sampling is ~20-40 min and enumeration is hours, so an accidental redo
# is the most expensive mistake available here. RESUME=0 forces a clean re-run.
if [ "$RESUME" = 1 ] && [ "$STAGE" = all ] && [ -s "$SAMPLE_DIR/records.csv" ]; then
    echo "=== [$CELL_TAG] SKIP sample -- $SAMPLE_DIR/records.csv exists. RESUME=0 to redo. ==="
    STAGE=enum
fi

# ---- 1. sample -------------------------------------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = sample ]; then
    echo "=== [$CELL_TAG] SAMPLE ($N_TRAJ traj) -> $SAMPLE_DIR ==="
    python "$WORKER" --mode sample \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" \
        --reward-name "$REWARD_NAME" --n-trajectories "$N_TRAJ" --batch-size "$SAMPLE_BATCH" \
        --device "$DEVICE" --run-dir "$TRAIN_DIR" --out-dir "$SAMPLE_DIR" \
        || { echo "ERROR: sample stage failed"; exit 1; }
    # The route contract writes route_status.json at sample time; surface it immediately rather
    # than letting an empty routes.json reach the competitor arm months later, where it would make
    # SPARROW price the EMPTY library as trivially Optimal at zero cost.
    if [ "$ROUTE_BEARING" = true ] && [ ! -s "$SAMPLE_DIR/routes.json" ]; then
        echo "ERROR: $CELL_TAG is route-bearing but wrote no routes.json -- unrecoverable without"
        echo "       a re-sample, so failing here rather than at analysis time."
        exit 1
    fi
fi

# ---- 2. pick_hubs ----------------------------------------------------------------------------------
# --pool all: rank flow over EVERY observed hub, not just the parents of the top-K candidates by
# reward. The v1 pre-filter was legacy continuity with the original recipe, not a justified choice;
# removing it is free (the flow estimate is read off the record's log-terms -- 0.17 s over 20,874
# hubs) and it removes a confound from the ordering ablation, which otherwise measures "flow orders
# well" on a pool that was itself reward-selected.
DEPTH_ARG=()
[ -n "$MIN_HUB_DEPTH" ] && DEPTH_ARG=(--min-hub-depth "$MIN_HUB_DEPTH")
if [ "$STAGE" = all ] || [ "$STAGE" = enum ]; then
    echo "=== [$CELL_TAG] PICK_HUBS (--pool all, top-$N_HUBS) -> $ENUM_DIR/hubs.csv ==="
    python "$REPO/experiments/lsd_hubs/campaign/pick_hubs.py" \
        --records "$SAMPLE_DIR/records.csv" --out "$ENUM_DIR/hubs.csv" \
        --pool all --order flow_desc --n-hubs "$N_HUBS" \
        --higher-is-better "$HIGHER_IS_BETTER" "${DEPTH_ARG[@]}" \
        || { echo "ERROR: pick_hubs failed"; exit 1; }
fi

# ---- 3. enumerate ----------------------------------------------------------------------------------
if [ "$STAGE" = all ] || [ "$STAGE" = enum ]; then
    ENUM_ARG=(); [ "$ENUM_MAX" -gt 0 ] && ENUM_ARG=(--enum-max-children "$ENUM_MAX")
    echo "=== [$CELL_TAG] ENUMERATE -> $ENUM_DIR ==="
    python "$WORKER" --mode enumerate \
        --config "$CONFIG" --checkpoint "$CHECKPOINT" "${GUIDANCE_ARG[@]}" \
        --reward-name "$REWARD_NAME" --hubs "$ENUM_DIR/hubs.csv" \
        --device "$DEVICE" --run-dir "$TRAIN_DIR" --out-dir "$ENUM_DIR" "${ENUM_ARG[@]}" \
        || { echo "ERROR: enumerate stage failed"; exit 1; }
fi

# ---- 4. record + verify ----------------------------------------------------------------------------
python "$TOOLS/ledger.py" record --stage enumerate --cell "$GEN/$TGT/$SEED" --arm "$ARM" \
    --origin generated --note "pool=all n_hubs=$N_HUBS min_hub_depth=${MIN_HUB_DEPTH:-none}" \
    || echo "WARNING: could not write the provenance row"

echo "=== [$CELL_TAG] campaign-stage verification ==="
python "$TOOLS/verify_cell.py" --cell "$GEN/$TGT/$SEED" --arm "$ARM" --stage campaign
VERIFY_RC=$?

echo "=== [$CELL_TAG] done (verify rc=$VERIFY_RC) ==="
echo "    next: back up this stage, then run the campaign."
exit $VERIFY_RC
