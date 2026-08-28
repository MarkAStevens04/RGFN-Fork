#!/bin/bash
#SBATCH --job-name=stage2
#SBATCH --time=12:00:00
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# STAGE 2 -- upsample a trained generator until its pool holds N diverse modes. See
# experiments/lsd_hubs/campaign/upsample_to_modes.py for why this stage exists.
#
# THE ENVIRONMENT BLOCK BELOW MIRRORS experiments/fixed_reward/scale5k/submit_baseline.sh AND MUST
# STAY IN STEP WITH IT. Two runs were lost on 2026-08-28 to a hand-rolled minimal launcher missing
# something that script sets:
#   * HF_HOME pointed at an empty .cache/hf, so S3-GFN's trainer tried to reach huggingface.co from
#     a compute node and died in its constructor before sampling a single molecule;
#   * no `module load cuda/11.8.0`, so the rgfn-env ingest step died with "Cannot load Graphbolt C++
#     library" AFTER sampling had already produced 4,000 molecules and scored them.
# Both were invisible until minutes in, which is why PREFLIGHT below checks them before the job
# commits to anything expensive. A launcher that fails in 5 seconds beats one that fails in 5
# minutes and looks like a generator property.
#
# Usage -- CELLS is a space-separated list of GENERATOR:TARGET:SEED triples:
#   CELLS="s3gfn:seh:42 saturn:seh:42" sbatch experiments/lsd_hubs/campaign/submit_stage2_upsample.sh
#
# synformer is NOT a valid generator here: its candidates are a slice of an accumulated GA
# population, so there is nothing to draw more of. Those cells stay pool-limited by construction.
#
# fraggfn IS valid (added 2026-08-28): it is a fragment-based GFlowNet with a real sampler, so
# --n-samples on its trained checkpoint draws more molecules the same way reinvent/saturn/s3gfn do.
# Its config path is special-cased below -- read the comment there before changing it.

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CELLS=${CELLS:?set CELLS to a list of GENERATOR:TARGET:SEED triples}
TARGET_MODES=${TARGET_MODES:-500}
CUTOFF=${CUTOFF:-0.5}                 # the campaign's tau. NOT the helper's 0.7 default.
ROUND=${ROUND:-4000}
STALL=${STALL:-10}
MAX_SCORED=${MAX_SCORED:-0}           # 0 = per-target default (50k surrogate / 20k docking)
OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/stage2}
FR_ROOT=${FR_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}

module load cuda/11.8.0 2>/dev/null || true
export TRITON_CACHE_DIR=$SCRATCH/.cache/triton MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export HF_HOME=$SCRATCH/.cache/huggingface TORCH_HOME=$SCRATCH/.cache/torch
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1     # compute nodes have no internet
export XDG_CACHE_HOME=$SCRATCH/.cache PYTHONUNBUFFERED=1 WANDB_MODE=offline
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$TORCH_HOME" "$OUT_ROOT"

echo "host=$(hostname)  CELLS=$CELLS"; nvidia-smi -L || true
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# ---- PREFLIGHT: fail in seconds, not minutes ----------------------------------------------------
echo "=== preflight ==="
[ -d "$HF_HOME/hub" ] || { echo "FATAL: no HF cache at $HF_HOME/hub" >&2; exit 2; }
conda run --no-capture-output -n rgfn python -c "
import dgl  # noqa: F401  -- the Graphbolt load that killed a run after it had already sampled
print('  rgfn env + dgl/graphbolt: OK')" || {
    echo "FATAL: rgfn env cannot import dgl -- is cuda/11.8.0 loaded?" >&2; exit 2; }
echo "  HF_HOME=$HF_HOME (offline)"

FAILED=""
for CELL in $CELLS; do
    GEN=${CELL%%:*}; REST=${CELL#*:}; TGT=${REST%%:*}; SD=${REST##*:}
    CFG_OVERRIDE=""
    case "$GEN" in
      saturn|tango) ENVNAME=saturn; RUNNER=validation/generators/saturn/run_saturn_fixed.py ;;
      reinvent)     ENVNAME=reinvent; RUNNER=validation/generators/reinvent/run_reinvent_fixed.py ;;
      s3gfn)        ENVNAME=s3gfn; RUNNER=validation/generators/s3gfn/run_s3gfn_fixed.py ;;
      fraggfn)
        ENVNAME=fraggfn; RUNNER=validation/generators/fraggfn/run_fraggfn_fixed.py
        # FragGFN is the ONE generator whose config does not follow ${GEN}_${TGT}_fixed.yaml, and
        # the convention does not merely miss -- it resolves to a file that EXISTS and is WRONG.
        # validation/configs/fraggfn_seh_fixed.yaml is the OLD 5,000-step build; the cells in
        # fraggfn_<sys>/ were trained on the *_norm.yaml configs at 157 steps (a327c3a). The
        # runner's resume guard compares the CONFIG's n_train_steps against the checkpoint's, so
        # the stale config would read 157 < 5000, silently re-train 4,843 steps, and re-spend the
        # training budget inside what is supposed to be a SAMPLING stage -- corrupting the
        # oracle-call accounting in FragGFN's favour. The `[ -s "$CFG" ]` guard cannot catch this
        # because the wrong file is present. Name the trained-on configs explicitly.
        case "$TGT" in
          seh)  CFG_OVERRIDE=validation/configs/fraggfn_seh_fixed_norm.yaml ;;
          drd2) CFG_OVERRIDE=validation/configs/fraggfn_drd2_fixed_norm.yaml ;;
          clpp) CFG_OVERRIDE=validation/configs/fraggfn_clpp_docking_fixed_norm.yaml ;;
          *) echo "FAILED $CELL — no normalized-budget fraggfn config for target $TGT" >&2
             FAILED="$FAILED $CELL"; continue ;;
        esac ;;
      synformer)    echo "SKIP $CELL — a GA population cannot be upsampled; pool-limited by construction"; continue ;;
      *) echo "FAILED $CELL — unknown generator" >&2; FAILED="$FAILED $CELL"; continue ;;
    esac
    CFG=${CFG_OVERRIDE:-validation/configs/${GEN}_${TGT}_fixed.yaml}
    RUN_DIR="$FR_ROOT/${GEN}_${TGT}/seed${SD}"
    OUT="$OUT_ROOT/${GEN}_${TGT}_seed${SD}"
    echo ""; echo "############ STAGE2 $CELL  ($(date '+%F %H:%M')) ############"
    [ -s "$CFG" ] || { echo "FAILED $CELL — no config at $CFG" >&2; FAILED="$FAILED $CELL"; continue; }
    [ -d "$RUN_DIR" ] || { echo "FAILED $CELL — no run dir at $RUN_DIR" >&2; FAILED="$FAILED $CELL"; continue; }

    EXTRA=(); [ "$MAX_SCORED" != "0" ] && EXTRA+=(--max-scored "$MAX_SCORED")
    conda run --no-capture-output -n rgfn python \
        experiments/lsd_hubs/campaign/upsample_to_modes.py \
        --generator "$GEN" --target "$TGT" --seed "$SD" \
        --cutoff "$CUTOFF" --target-modes "$TARGET_MODES" \
        --round-size "$ROUND" --stall-modes "$STALL" "${EXTRA[@]}" \
        --run-dir "$RUN_DIR" --cfg "$CFG" --env "$ENVNAME" --runner "$RUNNER" --out "$OUT"
    rc=$?
    if [ "$rc" -eq 0 ]; then echo "STAGE2 $CELL OK -> $OUT"; else echo "STAGE2 $CELL FAILED rc=$rc" >&2; FAILED="$FAILED $CELL"; fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED CELLS:$FAILED" >&2; exit 1; fi
echo "ALL CELLS OK: $CELLS"
