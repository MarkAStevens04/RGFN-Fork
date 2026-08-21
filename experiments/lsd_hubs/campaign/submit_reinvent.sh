#!/bin/bash
#SBATCH --job-name=reinvent_fr
#SBATCH --time=04:00:00                          # RL is the cost: 1000 steps x batch 128. See SIZING.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# REINVENT 4 FIXED-REWARD run — a ROUTE-LESS SMILES baseline, same class as S3-GFN (Logs/040, T3.1).
#
# Trains REINVENT's ChEMBL-pretrained SMILES RNN by RL against OUR frozen surrogate (injected through
# the GlueSurrogate plugin component), samples a pool from the TRAINED agent, and emits a standard
# candidate dataset with has_route=0. Routes are recovered downstream by MultiAiZ -> SPARROW, exactly
# as for S3-GFN, via submit_competitor_routes.sh.
#
# ONE SCRIPT, BOTH TARGETS AND ALL SEEDS — pass TARGET and SEED. Keeping the two surrogate targets in
# one script (rather than a per-target copy) is deliberate: every knob that must NOT drift between
# cells lives in the YAML, and the only things this script varies are the two that must.
#
# SIZING. 1000 RL steps x batch 128 = ~128,000 frozen-reward calls. The sEH MPNN scores a 128-SMILES
# batch on CPU in tens of ms, so wall-clock is dominated by REINVENT's own RNN sampling on the GPU.
# 4 h is a generous envelope; check the first job's actual time and trim before launching the rest.
#
#
# MULTI-CELL CHAINING. `CELLS="seh:42 seh:43 drd2:42"` runs those cells sequentially in ONE job.
# This is queue etiquette, not convenience: the account's QOS caps submitted jobs at 60 and this is a
# SHARED tree with three other agents, so eleven short cells submitted individually would crowd out
# their work for no gain. A cell that fails does not stop the rest; the exit status is non-zero if
# any failed, and each cell prints its own DONE/FAILED line.
# Submit:
#   TARGET=seh SEED=42 sbatch experiments/lsd_hubs/campaign/submit_reinvent.sh          # one cell
#   CELLS="seh:43 seh:44 drd2:42 drd2:43 drd2:44" \
#       sbatch experiments/lsd_hubs/campaign/submit_reinvent.sh                          # chained
# Smoke first — `-p debug` is 2 h max and starts near-instantly, but its QOS allows
# exactly ONE job at a time across the whole account:
#   TARGET=seh SEED=42 STEPS=5 N_SAMPLES=50 sbatch -p debug -t 00:30:00 .../submit_reinvent.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CELLS=${CELLS:-}
if [ -z "$CELLS" ]; then
    CELLS="${TARGET:?set TARGET (or CELLS=\"seh:42 drd2:43\")}:${SEED:-42}"
fi

# Per-cell overrides (apply to EVERY cell in CELLS; intended for smokes, not campaigns).
STEPS=${STEPS:-}                                  # empty = use the config's max_steps
N_SAMPLES=${N_SAMPLES:-}                          # empty = use the config's n_samples
OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}

# A PREFIX env on $SCRATCH, not $HOME: /home/markymoo is at ~95G of its 110G quota. Address it with
# `conda run -p`; `-n reinvent4` will not find it.
ENV_PREFIX=${REINVENT_ENV_PREFIX:-/scratch/markymoo/conda_envs/reinvent4}
[ -d "$ENV_PREFIX" ] || { echo "FATAL: no env at $ENV_PREFIX — run external/setup_reinvent.sh on a LOGIN node" >&2; exit 1; }

export PYTHONUNBUFFERED=1
export TORCH_HOME=$SCRATCH/.cache/torch
# NOTHING MAY WRITE TO $HOME: it is READ-ONLY on Balam compute nodes. Every one of these libraries
# defaults to a cache under $HOME and dies with PermissionError the first time it needs it — and a
# LOGIN-node smoke will never reproduce it, because $HOME is writable there. Found the hard way:
# Mamba's Triton JIT tried $HOME/.triton/cache 54 s into a debug job (job 73617).
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$SCRATCH/.cache/triton}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$SCRATCH/.cache/matplotlib}
export HF_HOME=${HF_HOME:-$SCRATCH/.cache/huggingface}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$SCRATCH/.cache/xdg}
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$HF_HOME" "$XDG_CACHE_HOME"
# Cap OpenMP. We do not request CPUs on Balam, so torch would otherwise spawn one thread per core on
# the node and contend with everything else on it; the sEH MPNN is tiny and gains nothing from more.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$TORCH_HOME"

# dgl/graphbolt CUDA-11.8 runtime for the INGEST child (the rgfn env imports glue->rgfn->dgl). Safe
# for the reinvent4 parent: cuda 11.8 exposes .11 sonames while torch cu121 loads its bundled .12
# libs via RUNPATH. Same contract as submit_s3gfn_seh.sh.
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# The ingest child (conda run -n rgfn) needs rgfn's torch-bundled CUDA libs on LD_LIBRARY_PATH, but
# those must NOT be on the reinvent4 parent's path. run_reinvent_fixed.py applies this to the ingest
# subprocess ONLY, from this variable.
_RGFN_SP=$(conda run -n rgfn python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
_RGFN_NVLIBS=$(find "$_RGFN_SP/nvidia" -name lib -type d 2>/dev/null | paste -sd:)
export RGFN_INGEST_LD_LIBRARY_PATH="${_RGFN_NVLIBS}:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "CELLS=$CELLS  ENV=$ENV_PREFIX"

FAILED=""
for CELL in $CELLS; do
    TARGET=${CELL%%:*}; SEED=${CELL##*:}
    CFG_C=${CFG:-validation/configs/reinvent_${TARGET}_fixed.yaml}
    # OUT_DIR is derived from TARGET/SEED, NOT from the caller's environment. A `RUN_DIR=...` on
    # the sbatch line is silently ignored here, which on 2026-08-21 let a 6-step debug smoke land on
    # top of the completed seed-42 run and truncate its 127,997-row history to 384 rows. Override the
    # destination with OUT_ROOT, which IS honoured.
    OUT_DIR="$OUT_ROOT/reinvent_${TARGET}/seed${SEED}"
    if [ -n "${RUN_DIR:-}" ]; then
        echo "FATAL: RUN_DIR is not supported by this script and would be ignored." >&2
        echo "  You probably meant:  OUT_ROOT=<dir> ... sbatch $0" >&2
        exit 1
    fi
    # REFUSE TO CLOBBER A COMPLETED RUN. A finished cell owns a candidates.csv; re-running onto it
    # destroys the training history that produced every published number for that cell. Set
    # FORCE_OVERWRITE=1 only when the intent really is to replace it.
    if [ -s "$OUT_DIR/fixed_reward/candidates/candidates.csv" ] && [ "${FORCE_OVERWRITE:-0}" != "1" ]; then
        echo "FATAL: $OUT_DIR already holds a completed run" >&2
        echo "  ($(($(wc -l < "$OUT_DIR/fixed_reward/candidates/candidates.csv") - 1)) candidates)." >&2
        echo "  Use OUT_ROOT=<somewhere-else>, or FORCE_OVERWRITE=1 to replace it deliberately." >&2
        exit 1
    fi
    echo ""
    echo "================ CELL $CELL ================"
    if [ ! -s "$CFG_C" ]; then
        echo "FAILED $CELL — no config at $CFG_C" >&2; FAILED="$FAILED $CELL"; continue
    fi
    mkdir -p "$OUT_DIR"
    echo "CFG=$CFG_C  OUT_DIR=$OUT_DIR"

    ARGS=(--cfg "$CFG_C" --run-dir "$OUT_DIR" --seed "$SEED")
    [ -n "$STEPS" ]     && ARGS+=(--max-steps "$STEPS")
    [ -n "$N_SAMPLES" ] && ARGS+=(--n-samples "$N_SAMPLES")

    conda run --no-capture-output -p "$ENV_PREFIX" \
        python validation/generators/reinvent/run_reinvent_fixed.py "${ARGS[@]}"
    RC=$?

    CANDS="$OUT_DIR/fixed_reward/candidates/candidates.csv"
    if [ "$RC" -eq 0 ] && [ -s "$CANDS" ]; then
        echo "DONE $CELL -> $OUT_DIR  ($(($(wc -l < "$CANDS") - 1)) candidates, has_route=0)"
    else
        # Never let an infrastructure failure be reported as a scientific limit (Logs/059).
        echo "FAILED $CELL rc=$RC — candidates.csv missing or empty." >&2
        FAILED="$FAILED $CELL"
    fi
done

echo ""
if [ -n "$FAILED" ]; then echo "FAILED CELLS:$FAILED" >&2; exit 1; fi
echo "ALL CELLS DONE: $CELLS"
echo "Next per cell: mode-saturation pre-flight, THEN submit_competitor_routes.sh"
echo "  (MultiAiZ is ~2.25 h/pool at N=500 — never spend it on a pool that cannot reach 100 modes)."
