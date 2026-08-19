#!/bin/bash
#SBATCH --job-name=saturn_fr
#SBATCH --time=02:00:00                          # oracle-budget bound, not step bound. See SIZING.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SATURN FIXED-REWARD run — a ROUTE-LESS SMILES baseline, same class as S3-GFN and REINVENT 4.
#
# Trains Saturn's Mamba SMILES model by RL (Augmented Memory) against OUR frozen surrogate, injected
# as a `glue_surrogate` oracle component, then samples a pool from the TRAINED agent and emits a
# standard candidate dataset with has_route=0. Routes are recovered downstream by MultiAiZ -> SPARROW
# via submit_competitor_routes.sh, exactly as for the other route-less entrants.
#
# SIZING. Saturn stops on its ORACLE BUDGET (10,000 calls), not a step count, so wall-clock depends
# on how fast it burns the budget: batch 64 with `allow_oracle_repeats: false` means ~156+ RL steps
# minimum, more as duplicates are rejected. The frozen reward itself is milliseconds. 2 h is a first
# estimate — check the first job's real time and trim, and note this is well inside `-p debug`'s 2 h
# limit if the queue on `compute` is congested.
#
#
# MULTI-CELL CHAINING. `CELLS="seh:42 seh:43 drd2:42"` runs those cells sequentially in ONE job.
# This is queue etiquette, not convenience: the account's QOS caps submitted jobs at 60 and this is a
# SHARED tree with three other agents, so eleven short cells submitted individually would crowd out
# their work for no gain. A cell that fails does not stop the rest; the exit status is non-zero if
# any failed, and each cell prints its own DONE/FAILED line.
# Submit:
#   TARGET=seh SEED=42 sbatch experiments/lsd_hubs/campaign/submit_saturn.sh          # one cell
#   CELLS="seh:43 seh:44 drd2:42 drd2:43 drd2:44" \
#       sbatch experiments/lsd_hubs/campaign/submit_saturn.sh                          # chained
# Smoke first — `-p debug` is 2 h max and starts near-instantly, but its QOS allows
# exactly ONE job at a time across the whole account:
#   TARGET=seh SEED=42 BUDGET=200 N_SAMPLES=50 sbatch -p debug -t 00:30:00 .../submit_saturn.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CELLS=${CELLS:-}
if [ -z "$CELLS" ]; then
    CELLS="${TARGET:?set TARGET (or CELLS=\"seh:42 drd2:43\")}:${SEED:-42}"
fi

# Per-cell overrides (apply to EVERY cell in CELLS; intended for smokes, not campaigns).
BUDGET=${BUDGET:-}                                # empty = use the config's oracle budget
N_SAMPLES=${N_SAMPLES:-}                          # empty = use the config's n_samples
OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}

# A PREFIX env on $SCRATCH, not $HOME: /home/markymoo is at ~95G of its 110G quota. Address it with
# `conda run -p`; `-n saturn` will not find it.
ENV_PREFIX=${SATURN_ENV_PREFIX:-/scratch/markymoo/conda_envs/saturn}
[ -d "$ENV_PREFIX" ] || { echo "FATAL: no env at $ENV_PREFIX — run external/setup_saturn.sh on a LOGIN node" >&2; exit 1; }

CLONE=external/saturn
[ -d "$CLONE" ] || { echo "FATAL: no Saturn clone at $CLONE — run external/setup_saturn.sh" >&2; exit 1; }

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
# Saturn is imported by path, not installed as a package, so the clone must be importable.
export PYTHONPATH="$(pwd)/$CLONE:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
# Cap OpenMP. We do not request CPUs on Balam, so torch would otherwise spawn one thread per core on
# the node and contend with everything else on it; the sEH MPNN is tiny and gains nothing from more.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$TORCH_HOME"

# dgl/graphbolt CUDA-11.8 runtime for the INGEST child (the rgfn env imports glue->rgfn->dgl). Safe
# for the saturn parent: cuda 11.8 matches its torch cu118 line exactly.
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# The ingest child (conda run -n rgfn) needs rgfn's torch-bundled CUDA libs on LD_LIBRARY_PATH, but
# those must NOT be on the saturn parent's path. run_saturn_fixed.py applies this to the ingest
# subprocess ONLY, from this variable.
_RGFN_SP=$(conda run -n rgfn python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
_RGFN_NVLIBS=$(find "$_RGFN_SP/nvidia" -name lib -type d 2>/dev/null | paste -sd:)
export RGFN_INGEST_LD_LIBRARY_PATH="${_RGFN_NVLIBS}:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "CELLS=$CELLS  ENV=$ENV_PREFIX  CLONE=$CLONE @ $(cat $CLONE/COMMIT_PINNED.txt 2>/dev/null)"

FAILED=""
for CELL in $CELLS; do
    TARGET=${CELL%%:*}; SEED=${CELL##*:}
    # VARIANT picks a config family and still varies the target per cell; a bare CFG cannot,
    # since it would pin one file across a mixed seh+drd2 chain. VARIANT=_beam -> the
    # Beam-Enumeration arm. CFG, if set, still wins for a single-cell run.
    CFG_C=${CFG:-validation/configs/saturn_${TARGET}${VARIANT:-}_fixed.yaml}
    OUT_DIR="$OUT_ROOT/saturn${VARIANT:-}_${TARGET}/seed${SEED}"
    echo ""
    echo "================ CELL $CELL ================"
    if [ ! -s "$CFG_C" ]; then
        echo "FAILED $CELL — no config at $CFG_C" >&2; FAILED="$FAILED $CELL"; continue
    fi
    mkdir -p "$OUT_DIR"
    echo "CFG=$CFG_C  OUT_DIR=$OUT_DIR"

    ARGS=(--cfg "$CFG_C" --run-dir "$OUT_DIR" --seed "$SEED")
    [ -n "$BUDGET" ]    && ARGS+=(--budget "$BUDGET")
    [ -n "$N_SAMPLES" ] && ARGS+=(--n-samples "$N_SAMPLES")

    conda run --no-capture-output -p "$ENV_PREFIX" \
        python validation/generators/saturn/run_saturn_fixed.py "${ARGS[@]}"
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
