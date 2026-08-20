#!/bin/bash
#SBATCH --job-name=synformer_fr
#SBATCH --time=06:00:00                          # projection-bound, NOT oracle-bound. See SIZING.
#SBATCH --partition=compute
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# SYNFORMER FIXED-REWARD run — a ROUTE-LESS SMILES baseline, same class as S3-GFN and REINVENT 4.
#
# Trains SynFormer's Mamba SMILES model by RL (Augmented Memory) against OUR frozen surrogate, injected
# as a `glue_surrogate` oracle component, then samples a pool from the TRAINED agent and emits a
# standard candidate dataset with has_route=1 — SynFormer carries routes BY CONSTRUCTION, so it does
# NOT need MultiAiZ; it prices via sparrow_select_frontier.py --route-source external.
#
# SIZING — A DIFFERENT BOTTLENECK FROM THE OTHER ENTRANTS. The frozen reward is milliseconds and the
# GA is trivial; the cost is SynFormer's PROJECTION — a transformer decode at search_width 24 /
# exhaustiveness 64 over ~200 molecules per generation, with a 180 s per-molecule ceiling — plus a
# 4 GB fingerprint index loaded per worker. 6 h is a first estimate with NO measurement behind it:
# take the real per-generation time from the first job and re-size before launching the rest.
#
# The stopping condition is the oracle budget (10,000 DISTINCT molecules, upstream's default). It
# counts distinct molecules, so a converging GA re-scores nothing and drains the budget slowly —
# another reason only the first run's wall-clock is trustworthy.
#
#
# MULTI-CELL CHAINING. `CELLS="seh:42 seh:43 drd2:42"` runs those cells sequentially in ONE job.
# This is queue etiquette, not convenience: the account's QOS caps submitted jobs at 60 and this is a
# SHARED tree with three other agents, so eleven short cells submitted individually would crowd out
# their work for no gain. A cell that fails does not stop the rest; the exit status is non-zero if
# any failed, and each cell prints its own DONE/FAILED line.
# Submit:
#   TARGET=seh SEED=42 sbatch experiments/lsd_hubs/campaign/submit_synformer.sh          # one cell
#   CELLS="seh:43 seh:44 drd2:42 drd2:43 drd2:44" \
#       sbatch experiments/lsd_hubs/campaign/submit_synformer.sh                          # chained
# Smoke first — `-p debug` is 2 h max and starts near-instantly, but its QOS allows
# exactly ONE job at a time across the whole account:
#   TARGET=seh SEED=42 BUDGET=200 N_SAMPLES=50 sbatch -p debug -t 00:30:00 .../submit_synformer.sh

set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"

CELLS=${CELLS:-}
if [ -z "$CELLS" ]; then
    CELLS="${TARGET:?set TARGET (or CELLS=\"seh:42 drd2:43\")}:${SEED:-42}"
fi

# Per-cell overrides (apply to EVERY cell in CELLS; intended for smokes, not campaigns).
BUDGET=${BUDGET:-}                                # empty = use the config's oracle budget
N_SAMPLES=${N_SAMPLES:-}                          # empty = use the config's n_samples
# POPULATION bounds a SMOKE's real cost. The budget does not: the first projection runs over the
# whole starting population before a single molecule is scored, and projection — not the oracle —
# is the expensive stage. A budget=20 smoke at the default population still projects 100 molecules.
POPULATION=${POPULATION:-}
TIME_LIMIT=${TIME_LIMIT:-}                        # per-MOLECULE projection ceiling, seconds
OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/experiments/fixed_reward}

# A PREFIX env on $SCRATCH, not $HOME: /home/markymoo is at ~95G of its 110G quota. Address it with
# `conda run -p`; `-n synformer` will not find it.
ENV_PREFIX=${SYNFORMER_ENV_PREFIX:-/scratch/markymoo/conda_envs/synformer}
[ -d "$ENV_PREFIX" ] || { echo "FATAL: no env at $ENV_PREFIX — run external/setup_synformer.sh on a LOGIN node" >&2; exit 1; }

CLONE=external/synformer
[ -d "$CLONE" ] || { echo "FATAL: no SynFormer clone at $CLONE — run external/setup_synformer.sh" >&2; exit 1; }

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
# SynFormer is imported by path, not installed as a package, so the clone must be importable.
export PYTHONPATH="$(pwd)/$CLONE:$(pwd)${PYTHONPATH:+:$PYTHONPATH}"
# Cap OpenMP. We do not request CPUs on Balam, so torch would otherwise spawn one thread per core on
# the node and contend with everything else on it; the sEH MPNN is tiny and gains nothing from more.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$TORCH_HOME"

# dgl/graphbolt CUDA-11.8 runtime for the INGEST child (the rgfn env imports glue->rgfn->dgl). Safe
# for the synformer parent: cuda 11.8 matches its torch cu118 line exactly.
module load cuda/11.8.0
source /home/markymoo/miniconda3/etc/profile.d/conda.sh

# The ingest child (conda run -n rgfn) needs rgfn's torch-bundled CUDA libs on LD_LIBRARY_PATH, but
# those must NOT be on the synformer parent's path. run_synformer_fixed.py applies this to the ingest
# subprocess ONLY, from this variable.
_RGFN_SP=$(conda run -n rgfn python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null)
_RGFN_NVLIBS=$(find "$_RGFN_SP/nvidia" -name lib -type d 2>/dev/null | paste -sd:)
export RGFN_INGEST_LD_LIBRARY_PATH="${_RGFN_NVLIBS}:${LD_LIBRARY_PATH:-}"

echo "host=$(hostname)"; nvidia-smi -L
echo "CELLS=$CELLS  ENV=$ENV_PREFIX  CLONE=$CLONE @ $(cat $CLONE/COMMIT_PINNED.txt 2>/dev/null)"

FAILED=""
for CELL in $CELLS; do
    TARGET=${CELL%%:*}; SEED=${CELL##*:}
    CFG_C=${CFG:-validation/configs/synformer_${TARGET}_fixed.yaml}
    OUT_DIR="$OUT_ROOT/synformer_${TARGET}/seed${SEED}"
    echo ""
    echo "================ CELL $CELL ================"
    if [ ! -s "$CFG_C" ]; then
        echo "FAILED $CELL — no config at $CFG_C" >&2; FAILED="$FAILED $CELL"; continue
    fi
    mkdir -p "$OUT_DIR"
    echo "CFG=$CFG_C  OUT_DIR=$OUT_DIR"

    ARGS=(--cfg "$CFG_C" --run-dir "$OUT_DIR" --seed "$SEED")
    [ -n "$BUDGET" ]     && ARGS+=(--budget "$BUDGET")
    [ -n "$N_SAMPLES" ]  && ARGS+=(--n-samples "$N_SAMPLES")
    [ -n "$POPULATION" ] && ARGS+=(--population "$POPULATION")
    [ -n "$TIME_LIMIT" ] && ARGS+=(--time-limit "$TIME_LIMIT")

    conda run --no-capture-output -p "$ENV_PREFIX" \
        python validation/generators/synformer/run_synformer_fixed.py "${ARGS[@]}"
    RC=$?

    CANDS="$OUT_DIR/fixed_reward/candidates/candidates.csv"
    if [ "$RC" -eq 0 ] && [ -s "$CANDS" ]; then
                # has_route=1 — SynFormer is the REACTION-AWARE entrant; the "has_route=0" here was a
        # sed leftover from the Saturn copy and would have told a reader the exact opposite of
        # the truth. Read it back from the manifest rather than asserting it.
        HR=$(python -c "import json;print(json.load(open('$OUT_DIR/fixed_reward/candidates/manifest.json'))['has_routes'])" 2>/dev/null || echo "?")
        echo "DONE $CELL -> $OUT_DIR  ($(($(wc -l < "$CANDS") - 1)) candidates, has_routes=$HR)"
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
