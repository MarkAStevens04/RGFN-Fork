#!/bin/bash
# ARM-A PILOT, depth-mix measurement (CPU only, login-node safe).
#
# WHAT THIS ANSWERS, and what it does NOT. It measures how much of a hub-batching library sits on
# DEPTH-0 hubs -- building blocks a chemist BUYS, priced at zero reactions. That share is currently
# unmeasured and is UNRECOVERABLE for v1: the hub-ordering arms kept only PNGs, not per-step curves.
# `--pool all` raises depth-0 exposure ~4x in the walked prefix, and depth-0 catalogue picking is the
# metric's named degenerate optimum, so the share has to be reported rather than assumed harmless.
#
# IT RUNS ON A 320,000-CALL CHECKPOINT, NOT A 10,000-CALL ONE. So it answers the DEPTH question, not
# the arm-A viability question. Label its outputs accordingly.
#
# THREE ARMS on the SAME cached enumeration (only the deliverable filter differs):
#   base    --min-synth-depth 0   everything, including one-step children of bought blocks
#   d2      --min-synth-depth 2   drops exactly the buy-and-couple molecules
#   d3      --min-synth-depth 3   stricter still
# `--min-synth-depth` filters MOLECULES by fully-nested reaction count from purchasable material;
# a depth-0 hub's one-step child has nested depth 1, so `d2` is the closest available proxy for the
# `--min-hub-depth 1` hub-side filter that pick_hubs.py does not yet have.
#
# Usage: bash experiments/benchmark_v2/pilot/run_depthmix.sh
set -uo pipefail

source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1 || true
# Under SLURM the batch script is COPIED to /var/spool, so BASH_SOURCE points there, not at the
# repo. SLURM_SUBMIT_DIR is where sbatch was invoked; fall back to BASH_SOURCE for a login run.
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/campaign/run_campaign.py ] || { echo "FATAL: not an RGFN-Fork checkout ($REPO)"; exit 1; }

CELL=${CELL:-scent_seh}
E=${E:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16/$CELL}
SNAP=${SNAP:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed42/additional_fragments/fragments_4000.json}
GATE=${GATE:-5.68}
OUT=${OUT:-/scratch/markymoo/rgfn_runs/v2_pilot/depthmix}
PY=/home/markymoo/miniconda3/envs/rgfn/bin/python

mkdir -p "$OUT"
echo "cell=$CELL enum=$E gate=$GATE out=$OUT"

for spec in "base:0" "d2:2" "d3:3"; do
    NAME=${spec%%:*}; MSD=${spec##*:}
    echo ""
    echo "############ arm=$NAME  --min-synth-depth $MSD ############"
    T0=$(date +%s)
    "$PY" experiments/lsd_hubs/campaign/run_campaign.py \
        --analysis-dir "$E/sample" \
        --enum-children "$E/enum/enum_children.json" \
        --snapshot "$SNAP" \
        --reward-threshold "$GATE" --similarity 0.5 --higher-is-better true \
        --budget-reactions 100 --budget-modes 300 \
        --child-policy free_frag --prebuild-k 0 \
        --min-synth-depth "$MSD" \
        --tag "depthmix_${CELL}_${NAME}" --out-dir "$OUT/$NAME"
    echo "[depthmix] arm=$NAME exit=$? wall=$(( $(date +%s) - T0 ))s"
done
echo ""
echo "DONE -> $OUT"
