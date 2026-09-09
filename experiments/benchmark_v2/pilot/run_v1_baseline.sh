#!/bin/bash
# Price a v1 (320,000-call) cell at the CURRENT gate, on its cached enumeration.
#
# WHY. The arm-A campaigns run at gate 5.68 (the 5%-FPR standard), while the committed v1 matrix was
# priced at 5.0. Comparing across those two would confound the budget effect with a gate change. This
# re-reads the v1 enumeration at 5.68 so the only remaining difference is the budget -- and the
# enumeration itself, which is NOT matched (v1: 200 hubs uncapped; arm A: 64 hubs at enum_max 4000)
# and is stated wherever the ratio is quoted.
#
# CPU only, but run it as a job: login-node imports of torch-geometric get SIGINT'd on this cluster.
#
# Usage: CELL=rgfn_seh sbatch -p debug --gpus-per-node=1 .../run_v1_baseline.sh
set -uo pipefail
source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1 || true
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
cd "$REPO" || { echo "FATAL: cannot cd to '$REPO'"; exit 1; }
[ -f experiments/lsd_hubs/campaign/run_campaign.py ] || { echo "FATAL: not an RGFN-Fork checkout"; exit 1; }

CELL=${CELL:-rgfn_seh}
E=${E:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16/$CELL}
GATE=${GATE:-5.68}
OUT=${OUT:-/scratch/markymoo/rgfn_runs/v2_pilot/v1_baseline/$CELL}
PY=/home/markymoo/miniconda3/envs/rgfn/bin/python
mkdir -p "$OUT"

# SCENT needs its promoted-fragment snapshot to expand nested builds; RGFN has no dynamic library
# and takes none. Passing one it cannot use would be a silent no-op at best.
SNAPARG=()
[ -n "${SNAP:-}" ] && SNAPARG=(--snapshot "$SNAP")

echo "cell=$CELL gate=$GATE enum=$E/enum/enum_children.json"
T0=$(date +%s)
"$PY" experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$E/sample" --enum-children "$E/enum/enum_children.json" "${SNAPARG[@]}" \
    --reward-threshold "$GATE" --similarity 0.5 --higher-is-better true \
    --budget-reactions 100 --budget-modes 300 \
    --child-policy free_frag --prebuild-k 0 \
    --tag "v1_${CELL}_gate${GATE}" --out-dir "$OUT"
echo "[v1-baseline] exit=$? wall=$(( $(date +%s) - T0 ))s -> $OUT"
