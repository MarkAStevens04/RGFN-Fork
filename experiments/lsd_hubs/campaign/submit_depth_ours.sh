#!/bin/bash
#SBATCH --job-name=depth_ours
#SBATCH --partition=compute
#SBATCH --time=4:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err
# Our side of the SYNTHETIC-DEPTH sweep on a target other than sEH, so the regime claim rests on
# more than one system. Depth here is counted from OUR 418 blocks (hub.depth + 1 + promoted builds);
# the competitor's is counted from ZINC, which is why the two axes must not share a plot.
# Needs rgfn-smoke-env (run_campaign imports glue -> dgl); see job 75746.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
TARGET=${TARGET:?set TARGET (drd2|clpp)}
SEED=${SEED:-42}
DEPTHS=${DEPTHS:-"1 2 3 4 5"}
SUF=""; [ "$SEED" != "42" ] && SUF="_seed${SEED}"
S=$SCRATCH/rgfn_runs/lsdflow/matrix16${SUF}/scent_${TARGET}
# Gate AND direction from the one source of truth; a hand-picked bar or a flipped direction is the
# silent failure this table exists to prevent (matrix16/targets.py).
read -r GATE HIB <<EOF
$(source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1; python - "$TARGET" <<'PYG'
import sys
sys.path.insert(0, "experiments/lsd_hubs/matrix16")
from targets import TARGETS
t = TARGETS[sys.argv[1]]
print(t.mode_reward_threshold, "true" if t.higher_is_better else "false")
PYG
)
EOF
[ -n "${GATE:-}" ] || { echo "FATAL: could not resolve gate for $TARGET" >&2; exit 1; }
SNAP=$(ls $SCRATCH/rgfn_runs/experiments/fixed_reward/scent_${TARGET}*/seed${SEED}/additional_fragments/fragments_*.json 2>/dev/null | sort -t_ -k2 -n | tail -1)
echo "host=$(hostname) target=$TARGET seed=$SEED gate=$GATE hib=$HIB snap=${SNAP:-none}"
[ -s "$S/enum/enum_children.json" ] || { echo "FATAL: no enumeration at $S" >&2; exit 1; }
source ~/bin/rgfn-smoke-env.sh
RC=0
for D in $DEPTHS; do
  echo ""; echo "############ ours: $TARGET seed $SEED min-synth-depth $D ############"
  python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$S/sample" --enum-children "$S/enum/enum_children.json" \
    ${SNAP:+--snapshot "$SNAP"} \
    --reward-threshold "$GATE" --higher-is-better "$HIB" --similarity 0.5 \
    --budget-reactions 100 --budget-modes 300 --child-policy free_frag \
    --min-synth-depth "$D" --tag "depth_ours_${TARGET}_s${SEED}_d${D}" \
    --out-dir "$SCRATCH/rgfn_runs/lsdflow_sparrow/results/depth_ours_${TARGET}_seed${SEED}/d${D}" || RC=1
done
echo "DEPTH-OURS rc=$RC"; exit "$RC"
