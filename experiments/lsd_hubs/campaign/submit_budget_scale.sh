#!/bin/bash
#SBATCH --job-name=budget_scale
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#SBATCH --gpus-per-node=1
#SBATCH --output=/scratch/markymoo/rgfn_runs/%x-%j.out
#SBATCH --error=/scratch/markymoo/rgfn_runs/%x-%j.err

# DOES A LARGER BUDGET EXHAUST THE COMPETITOR'S CHEAP POOL?
#
# THE HYPOTHESIS (the user's, and it needs no artificial constraint at all). The competitor sits at
# exactly 1.00 reaction per molecule because ~36-48% of its routed pool is ONE reaction from
# purchasable material, and at R=100 it only needs ~100 of them. Measured cheap supply:
#     seed 42: 125 one-step of 351 routed      seed 43: 182 of 384      seed 44: 191 of 394
# So the floor HOLDS at R=100 (supply > demand) and must BREAK somewhere past R=125-191, after which
# every further molecule costs 2+ reactions. Buying the entire routed pool costs 845-1051 reactions
# for 351-394 molecules, i.e. 2.2-2.7 rxn/mode at full exhaustion. If that is right, our advantage
# appears at large budgets WITHOUT any depth or catalogue-distinctness filter -- which answers the
# "you optimise for a case nobody cares about" objection far better than any constraint we impose.
#
# TWO REASONS THIS IS NOT A TRIM. (1) CLAUDE.md: trimming a solved SPARROW selection is not exact --
# it optimises jointly over shared intermediates, so a trim is feasible but suboptimal, a lower bound
# that flatters us. Each budget gets its own MILP. (2) OUR curves were generated with
# --budget-modes 300 and stop there, so reading them at R=300+ truncates US, not them. Phase 2
# re-runs our side with a 2000-mode budget so the comparison is not capped on our side.
#
# MILP TRACTABILITY: R=200/300 have previously hit an 1800 s wall where R=100 solved in seconds
# (CLAUDE.md). --max-seconds is generous here and `time_capped` MUST be read on every row -- a capped
# row is a LOWER bound on the competitor, i.e. it flatters us.
set -uo pipefail
cd "$HOME/projects/RGFN_Fork/RGFN-Fork"
export XDG_CACHE_HOME=$SCRATCH/.cache MPLCONFIGDIR=$SCRATCH/.cache/matplotlib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"
BUDGETS=${BUDGETS:-"50,100,150,200,300,400,500"}
TARGET=${TARGET:-seh}
SEEDS=${SEEDS:-"42 43 44"}
# Their pool dir varies by target: sEH uses stage2_pruned, DRD2 seeds 43/44 were REBUILT as
# stage2fix_pruned after the NaN gate fix, and ClpP's stage2 pools are the post-fix true-size ones
# (N115-N143) so the larger non-stage2 _pruned pools carry more routed molecules there.
POOL_GLOB=${POOL_GLOB:-"stage2_pruned_N*"}
# Gate from the one source of truth; a hand-picked bar or flipped direction is the silent failure.
read -r GATE HIB <<EOF
$(source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1; python - "$TARGET" <<'PYG'
import sys; sys.path.insert(0,"experiments/lsd_hubs/matrix16")
from targets import TARGETS
t=TARGETS[sys.argv[1]]; print(t.mode_reward_threshold, "true" if t.higher_is_better else "false")
PYG
)
EOF
[ -n "${GATE:-}" ] || { echo "FATAL: no gate for $TARGET" >&2; exit 1; }
P=$SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools
R=$SCRATCH/rgfn_runs/lsdflow_sparrow/results
source ~/bin/rgfn-smoke-env.sh
RC=0

echo "################ PHASE 1: competitor, re-solved at every budget ################"
for SEED in $SEEDS; do
  D=$(ls -d $P/s3gfn_${TARGET}_seed${SEED}_${POOL_GLOB} 2>/dev/null | head -1)
  [ -n "$D" ] && [ -s "$D/multiaiz_routes.json" ] || { echo "skip seed $SEED (no cached routes)"; continue; }
  echo "  pool: $(basename "$D")"
  echo ""; echo "=== seed $SEED ==="
  python experiments/lsd_hubs/campaign/sparrow_select_frontier.py \
      --routes "$D/multiaiz_routes.json" --pool "$D/pool_scores.csv" --route-source multiaiz \
      --gate "$GATE" --higher-is-better "$HIB" --cutoff 0.5 --budgets "$BUDGETS" \
      --max-seconds 5400 \
      --out-dir "$R/budget_scale_s3gfn_${TARGET}_seed${SEED}" \
      --tag "budget_scale_s3gfn_${TARGET}_seed${SEED}" || RC=1
done

echo ""; echo "################ PHASE 2: ours, mode budget raised so we are not the truncated side ################"
for SEED in $SEEDS; do
  SUF=""; [ "$SEED" != "42" ] && SUF="_seed${SEED}"
  S=$SCRATCH/rgfn_runs/lsdflow/matrix16${SUF}/scent_${TARGET}
  SNAP=$(ls $SCRATCH/rgfn_runs/experiments/fixed_reward/scent_${TARGET}*/seed${SEED}/additional_fragments/fragments_*.json 2>/dev/null | sort -t_ -k2 -n | tail -1)
  [ -s "$S/enum/enum_children.json" ] || { echo "skip seed $SEED"; continue; }
  echo ""; echo "=== seed $SEED ==="
  python experiments/lsd_hubs/campaign/run_campaign.py \
      --analysis-dir "$S/sample" --enum-children "$S/enum/enum_children.json" \
      ${SNAP:+--snapshot "$SNAP"} \
      --reward-threshold "$GATE" --higher-is-better "$HIB" --similarity 0.5 \
      --budget-reactions "${OURS_R:-500}" --budget-modes "${OURS_MODES:-2000}" --child-policy free_frag \
      --tag "budget_scale_ours_${TARGET}_s${SEED}" \
      --out-dir "$R/budget_scale_ours_${TARGET}_seed${SEED}" || RC=1
done
echo ""; echo "BUDGET-SCALE rc=$RC"; exit "$RC"
