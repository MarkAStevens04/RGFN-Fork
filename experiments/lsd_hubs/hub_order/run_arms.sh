#!/bin/bash
# CPU analysis phase of the hub-ordering ablation (Logs/053): run the campaign + the diversity sweep
# once per arm, over the SAME candidate pool, cost model and hero acquisition config — so the only
# thing that differs between arms is which hubs hub-batching walks, and in what order.
#
# Every arm reuses one cached enumeration set (merge_enum.py assembled each arm's 200 hubs in its own
# walk order), so this is pure CPU: no GPU, no re-scoring. best-candidate is recomputed per arm but
# is identical by construction — it never touches hub data — which doubles as a consistency check.
#
#   bash experiments/lsd_hubs/hub_order/run_arms.sh
#
# Env: OUT_ROOT, ANALYSIS, SNAPSHOT, THR (7.0), SIM (0.5), BUDGET_MODES (300), BUDGET_RXN (100),
#      CHILD_POLICY (free_frag), PREBUILD_K (20), TAG_SUFFIX, ARMS (space-separated subset).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=${REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
cd "$REPO" || exit 2

OUT_ROOT=${OUT_ROOT:-$SCRATCH/rgfn_runs/lsdflow/hub_order}
ANALYSIS=${ANALYSIS:-/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189}
CANON=${CANON:-/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363}
SNAPSHOT=${SNAPSHOT:-/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json}
THR=${THR:-7.0}
SIM=${SIM:-0.5}
BUDGET_MODES=${BUDGET_MODES:-300}
BUDGET_RXN=${BUDGET_RXN:-100}
CHILD_POLICY=${CHILD_POLICY:-free_frag}
PREBUILD_K=${PREBUILD_K:-20}
TAG_SUFFIX=${TAG_SUFFIX:-}
RESULTS="$SCRIPT_DIR/results"

# arm name -> enumeration dir. `incumbent` is the existing top-1000-candidates -> top-200-by-flow
# enumeration (Logs/029/031); the rest were assembled by merge_enum.py.
declare -A ENUMDIR=( [incumbent]="$CANON" )
for d in "$OUT_ROOT"/merged/*/; do
  [ -f "$d/enum_children.json" ] || continue
  ENUMDIR[$(basename "$d")]="${d%/}"
done

ARMS=${ARMS:-incumbent flow_top flow_bottom random cand_order_fixedset cand_order flow_bottom_600}

echo "[run_arms] thr=$THR sim=$SIM modes=$BUDGET_MODES child_policy=$CHILD_POLICY prebuild_k=$PREBUILD_K"
for arm in $ARMS; do
  dir=${ENUMDIR[$arm]:-}
  if [ -z "$dir" ]; then
    echo "[run_arms] SKIP $arm (no enumeration under $OUT_ROOT/merged/$arm)"; continue
  fi
  TAG="hubord_${arm}${TAG_SUFFIX}"
  OUT="$RESULTS/$TAG"
  mkdir -p "$OUT"
  # Shared inputs. NOTE the two drivers read --out-dir differently: run_campaign takes the FINAL
  # results dir, sweep_campaign takes the root it appends /<tag> to.
  ARGS=(--analysis-dir "$ANALYSIS" --enum-children "$dir/enum_children.json"
        --snapshot "$SNAPSHOT" --reward-threshold "$THR"
        --child-policy "$CHILD_POLICY" --prebuild-k "$PREBUILD_K" --tag "$TAG")
  echo "=== [run_arms] $arm  ($dir)"
  python experiments/lsd_hubs/campaign/run_campaign.py "${ARGS[@]}" --out-dir "$OUT" \
      --similarity "$SIM" --budget-modes "$BUDGET_MODES" --budget-reactions "$BUDGET_RXN" \
      > "$OUT/run_campaign.log" 2>&1 \
    && echo "  run_campaign OK" || echo "  run_campaign FAILED (see $OUT/run_campaign.log)"
  # The headline sweep: reactions needed to reach BUDGET_MODES modes, across the diversity cutoff.
  python experiments/lsd_hubs/campaign/sweep_campaign.py "${ARGS[@]}" --out-dir "$RESULTS" \
      --budget-modes "$BUDGET_MODES" --budget-reactions "$BUDGET_RXN" --baseline-cutoff "$SIM" \
      --n-hubs 200 \
      > "$OUT/sweep_campaign.log" 2>&1 \
    && echo "  sweep_campaign OK" || echo "  sweep_campaign FAILED (see $OUT/sweep_campaign.log)"
done

python experiments/lsd_hubs/hub_order/compare_hub_order.py --results "$RESULTS" --suffix "$TAG_SUFFIX"
