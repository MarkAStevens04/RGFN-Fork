#!/bin/bash
# Run the greedy-oracle comparison over every enumerated matrix16 cell (Logs/060).
#
# One process per cell on purpose: the login node's CPU limit is per-process, and a cell that fails
# must not take the sweep down with it. Per-generator child-policy defaults are copied verbatim from
# run_cell_campaign.sh / gate_sweep.sh so this driver can never silently evaluate a different
# configuration than the cell's own committed campaign number.
#
# Usage:  bash experiments/lsd_hubs/greedy_oracle/run_all_cells.sh [cell ...]
#         (no args = every cell with an enumeration on disk)
# Knobs:  BUDGET_MODES (300) BUDGET_REACTIONS (100) SIMILARITY (0.5) REGRESS (1) EXTRA ("")
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source ~/bin/rgfn-smoke-env.sh 2>/dev/null || {
    source /home/markymoo/miniconda3/etc/profile.d/conda.sh; conda activate rgfn; }

BUDGET_MODES=${BUDGET_MODES:-300}
BUDGET_REACTIONS=${BUDGET_REACTIONS:-100}
SIMILARITY=${SIMILARITY:-0.5}
REGRESS=${REGRESS:-1}
EXTRA=${EXTRA:-}

ALL_CELLS=(
    rgfn_seh scent_seh rxnflow_seh fraggfn_seh
    rgfn_drd2 scent_drd2 rxnflow_drd2 fraggfn_drd2
    scent_clpp rxnflow_clpp fraggfn_clpp
    scent_6td3 rxnflow_6td3 fraggfn_6td3
)
CELLS=("$@")
[ ${#CELLS[@]} -eq 0 ] && CELLS=("${ALL_CELLS[@]}")

OUT_ROOT="$REPO/experiments/lsd_hubs/greedy_oracle/results"
mkdir -p "$OUT_ROOT"
LEDGER="$OUT_ROOT/run_ledger.txt"

for CELL in "${CELLS[@]}"; do
    GEN="${CELL%%_*}"; TGT="${CELL#*_}"
    echo "=================================================================="
    echo "=== $CELL"
    if ! eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT" 2>/dev/null)"; then
        echo "SKIP $CELL — manifest emit failed" | tee -a "$LEDGER"; continue
    fi
    ENUM_JSON="$ENUM_DIR/enum_children.json"
    if [ ! -f "$ENUM_JSON" ] || [ ! -f "$SAMPLE_DIR/records.csv" ]; then
        echo "SKIP $CELL — no enumeration on disk" | tee -a "$LEDGER"; continue
    fi

    # Per-generator defaults, identical to run_cell_campaign.sh.
    if [ "$GEN" = scent ]; then CHILD_POLICY=free_frag; PREBUILD_K=20
    else                        CHILD_POLICY=reward;    PREBUILD_K=0; fi

    SNAP_ARG=()
    if [ "$GEN" = scent ]; then
        RUN_ROOT="$(cd "$(dirname "$CHECKPOINT")/../.." 2>/dev/null && pwd || true)"
        SNAP="$(find "$RUN_ROOT" -path '*additional_fragments/fragments_*.json' 2>/dev/null \
            | sort -t_ -k2 -n | tail -1 || true)"
        if [ -n "$SNAP" ]; then SNAP_ARG=(--snapshot "$SNAP")
        else echo "SKIP $CELL — SCENT snapshot missing (nested cost would not be comparable)" \
             | tee -a "$LEDGER"; continue; fi
    fi

    REG_ARG=(); [ "$REGRESS" = 1 ] && REG_ARG=(--regress)

    START=$(date +%s)
    python experiments/lsd_hubs/greedy_oracle/run_greedy_oracle.py \
        --analysis-dir "$SAMPLE_DIR" --enum-children "$ENUM_JSON" \
        --reward-threshold "$MODE_REWARD_THRESHOLD" --higher-is-better "$HIGHER_IS_BETTER" \
        --similarity "$SIMILARITY" \
        --budget-reactions "$BUDGET_REACTIONS" --budget-modes "$BUDGET_MODES" \
        --child-policy "$CHILD_POLICY" --prebuild-k "$PREBUILD_K" \
        "${SNAP_ARG[@]}" "${REG_ARG[@]}" $EXTRA --tag "$CELL"
    RC=$?
    echo "$(date +%FT%T) $CELL rc=$RC $(( $(date +%s) - START ))s policy=$CHILD_POLICY k=$PREBUILD_K" \
        | tee -a "$LEDGER"
done

echo "=== ledger: $LEDGER"
