#!/bin/bash
# CPU count-once campaign readout for ONE matrix cell (hub-batching vs best-candidate, Logs/029/033).
# Consumes the cell's sample_dir + enum_dir artifacts (produced by submit_cell.sh) and writes the
# head-to-head to experiments/lsd_hubs/matrix16/results/<gen>_<target>/. Pure CPU, seconds — safe on
# a login node. This is the default (count-once) evaluator; the SPARROW evaluator is a separate,
# concurrently-developed axis that reads the SAME enum_children.json.
#
# Usage:  bash experiments/lsd_hubs/matrix16/run_cell_campaign.sh <generator> <target>
# Knobs (env): SIMILARITY (0.5) BUDGET_REACTIONS (100) BUDGET_MODES (300)
#              CHILD_POLICY (reward) PREBUILD_K (0) SNAPSHOT (scent frag json; auto for scent)
set -uo pipefail

GEN=${1:?usage: run_cell_campaign.sh <generator> <target>}
TGT=${2:?usage: run_cell_campaign.sh <generator> <target>}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"

# The campaign imports glue (-> rgfn/dgl), so it needs the rgfn env even though it is CPU-only.
# rgfn-smoke-env.sh activates rgfn + sets LD_LIBRARY_PATH (dgl CUDA libs); login-safe.
source ~/bin/rgfn-smoke-env.sh 2>/dev/null || { source /home/markymoo/miniconda3/etc/profile.d/conda.sh; conda activate rgfn; }

eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || {
    echo "ERROR: manifest emit failed for '$GEN' '$TGT'"; exit 1; }

ENUM_JSON="$ENUM_DIR/enum_children.json"
for f in "$SAMPLE_DIR/records.csv" "$ENUM_JSON"; do
    [ -f "$f" ] || { echo "ERROR: missing artifact $f — run submit_cell.sh $GEN $TGT first"; exit 1; }
done

SIMILARITY=${SIMILARITY:-0.5}
BUDGET_REACTIONS=${BUDGET_REACTIONS:-100}
BUDGET_MODES=${BUDGET_MODES:-300}
CHILD_POLICY=${CHILD_POLICY:-reward}
PREBUILD_K=${PREBUILD_K:-0}

# SCENT: nested cost needs the promoted-fragment snapshot (fragments_<N>.json). Auto-discover it
# beside the checkpoint unless SNAPSHOT is given. Other generators: no snapshot (min_num_reactions).
SNAP_ARG=()
SNAP="${SNAPSHOT:-}"
if [ "$GEN" = "scent" ] && [ -z "$SNAP" ]; then
    # additional_fragments/ sits at the run root (seed42/additional_fragments), a couple levels above
    # the checkpoint (seed42/train/checkpoints/last_gfn.pt) — search up from the checkpoint dir and
    # take the highest-N snapshot (fragments_<N>.json), matching scent_worker._derive_snapshot.
    RUN_ROOT="$(cd "$(dirname "$CHECKPOINT")/../.." 2>/dev/null && pwd || true)"
    SNAP="$(find "$RUN_ROOT" -path '*additional_fragments/fragments_*.json' 2>/dev/null \
        | sort -t_ -k2 -n | tail -1 || true)"
    [ -n "$SNAP" ] && echo "[campaign] auto-discovered SCENT snapshot: $SNAP" \
        || echo "[campaign] WARNING no SCENT fragments snapshot found -> nested cost falls back to min_num_reactions"
fi
[ -n "$SNAP" ] && SNAP_ARG=(--snapshot "$SNAP")

echo "=== [$CELL_TAG] campaign: gate=$MODE_REWARD_THRESHOLD hib=$HIGHER_IS_BETTER sim=$SIMILARITY child=$CHILD_POLICY K=$PREBUILD_K ==="
python experiments/lsd_hubs/campaign/run_campaign.py \
    --analysis-dir "$SAMPLE_DIR" --enum-children "$ENUM_JSON" \
    --reward-threshold "$MODE_REWARD_THRESHOLD" --higher-is-better "$HIGHER_IS_BETTER" \
    --similarity "$SIMILARITY" --budget-reactions "$BUDGET_REACTIONS" --budget-modes "$BUDGET_MODES" \
    --child-policy "$CHILD_POLICY" --prebuild-k "$PREBUILD_K" \
    --tag "$CELL_TAG" --out-dir "$RESULTS_DIR" "${SNAP_ARG[@]}"

echo "=== DONE [$CELL_TAG] campaign -> $RESULTS_DIR ==="
