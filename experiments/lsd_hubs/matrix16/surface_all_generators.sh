#!/bin/bash
# The two-knob cost surface (reward bar x diversity cutoff, shaded by reactions/mode) for EVERY
# generator on one target -- the cross-generator version of Logs/052's single-cell figure.
#
# WHY: Logs/052 measured that surface for SCENT alone, which leaves the obvious question open --- is the
# shape (advantage broad across both knobs, collapsing only in the strict/high-bar corner) a property of
# hub-batching, or of SCENT? Running the identical grid on all four generators answers it, and it is
# free: pure CPU re-scoring of enumerations we already have.
#
# This is a thin driver over `campaign/tau_similarity_surface.py` (the other agent's, Logs/052) --- the
# surface maths, pool-limited hatching and plotting all live there and are NOT duplicated here. What
# this adds is (a) manifest-resolved paths per cell and (b) the per-generator child-policy convention
# used by run_cell_campaign.sh / gate_sweep.sh, so the numbers line up with every committed table:
# SCENT gets its hero free_frag + pre-select-K=20 (it has a dynamic fragment library); the baselines
# have no promoted fragments, so free_frag is degenerate for them and naive `reward` is the honest
# setting. Note the upstream script DEFAULTS to free_frag/K=20 (it was written for SCENT), so passing
# the baseline policy explicitly is load-bearing, not decorative.
#
# One process per cell on purpose: the login node's `ulimit -t` is per-process, so four sequential
# python invocations each get their own CPU budget instead of sharing one and tripping SIGXCPU.
#
# Usage:  bash experiments/lsd_hubs/matrix16/surface_all_generators.sh [target] [gen ...]
#         TAUS=4:8:0.5 SIMS=0.3:0.9:0.05 BUDGET=300 bash .../surface_all_generators.sh seh
# Output: results/surface/<cell>/{surface.csv,surface.json,surface.png,surface.pdf}
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1
source ~/bin/rgfn-smoke-env.sh >/dev/null 2>&1 || {
    source /home/markymoo/miniconda3/etc/profile.d/conda.sh; conda activate rgfn; }

M=experiments/lsd_hubs/matrix16
SURF=experiments/lsd_hubs/campaign/tau_similarity_surface.py
TARGET=${1:-seh}; shift || true
GENS=("$@"); [ ${#GENS[@]} -eq 0 ] && GENS=(rgfn scent rxnflow fraggfn)

TAUS=${TAUS:-4:8:0.5}
SIMS=${SIMS:-0.3:0.9:0.05}
BUDGET=${BUDGET:-300}
OUTROOT=$M/results/surface

for gen in "${GENS[@]}"; do
    eval "$(python "$M/manifest.py" --emit "$gen" "$TARGET")" || { echo "skip $gen (manifest)"; continue; }
    EN="$ENUM_DIR/enum_children.json"
    if [ ! -f "$EN" ]; then echo "skip $CELL_TAG (no enumeration)"; continue; fi

    # Reject a partial/stale enumeration rather than plotting a 117-cell surface off 10 hubs.
    NH=$(python -c "import json;print(len(json.load(open('$EN'))['hubs']))" 2>/dev/null || echo 0)
    NW=$(( $(wc -l < "$ENUM_DIR/hubs.csv") - 1 ))
    if [ "$NH" -lt $(( NW * 9 / 10 )) ]; then
        echo "skip $CELL_TAG: PARTIAL enumeration $NH/$NW hubs"; continue
    fi

    # Per-generator policy + SCENT's promoted-fragment recipe snapshot (nested cost model).
    if [ "$gen" = scent ]; then
        POL=(--child-policy free_frag --prebuild-k 20)
        RR="$(cd "$(dirname "$CHECKPOINT")/../.." 2>/dev/null && pwd || true)"
        sf="$(find "$RR" -path '*additional_fragments/fragments_*.json' 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)"
        [ -n "$sf" ] && POL+=(--snapshot "$sf")
    else
        POL=(--child-policy reward --prebuild-k 0)
    fi

    echo "=== $CELL_TAG  ($NH hubs)  taus=$TAUS  sims=$SIMS  budget=$BUDGET  ${POL[*]} ==="
    python "$SURF" \
        --analysis-dir "$SAMPLE_DIR" --enum-children "$EN" \
        --taus "$TAUS" --similarities "$SIMS" --budget-reactions "$BUDGET" \
        --higher-is-better "$HIGHER_IS_BETTER" "${POL[@]}" \
        --tag "$CELL_TAG" --out-dir "$OUTROOT/$CELL_TAG" \
        && echo "  done -> $OUTROOT/$CELL_TAG" || echo "  FAIL $CELL_TAG"
done

echo "=== surfaces under $OUTROOT ==="
ls -d $OUTROOT/*/ 2>/dev/null
