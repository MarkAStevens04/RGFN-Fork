#!/bin/bash
# Gate-threshold sweep for the matrix16 surrogate cells (Logs/034 motivation).
#
# Re-scores the count-once campaign — BOTH strategies (best-candidate + hub-batching) — at each
# target's `threshold_variants`, reusing the EXISTING enumeration (pure CPU, no re-enumeration).
# WHY: the default sEH-7.0 gate is optimistic (real inhibitors top ~7.7) and STARVES weaker-reward
# generators — e.g. only 1.2% of RxnFlow-sEH children clear 7.0, collapsing hub amortization
# (1.38 modes/hub, 1.23x) — whereas at the calibrated 5/6 bars it recovers (5-7 modes/hub, 2.0-2.3x).
# Sweeping the bar puts the cross-generator comparison on a non-starved footing.
#
# Child-policy is kept consistent with the headline results: SCENT uses its hero
# free_frag + pre-select-K=20 (it has a dynamic library); baselines use naive reward. best-candidate
# is child-policy-independent and emitted by every run.
#
# Usage:  bash experiments/lsd_hubs/matrix16/gate_sweep.sh [gen_target ...]   (default: all surrogate cells with an enum)
# Output: results/<cell>_thr<gate>/  +  results/gate_sweep/summary.csv (compiled table)
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO"
source ~/bin/rgfn-smoke-env.sh 2>/dev/null || {
    source /home/markymoo/miniconda3/etc/profile.d/conda.sh; conda activate rgfn; }

M=experiments/lsd_hubs/matrix16
OUT=$M/results
mkdir -p "$OUT/gate_sweep"

CELLS=("$@")
if [ ${#CELLS[@]} -eq 0 ]; then
    mapfile -t rows < <(python "$M/manifest.py" --list active)   # surrogate cells (seh/drd2 x 4 gens)
    CELLS=(); for r in "${rows[@]}"; do CELLS+=("$(echo "$r" | tr '\t' '_')"); done
fi

for cell in "${CELLS[@]}"; do
    gen=${cell%_*}; tgt=${cell##*_}
    eval "$(python "$M/manifest.py" --emit "$gen" "$tgt")"
    EN="$ENUM_DIR/enum_children.json"
    if [ ! -f "$EN" ]; then echo "skip $cell (no enumeration yet)"; continue; fi
    gates=$(python -c "import sys;sys.path.insert(0,'$M');from targets import get_target;print(*get_target('$tgt').threshold_variants)")
    if [ "$gen" = scent ]; then POL=(--child-policy free_frag --prebuild-k 20); else POL=(--child-policy reward); fi
    SNAP=()
    if [ "$gen" = scent ]; then
        RR="$(cd "$(dirname "$CHECKPOINT")/../.." 2>/dev/null && pwd || true)"
        sf="$(find "$RR" -path '*additional_fragments/fragments_*.json' 2>/dev/null | sort -t_ -k2 -n | tail -1 || true)"
        [ -n "$sf" ] && SNAP=(--snapshot "$sf")
    fi
    echo "=== $cell  gates=[$gates]  policy=${POL[*]} ==="
    for g in $gates; do
        tag="${cell}_thr${g%.0}"
        python "$M"/../campaign/run_campaign.py \
            --analysis-dir "$SAMPLE_DIR" --enum-children "$EN" \
            --reward-threshold "$g" --higher-is-better "$HIGHER_IS_BETTER" "${POL[@]}" "${SNAP[@]}" \
            --tag "$tag" --out-dir "$OUT/$tag" \
            >/scratch/markymoo/rgfn_runs/m16_gatesweep_${tag}.log 2>&1 \
            && echo "  done $tag" || echo "  FAIL $tag (see /scratch/.../m16_gatesweep_${tag}.log)"
    done
done

echo "=== compiling summary ==="
python - <<'PY'
import csv, glob, json, os, re
OUT="experiments/lsd_hubs/matrix16/results"
rows=[]
for d in sorted(glob.glob(f"{OUT}/*_thr*")):
    sp=f"{d}/summary.json"
    if not os.path.isdir(d) or not os.path.exists(sp): continue
    s=json.load(open(sp)); tag=os.path.basename(d)
    m=re.match(r"(.+)_thr(.+)$", tag); cell=m.group(1); gate=m.group(2)
    bc,hb=s["best_candidate"],s["hub_batching"]
    edge=(bc["reactions_per_mode"]/hb["reactions_per_mode"]) if (bc["reactions_per_mode"] and hb["reactions_per_mode"]) else None
    rows.append(dict(cell=cell, gate=float(gate), child_policy=s.get("child_policy"),
        best_modes=bc["total_modes"], best_rxn_per_mode=bc["reactions_per_mode"],
        hub_modes=hb["total_modes"], hub_rxn_per_mode=hb["reactions_per_mode"],
        edge=round(edge,3) if edge else None, hub_reward_gen_calls=hb["total_reward_gen_calls"]))
rows.sort(key=lambda r:(r["cell"], r["gate"]))
csvp=f"{OUT}/gate_sweep/summary.csv"
with open(csvp,"w",newline="") as fh:
    w=csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {csvp} ({len(rows)} cell x gate points)")
print(f"\n{'cell':14}{'gate':>5}{'best_modes':>11}{'best r/m':>9}{'hub_modes':>10}{'hub r/m':>8}{'edge':>7}")
for r in rows:
    print(f"{r['cell']:14}{r['gate']:>5}{r['best_modes']:>11}{r['best_rxn_per_mode']:>9}{r['hub_modes']:>10}{r['hub_rxn_per_mode']:>8}{(str(r['edge'])+'x'):>7}")
PY
echo "=== gate sweep done ==="
