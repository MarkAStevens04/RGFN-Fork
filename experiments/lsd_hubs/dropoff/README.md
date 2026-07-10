# `dropoff/` — per-hub filter funnel

How many of a hub's one-reaction terminal children survive each filter stage, and **which**
filter does the work:

```
raw children  --(binding/reward gate)-->  hits  --(greedy Tanimoto dedup)-->  hit-modes
```

Uses the canonical, paper-comparable mode definition from
`validation/lsdflow/metrics/diversity` (reward-gated greedy sphere-exclusion, ECFP Morgan r=3,
2048 bits, similarity 0.7 — matching upstream `TanimotoSimilarityModes`), so "mode" is never
reimplemented here. See Logs/026 for the write-up.

## Run

GFN-free — pure CSV + RDKit, runs on a login node:

```bash
source ~/bin/rgfn-smoke-env.sh
python experiments/lsd_hubs/dropoff/funnel.py \
    --records /scratch/markymoo/rgfn_runs/lsdflow/seh_stdlib_70140/enumerated_records.csv \
    --thresholds 6,7,7.5,8 --headline 7.0 --tag seh_70140
```

For docking differentials (lower-is-better) pass `--lower-is-better` and docking-scale
`--thresholds` / `--headline`.

## Files

- `funnel.py` — the analysis (parameterized; reusable across reward targets + DAGs).
- `funnel_<tag>_results.csv` — per-hub funnel table (committed).
- `funnel_<tag>_summary.json` — aggregate survival + redundancy (committed).

## Finding (sEH, job 70140, 12 enumerated hubs)

4,586 raw one-reaction children → **binding gate is the dominant dropoff**: 30% survive sEH≥6,
**3% survive ≥7**, 0% survive ≥8 (single-step diversification rarely reaches the paper's reward-8
bar). Tanimoto dedup is gentle by comparison (raw→structure 2.3×; the 135 ≥7 hits → 71 hit-modes,
1.9×). The two filters hit different hubs: **depth-0 fragment hubs** are killed by the *affinity*
gate (0 hits — diverse but weak; children/structure-mode ≈ 1.0–1.3), while **depth-3 hubs** carry
the hits (3–6% at ≥7) and more structural redundancy (children/structure-mode 2.7–6.5), yielding
19–29 distinct high-affinity hit-modes after both filters. Breadth comes from fragments; hits
come from depth.
