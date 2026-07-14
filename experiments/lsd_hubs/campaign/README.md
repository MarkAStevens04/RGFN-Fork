# `campaign/` — hub-batching vs best-candidate under a budget (Logs/028)

Compares two **independent, swappable** selection strategies for building a diverse library of hits
(modes) from a trained SCENT model, on reactions/mode (+ reward-gen calls, scaffolds, distinct
intermediates). The strategy logic is AL-ready in `glue.samplers.lsdflow.campaign`
(`BestCandidateStrategy` / `HubBatchingStrategy`, identical `CampaignResult` output, **disjoint
inputs** — best-candidate never touches hub/enumeration data). This dir is the offline analysis.

- **best-candidate** — top-reward modes from the generator's *sampled* pool; each built
  independently (promoted fragments shared as reusable stock). Mirrors RGFN's paper "top-k".
- **hub-batching** — walk pre-ranked hubs; build each scaffold once, diversify into modes.

**Cost** = true nested reactions, count-once (each distinct promoted fragment built once via its
logged route; hub scaffolds shared only in hub-batching). Two budget cases read off one curve:
Case 1 = modes at a reaction budget; Case 2 = reactions at a mode budget (≈ oracle calls).

## Layout

Scripts + this README live at the top; every generated artifact lands under **`results/<tag>/`**
(the dir carries the target, so filenames are untagged). e.g. `results/scent_seh/`. The heavy GPU
enumeration is NOT here — it stays on `$SCRATCH` (`campaign_enum_<tag>_<jobid>/`, 24 MB + 42 MB).

```
campaign/
  pick_hubs.py  run_campaign.py  sweep_campaign.py  hub_stats.py
  diversity_pairs.py  route_trees.py  synthesis_routes.py  submit_scent_seh_enum.sh
  results/<tag>/  summary.json curve*.csv curve.png  sweep_summary.json
                  pareto.{csv,png} fixed_modes.{csv,png} budget_efficiency.{csv,png}  hub_stats.csv
                  diversity_pairs/  gallery.png pairs.csv  route_trees.png route_trees.csv
                                    synthesis_protocol.md synthesis_steps.csv  schemes/scheme_cut*.png
```

## Pipeline

1. **`pick_hubs.py`** (CPU) — from a SCENT analysis `records.csv`, take top-K candidates by reward →
   their parent hubs → rank by single-candidate flow `F_hat` → `hubs.csv`. (The first hub strategy;
   swap this file for others.)
2. **GPU enumeration** — `submit_scent_seh_enum.sh` runs `pick_hubs` then the scent-env worker
   (`--mode enumerate`) to exhaustively enumerate + reward-score each hub's children on the frozen
   full library, emitting `enum_children.json` (child + reward + fragment added in the final
   reaction). Uses the **recipe** re-run checkpoint (2026-07-10, job 70180) → exact nested cost.
3. **`run_campaign.py`** (CPU) — loads candidates (`records.csv`+`compositions.json`),
   `enum_children.json`, and the recipe `fragments_<N>.json`; runs both strategies at one
   cutoff/budget → one head-to-head point → `results/<tag>/summary.json` + `curve_*.csv` + `curve.png`.
4. **`sweep_campaign.py`** (CPU) — the diversity/budget **sweeps**. Each simulation uses ONE
   predefined budget and yields ONE point; curves are built by re-running the greedy selection over
   the **cached** enumeration (no re-scoring, no GPU), varying only the diversity cutoff / budget.
   Produces three hub-vs-best plots: **Pareto** (modes at a fixed reaction budget vs cutoff),
   **cost** (reactions for M* modes vs cutoff), **budget-vs-efficiency** (modes vs reaction budget at
   a fixed cutoff) → `results/<tag>/{pareto,fixed_modes,budget_efficiency}.{csv,png}` +
   `sweep_summary.json`. ~44 s for sEH (26 runs).
5. **`hub_stats.py`** (CPU) — per-hub table (depth, #children, `U(h)`, mean `F_hat`, reward summary)
   from the enumeration → `results/<tag>/hub_stats.csv`.
6. **`diversity_pairs.py`** (CPU) — a visual read on what each diversity cutoff *means*: for every
   cutoff in the sweep, rebuild the hub-batching library and draw the two accepted modes with the
   **highest** pairwise Tanimoto (the closest still-distinct pair, shared MCS highlighted) →
   `results/<tag>/diversity_pairs/{gallery.png,pairs.csv}`. Reads only `enum_children.json` (acceptance
   is reward + Tanimoto; no cost table needed). See Logs/032.
7. **`route_trees.py`** (CPU) — schematic of where each pair's two molecules overlap: the shared
   building block/intermediate forking into the two products → `diversity_pairs/route_trees.{png,csv}`.
   Reads `pairs.csv` + `enum_children.json` + the recipe snapshot (`smiles_to_route`). See Logs/032.
8. **`synthesis_routes.py`** (CPU) — the chemist-actionable step-by-step protocol: names each reaction
   from its RGFN template, tags reactants buy/make, and emits per-molecule instructions + shopping
   lists (`synthesis_protocol.md`), a step table (`synthesis_steps.csv`), and one reaction-scheme PNG
   per pair (`schemes/`). Same inputs as `route_trees.py`. See Logs/032.

```bash
sbatch experiments/lsd_hubs/campaign/submit_scent_seh_enum.sh   # -> $SCRATCH/.../campaign_enum_seh_<jobid>/enum_children.json
source ~/bin/rgfn-smoke-env.sh
ARGS="--analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
     --enum-children /scratch/.../campaign_enum_seh_<jobid>/enum_children.json \
     --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
     --reward-threshold 7.0 --tag scent_seh"
python experiments/lsd_hubs/campaign/run_campaign.py   $ARGS   # one head-to-head point
python experiments/lsd_hubs/campaign/sweep_campaign.py $ARGS   # the 3 diversity/budget-sweep plots
python experiments/lsd_hubs/campaign/hub_stats.py \
    --enum-dir /scratch/.../campaign_enum_seh_<jobid> --reward-threshold 7.0 --tag scent_seh
```

`--reward-threshold` is target-specific (sEH ~7.0; DRD2 ~0.5); `sweep_campaign.py` also takes
`--cutoff-{min,max,step}` (default 0.30/0.90/0.05), `--budget-reactions` (R*=100),
`--budget-modes` (M*=300), `--baseline-cutoff` (**0.50** — the default diversity cutoff; also the
dashed marker on the Pareto/cost plots). All outputs land in `results/<tag>/`.

## Result (sEH, job 70295 = 50 hubs enumerated; default diversity cutoff 0.5)

| metric | best-candidate | hub-batching |
|---|---|---|
| reactions to generate 300 modes | 1,479 rxns (4.93 rxns/mode) | **820 rxns (2.73 rxns/mode)** |
| modes at a 100-reaction budget | 16 | **33** |
| reward-gen calls / mode | 0 | 62.9 (18,856 total) |
| distinct intermediates / hubs (300-mode budget) | 205 / 0 | 328 / **2** |
| scaffolds (of 300) · best sEH | 300 · 8.40 | 300 · 8.40 |

Hub-batching roughly halves reactions/mode — the win is **scaffold amortization** (300 modes from
just two flow-ranked hubs built once + cheap one-reaction diversifications), *not* intermediate
concentration (it uses more distinct intermediates). The 2.73 (not ~1) is because each final
reaction attaches a *promoted* dynamic-library fragment with its own nested route (242/298 marginal
modes cost 3 rxns = final + a fresh 2-rxn intermediate; only 26 reuse a built one). The cost is
enumeration scoring (63 reward-gen calls/mode). Per-hub stats: `results/scent_seh/hub_stats.csv`
(50 hubs; preliminary set 64 hubs, depth-1/2/3 = 22/29/13). Full enumeration (24 MB + 42 MB) on
`$SCRATCH` `lsdflow/campaign_enum_seh_70295/`. See Logs/029. All committed artifacts live under
`results/scent_seh/`. *(At the looser 0.7 cutoff: 706 vs 1,445, all from one hub — but 0.7 counts
near-identical molecules as distinct.)*

## Sweep results (sEH, `sweep_campaign.py`, similarity 0.30→0.90)

- **Hub-batching dominates efficiency across the whole diversity range it can serve** — ~2× more
  modes per 100 reactions (32–40 vs 16–18) and ~2× fewer reactions for 300 modes (644–824 vs
  1,386–1,479). best-candidate is flat in the cutoff; hub-batching's edge grows as diversity loosens
  (one hub serves everything) and erodes as it tightens (it must recruit more hubs: 1 at 0.70 → 28 at
  0.30).
- **Scaffold-concentration ceiling:** at the strictest cutoff (0.30) hub-batching **can't reach 300
  modes** (maxes at 135, 28/50 hubs exhausted) — its hubs' children are decorations of a few cores;
  best-candidate reaches 300 from the broad sampled pool. Partly an artifact of enumerating only 50
  hubs → more hubs raise the ceiling (ties into the `--no-flow` speedup).
