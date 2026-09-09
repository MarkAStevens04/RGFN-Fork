# `campaign/` — hub-batching vs best-candidate under a budget (Logs/028)

> **Looking for the COMPETITOR MATRIX** (6 generators × 3 targets × 3 seeds × 2 pools, Stage 1-4,
> `Logs/079`)? That is a different line of work sharing this directory — see
> **[`COMPETITOR_MATRIX.md`](COMPETITOR_MATRIX.md)** for where those results live, the quote rules
> that apply to them, and how to re-audit. Read it before comparing anything against those numbers:
> several of the rules there exist because breaking them already produced a wrong number.

Compares two **swappable** selection strategies for building a diverse library of hits
(modes) from a trained SCENT model, on reactions/mode (+ reward-gen calls, scaffolds, distinct
intermediates). The strategy logic is AL-ready in `glue.samplers.lsdflow.campaign`
(`BestCandidateStrategy` / `HubBatchingStrategy`, identical `CampaignResult` output). This dir is the
offline analysis.

- **best-candidate** — top-reward modes from the generator's *sampled* pool. Mirrors RGFN's paper
  "top-k". Each mode is charged its own assembly, **but** parent scaffolds that several picks happen
  to share ("accidental hub-batching") are built once — a swappable `HubAssignmentPolicy` (default
  `MostSharedAssignment`; immediate-parent hubs from `records.csv`).
- **hub-batching** — walk pre-ranked hubs; build each scaffold once, diversify into modes. The
  **within-hub child policy** (Logs/037, `--child-policy`, `glue.samplers.lsdflow.child_select`)
  decides which of a hub's children get offered to the mode selector:
  - `reward` (default) — **naive hub-batching**: reward-first, keep all, no fragment-cost awareness.
    Byte-identical to the original approach (Logs/029); the no-cost-awareness baseline/control.
  - `free_frag` — keep only children whose final reaction attaches an *already-available* fragment
    (base stock / already built / part of this hub / **pre-select stock**) → each kept child costs
    exactly **1** marginal reaction. ~1.2 rxn/mode with enough hubs (2.2–2.5× cheaper than reward/best),
    but pays a large enumeration bill (walks many hubs) and hits the strict-diversity ceiling sooner.

  **pre-select-K** (Logs/037, `--prebuild-k K --rank-by build_score`) sits on top of `free_frag`:
  pre-synthesize the top-K fragments (ranked by `campaign.rank_fragments`; `build_score` =
  `(reward−bar)·fanout / build_reactions`, or `fanout` / `reward` ablations), charge them **once** up
  front (no hub double-count), then free-fill. Because a stocked fragment is reused across many hubs,
  each hub yields more free children → fewer hubs walked → **far fewer reward-gen (≈ oracle) calls** at
  ~constant reactions/mode. K is the reactions↔calls dial. Sweep it with `preselect_sweep.py`.

**Cost = ONE count-once model for both strategies (Logs/033):** a molecule's cost = *assembly
couplings* (`num_reactions − Σ nested build cost of each attached promoted fragment` — SCENT's
`num_reactions` is fully nested, so this recovers the shallow join steps) **+** each distinct
promoted fragment built once (closure). Shared hubs (scaffolds) have their couplings charged once —
by design for hub-batching, accidentally for best-candidate. Both strategies amortize identically;
only the SELECTION differs. Two budget cases read off one curve: Case 1 = modes at a reaction budget;
Case 2 = reactions at a mode budget (≈ oracle calls).

**Three accounting axes.** (1) `reactions/mode` — the *synthesis* (bench) cost above. (2)
`reward_gen_calls` — a *count* of reward-generator invocations (enumerated children scored;
best-candidate reuses sampled scores → 0). (3) **measured compute time** (Logs/039) — real wall-clock,
differentiated by component, so we can say *how much longer the computer actually works* to do
hub-batching vs best-candidate, and *where* that time goes. Axis 3 is measured, not modelled: the GPU
enumeration worker records per-hub `enumeration_s` / `reward_gen_s` / `flow_extract_s` (+ one-time
`setup_s`) into `enum_timings.json`; each driver times its CPU mode-selection live and attributes the
measured per-hub times over the exact hubs the strategy walks (`validation.lsdflow.metrics.cost.compute_time`).

## Layout

Scripts + this README live at the top; every generated artifact lands under **`results/<tag>/`**
(the dir carries the target, so filenames are untagged). e.g. `results/scent_seh/`. The heavy GPU
enumeration is NOT here — it stays on `$SCRATCH` (`campaign_enum_<tag>_<jobid>/`, 24 MB + 42 MB).

```
campaign/
  pick_hubs.py  run_campaign.py  sweep_campaign.py  tau_similarity_surface.py  preselect_sweep.py
  batch_size_distribution.py
  hub_stats.py  diversity_pairs.py  route_trees.py  synthesis_routes.py
  submit_scent_seh_enum.sh  submit_scent_seh_enum_timed.sh  merge_enum_timings.py
  results/<tag>/  summary.json curve*.csv curve.png  sweep_summary.json
                  pareto.{csv,png} fixed_modes.{csv,png} budget_efficiency.{csv,png}  hub_stats.csv
                  compute_time.{csv,png} compute_time_by_cutoff.{csv,png} preselect_compute.png  (Logs/039)
                  diversity_pairs/  gallery.png pairs.csv  route_trees.png route_trees.csv
                                    synthesis_protocol.md synthesis_steps.csv  schemes/scheme_cut*.png
```

## Pipeline

1. **`pick_hubs.py`** (CPU) — from a SCENT analysis `records.csv`, take top-K candidates by reward →
   their parent hubs → rank by single-candidate flow `F_hat` → `hubs.csv`. Hub-batching walks that
   file top-to-bottom, so **the row order is the strategy**. Two knobs select a different strategy
   without touching anything else (Logs/053; defaults reproduce the above byte-for-byte):
   `--pool {topk_candidates,all}` (the reward pre-filter, or every observed hub) and
   `--order {flow_desc,flow_asc,random,candidate_reward}`, plus `--restrict-to` to apply an order to
   a fixed hub set. The ablation that exercises them lives in
   [`../hub_order/`](../hub_order/) — and found the pre-filter is *not* load-bearing
   (`--pool all --order flow_desc` is marginally better), while reversing or randomising the order
   costs 1.7×/1.5×.
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
5. **`tau_similarity_surface.py`** (CPU) — the same comparison over **both** quality knobs at once:
   reactions/mode on a (reward bar τ) x (diversity cutoff) grid at a FIXED reaction budget, so the
   robustness claim is a surface rather than the two 1-D slices `sweep_campaign.py` (cutoff at fixed τ)
   and `matrix16/gate_curve.py` (τ at fixed cutoff) give separately. Reuses this file's own
   `_load_candidates` / `_load_enumerated_hubs` / `build_strategy`, so each cell is the same computation
   as a `run_campaign.py` run at that (τ, cutoff). Cells where a strategy exhausts its library before
   spending the budget are marked `pool_limited` and hatched — never counted as wins.
   → `results/<tag>/surface.{csv,json,png,pdf}`. 221 cells ≈ 11 min; `submit_tau_similarity_surface.sh`
   runs it on `debug` (writes to `$SCRATCH`, since `$HOME` is read-only on compute). See Logs/052.

6. **`hub_stats.py`** (CPU) — per-hub table (depth, #children, `U(h)`, mean `F_hat`, reward summary)
   from the enumeration → `results/<tag>/hub_stats.csv`.
7. **`diversity_pairs.py`** (CPU) — a visual read on what each diversity cutoff *means*: for every
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
   per pair (`schemes/`). Same inputs as `route_trees.py`, **plus** the ground-truth reconstruction
   data: the final hub→child step uses the logged `reaction` field in `enum_children.json` (not an
   inference), and `--routes routes.json` (sample mode) makes the **hub build itself** instead of
   showing as a bought leaf. Both are optional — it falls back gracefully if absent. See Logs/032.

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

## Result (sEH, job 70295 = 50 hubs enumerated; default diversity cutoff 0.5; **fair cost model, Logs/033**)

| metric | best-candidate | hub-batching |
|---|---|---|
| reactions to generate 300 modes | 929 rxns (3.10 rxns/mode) | **819 rxns (2.73 rxns/mode)** |
| modes at a 100-reaction budget | 29 | **33** |
| reward-gen calls / mode | 0 | 62.9 (18,856 total) |
| distinct intermediates / shared hubs | 205 / 254 (accidental) | 328 / **2** (by design) |
| scaffolds (of 300) · best sEH | 300 · 8.40 | 300 · 8.40 |

Hub-batching still wins, but by **~1.13×, not ~2×** — most of the old "halving" was a fragment
double-count (SCENT's `num_reactions` already nests the promoted-fragment builds; the campaign used
to add them again). Of best-candidate's drop from the old 1,479 → 929: the **double-count fix is
−530**, crediting accidental hub-sharing only **−20** (254 shared hubs across 300 modes — diverse
modes have diverse scaffolds, so little accidental sharing at cutoff 0.5). The remaining hub-batching
win is **scaffold amortization** (300 modes from two flow-ranked hubs built once + cheap one-reaction
diversifications), paid for with **enumeration scoring** (63 reward-gen calls/mode vs 0). Per-hub
stats: `results/scent_seh/hub_stats.csv` (50 hubs; preliminary set 64 hubs, depth-1/2/3 = 22/29/13,
cost-independent). Full enumeration (24 MB + 42 MB) on `$SCRATCH` `lsdflow/campaign_enum_seh_70295/`.
See Logs/033 (correction) + Logs/029. Committed artifacts under `results/scent_seh/`. *(Looser
cutoffs credit more accidental sharing: at 0.90 best-candidate falls to 728 rxns.)*

## Sweep results (sEH, `sweep_campaign.py`, similarity 0.30→0.90; **fair cost model**)

Reactions to reach 300 modes, corrected model:

| cutoff | best-candidate | hub (50) | hub (200) |
|---|---|---|---|
| 0.30 (strict) | 1,123 | can't (135 max) | 882 |
| 0.50 | 929 | 819 | 816 |
| 0.70 | 793 | 706 | 770 |
| 0.80 | 722 | 656 | 828 |
| 0.90 (loose) | 728 | 644 | 822 |

- **Hub-batching's edge shrank to ~1.1–1.3× at strict-to-mid cutoffs** (was ~2×). best-candidate's
  cost now *falls* as diversity loosens (near-identical modes share more parent hubs → more
  accidental batching); at loose cutoffs (≥0.75, 200-hub enum) best-candidate **beats** hub-batching
  on reactions — hub-batching's residual cost there is its 7k–19k reward-gen (enumeration) calls, not
  synthesis. best-candidate is identical across the two enumerations (it never uses hub data).
- **Scaffold-concentration ceiling (cost-independent):** at cutoff 0.30 the 50-hub run **can't reach
  300 modes** (135 max); scaling to 200 hubs breaks it (882 rxns). Below ~0.30 *neither* strategy
  reaches 300 — a chemistry floor (see Logs/031).

## Threshold variants (Logs/035) — the "mode"/hit bar is a CLI knob

The `--reward-threshold` (what counts as a hit before the diversity filter) was **7.0** in Logs/029/033,
but Logs/034 showed 7.0 is mis-calibrated for the sEH proxy (real inhibitors top out ~7.7; empirically
useful bar ~5–6). Re-running at **5 and 6** is a pure-CPU re-selection over the *same cached* enumeration
(no GPU, no re-scoring) — just a different `--tag` so nothing overwrites. Result-dir map:

| bar | 50-hub tag | 200-hub tag |
|---|---|---|
| **7.0** (baseline; `029`/`031`/`033`) | `results/scent_seh/` | `results/scent_seh_1kx200/` |
| 6.0 (`035`) | `results/scent_seh_thr6/` | `results/scent_seh_1kx200_thr6/` |
| 5.0 (`035`) | `results/scent_seh_thr5/` | `results/scent_seh_1kx200_thr5/` |

Each dir's `summary.json`/`sweep_summary.json` records its own `"reward_threshold"`. `compare_thresholds.py`
reads the three bars' committed `{pareto,fixed_modes}.csv` and overlays them (strategy = colour, threshold =
linestyle) → `results/threshold_comparison/{cost,pareto}_{50hub,200hub}.png`. **Finding:** the cost/Pareto
comparison is threshold-robust — best-candidate is identical across bars (its top-300 modes are all
high-reward), hub-batching matches the bar-7 curve from cutoff ~0.55 up and only gets cheaper at strict
diversity; the real change is hub-batching's enumeration bill halving (one hub, not two, serves 300 modes
at cutoff 0.5). See Logs/035.

```bash
# thresholds 5 and 6 (50-hub shown; swap $ENUM50 -> $ENUM200 + tag *_1kx200_thr* for 200-hub)
for T in 5.0 6.0; do
  python experiments/lsd_hubs/campaign/run_campaign.py   $ANALYSIS_ARGS --reward-threshold $T --tag scent_seh_thr${T%.0}
  python experiments/lsd_hubs/campaign/sweep_campaign.py $ANALYSIS_ARGS --reward-threshold $T --tag scent_seh_thr${T%.0}
done
python experiments/lsd_hubs/campaign/compare_thresholds.py   # overlay figures
```

## Compute-time accounting (Logs/039) — measured wall-clock, differentiated by component

The third axis: **how much longer the computer actually works** for hub-batching vs best-candidate,
and *where* the time goes. Measured live during a re-run, never inferred. Components:

| component | best-candidate | hub-batching |
|---|---|---|
| `setup_s` (model load + library freeze) | 0 | once (if it walks ≥1 hub) |
| `hub_pick_s` (Stage-2 ranking) | 0 | `pick_hubs.py` |
| `enumeration_s` (RDKit children) | 0 | Σ over walked hubs (measured) |
| `reward_gen_s` (proxy/docking scoring) | 0 marginal¹ | Σ over walked hubs (measured) |
| `flow_extract_s` (P_F/P_B for U(h)) | 0 | Σ over walked hubs (measured) |
| `mode_selection_s` (diversity filter) | measured live | measured live |

¹ best-candidate reuses rewards computed during Stage-1 sampling (the shared pool). Stage-1 sampling
time is tracked separately (`sample_timings.json`) and cancels in the head-to-head.

**Pipeline.** The worker (`scent_worker.py`) is instrumented to write per-hub `enum_timings.json`. To
get real numbers, re-run the enumeration **with timers on** — split into slices for the 2h `debug`
limit, then merge:

```bash
# 1) time the exact 200-hub enumeration in 6 parallel debug slices (writes slice*/enum_timings.json)
for s in 0 1 2 3 4 5; do sbatch experiments/lsd_hubs/campaign/submit_scent_seh_enum_timed.sh $s; done
# 2) merge into one enum_timings.json beside the canonical enum_children.json (+ integrity check)
python experiments/lsd_hubs/campaign/merge_enum_timings.py \
    --timed-dir   /scratch/.../lsdflow/campaign_enum_seh_timed \
    --canonical   /scratch/.../lsdflow/campaign_enum_seh_70363 \
    --out         /scratch/.../lsdflow/campaign_enum_seh_70363/enum_timings.json \
    --pick-hubs-timing /scratch/.../lsdflow/campaign_enum_seh_timed/pick_hubs_timing.json
```

**Reading it out.** All four drivers auto-detect `enum_timings.json` (and `pick_hubs_timing.json`)
beside `--enum-children` (override with `--enum-timings` / `--hub-pick-timing`); with no file they
skip the compute-time section (backward-compatible). Each adds:
- `run_campaign.py` → `compute_time.{csv,png}` (per-component stacked bar + head-to-head) + a
  `compute_time` block in `summary.json`.
- `sweep_campaign.py` → `compute_time_by_cutoff.csv` + `compute_time_by_cutoff.png` (total compute vs diversity
  cutoff) + a `compute_time` block in `sweep_summary.json`.
- `preselect_sweep.py` → `ct_*`/`compute_total_s` columns in `preselect.csv` + `preselect_compute.png`
  (the measured-time companion to the reactions↔calls Pareto — pre-select-K cutting real seconds).
- `batch_size_distribution.py` → `ct_*`/`compute_total_s` columns per K in `batch_stats.csv`.
