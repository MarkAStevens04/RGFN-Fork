# SCENT sEH — how big are the hub synthesis batches, and are they even?
**Date:** 2026-07-15, ~3pm

## Question

When we build a diverse hit library by "batching" from hubs — build one shared scaffold, then run
many diversifying reactions on it — how many molecules does each hub actually contribute, and is that
number roughly even across hubs, or does one hub become a monster synthesis step hundreds of times
bigger than the others?

## Context & Summary

Hub-batching (entries `029`/`033`/`035`/`037`) builds a diverse library cheaply by reusing a shared
scaffold: build the scaffold once, then attach a different fragment for each new hit. Entry `037`
found the winning recipe — **pre-select-K**: pre-synthesize the K most widely-reusable building
blocks up front, then only keep hits that need no further new synthesis. That work optimized the
*total* cost (reactions per hit, oracle calls). It never looked at how the work is *distributed*
across the individual hubs. That matters operationally: a hub is one bench step, and if one hub yields
300 products while another yields 2, that single step is 300× the size of the others — awkward to run
even if the average looks fine. The raw numbers looked alarming: in the 200-hub enumeration, the
biggest scaffold can, in principle, undergo **2,549–11,968** distinct one-reaction products.

This experiment measures, for each hub, the **batch size** = the number of distinct diverse hits we
actually keep from it (products that clear the reward bar and survive a Tanimoto diversity filter —
the real `k` in the "`depth(h)+k`" cost model). We ran the exact pre-select-K campaign from `037` on
SCENT/sEH for a 300-hit library across five stock sizes (K = 0, 16, 50, 100, 200) and looked at how
the 300 hits spread across the hubs used, plus standard imbalance statistics.

## Answer

The "one giant synthesis step" fear does not hold up. The huge raw fan-out is a mirage: once the
reward bar and diversity filter are applied, a scaffold with ~12,000 possible children yields only a
**handful** of kept hits, because distinct diverse hits require distinct fragments and only a few of
those both score well and are actually available. Across the whole sweep the **largest single hub
batch is 65 molecules** (and that is the K=200 extreme, where we deliberately use only 10 hubs);
typical batches are single- to low-double-digit. The imbalance is moderate (Gini 0.37–0.44) and the
worst hub-to-hub size ratio is 28×, not 300×. K is a clean **consolidation dial**: raising K packs the
same 300 hits into fewer, larger, *relatively* more-even hubs (Gini falls, biggest/smallest ratio
falls from 17× to 9×), at the cost of concentrating more of the library into a few steps (the single
biggest hub goes from 6% to 22% of the library). Low K = many small parallel steps; high K = few
larger consolidated steps. Neither regime produces a pathological batch.

## Relevance to our Publication

The LSD-Flow / hub-batching story sells "build once, diversify late" as a *practical* synthesis
saving, so a reviewer (and a real medicinal chemist) will immediately ask whether the saving is an
artifact of dumping everything onto one impossible scaffold. This entry answers that pre-emptively
with the batch-size distribution: the batches are bench-plausible (≤ 65, typically far smaller) and
tunable via K, and it shows raw enumeration fan-out is the wrong quantity to reason about synthesis
load — the reward+diversity-filtered batch is. It strengthens the cost/Pareto section (`037`) with the
operational-feasibility angle, and the K dial gives chemists a knob they can read directly ("many
small plates vs a few big ones").

## Next Experiments

**Refining for publication**
- **Library-size sensitivity.** The largest batch is measured at a fixed 300-hit target; at high K
  (few hubs) a 1,000-hit target would scale the biggest hub up proportionally. Re-run the sweep at a
  couple of library sizes to state where, if ever, a step gets uncomfortably large.
- **Second target (DRD2)** — repeat the batch-size distribution to show the balance finding is not
  sEH-specific (the DRD2 analysis DAG + enumeration already exist from `037`'s to-do list).
- **Diversity-cutoff sensitivity** — the batch size depends on the Tanimoto cutoff (0.5 here); trace
  how the distribution tightens/loosens at 0.6/0.7 (the mode-definition robustness from `035`).

**Next steps in project**
- If library-size sensitivity ever shows a dominant step, add the deferred **per-hub batch cap**
  (`k_max`) to `HubBatchingStrategy` and measure the reactions/oracle-call cost of enforcing balance
  — the balancing knob we intentionally did not build here.
- Wire naive/free-frag/pre-select-K into the active-learning loop (`LSDFlowAcquisition`) and report
  batch sizes as an operational metric alongside the oracle-efficiency curve.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Strategy code (ours — `glue/`, unchanged this entry, the object under test):**
- `./glue/samplers/lsdflow/campaign.py` — `HubBatchingStrategy` (stamps `source_hub` on every accepted
  mode — the field this analysis groups by), `FreeFragChildPolicy` wiring, `rank_fragments` for the
  pre-select-K stock.
- `./glue/samplers/lsdflow/child_select.py` — `FreeFragChildPolicy` (the child policy used).

**Analysis driver (ours — new this entry):**
- `./experiments/lsd_hubs/campaign/batch_size_distribution.py` — re-uses `run_campaign.py`'s loaders
  (`_load_candidates`, `_load_enumerated_hubs`, `build_strategy`) + the `preselect_sweep.py` machinery,
  runs the free-frag + pre-select-K campaign to the mode budget for each K, groups accepted modes by
  `source_hub` into per-hub batch sizes, and writes the distribution + imbalance stats + two figures.
  Diagnostic only (no balancing); pure-CPU re-selection over the cached enumeration (no GPU, no
  re-docking).

**Results (committed — small artifacts):** `./experiments/lsd_hubs/campaign/results/scent_seh_batch_dist/`
- `batch_stats.csv` — one row per K: n_hubs, total_modes, min/median/mean/max batch, max/min ratio,
  CV, Gini, top-1/top-5 hub share.
- `batch_per_hub.csv` — every used hub's batch size (accepted hits) + its raw enumerated-children
  count + depth, ranked largest-first per K (shows the raw→kept collapse).
- `batch_summary.json` — inputs + per-K stats.
- `batch_distribution.png` — box+strip of per-hub batch sizes per K, and imbalance (Gini / top-share /
  max-min ratio) vs K.
- `batch_ranksize.png` — sorted (rank-size) batch curves per K: the dominance shape in one view.

**Inputs (on `$SCRATCH`, `/scratch/markymoo/rgfn_runs/`, git-ignored — same as `037`):**
- `lsdflow/scent_seh_70189/{records.csv,compositions.json}` — SCENT sEH analysis DAG (candidate pool +
  per-molecule promoted fragments).
- `lsdflow/campaign_enum_seh_70363/enum_children.json` — the 200-hub enumeration: each ranked hub's
  children + `added_promoted` (200 hubs, 828,448 enumerated children).
- `experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` —
  recipe snapshot with nested build costs (`smiles_to_route`), for the cost table + pre-select ranking.

## Relevant Versions

Branch `Hub-Analysis`. `batch_size_distribution.py` + `results/scent_seh_batch_dist/` are **new and not
yet committed.** The `glue/` strategy code under test is committed at `63fdede` ("In depth k
investigation for pre-select k fragments").

Files to commit: `experiments/lsd_hubs/campaign/batch_size_distribution.py` and
`experiments/lsd_hubs/campaign/results/scent_seh_batch_dist/`. [TODO — add commit hash after pushing.]

## Relevant Resources

**Sources** — entries `037` (pre-select-K, the campaign this reuses verbatim), `033` (fair count-once
cost model), `029` (naive hub-batching), `035` (mode-threshold robustness). `[gainski2025scent]`
(SCENT dynamic library), `[bengio2021gflownet]` (modes / diverse top-k).

**Packages** — `rgfn` conda env (activated via `~/bin/rgfn-smoke-env.sh`); RDKit (Tanimoto/Murcko in
`mode_select`/`campaign`), matplotlib (figures). All pure-CPU on the Balam login node.

## Method

1. Ran the batch-size analysis over the cached 200-hub SCENT sEH enumeration for K ∈ {0,16,50,100,200},
   free-frag child policy, build-score pre-select ranking, reward bar 7.0, Tanimoto cutoff 0.5, mode
   budget 300 (identical config to entry `037`'s Pareto):

   ```
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/campaign/batch_size_distribution.py \
     --analysis-dir /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189 \
     --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
     --snapshot /scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
     --reward-threshold 7.0 --tag scent_seh_batch_dist
   ```
2. For each K: built `HubBatchingStrategy(FreeFragChildPolicy, prebuilt_fragments=top-K)`, ran to the
   300-mode budget, grouped `result.accepted` by `source_hub` → per-hub batch sizes, computed
   distribution + imbalance stats (Gini via the sorted-rank formula; CV = pop-stdev/mean; top-N share =
   sum of N largest batches / total modes), and recorded each used hub's raw enumerated-children count
   for the raw→kept comparison.

## Results

**Per-hub batch-size distribution, 300-hit library, SCENT sEH (enum `70363`, cutoff 0.5, bar 7.0).**
Batch size = accepted diverse hits charged to a hub.

| K | hubs used | modes | min | median | mean | max | max/min | max/median | Gini | CV | top-1 share | top-5 share |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 (free-frag) | 57 | 300 | 1 | 4 | 5.26 | 17 | 17.0 | 4.25 | 0.443 | 0.813 | 0.057 | 0.24 |
| 16 | 45 | 300 | 1 | 5 | 6.67 | 21 | 21.0 | 4.20 | 0.385 | 0.720 | 0.070 | 0.28 |
| 50 | 29 | 300 | 1 | 10 | 10.35 | 28 | 28.0 | 2.80 | 0.369 | 0.680 | 0.093 | 0.38 |
| 100 | 19 | 300 | 2 | 13 | 15.79 | 37 | 18.5 | 2.85 | 0.366 | 0.650 | 0.123 | 0.503 |
| 200 | 10 | 300 | 7 | 26.5 | 30.0 | 65 | 9.29 | 2.45 | 0.373 | 0.673 | 0.217 | 0.813 |

Reading it: the biggest single synthesis step anywhere in the sweep is **65** products (K=200); at the
low-K, many-small-steps end it is **17** (K=0). Raising K packs the same 300 hits into fewer hubs, so
batches grow and the worst hub-to-hub ratio *falls* (17× → 9×) and Gini *falls* (0.44 → 0.37) — the
used hubs become relatively more uniform — but concentration rises (the single biggest hub goes from
6% to 22% of the library; top-5 from 24% to 81%). (Reactions/mode + oracle-call columns match `037`
exactly, confirming the same campaign: K=0 → 1.22 rxn/mode / 315,539 calls; K=200 → 1.85 / 52,865.)

**Raw fan-out → kept batch collapse (K=0, hubs with the largest enumerable fan-out that were used):**

| raw one-reaction children | kept diverse hits (batch) | depth | hub |
|---|---|---|---|
| 11,968 | 7 | 1 | `Brc1ccc(NCC2CCN2)cc1` |
| 11,968 | 2 | 1 | `Brc1ccc(NCC2Cc3ccccc3CN2)cc1` |
| 9,878 | 1 | 1 | `OB(O)c1cccc(NCC2Cc3ccccc3CN2)c1` |
| 9,487 | 13 | 1 | `O=C(O)c1ccc2[nH]c(C3CNC3)nc2c1` |

The scaffold that can make ~12,000 different products contributes only a **single-digit** kept batch
— because distinct diverse hits need distinct fragments, and only a few clear the reward bar *and* are
already available under free-frag. The raw enumeration count is the oracle-scoring load (the `037`
axis), **not** the synthesis batch size; they differ by ~100–1000×. Even the largest kept batch (65,
K=200, hub `O=C(O)c1ccc2nc(C3CNC3)oc2c1`) comes from a raw fan-out of 9,428.
