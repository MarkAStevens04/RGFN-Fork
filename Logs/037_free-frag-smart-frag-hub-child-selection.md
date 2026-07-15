# SCENT — fragment-aware hub-batching: free-frag, and pre-select-K for cheap enumeration
**Date:** 2026-07-14 → 07-15

## Question

The synthesis-cheapest way to batch a diverse hit library from hubs (free-frag) has to walk *many*
hubs, which means enumerating and scoring a huge number of candidate molecules — an expensive bill
that in the active-learning loop is paid in **oracle calls**. Can we cut that bill — without giving up
the cheap synthesis — by pre-committing to a few widely-reusable building blocks up front?

## Context & Summary

Earlier entries (`029`/`033`/`035`) compared **best-candidate** (top-reward sampled molecules) vs
**naive hub-batching** (build a shared scaffold once, diversify it by attaching one fragment per hit).
Naive hub-batching costs ~2.7 reactions/hit because the attached fragment is usually a *promoted*
dynamic-library block that itself takes 2–3 reactions to make. This entry makes hub-batching
fragment-aware and chases a second cost axis — the enumeration/oracle bill — that the AL loop actually
cares about.

- **free-frag** keeps only children whose final reaction attaches an *already-available* fragment
  (base stock, or one already built) → each hit costs exactly **1** marginal reaction → **1.22
  rxn/hit** (2.5× cheaper than best-candidate). But diversity *within* a hub requires *distinct*
  fragments (measured: 241/270 fragments serve exactly one diverse mode), so free-frag gets its
  diversity from **base** fragments and must walk **57 hubs** to find 300 diverse base-decorated hits
  → **315k reward-gen (≈ oracle) calls**, 45× the naive 7k. That enumeration bill is the real cost.
- **Structural key:** promoted fragments are massively **cross-hub** — each of the 1,600 appears (as a
  hit child) in a *median of ~50* of the 200 hubs. So a *few* widely-reused blocks could serve diverse
  hits across many hubs, if we commit to building them.
- **pre-select-K** (the winner): compute each fragment's fan-out; pre-synthesize the **top-K by
  build-score** `(reward − bar)·fanout / build_reactions`; charge them **once** up front; then run
  free-frag. Now each hub yields more free children (base **+** the K stock), so we walk fewer hubs →
  fewer calls, for the price of K upfront builds. K is a clean reactions↔calls dial.

**Explored and removed** (kept lean per the researcher's call): a *static* **smart-frag**
(`reward − β·cost/utility` with SCENT's `smiles_to_mean_reward` utility) — the utility is compressed
into a ±4% band so it can't discriminate fragments; and a *dynamic* **fan-out move-on** policy — it
kept building fragments and never reached the low-reaction regime. Both are documented here as
negatives; their code was deleted.

## Answer

**Pre-select-K dominates free-frag on the enumeration axis at essentially no synthesis cost — in a
small-K window it dominates on *both* axes at once.** Stocking the top ~16 universal blocks is not just
a free lunch, it's a small *net win*: total reactions actually dip slightly **below** free-frag
(364 vs 367 at K=16) *and* oracle calls fall ~21% (248k vs 315k) — because each pre-built high-fan-out
block (~2 reactions) lets us skip hubs whose scaffold builds cost more than that. So there is no reason
to run plain free-frag (K=0); a small stock strictly beats it. Past that, K is a clean dial:
**K=100 → 1.41 rxn/mode at 85k calls (−73%)**, K=200 → 1.85 at 53k (−83%); median reward even drifts
*up* (7.30 → 7.51),
because the high-build-score blocks are high-reward. The pre-selected fragments are chemically sensible
**universal building blocks** — bromo-aryl amides/amines with small saturated N-heterocycles
(azetidine/pyrrolidine), each usable in 130–175 of the 200 hubs. The knob is exactly "how many good
reusable blocks to stock," which a chemist reads directly. Fan-out is the load-bearing signal (spans
5–187 hubs); per-fragment max reward is compressed near the top (a knife-edge), and SCENT's mean-reward
utility is compressed further still (why smart-frag failed).

## Relevance to our Publication

This is the **oracle-efficiency** axis of the "why batch from a hub" story. In the AL loop the scarce
budget is oracle calls, and pre-select-K cuts them **~2–6×** at ~constant synthesis cost by
front-loading a handful of reusable blocks — the exact trade the oracle-efficiency curve
(`[bengio2021gflownet]` Fig. 7 analogue, Objective 1) has to make. It is chemist-legible ("stock these
20 blocks, then diversify"), the code is production `glue/` (AL-importable), and the pieces are
swappable for ablations (child policy + pre-select ranking are registries).

## Next Experiments

**Refining for publication**
- **Second target (DRD2)** — repeat the pre-select-K sweep (analysis DAG + enumeration already exist)
  to show the win isn't sEH-specific.
- **Ranking ablation** — the `--rank-by` flag (build_score / fanout / reward) is one switch; report how
  much the reward-weighting and cost-division each buy (measured start: fanout-only gives slightly
  fewer calls, reward-only gives higher median reward but more reactions).
- **Cutoff sweep for pre-select-K** — trace the reactions↔calls Pareto across diversity cutoffs.

**Next steps in project**
- Wire naive/free-frag/pre-select-K into the active-learning loop (`LSDFlowAcquisition`) and measure
  the oracle-efficiency curve — the capstone the campaign was built to feed.
- The RGFN-vs-SCENT hub-coincidence study (proposal §8): are the pre-selected universal blocks the same
  intermediates SCENT promoted?

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Strategy code (ours — `glue/`, AL-importable). Lean set after the rework:**
- `./glue/samplers/lsdflow/child_select.py` — `ChildSelectionPolicy` ABC + two policies:
  `RewardChildPolicy` (**naive** hub-batching — reward-first, keep all, no fragment-cost awareness;
  byte-identical to the historical sort) and `FreeFragChildPolicy` (keep only already-available-fragment
  children). `make_child_policy` factory. To add an ablation policy: subclass + register in `_POLICIES`.
- `./glue/samplers/lsdflow/campaign.py` — `HubBatchingStrategy` gains `prebuilt_fragments` (pre-select-K
  stock, charged once via `_charge_promoted` → **no hub double-count**) + the fan-out/pre-select helpers
  `fragment_fanout()` and `rank_fragments(..., method=...)` with the swappable `RANK_METHODS`
  (`build_score` default / `fanout` / `reward`).
- `./glue/samplers/lsdflow/__init__.py` — re-exports the lean set + `fragment_fanout` / `rank_fragments`.
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — SCENT mean-reward utility machinery
  removed (the compressed-utility dead end); `$`-cost machinery retained.

**Analysis drivers (ours — `experiments/lsd_hubs/campaign/`):**
- `run_campaign.py` / `sweep_campaign.py` — `--child-policy {reward,free_frag}`, `--prebuild-k`,
  `--rank-by {build_score,fanout,reward}`.
- `preselect_sweep.py` — **NEW**, reusable: sweeps K (default coarse K = {0,5,10,20,50,100,200}),
  writes `preselect.csv` + `preselect_summary.json` + a 2-panel `preselect_pareto.png` (reactions↔calls
  Pareto **and** total-reactions-vs-K, the latter showing the small-K dip) next to the naive /
  best-candidate reference points.

**Results (committed):**
- `results/scent_seh_1kx200_preselect/` — the default coarse pre-select-K Pareto (200-hub, enum `70363`).
- `results/scent_seh_1kx200_preselect_0to50/` — the dense K=0…50 sweep (the small-K dip).
- `results/{scent_seh,scent_seh_1kx200}_freefrag/` — free-frag (K=0) baselines + cutoff sweeps.
- `results/{scent_seh,scent_seh_1kx200}/` — naive-hub-batching vs best-candidate baselines (`033`/`035`).

**Inputs (on `$SCRATCH`, `/scratch/markymoo/rgfn_runs/`):**
- `lsdflow/scent_seh_70189/` — SCENT sEH analysis DAG (`records.csv`, `compositions.json`).
- `lsdflow/campaign_enum_seh_{70295 (50-hub),70363 (200-hub)}/enum_children.json` — each hub's children
  + `added_promoted` (the promoted fragment attached in the final reaction) — the field everything keys on.
- `experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` —
  recipe snapshot (`033`); nested build costs (`smiles_to_route`).

## Relevant Versions

Branch `Hub-Analysis`. `child_select.py` / `campaign.py` / `__init__.py` / `dynamic_amortization.py` /
`run_campaign.py` / `sweep_campaign.py` edited; `preselect_sweep.py` new; `results/scent_seh_1kx200_preselect/`
new. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources** — entries `029` (naive hub-batching + the 3-reactions-per-diversification decomposition),
`033` (fair count-once cost model), `035` (threshold robustness). `[gainski2025scent]` (Dynamic
Library), `[bengio2021gflownet]` (modes / top-k / oracle efficiency).

**Packages** — `rgfn` env (campaign + analysis, pure CPU).

## Method

1. **Lean rework of the child-selection code:** kept `reward` (naive) + `free_frag`; added
   `prebuilt_fragments` to the strategy (pre-select-K) with the `fragment_fanout` / `rank_fragments`
   helpers; deleted the static smart-frag, the SCENT-utility machinery, and the fan-out move-on policy.
   Unit-tested: naive `reward` unchanged, the pre-select **double-count guard** (a stock fragment a hub
   scaffold also uses is charged once), pre-select unlock, and the three ranking methods. **Regression:
   naive `reward` reproduces the committed `033` numbers byte-identically (819 / 929).**
2. **Fan-out measurement** (read the cached 200-hub enumeration, no re-run): fan-out per fragment
   (median ~50 hit-hubs; 795/1,600 in ≥50 hubs) and per-fragment max reward (compressed near 8).
3. **Pre-select-K sweep** (`preselect_sweep.py`) K ∈ {0,5,10,20,50,100,200} on the 200-hub enumeration,
   cutoff 0.5, bar 7.0; + the `--rank-by` ablation at K=50. Pure-CPU on the Balam login node.

## Results

**Pre-select-K Pareto (200-hub enum `70363`, cutoff 0.5, bar 7.0). K=0 is plain free-frag.**

| strategy | rxns/mode | reactions | hubs used | distinct frags | reward-gen calls | median sEH |
|---|---|---|---|---|---|---|
| naive hub-batching (reward) | 2.72 | 816 | 2 | 321 | 7,086 | 7.49 |
| free-frag (K=0) | 1.22 | 367 | 57 | 26 | 315,539 | 7.30 |
| **pre-select K=20** | 1.22 | 365 | 43 | 37 | 241,158 | 7.30 |
| **pre-select K=50** | 1.26 | 378 | 29 | 61 | 158,660 | 7.34 |
| **pre-select K=100** | 1.41 | 422 | 19 | 108 | 85,165 | 7.44 |
| pre-select K=200 | 1.85 | 556 | 10 | 202 | 52,865 | 7.51 |
| best-candidate (control) | 3.10 | 929 | 254 | 205 | 0 | 8.10 |

K≤20 cuts calls at no reaction cost (free lunch); higher K trades a little synthesis for large call
savings. Best/median sEH stay 8.3–8.4 / 7.3–7.5 throughout.

**Small-K dip — pre-select strictly beats free-frag on both axes (dense sweep K=0…50).** Resolving K at
integer resolution (`preselect_sweep.py --k-list 0,1,…,50`) shows total reactions *dip below* free-frag
(K=0 = 367) across a broad shallow basin **K≈12–31**, minimum **K=16 → 364 (−3)** (also 364 at K=18/19/23):

| K | reactions (Δ vs K0) | hubs | oracle calls | median |
|---|---|---|---|---|
| 0 (free-frag) | 367 | 57 | 315,539 | 7.302 |
| **16** | **364 (−3)** | 45 | **248,356** | 7.290 |
| 23 | 364 (−3) | 41 | 237,739 | 7.294 |
| 31 | 365 (−2) | 35 | 189,108 | 7.317 |
| 50 | 378 (+11) | 29 | 158,660 | 7.343 |

The mechanism is the one predicted: a pre-built block costs ~2 reactions but, being high-fan-out, unlocks
free children across many hubs, letting us skip hubs (57 → 45 at K=16) whose scaffold builds cost more
than the block. The dip is small (~0.8%) and a little noisy (dropping a hub is discrete, so reactions
wiggle ±1–3), but consistent; calls fall monotonically the whole way. Net: a small stock (K≈16–20) is
**Pareto-dominant over free-frag** — fewer reactions *and* ~21–25% fewer oracle calls. **K=0 is verified
byte-identical to the naive free-frag baseline** (367 rxns / 315,539 calls / 57 hubs / 26 frags — a
built-in control that `--prebuild-k 0` is the same code path). Dense curve + 2-panel figure (Pareto +
reactions-vs-K) in `results/scent_seh_1kx200_preselect_0to50/`; the committed default sweep
(`results/scent_seh_1kx200_preselect/`) keeps the coarse K = {0,5,10,20,50,100,200} range.

**Ranking ablation (`--rank-by`, K=50, 200-hub):** build_score → 1.26 rxn/mode, 159k calls, med 7.34;
**fanout** → 1.26, **142k** calls (fewest — widest reuse), med 7.35; **reward** → 1.32, 172k calls, med
**7.42** (highest reward, but pricier). One flag; isolates each signal.

**Fan-out distribution (the reason pre-select works):** all 1,600 promoted fragments appear in ≥5 hubs;
median 50 hit-hubs; 795 in ≥50 hubs. The top-K by build-score are bromo-aryl amide/amine blocks with
azetidine/pyrrolidine, fan-out 130–175, max reward ~8.3 — genuine universal building blocks.

**Removed negatives (documented, code deleted):** static smart-frag reduced reactions only ~6–13% on a
lean 50-hub pool and was ~neutral on 200 hubs (SCENT utility too compressed, ±4% band); the dynamic
fan-out move-on stayed at 2.1–2.6 rxn/mode (kept building fragments) and, pushed to more hubs via a
higher move-on threshold, exploded to 828k calls without reaching 300 modes.
