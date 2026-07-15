# SCENT — a fair cost model: fixing the fragment double-count + crediting accidental hub-batching
**Date:** 2026-07-14, ~2pm

## Question

When we compare "batch from hubs" against "just pick the best candidates" on synthesis reactions per
diverse hit, is the comparison actually fair — and how much of hub-batching's apparent advantage
survives once both strategies count every reusable piece exactly once?

## Context & Summary

Entry `029` reported that hub-batching "roughly halves" the synthesis cost per hit (820 vs 1,479
reactions for 300 sEH modes at diversity cutoff 0.5). Two things about that comparison were unfair,
and this entry corrects both:

1. **The cost model double-counted dynamic-library fragments.** SCENT's per-molecule `num_reactions`
   is *fully nested* — it already includes the reactions that build every promoted fragment the
   molecule attaches (confirmed in SCENT's own `reaction_env.py`: the seed carries its fragment's
   `num_reactions`, each attached reactant adds its fragment's `num_reactions`, and each coupling
   adds +1; and empirically, for all 245 promoted fragments that appear as intermediates,
   `num_reactions` equals their full nested build cost exactly). The campaign then *added* the
   fragment builds a second time. Net effect: the per-molecule assembly term was ~2× inflated before
   the extra fragment charge even landed. The docstring said "count once"; the code did not.

2. **Best-candidate got no credit for the hubs it shares by accident.** When you pick the top-N
   diverse molecules, several of them descend from a common parent scaffold. A chemist building that
   library would make the shared scaffold once — but the old model charged every molecule its full
   assembly independently, inflating best-candidate and flattering hub-batching.

The fix is one **count-once model applied identically to both strategies**: a molecule's cost splits
into (a) *assembly couplings* — the shallow join steps, `num_reactions − Σ(nested build cost of each
attached promoted fragment)` — and (b) *promoted-fragment builds*, each distinct fragment charged
once. Shared scaffolds ("hubs") have their couplings charged once too: by design for hub-batching, and
now *accidentally* for best-candidate, which assigns each mode to the parent hub most reused among the
accepted modes (a swappable policy; immediate-parent hubs from `records.csv`). The only remaining
difference between the two strategies is the **selection** — the accounting is now shared.

## Answer

**Most of hub-batching's "halving" was a cost-model artifact, not a real synthesis saving.** With the
corrected count-once model, building 300 diverse sEH hits at cutoff 0.5 costs best-candidate **929
reactions (3.10 / mode)** vs hub-batching's **819 (2.73 / mode)** — hub-batching still wins, but by
**~1.13×, not the ~1.8× of entry 029**. Decomposing best-candidate's drop from the old 1,479: the
**double-count fix accounts for 530 of the 550 reactions saved** (1,479 → 949), and crediting
accidental hub-sharing accounts for only **~20 more** (949 → 929, 254 shared hubs across 300 modes).
Accidental sharing is small here because diverse modes tend to have diverse scaffolds — but it grows
as the cutoff loosens (near-identical modes share parents): at cutoff 0.90 best-candidate falls to 728
reactions. In fact, with 200 enumerated hubs, at loose cutoffs (≥ 0.75) best-candidate **beats**
hub-batching on raw reactions (e.g. 0.80: best 722 vs hub 828). Hub-batching's durable advantage is
therefore narrower and lives at strict-to-mid cutoffs; and its real remaining cost is not synthesis
but the **7k–19k reward-generator evaluations** it spends enumerating hub neighborhoods (best-candidate
spends 0). The honest headline is "hub-batching modestly reduces synthesis reactions at meaningful
diversity, paid for with a large enumeration-scoring bill," not "halves cost."

## Relevance to our Publication

This is a correctness fix to the headline cost comparison, exactly the kind of thing a careful
reviewer (or chemist) checks: do both strategies pay for shared intermediates the same way? They now
do, on one auditable model, so the reactions-per-mode numbers are defensible. It also reframes the
contribution more honestly — hub-batching trades enumeration-scoring calls for a modest synthesis
saving — which is the precise axis the active-learning loop (Objective 1's oracle-efficiency curve)
has to balance, and it keeps the two strategies swappable behind the same `CampaignResult` interface.

## Next Experiments

**Refining for publication**
- **DRD2** through the same corrected model (both the head-to-head and the sweep) to show the fair
  comparison isn't sEH-specific.
- **Full-backbone sharing** (deeper than immediate-parent): needs the sample re-run's per-molecule
  `routes.json`, then count-once over the union of full routes. Would credit best-candidate for
  shared *grandparent* scaffolds too — expected to shrink hub-batching's edge a little further.
- **Alternate hub-assignment policies** (the interface is swappable): cheapest-parent and
  canonical/most-frequent, to bound how policy-sensitive the accidental-batching credit is.

**Next steps in project**
- Wire both strategies into the active-learning loop on the corrected cost and measure the
  oracle-efficiency curve (`[bengio2021gflownet]` Fig. 7 analogue) — reward-gen calls vs diverse hits.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Corrected cost model (ours — `glue/`):**
- `./glue/samplers/lsdflow/campaign.py` — the rewrite. New `shallow_couplings()` (nested
  `num_reactions` → assembly couplings); `HubAssignmentPolicy` ABC + `MostSharedAssignment` (default,
  swappable) + `NoHubSharing`; `BestCandidateStrategy` now three-pass (select → assign shared hubs →
  count-once cost) with a validity filter (`couplings(hub) < couplings(mode)`, so a cross-trajectory
  parent can never inflate the marginal); `HubBatchingStrategy` charges `shallow_couplings(hub)`
  instead of the nested depth. Both strategies now share the identical accounting.
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — `FragmentCostTable` (unchanged;
  `shared_build_cost([f])` supplies each fragment's nested build cost for the couplings subtraction).

**Analysis drivers (ours — `experiments/lsd_hubs/campaign/`):**
- `run_campaign.py` — `_load_candidates` now also reads each terminal's observed parent hubs
  (`records.csv` `hub_key`); new shared `build_strategy()` constructs either strategy on the corrected
  model (passes `compositions` + the assignment policy to best-candidate). Used by both drivers.
- `sweep_campaign.py` — threads `compositions` through `build_strategy` (via `_run`); plots unchanged.

**Inputs (on $SCRATCH):** sEH analysis DAG `lsdflow/scent_seh_70189/` (`records.csv`,
`compositions.json`); enumerations `lsdflow/campaign_enum_seh_70295/` (50 hubs) +
`campaign_enum_seh_70363/` (200 hubs); recipe snapshot
`fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`.

**Results (committed):** `experiments/lsd_hubs/campaign/results/scent_seh/` (50-hub) +
`results/scent_seh_1kx200/` (200-hub) — regenerated `summary.json`, `curve_*.csv`,
`{pareto,fixed_modes,budget_efficiency}.{csv,png}`, `sweep_summary.json`.

## Relevant Versions

Branch `Hub-Analysis`. Corrected `campaign.py` + the two drivers + regenerated `results/` +
supersede notes on `029`/`031`. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources** — entries `029` (the comparison this corrects), `028` (the `FragmentCostTable`), `027`
(SCENT cross-env adapter). `[gainski2025scent]` (dynamic library), `[bengio2021gflownet]` (modes/top-k).

**Packages** — SCENT clone `external/scent` (`rgfn/gfns/reaction_gfn/reaction_env.py` — the
`num_reactions` accounting confirmed here); `rgfn` env (campaign, CPU).

## Method

1. Traced SCENT's state `num_reactions` to `reaction_env.py`; confirmed it is fully nested. Verified
   empirically: for all 245 promoted fragments in `compositions.json`, `num_reactions` == the nested
   closure cost (never the shallow step count). Confirmed `couplings = num_reactions − Σ(nested frag
   builds)` is a clean 1–4 for the top-1000 (0 negatives).
2. Rewrote `campaign.py` to the one count-once model (shallow couplings + fragments-once + shared
   hubs, both strategies); added the swappable `MostSharedAssignment`; added the prefix-validity
   filter after an initial version let a cross-trajectory parent *raise* best-candidate's cost
   (sharing must never exceed no-sharing — now asserted in the isolation test).
3. Regenerated both result sets (`run_campaign.py` + `sweep_campaign.py`, CPU, ~42 s / ~73 s).

## Results

**Head-to-head (sEH, hit bar ≥ 7.0, cutoff 0.5):**

| metric | best-candidate | hub-batching |
|---|---|---|
| reactions to generate 300 modes | 929 (3.10 / mode) | **819 (2.73 / mode)** |
| Case 1: modes at 100 reactions | 29 | **33** |
| reward-gen calls (/ mode) | 0 (0) | 18,856 (62.9) |
| distinct promoted intermediates | 205 | 328 |
| shared hubs used | 254 (accidental) | 2 (by design) |
| Bemis-Murcko scaffolds (of 300) | 300 | 300 |

**Effect decomposition (best-candidate, 300 modes, cutoff 0.5):**

| model | reactions | / mode |
|---|---|---|
| old (nested `num_reactions` + fragment charge = double-count) | 1,479 | 4.93 |
| corrected count-once, no hub sharing | 949 | 3.16 |
| corrected + accidental hub credit (most-shared) | **929** | **3.10** |

→ hub-batching's edge: **1.81× → 1.13×**. Double-count fix saved 530 reactions; accidental hubs saved
20 more.

**Sweep — reactions to reach 300 modes (corrected):**

| cutoff | 50-hub best | 50-hub hub | 200-hub best | 200-hub hub |
|---|---|---|---|---|
| 0.30 (strict) | 1,123 | can't (135 max) | 1,123 | 882 |
| 0.50 | 929 | 819 | 929 | 816 |
| 0.70 | 793 | 706 | 793 | 770 |
| 0.80 | 722 | 656 | **722** | 828 |
| 0.90 (loose) | 728 | 644 | **728** | 822 |

(best-candidate is identical across the two enumerations — it never uses hub data; only its accidental
sharing, read from `records.csv`, matters.) **Findings:** (a) hub-batching still wins at strict-to-mid
cutoffs, but by ~1.1–1.3× (not ~2×). (b) best-candidate's cost *falls* as diversity loosens (more
accidental scaffold sharing among near-identical modes). (c) with 200 hubs, at loose cutoffs (≥ 0.75)
best-candidate **beats** hub-batching on reactions — hub-batching's residual cost there is the
enumeration-scoring bill (7k–19k reward-gen calls), not synthesis. Per-hub stats (`hub_stats.csv`) and
the strict-cutoff ceiling behaviour (entry `031`) are unaffected — they don't depend on the cost model.
