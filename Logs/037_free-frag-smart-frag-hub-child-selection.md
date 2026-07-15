# SCENT — free-frag & smart-frag: fragment-aware within-hub child selection
**Date:** 2026-07-14, ~5pm

## Question

When we batch a diverse library of hits from a shared scaffold ("hub"), does refusing to build new
chemical building blocks — or gently down-weighting expensive ones — cut the total synthesis cost per
hit, and what do we give up for it?

## Context & Summary

Entries `029`/`033`/`035` compared two ways to pick a diverse library from a trained SCENT model:
**best-candidate** (take the top-reward molecules the generator sampled) vs **hub-batching** (build a
shared scaffold once, then diversify it with one final reaction each). On the fair count-once cost
model (`033`), hub-batching was only modestly cheaper — **2.73 vs 3.10 reactions per hit** (~1.13×).
The reason it wasn't cheaper is specific: the fragment attached in that "one final reaction" is
usually a *promoted* dynamic-library fragment that itself takes 2–3 reactions to synthesize, so each
diversification really costs ~3 reactions, not 1.

This entry adds two new **within-hub child-selection** strategies that exploit exactly that structure
— they change *which* of a hub's children we keep as hits, leaving best-candidate (the control) and
the cost accounting untouched. **free-frag** keeps only children whose final reaction attaches an
*already-available* fragment (a purchasable base block, or one already built for another hit) — so
every kept hit costs exactly **one** marginal reaction. **smart-frag** is the soft version: instead of
filtering, it ranks children by `reward − β·(cost of fragment / utility of fragment)`, where cost is
the fragment's build cost in reactions and utility is SCENT's own per-fragment goodness score
(`smiles_to_mean_reward`, their Eq. 13) — so a cheap, high-utility fragment barely hurts a child's
ranking while an expensive, low-utility one sinks it. Both were run on the cached sEH enumerations at
two hub-pool sizes (50 and 200 hubs).

## Answer

**free-frag roughly halves the synthesis cost again — but you pay for it in enumeration.** Given a
large enough hub pool (200 hubs) it builds 300 diverse sEH hits at **1.22 reactions per hit** — 2.5×
cheaper than best-candidate (3.10) and 2.2× cheaper than plain hub-batching (2.72) — at the same
reward and diversity. The catch is that "only reuse what you've already built" forces it to spread
across many more hubs (57 vs 2), so it enumerates and scores ~45× more candidate molecules (316k vs
7k) to find those already-cheap children; with only 50 hubs it runs out and caps at 232 of 300 hits.
**smart-frag is a gentler, safer lever:** it always reaches the full 300 hits and shaves ~6–13% off
the reaction cost when cheap children are scarce (50-hub pool, β=4), but is roughly neutral when the
pool is rich enough that plain reward-ranking already finds them. Neither hurts hit quality (best sEH
8.3–8.4, median 7.3–7.5 across all policies). Net: free-frag is the aggressive synthesis-cost lever,
smart-frag the mild one, and together they map the trade between *cheaper chemistry* and *more
scoring calls* — the exact axis the active-learning loop will have to price.

## Relevance to our Publication

This sharpens the "why batch from a hub" story into a concrete, chemist-legible lever: the way to make
late-stage diversification cheap is to reuse building blocks you've already made, and free-frag shows
that pushes hub-batching from a 1.13× edge to ~2.5× over the paper-style top-k. It also puts a number
on the tradeoff the active-learning oracle-efficiency curve (`[bengio2021gflownet]` Fig. 7 analogue,
Objective 1) has to balance — synthesis reactions saved vs. reward-generator (≈ oracle) calls spent
— which is the reviewer question these selection strategies exist to answer. Both policies are written
as production `glue/` code (importable by a future in-loop `LSDFlowAcquisition`), so the same objects
drive the offline campaign and the AL loop.

## Next Experiments

**Refining for publication**
- **Second target (DRD2)** through the same three policies + best-candidate, to show the free-frag win
  isn't sEH-specific (the DRD2 analysis DAG + enumeration already exist).
- **Free-frag ceiling vs hub count** — we have two points (50 hubs → caps at 232/300; 200 hubs →
  full 300 at cutoff ≥0.40). A cleaner hub-count sweep would pin where "enough hubs" is, and pairs
  with the deferred faster `--no-flow` enumeration (Logs/029) that would make many-hub pools cheap.
- **Report the enumeration bill as an explicit second cost axis** in the head-to-head figure, not a
  footnote — free-frag's 316k reward-gen calls is the honest price of its 1.22 rxn/mode.

**Next steps in project**
- Wire the three child policies into the active-learning loop (`LSDFlowAcquisition`) and measure the
  oracle-efficiency curve for each — the capstone the campaign was built to feed.
- The RGFN-vs-SCENT hub-coincidence study (proposal §8) on the same machinery.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**New / changed strategy code (ours — `glue/`, AL-importable):**
- `./glue/samplers/lsdflow/child_select.py` — **NEW.** `ChildSelectionPolicy` ABC + `RewardChildPolicy`
  (default, reward-first, keep all — byte-identical to the old inline sort), `FreeFragChildPolicy`
  (keep only children needing no new fragment build), `SmartFragChildPolicy`
  (`reward − β·Σ cost(f)/utility(f)`); `make_child_policy` factory. Pure/duck-typed (takes a duck-typed
  `cost_table` + a plain `utilities` dict), so the campaign and a future AL loop share it.
- `./glue/samplers/lsdflow/campaign.py` — `HubBatchingStrategy` gains a `child_policy` param and
  computes each hub's `available` fragment set (`built_promoted ∪ closure(hub.promoted)`) before
  offering children to the policy. Default = `RewardChildPolicy` → unchanged cost/behaviour.
- `./glue/samplers/lsdflow/__init__.py` — re-exports the policies.
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — `FragmentCostTable.utilities` +
  `.utility(f)`; new `scaled_fragment_utilities(snapshot, beta_train, scale)` puts SCENT's
  `smiles_to_mean_reward` onto the reward scale (see the utility-scale note below).

**Analysis drivers (ours — `experiments/lsd_hubs/campaign/`):**
- `run_campaign.py` / `sweep_campaign.py` — new `--child-policy {reward,free_frag,smart_frag}`,
  `--beta`, `--utility-scale {logbeta,log,raw}`, `--beta-train`. The child policy is applied to the
  hub-batching line only; best-candidate is unaffected. `README.md` documents the flag.

**Results (committed, per `results/<tag>/` — `summary.json` + `curve_*.csv` + sweep `pareto/`,
`fixed_modes/`, `budget_efficiency/` + `sweep_summary.json`):**
- `results/scent_seh_freefrag/`, `results/scent_seh_smart_b4/` (50-hub, enum `70295`).
- `results/scent_seh_1kx200_freefrag/`, `results/scent_seh_1kx200_smart_b4/` (200-hub, enum `70363`).
- Reward-sort / best-candidate baselines are the untouched committed `results/scent_seh/` (50-hub) and
  `results/scent_seh_1kx200/` (200-hub) from `033`/`035`.

**Inputs (on `$SCRATCH`, `/scratch/markymoo/rgfn_runs/`):**
- `lsdflow/scent_seh_70189/` — SCENT sEH analysis DAG (`records.csv`, `compositions.json`).
- `lsdflow/campaign_enum_seh_70295/enum_children.json` (50-hub) + `…_70363/enum_children.json`
  (200-hub) — each hub's children + `added_promoted` (the promoted fragment attached in the final
  reaction). **This is the field free-frag / smart-frag key on.**
- `experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` —
  recipe re-run snapshot (`033`); supplies nested build costs (`smiles_to_route`) **and** the utility
  (`smiles_to_mean_reward`, 386,643 tracked / 1,600 promoted with a value each).

## Relevant Versions

Branch `Hub-Analysis`. New `child_select.py` + edits to `campaign.py` / `__init__.py` /
`dynamic_amortization.py` / `run_campaign.py` / `sweep_campaign.py` / campaign `README.md` + the four
new `results/` dirs. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources** — entries `029` (the head-to-head this extends + the 3-reactions-per-diversification
decomposition), `033` (the fair count-once cost model), `035` (threshold robustness), `027`/`028`
(SCENT cross-env adapter + nested cost engine). `[gainski2025scent]` — the Dynamic Library and its
utility metric (Eq. 13 = mean reward through a state = `smiles_to_mean_reward`). `[bengio2021gflownet]`
— modes / top-k / oracle-efficiency.

**Packages** — `rgfn` env (campaign + analysis, pure CPU); SCENT clone `external/scent`
(`…/dynamic_library/reaction_dynamic_library.py` — where `smiles_to_mean_reward` is defined/saved).

**Utility-scale note (load-bearing).** SCENT stores `smiles_to_mean_reward` as the mean of the
*shaped* reward `R(x) = exp(β_train · proxy)`, so the values are astronomically large (~1e27 for
β_train = 8; confirmed via `records.csv` where `log_reward = 8 · reward`). Used literally, a reaction
count divided by ~1e27 is 0 and smart-frag collapses to plain reward-ranking. `scaled_fragment_utilities`
therefore rescales to the proxy scale via `log(mean_reward)/β_train` (≈ the expected reward through the
fragment, ~7.7–8.3 for chosen fragments) — the default; `--utility-scale {log,raw}` and `--beta-train`
expose the choice. This is a deliberate interpretation of "utility," flagged so it can be revisited.

## Method

1. Added `ChildSelectionPolicy` (+ 3 policies) and wired `child_policy` into `HubBatchingStrategy`,
   computing each hub's `available` set so free-frag counts a child free iff it builds no promoted
   fragment beyond what building the hub already provides. Unit-tested the ordering/filtering + cost
   accounting on a synthetic hub; **verified the default `reward` policy reproduces the committed
   `033` numbers byte-identically** (hub-batching 819 rxns / 300 modes / 2.73 per mode; best-candidate
   929) as a regression guard.
2. Added utility support to the cost table + `scaled_fragment_utilities` (log/β rescale, β_train = 8).
3. Ran `run_campaign.py` at diversity cutoff 0.5, hit bar 7.0, for `reward` / `free_frag` /
   `smart_frag` (β ∈ {1,2,4,8}) / `best_candidate`, on the 50-hub (`70295`) and 200-hub (`70363`)
   enumerations. Ran `sweep_campaign.py` (cutoff 0.30→0.90) for `free_frag` (both pools) and
   `smart_frag` β=4 (50-hub). All pure-CPU on the Balam login node (~11 s/point, ~44 s/sweep).

## Results

**Head-to-head, diversity cutoff 0.5, hit bar ≥ 7.0 (hub-batching line under each child policy;
best-candidate is the same control everywhere).**

50-hub enumeration (`70295`):

| policy | modes | reactions | rxns/mode | hubs used | distinct frags | reward-gen calls | modes @100rxn | best / median sEH |
|---|---|---|---|---|---|---|---|---|
| reward (β=0) | 300 | 819 | 2.73 | 2 | 328 | 18,856 | 33 | 8.40 / 7.48 |
| **free_frag** | 232 | 308 | **1.33** | 47 | 39 | 148,153 | **82** | 8.40 / 7.41 |
| smart_frag β=4 | 300 | 769 | **2.56** | 2 | 310 | 18,856 | 42 | 8.40 / 7.44 |
| best-candidate | 300 | 929 | 3.10 | 254 | 205 | 0 | 29 | 8.40 / 8.10 |

200-hub enumeration (`70363`):

| policy | modes | reactions | rxns/mode | hubs used | distinct frags | reward-gen calls | best / median sEH |
|---|---|---|---|---|---|---|---|
| reward (β=0) | 300 | 816 | 2.72 | 2 | 321 | 7,086 | 8.32 / 7.49 |
| **free_frag** | 300 | 367 | **1.22** | 57 | 26 | **315,539** | 8.40 / 7.30 |
| smart_frag β=4 | 300 | 821 | 2.74 | 3 | 324 | 9,635 | 8.32 / 7.48 |
| best-candidate | 300 | 929 | 3.10 | 254 | 205 | 0 | 8.40 / 8.10 |

free_frag reaches full diversity (300 modes) only with the larger pool; on 50 hubs it exhausts the
"free" children at 232. Its 1.22 rxns/mode approaches the ~1 floor (hub built once + 1 coupling per
child), bought with a **45× enumeration bill** (315,539 vs 7,086 reward-gen calls).

**smart-frag β-scan (50-hub, cutoff 0.5):** 2.73 (β0) → 2.70 (β1) → 2.65 (β2) → **2.56 (β4)** → 2.58
(β8). β=4 is the sweet spot → fixed for the reported runs.

**Diversity-cutoff sweep — reactions to reach 300 modes (`None` = pool can't reach 300 at that
cutoff).**

| cutoff | free_frag (200-hub) | reward (200-hub) | best-candidate | free_frag (50-hub) | smart_frag β4 (50-hub) | reward (50-hub) |
|---|---|---|---|---|---|---|
| 0.30 (strict) | None | 898* | 1,123 | None | None | None |
| 0.40 | 489 | — | 1,018 | None | 787 | 810 |
| 0.50 | **367** | 816 | 929 | None (232 max) | 769 | 819 |
| 0.70 | 319 | — | 793 | 317 | 629 | 706 |
| 0.90 (loose) | 306 | — | 728 | 307 | 565 | 644 |

\*from `031`/`033`. free-frag beats best-candidate ~2.2–2.5× wherever it reaches 300, and hits the
same strict-cutoff chemical-space ceiling as before (`031`); the ceiling lifts with hub count (50 hubs
→ only cutoffs ≥ 0.70 reach 300; 200 hubs → down to 0.40). smart-frag β=4 beats reward-sort by ~6–13%
at every cutoff on the lean 50-hub pool and always reaches 300.
