# SCENT / sEH + DRD2 — what a reaction count cannot see: reaction yield

**Date:** 2026-08-24, ~5pm

## Question

Our headline says "how many distinct good molecules do you get for 100 reactions?" — but does a
reaction count hide the fact that some reactions are much more likely to actually work than others,
and does our hub-batching method make that worse?

## Context & Summary

Every cost number in this project is a count of **reactions**. Entry [065] settled that as the
primary readout, and entries [050]/[056] built the headline on it: hub-batching delivers roughly
three times as many distinct molecules as picking the best candidates one at a time, at the same
budget of 100 reactions.

A reaction count treats all reactions as equal. Real ones are not: they have **yields** — the
fraction of material that survives the step. SCENT's own building-block library ships a per-reaction
yield column, spanning 0.65 to 0.95, which we had never looked at. So a 100-reaction library could in
principle be 100 reactions of the safest chemistry or 100 reactions of the riskiest, and our metric
would report the same number.

This came up while building the route dataset. Writing a reaction-naming table for the synthesis
schemes surfaced two reactions in the library that consume the same starting materials — a
carboxylic acid plus an amine — but make different products at different yields: an amide at 0.75,
and a direct reduction to an amine at 0.70. Because they apply to *identical* starting materials and
the library offers exactly four variants of each, the model faces a clean 50/50 choice, which makes
its actual behaviour measurable.

We measured three things. First, whether the model systematically picks lower-yielding reactions than
the ones available to it at each step. Second — and this is the part that decides whether the finding
is a flaw or a feature — *why* it does that, by re-running the higher-yielding alternative on exactly
the same starting materials and scoring both products with the same reward the model was trained on.
Third, whether hub-batching inherits, amplifies, or dampens the effect compared with best-candidate.

## Answer

The model does consistently choose below-average-yield reactions — in 77–87% of all steps — but it is
not being careless: it is following the reward, and on DRD2 it is chemically right to. The molecule
the low-yield reaction makes scores far better than the high-yield alternative (0.56 vs 0.35, with
the amine preferred in 94% of matched pairs), because DRD2 ligands need a basic amine that an amide
cannot provide. On sEH, where the reward mildly prefers the amide instead, the model uses the
low-yield reaction *less* than chance. So the yield loss is the price of reaching the pharmacophore,
and our reaction count simply cannot see that price being paid — compounded over a route it comes to
8-11% of expected yield on sEH and 16-17% on DRD2.

The more consequential finding is about hub-batching, and it is not about yield averages at all.
Hub-batching's advantage comes from many molecules sharing one synthetic prefix — which means their
failures are no longer independent. One bad step in a shared prefix loses the entire batch. The
expected number of molecules is unchanged by this (the maths cancels), but the **spread** is not:
hub-batching's standard deviation rises 1.8× to 4.8× once shared prefixes are accounted for, with
94–100% of its library sitting behind a handful of prefixes, the largest single batch holding up to
63 of 96 molecules, and the worst prefix failing with probability 0.30–0.44. Best-candidate, whose
routes are essentially independent, shows no such concentration. Hub-batching still wins decisively
on expected molecules delivered (2.7–5.3×), so this is a risk caveat on a real win, not a reversal.

## Relevance to our Publication

This is the limitation section reviewers will otherwise write for us. Any chemistry-aware reviewer at
NeurIPS — or especially at a venue like *Journal of Chemical Information and Modeling* — will ask why
a paper about synthesis cost counts reactions and ignores yield, given that our own library ships
yield data. Answering it ourselves with a measurement, and showing the direction of the bias
(reward-following, not careless) plus its size (8-17% of expected yield), is far stronger than
conceding it. The risk-concentration result is better still: it is a genuinely new, quantified
statement about what parallel-chemistry batching *costs* you, on the same axis SPARROW
(`[fromer2024sparrow]`) built its method around, and it points at a concrete fix rather than just an
apology.

## Next Experiments

**Refining for publication**

- Re-price the headline on expected molecules delivered rather than modes selected, as a secondary
  readout, so the yield-adjusted comparison is visible in one table.
- Extend the risk measurement across the diversity-cutoff frontier, not just the single canonical
  cutoff, to show the concentration is not an artifact of one operating point.
- Two seeds per target is thin for the amplification question specifically, which disagreed between
  DRD2 seeds (see Results) — the seed-42 cells cannot help here (entry [070]: no routes), so this
  needs the re-trains in `docs/RETRAIN_RUNBOOK.md`.

**Next steps in project**

- Make hub selection yield-aware: rank hubs by expected surviving molecules rather than raw fanout,
  which is a one-line change to the ranking function and directly targets the risk concentration.
- Spread a fixed budget over more, smaller batches and measure the trade against total molecules —
  the natural knob for decorrelating failure, and one the pre-select-K work (entry [037]) already
  built the machinery for.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/lsd_hubs/campaign/yield_preference.py` — joins SCENT's per-template yields to the
  anchored template ids by exact SMARTS, validates the matched acid+amine pair, runs the
  state-conditional revealed-preference test, and runs the reward counterfactual.
- `./experiments/lsd_hubs/campaign/yield_by_strategy.py` — re-runs both selection strategies at the
  fixed 100-reaction budget, prices every accepted molecule's full route, and computes the
  batch-level risk concentration.
- `./experiments/lsd_hubs/campaign/reaction_names.py` — template id → named reaction. Written for the
  route dataset's schemes; this is what surfaced the two competing acid+amine families in the first
  place.

**Datasets**

- `./external/scent/data/small/templates.txt` — the 112 reaction templates the runs actually load;
  expanded at load time to 206 **anchored** templates (one per reactant slot) by
  `rgfn/gfns/reaction_gfn/api/reaction_data_factory.py`. The anchored index is the id that appears in
  `routes.json`, which is why neither source file's row numbers can be used as ids.
- `./external/scent/data/small/templates_yields.csv` — 122 rows carrying `Family`, `Comments` and
  `yield` for those templates. Not row-aligned with `templates.txt`; joined by exact SMARTS only.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/{seed43,seed44,drd2_seed43,drd2_seed44}/`
  — the frozen enumeration behind the four route-complete cells (entry [070]). `routes.json` gives
  each molecule's synthetic prefix; `enum_children.json` `children[].reaction` gives the final step,
  so a hub child's full route is the concatenation of the two.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed4{3,4}/scent_{seh,drd2}/sample/records.csv` — the
  sampled candidate pool that best-candidate selects from. Verified against the snapshot before use:
  100% of `child_key` values are present in both `routes.json` and `compositions.json`.
- `./external/scent/oracle/drd2_current.pkl` — the frozen TDC DRD2 activity oracle, the same reward
  the DRD2 cells were trained against, used here to score counterfactual products.

**Results**

- `./experiments/lsd_hubs/campaign/results/yield_preference/revealed_preference.csv` — per cell:
  chosen vs available yield, and the compounded route-level gap.
- `.../yield_preference/usage_by_cell.csv` — per cell template usage, mean step and route yields, and
  the acid+amine split.
- `.../yield_preference/counterfactual_summary.csv` + `counterfactual_<cell>.csv` — matched
  amine/amide pairs with both scores.
- `.../yield_preference/strategy_yield_<tag>.csv` + `strategy_yield_detail_<tag>.csv` — per-arm yield
  and risk summary, plus every selected molecule with its route yield and named reaction sequence.
- `./experiments/lsd_hubs/campaign/results/template_map.csv` — all 168 observed templates with step
  counts and assigned names.

## Relevant Versions

Branch `Hub-Analysis`, on top of `4ded0f2` ("Audit every cell for a usable recipe: 5 of 36 hold one,
and most of the rest cannot be recovered").

Not committed: `experiments/lsd_hubs/campaign/{yield_preference.py,yield_by_strategy.py,reaction_names.py}`
and `experiments/lsd_hubs/campaign/results/yield_preference/`.
`[TODO — add commit hash after pushing]`

## Relevant Resources

**Sources**

- `[fromer2024sparrow]` — the parallel-chemistry / shared-intermediate framing this risk result
  speaks to.
- `[koziarski2024rgfn]` — reaction-template action space; the anchored-template expansion.

**Packages**

- RDKit — template application and substructure matching (`yield_preference.py`,
  `yield_by_strategy.py`).
- scikit-learn (via the pickled TDC oracle) — DRD2 scoring in `yield_preference.py`.
- `rgfn.gfns.reaction_gfn.proxies.seh_proxy.SEHProxyWrapper` — sEH scoring, CPU.

## Method

All steps CPU-only on a Balam login node, under `source ~/bin/rgfn-smoke-env.sh`.

1. **Build the yield map and validate the matched pair.**
   `python experiments/lsd_hubs/campaign/yield_preference.py --check`
   Joins all 112 templates to a yield (112/112 by exact SMARTS), expands to 206 anchored ids, asserts
   the two acid+amine families are a 1:1 action pair with identical reactant applicability, and
   asserts the operative yields are amide 0.75 / direct 0.70.

2. **Revealed-preference test.** Same invocation. For 6,000 sampled steps per cell, collect every
   template whose reactant pattern matches the molecule actually in hand, and compare the chosen
   template's yield against the mean of that applicable set. Applicability ignores partner-fragment
   availability, which over-states the applicable set and therefore *under*-states any preference.

3. **Reward counterfactual.**
   `python experiments/lsd_hubs/campaign/yield_preference.py --counterfactual --n-pairs 4000`
   For unique (molecule-in-hand, added-reactant, template) triples that used the 0.70 reduction,
   re-run the 0.75 amide template of matching amine substitution on the same two reactants, and score
   both products with that cell's own frozen reward. Triples whose recorded product could not be
   reproduced are excluded rather than guessed; 4000/4000 reproduced in every cell.

4. **Strategy comparison.** Per cell, with an analysis dir assembled from the snapshot
   (`compositions.json`, `enum_children.json`, `routes.json`) plus the matching `records.csv`:
   ```
   python experiments/lsd_hubs/campaign/yield_by_strategy.py \
       --adir <adir> --target {seh|drd2} --gate {7.0|0.5} \
       --snapshot .../fragments_4000.json --tag <tag>
   ```
   Runs `BestCandidateStrategy` and `HubBatchingStrategy` (child policy `free_frag`, similarity 0.5)
   to a `("reactions", 100)` budget via the shared `build_strategy` used by `run_campaign.py`, then
   prices each accepted molecule's full route.

## Results

### Generator: chosen yield vs what was available at the same state (6,000 steps/cell)

| cell | chosen | available | delta | chose below | route yield | route yield at available | forgone |
|---|---|---|---|---|---|---|---|
| scent_seh s43 | 0.7452 | 0.7715 | −0.0263 | 76.9% | 0.499 | 0.542 | 7.9% |
| scent_seh s44 | 0.7404 | 0.7742 | −0.0338 | 81.8% | 0.472 | 0.527 | 10.6% |
| scent_drd2 s43 | 0.7310 | 0.7772 | −0.0462 | 85.8% | 0.400 | 0.479 | 16.4% |
| scent_drd2 s44 | 0.7296 | 0.7780 | −0.0484 | 86.9% | 0.404 | 0.486 | 16.9% |

Library baseline mean yield = 0.7686 over 112 templates; distribution 0.65×4, 0.67×4, 0.70×13,
0.75×58, 0.80×13, 0.85×11, 0.90×2, 0.95×7. "Route yield" compounds the per-step figure over the
cell's mean route length.

### The matched acid+amine choice (all routes, pooled 747,472 steps)

| cell | steps | mean step yield | mean route yield | direct 0.70 | amide 0.75 | direct share of acid+amine |
|---|---|---|---|---|---|---|
| scent_seh s43 | 127,260 | 0.746 | 0.517 | 11.3% | 24.3% | 31.7% |
| scent_seh s44 | 154,826 | 0.741 | 0.492 | 15.8% | 21.6% | 42.3% |
| scent_drd2 s43 | 238,428 | 0.732 | 0.419 | 46.4% | 0.1% | 99.8% |
| scent_drd2 s44 | 226,958 | 0.730 | 0.422 | 51.8% | 0.1% | 99.9% |

Uniform baseline is 50.0% (four anchored ids each, identical reactant applicability).

### Reward counterfactual — why the low-yield reaction gets chosen (n=4,000 matched pairs/cell)

| cell | target | recorded amine (0.70) | counterfactual amide (0.75) | amine higher |
|---|---|---|---|---|
| scent_seh s43 | sEH proxy | 6.345 | 6.420 | 38.0% |
| scent_seh s44 | sEH proxy | 6.145 | 6.237 | 35.5% |
| scent_drd2 s43 | DRD2 activity | 0.5585 | 0.3478 | 93.8% |
| scent_drd2 s44 | DRD2 activity | 0.5766 | 0.3319 | 94.4% |

The sign of the reward preference matches the sign of the usage preference in every cell: DRD2 wants
the amine and takes the 0.70 route ~100% of the time; sEH mildly prefers the amide and takes the 0.70
route below chance.

### Selection: hub-batching vs best-candidate at 100 reactions

| cell | arm | modes | rxns | route yield | step yield | route len | % steps direct | E[molecules] |
|---|---|---|---|---|---|---|---|---|
| seh s43 | best-candidate | 28 | 101 | 0.597 | 0.744 | 1.82 | 7.8% | 16.7 |
| seh s43 | hub-batching | 80 | 101 | 0.557 | 0.745 | 1.99 | 6.3% | 44.6 |
| seh s44 | best-candidate | 28 | 103 | 0.627 | 0.765 | 1.89 | 13.2% | 17.6 |
| seh s44 | hub-batching | 83 | 100 | 0.570 | 0.770 | 2.13 | 10.2% | 47.3 |
| drd2 s43 | best-candidate | 27 | 101 | 0.359 | 0.732 | 3.37 | 24.2% | 9.7 |
| drd2 s43 | hub-batching | 96 | 100 | 0.535 | 0.732 | 2.00 | 55.7% | 51.4 |
| drd2 s44 | best-candidate | 28 | 100 | 0.356 | 0.730 | 3.36 | 31.9% | 10.0 |
| drd2 s44 | hub-batching | 94 | 100 | 0.544 | 0.749 | 2.16 | 33.0% | 51.1 |

Hub-batching's route yield is **higher** on DRD2 (0.535/0.544 vs 0.359/0.356) purely because its
routes are shorter (2.0–2.2 vs ~3.4 steps), and slightly lower on sEH. Expected molecules delivered
favours hub-batching by 2.7× (seh) to 5.3× (drd2).

**The amplification claim does not replicate across seeds.** On drd2 s43 hub-batching more than
doubles the low-yield share (24.2% → 55.7%), consistent with hub ranking favouring acid hubs with
large amine fanout; on drd2 s44 the same comparison is flat (31.9% → 33.0%), and on both sEH cells
hub-batching is slightly *lower* than best-candidate. Two seeds cannot separate a real
target-specific effect from seed noise here, so this is recorded as unresolved, not as a finding.

### Risk concentration (same runs)

| cell | arm | sd if independent | sd with shared prefixes | shared batches | largest batch | % library in batches | E[lost to prefix failure] | worst prefix fails |
|---|---|---|---|---|---|---|---|---|
| seh s43 | best-candidate | 2.5 | 2.6 | 2 | 2 | 14% | 0.8 | 0.30 |
| seh s43 | hub-batching | 4.4 | 7.8 | 16 | 9 | 94% | 15.8 | 0.30 |
| seh s44 | best-candidate | 2.4 | 2.4 | 1 | 2 | 7% | 0.4 | 0.25 |
| seh s44 | hub-batching | 4.5 | 8.8 | 12 | 12 | 95% | 13.7 | 0.30 |
| drd2 s43 | best-candidate | 2.4 | 2.5 | 0 | 0 | 0% | 0.0 | 0.00 |
| drd2 s43 | hub-batching | 4.9 | 23.6 | 4 | 63 | 100% | 21.4 | 0.30 |
| drd2 s44 | best-candidate | 2.5 | 2.6 | 0 | 0 | 0% | 0.0 | 0.00 |
| drd2 s44 | hub-batching | 4.7 | 14.5 | 7 | 26 | 100% | 19.0 | 0.44 |

Expected molecules is **identical** under both failure models — the shared-prefix survival
probability factors out of the sum over a batch's children — so only the spread and the exposed mass
distinguish them. Both arms are scored with the same batch model (a molecule with no shared hub is a
batch of one), so the contrast is structural, not an accounting difference.

### One trap worth recording

`templates_yields.csv` lists the amide reaction **twice**: at yield 0.95 with an isotope-labelled acid
carbon (`[12C:3]`), and at 0.75 with a plain `[C:3]`. `templates.txt` — the file the runs load —
contains **no** isotope labels, so the operative amide yield is 0.75 and the 0.95 row is dead. Taking
the 0.95 turns a 0.05 yield gap into a fictitious 0.25 one and would have made this entry's headline
roughly five times larger than the truth. `yield_preference.py --check` now asserts the joined values.
Relatedly, the file's `Family` column is block-style (blank = same as above), and the plain duplicates
are appended at the end with blank family cells, so forward-filling attributes them to an unrelated
family — family labels from that file are display-only and must not be trusted.
