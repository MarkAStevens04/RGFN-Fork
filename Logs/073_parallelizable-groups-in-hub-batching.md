# SCENT / sEH + DRD2 — are the molecules we hand over ones a chemist could make side by side?

**Date:** 2026-08-26, ~12pm

> **Scope note: this entry is a CURIOSITY, not a publication claim.** It was run to satisfy a
> question about the shape of our delivered libraries, not to support a headline. Nothing here has
> been replicated across generators. Of the two third filters in §Results-C, the availability-gated
> one is a diagnostic that did not work; the mode-gated one (§C2) did, and was accepted on 2026-08-26
> as a candidate change to the method — see *Relevance to our Publication*, which also states what it
> still needs before it can be claimed. Read that section before quoting any number from here.

## Question

When we hand a chemist the library our method selected, do the molecules fall into groups that could
be made side by side in one batch of identical reactions — and would the obvious alternative method
have given them that too?

## Context & Summary

**Context** — Our method's whole pitch is "build a shared intermediate once, then diversify off it",
and entry `038` already showed the resulting batches are bench-plausible in *size* (the largest is 65
molecules, most are far smaller, and the spread across intermediates is moderate). But size is not
the same as workability. A batch of twelve molecules off one shared intermediate is only *one* job if
all twelve are made by the same reaction; if they are made by five different reactions it is five
jobs that happen to share a starting material. Nobody had looked at which reaction each selected
molecule was actually made by, even though the enumeration has recorded that all along. The question
also touches something the paper already asserts: the outline's construct-validity argument claims
that one shared intermediate diversified with many purchasable partners *is* parallel and
combinatorial synthesis. That is currently an argument, not a measurement.

**Summary** — For each molecule in a delivered library we read off the single reaction that made it —
the last step, which is the only work that differs between members of a batch — and grouped the
library two ways. **Class A** puts together molecules that share *both* their starting material and
their reaction: one substrate, one set of conditions, a different purchasable partner per well. That
is a parallel array in the strict sense. **Class B** is looser and groups purely by reaction,
ignoring what it was run on: a plate of amide couplings sitting on different scaffolds. We ran the
same measurement on the obvious alternative strategy — take the highest-scoring molecules one at a
time — on identical chemistry, identical hit threshold and identical diversity cutoff, so the only
thing that differs between the two is the chooser. We then tried *demanding* parallelism rather than
merely measuring it, with a third filter, in two versions. The first keeps only molecules with at least ten
parallelizable siblings available. The second is stricter and asks how many *genuinely different*
molecules each group could deliver on its own — by running the selection inside the group first — and
keeps only groups that clear ten. Because a group that is internally varied can still turn out to
duplicate things chosen elsewhere, we then measured how much of each group actually survived.

## Answer

**The strict kind of parallelism is real, large, and entirely a product of our selection; the loose
kind is real but is not ours to claim.** In every cell, our libraries contain genuine parallel arrays
— the largest is 26 or 27 molecules made from one substrate by one reaction — and roughly three
quarters of the library sits in an array of four or more. The alternative strategy, given exactly the
same chemistry, produces essentially none: its arrays are singletons, with at most two to five
molecules ever sharing a substrate-and-reaction, and never ten. Group by reaction alone and the
distinction evaporates: at equal library size both strategies have big groups, because a handful of
reactions dominate this chemistry library regardless of who is choosing. So the honest statement is
that our method creates *arrays*, and the underlying model creates the reaction concentration.

**Demanding parallelism works, but only if you demand the right thing.** Requiring ten *available*
siblings changes nothing — the intermediates our method picks already have hundreds, so the demand
never bites. Requiring that a group could deliver ten genuinely different molecules on its own does
bite, and it is close to free: it costs at most three molecules out of eighty at the same budget, and
it consolidates the library from ten to fifteen arrays down to three to eight, with essentially
everything landing in an array of four or more and roughly double the fraction in arrays of ten or
more. The collapse that was expected is real but smaller than feared — groups keep 61-91% of what they
could have delivered — and it bites unevenly: on sEH almost every gated group still ends up with ten
or more, on DRD2 only about half do. Offering each group's members together rather than interleaved
made almost no difference, which says the loss is genuine overlap between groups rather than an
artifact of the order.

One caveat on all the counts. A count of molecules overstates the width of an array wherever the
shared substrate has two equivalent reactive sites, because one reaction then yields two separable
products from a single vessel — so 78 molecules can be 66 wells.

## Relevance to our Publication

The outline's construct-validity section already claims that a shared intermediate diversified with
diverse purchasable starting materials *is* parallel and combinatorial synthesis, and that the hub
limit *is* late-stage diversification. Those are currently arguments from the shape of the method.
This entry is the measurement that would let them be stated as facts, with the strongest version
being the contrast: given identical chemistry and identical filters, one chooser produces arrays of
26 and the other produces singletons. It also pairs with entry `071`'s risk finding — the same
concentration that makes a library parallelizable is what makes its failures correlated — so the two
belong in the same paragraph rather than in separate places.

**Decision taken 2026-08-26: the mode-gated filter (§C2) is carried forward as a candidate change to
the method, not filed as a diagnostic.** The trade it offers is the kind a lab actually cares about —
give up at most three molecules of eighty and under 7% more reactions, and get a library that is three
to eight plates instead of ten to fifteen, with essentially every molecule in an array of four or
more. That is cheap for a substantial change in how the delivered library is run. What it needs before
it can be *claimed* is unchanged by the decision: a second generator, and a check that the reaction
premium is stable rather than a property of these four cells.

**The measurement itself is still not ready to be cited, and the reason is scope.** It rests on one
generator (SCENT) and two seeds per target, because those are the only cells that hold the recorded
reactions this analysis needs (entry `070`: 5 of 36). The reaction *names* are a reading aid derived
from the templates, not identifiers. And no chemist has reviewed whether the arrays are actually
runnable — several of the shared substrates are polyfunctional, which is a selectivity question this
analysis cannot answer and does not raise on its own. Treat the grouping as a promising direction for
the route dataset's `batches.csv` layer, not as a result.

## Next Experiments

**Refining for publication**

- **Get a chemist to look at the three biggest arrays.** The largest sEH array runs an amination on a
  substrate that carries both an aryl bromide and a free secondary amine. Whether that is a clean
  26-well plate or a selectivity problem is the whole question, and it is not answerable from the
  templates — which only say a bond *can* form, never that it forms preferentially.
- **Repeat on a second generator once routes exist.** Every RGFN and RxnFlow cell is missing the
  sampled routes this needs, so the "our selection, not our generator" claim currently rests on one
  model. The re-training runbook is the gate.
- **Report wells, not just molecules, wherever an array is quoted.** The molecule count is the
  library size; the well count is the bench workload, and on DRD2 they differ by 15%.

**Next steps in project**

- **Fold the grouping into the route dataset.** The schema already specifies a `batches.csv` keyed on
  the shared intermediate; adding the reaction class turns each batch into the set of plates it
  actually is, which is what makes the dataset a synthesis guide rather than a molecule list.
- **Qualify the mode-gated filter on a second generator** (decision above). Two things to establish:
  that the reaction premium stays small (0.6-7.1% here) rather than being a property of SCENT's
  dynamic library, and that the gate's threshold does not need per-target tuning — sEH and DRD2 already
  behave differently under it, which is the first hint it might.
- **Close the remaining gap by making the gate campaign-aware.** The gate counts a group's diversity
  in isolation, which is why about half of DRD2's gated groups still fall below ten. Counting instead
  against what has already been accepted would remove that failure mode by construction. That is a
  different method rather than a diagnostic, so it was deliberately left out of this entry.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/campaign/parallel_groups.py` — **new, and the whole of this entry.**
  Re-runs both selection arms over a frozen enumeration, joins each accepted molecule to the reaction
  that made it, and emits the Class A / Class B groupings plus the optional third-filter arm. Three
  things in it are load-bearing rather than incidental: (1) it re-derives the hub list itself in order
  to attach reaction classes, then asserts field-by-field parity against
  `run_campaign._load_enumerated_hubs` — without that check a loader drift would silently describe a
  library we never selected; (2) it counts **wells** = distinct `(parent, reagent set)` pairs beside
  molecules, because a polyfunctional substrate turns one reaction into two products; (3) it marks
  ambiguous joins rather than resolving them (see Method step 3).
- `./experiments/lsd_hubs/campaign/reaction_names.py` — template → named reaction. Pre-existing
  (entry `071`), used here unchanged as the grouping key. 168 templates collapse to 42 named classes.
- `./experiments/lsd_hubs/campaign/run_campaign.py` — provides `_load_candidates`, `build_strategy`,
  `_load_enumerated_hubs` (the parity reference) and `run_timed`. Unchanged.
- `./experiments/lsd_hubs/campaign/batch_size_distribution.py` — entry `038`; its `_gini` / `_stats_for`
  are imported so the imbalance statistics here are computed identically to the batch-size ones.
- `./glue/samplers/lsdflow/campaign.py` — `HubBatchingStrategy` / `BestCandidateStrategy`, the objects
  under test. **Not modified.** The third filter is a local `ChildSelectionPolicy` subclass in the
  experiment script, and the reaction-carrying child is a local `EnumChild` subclass, so `glue/` stays
  free of the naming table and the baseline selection stays byte-identical.
- `./experiments/lsd_hubs/matrix16/check_route_readiness.py` — entry `070`; used to pick the cells.

**Datasets** (inputs; no new compute)

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/{seed43,seed44,drd2_seed43,drd2_seed44}/enum_children.json`
  — the frozen 200-hub enumerations. Each child carries `reaction` = the final step's template,
  reactants, input and product; that field is the entire input this analysis needs and it is 100%
  populated on all four cells. Read from the frozen copy, not the live path, per
  `docs/ROUTE_DATASET_SCHEMA.md` §7. All four were md5-verified equal to their live counterparts
  under `matrix16_seed4{3,4}/`, so this is provenance safety rather than a divergence.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed4{3,4}/scent_{seh,drd2}/sample/{records.csv,compositions.json,routes.json}`
  — the candidate pool the alternative strategy chooses from, the per-molecule promoted fragments the
  cost model needs, and the sampled routes. `routes.json` is what supplies the **alternative arm's**
  final step: its molecules are sampled terminals rather than enumerated children, so their last step
  is not in the enumeration.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_{seh,drd2}_5k/seed4{3,4}/additional_fragments/fragments_4000.json`
  — promoted-fragment recipes, for the nested count-once cost model and the pre-select-20 ranking.

**Results** (committed, small — 6.5 MB total)

`./experiments/lsd_hubs/campaign/results/parallel_groups/`

- `<cell>/modes_<arm>.csv` — **one row per selected molecule**: smiles, reward, cumulative reactions,
  marginal reactions, parent (stereo-stripped) and `parent_input` (stereo-aware), named reaction,
  template id, reagents. One per arm: `modes_hub_batching.csv`, `modes_best_candidate.csv`, and in the
  `_m10` cells `modes_hub_batching_modes10_{reward,group}.csv`. This file did not previously exist
  anywhere: the committed campaign curves record `source_hub` but **not** the SMILES, so the delivered
  library could not be joined to its chemistry without re-running the selection.
- `<cell>/groups.csv` — one row per group (both classes, both slices, all arms) with molecules,
  wells, multi-product wells, distinct parents/templates, reward range, and selection-step span.
- `<cell>/group_stats.csv` — one row per (arm, slice, class): group-count and size distribution,
  Gini, and coverage at group sizes ≥2/4/8/10/24/96 in both molecules and wells.
- `<cell>/class_b_decomposition.csv` — for each Class B group, how many Class A arrays it splits into.
  This is the single most legible number in the entry.
- `<cell>/parallel_groups.png` — rank-size curves, Class A beside Class B, at the reaction budget.
- `<cell>_m10/collapse.csv` — **the collapse measurement**: per used group, local capacity vs
  delivered modes, with `hub_truncated` marking the one group whose hub the budget cut short so it is
  excluded rather than averaged in.
- `filter_sweep.csv` — the availability-gate threshold sweep, with a `status` column flagging the one
  pool-exhausted row so its reaction total is never read as a cost win.
- `mode_gate.csv` — the mode-gate summary: baseline vs both offering orders, per cell, with the
  gated-in/out counts and the retention figures. **This is the table to read first.**
- `<cell>/summary.json` — inputs (with the enumeration path), the configuration, per-arm totals, the
  join diagnostics (unjoined / ambiguous / `bc_parent_equals_assigned_hub`), the `mode_gated_arms`
  block, and every row of `group_stats.csv`. The provenance record for the cell.
- Cells: `scent_seh_seed43`, `scent_seh_seed44`, `scent_drd2_seed43`, `scent_drd2_seed44`, plus
  `_par{10,50,200,400,800}` (availability gate) and `_m10` (mode gate, both orders) variants.

**Job Logs** — none. Everything ran interactively on `balam-login01`; **~33 s per cell** for all four
arms including both micro-selection passes, pure CPU. The within-group micro-selections are cheap
because they only touch the children of hubs the campaign actually walks (~15-60k of the 318-792k
enumerated), and the ECFP cache is shared with the campaign's own selector, so no fingerprint is
computed twice.

### Relevant Versions

Branch `Hub-Analysis`. Most recent commit at time of running: `c5f428e` ("The ClpP pre-flight gated
on the training reward, so every docking cell read as empty").

The `glue/` strategy code under test is committed and **unmodified by this entry**. New and
uncommitted:

- `experiments/lsd_hubs/campaign/parallel_groups.py`
- `experiments/lsd_hubs/campaign/results/parallel_groups/`
- `Logs/073_parallelizable-groups-in-hub-batching.md`

Also still uncommitted from entry `071`, and depended on here: `reaction_names.py`.

[TODO — add commit hash after pushing.]

### Relevant Resources

**Sources**

- Entry `038` (per-hub batch sizes — the distribution this refines), entry `037` (the pre-select-K
  policy these libraries were selected with), entry `070` (the route-artifact audit that determined
  which cells are usable), entry `071` (reaction-level behaviour of the same cells, and the
  risk-concentration finding this pairs with), entry `065` (the fixed-reaction-budget readout).
- `docs/ROUTE_DATASET_SCHEMA.md` §4.2 — the `batches.csv` batch layer this measurement would populate;
  §7 — the frozen-snapshot rule followed here.
- `docs/paper_planning/current-outline.md` §2.4 — the construct-validity claim being measured.
- `[fromer2024sparrow]` — *"Optimal Compound Downselection to Promote Diversity and Parallel
  Chemistry"*: the nearest precedent for treating shared intermediates as the unit of experimental
  work, and the reason "parallel chemistry" is the right frame for this question.
- `[gainski2025scent]` (the dynamic library these cells were trained with),
  `[bengio2021gflownet]` (the mode definition).

**Packages** — `rgfn` conda env via `~/bin/rgfn-smoke-env.sh`; RDKit (fingerprints inside
`mode_select`), matplotlib (figures). No GPU, no docking, no scoring.

### Method

1. **Chose the cells by audit, not assumption.** Ran `check_route_readiness.py` across all three
   scratch trees: 5 of 36 cell-seeds hold complete recipes, all SCENT — `seh`/`drd2`/`6td3` at seed 43
   and `seh`/`drd2` at seed 44. Took the four surrogate cells. **Excluded `scent_6td3` seed 43**
   although it is route-complete: its selection was made against the 6TD3 differential that entry
   `072` showed our own candidates exploit, so grouping molecules chosen by a superseded gate would
   describe a library we would not now deliver.
2. **Confirmed the input exists before building on it.** On `seed43`: 200 hubs, 702,574 children,
   **100.0000%** carrying a reaction, **exactly one step each** (so "the final step" is unambiguous),
   and the step's `input` equal to `hub_input` for **all** 702,574 children. Also audited the naming
   on this corpus: 22 of 702,574 children unnamed (0.003%), and the dominant class
   (`Urea formation (amine + amine)`, 41%) checked against its four templates (ids 125-129) to confirm
   the product genuinely inserts a carbonyl between two nitrogens rather than being a loose match.
3. **Ran both arms per cell** at the cell's committed configuration — the same one
   `run_cell_campaign.sh` uses, so the libraries analysed are the published ones: SCENT gets
   `free_frag` + pre-select-K=20, similarity 0.5, gate 5.0 (sEH) / 0.5 (DRD2) from `targets.py`,
   to a 300-mode budget:

   ```
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/campaign/parallel_groups.py \
     --analysis-dir /scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43/scent_seh/sample \
     --enum-children /scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/seed43/enum_children.json \
     --snapshot /scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed43/additional_fragments/fragments_4000.json \
     --reward-threshold 5.0 --higher-is-better true --similarity 0.5 \
     --child-policy free_frag --prebuild-k 20 --budget-reactions 100 --budget-modes 300 \
     --tag scent_seh_seed43
   ```

   Loader parity against `run_campaign._load_enumerated_hubs` passed on all four cells, and the arm
   totals reproduce the committed campaign (e.g. `seh_seed43`: 300 modes / 326 reactions / 6 hubs).
4. **Joined each molecule to its final step.** Hub-batching joins on `(source_hub, smiles)` into the
   enumeration — **0 of 300 unjoined** in every cell. `source_hub` is the *stereo-stripped* hub key,
   so stereoisomeric hubs collide on it; 5/7/0/62 keys per cell are genuinely ambiguous (the colliding
   children were made by different reactions). Those are **marked, not resolved** — keeping whichever
   came first would put a wrong reaction name on a molecule with nothing to show it. **0 of 300
   selected modes hit an ambiguous key in any cell**, so the join is exact in practice.
   The alternative arm takes its final step from `routes.json` (0 of 300 missing), and defines the
   parent as that step's own `input` — the true immediate precursor — **not** the hub the cost model
   assigned it to. Those differ for 62-100 of 300 molecules per cell, which is why the distinction is
   made explicitly: the assigned hub is the cheapest valid shared prefix for *accounting*, while
   parallel chemistry is about what is weighed into the well.
5. **Grouped and sliced.** Class A on `(parent, named reaction)`, Class B on `named reaction` alone,
   each read at the 100-reaction budget (the primary readout) and over the full 300-mode library — the
   first being a prefix of the second, so both come off one run.
6. **Third filter, version 1 — availability (`--min-group N`).** An arm whose child policy keeps only
   children in a `(this hub, reaction class)` group of ≥N **reward-passing** siblings, then delegates
   to `free_frag`. Swept N = 10, 50, 200, 400, 800 on `seh_seed43` and `drd2_seed44`; N = 10 on all
   four cells.
7. **Third filter, version 2 — deliverable modes (`--min-modes N`).** The same shape, but the gate runs
   a **micro greedy selection inside each group**: the group's children are first ordered by the inner
   policy (so the count credits only what the campaign would actually be offered), then fed to a
   **fresh** `DiverseThresholdModeSelector` with the cell's own bar and cutoff. The count is therefore
   "modes this group could deliver if it had the whole diversity budget to itself", capped at
   `--local-cap 200`. N = 10 on all four cells, in **both** offering orders (`--order-modes
   reward,group`): groups interleaved by reward, and groups offered contiguously best-first. Groups are
   ranked for the contiguous arm by their best child's reward, *not* by size, so reward priority is
   unchanged between the two arms and the only difference is contiguity.

   ```
   python experiments/lsd_hubs/campaign/parallel_groups.py \
     --analysis-dir /scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43/scent_seh/sample \
     --enum-children /scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820/seed43/enum_children.json \
     --snapshot /scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed43/additional_fragments/fragments_4000.json \
     --reward-threshold 5.0 --higher-is-better true --similarity 0.5 \
     --child-policy free_frag --prebuild-k 20 --budget-reactions 100 --budget-modes 300 \
     --min-modes 10 --tag scent_seh_seed43_m10
   ```

   **The `--local-cap 200` ceiling was never reached** — 0 of 190 / 154 / 730 / 812 groups scored, and
   0 of the groups actually used, saturated it. So every `local_modes` below is the exact within-group
   capacity, and the retention figures in §C3 are exact rather than upper bounds. Worth stating because
   a saturated cap would have made retention look better than it is, silently.
8. **Collapse measurement, with truncation separated out.** For every group present in the accepted
   set, `collapse.csv` records its local capacity beside what it delivered. **One hub is always
   partially consumed** — `HubBatchingStrategy` offers every child of a hub before moving on and
   breaks out of both loops only when the budget stops it, so exactly the hub holding the last accepted
   mode is truncated. Those groups are flagged `hub_truncated` and **excluded from the retention
   summary**; without that, the budget ending would have been counted as collapse (it inflated the
   apparent loss from 13% to 25% on `seh_seed43` before the correction).

### Results

**A. Class A — same parent, same reaction (a strict parallel array).** At the 100-reaction budget.
"wells" = distinct `(parent, reagent set)` pairs; "≥4" / "≥10" = fraction of the library in an array
of at least that many molecules.

| cell | arm | molecules | wells | parents | arrays | largest array | ≥4 | ≥10 |
|---|---|---|---|---|---|---|---|---|
| `scent_seh_seed43` | hub-batching | 79 | 74 | 2 | 14 | **26** (22 wells) | 0.77 | 0.52 |
| | best-candidate | 27 | 27 | 26 | 26 | 2 | 0.00 | 0.00 |
| `scent_seh_seed44` | hub-batching | 80 | 80 | 1 | 10 | **27** (27 wells) | 0.89 | 0.68 |
| | best-candidate | 27 | 27 | 27 | 27 | 1 | 0.00 | 0.00 |
| `scent_drd2_seed43` | hub-batching | 78 | 69 | 2 | 14 | **16** (15 wells) | 0.77 | 0.36 |
| | best-candidate | 26 | 26 | 26 | 26 | 1 | 0.00 | 0.00 |
| `scent_drd2_seed44` | hub-batching | 78 | 66 | 4 | 15 | **21** (14 wells) | 0.89 | 0.42 |
| | best-candidate | 28 | 28 | 28 | 28 | 1 | 0.00 | 0.00 |

At a fixed budget the two arms deliver different library sizes (79-80 vs 26-28 — that *is* the
headline result of entries `065`/`050`), so the same comparison is repeated at **equal library size**
(300 modes) where size cannot explain anything:

| cell | arm | reactions | parents | arrays | largest | ≥4 | ≥10 |
|---|---|---|---|---|---|---|---|
| `scent_seh_seed43` | hub-batching | 326 | 6 | 54 | 31 | 0.81 | 0.45 |
| | best-candidate | 955 | 269 | **279** | 5 | 0.017 | 0.00 |
| `scent_seh_seed44` | hub-batching | 328 | 8 | 58 | 31 | 0.77 | 0.48 |
| | best-candidate | 971 | 280 | **287** | 2 | 0.00 | 0.00 |
| `scent_drd2_seed43` | hub-batching | 336 | 13 | 83 | 17 | 0.71 | 0.24 |
| | best-candidate | 1137 | 299 | **299** | 2 | 0.00 | 0.00 |
| `scent_drd2_seed44` | hub-batching | 338 | 15 | 85 | 21 | 0.67 | 0.34 |
| | best-candidate | 1118 | 298 | **300** | 1 | 0.00 | 0.00 |

279-300 arrays for 300 molecules: the alternative arm's libraries are **pure singletons**. This is the
result, and it survives equalising library size.

**B. Class B — same reaction, any parent — is NOT a property of the selection.** At equal library
size the two arms are indistinguishable:

| cell | hub-batching (groups / largest / ≥10) | best-candidate (groups / largest / ≥10) |
|---|---|---|
| `scent_seh_seed43` | 17 / 99 / 0.91 | 16 / 73 / 0.88 |
| `scent_seh_seed44` | 18 / 65 / 0.88 | 17 / 57 / 0.87 |
| `scent_drd2_seed43` | 14 / 79 / 0.98 | 12 / 114 / 0.94 |
| `scent_drd2_seed44` | 17 / 117 / 0.92 | 12 / 110 / 0.92 |

Both concentrate on a dozen-odd reaction classes because the chemistry library and the reward do —
consistent with entry `071`, which measured the reward driving reaction choice directly. **Class B
should not be claimed as an advantage of our method.** (An earlier version of this analysis keyed
wells on the reagent alone; within a Class A group the parent is constant so that is equivalent, but
across a Class B group it collapsed the same amine reacting with four different scaffolds into one
well and made best-candidate's Class B read ~5× more concentrated than it is — 114 molecules scored
as 23 wells. Corrected before any number above was taken.)

**The decomposition makes the difference legible in one line** (`scent_seh_seed43`, 100 reactions):

| Class B group | hub-batching → Class A arrays | best-candidate → Class A arrays |
|---|---|---|
| Buchwald-Hartwig amination | 28 molecules → **2 arrays** (26 + 2) | 3 molecules → 3 arrays of 1 |
| Amide coupling | 18 molecules → **2 arrays** (15 + 3) | 8 molecules → 8 arrays of 1 |
| Urea formation (amine + amine) | 9 molecules → **1 array** of 9 | — |
| Suzuki coupling | — | 5 molecules → 5 arrays of 1 |

**C1. Gating on AVAILABLE siblings does almost nothing.** Requiring ≥N reward-passing siblings per
`(hub, reaction)`:

| cell | min-group | children dropped | modes @100 rxn | arrays | largest | ≥4 | ≥10 | full-curve |
|---|---|---|---|---|---|---|---|---|
| `scent_seh_seed43` | 0 (baseline) | — | 79 | 14 | 26 | 0.77 | 0.52 | 300 modes / 326 rxn |
| | 10 | 390 | 79 | 13 | 26 | 0.80 | 0.53 | 300 / 326 |
| | 50 | 1,275 | 79 | 13 | 26 | 0.80 | 0.53 | 300 / 327 |
| | 200 | 5,709 | 79 | **7** | 26 | 0.96 | 0.73 | 300 / 330 |
| | 800 | 24,198 | 79 | **5** | 27 | **0.98** | **0.90** | 300 / 336 |
| `scent_drd2_seed44` | 0 (baseline) | — | 78 | 15 | 21 | 0.89 | 0.42 | 300 / 338 |
| | 10 | 2,041 | 78 | 15 | 21 | 0.86 | 0.42 | 300 / 338 |
| | 200 | 48,163 | 77 | 14 | 23 | 0.86 | 0.46 | 300 / 354 |
| | 400 | 294,434 | 76 | 13 | 24 | 0.84 | 0.63 | 300 / 341 |
| | 800 | 416,453 | 75 | 12 | 12 | 0.88 | 0.45 | ⚠ **196 modes** / 233 rxn — **pool-exhausted**, not a cost win |

At N=10 nothing moves, because the walked intermediates have hundreds of qualifying children per
reaction class. Only at N=200-800 does the gate bite, and then it bites bluntly: sEH tolerates 800
(0 modes lost, 14 arrays → 5, 0.90 in arrays ≥10) while DRD2 breaks between 400 and 800, where the
filtered pool runs out at 196 modes. **The gate is measuring the wrong quantity** — availability, not
what a group can deliver — which is what §C2 fixes.

**C2. Gating on DELIVERABLE modes works, and is nearly free.** Same shape, but a group is admitted
only if a micro greedy selection *inside* it yields ≥10 mutually distinct molecules. Class A at the
100-reaction budget:

| cell | gate | modes | arrays | largest | ≥4 | ≥10 | rxn for 300 |
|---|---|---|---|---|---|---|---|
| `seh_seed43` | none | 79 | 14 | 26 | 0.77 | 0.52 | 326 |
| | ≥10 available | 79 | 13 | 26 | 0.80 | 0.53 | 326 |
| | **≥10 modes, reward order** | 79 | **4** | 28 | **1.00** | **0.94** | 331 |
| | **≥10 modes, group order** | 79 | **3** | 32 | **1.00** | **1.00** | 331 |
| `seh_seed44` | none | 80 | 10 | 27 | 0.89 | 0.68 | 328 |
| | ≥10 available | 80 | 10 | 27 | 0.89 | 0.68 | 328 |
| | **≥10 modes, reward order** | 79 | **4** | 31 | **1.00** | 0.89 | 330 |
| | **≥10 modes, group order** | 79 | **4** | 33 | **1.00** | 0.91 | 330 |
| `drd2_seed43` | none | 78 | 14 | 16 | 0.77 | 0.36 | 336 |
| | ≥10 available | 78 | 14 | 16 | 0.77 | 0.36 | 336 |
| | **≥10 modes, reward order** | 76 | **8** | 17 | 0.99 | 0.61 | 351 |
| | **≥10 modes, group order** | 76 | **8** | 17 | 0.99 | 0.68 | 351 |
| `drd2_seed44` | none | 78 | 15 | 21 | 0.89 | 0.42 | 338 |
| | ≥10 available | 78 | 15 | 21 | 0.89 | 0.42 | 338 |
| | **≥10 modes, reward order** | 75 | **5** | 24 | **1.00** | **1.00** | 362 |
| | **≥10 modes, group order** | 75 | **5** | 24 | **1.00** | **1.00** | 362 |

Cost: **0-3 molecules** out of 78-80 at the budget, and **+2 to +24 reactions** (0.6-7.1%) for the
full 300-mode library. Return: arrays consolidate from 10-15 to **3-8**, ≥4 coverage reaches
**0.99-1.00** in every cell, and ≥10 coverage roughly doubles. The gate rejects most groups it sees
(173/190, 136/154, 702/730, 779/812) but the ones it keeps are the ones that were carrying the library
anyway.

**C3. The collapse is real, is 9-39%, and is NOT caused by interleaving.** Per group, local capacity
(what the micro-selection said it could deliver alone) vs what it actually delivered. **Excluding the
one hub the budget truncated** — see Method step 8; without that correction the apparent loss on
`seh_seed43` reads 25% instead of 13%.

| cell | order | closed groups | still ≥10 after selection | delivered / local | retained |
|---|---|---|---|---|---|
| `seh_seed43` | reward | 16 | **14/16** | 294 / 339 | 87% |
| | group | 16 | 13/16 | 293 / 339 | 86% |
| `seh_seed44` | reward | 16 | **14/16** | 287 / 317 | 91% |
| | group | 16 | **14/16** | 289 / 317 | 91% |
| `drd2_seed43` | reward | 27 | 15/27 | 290 / 362 | 80% |
| | group | 27 | 17/27 | 296 / 362 | 82% |
| `drd2_seed44` | reward | 31 | 14/31 | 294 / 482 | 61% |
| | group | 31 | 14/31 | 294 / 482 | 61% |

(Full 300-mode library. At the 100-reaction budget only 1-7 groups are closed, so those retention
figures — 100%/90%/78%/93% — rest on very few groups and are reported for completeness only.)

So the effect the collapse was predicted to have is confirmed and quantified: **a group gated in at
≥10 still delivers ≥10 for 13-14 of 16 groups on sEH (81-88%) but only 14-17 of 27-31 on DRD2
(45-63%)**, and DRD2's groups give
up 39% of their local capacity. The two offering orders differ by **0-2 percentage points** in
retention — contiguity buys essentially nothing, so the loss is genuine overlap between groups and
hubs, not an artifact of interleaving them. That is a clean negative result for the ordering
hypothesis and it is why the gate's remaining failure mode is a *gate* problem (it counts diversity in
isolation) rather than an *ordering* problem.

**And the two targets differ for a legible reason: the gated groups are not far above the threshold.**
Median local capacity of a gated-in group is 21 (`seh_s43`), 13.5 (`seh_s44`), 12.0 (`drd2_s43`) and 15
(`drd2_s44`). A group with a capacity of 12 that retains 61% lands at 7 — below the bar — whereas one
with a capacity of 21 retaining 87% lands at 18 and stays comfortably above it. So the hold rate is
roughly what capacity and retention together predict, and the fix implied by that is either a higher
threshold (buying headroom) or a campaign-aware gate (removing the shortfall). Both are untested.

**D. Molecules overstate wells wherever the substrate is polyfunctional.** Concretely, on
`scent_drd2_seed43` the hub `Oc1ccc(CCNCC2CCN2)cc1` carries two secondary amines, so template 74
(ketone reductive amination) fires at either site: 9 selected molecules from **6** ketones, three of
which give two regioisomers from one vessel. Both products are real and score differently (0.995 vs
0.991), so the mode count is not wrong — but as parallel chemistry it is 6 wells, not 9.

| cell | Class A molecules @100 rxn | wells | wells giving >1 product |
|---|---|---|---|
| `scent_seh_seed43` | 79 | 74 | 3 |
| `scent_seh_seed44` | 80 | 80 | 0 |
| `scent_drd2_seed43` | 78 | 69 | 9 |
| `scent_drd2_seed44` | 78 | 66 | 12 |

**E. One incidental observation worth recording.** On `scent_seh_seed44`, all 80 molecules at the
100-reaction budget come from a **single** substrate (`O=C(O)C1Cc2ccccc2CN1`) diversified 10 ways — the
Class A and Class B group counts are identical because the parent never varies. That is the
late-stage-diversification limit the outline describes, reached without being asked for; it is also
the worst case for entry `071`'s correlated-failure finding, since one bad prefix step would lose the
entire library.
