# SCENT — synthesis-recipe re-runs (all targets) + nested-amortization cost model
**Date:** 2026-07-10, ~5pm

**[OUTDATED]** (headline cost result only): this entry costed a *fixed* set of molecules (one
representative per mode from a hub selection) two ways — hub-amortized vs independent — which makes
the shared-intermediate cost cancel out (an artifact of comparing the *same* molecules). The right
comparison is between two *selection strategies run from scratch under a budget*; see **entry 029**.
Still current and carried forward: the recipe re-runs (jobs 70180/70184), the recipe logging, and
the nested `FragmentCostTable` — 029 builds directly on them; only the fixed-set reactions/mode
comparison (the Answer/Results tables here) is superseded.

## Question

When we cost a batchable library of SCENT molecules, how do we charge the synthesis of SCENT's
reusable "shortcut" intermediates **fairly** — building each one only once, even when it is used
many times and even when one shortcut is itself built from another — and can we log *how* to make
each intermediate so a chemist could actually follow it?

## Context & Summary

SCENT is our cost-aware generator: during training it promotes high-reward intermediate molecules
into its own building-block library (a "dynamic library") so it can attach them in a single step
later. Entry `027` wired SCENT into our LSD-Flow hub analysis and — importantly — showed that to
analyze the *real* trained SCENT we must "unfreeze" those promoted fragments (a plain checkpoint
load leaves the model stuck with only the 418 purchasable base blocks). But a promoted intermediate
is **not free** the way a purchasable block is: it took real reactions to make. Our hub-batching
cost metric ("reactions per distinct compound") has to reflect that — and, crucially, charge each
promoted intermediate's build cost **exactly once** across a library, because in a real lab you'd
synthesize a batch of that intermediate once and draw from it, whether or not you use the hub trick.
A promoted intermediate can even be built from *another* promoted intermediate, so the accounting
has to unwind that nesting. The only place to learn how each intermediate is actually made is
*during* SCENT's training (once promoted, the model treats it as an atomic block). So this
experiment (a) re-runs SCENT with synthesis-route logging on every target that actually promotes
fragments, and (b) builds the cost model that consumes those routes — charging each distinct
intermediate once, expanding nesting, in both reactions and SCENT's own dollar-cost. The same routes
double as chemist-facing "how to make this intermediate" instructions.

## Answer

Charging SCENT's promoted intermediates honestly — each distinct one built exactly once, nested
through its logged route — **roughly doubles** the true reactions-per-mode the old metric reported,
because SCENT's "cheap" one-step fragment attachments hide 1–2 reactions of intermediate synthesis
each. On sEH, a flow-selected 68-mode library draws on 77 distinct promoted fragments (including
nested ones — 438 of the 1,600 promoted sEH fragments are themselves built from *other* promoted
fragments), which cost 98 reactions to build once: base hub cost 1.15 → **2.59** reactions/mode
(independent 1.77 → 3.21). Crucially the hub-vs-independent **saving is unchanged** — the shared
intermediate build is identical in both plans and cancels — so hub batching still wins; it was the
*absolute* cost that the old metric undercounted. DRD2 shows the same pattern, smaller (25 promoted
fragments, almost no nesting: 2.25 vs 3.08). The 6TD3/ClpP docking runs (400 iterations) never reach
SCENT's first promotion (iteration 1,000), so they have no promoted fragments to cost — analyzed on
the base library, and only worth a (much more expensive) longer docking re-run if we want promotions.

## Relevance to our Publication

The reactions-per-mode / amortization number is the headline "why batch from a hub" result, and the
first reviewer question for a *cost-aware* baseline like SCENT is "did you cost its reused
intermediates honestly, or double-count them?" Charging each distinct intermediate exactly once —
with nesting — is what makes the comparison defensible, and reporting it in both reactions
(RGFN-comparable) and SCENT's own dollar-cost tables (reviewer-proof) covers both framings. The
synthesis routes also move the work toward something a chemist could act on, not just a benchmark
number.

## Next Experiments

**Refining for publication**
- **Full recipe re-runs finish** (sEH job 70180, DRD2 job 70184; ~12–14 h each) → re-run the frozen
  SCENT hub analysis + nested cost model on their final checkpoints for the *accurate* numbers.
- **6TD3 / ClpP promotion question:** decide whether to run longer docking re-runs so they promote
  fragments (expensive), or report them on the base library (no promoted intermediates to amortize).
- **RGFN cross-check:** RGFN has no promoted fragments (all 418 base blocks are free), so its cost is
  unchanged; confirm the cost model reduces to the entry-025/026 behavior for RGFN.

**Next steps in project**
- Fold the nested cost into the RGFN-vs-SCENT hub-coincidence study (§8): a hub that SCENT already
  promoted is a build-once block in both plans (no extra amortization); the new savings come from
  hubs SCENT missed.
- Wire the captured routes toward a chemist-facing "synthesize these intermediates" view (logging
  only for now; no UI).

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Recipe logging + re-runs (ours):**
- `./validation/generators/scent/recipe_logging.py` — monkeypatches `DynamicLibrary` to log each
  promoted fragment's min-reaction synthesis route into `fragments_<N>.json` (entry `027`; SCENT
  clone untouched).
- `./validation/generators/scent/run_scent_fixed.py` — `--log-recipes` flag enables it.
- `./experiments/fixed_reward/scent_seh/submit_fixed_scent_seh_recipes.sh` — sEH recipe re-run (job **70180**).
- `./experiments/fixed_reward/scent_drd2/submit_fixed_scent_drd2_recipes.sh` — DRD2 recipe re-run (job **70184**).

**Nested-amortization cost model (ours, this entry):**
- `./validation/lsdflow/adapters/workers/scent_worker.py` — extended to emit per-molecule
  dynamic-fragment **composition** (`compositions.json`; which promoted fragments each hub/child
  uses), from the sampled trajectories.
- `./validation/lsdflow/adapters/scent_adapter.py`, `.../adapters/base.py` (`FlowSample.compositions`),
  `.../dag/graph.py` (`HubDAG.save` persists `compositions.json`) — carry it through the harness.
- `./validation/lsdflow/metrics/cost/dynamic_amortization.py` — `FragmentCostTable`: recursive
  per-fragment unit cost from the routes (reactions = route step count; $ = marginal, from SCENT's
  cost tables), memoized, closed under nesting. Falls back to `min_num_reactions` when no routes.
  (Its earlier same-molecules `amortized_library_cost` was removed in the addendum reframe.)
- **Campaign (the addendum's hub-batching-vs-best-candidate comparison, supersedes the initial
  `experiments/lsd_hubs/amortized_cost/`):** `glue/samplers/lsdflow/{campaign.py,mode_select.py}`
  (AL-ready strategies) + `experiments/lsd_hubs/campaign/{pick_hubs.py,run_campaign.py,submit_scent_seh_enum.sh,README.md}`.

**Recipe re-run checkpoints (inputs, on $SCRATCH):**
- `.../fixed_reward/scent_seh/2026-07-10_17-28-06/` (job 70180) + `.../scent_drd2/2026-07-10_17-28-06/`
  (job 70184) — `train/checkpoints/{last_gfn.pt,guidance_models.pt}` + `additional_fragments/fragments_4000.json`
  (1,600 fragments + routes each). Supersede the recipe-less `2026-07-07_*` anchors.

**Analysis outputs:**
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/` + `scent_drd2_70190/` — full 30k DAG
  (`records.csv`, `compositions.json`, `enumerated_records.csv`, `graph.gpickle`) + report. Small
  artifacts committed to `./validation/lsdflow/results/scent_{seh,drd2}_pilot/`
  (`meta.json`, `report.json`, `acquisitions.csv`, `enumeration.json`; large `records`/`hub_summary`/
  `graph` left on scratch).

**Job Logs:** re-runs **70180**/**70184**, dependent analyses **70189**/**70190** (all COMPLETED;
`/scratch/markymoo/rgfn_runs/{fr_scent_*_recipes,lsdflow_scent_*}-*.{out,err}`).

## Relevant Versions

Branch `Hub-Analysis`. Recipe-logging files landed with entry `027`; this entry adds the cost model
+ per-target recipe submits. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources**
- `docs/LSD_FLOW_PROPOSAL.md` §11 (reactions-per-mode PRIMARY + amortization-ratio SECONDARY using
  SCENT's yield/reactant-cost tables); entries `024` (P_B recovery, which targets re-ran), `027`
  (freeze + enumeration + recipe logging), `025`/`026` (the RGFN cost baseline this generalizes).
- `[gainski2025scent]` — SCENT dynamic library + cost model.

**Packages**
- `scent` conda env (its `rgfn` fork) — the re-runs; `rgfn` env — the harness + cost model.

## Method

1. Confirmed which entry-024 targets promote fragments: sEH + DRD2 (5,000 iters → `fragments_1000..4000.json`)
   do; 6TD3 + ClpP (400 iters, first promotion at 1,000) do **not** (no `additional_fragments/`). A
   single docking re-run cannot reach the first promotion either — 24 h walltime cap ÷ ~2.2 min/iter
   ≈ 650 iters < 1,000 — so promoting on the docking targets needs a checkpoint+requeue chain, not a
   single job; deferred.
2. Launched recipe re-runs for the promoting targets: `sbatch .../submit_fixed_scent_seh_recipes.sh`
   (**70180**), `sbatch .../submit_fixed_scent_drd2_recipes.sh` (**70184**).
3. **Queued the full weekend pipeline** (empty queue over the weekend): the frozen SCENT hub analysis
   (`submit_scent_seh.sh` / `submit_scent_drd2.sh`, now auto-discovering the newest checkpoint + running
   enumeration + persisting `compositions.json`) as **`--dependency=afterok`** jobs on the re-runs —
   sEH **70180 → 70187**, DRD2 **70184 → 70188** — so the whole chain (train → sample → enumerate →
   persist) runs unattended.
4. Built the per-molecule composition capture (worker `extract_flow_records` → `compositions.json`,
   persisted via `FlowSample.compositions` + `HubDAG.save`) and the nested-amortization cost model
   (`dynamic_amortization.py`). Unit-tested the cost logic (nested closure; no-double-count $;
   `min_num_reactions` fallback; RGFN no-promoted reduction). Validated end-to-end on a partial recipe
   run + the fully-trained recipe-less anchor before the weekend jobs ran.
5. All four jobs COMPLETED over the weekend. Ran the amortized cost model on both analyses:
   ```
   python experiments/lsd_hubs/amortized_cost/cost.py \
     --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
     --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
     --log-z 74.33 --reward-threshold 7.0 --tag scent_seh_70189      # DRD2: --reward-threshold 0.5
   ```
   Caught + fixed the DRD2 hit-bar bug (see Results); copied small artifacts into the repo.

## Results

**Re-runs completed (weekend).** sEH job **70180** (9 h 28 m) + DRD2 **70184** (12 h 30 m): each wrote
`last_gfn.pt` + `guidance_models.pt` + `fragments_4000.json` with **1,600 promoted fragments and
1,600 routes**. Nesting (fragments built from other promoted fragments): **sEH 438**, DRD2 6 — sEH's
cost-aware library recursively reuses its own intermediates far more than DRD2's.

**Dependent analyses completed.** sEH **70189** (47 m): 30k trajectories, 20,519 hubs, **2,709
multichild**, logZ 74.33; flow acquisition beats the `parent_of_topk` control on base cost (1.15 vs
1.65 rxn/mode). DRD2 **70190** (1 h 19 m): 28,865 hubs, 864 multichild, logZ 61.90. Enumeration
(frozen 2,018-frag library) produces huge neighborhoods — sEH depth-0 hubs enumerate to
2,459–7,627 children / 581–857 modes at sEH ≥ 8.2.

**Nested amortized cost (flow acquisition = highest_terminating_flow × topk_reward):**

| target | modes | promoted frags (closure) | base rxn/mode (hub / indep) | +promoted build /mode | **augmented rxn/mode (hub / indep)** | saving |
|---|---|---|---|---|---|---|
| sEH (70189) | 68 | 77 | 1.15 / 1.77 | +1.44 (98 rxns / $348, once) | **2.59 / 3.21** | 42 rxns |
| DRD2 (70190) | 85 | 25 | 1.66 / 2.49 | +0.59 (50 rxns / $118, once) | **2.25 / 3.08** | 71 rxns |

Charging promoted intermediates ~doubles the base per-mode cost; the hub-vs-independent saving is
unchanged (shared build cancels). Costs the **sampled** acquisition (mode reps have per-molecule
composition); the **enumerated** neighborhoods need enumerate-mode composition to augment (follow-up).

**Cost-model unit tests (pass).** Synthetic nested (G = base+base; F = base+G; molecule uses F):
closure {F, G}, shared 2 reactions, marginal $ 7+5 (no double-count); `min_num_reactions` fallback;
RGFN (no promoted) reduces to the entry-025/026 base cost exactly.

**Bug found + fixed.** The DRD2 analysis first returned **0 modes everywhere** — `submit_scent_drd2.sh`
had inherited the sEH hit bar (`MODE_REWARD=7.0`) via the sed that created it, but DRD2's proxy is a
0–1 probability (p50 0.97). Recomputed post-hoc at the DRD2 "active" bar 0.5 (numbers above); the
submit default is corrected to 0.5.

## Addendum — hub-batching vs best-candidate budget campaign (2026-07-11)

Reframed the cost model on feedback: instead of costing one fixed molecule set two ways (which made
the promoted-fragment cost cancel — an artifact), we now compare two **selection strategies** under
a budget, which is what the eventual AL loop will do.

- **best-candidate** — top-reward modes from the sampled pool, each built independently (promoted
  fragments shared as reusable stock). Mirrors RGFN's paper "top-k".
- **hub-batching** — walk pre-ranked hubs (parents of the top-K candidates by single-candidate flow),
  build each scaffold once, diversify its enumerated children into modes.

Two budget cases off one modes-vs-reactions curve: Case 1 = modes at a reaction budget; Case 2 =
reactions at a mode budget (≈ oracle calls). Metrics: reactions/mode, reward-gen calls/mode,
scaffolds, best/median reward, distinct intermediates + hubs used.

**Design honored two hard constraints from the researcher:** (1) the strategies are *independent* —
`BestCandidateStrategy` needs only the candidate pool (never any hub/enumeration data), so a pure
best-candidate run can't accidentally depend on hubs (the earlier "same molecules for both" mistake
can't recur); (2) they're *swappable* — different constructors, **identical** `CampaignResult`
output, so the AL loop reads `[p.smiles for p in strategy.run(budget).accepted]` for either.

New AL-ready code: `glue/samplers/lsdflow/campaign.py` (the two strategies + `CampaignResult`) +
`glue/samplers/lsdflow/mode_select.py` (pluggable diverse+threshold mode acceptor). Analysis:
`experiments/lsd_hubs/campaign/` (`pick_hubs.py` → GPU `submit_scent_seh_enum.sh` → `run_campaign.py`).
Cleanup (all superseded/dangling; deleted): the same-molecules `amortized_library_cost` +
`experiments/lsd_hubs/amortized_cost/`, the never-used `cost/{base,reactions_per_mode,registry}.py`
ABC, `glue/samplers/lsdflow/{acquisition.py,molecule/}` (the old acquisition + molecule strategies),
and the harness's acquisition-combo cost (`run.py` `_analyze_combo`/`_per_mode_cost` — the harness
now only produces the flow field + descriptive enumeration; cost lives in the campaign). Worker
`--mode enumerate` now emits `enum_children.json` (each child + the promoted fragment added in its
final reaction).

**Preliminary (best-candidate, sEH, validated on real data):** 300 modes cost **1,445 reactions
(4.82/mode)** spanning **200 distinct promoted intermediates**; 18 modes fit in a 100-reaction
budget. Hub-batching (should concentrate intermediates → fewer reactions/mode) awaits the
enumeration job **70295**; `run_campaign.py` fills the comparison + curve once it lands.
