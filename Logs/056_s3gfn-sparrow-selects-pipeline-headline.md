# sEH — the competitor pipeline run the way a chemist would actually run it

**Date:** 2026-08-04, ~4pm

## Question

If a chemist used the standard toolchain — generate a batch of molecules, plan all their syntheses,
then let an optimizer pick which ones to actually make — would our method still deliver the same
library for fewer reactions?

## Context & Summary

Every head-to-head we had run so far compared *selection strategies on our own generator's output*.
That answers "is hub-batching better than picking the best individual molecules?", but it does not
answer the question a reviewer actually cares about: is our whole pipeline better than the pipeline a
working chemist would use? Worse, on the one axis where we had compared against an outside generator
(entry [048]), **we lost**: at a budget of 100 reactions the non-reaction baseline S3-GFN delivered
34–35 distinct molecule families and our method delivered 20–27.

Entry [047] had already identified why. To price our molecules, we were re-deriving their syntheses
from scratch with a retrosynthesis planner — and that planner could only solve 48.7% of our
molecules, because our building blocks are not in its catalogue, while S3-GFN builds directly from
that catalogue and solves ~90%. So the comparison was mostly measuring whose chemistry the planner
happened to recognize, not whose library was cheaper to make. A real chemist would never re-derive
our routes; our generator hands them the route it used.

This entry rebuilds the comparison the way the workflow actually runs. The competitor gets the full
modern toolchain and every advantage we can reasonably give it: a strong generative model, a
state-of-the-art batch retrosynthesis planner that is explicitly designed to find shared
intermediates, its own home catalogue, and — critically — **the optimizer is allowed to do its own
job**, choosing which molecules are worth making rather than being handed a pre-chosen list to price.
Our side is priced on the routes our generator produced by construction. Both sides must deliver the
same thing: 100 distinct, high-scoring molecule families.

## Answer

**Run as a real pipeline, our method needs about three times fewer reactions to deliver the same
100-family library — and the earlier result that had us losing was an artifact of who was made to do
the retrosynthesis.** The competitor pipeline needs ~411 reactions to reach 100 families; ours needs
131. Read the other way, at our budget of 131 reactions the competitor delivers 32 families to our
100. This holds even though the competitor's molecules score *higher* than ours on the binding
surrogate and even though its syntheses solve at a 95.6% rate, so neither molecule quality nor
planner failure explains the gap.

**The number to quote depends on how charitable we are to the competitor, and we should be explicit
about that.** The 3.1× compares each pipeline as it would actually be run, with the optimizer making
its own selections. But the optimizer has no notion of diversity, so we also gave the competitor a
diversity-aware selector on top of the same strong planner — its best possible configuration. That
reaches 100 families in **235** reactions, so the **conservative margin is 1.79×**. Both numbers are
real; 1.79× is the one that survives a reviewer arguing we judged the optimizer at a task it was not
built for, and it is the one to lead with.

Two secondary findings came out of the same runs. First, the optimizer has a measurable blind spot:
because cheap syntheses come from molecules that share intermediates, and molecules that share
intermediates look alike, its selections are **less diverse than the pool it draws from** exactly
when the budget is tight — 27% distinct families versus the pool's own 41%. Second, our
"pre-synthesize the top-K fragments up front" option has a **break-even library size**: below it the
up-front cost is wasted and the option actively hurts, above it the sharing pays that cost back.

## Relevance to our Publication

This is the experiment that turns the paper's claim from an internal comparison into an external one.
Reviewers at NeurIPS will ask the obvious question — *you compared your method against your own
ablations; what happens against what people actually use?* — and the honest answer we had before this
entry was unfavourable. We can now state the comparison in the form the question demands: same
target, same scoring function, same quality bar, same deliverable, competitor given the stronger
planner and its own catalogue, optimizer allowed to make its own decisions, and our method still
delivers the library for a third of the synthetic effort.

It also lets us report the costs honestly rather than defensively. Our method spends far more calls
to the scoring function than the competitor does, and the competitor spends 2.25 hours of
route-planning that our method never pays. Naming both, instead of engineering a single "fair" budget
that flatters one side, is a stronger position than a forced parity constraint a reviewer could
poke at.

## Next Experiments

**Refining for publication**

- **Repeat the head-to-head on the second target.** Everything here is sEH. Note this is *not* the
  cheap extension it first appears: the competitor generator has only ever been trained on sEH, so a
  second target requires training it first, not merely re-planning.
- **Give the competitor our building blocks too.** We deliberately gave it only the standard
  commercial catalogue, which is the realistic setting, but a reviewer can argue our blocks are
  purchasable as well. Re-running the planning step with the merged catalogue removes that objection
  and is expected to strengthen the competitor.
- **Error bars.** Still the single largest gap in the benchmark: every number here is one seed.
- **Choose the pre-synthesis setting per target rather than globally.** The break-even finding means
  the current global setting is the wrong choice for at least one target at the 100-family scale.

**Next steps in project**

- Fold this into the headline figure so the reader sees both pricing regimes and understands why the
  ordering changes between them.
- Extend the same head-to-head to the docking-based targets, where synthesis cost matters most
  because each measurement is expensive.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh` — standalone MultiAiZ route discovery
  over a fixed pool. Exists to break the plan-once/price-many coupling: `eval/multiaiz.py` re-runs
  discovery inside every `score()` call (the `subprocess.run` at line 97 is unconditional and the
  snapshot dir is keyed by call counter), so varying a SPARROW parameter used to re-pay the whole
  discovery. This persists `multiaiz_routes.json` and skips if it already exists.
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — the competitor pipeline's frontier.
  Consumes the cached routes artifact, builds the merged reaction network once (it does not depend on
  the budget), then sweeps SPARROW's reaction budget, counting modes in each selection.
- `./validation/lsdflow/adapters/workers/sparrow_worker.py` — **modified**: added SELECTION mode
  (`--select`, `--max-rxns`, `--max-targets`, `reward` objective) alongside the existing PRICING mode,
  and emits `selected_target_smiles` (pricing never needed it; mode counting does).
- `./experiments/lsd_hubs/campaign/reconcile_t15.py` — **modified**: exposes `--child-policy`,
  `--prebuild-k`, `--rank-by`. Before this, the reconciler never passed them, so entry [049]'s audit
  silently used the naive policy rather than the configuration the headline reports.
- `/tmp/.../scratchpad/mode_saturation.py` — pre-flight pool sizing (scratch, not committed).
- `/tmp/.../scratchpad/build_s3gfn_pools.py` — emits the nested top-N pools (scratch, not committed).

**Models / Datasets**

- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/71007/fixed_reward/candidates/candidates.csv`
  — S3-GFN's 2,000-molecule sEH candidate pool, scored by the same frozen sEH surrogate our generator
  trains against. The source of the competitor's candidates.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_N{50,100,250,500}/` — the
  nested competitor pools (`pool.smi`, `pool_scores.csv`) plus their cached
  `multiaiz_routes.json` and `discovery_timing.json`. Nested by reward rank so the cost-scaling curve
  is attributable to pool size rather than composition.
- `data/models/aizynthfinder/config.yml` — plain ZINC stock + USPTO templates. Deliberately the
  competitor's own catalogue, not the merged one (see Method step 2).
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974/` — our sEH library's by-construction
  routes and enumerated hubs; the native-route side of the comparison.
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_drd2_native_72145/` — the DRD2 equivalent, produced
  earlier the same day.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_{seh,drd2}/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`
  — the recipe-carrying promoted-fragment snapshots (1600/1600 recipes each), required for the
  cost-model audit.

**Results**

- `./experiments/lsd_hubs/campaign/results/s3gfn_seh_greedy_N500/` — the competitor's STRONGEST
  configuration (same cached MultiAiZ routes, but a diversity-aware greedy selection instead of the
  optimizer's own). Run because it was the one comparison that could only ever *weaken* our claim.
- `./experiments/lsd_hubs/campaign/results/s3gfn_seh_select_N500/` — the headline frontier
  (`select_frontier.csv`, `select_frontier_summary.json`, per-budget `milp_R*.json`).
- `./experiments/lsd_hubs/campaign/results/scent_seh_native_freefrag_reconcile/` — the cost-model
  audit on the configuration the headline actually reports.
- `./experiments/lsd_hubs/campaign/results/scent_{seh,drd2}_freefrag_k0_reconcile/` — the K=0 arms
  that isolate the up-front pre-synthesis charge.
- `./experiments/lsd_hubs/campaign/results/scent_drd2_native_freefrag_reconcile/` — the failing DRD2
  arm (kept deliberately: it is the finding, not a discarded run).
- `./experiments/lsd_hubs/campaign/results/t15_regress_naive_reconcile/` — regression proving the
  reconciler's defaults still reproduce entry [049] bit-for-bit.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/maiz_disc_N{50,100,250,500}-{72187,72188,72189,72190}.{out,err}`

### Relevant Versions

Branch `Hub-Analysis`. Most recent commit at time of writing: `ab16c1c` ("Docking launcher: fix two
smoke-found bugs, force the optimal QV2 batch") — unrelated to this entry.

**Not yet committed.** These need to be committed for this entry to be reproducible:

```
M  validation/lsdflow/adapters/workers/sparrow_worker.py
M  experiments/lsd_hubs/campaign/reconcile_t15.py
?? experiments/lsd_hubs/campaign/submit_multiaiz_discover.sh
?? experiments/lsd_hubs/campaign/sparrow_select_frontier.py
?? experiments/lsd_hubs/campaign/results/s3gfn_seh_select_N500/
?? experiments/lsd_hubs/campaign/results/scent_seh_native_freefrag_reconcile/
?? experiments/lsd_hubs/campaign/results/scent_seh_freefrag_k0_reconcile/
?? experiments/lsd_hubs/campaign/results/scent_drd2_native_freefrag_reconcile/
?? experiments/lsd_hubs/campaign/results/scent_drd2_freefrag_k0_reconcile/
?? experiments/lsd_hubs/campaign/results/t15_regress_naive_reconcile/
```

`[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources**

- `[fromer2024sparrow]` — SPARROW, the batch synthesis-planning-and-selection MILP used both as the
  competitor's selector and as our independent cost auditor.
- `[ianez2026multiaiz]` — MultiAiZ, the convergent batch retrosynthesis planner (MolecularAI); runs
  AiZynthFinder over a target set for several cycles, appending discovered intermediates to stock so
  targets converge on shared chemistry.
- `[kim2026s3gfn]` — S3-GFN, the non-reaction baseline generator (GP-MolFormer + soft-synthesizability
  regularization); emits SMILES with no route structure, so a chemist must plan them.

**Packages**

- `aizynthfinder` 4.4.1 (`aizynth` env) — used by `validation/lsdflow/adapters/workers/multiaiz_worker.py`
- `sparrow` + PuLP/CBC (`sparrow` env) — used by `validation/lsdflow/adapters/workers/sparrow_worker.py`
- RDKit (`rgfn` env) — mode counting via `validation/lsdflow/metrics/diversity.py`

### Method

1. **Pool sizing (pre-flight).** Computed how many distinct modes each candidate pool contains as a
   function of its size, using the project's canonical mode metric, to find the smallest pool from
   which 100 modes is reachable. This corrected an earlier estimate that was derived from a
   routability-limited endpoint and was wrong by ~3×.

2. **Built nested competitor pools.** Top-N by surrogate score, above the 7.0 gate, deduplicated, for
   N ∈ {50, 100, 250, 500}; nested so a cost increase is attributable to size, not composition.
   Stock fixed to **plain ZINC** — S3-GFN is ZINC-native, so this is the catalogue a chemist would
   actually hold. (Merging in our own 418 blocks is a planned extension; it should help the
   competitor.)

3. **Route discovery, once per pool.** Four SLURM jobs (72187/72188/72189/72190), MultiAiZ with the
   paper's `n_iters=5`, each persisting `multiaiz_routes.json` + `discovery_timing.json`. The N=500
   artifact is the headline input; the smaller pools are the cost-scaling curve. Because MultiAiZ is
   set-based, routes depend on pool composition, so **only the N=500 artifact is valid for the
   headline** — the cache key is the pool, never the molecule.

4. **Added SELECTION mode to the SPARROW worker.** Verified the objective's sign convention against
   `LinearSelector.set_objective` rather than assuming it: the problem minimizes
   `-w₀·Σ(reward×selected) + w₁·Σ(SM cost) + w₂·Σ(reaction penalty)`. So `weights=[1,0,0,0,0]` plus a
   hard `max_rxns` maximizes selected reward subject to a reaction budget. A hard budget was chosen
   over a reaction *penalty* deliberately — a penalty requires inventing an exchange rate between
   reward units and reactions, which is a free parameter that changes the answer.

5. **Validated both modes.** Pricing mode reproduces the prior DRD2 answer exactly (117 reactions /
   100 targets). Selection mode is monotone and `Optimal` at every budget and, at `max_rxns=117`,
   recovers all 100 targets — i.e. the two modes agree at the boundary where they must.

6. **Swept the competitor frontier.** Network built once from the cached routes; MILP re-solved per
   budget over R ∈ {50…1000}; modes counted in each returned selection with the canonical metric.

7. **Ran the competitor at its strongest** (`--selection greedy`): the same cached routes, but the
   library chosen by a reward-ranked τ-diverse greedy selection and SPARROW used only to PRICE it.
   This removes the objection that the optimizer was judged on diversity, which it does not optimize.

8. **Re-audited our own cost model on the configuration the headline reports.** Exposed the
   acquisition-policy arguments in the reconciler and re-ran with `free_frag` + pre-select-K=20,
   then with K=0 to isolate the up-front charge.

### Results

**Head-to-head at the 100-mode deliverable** (τ=0.5, gate 7.0, both arms priced by SPARROW):

| pipeline | reactions for 100 modes | rxn/mode |
|---|---|---|
| S3-GFN (500) → MultiAiZ → SPARROW selects *(as run)* | **411** (interp. 98@400, 117@500) | 4.11 |
| S3-GFN (500) → from-scratch AiZynth → diverse greedy | 269 | 2.69 |
| **S3-GFN (500) → MultiAiZ → diverse greedy** *(their best)* | **235** | 2.35 |
| LSD-Flow, routes re-derived from scratch | 305 | 3.05 |
| **LSD-Flow native, free-frag K=0** | **131** | 1.31 |
| LSD-Flow native, free-frag K=20 | 135 | 1.35 |

→ **3.13× fewer reactions** against the pipeline as actually run; equivalently at 131 reactions it
reaches **32 modes vs our 100**. Against the competitor's **strongest** configuration (235) the
margin is **1.79×** — the conservative number.

**Note the ordering under from-scratch re-derivation: 305 (ours) vs 269 (theirs) — we LOSE.** That is
the same comparison entries [041]/[048] reported, and it is why this entry exists. The reversal comes
entirely from whether our routes are re-derived or used as generated.

**Prior framing, for contrast** (from entries [041]/[048], from-scratch AiZynth pricing, modes at a
100-reaction budget): S3-GFN 34–35 vs hub-batching 20–27 — the competitor won. Same generators; the
ordering reverses on pricing regime alone.

**The competitor is not a weak baseline:**

| pool | n | median score | max | above 7.0 gate |
|---|---|---|---|---|
| S3-GFN | 2,000 | **8.03** | 8.48 | **99.7%** |
| SCENT (ours, sampled) | 29,997 | 7.61 | 8.40 | 82.9% |

Its routability under its own catalogue was **478/500 = 95.6%** (7,613 candidate routes), so the gap
is not planner failure either.

**SPARROW's diversity blind spot** (selection is *less* diverse than the pool when the budget binds;
the N=500 pool's own mode rate is **41.2%**):

| budget (rxns) | selected | modes | mode rate |
|---|---|---|---|
| 50 | 47 | 16 | 0.34 |
| **100** | **89** | **24** | **0.27** |
| 200 | 164 | 51 | 0.31 |
| 400 | 278 | 98 | 0.35 |
| 600 | 378 | 140 | 0.37 |
| 1000 | 478 | 192 | 0.40 |

**Route-planning cost — the axis only the competitor pays** (LSD-Flow pays zero; linear in pool size):

| N targets | discovery | s/target |
|---|---|---|
| 50 | 833 s | 16.66 |
| 100 | 1,599 s | 15.99 |
| 250 | 4,007 s | 16.03 |
| **500** | **8,085 s (2.25 h)** | 16.17 |

**Cost-model audit across acquisition policies** (100 modes, native routes, all MILP `Optimal`, full
1.00 route coverage). This closes a coherence gap: entry [049]'s 3.12% was measured on the *naive*
policy, not the configuration the headline reports.

| target | policy | K | count-once | SPARROW | rel. diff | rxn/mode |
|---|---|---|---|---|---|---|
| sEH | naive | 0 | 288 | 279 | 3.12% | 2.88 |
| sEH | free-frag | 0 | 125 | 131 | 4.80% | 1.25 |
| **sEH** | **free-frag** | **20** | **134** | **135** | **0.75%** | 1.34 |
| DRD2 | naive | 0 | 117 | 117 | 0.00% | 1.17 |
| **DRD2** | **free-frag** | **0** | **110** | **110** | **0.00%** | **1.10** |
| DRD2 | free-frag | 20 | 134 | 117 | **12.69%** (gate failed) | 1.34 |

**Pre-select-K break-even.** The DRD2 K=20 gate failure is not a cost-model defect. Pre-select-K
charges 26 reactions up front to pre-synthesize fragments; the 100-mode selection does not use ~17 of
them, and SPARROW — which only counts what hindsight shows was needed — charges 117 where we charge
134. Setting K=0 collapses the difference to **exactly 0.00% (110/110)**, confirming the up-front
charge is the entire cause. The direction matters: count-once charges reactions we genuinely spent,
so our published numbers are **conservative**. At 300 modes the same K=20 setting *wins* on DRD2
(1.16 vs 2.65 naive, entry [050]), so the option has a break-even library size rather than being
uniformly good or bad.

**Regression.** The reconciler's defaults still reproduce entry [049] bit-for-bit — best-candidate
314/299 (4.78%), hub-batching 288/279 (3.12%) — so no previously published number is invalidated.

**Reported asymmetries** (deliberately reported, not matched):

| axis | competitor | LSD-Flow |
|---|---|---|
| candidate pool the selector saw | 500 | 26,069 candidates / 64 hubs |
| surrogate calls | ~500 | far higher (155,764 for 300 modes at K=20, entry [050]) |
| route-planning compute | 2.25 h | 0 |
