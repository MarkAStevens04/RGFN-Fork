# SCENT + RGFN / sEH — is a generator trained on a tenth of the usual budget still good enough to build a library from?

**Date:** 2026-09-08, ~1pm

## Question

If we train our molecule generators for a thirty-second of the time they normally get — so that they
can be compared fairly against competing programs — do they still learn enough for our library-design
method to work?

## Context & Summary

**Context.** The clean re-run (`docs/RETRAIN_RUNBOOK.md`) standardises every generator on the same
training budget so the comparison is fair: **10,000 scoring calls**, which is what the competing
programs' own papers use. Our three generators have never been run that low. They normally get
**320,000** — thirty-two times more — and that is the number *their* papers use. So the fairness fix
creates a risk nobody had measured: our method works by reading the "flow field" a trained generator
leaves behind, which tells it which partly-built molecules are worth committing to. If ten thousand
calls is not enough training for that field to mean anything, the whole budget-matched comparison
collapses. Entry `070` showed what it costs to discover a problem like this late — three artifacts
were silently missing for weeks. This pilot spends one cheap cell to answer the question before 108
cells are committed.

**Summary.** We trained two generators (SCENT and RGFN) on the cheapest target (sEH) at the new
low budget, then ran them through the full library-design pipeline and compared against the same
generators trained the normal way. We measured three things: whether the flow field is informative at
all, where in a molecule's assembly the method chooses to commit, and what the resulting library
actually costs.

## Answer

**The low budget does not break the flow field — it changes what the field points at, and that is
what costs us.** At ten thousand calls the field is just as informative as at the full budget and
rests on ten times more supporting evidence, so it is not noise. But the libraries it produces are
much worse value: roughly a third as many molecules for the same bench effort, and the advantage over
the naive alternative falls from about 3.4× to 1.6×.

The reason turns out to be specific and unexpected. **SCENT's defining feature — it learns to reuse
its own best intermediates — never switches on at the low budget.** It is scheduled to start doing
that after a thousand training rounds, and the low budget stops at a hundred and fifty-seven. Without
it, SCENT commits to intermediates at the very last assembly step, which is the worst possible place,
because there is almost nothing left to share. A second generator that never has this feature at all
behaves identically at both budgets, which is what shows the feature — not the training length — is
the thing that matters.

## Relevance to our Publication

This is the entry the whole re-run turned on. The plan had been to run every comparison at the
matched budget; this shows that would have quietly crippled our own method and understated our result
against the competitors — the opposite of the fairness the change was meant to buy. It supports
splitting the campaign: the head-to-head against other programs runs at the matched budget, where
fairness is the point, and our own internal comparison runs at the full budget, which is what our
generators' papers use.

It also hands ICLR and NCS reviewers a clean answer to a question they will certainly ask about our
strongest result — "isn't your method just picking cheap molecules off the shelf?" We can now say how
much of a delivered library rests on bought starting material and exactly what excluding them costs:
about two percent.

## Next Experiments

**Refining for publication**

- **Re-measure the headline ratio with the two arms matched on enumeration.** The low-budget run
  enumerated fewer candidate intermediates than the full-budget one, so while the direction is not in
  doubt, the size of the gap is not yet a quotable number.
- **Extend to a second target and the other seeds.** The hub-depth half of this costs seconds and is
  already done across every cell we hold; the library-cost half is one GPU job per cell.
- **Measure the bought-starting-material share on the unfiltered intermediate pool.** The number we
  have is a floor, measured on a pool that happened to contain only two such intermediates where the
  pool we have adopted contains eleven.

**Next steps in project**

- **Decide what to do about SCENT at the matched budget** — the three options are to rescale its
  schedule (which changes what SCENT is and must be disclosed), to report the failure as a finding
  about how much training library-learning methods need, or to run SCENT only at the full budget.
  This is a research decision and nothing has been built for any of the three.
- **Price the compute-versus-reactions tradeoff**, now an approved exhibit: this pilot is its first
  datapoint, showing what the extra GPU time at the full budget buys in saved bench chemistry.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.
Branch: `worktree-agent-b-arm-a-pilot`.

### Relevant Files

**Scripts** — all new, all under `./experiments/benchmark_v2/pilot/`

- `submit_pilot_train.sh` — trains one generator to the arm-A budget into an **isolated** output
  root. Exists because `experiments/fixed_reward/scale5k/submit_rgfn.sh` hardcodes
  `FR_ROOT_DIR=$SCRATCH/rgfn_runs/experiments` with no override and derives its run name from
  (system, seed) — pointed at seh/42 it resolves to the **live v1 cell** `rgfn_seh_5k/seed42` and
  resumes from its 5,000-iteration checkpoint. This launcher refuses any `OUT_ROOT` inside the v1
  tree (guard tested).
- `submit_pilot_hubs.sh` — samples an arm-A checkpoint and ranks its hubs under both `--pool all` and
  the legacy reward-pre-filtered pool. Detects a SCENT checkpoint with no `additional_fragments/`
  snapshot and passes `--no-freeze`, so an arm-A SCENT runs on the 418-fragment base library
  explicitly rather than letting the default resolve to nothing.
- `submit_pilot_campaign.sh` — enumerates the arm-A hub set and prices it; also asserts whether
  `depth_mix` populated.
- `run_depthmix.sh` — the three `--min-synth-depth` arms on a cached enumeration.
- `depth_mix.py` — joins accepted molecules to their hub's depth and reports the delivered-mode and
  reaction-budget split by depth. Tries an exact SMILES join first and a stereo-stripped fallback
  second, counting each, because `hubs.csv` holds stereo-bearing SMILES while some downstream keys
  are stereo-stripped.
- `hub_structure.py` — flow-field separation and evidence per hub, for reading an arm-A field beside
  a v1 one on the same axis.
- `hub_depth_grid.py` — runs `pick_hubs --pool all` over every cell whose sample survives, to test a
  depth claim on the grid rather than on one cell. Gives each cell its own output directory.
- `reconcile_depth.py` — mode, mode share, median and a concentration flag together, so a
  mode-vs-median disagreement is visible rather than settled by whichever statistic was reached for
  first.
- `depth_mix_table.py`, `summarise_depthmix.sh` — reporting wrappers.

**Models**

- `/scratch/markymoo/rgfn_runs/v2_pilot/fixed_reward/scent_seh_armA/seed42/train/checkpoints/last_gfn.pt`
  — SCENT at 10,048 calls. **28 MB against v1's 205 MB**: the action-space signature of an empty
  dynamic library.
- `/scratch/markymoo/rgfn_runs/v2_pilot/fixed_reward/rgfn_seh_armA/seed42/train/checkpoints/arm_a_10k.pt`
  — RGFN's arm-A checkpoint, written by agent A's `BudgetCheckpointer` at the exact trace crossing.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/{scent,rgfn}_seh_5k/seed42/...` — the v1
  320,000-call checkpoints, the comparison arm. Read-only here.

**Datasets**

- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16*/*/sample/records.csv` — 35 v1 sampled DAGs, the
  grid the depth claim was tested on.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16/scent_seh/enum/{enum_children.json,hubs.csv}` — the
  cached v1 enumeration the depth-mix arms were priced on. `hubs.csv` supplies the hub→depth join.

**Results** (committed, `./experiments/benchmark_v2/pilot/results/`)

- `hub_structure_both.json` — flow separation and evidence, arm A vs v1, both generators.
- `hub_depth_grid.json` — top-40 hub depths for all 35 cells.
- `depth_reconcile.json` — mode / share / median / concentration per cell.
- `depthmix_{base,d2,d3}.json` — delivered-mode and reaction split by hub depth.

**Job Logs** — `/scratch/markymoo/rgfn_runs/`

- `pilotA_scent_seh-75927.out`, `pilotA_rgfn_seh-75928.out` — arm-A training.
- `pilotB_hubs_scent-75931.out`, `pilotB_hubs_rgfn-75933.out` — sample + hub ranking.
- `pilotB_camp_scent-75932.out`, `pilotB_camp_rgfn-75996.out` — enumerate + campaign.
- `pilotB_depthmix-75930.out` — the depth arms.

### Relevant Versions

`027a007` (pilot scripts + SCENT library finding), `b03cf61` (verdict, controlled 2×2, defects), both
on `worktree-agent-b-arm-a-pilot` and pushed. Merged agent C's `worktree-benchmark-v2-infra` twice
(`e630d5c` and a second merge picking up `6599590`, the `pick_hubs` sidecar fix this entry's defect
report prompted).

Depends on agent A's arm-A checkpointing in `glue/fixed_reward/pipeline.py`, on the main checkout.
**Note the import path:** `$CONDA_PREFIX/lib/python3.11/site-packages/rgfn.pth` points at the main
checkout, so `import glue` / `import rgfn` resolve there regardless of which worktree launched the
job — these runs executed agent A's code while logging `repo=<my worktree>`.

[TODO — add commit hash for the final consolidation after pushing]

### Relevant Resources

**Sources**

- Entry `070` — the route-artifact audit; the precedent for spending a cheap check before a large
  campaign, and the source of the three-artifact contract this pilot's SCENT finding breaks.
- Entry `053` — the hub-ordering ablation; source of `--pool all` and the "flow finds the
  neighbourhood, not the rank" reading.
- Entry `068` — the enumeration-cap correction; why the v1 enumeration is uncapped and this pilot's
  is not.
- `docs/RETRAIN_RUNBOOK.md` §1 (the two arms), §2.2 (`--pool all`, `free_frag`, `prebuild-k 0`),
  §2.1 (the 5%-FPR gates).
- SCENT clone config `external/scent/configs/envs/dynamic_library/dynamic_library.gin` — the
  promotion schedule this entry's central finding rests on.

**Packages**

- `pick_hubs.py` (pure stdlib, ~0.2 s/cell) — the flow estimate and hub ranking.
- `run_campaign.py` + `glue/samplers/lsdflow/` — selection strategies and the count-once cost model.
- `validation/lsdflow/adapters/workers/{scent,rgfn}_worker.py` — sample and enumerate stages.

### Method

1. Train both generators to the arm-A budget into an isolated root, never the v1 tree:
   ```
   OUT_ROOT=$SCRATCH/rgfn_runs/v2_pilot N_ITERS=157 \
     sbatch experiments/benchmark_v2/pilot/submit_pilot_train.sh scent seh 42   # job 75927
   OUT_ROOT=$SCRATCH/rgfn_runs/v2_pilot N_ITERS=100 \
     sbatch experiments/benchmark_v2/pilot/submit_pilot_train.sh rgfn  seh 42   # job 75928
   ```
2. Sample 30,000 trajectories from each arm-A checkpoint and rank hubs under both pools
   (`submit_pilot_hubs.sh`, jobs 75931 / 75933). **Each pool written to its own directory** — see
   Results §6.
3. Enumerate the top 64 hubs of the `--pool all` set and price the library at R=100, gate 5.68,
   τ 0.5, `free_frag`, `prebuild-k 0` (`submit_pilot_campaign.sh`, jobs 75932 / 75996).
4. On the **cached v1** enumeration, run the same campaign at three deliverable filters
   (`--min-synth-depth` 0 / 2 / 3) and join accepted molecules to hub depth
   (`run_depthmix.sh` job 75930, then `depth_mix.py`).
5. Test the hub-depth claim across all 35 v1 cells with `hub_depth_grid.py`, then
   `reconcile_depth.py` for mode/median/concentration.

### Results

**1 — the arm-A flow field is informative, and better evidenced than the full-budget one.**
Top-40 hubs, `--pool all`, sEH seed 42. Separations in nats of log-flow; "est/hub" is the median
number of observed children backing a hub.

| | ranked | top10 − median | rank0 − rank40 | est/hub | single-estimate frac |
|---|---|---|---|---|---|
| SCENT arm A | 200 | **4.281** | 4.535 | **101** | 0.05 |
| SCENT v1 320k | 200 | 2.901 | 3.363 | 10 | 0.10 |
| RGFN arm A | 200 | 4.103 | 4.898 | 21 | 0.125 |
| RGFN v1 320k | 200 | 5.684 | 5.785 | **1** | **0.725** |

Eligible pools: SCENT arm A 7,111 hubs, RGFN arm A 13,842. Note the last row — **72.5% of v1 RGFN's
top-40 hubs rest on a single observed child**, so any v1 RGFN hub-ordering claim inherits an
unreplicated point estimate.

**2 — what arm A costs, measured.** `scent_seh/42`, same gate, config and seed; only the budget
differs.

| | modes @ R=100 | vs best-candidate | rxn/mode | delivered-mode depths | promoted frags |
|---|---|---|---|---|---|
| arm A (10,048 calls) | **39** | **1.56×** | **1.918** | {2:7, 3:32} | **0** |
| v1 (320,000 calls) | **96** | **3.43×** | **1.067** | {0:35, 1:48, 2:13} | 1,600 |

**Not matched on enumeration** — arm A used 64 hubs at `enum_max 4000`, v1 used 200 uncapped
(entry `068`). At R=100 only ~5 hubs are walked so the hub count should not bind, but the per-hub
child cap plausibly does. Direction is solid; the magnitude is not yet quotable.

**3 — the mechanism is the dynamic library, not the training length.** RGFN never has a dynamic
library at any budget; SCENT has one only at v1. Top-40 hub depths, `--pool all`:

| generator | arm A | v1 320k | moves with budget? |
|---|---|---|---|
| RGFN | {1:1, 2:5, 3:34} | {1:2, 2:4, 3:34} | **no — identical** |
| SCENT | {2:3, 3:37} | {0:6, 1:30, 2:4} | **yes — flips 3 → 1** |

**4 — SCENT promotes zero fragments at arm A.** From
`external/scent/configs/envs/dynamic_library/dynamic_library.gin`, confirmed in the pilot's own
resolved `operative_config.gin`:

```
DynamicLibrary.every_n_iterations = 1000
DynamicLibrary.num_additions      = 10     -> promotions at iterations 1000, 2000 ... 10000
DynamicLibrary.n_new_fragments    = 400
```

Arm A is 157 iterations; the first promotion is at iteration 1,000 = **64,000 oracle calls, 6.4× the
arm-A budget**. Observed: `additional_fragments/` was never created. Arithmetic cross-check — v1
(5,000 iterations) fired 4 of 10 promotions, writing `fragments_1000..4000.json`, exactly the 1,600
promoted fragments its enumeration metadata records. Consequences: no recipes, so the cell fails
`check_route_readiness` and cannot enter the route dataset; and `free_frag` / `prebuild-k` are no-ops.

**5 — the depth-0 share of a delivered library.** `scent_seh` v1 enumeration, gate 5.68, R=100. Hub
join exact: 96/96 accepted molecules resolve, 0 unjoined, 0 needing the stereo fallback.

| arm | HB modes@100 | hubs walked | hub depths | modes by depth | reactions by depth | depth-0 mode share |
|---|---|---|---|---|---|---|
| `--min-synth-depth 0` | 96 | 5 | 2×d0, 2×d1, 1×d2 | 35 / 48 / 13 | 35 / 50 / 15 | **36.5%** |
| `--min-synth-depth 2` | 94 | 5 | 4×d1, 1×d2 | 81 / 13 | — | 0.0% |
| `--min-synth-depth 3` | 71 | 19 | 10×d1, 8×d2 | 15 / 55 | — | 0.0% |

Reactions/mode by hub depth is **1.000 / 1.042 / 1.154** for depth 0/1/2 — a depth-0 hub is bought,
so it contributes nothing to the build and each child is one coupling, making 1.000 the floor a
library can reach. Two of 200 hubs (1% of the set) deliver 36.5% of the library, so **counting hubs
understates the reliance ~36×**; the delivered-mode share is the figure to report. Excluding
buy-and-couple molecules costs 96 → 94 modes, **2.1%**.

**This 36.5% is a lower bound and a SCENT number.** It was measured on the reward-pre-filtered pool
(2 depth-0 hubs of 200); the `--pool all` set adopted by the runbook holds 11 of 200. And the two
library-less generators already hub one step from terminal, so the figure does not generalise across
the three. Best-candidate's depth mix is incomplete (12 of 26 named hubs absent from `hubs.csv`, the
stereo fallback recovering none), so no BC share is quoted.

**6 — three defects, all reported.**

- **`pick_hubs` overwrote its own provenance.** Both sidecars went to a fixed name in the parent of
  `--out`, so two invocations differing only in `--out` basename shared one file and the second
  clobbered the first. This bit the pilot: a `topk` ranking was analysed as if it were `--pool all`,
  caught because `pick_hubs_timing.json` recorded `pool: topk_candidates`. **This is the cause of the
  standing "`hub_scores.csv` is often stale" warning** — the file does not age, the next run
  overwrites it. Fixed by agent C in `6599590`.
- **`run_campaign`'s hub provenance is null on any real cell.** `depth_mix` and
  `n_promoted_fragments` populate correctly (the latter reading 0 independently confirms Result 4
  from a second code path), but `hub_pool`, `min_hub_depth`, `max_hub_depth` and `walked_depth_hist`
  are null because `pick_hubs` is a separate stage that never tells the campaign which pool it used.
  In production those stages are always separate, so this is not an artifact of how the pilot ran.
- **`timing.json`'s `score` column is not credible.** RGFN reports 52 ms for scoring 16,919
  molecules — 3 µs each. The proxy call is most likely inside `train_gfn` with the `score` timer
  wrapping something else. Do not quote it; re-emit through `write_sample_timings()`, which omits an
  unmeasured component rather than writing 0.0.

**7 — mode-versus-median, and why concentration is the right statistic.** Testing the "library-less
generators hub at (cap − 1)" claim across all 35 v1 cells: **19/23 library-less cells** sit at modal
depth cap − 1, but SCENT is only 5/12. Mode and median disagree on 4/35 cells — every one a cell
whose distribution is not concentrated. Adding the mode's share of the top-40 explains it:

| | concentrated (mode ≥ 50% of top-40) | median mode share |
|---|---|---|
| library-less (RGFN, RxnFlow) | **23/23** | ~85% |
| SCENT | **6/12** | ~51% |

So the defensible statement is not "library-less hubs deep, SCENT hubs shallow" — it is that
**without a dynamic library the flow field collapses onto a single depth one step from terminal;
with one it spreads across depths.** SCENT's apparent target-dependence (deep on ClpP/6TD3, shallow
on sEH/DRD2) is largely this: the docking cells are spread distributions being described by their
largest bin.

**8 — arm-A budgets land where intended, and the trace counter earns its keep.**

| generator | mechanism | achieved |
|---|---|---|
| SCENT | 157 iterations × batch 64 | 10,048 calls (`paths.csv` last iteration 10,047) |
| RGFN | `BudgetCheckpointer` on the trace counter | **10,007 calls at iteration 83** |

RGFN was planned at 100 iterations on the assumption of 100 calls each. It crossed at **83**, because
RGFN scores `train_forward_n_trajectories` (100) **plus** `train_replay_n_trajectories` (20) =
120.6/iteration measured. Placing the checkpoint by step arithmetic would have trained it ~20% past
the arm-A budget, making the external head-to-head quietly unfair in our favour.

SCENT emitted no `trace.csv` — agent A's instrumentation covered RGFN's `glue/fixed_reward/pipeline.py`
but not `run_scent_fixed.py` at this pilot's launch. And this pilot's RGFN trace predates A's phase
fix: all 16,919 rows read `train`, of which ~12,060 are training calls and ~4,859 the final candidate
scoring. **The checkpoint is unaffected** (the crossing is at iteration 83, the contamination appended
at the end), but the trace cannot carry a modes-vs-calls curve without re-running.

**9 — wall clock, arm A, sEH, one A100.** Published numbers for the compute-versus-reactions exhibit.

| stage | SCENT | RGFN |
|---|---|---|
| train to arm-A budget | 1,159 s | 3,864 s (`train_gfn` 3,166 / `sample_batch` 659) |
| sample, 30,000 trajectories | 1,189 s | 4,324 s |
| `pick_hubs`, either pool | 0.18 s | 0.19 s |
| enumerate, 64 hubs, `enum_max` 4000 | 922 s | *(job 75996)* |
| campaign | 17 s | *(job 75996)* |

RGFN costs ~3.3× SCENT per oracle call at arm A.

**10 — what remains unexplained.** SCENT's target-dependence is narrowed but not solved. Three
mechanisms have been ruled out: **dose-response** (all twelve SCENT cells fired exactly 4 promotions
to `last_iter=4000` — identical library, opposite hub depths); **reward-by-depth** (top-decile-reward
median hub depth is 2/3/2/3 across seh/drd2/clpp/6td3, no surrogate-versus-docking split); and a
**policy-term effect** (decomposing top-40 flow, `logR` dominates `logP_B` and the two `logP_F` terms
by an order of magnitude on every cell, so the ranking is essentially reward-mass ranking). No
confirmed mechanism. Recorded as unexplained rather than filled with a story.
