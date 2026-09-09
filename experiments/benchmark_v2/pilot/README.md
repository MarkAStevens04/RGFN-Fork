# Arm-A viability pilot (agent B)

**The question.** `benchmark_v2` standardises training at **10,000 oracle calls** (arm A) so the nine
generators are comparable. That is the PMO convention and it is each *competitor's* own paper budget —
but it is **not** a budget our three reaction-GFNs have ever been run at. RGFN gets ~100 gradient
steps there, SCENT and RxnFlow ~157. Hub-batching reads the trained **flow field**, so if that field
carries no structure at 10,000 calls, the whole budget-matched headline collapses. This pilot answers
that for one cell before 108 cells are committed.

**Scope, stated up front.** One target (sEH, the cheapest — surrogate reward, no docking), one seed
(42), two generators. It is a go/no-go probe, not evidence.

---

## VERDICT — arm A breaks SCENT specifically, not hub-batching in general

All four cells priced at the **same gate (5.68)**, same config, same target and seed; only the budget
differs within each generator.

| | HB modes @ R=100 | best-candidate | ratio | rxn/mode |
|---|---|---|---|---|
| RGFN v1 (320,000 calls) | 57 | 25 | 2.28x | 1.890 |
| **RGFN arm A (10,007 calls)** | **65** | 25 | **2.60x** | **1.965** |
| SCENT v1 (320,000 calls) | **96** | 28 | **3.43x** | **1.067** |
| **SCENT arm A (10,048 calls)** | **39** | 25 | **1.56x** | **1.918** |

**RGFN is unharmed** by the 32x budget cut -- slightly better, though on one seed that is a claim of
no damage, not of improvement. **SCENT is halved.** And SCENT at arm A costs 1.918 rxn/mode against
RGFN's 1.965 at the same budget, within 2.5%: stripped of its library it lands on the library-less
cost profile rather than degrading in a SCENT-specific way.

So the earlier framing here -- "arm A is viable for the external head-to-head, NOT for hub-batching"
-- was too broad and is superseded. Hub-batching is fine at arm A on a generator that never had a
library. The penalty is entirely SCENT's, and entirely the library.

**The enumeration mismatch does not explain it and cuts the other way.** Both arm-A runs used the
*smaller* enumeration (64 hubs at `enum_max` 4000 against v1's 200 uncapped). The same handicap
applied to both generators; only SCENT collapsed, and RGFN improved under it. It remains a caveat on
exact magnitudes, not a candidate explanation.

---

## Finding 0 — the mechanism is the LIBRARY, not the budget (controlled 2x2)

The cleanest result here. RGFN never has a dynamic library at any budget; SCENT has one only at v1.
Top-40 hub depths under `--pool all`, sEH seed 42:

| generator | arm A (10k calls) | v1 (320k calls) | moves with budget? |
|---|---|---|---|
| **RGFN** (never has a library) | {1:1, 2:5, 3:34} | {1:2, 2:4, 3:34} | **no -- identical** |
| **SCENT** (library only at v1) | {2:3, 3:37} | {0:6, 1:30, 2:4} | **yes -- flips 3 -> 1** |

Budget alone does not make a flow field hub deep: RGFN is depth-3 at both budgets. SCENT moves
*because* the budget is what decides whether it has a library. That isolates the mechanism to the
library rather than to training length, which no single-generator comparison could do.

---

## Finding 1 — ⛔ SCENT's dynamic library never activates at arm A

**Confirmed three ways: from the clone config, from the pilot's own resolved `operative_config.gin`,
and by direct observation of the finished run.**

SCENT promotes fragments on a fixed **iteration** schedule, not a budget-relative one:

```
DynamicLibrary.every_n_iterations = 1000
DynamicLibrary.num_additions      = 10     ->  promotions at iterations 1000, 2000 ... 10000
DynamicLibrary.n_new_fragments    = 400
```

Arm A is 157 iterations. **The first promotion is at iteration 1,000 = 64,000 oracle calls — 6.4× the
entire arm-A budget.** The finished pilot run confirms it: `additional_fragments/` was never created,
so **zero** fragments were promoted. Its checkpoint is **28 MB against v1's 205 MB**, the action-space
signature of an empty library.

Cross-check on the arithmetic: v1 (5,000 iterations = 320,000 calls) fired only **4 of the 10**
scheduled promotions, writing `fragments_1000..4000.json` — exactly the 1,600 promoted fragments its
enumeration metadata records.

**Three consequences, all specific to arm A:**

1. **SCENT's distinguishing feature is inert.** At 10,000 calls it is a reaction-GFN with no library
   learning — architecturally nearer RGFN than the SCENT the project has been reporting.
2. **No recipes exist** (artifact ③), so the cell fails `check_route_readiness`. SCENT arm-A cells
   **cannot enter the route dataset**, which contradicts both runbook §6.4's hard gate and the
   "matrix-wide route dataset" claim.
3. **`--child-policy free_frag` is a no-op** (nothing promoted to filter on) and `--prebuild-k` has
   nothing to stock. The hub-batching configuration standardised in runbook §2.2 is meaningful only
   at arm B.

This is a research decision. Options, none of them built for: scale the promotion schedule to the
budget (a disclosed deviation that changes what SCENT *is*); accept and report it as a finding about
library-learning methods' sample efficiency; or run SCENT at arm B only.

---

## Finding 2 — the depth-0 share of a delivered library, measured

Previously unmeasured, and **unrecoverable for v1** (the hub-ordering arms kept only plots, not
per-step curves). `scent_seh`, gate 5.68, τ 0.5, `free_frag` + `prebuild-k 0`, R=100.
The hub join is **exact**: 96/96 accepted molecules resolve to a hub in `hubs.csv`, 0 unjoined.

| arm | HB modes@100 | hubs walked | hub depths | modes by depth | reactions by depth | **depth-0 mode share** |
|---|---|---|---|---|---|---|
| base | 96 | 5 | 2×d0, 2×d1, 1×d2 | 35 / 48 / 13 | 35 / 50 / 15 | **36.5%** |
| `--min-synth-depth 2` | 94 | 5 | 4×d1, 1×d2 | 81 / 13 | — | 0.0% |
| `--min-synth-depth 3` | 71 | 19 | 10×d1, 8×d2 | 15 / 55 | — | 0.0% |

**Three readings.**

1. **Counting hubs understates the reliance ~36×.** Two of 200 hubs — 1% of the set — deliver 36.5%
   of the library. The *delivered-mode* share is the number to report, never the hub count.
2. **The mechanism, and it is exact.** Reactions/mode by hub depth is **1.000 / 1.042 / 1.154** for
   depth 0 / 1 / 2. A depth-0 hub is *bought*, so it contributes nothing to the build and each child
   is a single coupling: **1.000 rxn/mode is the theoretical floor a library can reach**. That is
   precisely why the flow ranking puts such hubs at the top, and why their share of the delivered
   library so far exceeds their share of the hub set.
3. **And it is cheap to give up.** Excluding buy-and-couple molecules entirely costs **96 → 94 modes
   (−2.1%)**; the walk simply moves onto depth-1 hubs, delivering 81 modes off them instead of 48.
   This is the answer to the degenerate-optimum objection: the headline can be reported with
   catalogue picking excluded for ~2%. It only bites at `--min-synth-depth 3` (71 modes, −26%, and
   the walk spreads over 19 hubs instead of 5).

**⚠ 36.5% is a LOWER BOUND, not an estimate.** It was measured on the **v1 reward-pre-filtered** hub
set, which contains 2 depth-0 hubs of 200. The `--pool all` set that `benchmark_v2` adopts holds
**11** of 200 (eligible pool: 54 depth-0 of 20,874). If two depth-0 hubs deliver 35 of 96 modes,
eleven plausibly deliver more. The true `--pool all` figure needs a fresh enumeration of the
unfiltered hub set — cheap on a surrogate target, **not yet run**.

**Two further caveats.** This runs on a **320,000-call** checkpoint, so it answers the *depth*
question, not the *viability* one. And best-candidate's depth mix is **incomplete** — 12 of its 26
named hubs are absent from `hubs.csv` and a stereo-strip fallback recovers none — so no BC depth-0
share is quoted.

---

## Finding 3 — arm-A budgets land exactly

| generator | mechanism | achieved | evidence |
|---|---|---|---|
| SCENT | 157 iterations × batch 64 | **10,048 calls** | `paths.csv` last iteration 10,047, 10,048 rows |
| RGFN | agent A's `BudgetCheckpointer` on the trace counter | **10,007 calls at iteration 83** | `arm_a.json`: `fired: true, crossed_at_n_scored: 10007, saved_at_iteration: 83` |

**The RGFN row is the case for the trace counter over arithmetic.** I planned 100 iterations expecting
100 calls each. It crossed at iteration **83**, because RGFN scores `train_forward_n_trajectories`
(100) **plus** `train_replay_n_trajectories` (20) = 120.6/iteration measured. Placing the checkpoint by
step count would have trained RGFN ~20% past the arm-A budget and made the external head-to-head
quietly unfair in our favour.

**SCENT emitted no `trace.csv`.** At this pilot's launch, agent A's instrumentation covered RGFN's
`glue/fixed_reward/pipeline.py` but not `validation/generators/scent/run_scent_fixed.py`. SCENT's
budget is exact anyway because its batch is fixed and its schedule deterministic — but that is
arithmetic, not measurement, and it does not generalise to a generator with replay.

---

## Finding 4 — hazard: worktree isolation does not isolate Python

`$CONDA_PREFIX/lib/python3.11/site-packages/rgfn.pth` contains the **main checkout** path, so
`import glue` / `import rgfn` resolve there no matter which worktree launched the job. This pilot's
own log records `repo=<my worktree>` while executing another agent's uncommitted `glue/` code.

Two consequences: a job's results are **not attributable to its worktree's commit**, and two jobs
started minutes apart can silently run different code. Python reads a module fully at import, so a
mid-run edit cannot corrupt a running process — but nothing records which version ran. This belongs
beside the "never edit a running bash script" hazard in the runbook.

---

## Reproducing

```bash
# stage 1 — train to arm A into an isolated root (REFUSES any OUT_ROOT inside the v1 tree)
OUT_ROOT=$SCRATCH/rgfn_runs/v2_pilot N_ITERS=157 \
  sbatch experiments/benchmark_v2/pilot/submit_pilot_train.sh scent seh 42

# stage 2 — sample the arm-A checkpoint and rank its hubs (--pool all, runbook 2.2)
GEN=scent CKPT=<run>/train/checkpoints/last_gfn.pt GUIDANCE=<run>/.../guidance_models.pt \
  OUT=$SCRATCH/rgfn_runs/v2_pilot/hubs/scent_armA \
  sbatch experiments/benchmark_v2/pilot/submit_pilot_hubs.sh

# depth mix on a cached enumeration (debug partition; login-node imports get SIGINT here)
sbatch -p debug --gpus-per-node=1 experiments/benchmark_v2/pilot/run_depthmix.sh
bash experiments/benchmark_v2/pilot/summarise_depthmix.sh
```

`submit_pilot_train.sh` exists because `submit_rgfn.sh` hardcodes `FR_ROOT_DIR` with no override and
would resolve seh/42 to the **live v1 cell**; this one refuses any `OUT_ROOT` under the v1 tree.

Committed results: `results/depthmix_{base,d2,d3}.json`.

---

## Timings — arm A, sEH, one A100 (published numbers, per runbook's compute-vs-reactions exhibit)

| stage | SCENT | RGFN |
|---|---|---|
| train to arm-A budget | 1,159 s (19 min) | 3,864 s (64 min) |
| — `train_gfn` | — | 3,166 s |
| — `sample_batch` | — | 659 s |
| stage-2 sample, 30,000 trajectories | 1,189 s | 4,324 s |
| `pick_hubs` (either pool) | 0.18 s | 0.19 s |
| enumerate, 64 hubs, `enum_max` 4000 | 922 s | — |
| campaign | 17 s | — |

RGFN costs ~3.3x SCENT per oracle call at arm A.

**DO NOT quote RGFN's `score: 0.052 s`.** `timing.json` reports 52 ms for scoring 16,919 molecules --
3 us each -- which is not credible for an MPNN forward pass. The proxy call is most likely happening
inside `train_gfn` and the `score` timer wrapping something else, i.e. mis-attributed rather than
genuinely free. A near-zero column reads as "measured and fast" when it may be "not measured here";
confirm the measurement happened before treating it as a finding. These timings should be re-emitted
through C's `write_sample_timings()`, which omits an unmeasured component rather than writing 0.0.

---

## Defects found, and where they landed

1. **`pick_hubs` silently overwrites its own provenance.** Both sidecars go to a FIXED name in the
   parent of `--out` (`hub_scores.csv`, `pick_hubs_timing.json`, lines 276/289), so two invocations
   differing only in `--out` basename -- exactly what a pool comparison does -- share one sidecar and
   the second clobbers the first. It bit this pilot twice: I analysed a `topk` ranking believing it
   was `--pool all`, caught it because `pick_hubs_timing.json` said `pool: topk_candidates`, and
   re-ran into separate directories. **This is the cause of the project's standing "hub_scores.csv is
   often STALE, never join against it" warning** -- the file does not age, the next run overwrites it.
   Reported to agent C.
2. **`run_campaign`'s hub provenance is null on any real cell.** `depth_mix` and
   `n_promoted_fragments` populate correctly (verified here -- and `n_promoted_fragments: 0`
   independently confirms Finding 1 from a second code path). But `hub_pool`, `min_hub_depth`,
   `max_hub_depth` and `walked_depth_hist` came back **null**, because `pick_hubs` is a separate
   stage and the campaign is never told which pool produced the enumeration. In the production
   pipeline those stages are *always* separate, so this is not an artifact of how I ran it.
3. **v1 RGFN's flow ranking rests on unreplicated estimates.** Median observed children backing a
   top-40 hub: **1**, with **72.5%** singletons -- against 21 (12.5% singletons) for the arm-A RGFN
   sample and 10 for v1 SCENT. Any v1 RGFN hub-ordering claim inherits that.

---

## Still open

- **`--pool all` depth-0 share** — needs a fresh enumeration of the unfiltered hub set.
- **Matched-enumeration re-measure** of the arm-A vs v1 ratio (64/capped vs 200/uncapped).
- **DRD2 and the other seeds** — `pick_hubs` is 0.2 s, so the depth grid extends for seconds; the
  campaign half does not.
- **Re-emit timings** through `write_sample_timings()`.
