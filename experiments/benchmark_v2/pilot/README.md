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
| RGFN | agent A's `BudgetCheckpointer` on the trace counter | pending | `trace.csv` + `arm_a.json` |

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

## Still open

- **Arm-A hub structure** — the actual go/no-go. Jobs in flight.
- **`--pool all` depth share** — needs a fresh enumeration of the unfiltered hub set.
- **Per-stage wall clock** for the cost model — partial (SCENT arm-A training: 19 min on one A100).
