# SynFormer — why nine training runs died at hour ten, and what it took to make the tenth survive

**Date:** 2026-08-27, ~2pm

## Question

Why did every attempt to train the one competitor that generates molecules together with their
recipes stall part-way through, and can it be made to run reliably?

## Context & Summary

**Context.** The benchmark compares our method against five competitors on the same question: given a
fixed budget of 100 chemical reactions, how many genuinely different high-quality molecules does each
produce? Four of the five emit only molecules, so we buy them recipes afterwards with a retrosynthesis
planner — about five hours of GPU per cell. SynFormer (`[gao2025synformer]`) is the exception and the
most interesting comparison in the set: it generates molecules *as* synthesis pathways, so its recipes
are free, exactly as ours are. That makes it the closest baseline to our own method, and the one a
reviewer will look at hardest when asking whether our advantage comes from the flow field or merely
from being reaction-grounded.

Nine cells (three targets x three seeds) were launched on 2026-08-22/23 with a three-day walltime
each. Eight hit the walltime; one finished. Every one of them stopped producing molecules at hour
9-12 and then sat silent for the remaining ~60 hours. Twenty-seven GPU-days bought one usable cell.
Entry [067]'s pool machinery and entry [070]'s recipe audit both depend on having this entrant, so
the column could not simply be dropped.

**Summary.** We established the authors' own configuration as a control before touching any of our
own modifications, then reintroduced our differences one at a time. That sequence located the fault
in our code rather than in SynFormer, and turned up four separate reasons a worker process could not
be spawned — each with a different symptom, each individually sufficient to break the run. Two of the
three targets are now fixed and verified end-to-end; the third is understood and blocked on a
structural change we have not made.

## Answer

The runs were never slow: they were spawning worker processes that could not start, and then waiting
for results that would never arrive. The cause was in our harness, not in SynFormer — four distinct
defects, of which the most consequential was a single stray import line that opened graphics-card
handles on behalf of code that never needed them, breaking two targets that never touch that code at
all. Two targets (DRD2, ClpP) now complete generations, bound their memory, and emit recipes; the
third (sEH) cannot be fixed the same way, because loading its reward model in the coordinating
process is fundamentally incompatible with spawning workers from it.

Two things are worth carrying beyond SynFormer. First, the memory exhaustion that looked like the
problem was a *consequence*: the authors discard their workers every generation, we kept ours alive,
and the mechanism that would have bounded the leak was exactly the one the defects broke. Second,
the diagnostic that finally worked measures the fault directly and never asks the library whether a
GPU is present — because asking is itself one of the four ways to cause it.

## Relevance to our Publication

ICLR reviewers will ask whether the strongest reaction-aware baseline was given a fair run, and
SynFormer is that baseline — the only competitor that, like our method, carries routes by
construction. An empty or one-cell SynFormer column invites the reading that the comparison was
arranged rather than measured. This entry also produces a fairness statement we can make positively:
running the authors' own loop unmodified showed that *their* generation step crashes stochastically
on their own crossover code, so one of our deviations is a required bug fix rather than a preference.
Being able to say precisely which deviations were forced, and to show the control run that
established it, is stronger than asserting the configuration was fair.

## Next Experiments

**Refining for publication.** The verified runs used 120-400 molecule budgets where production cells
use 10,000, and the original failure only surfaced at hour ten — so a full-length DRD2 or ClpP cell
must complete before the column is quotable. Reviewers will also reasonably ask whether we can
reproduce SynFormer's own published numbers; that is a separate experiment from this one, needs two
datasets we do not currently cache, and would be the cleanest possible evidence that the baseline was
run competently.

**Next steps in project.** Complete the five runnable cells (DRD2 seeds 42/44, ClpP seeds 42/43/44),
then decide whether the three sEH cells justify moving the reward model out of the coordinating
process — the pattern the docking path already uses. sEH is the target with the most complete data
for every other competitor, so an absent SynFormer sEH column is the most visible gap in the matrix.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/synformer_baseline/upstream_control.py` — runs SynFormer's own GraphGA-SF loop with
  none of our modifications, importing their `projection`, `make_mating_pool`, `reproduce` and
  `sanitize` directly rather than reimplementing them. One flag per divergence (`--workers`,
  `--routes`, `--torch-in-parent`, `--patched-reproduce`) so each can be enabled alone. Written
  because with three divergences live a failure has four candidate causes.
- `./experiments/synformer_baseline/submit_upstream_control.sh` — SLURM wrapper for the control.
- `./validation/generators/synformer/run_synformer_fixed.py` — our adapter. Carries `_cuda_probe`,
  which reports the nvidia file-descriptor count and thread count at each process-spawn site and
  warns when either is unsafe.
- `./validation/generators/synformer/fixed_reward.py` — reward providers for the three targets. Holds
  three of the four fixes.
- `./glue/oracles/docking_server.py` — persistent docking server and its client.
- `/scratch/markymoo/rgfn_runs/forktest.py` — minimal test with SynFormer removed entirely: spawn a
  worker before the sEH reward model is loaded, then after, and ask each to use the GPU.

**Datasets**

- `./external/synformer/data/trained_weights/sf_ed_default.ckpt` — SynFormer's published checkpoint
  (2.7 GB). Stores `chem.fpindex` and `chem.rxn_matrix` as paths RELATIVE to the clone, which is why
  every worker must run with the clone as its working directory.
- `./external/synformer/data/processed/comp_2048/fpindex.pkl` — 4 GB building-block index, loaded per
  worker. Its size is why keeping workers alive across generations looked worth doing.
- `./external/synformer/data/chembl_filtered_1k.txt` — the authors' bundled starting population.

**Results**

- `/scratch/markymoo/rgfn_runs/smoke/sf_forkclean/` — DRD2 verification: five worker teardowns, full
  budget reached, recipes written.
- `/scratch/markymoo/rgfn_runs/smoke/sf_clpp_v3/` — ClpP verification through the production
  submission path, with the docking server live.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/synformer_drd2/seed43/` — the single cell that
  survived the original nine, 2,000 candidates all carrying recipes.

**Job Logs**

Root: `/scratch/markymoo/rgfn_runs`

- `ch_sf_{seh,drd2,clpp}_{42,43,44}-747{15..23}.{out,err}` — the original nine. SLURM records
  `Detected 1-3 oom_kill events` on each.
- `sf_control-74992.out`, `sf_ctl_routes-75001.out`, `sf_ctl_torch-75029.out`,
  `sf_ctl_long-74999.out` — the four control runs.
- `forktest-75080.out` — the test that settled sEH.
- `sf_clpp_vfy-75089.out` — where `_cuda_probe` caught the fourth hazard before any worker died.
- `sf_forkclean-75072.out`, `sf_clpp_v3-75094.out` — the two verifications.

## Relevant Versions

```
efd5586  Record what is actually verified, what is not, and one retraction
f35f82b  Asking whether a GPU is available opened the descriptors that broke the fork
b430f4c  sEH cannot be made fork-safe in-process; three guards each fixed a symptom and the fork still failed
39f37ef  A module-scope import opened 6 nvidia fds in the parent, and that is what broke every pool rebuild
1662762  Upstream's own GA crashes stochastically, and the base loop is otherwise validated
```

## Relevant Resources

**Sources**

- Gao, Luo, Coley (2025), *Generative artificial intelligence for navigating synthesizable chemical
  space*, PNAS. doi:10.1073/pnas.2415665122. Cite key `[gao2025synformer]`.
  Upstream code: https://github.com/wenhao-gao/synformer

**Packages**

- `synformer` (clone `bef02bc`), used by `validation/generators/synformer/run_synformer_fixed.py`.
  Its `sampler/analog/parallel.py::WorkerPool` is the process pool at issue; its
  `experiments/graphga_sf_opt.py` is the loop the control imports.
- `torch` 2.1.0+cu118 in the `synformer` env — the version whose `torch.cuda.device_count` is
  lru-cached, which is hazard 3.
- `gflownet.models.bengio2021flow`, used by `SEHFrozenReward` — the import that is hazard 1.

## Method

1. Established the authors' configuration as a control, changing nothing:
   `sbatch experiments/synformer_baseline/submit_upstream_control.sh`
   Recorded per-generation wall-clock and the resident memory of the coordinating process and each
   worker.
2. Enabled one divergence at a time on the same control:
   `TORCH_IN_PARENT=1 ...`, `ROUTES=1 ...`, `PATCHED_REPRODUCE=1 GENERATIONS=16 ...`
3. Tested the premise for sEH with SynFormer removed:
   `python /scratch/markymoo/rgfn_runs/forktest.py` (job 75080).
4. Instrumented every process-spawn site with `_cuda_probe`, reporting nvidia file-descriptor count
   and thread count. Deliberately does not call `torch.cuda.is_available()` or `device_count()`.
5. Verified DRD2 with per-generation worker teardown:
   `CFG=<recycle_workers_every_gens:1> TARGET=drd2 SEED=42 BUDGET=200 ... sbatch -p debug ...`
6. Verified ClpP through the production path, docking server included:
   `CELLS="synformer:clpp:42" N_ITERS_OVERRIDE=400 N_SAMPLES_OVERRIDE=30 OUT_ROOT=<smoke dir> sbatch -p debug experiments/fixed_reward/scale5k/submit_baseline_chain.sh`

## Results

**The four hazards.** Each prevents a worker from starting, with a different symptom:

| # | cause | worker symptom | affected |
|---|---|---|---|
| 1 | `from gflownet.models import bengio2021flow` at module scope opens 6 `/dev/nvidia*` descriptors at import | dies, `CUDA error: initialization error` | all three targets |
| 2 | loading the sEH model takes the coordinating process from 64 to 128-191 threads | hangs at exactly 2.9 GiB, `futex_do_wait`, indefinitely | sEH |
| 3 | `torch.cuda.device_count` lru-cached as 0 while the GPU was hidden | dies, `No CUDA GPUs are available` | sEH |
| 4 | `torch.cuda.is_available()` inside `_free_gpu_cache`, reached from `_dock` every docking batch | dies, `CUDA error: initialization error` | ClpP only |

Descriptor counts, same process, measured directly:

| step | nvidia fds |
|---|---|
| after `import torch` | 0 |
| `torch.cuda.is_initialized()` -> False | 0 |
| `torch.cuda.is_available()` -> True | **6** |
| `from gflownet.models import bengio2021flow` | **6** |

`torch.cuda.is_initialized()` reports False throughout, so the library's own readiness flag does not
reveal the condition. The file-descriptor count does.

**The authors' cadence bounds memory completely** (control 75001, four generations plus the initial):

| generation | molecules | worker memory after teardown |
|---|---|---|
| 0 | 100 | 0.0 GiB |
| 1 | 200 | 0.1 GiB |
| 2 | 196 | 0.1 GiB |
| 3 | 197 | 0.1 GiB |
| 4 | 200 | 0.1 GiB |

Recipes survive that cadence: **893 of 893** projected rows carried a route (100.0%). Keeping the
pool alive across generations is our divergence, and is what allowed ~260 GiB to accumulate.

**Upstream's own loop crashes stochastically** (control 74999, failed at 51 min):
`IndexError: tuple index out of range` from their `reproduce`, which catches only `ValueError` while
their `crossover_non_ring` does `rxn.RunReactants((fa, fb))[0]` (`experiments/crossover.py:146`) and
raises `IndexError` when a reaction yields no product. Our adapter already widens that catch; this is
independent evidence the widening is required.

**sEH is structurally blocked** (job 75080, SynFormer removed):

| test | result |
|---|---|
| spawn a worker BEFORE the sEH model loads | OK, `device_count=1` |
| coordinating process after model load | fds=0, threads=33 |
| spawn a worker AFTER the sEH model loads | **FAIL, `No CUDA GPUs are available`** |

The coordinating process measured clean on both metrics and the spawn failed anyway, so descriptor
and thread counts are necessary but not sufficient. Three successive guards each fixed their own
measured symptom and the spawn still failed; all three were reverted rather than left in place.

**Verification, DRD2** (job 75072): five worker teardowns, each followed by a working generation;
budget reached (200/200); 40 recipes written; candidates ingested. Reward mean rose 0.247 -> 0.636.

**Verification, ClpP through the production path** (job 75094):

| generation | scored | worker memory | fds at spawn |
|---|---|---|---|
| 1 | 233/400 | 40.1 GiB | 0 |
| 2 | 353/400 | 22.0 GiB | 0 |

Memory falls across the teardown rather than accumulating. Best score 11.6 on the `clip(-vina)`
scale, i.e. raw Vina about -11.6 kcal/mol, well past the -8.0 gate calibrated in entry [045].

**Not verified.** Both verifications used 120-400 molecule budgets against a production budget of
10,000, and the original failure surfaced only at hour ten. These results support "safe to launch",
not "will complete". The three sEH cells remain blocked: `scripts/score_batch.py` registers only
docking oracles, so scoring the surrogate in a separate process needs an entry point that does not
yet exist.
