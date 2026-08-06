# GPU docking on a second cluster — what a newer GPU actually buys, and what really sets the wall-clock

**Date:** 2026-08-06, ~1pm

## Question

Would running our expensive molecular-docking step on the newer, faster graphics cards of a second
computer cluster get the remaining docking work done meaningfully sooner?

## Context & Summary

The one big piece of the hub-batching study still unfinished (entry `050`) is the two targets whose
molecules must be scored by real docking simulation rather than a fast approximate model. That work
is large: every scaffold we pick has to have *all* of its one-step chemical descendants docked, which
is roughly two million docking runs across the study — the single most expensive thing left on the
project's critical path.

Entry `036` measured how fast that docking runs on the graphics cards of our usual cluster (Balam) and
found two things: hand the docking program 200 molecules at a time rather than 25 and it goes 3.3×
faster, but running *two* docking programs side by side on one card buys nothing. That second result
came with a caveat we could not resolve there — two programs together needed 39.9 GB of the card's
40 GB, so we could not tell whether the card was genuinely working flat out or was simply out of
memory. The distinction matters a lot: if it was a memory limit, then a bigger card would let us
double throughput for free.

We now have access to a second cluster (Trillium) whose cards are both newer and twice the memory,
and which has roughly six times as many of them. This entry re-runs entry `036`'s exact benchmark
there, repeats one real scaffold's worth of the actual production work on both clusters for a
like-for-like comparison, and — because the honest bottleneck turned out not to be the hardware at
all — measures how many cards we can actually get hold of and how long we wait for them.

## Answer

**The newer card is only about 1.3× faster, and the reason two docking programs gain nothing is now
settled: it is not a memory limit.** On the bigger card two programs run with 32 GB to spare and
still deliver no extra molecules per second. So docking throughput cannot be bought with better
hardware or by packing more work onto each card — **only by using more cards.** That reframes the
remaining work as a scheduling problem rather than a hardware one, and on that axis the second
cluster is far less of a win than its size suggests: the typical share a project like ours actually
receives there is the same handful of cards we already get on Balam, just far less predictably.

Two things we did not go looking for turned out to matter more than the speed measurement. First,
**half the docking work we thought was ready cannot run at all** — two of the four molecule
generators have no docking support in their evaluation code, so the study is four cells, not eight.
Second, the provisional quality bar for the second docking target looks **well calibrated**, in
contrast to the first target's bar, which is so permissive as to be nearly meaningless.

## Relevance to our Publication

Two of the project's cost axes are *measured* wall-clock, not modelled (the standing directive behind
entry `039`), so the per-molecule docking rates here are the numbers the docking cells' compute
column will be built from — and they are now measured on both machines, which lets us report a
hardware-independent cost rather than one entangled with whichever cluster happened to be free.

The concurrency result is a small, self-contained methods contribution of the kind Digital Discovery
and JCIM reviewers respond well to: it closes a question our earlier data was structurally unable to
answer, and it tells anyone reproducing or extending this work that the docking oracle scales with
card *count* and not card *quality*. The generator-coverage finding is more consequential for the
paper's shape — it means the docking half of the four-generator matrix can only ever be a
two-generator comparison unless two more evaluation paths are written.

## Next Experiments

**Refining for publication**
- Finish the four runnable docking cells and report, for each, what fraction of enumerated molecules
  clear the quality bar — a bar that admits almost everything makes a cost comparison look good for
  reasons that have nothing to do with the method.
- Re-examine the first docking target's quality bar. It currently admits 88% of candidates, which is
  close to no filter at all; entry `045` calibrated it against known binders, so the recalibration
  should be argued from that evidence rather than chosen for convenience.
- Decide whether the two unsupported generators get docking evaluation paths. Without them the
  docking comparison is two generators wide while the cheap-scoring comparison is four.

**Next steps in project**
- Treat card count, not card speed, as the lever for any future docking-heavy work — including the
  active-learning loop, where the same oracle is queried every round.
- If the docking cells become a bottleneck again, the remaining efficiency ideas are on the
  computer's side of the work rather than the graphics card's (entry `036` reached the same
  conclusion from the other direction).

---

# Re-creation

## Relevant Files

Root: `./` (repo root `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`). `/home` and `/scratch` are
**shared between Balam and Trillium**, so one checkout, one set of conda envs, and one set of
checkpoints serve both clusters — no rebuild, no data copy.

**Scripts**
- `./experiments/fixed_reward/docking_benchmark/bench_batch_concurrency.py` — entry `036`'s
  benchmark, **run unmodified** so the two clusters are compared by the same code on the same
  molecule pool. Part A sweeps the docking batch size end-to-end through the live oracle; Part B
  compares one raw QuickVina2-GPU process against two concurrent ones on a single card.
- `./experiments/fixed_reward/docking_benchmark/submit_batch_concurrency_trillium.sh` — **new here.**
  Trillium launcher for the above. Exactly three deltas from the Balam version, all documented in its
  header: no `module load cuda/11.8.0` (absent on Trillium — `~/bin/rgfn-smoke-env.sh` supplies the
  CUDA libs instead), no `--exclude=balam008`, and partition `compute` rather than `debug` (Trillium's
  `debug` QOS allows one job at a time).
- `./experiments/lsd_hubs/matrix16/submit_docking_cell.sh` — the production docking-enumeration
  launcher, **run unmodified on Trillium**. It needed no port because it already sources the
  cluster-agnostic smoke-env helper rather than loading a CUDA module, carries no `--exclude`, and
  gates node health by probing the docking server rather than by an excluded-node list.
- `./experiments/lsd_hubs/matrix16/smoke_cell.sh` — the substance-asserting end-to-end smoke, used to
  validate the 6TD3 docking path before committing GPU-hours to it.
- `./scripts/preflight_dock.py` — per-oracle "can this node actually produce a pose" gate; used
  directly on the Trillium login node for the first correctness check of both oracles.

**Models** — the entry-`030` matrix checkpoints under
`/scratch/markymoo/rgfn_runs/experiments/fixed_reward/<cell>/`, reached unchanged from Trillium. The
SCENT cells additionally need their `guidance_models.pt` sidecar (entry `024`).

**Datasets**
- `./experiments/active_learning/6td3/seed_6td3.csv` — the 408-molecule pool; the benchmark docks the
  same 200 of them at every batch size on both clusters, so only the process count differs.
- `./data/targets/ClpP.pdbqt` — receptor for the single-target oracle (human ClpP 7UVU, entry `045`).
- `./experiments/oracle_validation/docking_6td3/6TD3_tier{1,2}.pdbqt`, `crystal_RC8.pdb` — the
  two-tier receptors and autobox ligand for the differential oracle.

**Results**
- `/scratch/markymoo/rgfn_runs/docking_benchmark/trillium_721183/benchmark_results.json` — per-config
  seconds/molecule, molecules/second, pose counts, and sampled card utilisation/memory. Directly
  comparable to `experiments/fixed_reward/docking_benchmark/results/70623/` (the Balam run).
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16_trillium_calib/{scent,rxnflow}_clpp/enum/slice1of999/`
  — the two production-path calibrations, deliberately written to an **isolated** scratch tag so they
  can never be mistaken for a production slice by the merge step's `slice*of*` glob.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16_smoke/scent_6td3/` — the 6TD3 smoke artifacts.

**Job Logs** — `/scratch/markymoo/rgfn_runs/trig_{dock_bench,calib_scent_clpp,calib_rxnflow_clpp}-*.{out,err}`
(Trillium jobs 721183/721184/721189), `smoke_scent_6td3-721320.out`, and for the Balam side of the
like-for-like comparison `m16_dock_scent_clpp_calib-72283.out` (job 72283).

## Relevant Versions

Branch `Hub-Analysis`. Built on `27a1123` ("manifest: a docking cell is not 'ready' if its worker
cannot dock" — the fix for the coverage finding below, authored by the agent on Balam in this same
shared checkout).

**Not yet committed:** `Logs/057_trillium-h100-docking-throughput-and-queue.md`,
`experiments/fixed_reward/docking_benchmark/submit_batch_concurrency_trillium.sh`, and this entry's
row in `docs/RESEARCH_CONTEXT.md`. [TODO — add commit hash after pushing.]

⚠️ Two agents share this one working tree (`/home` is shared). Commit these three paths **by name** —
`git add -A` would sweep in the other agent's in-progress SPARROW edits.

## Relevant Resources

**Sources**
- Tang et al., *Vina-GPU 2.1* (bioRxiv 2023.11.04.565429) — QuickVina2-GPU-2.1; recommends
  `thread` ≈ 5000, capped below 10000. We run `thread=8000`, unchanged from entry `036`.
- Prior entries: `036` (the Balam batch/concurrency measurement this replicates, and the memory-bound
  caveat it left open), `050` (the docking cells as the matrix's last open item), `045` (the ClpP
  bar's calibration against known binders), `039` (compute time must be measured, not modelled),
  `013`/`014` (nodes that pass a health probe yet produce no poses — why a live probe gates each run),
  `012` (`$HOME` is read-only on compute nodes).
- [SciNet Slurm documentation](https://docs.scinet.utoronto.ca/index.php/Slurm) — confirms `def-*`
  accounts are Rapid Access Service allocations and that wait depends on allocation size and recent
  usage. It does not document the GPU cluster's priority weights or per-user limits, so those were
  read directly from the scheduler (`scontrol show config`, `sacctmgr`, `sshare`). The
  [Alliance Trillium page](https://docs.alliancecan.ca/wiki/Trillium_Quickstart) is Cloudflare-gated.

**Packages**
- QuickVina2-GPU-2.1 (`$SCRATCH/vina_gpu/Vina-GPU-2.1/QuickVina2-GPU-2.1/QuickVina2-GPU-2-1`) — the
  docking engine; the Balam build runs on Trillium with no rebuild.
- gnina v1.3.2 (`$GNINA` → `/scratch/markymoo/gnina/run_gnina.sh`) — CNN pose selection for the
  two-tier differential (kept as pose picker on the evidence of entry `008`).
- Oracles: `glue/oracles/docking_seh_oracle.py` (`DockingClpPOracle`),
  `glue/oracles/docking_gpu_differential_oracle.py` (`Docking6TD3GpuOracle`), served over
  `glue/oracles/docking_server.py`.

## Method

1. **Correctness before speed**, on the Trillium login node (which has H100s), via
   `source ~/bin/rgfn-smoke-env.sh`:
   `$SCRATCH/vina_gpu/opencl_healthcheck`, then
   `python scripts/preflight_dock.py --oracle docking_clpp` and `--oracle docking_6td3_gpu`.
2. **Benchmark replication** (Trillium job 721183, one H100, partition `compute`):
   `sbatch experiments/fixed_reward/docking_benchmark/submit_batch_concurrency_trillium.sh`, which
   runs `bench_batch_concurrency.py --targets clpp 6td3 --n 200 --batch-sizes 25 50 100 200
   --concurrency-n 96 --repeats 2`. Compared against Balam job 70623 (entry `036`).
3. **Like-for-like production calibration** — re-run the *same hub* Balam had already enumerated,
   using an isolated scratch tag so the production tree is untouched:
   ```bash
   MATRIX16_SCRATCH=/scratch/markymoo/rgfn_runs/lsdflow/matrix16_trillium_calib \
   HUBS_FILE=/scratch/markymoo/rgfn_runs/lsdflow/matrix16/<cell>/enum/hubs.csv \
   STAGE=enum ENUM_MAX=4000 sbatch -J tri_calib_<cell> -p compute --time=01:45:00 \
     --export=ALL,STAGE=enum,HUBS_FILE=...,MATRIX16_SCRATCH=...,ENUM_MAX=4000 \
     experiments/lsd_hubs/matrix16/submit_docking_cell.sh <gen> clpp 1 999
   ```
   Slice `1 999` selects hub row 0 deterministically (`i % 999 == 0`), so it reproduces exactly the
   hub Balam job 72283 measured. Jobs 721184 (`scent_clpp`) and 721189 (`rxnflow_clpp`).
4. **6TD3 path validation:** `bash experiments/lsd_hubs/matrix16/smoke_cell.sh scent 6td3 200 6`
   (job 721320) — writes to the isolated `matrix16_smoke` tag and asserts substance, not exit codes.
5. **Scheduling measurement** — no jobs submitted for this beyond the above:
   `scontrol show config`, `sacctmgr show qos`, `sshare`, `scontrol show partition|node|reservation`,
   plus a two-week cluster-wide `sacct -a -X -S 2026-07-23 -o JobID,Account,User,Submit,Start,End,
   Timelimit,Elapsed,State,AllocTRES --parsable2` reduced per-user into concurrent-GPU timelines.

## Results

### Docking throughput vs batch size — the same 200 molecules, both clusters

Balam rows are from entry `036` (job 70623, A100-40GB); Trillium rows are new (job 721183, H100-80GB).
`n_ok` was 199/200 in every configuration on both machines — the one failure is a molecule-prep
failure, not a hardware difference, so scores remain batch- and cluster-invariant.

| batch | procs | ClpP A100 | ClpP H100 | speed-up | 6TD3 A100 | 6TD3 H100 | speed-up |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 25 | 8 | 1.312 | 1.022 | 1.28× | 1.750 | 1.613 | 1.08× |
| 50 | 4 | 0.779 | 0.613 | 1.27× | 1.107 | 0.993 | 1.11× |
| 100 | 2 | 0.534 | 0.402 | 1.33× | 0.785 | 0.683 | 1.15× |
| **200** | **1** | **0.398** | **0.298** | **1.34×** | **0.632** | **0.528** | **1.20×** |

Seconds per molecule. Batch 200 remains the optimum on the newer card and the curve has the same
shape, so entry `036`'s `docking_batch_size=200` decision needs no revisiting. The differential
oracle gains less (1.20× vs 1.34×) — it spends part of each molecule in gnina's CNN rescoring, which
does not speed up as much as the search. Card utilisation stays low on both machines (H100: 47.1%
ClpP, 30.7% 6TD3 at batch 200).

### One versus two docking processes on one card — the question entry `036` could not answer

| target | cluster | 1 process | 2 concurrent | speed-up | memory used | verdict |
|---|---|---:|---:|---:|---:|---|
| ClpP | A100-40GB | 1.86 lig/s | 1.87 | 1.01× | ~39.9 / 40 GB | saturated **or** out of memory — indistinguishable |
| ClpP | **H100-80GB** | 2.43 lig/s | 2.54 | **1.05×** | **48.3 / 80 GB** | **saturated — 32 GB spare** |
| 6TD3 | A100-40GB | 1.65 lig/s | 1.76 | 1.07× | ~39.9 / 40 GB | same ambiguity |
| 6TD3 | **H100-80GB** | 2.10 lig/s | 2.34 | **1.11×** | 48.3 / 80 GB | **saturated — 32 GB spare** |

This is the entry's cleanest result. On the 40 GB card the no-gain finding was confounded: two
processes sat at 39.9 of 40 GB, so "the card is already busy" and "there is no room for a second
process" predict the same observation. On the 80 GB card two processes fit with **32 GB unused** and
still add nothing (utilisation rises only 36.7% → 41.7%). Entry `036`'s hypothesis — that the ceiling
is per-molecule work on the computer's side (launch overhead and pose file I/O) rather than graphics
compute — is therefore confirmed rather than assumed. **Consequence: docking throughput scales with
the number of cards, not their capability or how much work is packed onto each.**

### Same hub, both clusters — the production path, not a synthetic pool

Balam job 72283 and Trillium job 721184 enumerated the **identical** scaffold and got the **identical
1,860 descendants**, so this is a controlled comparison of the real workload.

| component | Balam A100 | Trillium H100 | speed-up |
|---|---:|---:|---:|
| enumeration | 51.7 s | 38.7 s | 1.34× |
| **docking (reward)** | **1706.6 s** | **1288.6 s** | **1.32×** |
| flow extraction | 8.1 s | 6.5 s | 1.25× |
| total (excl. setup) | 1766.4 s | 1333.8 s | **1.32×** |
| **seconds per molecule** | **0.9497** | **0.7171** | **1.32×** |
| docking server's own s/mol | 0.9261 | 0.6988 | 1.33× |

Docking is **96.6%** of this cell's cost. The production figure (1.32×) and the synthetic benchmark
(1.34×) agree to within 0.02×, which is the check that the benchmark measures the thing we care about.

A second calibration, `rxnflow_clpp` (job 721189, hub row 0, 1,360 descendants): **0.4185 s/molecule**
end-to-end, 0.3668 s/molecule of docking — so **SCENT costs ~1.7× per molecule what RxnFlow does**,
because its promoted-fragment molecules are larger and dock more slowly. RxnFlow's flow-extraction is
65 ms/molecule (16% of its cost) against 4 ms for SCENT.

### Only four of the eight docking cells can run

Found while planning the work, not by testing a hypothesis. `manifest.py --emit` reported
`STATUS=ready` for all eight docking cells, because it checks only that the checkpoint, sidecar and
candidates exist. But the per-generator evaluation workers tell a different story:

| worker | docking references | behaviour on a docking target |
|---|---:|---|
| `rxnflow_worker.py` | 9 | wired |
| `scent_worker.py` | 3 | wired |
| `rgfn_worker.py` | **0** | no docking path |
| `fraggfn_worker.py` | **0** | `SystemExit("reward '<x>' not wired. Docking targets (6td3/clpp) are deferred.")` |

The failure mode was expensive-looking and misleading: `submit_docking_cell.sh` gates on
`[ "$STATUS" = "ready" ]`, so an `rgfn`/`fraggfn` docking cell passed the gate, started the docking
server, paid ~35–45 s of oracle construction *plus* a live preflight dock, and only then died inside
the worker — reading like a cluster problem rather than missing code. Fixed in **`27a1123`**, which
derives a `docking_wired` flag from the worker source: docking now reports **4 ready / 4
worker-not-wired**, the surrogate cells stay 8/8, and the flag flips automatically if a generator is
wired later. **The docking half of the matrix is a two-generator comparison** (RxnFlow, SCENT) unless
two more workers are written — where the cheap-scoring half is four generators wide (entry `055`).

### The 6TD3 quality bar looks well calibrated; ClpP's does not

`scent_6td3` smoke (job 721320; 200 trajectories, 6 scaffolds, 240 enumerated descendants), which
asserts substance rather than exit status:

```
OK  oracle failure rate 0.0% < 20%          (a wedged card shows up here, not as bad chemistry)
OK  `reward` is RAW energy: 200/200 negative, mean -2.70
OK  two-column contract: log_reward == 4*ReLU(-raw) on 200/200 rows (implied beta 4)
OK  gate <-2 qualifies 126/200 = 63.0%
SMOKE PASSED
```

Three things this establishes. The 6TD3 docking path works end to end (first time for either
cluster). The two-column contract holds on every row with β recovered exactly — so the trap
`_docking.py` exists to prevent (gating the rectified reward column instead of the raw energy, which
silently qualifies nothing) is absent on this path. And the **provisional −2.0 bar admits 63%**,
which is a real filter — unlike ClpP's calibrated −8.0, which the Balam agent measured admitting
**88%** of enumerated descendants for SCENT, close to no filter at all. So the recalibration concern
belongs to ClpP, not 6TD3. Caveat: 240 descendants over 6 scaffolds validates plumbing and sign
conventions, not the 63% figure at production scale.

### What actually sets the wall-clock: getting cards, not card speed

Trillium's GPU side is a **separate scheduler** (`grillium`) from both its CPU side and Balam, and the
three are **not federated** — only `grillium` is visible from its login node, so jobs must be
submitted from each cluster's own login node and cross-submission is impossible.

**The pool is smaller than the headline.** 252 H100s configured, but 20 sit on permanently
administratively-held nodes (`trig0004/0010/0057/0061` plus four VMs, all carrying `MAINT,IGNORE_JOBS`
reservations with 365-day durations — dead nodes and driver testing). Effective pool ≈ **232**, running
~228 allocated: **saturated**. Reading unallocated GPUs from `AllocTRES` without also checking node
state overstates availability by 5× — a mistake worth naming, because it turns "the scheduler is
idling while I wait" into "the cluster is genuinely full."

**Priority is essentially all fairshare, and our account has almost none.**

| factor | weight | our contribution |
|---|---:|---|
| Partition | 10,000,000 | **0** — partition `compute` has `PriorityJobFactor=0` (only `debug` has 1, and its QOS allows one job at a time, 2 h max) |
| FairShare | 5,000,000 | ~245,000 (factor 0.049) |
| QOS | 5,000,000 | **0** — QOS `normal` has priority 0; the `bigprio`/`bump`/`reframe` QOSes carry real priority but are not granted to us |
| JobSize | 62,500 | ~479 |
| Age | 50,000 | grows slowly (`PriorityMaxAge` 28 d) |

Against a **median priority of 411,061 across ~860 pending jobs**, ours sits in the bottom quartile.
`def-naeilum` holds `RawShares=1`, as do **3,312 other default accounts**, while allocation-holding
accounts hold 10,000–48,000. Usage depresses it further and recovers slowly: holding two cards for 17
minutes moved `RawUsage` 312 → 2,111 and the fairshare factor 0.049 → 0.030, against a 7-day decay
half-life with no periodic reset.

**Two scheduler behaviours explain everything we observed.** `SchedulerParameters` contains
**`bf_max_job_user=3`** — the backfill scheduler tests at most three of one user's jobs per cycle — and
`sched/backfill` will start a low-priority job that fits a hole without delaying a higher-priority
reservation. Both were then confirmed live:

- Two jobs submitted at 12:32:37 started at **12:32:43 and 12:33:06**. A whole-node 4-GPU job
  (`rrg-fwood` 717543) had been **cancelled at 12:31:06**, 91 s earlier; four jobs from **three
  different users** consumed those four cards within 26 s. Pure backfill luck, not entitlement.
- A third job submitted 3 min later (12:36:16) waited until **13:08:02 — 32 min**.
- Later, eight slices submitted together: **exactly 3 started immediately** on three separate nodes,
  five queued. `bf_max_job_user=3`, observed directly.

**`squeue --start` is worst-case, not a forecast.** That third job was advertised as starting the
*next day* (`2026-08-07T11:26`, ~23 h out) and started in 32 min. The estimate is the backfill
reservation's pessimistic bound and must not be quoted as an expected wait.

**What default-allocation users actually achieve** (288 of them with GPU jobs, two weeks):

| | max concurrent GPUs |
|---|---:|
| p25 | 2 |
| **p50** | **4** |
| p75 | 12 |
| p90 | 21 |
| p95 | 29 |
| max | 72 |

62.5% reach ≥4 cards, 35.4% ≥8, 20.5% ≥16, 4.2% ≥32. Queue wait for jobs requesting 6–12 h: median
0.2 h, p75 6.7 h, **p90 68.6 h**. So the median experience is the **same four cards we already get
reliably on Balam**, with much heavier tails in both directions — and a practical corollary we acted
on: request the walltime the work needs, not the maximum, because short requests fit backfill holes
far more often (71% of ≤2 h requests start within the hour against 50% of >12 h requests).

### What the remaining work costs

Volume, projecting each generator's own measured fan-out onto its docking cells, restricted to the
four cells that can actually run:

| cell | children | s/child (Trillium) | Trillium GPU-h | Balam GPU-h |
|---|---:|---:|---:|---:|
| `rxnflow_clpp` | 197,128 | 0.418 | 22.9 | 30.2 |
| `rxnflow_6td3` | 197,128 | 0.690 | 37.8 | 45.3 |
| `scent_clpp` | 391,726 | 0.717 | 78.0 | 103.0 |
| `scent_6td3` | 391,726 | 1.252 | 136.2 | 163.5 |
| **total** | **1,177,708** | | **274.9** | **342.0** |

Plus the sample stage for the two 6TD3 cells (neither has one; both ClpP cells do). The 6TD3 figures
scale ClpP's docking component by the measured H100 ratio 0.528/0.298 = 1.772×. `scent_6td3` alone is
half the bill.

At four cards — the median share on either cluster — that is **2.9 days on Trillium against 3.6 on
Balam**; at eight cards, 1.4 against 1.8. **The saving is ~25% at equal card count, and everything
beyond that comes from card count, which on a default allocation is opportunistic rather than
something to plan around.** Both clusters are worth running simultaneously, which costs nothing to
arrange: `submit_docking_cell.sh` runs unmodified on both, hub slices are disjoint by construction
(`i % N == idx-1`), and `merge_docking_slices.sh` unions them and refuses to publish unless every
scaffold appears exactly once.

### Portability: no code changes were required

`submit_docking_cell.sh` ran unmodified on Trillium to completion twice, producing the full artifact
set (`enum_children.json`, `enum_per_hub.json`, `enum_timings.json`, `enumerated_records.csv`,
`meta.json`). Correcting three assumptions recorded before Trillium's compute nodes were reachable:
`--account` is **not** required; an untyped `--gpus-per-node=1` yields **1 card + 24 cores on a shared
node**, not a whole node (`DefCpuPerGPU=24`); and Trillium uses the **same partition names** as Balam
(`debug`/`compute`/`*_full_node`). The oracles also agree numerically across clusters — ClpP probes
returned −5.9/−7.9/−6.6 against Balam's −5.8/−7.9/−6.5, and 6TD3 returned −2.44/−1.19 against
reference values −2.20/−1.26, both within Vina's run-to-run spread.

### In flight at time of writing

Launched on Trillium under the cell split agreed with the Balam agent (each cluster takes one ClpP and
one 6TD3 cell, so no cell directory is ever shared): `rxnflow_clpp` enumeration as 8 slices at 8 h
each (jobs 721347–721354; 3 started immediately, 5 queued) and the `scent_6td3` sample stage (job
721346, ~4 h projected). Balam holds `scent_clpp` (12 slices, all confirmed `PENDING` with no progress
after ~2 h) and `rxnflow_6td3`. `scent_6td3`'s slice count is deliberately **not** yet chosen — it
will be sized from a hub-0 calibration once its sample produces `hubs.csv`, since fan-out estimates
for this pipeline have been wrong twice before (22% low for RxnFlow; 6 slices would have exceeded
SCENT's walltime).
