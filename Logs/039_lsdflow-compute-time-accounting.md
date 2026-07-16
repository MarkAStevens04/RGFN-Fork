# SCENT sEH — measured compute-time accounting for hub-batching vs best-candidate
**Date:** 2026-07-15, ~5pm

## Question

When we build a diverse hit library by hub-batching instead of picking the best sampled
candidates, **how much longer does the computer actually have to work — and where, exactly, does
that time go?**

## Context & Summary

The LSD-Flow hub-batching campaign (entries `029`/`033`/`037`/`038`) compares two ways to assemble a
diverse library of hits: **best-candidate** (keep the top-reward molecules the generator already
sampled) and **hub-batching** (build one shared scaffold, then enumerate and score many diversifying
reactions on it). We already account for two costs carefully: **reactions per hit** (the bench
synthesis cost) and the **number of reward-generator calls** (how many enumerated children we score).
But a call *count* is not *time*: one call can be a fast surrogate proxy (~milliseconds) or a docking
oracle (~a second), and hub-batching makes tens-to-hundreds of thousands of them while best-candidate
reuses scores it already had. So the count alone can't tell us how much extra wall-clock hub-batching
really costs.

This experiment adds a third accounting axis: **measured compute time, tracked live as the pipeline
runs and broken down by component** — enumeration (building a hub's children), reward generation
(scoring them), flow-extraction (the policy probabilities for hub ranking), and mode selection (the
diversity filter). We instrument the actual GPU enumeration worker to record real per-hub wall-clock
for each component, re-run the sEH SCENT enumeration with the timers on, and have the four campaign
drivers attribute those measured times over the exact hubs each strategy walks — reporting, for every
operating point, how much longer the computer works to do hub-batching versus best-candidate and which
component that time is spent on.

## Answer

Hub-batching makes the computer work **~150× longer** than best-candidate to produce the same 300-mode
library — and **measuring** (not modelling) where that time goes overturns what the call-count
suggested. At the default diversity cutoff, hub-batching costs **~170 s** of compute vs best-candidate's
**~1 s**; but of that 170 s, the reward generator (the thing we count so carefully) is only **~5%** —
the real cost is **enumeration**, the RDKit work of *building* each scaffold's children (**~64%**),
plus flow-extraction (~14%) and one-time model setup (~16%). So for a fast surrogate reward, the
reward-gen *call count* badly under-represents compute; the enumeration to generate those children is
~12× more expensive than scoring them. Best-candidate's marginal compute is essentially just the O(n²)
diversity filter. Net reading for the paper: **hub-batching trades compute for bench synthesis** — it
spends ~169 extra seconds of computer time to save ~113 bench reactions (929→816) at cutoff 0.5. Two
strong levers surfaced, both measured: the **diversity cutoff** (strict 0.30 costs 13,485 s vs 169 s at
0.50 — stricter diversity forces enumerating almost every hub) and **pre-select-K** (K=200 cuts compute
6.1× vs free-frag, 7,472 s→1,219 s, tracking the call reduction). (For an expensive docking reward the
reward-gen term would dominate instead — the same instrumentation captures it, no code change.)

## Relevance to our Publication

The hub-batching story (targeted at **Digital Discovery** / **J. Cheminformatics**) sells "build once,
diversify late" as a *practical* saving. A reviewer will immediately ask what that saving costs in
compute — hub-batching's up-front enumeration and scoring of a scaffold's whole child set is real work
best-candidate never does. This entry answers that head-on with **measured, not modeled**, per-component
wall-clock: it makes the reactions-per-hit saving (`033`/`037`) honest by putting the compute price
next to it, and the component breakdown shows reviewers exactly where the pipeline spends its time.

## Next Experiments

**Refining for publication**
- **Time a docking target** (6TD3 / ClpP) the same way — the sEH proxy makes reward-gen look
  negligible; with a real oracle the reward-gen term should dominate, and the head-to-head flips. This
  is the sentence a reviewer will want ("what if the reward is expensive?"), and the instrumentation
  already handles it (no code change — just re-run the enumerate worker against the docking oracle).
- **Per-depth / per-hub compute** breakdown: enumeration time scales with the hub's fan-out; show
  which hubs are compute-expensive (the biggest-fanout scaffolds) so the cost is attributable to
  chemistry, not a black box.
- **Enumeration is the bottleneck, not scoring** — profile whether the RDKit enumeration is
  optimizable (it is ~64% of hub-batching's compute here); a faster enumerator would move the Pareto.

**Next steps in project**
- Wire the compute-time accountant into the AL loop's `LSDFlowAcquisition` (proposal §4a) so a live
  multi-round run reports the same per-component breakdown alongside the oracle-efficiency curve —
  making "compute per oracle call" a first-class, measured axis in the end-to-end system.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Instrumentation (ours — new/edited this entry):**
- `./validation/lsdflow/adapters/workers/scent_worker.py` — GPU enumeration worker; per-hub wall-clock
  timers (enumeration / reward-gen / flow-extract, stdlib-only, CUDA-synchronized) + one-time
  `setup_s` → `enum_timings.json`; `sample` mode → `sample_timings.json`.
- `./validation/lsdflow/metrics/cost/compute_time.py` — **new**; the compute-time accountant
  (`EnumTimings` load/merge keyed by unique `hub_input`, `account_strategy` attribution over the walk +
  live mode-selection, `head_to_head`).
- `./glue/samplers/lsdflow/campaign.py` — `EnumeratedHub.hub_input` + `CampaignResult.walked_hub_ids`
  (the unique-hub-identity walk record; stereoisomers share `hub_key` but are distinct here).
- `./experiments/lsd_hubs/campaign/run_campaign.py` — shared helpers (`run_timed`,
  `load_enum_timings`, `load_hub_pick_s`, `compute_time_section`, `plot_compute_time`) + `compute_time.{csv,png}`.
- `./experiments/lsd_hubs/campaign/{sweep_campaign,preselect_sweep,batch_size_distribution}.py` — surface
  the breakdown (`compute_time_by_cutoff.csv`, `preselect_compute.png`, per-K `ct_*` columns).
- `./experiments/lsd_hubs/campaign/pick_hubs.py` — Stage-2 timing → `pick_hubs_timing.json`.
- `./experiments/lsd_hubs/campaign/submit_scent_seh_enum_timed.sh` — **new**; one timed hub slice on `debug`.
- `./experiments/lsd_hubs/campaign/merge_enum_timings.py` — **new**; union the slices + integrity check.

**Results (committed — small artifacts, regenerated with measured timings):**
- `./experiments/lsd_hubs/campaign/results/scent_seh_1kx200/` — `compute_time.{csv,png}` (run_campaign
  per-component stacked bar) + `compute_time_by_cutoff.{csv,png}` (sweep) + `compute_time` blocks in
  `summary.json` / `sweep_summary.json`.
- `./experiments/lsd_hubs/campaign/results/scent_seh_1kx200_preselect/` — `preselect_compute.png` +
  `ct_*`/`compute_total_s` columns in `preselect.csv`.
- `./experiments/lsd_hubs/campaign/results/scent_seh_batch_dist/` — `ct_*`/`compute_total_s` per-K in
  `batch_stats.csv`.

**Inputs (on `$SCRATCH`, git-ignored):**
- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/` — SCENT sEH analysis DAG (candidate pool).
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/` — canonical 200-hub enumeration
  (828,448 children); the timed re-run reproduces its per-hub work. Now also holds the merged
  `enum_timings.json` + `pick_hubs_timing.json`.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_timed/` — the 6 timed slices + `chain.log`.
- SCENT recipe checkpoint `.../scent_seh/2026-07-10_17-28-06/` (job 70180).

**Job logs:** `/scratch/markymoo/rgfn_runs/enum_timed_seh-707{02,00,99,98,97,96}.out` (6 slices) +
`enum_timed_smoke-70695.out` (instrumentation smoke).

## Relevant Versions

Branch `Hub-Analysis`. New/edited files **not yet committed.** `[TODO — add commit hash after pushing.]`

## Relevant Resources

**Sources** — entries `038`/`037`/`033`/`029` (the campaign this instruments), `036` (measured
docking batch-size timing), `012`/`011` (per-phase / substep timing precedent). `[gainski2025scent]`,
`[bengio2021gflownet]`.

**Packages** — `rgfn` + `scent` conda envs (`~/bin/rgfn-smoke-env.sh`); existing timing infra
`glue/active_learning/timing.py` (`PhaseTimer`/`DockAccountant`), `glue/oracles/step_timing.py`
(`OracleStepTimer`); RDKit; matplotlib.

## Method

1. **Instrument the worker** (`scent_worker.py`, both modes): `time.perf_counter` around enumeration
   (RDKit child construction), the reward-generator call (`reward.compute_reward_output`), and
   flow-extraction (`assign_log_probs` → P_F/P_B), with `torch.cuda.synchronize()` at each boundary so
   async GPU work is charged to the right component. Per-hub timings + one-time `setup_s` →
   `enum_timings.json`; Stage-1 sampling vs flow-extract → `sample_timings.json`.
2. **Record the walk**: `HubBatchingStrategy` stamps `CampaignResult.walked_hub_keys` (hubs iterated,
   in order) so measured per-hub times attribute over exactly the hubs a strategy scored (verified:
   attributed `n_children_scored` == `total_reward_gen_calls`).
3. **Accountant** (`validation/lsdflow/metrics/cost/compute_time.py`): `EnumTimings` loads/merges the
   worker timings; `account_strategy` attributes them over the walk + adds the driver's live-measured
   mode-selection time; `head_to_head` gives the extra compute of hub-batching over best-candidate.
4. **Timed re-run** on Balam `debug` (2 h limit, QOS = 1 job at a time): the strictest operating point
   walks 198/200 hubs, so the full 200-hub enumeration is re-run with timers on, split into **6
   balanced hub slices** (~145k children each) run **sequentially**
   (`submit_scent_seh_enum_timed.sh`), then unioned by `merge_enum_timings.py` into one
   `enum_timings.json` beside the canonical `enum_children.json` (integrity: per-hub child counts vs
   the cached `enum_per_hub.json`). `pick_hubs.py` reproduces the exact 200-hub set (hub-pick 0.16 s).
5. **Regenerate** the four drivers (`run_campaign` / `sweep_campaign` / `preselect_sweep` /
   `batch_size_distribution`) over the canonical cache + merged timings → per-component breakdown +
   hub-vs-best-candidate head-to-head in each `results/<tag>/`.

## Results

**Timed run.** Full 200-hub sEH enumeration re-run with timers on, 6 sequential `debug` slices (jobs
70696–70702, ~45–60 min each), merged → `enum_timings.json` (200/200 hubs, 828,448 children, 0 per-hub
mismatches). Whole-enumeration totals: **setup 26.7 s, enumeration 15,612 s, reward-gen 960 s,
flow-extract 2,998 s** (i.e. across ALL 200 hubs; a strategy pays only the hubs it walks). A
stereoisomer-collision on the stereo-stripped `hub_key` (3 pairs, e.g. `[C@H]`/`[C@@H]` at ranks
25/104) was fixed by attributing on the unique `hub_input` — both isomers are one scaffold for the
reactions model but were separately enumerated for compute, so both are counted.

**Head-to-head (naive hub-batching, cutoff 0.5, 300 modes) — `run_campaign`, `scent_seh_1kx200`:**

| component | hub-batching (s) | best-candidate (s) |
|---|---|---|
| setup (load + freeze) | 26.7 | 0 |
| hub pick | 0.16 | 0 |
| **enumeration** | **108.8** | 0 |
| reward-gen | 9.2 | 0 |
| flow-extract | 23.5 | 0 |
| mode-selection | 1.4 | 1.2 |
| **total** | **169.7** | **1.2** |

Hub-batching costs **+168.6 s (147×)** more compute, to save **113 bench reactions** (929→816, the
Logs/033 reactions/mode edge). Enumeration is **64%** of hub-batching's compute; reward-gen only **5%**
— the walk scored 7,086 children, and *building* them cost ~12× more than *scoring* them.

**Compute vs diversity cutoff (`sweep_campaign`, hub-batching):** 0.30 → **13,485 s** (walks ~126–198
hubs), 0.45 → 464 s, 0.50 → 169 s, 0.60 → 74 s, 0.90 → 73 s. Strict diversity forces enumerating far
more hubs → compute explodes; best-candidate stays ≈0.2–43 s (its O(n²) filter, which also grows at
strict cutoffs).

**Measured-compute Pareto (`preselect_sweep`, free-frag + pre-select-K):** the reward-gen-call Pareto
(Logs/037) now has a wall-clock twin — pre-select-K cuts real compute in step with calls:

| K | reward-gen calls | measured compute (s) | of which enumeration (s) |
|---|---|---|---|
| 0 (free-frag) | 315,539 | 7,472 | 5,941 |
| 20 | 241,158 | 5,685 | 4,524 |
| 50 | 158,660 | 3,732 | 2,951 |
| 100 | 85,165 | 2,005 | 1,579 |
| 200 | 52,865 | 1,219 | 945 |

K=200 vs free-frag = **6.1× less compute** (and enumeration dominates at every K). `batch_size_distribution`
carries the same `ct_*`/`compute_total_s` columns per K.

**Integrity:** attributed `n_children_scored` == `total_reward_gen_calls` at every checked point
(cutoff 0.30 walk = 571,946 children, 0 missing); no `missing_hub_keys` anywhere.
