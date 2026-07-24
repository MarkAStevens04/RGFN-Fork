# Four systems × four generators — publication-scale re-run (uniform 5,000 iterations)

**Date:** 2026-07-13, ~11am (START stub — campaign in bring-up)

## Question

When every generator (RGFN, FragGFN, RxnFlow, SCENT) is trained for the same, full
publication-scale number of steps, how do they compare head-to-head across all four benchmark
systems (sEH, DRD2, 6TD3, ClpP)?

## Context & Summary

Our four-way generator comparison so far mixed two very different training budgets. The two
*surrogate-reward* systems — sEH and DRD2, where the reward is a fast pretrained model — were
already trained for the full ~5,000 steps used in the RGFN and SCENT papers. But the two
*docking* systems — 6TD3 (the validated glue differential) and ClpP — were capped at only
**400** steps (entry `021`), because docking every molecule in the loop is slow (~1 s/molecule)
and 5,000 steps would run for one to two weeks per model. That mismatch means the docking
results are not directly comparable to the surrogate results, nor to the numbers the papers
report. This campaign closes that gap: re-run the **entire 4×4 matrix at a uniform 5,000
steps**. (The project lead initially also asked to raise the maximum number of reactions by one;
a GPU smoke measured that at ~3× the per-iter compute while amplifying RGFN's size-drift toward
larger, less-drug-like molecules, so the reaction cap was **kept at the original 4** for this
campaign — the "+1 reaction" is deferred to a possible later sensitivity study. See Results.)

Because the docking runs take ~9 days each at 5,000 steps and Balam compute jobs die at a 3-day
wall, the campaign also needs **automatic re-queueing** (checkpoint → resubmit → resume) for
every generator, and — to keep the cross-environment baselines from wasting ~2 days each on
per-step subprocess startup — a **persistent docking server**. We instrument the whole thing
so we can see **where wall-clock goes** and **how busy the docking server actually is**: if the
server sits idle most of the time, additional random seeds can be run concurrently for error
bars at little extra wall-clock (needing only more GPUs, not more time).

## Answer

*(TODO — fill in END mode: the matched 5,000-step, +1-reaction four-way comparison across all
four systems, and whether the docking-server utilization justifies running 3 seeds.)*

## Relevance to our Publication

Reviewers (Digital Discovery / JCIM tier, per `RESEARCH_CONTEXT.md`) will ask whether the
generator comparison is fair. Two budgets in one table is an obvious apples-to-oranges target;
a uniform, publication-matched step count across every generator and system removes it. The
+1-reaction setting is a controlled sensitivity check on molecule-building depth. The
timing/utilization instrumentation also directly supports the "≥3 seeds with error bars"
requirement (Objective 4) by telling us the true marginal cost of a seed.

## Next Experiments

**Refining for publication**
- If server utilization is low, run **3 seeds** per cell for error bars (Objective 4).
- Post-hoc synthesizability (AiZynthFinder) + SA + diversity/novelty over every cell's
  candidate dataset, as in entries `018`/`020`/`021`.

**Next steps in project**
- Fold the auto-requeue + docking-server infrastructure back into the standard launch layer so
  any future long docking run inherits it.
- **Apply the exp-036 docking-batch-size fix to future docking runs** (NOT this live campaign):
  raise `docking_batch_size` ≥ the per-step sample count (e.g. 200) on the single-target oracle
  (`glue/oracles/docking_seh_oracle.py`, TODO in place) for a **3.3× faster** ClpP/sEH dock at
  identical scores. It was deliberately not applied here — a mid-campaign change would split a
  cell across two settings, and the finish line is gated by `rgfn_6td3` (a 6TD3 *differential*
  cell whose oracle already docks in one process), which the fix doesn't touch. See Logs/036.

---

# Re-creation

## Relevant Files

Root: `./` (repo root `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`).

**Configs (new campaign variants — 5,000 iters + max reactions 5; all `_5k`, verified by gin-parse/YAML-load 2026-07-13).** Each `include`s (RGFN) or copies (baselines) its 400-iter/4-reaction base and overrides only iters + reaction cap; FragGFN keeps `max_nodes=9`:
- RGFN (gin): `./configs/glue/fixed_reward_{6td3,clpp,seh_proxy_stdlib,drd2_stdlib}_5k.gin`
- RxnFlow (yaml): `./validation/configs/rxnflow_{6td3_docking,clpp_docking,seh,drd2}_fixed*_5k.yaml`
- FragGFN (yaml): `./validation/configs/fraggfn_{6td3_docking,clpp_docking,seh,drd2}_fixed_5k.yaml`
- SCENT (gin): `./validation/configs/scent_{6td3,clpp,seh,drd2}_fixed_5k.gin`

**Scripts / infrastructure:**
- `./scripts/fixed_reward.py` — RGFN single-shot entry point (already supports `--resume-from`).
- `./scripts/score_batch.py` — cross-env oracle bridge (per-step spawn today; the server replaces it).
- `./glue/oracles/docking_server.py` — **NEW** persistent docking server (construct oracle once,
  serve batches over AF_UNIX, log busy/idle utilization) + stdlib `DockingServerClient`.
  Protocol smoke-tested with the mock oracle (login node, no GPU). Socket must be a short
  node-local path (AF_UNIX ~108-byte cap).
- `./experiments/fixed_reward/scale5k/launch_chain.sh` — **NEW** 3-day auto-requeue as a
  login-node-submitted dependency **chain** (N links, each `afterany` the previous). Replaces
  the first (self-`sbatch`) design, which a smoke test proved Balam **forbids** ("job submission
  ... cannot be done from a compute job"). Each link resumes from the last checkpoint and no-ops
  once the completion marker exists.
- `./experiments/fixed_reward/scale5k/submit_rgfn.sh` — **NEW** RGFN per-link submit
  (`<6td3|clpp|seh|drd2> [seed]`): early-exit if complete, resume-or-fresh (+`*_cache` strip),
  OpenCL health gate for docking, stable per-(system,seed) run dir, `N_ITERS_OVERRIDE` for smokes.
- `./scripts/fixed_reward.py` — added `--gin-binding` (repeatable, overrides config; used for
  the smoke `Trainer.n_iterations` override).
- `./validation/generators/{fraggfn,rxnflow}/al_loop.py` + `run_{fraggfn,rxnflow}_fixed.py` —
  **NEW** checkpoint/resume for the 3-day chain: `save_checkpoint`/`load_checkpoint` (full state —
  model + both optimizers + both LR scheds + sampling_model + step; gflownet's `_save_state`
  omits these), periodic save in `_train_steps`, `--run-dir` (stable dir) + resume +
  remaining-steps in the runners. py_compiled in both envs; GPU resume smoke pending.
- `./validation/generators/{fraggfn,rxnflow}/fixed_reward.py` (`DockingBridgeReward`) +
  `./validation/generators/scent/docking_bridge_proxy.py` (`DockingBridgeProxy`) — **NEW**
  server-mode: dock over the persistent server when `RGFN_DOCK_SOCKET` is set (stdlib client
  path-imported, no glue import), else per-step spawn. `glue/oracles/docking_server.py` gained
  `client_from_env`. Cross-env round-trip validated (mock).
- `./validation/generators/scent/{fixed_reward.py,run_scent_fixed.py}` — **NEW** faithful SCENT
  resume: `--run-dir` + `Trainer.resume_path`, valid_every 1e8→250 (periodic ckpt), guidance
  sidecar saved per-checkpoint (`make_checkpoint` monkeypatch) + reloaded on resume
  (`_load_guidance_models`).
- `./experiments/fixed_reward/scale5k/submit_baseline.sh` — **NEW** per-link baseline submit
  (`<gen> <system> [seed]`): launches the docking server for 6td3/clpp cells (rgfn-env subshell,
  `/tmp` socket, OpenCL gate, trap-kill), runs the resume-capable baseline trainer with a stable
  `--run-dir`, early-exits on candidates.csv. Runners got `--n-train-steps`/`--n-samples`
  smoke overrides. bash -n + py_compile clean; baseline GPU smoke in flight (job 70369).

**Timing / utilization:**
- `./glue/active_learning/timing.py` — `PhaseTimer` (+`total_for`) and **NEW** `DockAccountant`
  (per-run docking-time accounting; unit-tested).
- `./glue/proxies/oracle_reward_proxy.py` — times each oracle call via `DockAccountant`.
- `./glue/fixed_reward/pipeline.py` — writes `dock_timing.json` (+`docking_share`) after training.

**Job Logs:** `/scratch/markymoo/rgfn_runs/*.out` *(TODO — job IDs, once smoke tests + launch run)*

## Relevant Versions

Branch `Hub-Analysis` at start. *(TODO — add commit hash after the campaign infra is committed.)*

## Relevant Resources

**Sources**
- `[koziarski2024rgfn]` — RGFN paper (Appendix B.2: 4,000 steps @ batch 100; max 4 reactions).
- `[gainski2025scent]` — SCENT (5,000 iters @ batch 64 for SMALL/MEDIUM; max 4 reactions).
- `[seo2024rxnflow]` — RxnFlow (reports budget in oracle calls; own model caps at 3 reaction steps).
- Prior entries `019`, `020`, `021` (fixed-reward pipeline + 400-iter docking matrix).

**Packages**
- QuickVina2-GPU + gnina (docking oracle), gin-config, torch, dgl.

## Method

*(TODO — the launch commands + auto-requeue mechanism, filled in as the campaign runs.)*

## Results

**RGFN smoke tests (debug partition, 2026-07-13) — path validated + cap-5 cost measured.**
- Full RGFN cap-5 6TD3 docking path end-to-end (job 70354): resumed from an iter-4 checkpoint
  (chain-resume), trained to iter 8, sampled, emitted 25 candidates + `dock_timing.json`
  (`464 mols, 1.28 s/mol, docking_share 52.7%` on this short run). Confirms resume + docking +
  emit + timing instrumentation.
- **Per-iter cost, cap-4 vs cap-5 (the +1 reaction):**

  | run | cap-4 (baseline) | cap-5 (the +1) | ratio |
  |---|---|---|---|
  | sEH surrogate (no docking) | ~48 s/it (job 70356) | ~150 s/it (smoke A) | ~3.1× |
  | 6TD3 differential docking | ~140–280 s/it (Logs/021) | ~360–430 s/it, rising (smoke C/70354) | ~2–3× |

  Cause: the extra reaction amplifies RGFN's known **size-drift** → larger molecule graphs →
  slower GFN forward + slower docking. Full-run (5000-iter) estimates: surrogate cap-4 ~2.8 d →
  cap-5 ~9–12 d; docking cap-4 ~9 d → cap-5 ~22–25 d per run. Parallelized matrix wall-clock:
  cap-4 ~1.5 wk vs cap-5 ~3.5 wk; cap-5 ≈ 3× GPU-hours and ~8 requeue links/docking-run vs ~3.
- At cap-5 **docking is ~50% of wall-clock** (vs the historical <2% at cap-4, Logs/014) → the
  persistent docking server matters more, and seed-sharing per docking GPU is limited (~2, not 3).

**Baseline orchestration validated on GPU (debug, 2026-07-13):** FragGFN 6TD3 via
`submit_baseline.sh` (job 70369) ran end-to-end — OpenCL gate → persistent docking server
launched → FragGFN bridge connected in **server-mode** → trained via docking-through-server →
emitted candidates + wrote its checkpoint → COMPLETE. The server logged **95.2% utilization**
over that tiny 6-iter run (388 mols, 0.98 s/mol; a full run will show the real GFN-vs-dock
balance). Resume/RxnFlow/SCENT smokes in progress.

**Launch layer:** `experiments/fixed_reward/scale5k/{launch_chain.sh,launch_matrix.sh}` — the
matrix fires as 16 auto-requeue dependency chains. Sized to the `normal`-QOS **60-submitted-job
cap**: DOCK_LINKS=5 (docking cells) + SURR_LINKS=2 (surrogate) = 56 jobs, with a pre-flight
headroom check.

**ALL GPU SMOKES PASSED (debug, 2026-07-13):** RGFN full path; FragGFN fresh+resume; RxnFlow
fresh; SCENT fresh+resume (guidance sidecar saved every checkpoint + reloaded on resume,
loaded=2/unmatched=0); launch_chain afterany wiring. Three bugs found+fixed via the smokes:
(1) gflownet `shutil.rmtree(run_dir/train)` on trainer construction wiped the baseline resume
checkpoint → moved fraggfn/rxnflow ckpt to sibling `run_dir/checkpoints/`; (2) optimizer state
loaded on CPU while model moved to GPU later → Adam device-mismatch → move model to device before
load; (3) a `@gin.configurable` decorator displaced off `DockingBridgeProxy` during the SCENT
server-mode edit → moved back onto the class.

**LAUNCHED 2026-07-13 ~18:20 — full 16-cell matrix, seed 42, jobs 70457-70512** (56 jobs, 16
dependency chains via `launch_matrix.sh 42`). Docking cells 5 links (~15 d), surrogate 2 links
(~6 d). Monitor: `squeue -u markymoo`.

**3-SEED EXTENSION — seeds 43,44 (+ seed-42 FragGFN re-run) launched 2026-07-23.** To get error
bars, two more trials across all 16 cells (cluster idle). Two config changes for the new seeds
(seed 42 for the other three generators is kept as-is):
- **ClpP `docking_batch_size` 25→200** (`glue/oracles/docking_seh_oracle.py` default): 3.3× faster
  ClpP docking at byte-identical, batch-invariant scores (exp 036 / Logs/036). Covers both the RGFN
  in-env oracle and the baseline docking server (neither overrides the default).
- **FragGFN `max_nodes` 9→6 for all four cells** (paper-faithful; Logs/046 found 9 → oversized MW →
  reward collapse, e.g. DRD2). The seed-42 FragGFN cells had finished at the wrong value 9, so they
  were archived to `fraggfn_<sys>_5k/seed42_maxnodes9/` and re-run at 6 — the FragGFN row is now
  seeds 42/43/44 all at 6. RGFN/RxnFlow/SCENT are unaffected (their max-reaction cap is separate).

Because 36 cells (2 seeds × 16 + 4 seed-42 FragGFN) exceed the 60-submit/30-run per-user caps
(shared with other jobs), they are driven by a cap-aware orchestrator
`experiments/fixed_reward/scale5k/orchestrate.sh` (idempotent; tags each cell
`c5_<gen>_<sys>_s<seed>` in its SLURM job name; launches idle cells as 2-link afterany mini-chains,
docking-first, up to the live submit budget; re-run to top up + launch waiters). crontab is
disallowed for this user, so it is re-run on check-ins; each cell's 2-link chain gives a ~6-day
autonomous buffer. First wave: 17 docking cells (jobs 71441–71474). Data layout is unchanged —
`<gen>_<sys>_5k/seed{42,43,44}/` — with no aggregation layer (analyzed downstream).

*(TODO — the full 4×4 × 3-seed matrix results once complete.)*
