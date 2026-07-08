# sEH / RGFN — LSD-Flow phase 1: post-hoc hub extraction from the trained flow field
**Date:** 2026-07-08, ~3pm

## Question

Are batchable "hub" scaffolds — pre-terminal molecules from which many *different*
high-reward molecules can each be made in one final reaction — actually present in a trained
reaction-GFlowNet, and can we pull them out *after* training without retraining or changing the
reward?

## Context & Summary

Our generator (RGFN) builds molecules by composing chemical reactions, and it's trained to
sample high-reward molecules with lots of diversity. The **LSD-Flow** idea
(`docs/LSD_FLOW_PROPOSAL.md`) is that this diversity is not just a property of the *final*
molecules — it should mean there are intermediate scaffolds ("hubs") sitting in the model's
internal "flow" from which many good, distinct products branch off in a single reaction. If so,
a chemist could build that one hub scaffold once and then run several quick final reactions to
get a whole diverse library cheaply — "late-stage diversification." This experiment is the first
build and test of that idea (it replaces an earlier, abandoned hub-analysis pipeline, deleted
2026-07-08; none of that code was reused).

We built the machinery in two halves — the reusable acquisition primitives that live with the
training pipeline (`glue/`), and the comparative analysis world that measures them
(`validation/lsdflow/`) — and ran it on an already-trained model: the RGFN run that was trained
once against the sEH scorer on our shared standard building-block library (entry `020`, job
69616, 5,001 training steps). We sampled 10,000 molecules from the trained model, reconstructed
the flow value and an internal-consistency ("uncertainty") signal for every hub, ranked hubs
under six different strategies, and then had each strategy pick a 96-molecule library and
measured how many distinct chemical families ("modes") it covered and how many reactions it
would take to make versus building every molecule from scratch.

## Answer

The machinery works end-to-end on a real trained model, and batchable hubs **do** exist in the
flow: of 8,183 hubs seen, 613 had at least two distinct one-reaction products (up to 9), and
choosing a 96-molecule library from shared hubs cut the reactions-per-distinct-compound by
~40% versus building each molecule independently. But two honest caveats sharpen what comes
next. First, **91% of those multi-product hubs sit at the model's maximum reaction depth**,
where the model is *forced* to stop — so the raw hub count over-represents a boundary artifact
of the 4-reaction cap rather than genuine mid-synthesis diversification (only ~42 hubs are
shallow, depth 0–1, the "build-early, diversify-late" case the idea is really about). Second,
**flow-based hub ranking did not beat a naive "just take the parents of the best molecules"
control on synthesis cost** (3.55 vs 3.47 reactions/mode) — so if the flow signal earns its
keep, it will be on diversity/mode-coverage, which is exactly what the planned falsification
("severe test") suite is built to check.

## Relevance to our Publication

This is the foundation experiment for a potential methods contribution (target: Digital
Discovery, or a NeurIPS AI4Science / ICLR-MLDD workshop) whose thesis is that batchable
synthetic structure is *latent* in any trained reaction GFlowNet and can be harvested post-hoc.
This entry is the first evidence that the structure is there and extractable, and — just as
important for reviewers — it's framed as a severe test: we report where the method does not yet
win (cost parity with the naive control, boundary-dominated hubs) rather than only the
favorable number. It also proves the code architecture the paper depends on: the acquisition
logic ships as production `glue/` code while the comparative harness stays on the `validation/`
axis, so the same hub-selection can later be dropped into the active-learning loop.

## Next Experiments

**Refining for publication**
- **Control the sample-size confound:** the untrained-vs-trained hub-density jump was measured
  at different N (300 vs 10,000). Re-run the trained model at N=300 and/or an untrained model at
  N=10,000 so the density comparison is apples-to-apples.
- **Separate boundary hubs from interior hubs:** report hub statistics split by "children hit
  the reaction cap (forced stop)" vs "children chose to stop" — the latter is the real
  diversification signal.
- **Run the severe-test suite** (diversity/mode-coverage, depth parity, total-reward parity,
  mode dominance, noise floor) — this is where flow selection must out-perform the control.
- **Raise the reaction cap** (`max_num_reactions`) and re-measure, to see whether interior
  (non-boundary) hubs densify when the boundary is pushed out.
- **Enumerate, don't just sample:** add the phase-2 `enumerate_children` path so a chosen
  shallow hub's terminal children are enumerated exhaustively (surrogate rewards like sEH are
  free to enumerate), instead of relying on sampling to happen to hit them.

**Next steps in project**
- Run the same analysis across RGFN's other three fixed-reward targets (DRD2, ClpP, 6TD3).
- Wire the hub-based selection into the active-learning loop as an acquisition function and
  test whether hub-diversified query batches improve the oracle-efficiency curve (Objective 1).
- Bring in SCENT (once its backward-policy-recoverable retrain lands, entry `024`/§9) for the
  RGFN-vs-SCENT hub-coincidence study that is the paper's spine, then FragGFN/RxnFlow.

# Re-creation

## Relevant Files

Root: `./` (repo root). Analysis is GFN **inference only** (no docking), so it runs on a Balam
login node under `source ~/bin/rgfn-smoke-env.sh`.

**Production-side primitives (ours, new — `glue/`, imported by the AL loop):**
- `./glue/metrics/lsdflow_flow.py` — flow recovery `log F_hat(h;x) = logR + logP_B − logP_F(move) − logP_F(stop)`, log-space, plus the median-consensus / total-terminating aggregators and the reward-free visitation estimate.
- `./glue/metrics/uncertainty.py` — `U(h)`, the variance of a hub's per-child flow estimates (the flow-matching residual / internal-consistency signal).
- `./glue/samplers/lsdflow/records.py` — `FlowRecord` (one observed `hub→product` transition) + the HubDAG-shaped duck-type protocols the strategies read.
- `./glue/samplers/lsdflow/dag.py` — `LiteHubDAG`, the lightweight in-loop hub aggregation (children deduped by canonical key so `U(h)` isn't deflated).
- `./glue/samplers/lsdflow/rgfn_extract.py` — the rgfn-native extractor: walks each sampled trajectory, composes the last reaction's A/B/C micro-steps into one molecule→molecule move, reads the final stop micro-step as `P_F(stop|x)` and the learned C-step backward prob as `P_B`.
- `./glue/samplers/lsdflow/hub/` — 6 hub-selection strategies (`highest_terminating_flow`, `highest_flow`, `most_modes`, `parent_of_topk` [control], `highest_visitation` [reward-free], `lowest_uncertainty`) + registry.
- `./glue/samplers/lsdflow/molecule/` — 3 per-hub product-selection strategies (`topk_reward`, `prob_weighted`, `uniform_random`) + registry.
- `./glue/samplers/lsdflow/acquisition.py` — `LSDFlowAcquisition`, the AL-facing entry point (trajectories+objective → flat molecule batch; `select_grouped` keeps hub→children for cost accounting).

**Validation-side analysis (ours, new — `validation/lsdflow/`, imports `glue/`, never imported back):**
- `./validation/lsdflow/adapters/base.py` — `GFNAdapter` ABC + `FlowSample` canonical schema (the §4b six-method contract; phase-2 enumeration methods raise NotImplementedError until wired).
- `./validation/lsdflow/adapters/rgfn_adapter.py` — the in-process RGFN adapter: rebuilds the objective + the pure-policy `valid_sampler` from a gin config + checkpoint, loads `last_gfn.pt`, samples, extracts flow records.
- `./validation/lsdflow/adapters/registry.py` — declares all four target models + env + build status (only RGFN wired in v1).
- `./validation/lsdflow/dag/{node.py,graph.py,build.py}` — canonical stereo-stripped node key + the rich `HubDAG` (networkx view + per-node stats + CSV/JSON/gpickle persistence).
- `./validation/lsdflow/metrics/diversity.py` — Butina modes (ECFP4 @ Tanimoto 0.65) + Bemis-Murcko scaffolds.
- `./validation/lsdflow/metrics/cost/` — reactions-per-mode cost (hub batch = `depth(h)+k` vs independent = `Σ depth(x_j)`).
- `./validation/lsdflow/harness/{config.py,run.py}` — the vertical-slice driver.
- `./validation/lsdflow/README.md` — subtree map + the anchor-checkpoint provenance caution.

**Model under analysis (from entry `020`, job 69616 — the anchor):**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-59-53/train/checkpoints/last_gfn.pt` — RGFN trained once against the frozen sEH MPNN on `glue_standard_v1`, **epoch 5001** (optimizer step 5002, logZ 75.4). Its `fixed_reward/candidates/candidates.csv` reproduces entry `020`'s numbers (median sEH 7.263 / max 8.354), confirming provenance. **Caution:** the sibling dirs `2026-07-02_13-45-03` (epoch 30) and `2026-07-02_14-18-34` (epoch 40) are cancelled early attempts (jobs 69613/69615) — not valid for analysis.
- `./configs/glue/fixed_reward_seh_proxy_stdlib.gin` — the config the adapter rebuilds the objective/sampler from.

**Results (ours, this entry):**
- `./validation/lsdflow/results/seh_rgfn_pilot/` — persisted DAG (`records.csv`, `hub_summary.csv`, `graph.gpickle`), run `meta.json`, `report.json` (per-strategy hub rankings + diagnostics), `acquisitions.csv`.

**Job Logs:** run interactively on the login node; stdout captured under the session scratchpad (`lsdflow_10k_trained.log`).

## Relevant Versions

Branch `Hub-Analysis`. All LSD-Flow files above (`glue/metrics/lsdflow_flow.py`,
`glue/metrics/uncertainty.py`, `glue/samplers/lsdflow/**`, `validation/lsdflow/**`) plus the
two one-line registration edits (`glue/samplers/__init__.py`, `glue/metrics/__init__.py`) and
the `docs/REFACTOR_LOG.md` entry are **not yet committed**.
[TODO — add commit hash after pushing.]

## Relevant Resources

**Sources**
- LSD-Flow design spec: `docs/LSD_FLOW_PROPOSAL.md` (the §2 flow-recovery math, §4b adapter
  contract, §6 DAG model, §7 severe tests, §11 metrics this entry implements).
- RGFN (`[koziarski2024rgfn]`) — the reaction-GFlowNet + its micro-step (A: template/stop,
  B: reactant, C: commit) action space that `rgfn_extract` composes.
- GFlowNet / trajectory balance (`[bengio2021gflownet]`) — the flow the recovery inverts.
- Entry `020` — the trained sEH model under analysis; entry `024` — the SCENT P_B recovery this
  will build on for the phase-3 hub-coincidence study.

**Packages**
- `rgfn` env (py3.11): torch, dgl, RDKit, gin — the whole slice runs here. Login-node CUDA libs
  via `~/bin/rgfn-smoke-env.sh`.
- `networkx` 3.6.1 — the persisted DAG graph view.

## Method

1. Built the primitives + analysis subtree (files above); `py_compile` + a pure-logic unit test
   of the aggregation/strategies/acquisition on synthetic records; import smoke of the full
   `validation.lsdflow` stack. Verified the one-way dependency rule (`glue/` has no
   `import validation`).
2. Selected the anchor checkpoint by cross-checking Logs/020 against on-disk metadata: confirmed
   `2026-07-02_14-59-53` is the completed 5,001-iter run (metrics `epoch`, optimizer step,
   `candidates.csv` matching the log) and rejected the two cancelled early stubs.
3. Ran the harness on the trained checkpoint (login-node A100):
   ```
   source ~/bin/rgfn-smoke-env.sh
   python -m validation.lsdflow.harness.run \
     --checkpoint .../seh_proxy_stdlib/2026-07-02_14-59-53/train/checkpoints/last_gfn.pt \
     --config-path configs/glue/fixed_reward_seh_proxy_stdlib.gin \
     --n-trajectories 10000 --sample-batch-size 200 --run-id seh_stdlib_69616_10k \
     --out-dir validation/lsdflow/results/seh_rgfn_pilot
   ```
   It sampled 10,000 trajectories from the pure trained forward policy (rewards attached),
   recovered `F_hat`/`U(h)` per hub, ranked hubs under all six strategies, ran four
   (hub × molecule) acquisition combos through the reactions-per-mode + Butina-mode metrics,
   computed the flow-vs-visitation diagnostic, and persisted the DAG + report.

## Results

**Flow field / hub structure (10,000 trajectories, 9,995 valid terminals, 25,628 distinct
molecule nodes; logZ = 74.99):**

| quantity | value |
|---|---|
| total hubs | 8,183 |
| multi-child hubs (≥2 distinct one-reaction products) | 613 (7.5%) |
| max children per hub | 9 |
| mean depth of multi-child hubs | 2.82 |

*Children-count distribution (multi-child hubs):* 426 have 2, 121 have 3, 37 have 4, 14 have 5,
6 have 6, 3 have 7, 4 have 8, 2 have 9.

*Depth distribution (multi-child hubs):* depth 0: 15, depth 1: 27, depth 2: 13, **depth 3:
558** — i.e. 91% sit one reaction below the `max_num_reactions=4` cap, where the final reaction
forces a stop (`P_F(stop)=1`). Only ~42 hubs are shallow (depth 0–1) genuine diversification
points.

**Acquisition — 96-molecule library, `per_hub=8`, reactions-per-mode (lower = cheaper):**

| hub strategy × molecule strategy | hubs used | modes | scaffolds | rxn/mode (hub) | rxn/mode (independent) | reactions saved |
|---|---|---|---|---|---|---|
| highest_terminating_flow × topk_reward | 33 | 51 | 82 | **3.55** | 6.24 | 137 |
| highest_terminating_flow × uniform_random | 33 | 51 | 82 | 3.55 | 6.24 | 137 |
| lowest_uncertainty × prob_weighted | 48 | 51 | 52 | 4.53 | 7.18 | 135 |
| parent_of_topk × topk_reward (control) | 34 | 53 | 82 | 3.47 | 6.00 | 134 |

Hub-amortized selection cuts reactions-per-mode ~40–45% vs building independently — but the
flow-ranked strategy (3.55) does **not** beat the naive control (3.47) on cost, and the control
covers marginally more modes (53 vs 51). (topk_reward ≡ uniform_random here because most
selected hubs have ≤8 children, so both pick nearly all of them.)

**Diagnostics (TB-integrity, over the 613 multi-child hubs):**
- Pearson(consensus `log F`, total-terminating `log F`) = **0.926** (sanity check — the two
  flow aggregators agree, as they should).
- Pearson(DB-recovered `log F`, reward-free visitation `log F`) = **0.086** (near zero). The two
  independent flow estimates barely agree — expected to be large under a converged GFN (§2/§8),
  so this is a flag. Most likely a noise-floor + boundary artifact (multi-child hubs have tiny
  visit counts, 2–9, and deep hubs show very high `U(h)`, 16–38), but it warrants the
  severe-test noise-floor analysis before any TB-integrity claim.

**Machinery-validation note (not a scientific result):** an earlier 300-traj smoke and a 20k run
were executed on a *cancelled 30-iteration* checkpoint before the anchor was corrected; they
validated the code path (2/292 multi-child hubs there) but carry no scientific weight. The
trained-vs-untrained density comparison (0.7% → 7.5% multi-child) is confounded by N (300 vs
10,000) and must be re-run at matched N before being cited.
