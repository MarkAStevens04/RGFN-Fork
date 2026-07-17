# LSD-Flow: Post-Hoc Hub Selection for Batched Late-Stage Diversification

**A design proposal for the coding agent.** This document specifies the architecture, data
model, interfaces, metrics, and experiment matrix for a post-hoc analysis pipeline that
extracts batchable "synthetic neighborhoods" from already-trained reaction GFlowNets, and exposes
its hub-selection logic as an acquisition function the active-learning loop can consume.

It is written to be built against. Sections marked *contract* / *interface* are hard specs.

**Placement summary (read this first — it drives the whole layout).** LSD-Flow splits across your
two repo axes on purpose:

- The **acquisition primitives** (flow recovery, `U(h)`, visitation estimate, hub/molecule
  selection strategies + registry) are **pipeline science** and live in `glue/`. They are the same
  category as the existing `glue/samplers/` (batch-selection) and `glue/metrics/`. The AL loop
  imports these directly.
- **Everything analytical/comparative** (adapters + per-env workers, the canonical DAG, severe-test
  suite, RGFN-vs-SCENT hub-coincidence study, Pareto/cost analysis, experiment matrix) is
  **validation-axis** and lives self-contained under `validation/lsdflow/`. It imports the
  primitives from `glue/` (allowed by the one-way rule) and is **never imported back** by the
  pipeline.

This is the same shape as your RGFN entrant: the real strategy logic is production code in `glue/`,
and the `validation/` subtree is the thin harness that measures it. It respects the one-way
dependency rule (`validation/` -> `glue/`+`rgfn/`, never the reverse) while still letting the
production-side AL loop call the acquisition functions.

Status of open design questions: **all resolved.** Model scope, adapter contract, P_B
recoverability, flow/uncertainty math, cost/information metrics, placement, the AL interface, and
build ordering are settled. The one prerequisite (SCENT `nn.ModuleList` patch + retrain) is
researcher-owned, not this agent's work (section 9).

---

## 1. Thesis and what the pipeline must demonstrate

Reaction-based GFlowNets learn a flow field over a DAG whose nodes are molecules and whose edges are
reactions. The claim: **high-reward, synthesizable, *batchable* neighborhoods are already latent in
that flow field** and can be harvested *after* training with no retraining and no reward
modification. A "hub" is a pre-terminal state `h` from which many diversifying reactions yield
distinct high-reward products. Building `h` once and running many final reactions on it amortizes
the synthesis of a diverse library -- late-stage diversification, recovered as a property of the flow.

The pipeline must (a) discover hubs under interchangeable strategies, (b) select a product batch from
each hub under interchangeable strategies, (c) score batches with interchangeable cost and
diversity/information metrics, (d) run across four trained models and four reward targets, and
(e) expose hub-based batch selection as an acquisition function for the AL loop. Adding a new
strategy or metric must be trivial.

The scientific framing is **severe testing**: every analysis is written to try to *falsify* the
claim (hubs are secretly low-diversity, or deep/expensive, or one dominant mode); publication value
comes from those attempts failing.

**Positioning relative to SCENT.** We complement, not compete. SCENT bakes cost-awareness into
*training* via a cost-guided backward policy and a dynamic library. LSD-Flow shows the batchable
structure is extractable *post-hoc* from any trained reaction GFN -- including SCENT itself. Running
LSD-Flow on SCENT is the spine of the paper (section 8).

---

## 2. The flow recovery and the uncertainty signal (the math the code implements)

For a hub `h` and a terminal product `x` reachable from `h` by one reaction (then `stop`):

```
F_hat(h ; x) = R(x) * P_B(h | x) / ( P_F(x | h) * P_F(stop | x) )
```

- `R(x)` -- reward at the terminal product. **Required input** (reward is the only source term).
- `P_B(h | x)` -- backward prob of undoing the final reaction. Must be the model's **own trained**
  backward policy (section 5).
- `P_F(x | h)` -- forward prob of the diversifying reaction. For phased models (RGFN/SCENT) this is
  the product of the A->B->C micro-step softmaxes composing one molecule->molecule move.
- `P_F(stop | x)` -- the terminating factor. **Do not drop this**; it equals 1 only for true leaves,
  and omitting it biases `F_hat` upward for products eager to react further.

All computed in **log space** from the per-step `logP_F`/`logP_B` that `objective.assign_log_probs`
produces plus the shaped `log R(x)`.

**The uncertainty signal.** Each terminal child `x_i` of `h` independently estimates the same
`F(h)`; under a converged GFN they agree. Define

```
U(h) = Var_i [ log F_hat(h ; x_i) ]        # variance across sampled children, in log space
```

`U(h)` is the local flow-matching residual = an epistemic-uncertainty signal with no ensemble/
dropout/extra model. Three roles across the two phases:

- **Hub-selection criterion** (phase 1, fixed reward): rank/gate hubs by consistency.
- **Diagnostic** (phase 1): does `U(h)` correlate with disagreement between flow estimators?
- **Acquisition signal** (phase 2, AL): high-`U` hubs are where the model is internally inconsistent
  about downstream reward -> most informative to query. **This strong claim requires the AL phase --
  do not over-promise it in phase 1.**

Log space is mandatory (estimates are heavy-tailed; raw-space variance is dominated by one outlier).

**Reward-free cross-check.** A trained GFN visits states proportional to their flow, so empirical
visitation frequency of `h` is a reward-free MC estimate of `F(h)`. Disagreement between the
DB-based estimate and the visitation estimate (after the `logZ` shift) is a diagnostic -- and
directly tests whether SCENT's cost guidance broke trajectory balance. Ship it as one more
hub-selection strategy.

---

## 3. Repository layout (split across the two axes)

> **Current state vs. this design.** The authoritative map of what exists today is
> `validation/lsdflow/README.md` + `experiments/lsd_hubs/campaign/README.md`; where the layout below
> differs, the READMEs win. Two things shipped differently from the sketch: (1) the batch-selection
> logic is the **campaign strategies** — `glue/samplers/lsdflow/campaign.py`
> (`BestCandidateStrategy` / `HubBatchingStrategy`) with within-hub `child_select.py` policies and a
> `mode_select.py` diversity selector — not a separate `molecule/` strategy registry; (2) the
> implemented cost model is **count-once** (`validation/lsdflow/metrics/cost/dynamic_amortization.py`,
> §11). The AL-facing acquisition entry point (§4a), the severe-test / hub-coincidence suite (§7/§8),
> and the multi-generator library-efficiency benchmark (`docs/LSD_FLOW_BENCHMARK_PLAN.md`) are
> planned, not built.

### 3a. Production side -- the acquisition primitives (`glue/`)

These are the only files the AL loop imports. Small, pure, no comparative machinery.

```
glue/
  metrics/
    lsdflow_flow.py      # F_hat(h;x) recovery + visitation estimate, log-space (the section 2 math)
    uncertainty.py       # U(h)
  samplers/
    lsdflow/             # hub selection IS a batch-selection strategy -> lives with glue/samplers/
      __init__.py
      hub/
        base.py          # HubSelectionStrategy ABC: HubDAG -> ranked hubs
        highest_flow.py
        highest_terminating_flow.py   # flow from children that stop after 1 rxn
        most_modes.py
        parent_of_topk.py             # control
        highest_visitation.py         # reward-free
        lowest_uncertainty.py         # gate/rank by U(h)
        registry.py
      campaign.py        # the batch-selection strategies: BestCandidateStrategy | HubBatchingStrategy
      child_select.py    # within-hub child policies (reward | free_frag)
      mode_select.py     # diversity mode-acceptance
      records.py         # FlowRecord + the HubDAG-shaped duck type
      dag.py             # LiteHubDAG (lightweight in-env aggregation)
      rgfn_extract.py    # rgfn-native trajectory -> FlowRecord
      rgfn_enumerate.py  # exhaustive one-reaction child enumeration
      # (planned, §4a) acquisition.py — AL-facing hub acquisition sampler
```

`glue.registry` must import `glue.samplers.lsdflow` and the two `glue.metrics` modules so gin sees
the classes.

### 3b. Validation side -- the analysis world (`validation/lsdflow/`)

Self-contained. Imports primitives from `glue/`; never imported by the pipeline.

```
validation/lsdflow/
  __init__.py
  adapters/
    base.py              # GFNAdapter ABC -- the six-method contract (section 4b)
    protocol.py          # RPC schema: SMILES + action-ids + logprobs
    client.py            # in-process handle -> per-env worker subprocess
    workers/
      rgfn_worker.py     # runs in the rgfn env
      scent_worker.py    # runs in the scent env (SCENT's package is also named `rgfn`)
      fraggfn_worker.py  # runs in the recursion-gflownet env
      rxnflow_worker.py  # runs in the RxnFlow env
    registry.py          # name -> (conda env, entrypoint)
  dag/
    node.py              # canonical node identity (section 6)
    graph.py             # HubDAG: aggregation of sampled trajectories
    build.py             # trajectories -> HubDAG
  analysis/
    severe_tests.py      # the falsification suite (section 7)
    hub_coincidence.py   # RGFN-vs-SCENT flow hubs == SCENT-promoted intermediates (section 8)
    pareto.py            # diversity vs concurrency front
  metrics/
    cost/
      dynamic_amortization.py       # count-once synthesis cost (FragmentCostTable, nesting closure)
      compute_time.py               # measured per-hub compute-time accounting
    diversity.py         # paper-comparable modes (Morgan r=3/2048, Tanimoto 0.7) + scaffolds
    # (planned) information.py — proxy-error-reduction per oracle call (AL phase)
  harness/
    matrix.py            # model x reward x strategy x metric sweep driver
    config.py            # dataclass configs, gin/YAML-loaded
    run.py               # entrypoint
  configs/               # run specs (model x reward x strategy x budget x seeds)
  results/               # committed tables + plots (small artifacts only)

docs/
  LSD_FLOW_PROPOSAL.md   # this document

experiments/
  scent_retrain/         # the ModuleList-patched SCENT retrain runs (section 9) -- its own run group
```

**Why the `HubDAG` sits in `validation/` even though strategies in `glue/` operate on it:** the DAG
is built from adapter output (validation-only, cross-env). The `glue/` strategies are written
against a lightweight `HubDAG`-shaped **protocol/duck type**, and in the AL path they operate on the
DAG the loop already has from its own sampling (no adapter, single env). So the strategies never
import the validation `HubDAG` concretely -- they take anything exposing the node/edge/flow fields
in section 6. This keeps the one-way rule intact.

---

## 4a. The AL-facing acquisition interface (production side) — PLANNED (not yet built)

**Decision: the AL loop consumes molecules, not hubs.** Hubs are an internal detail of *how* the
batch is chosen. This keeps the production-side change to a single new sampler plugin -- `loop.py`
and the Bengio-Alg.1 driver are untouched, and the loop's `select_batch` signature does not change.

```
# glue/samplers/lsdflow/acquisition.py
class LSDFlowAcquisition:
    """Batch-selection strategy. Chooses hubs internally, returns their diversified products."""
    def __init__(self, hub_strategy, molecule_strategy, batch_size, ...): ...
    def select_batch(self, sampled_trajectories, proxy) -> list[Molecule]:
        # 1. build a lightweight HubDAG from the trajectories the loop already sampled
        # 2. hub_strategy ranks hubs (may use U(h) from glue.metrics.uncertainty)
        # 3. molecule_strategy picks products per selected hub
        # 4. flatten to a molecule batch of size batch_size; return
```

The loop hands it the trajectories + current proxy and gets a flat molecule batch back -- identical
contract to any other sampler. Hub structure, `U(h)`, and amortized-cost accounting stay inside.

**Deferred to phase 2 (explicitly not built now):** a hub-aware loop that acquires a parent + its
children as a unit and credits the parent's synthesis once in the oracle budget. That would change
the loop's batch abstraction and touch `glue/active_learning/loop.py`; we defer it until fixed-
reward results justify it. v1 is a drop-in sampler only.

---

## 4b. The analysis adapter contract (validation side)

**Non-negotiable constraint from repo review:** the four models are two API families in four
mutually-incompatible conda envs; no two co-import in one process. The adapter is **not one imported
class** -- it is a subprocess/RPC contract. Each model runs in a per-env worker; the main process
exchanges **SMILES + action-ids + logprobs** over the boundary. Reuse the `scripts/score_batch.py`
bridge shape (`conda run -n <env> ...`).

Six methods per worker; `GFNAdapter` is the in-process client mirror.

```
sample_trajectories(n)            -> list[Trajectory]   # canonical schema, rewards attached
forward_logprob(state, move)      -> float              # molecule->molecule; composes A/B/C for phased
backward_logprob(child, parent)   -> float              # the trained P_B (section 5)
stop_logprob(state)               -> float              # P_F(stop | state); required by section 2
enumerate_children(state)         -> list[(move, product_state)]   # applicable rxn/reactant -> product
reward(state)                     -> float              # active oracle/proxy; f(list[SMILES])->list[float]
```

**"One generative move" is model-defined.** For RGFN/SCENT the atomic policy step is a micro-step
(A: template/stop, B: reactant, C: commit), so `forward_logprob`/`enumerate_children` operate at
*reaction* granularity by composing A->B->C. For FragGFN/RxnFlow the atomic graph action already is
one move. Never assume an atomic `(state, action) -> product`; pass a model-defined `Move`.

Per-model support (leaks handled explicitly):

| method | RGFN | FragGFN | RxnFlow | SCENT |
|---|---|---|---|---|
| sample_trajectories | full | full | full (must sample) | full |
| forward_logprob | exact | exact | exact per-action; normalizer approx unless ratio=1.0 | exact |
| backward_logprob | exact, ckpt | analytic (uniform) | approx heuristic | exact, ckpt **after patch** (section 9) |
| stop_logprob | yes | yes | yes (set allow_stop) | yes |
| enumerate_children | ~10^2-10^3 | ~10^2 | 10^3-10^6, sample unless stdlib | ~10^2-10^3 after library freeze |
| reward | yes | yes | yes | yes |

- **RGFN** -- full six, cleanest `U(h)`. The anchor.
- **SCENT** -- full six *after the ModuleList patch + retrain* (section 9). Core target. Freeze the
  dynamic library to its final promoted-fragment snapshot (`additional_fragments/fragments_*.json`,
  `chosen_smiles`) before `enumerate_children`; not needed for sampled-trajectory flow recovery.
- **FragGFN** -- all six, but P_B is fixed-uniform and "one move" is a fragment attachment, not a
  reaction. **Control only; never cross-compare its hub values to reaction models.** The "does
  reaction structure matter?" ablation.
- **RxnFlow** -- include only at `sampling_ratio=1.0` on the SMALL library, **library-level claims
  only** (diversity, cost, reactions-per-mode). **No `U(h)`** (heuristic P_B muddies it). Robustness
  row.

---

## 5. Why P_B must be the model's own trained policy

`U(h)` means "flow-matching residual" only if `P_B` is the policy `P_F` was trained against, so
per-child estimates agree at convergence:

- RGFN: learned P_B, recoverable -> clean.
- FragGFN: fixed-uniform P_B, *trained with* uniform P_B -> consistent (uniform is correct; `U(h)`
  then reflects forward-policy inconsistency alone). Control, so we don't lean on it.
- RxnFlow: retro-heuristic P_B, not TB-paired -> `U(h)` muddied. No `U(h)`.
- SCENT: cost-guided P_B. The recovery is **agnostic to how P_B is shaped** (enters only as a
  path-product), so cost-guided P_B is fine *if it is the trained one*. Repo review proved the
  guidance weights are **not** in the checkpoint (stored in a plain `list`, invisible to
  `state_dict()`); reseeding them moves P_B on 75% of multi-parent backward choices. Refit-from-
  static-costs is **not** faithful (injects systematic prod(P_B'/P_B) bias). Fix = the one-line
  ModuleList patch + retrain (section 9); afterward P_B is exactly recoverable, and the recovered
  `F(h)` is **cost-aware for free** (SCENT's P_B tilts toward cheap, decomposable parents -- that
  tilt propagating into the hub score is the synergy the paper sells).

Confirmed non-issues: TB balances against **raw** `P_F` (exploitation penalty wraps only the
training sampler, never the loss), so `assign_log_probs` gives the right `P_F`; `logZ` is restored
to a trained value (needed for absolute flow and the visitation cross-check).

---

## 6. Canonical DAG data model

One shared node identity across all four models so aggregation is model-agnostic. The `glue/`
strategies operate on anything exposing these fields (duck-typed), so they never import the
validation `HubDAG` concretely.

- **Node key:** RDKit canonical SMILES, **stereo stripped for the cross-model key**
  (`MolToSmiles(mol, isomericSmiles=False)`) -- FragGFN strips stereo while reaction models keep it,
  so keeping it would block same-skeleton key matches. Keep a stereo-aware secondary key for
  within-reaction-model work.
- **Node fields:** canonical key; MC visit count (reward-free estimator); cached reward(s) keyed by
  oracle; per-child flow estimates (for `U(h)`); depth (min #reactions from source); per-trajectory
  is_terminal flags.
- **Edge fields:** `Move` (template + reactant for reaction models; graph action for FragGFN),
  `logP_F`, `logP_B`.
- **Storage:** in-memory `networkx` per run; persist to `validation/lsdflow/results/` (parquet
  nodes+edges or pickled graph) keyed by `(model, reward, run_id)`, mirroring the per-experiment
  convention (config + git hash copied in).

**Reward reuse (docking cost).** Rewards for *sampled* terminal children come free from training --
**reuse cached training rewards, do not re-dock.** Only post-selection full enumeration of a
*selected* hub's children pays fresh oracle cost; for docking (6TD3/ClpP) **cap
`#selected_hubs x #children`** against a per-round budget. Surrogates (sEH/DRD2) enumerate freely.

---

## 7. Severe-test analysis suite (`validation/lsdflow/analysis/severe_tests.py`)

Each written to *falsify* the thesis; value = they fail to falsify.

1. **Diversity, not redundancy.** Modes per batch (Butina/ECFP, primary), Bemis-Murcko scaffolds,
   #Circles. Falsifier: batches are near-duplicates.
2. **Depth parity.** Mean #reactions to build the parent hub vs to reach an independently-found
   high-reward molecule. Falsifier: hub parents are deep, so "build once" saving is illusory.
3. **Total-reward parity.** Batch reward vs independent top-k at matched synthesis budget.
   Falsifier: batching sacrifices reward.
4. **Mode coverage, not one dominant child.** Does flow selection recover only the global mode?
   Per-child reward distribution within a hub. Falsifier: one child dominates.
5. **Noise floor.** Is `U(h)`/flow ranking stable across seeds and `N`? Log effective sample count
   per hub. Falsifier: estimates too noisy to act on.

---

## 8. RGFN-vs-SCENT hub-coincidence study (`validation/lsdflow/analysis/hub_coincidence.py`)

The paper's spine. SCENT's Dynamic Library caches high-reward intermediates via a utility metric
(their Eq. 13, mean reward over trajectories through a state) -- a training-time cousin of
flow-based hub value. Experiment: **do the hubs LSD-Flow discovers post-hoc coincide with the
intermediates SCENT learned to promote** (`chosen_smiles` in the frozen snapshot)?

- **Coincide** -> two independent mechanisms converge on the same batchable hubs; validates both.
- **Diverge** -> LSD-Flow finds hubs SCENT missed (especially the *highest-terminating-flow* variant,
  which rewards diversification parents rather than good general building blocks); flow captures
  something reward-averaging utility misses.

Both publishable; both land the thesis on top of current SoTA. Report Jaccard of hub sets, rank
correlation of hub value vs SCENT utility, plus the visitation-vs-balance-recovered agreement test
as a TB-integrity check.

---

## 9. Prerequisite owned by the researcher (not this agent)

Before SCENT is a live target, the researcher will: wrap `JointlyGuidedBackwardPolicy.policies` in
`nn.ModuleList(...)` (one line -- makes the two 262,913-param guidance MLPs visible to
`state_dict()`, self-healing for all future checkpoints, and makes `load_state_dict` flag them as
real-missing if ever absent) and **retrain the SCENT fixed-reward matrix** (sEH/DRD2/ClpP/6TD3) into
`experiments/scent_retrain/`. These re-runs are phase-1 work.

The only config deviation from SCENT's published setup is the **reward target** (our 6TD3/ClpP/sEH/
DRD2 vs their sEH/GSK3b/JNK3). Everything else -- including the **SMALL chemical & reaction library
(418 fragments / 112 templates), used as the shared substrate for all four models** -- matches
SCENT's published config, keeping our SCENT numbers comparable to their Table 1 on overlapping
targets. Verify `fragments_4000.json` is the final library state (no promotion in iters 4001-5000)
with a one-line training-log check.

Until patched checkpoints exist, the SCENT worker is stubbed but wired; the file tree does not
change when it goes live.

---

## 10. Build order

1. **RGFN adapter + DAG + flow recovery + `U(h)`** end-to-end on one fixed reward (sEH). The anchor,
   ready today. Prove the whole vertical slice, including the `glue/`<->`validation/` split working
   across the env boundary.
2. **Strategy + metric registries** -- full hub/molecule/cost/diversity plugins; RGFN across all four
   fixed rewards. **Wire `LSDFlowAcquisition` into the AL loop as a sampler** (no `loop.py` changes)
   and smoke-test that it returns a molecule batch.
3. **SCENT adapter** once patched checkpoints exist; the hub-coincidence study.
4. **FragGFN** (control) and **RxnFlow-on-stdlib** (library-level robustness row).
5. **Severe-test suite + Pareto front**; write-up of phase-1 (fixed-reward) results.
6. **Active-learning phase**: retrainable proxy (already first-class -- `LearnedGlueProxy` /
   `AtomMPNNProxy` / `LearnedDockingProxy`, all with `fit()`), 6TD3/ClpP docking oracle,
   information-per-oracle-call on held-out hits + property-matched decoys (point at your existing
   `experiments/oracle_validation/docking_{6td3,crbn}/*_results.csv`), and `U(h)` as acquisition
   signal. Capstone, one section -- the paper stands on phases 1-5 without it.

---

## 11. Metric definitions (pin before coding registries)

- **Mode** (primary): Butina / sphere-exclusion cluster on ECFP4 at a fixed Tanimoto cutoff (start
  0.65, config knob). "One mode = one representative you'd actually synthesize." Secondary: unique
  Bemis-Murcko scaffolds; #Circles.
- **Cost -- reactions per mode** (PRIMARY): `total_reactions_to_build_batch / n_modes`, lower better.
  The implemented model is **count-once** (`validation/lsdflow/metrics/cost/dynamic_amortization.py`
  + `campaign.py`): every synthesis step is charged **exactly once** — a shared hub's assembly
  couplings once (`shallow_couplings = num_reactions − Σ nested build cost of each attached SCENT
  promoted fragment`), each distinct promoted dynamic-library fragment built once (closure under
  nesting), applied identically to both strategies so only the *selection* differs. (The earlier
  `depth(h) + k` sketch double-counted SCENT's fully-nested `num_reactions`; see Logs/028/033.)
  Encodes the LSD value proposition and reads to a chemist. State that the real saving is reaction
  *time*, not reactant cost.
- **Cost -- amortization ratio** (SECONDARY, SCENT-comparable): `(depth(h) + k*C_step)` vs
  `sum_j C_full(x_j)`, using SCENT's own yield/reactant-cost tables so it's reviewer-proof.
- **Information** (phase 2): proxy-error reduction per oracle call on held-out known hits + property-
  matched decoys (your `docking_{6td3,crbn}/*_results.csv` are exactly this set). Track **both**
  regression error and **ranking/enrichment (AUROC / BEDROC)** -- average accuracy and hit-vs-decoy
  ranking differ, and ranking is what a screen needs. Track over/under-estimation (calibration).
  Ground truth here is *docking*, not experimental affinity -- say so.
- **Concurrency** (Pareto axis): batch size from one shared parent; plotted against diversity to map
  the front.

---

## 12. Baseline suite

Independent top-k (no batching); random hub; parent-of-top-N (the control strategy); SCENT's
generated library at the library level on the shared SMALL substrate. The SCENT-library fairness
knob (freeze its dynamic library to match vocabularies vs. let it explore a superset and frame
accordingly) is a phase-2 decision, not a v1 blocker.
