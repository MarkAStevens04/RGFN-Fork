# Active-Learning Pipeline Architecture — the generalizable design

**What this doc is.** The spec for the *live* active-learning (AL) loop as a **generalizable,
modular pipeline**: one loop that runs across many molecule **proposers** (reaction-GFNs *and*
non-chemistry-aware SMILES generators), many **batching strategies** (flow-based hub-batching *and*
retrosynthesis-based batching), many **oracles**, and a swappable learned **proxy** — so any
`(proposer × batching × oracle × proxy)` combination is a *config choice*, not new plumbing.

It complements, and does not duplicate, the two existing design docs:
- `LSD_FLOW_PROPOSAL.md` — the flow-recovery math + hub-selection primitives (`glue/samplers/lsdflow/`).
- `LSD_FLOW_BENCHMARK_PLAN.md` — the *post-hoc, fixed-pool* library-efficiency benchmark (Generator →
  Selection → Evaluator). **This doc is the same three stages wrapped in the live AL loop**
  (fit `M` → train/query proposer → **acquire batch** → dock → grow `D` → trace). Where the benchmark
  measures a frozen pool, the AL loop closes the feedback loop; both consume the *same* pluggable
  stages and file contracts, and the retrosynthesis-batching path **reuses the benchmark's SPARROW
  machinery verbatim** (`validation/lsdflow/eval/`).

Prime directives (inherited from the repo): **simplicity + modularity**; never edit `rgfn/` or
`configs/` (except `configs/glue/`); `validation/` may import `glue/`/`rgfn/`, never the reverse;
heavy third-party generators install via `external/setup_*.sh` and cross conda-env boundaries **by
subprocess, never by import**.

---

## 1. The four orthogonal axes

```
  Proposer ──pool/flow──▶  Acquisition (batching) strategy ──batch──▶  Oracle ──▶ [fit M, grow D, trace]
 (reaction-GFN            (how the docking batch is chosen)            (the expensive
  OR SMILES gen)                                                       lab measurement)
                                          ▲
                                    Proxy M (fast surrogate,
                                    refit on oracle labels each round)
```

| Axis | Contract | Examples (built / planned) |
|---|---|---|
| **Proposer** | a per-env worker that samples molecules (+ a flow field / routes if it has them) and scores them with the round's `M` | RGFN (in-env) · SCENT · RxnFlow · FragGFN · *S3-GFN, REINVENT, VAE-BO (planned)* |
| **Acquisition / batching** | `given(proposer_state, M) → molecule batch (+ provenance)` | see §2 |
| **Oracle** | `GlueOracle` subclass, dispatched by name through the bridge | docking (6TD3 differential, sEH, ClpP), mock · *MD / co-folding (planned)* |
| **Proxy `M`** | `fit(smiles, labels)` + `predict` (a `CachedProxyBase`) | `LearnedGlueProxy` (RGFN) · `LearnedDockingProxy` (SCENT) |

The axes are independent: a new oracle is a `GlueOracle` + one registry line; a new proposer is a new
per-env worker; a new batching strategy is a new selector behind the §2 contract.

---

## 2. Batching is a first-class strategy *family* (the load-bearing axis)

"How is the docking batch chosen" is not one thing — there are fundamentally different **paradigms**,
and which ones apply depends on what the proposer produces:

| Paradigm | Mechanism | Requires | Applies to |
|---|---|---|---|
| **flow-hub-batching** (LSD-Flow) | read high-traffic pre-terminal **hubs** from the flow, enumerate each hub's 1-reaction children, UCB-rank by `z(reward)+λ·z(U(h))`, diversify into modes; pre-select-K reusable fragments | a reaction **flow field** + child enumeration | RGFN, SCENT, RxnFlow, FragGFN |
| **retrosynthesis-batching** | route the proposed molecules post-hoc (AiZynth), **merge shared intermediates**, pick batches that co-synthesize (SPARROW MILP) | a retrosynthesis planner (no flow field needed) | **any** proposer — the *only* batching available to non-reaction SMILES generators |
| **best-candidate** | rank the pool by `M`, keep top-`M` past the hit-bar + Tanimoto-diverse | just a proxy ranking | any proposer (the universal control) |
| **random / policy** | uniform / learned sampling, no batching logic | sampling | any proposer (the Fig.7 floors) |

**Why this matters for the end state:** a non-chemistry-aware proposer has *no* flow field and *no*
route-by-construction, so hub-batching is impossible for it — its route to a co-synthesizable batch is
**retrosynthesis-batching**. Both paradigms emit the same thing (a molecule batch) into the same loop
and the same oracle, so the AL loop is paradigm-agnostic; only the selector differs.

The uncertainty signal `U(h)` (variance of hub-flow estimates) is the exploration term of the
flow-hub-batching acquisition; retrosynthesis-batching has its own information signal (proxy-error /
enrichment per oracle call — `LSD_FLOW_BENCHMARK_PLAN.md` §11).

---

## 3. The architectural principle: shared *contracts*, thin per-env loops

The generators live in **mutually-incompatible conda envs** (SCENT's package is *also* named `rgfn`
and shadows ours; RxnFlow/FragGFN/S3-GFN each isolated). No two co-import in one process, so there is
**no single AL-loop class imported everywhere**. Generality instead comes from a small set of
**shared contracts** that every env speaks, with a **thin per-env loop** that owns only the
env-specific "train/query the proposer" step:

- **The oracle bridge** — `scripts/score_batch.py` (`--oracle <name>`): one scorer, every env shells to
  it (`conda run -n rgfn ...`). Adding an oracle = a `GlueOracle` subclass + one `ORACLES` entry.
- **The acquisition selector** — `validation/lsdflow/select_acquisition.py`: the heavy selection runs
  in the `rgfn` env (where `glue` imports cleanly) on **files** the proposer's env-worker wrote. One
  selector serves every cross-env generator; the in-env RGFN loop calls the equivalent
  `glue/samplers/lsdflow/acquisition.py` directly.
- **The per-env worker contract** — `<gen>_worker.py --mode {sample,enumerate}` emits the canonical
  files (§4). `scent_worker.py` is the reference; the uniform contract across all four generators is
  being standardized (matrix16 work).
- **The candidate-dataset format** (`docs/CANDIDATE_DATASET_FORMAT.md`, `glue/datasets/candidates.py`) —
  shared provenance (`has_route`, routes, per-batch metrics).
- **The trace + curve** — every arm writes `oracle_calls.csv` (oracle & reward-gen call axes),
  `acquisition_timings.csv` (per-component wall-clock), `phase_timings.csv`; `validation/harness/
  acquisition_curve.py` reads them across arms/seeds into the top-k-vs-oracle-calls figure.

**Consequence:** the four `al_loop.py`s (`glue/active_learning/loop.py` in-env; `validation/generators/
{scent,fraggfn,rxnflow}/al_loop.py` cross-env) are deliberately near-identical thin drivers over these
contracts. New generator ⇒ new thin loop + worker, **not** new acquisition/oracle/trace code.

---

## 4. The contracts, concretely

**Per-env worker → selector (the acquisition file contract).** Each round, the proposer's env writes:
- `enum_children.json` — `{"hubs":[{hub_key, hub_input, depth, uncertainty (U(h)), n_effective,
  children:[{smiles, reward (M's value), added_promoted}]}]}` — flow-hub-batching input.
- `compositions.json` — `child_key → {promoted:[...], num_reactions}` — per-molecule promoted fragments
  (cost model).
- `records.csv` — the sampled pool (`hub_key, child_key, reward, …`) — best-candidate input; also the
  molecule pool that **retrosynthesis-batching** routes.
- a promoted-fragment snapshot (`fragments_<N>.json`) — `FragmentCostTable` for pre-select-K.

**Selector → loop:** `chosen.csv` (`smiles[, source_hub]`) — the round's docking batch.

**Loop → oracle:** the batch SMILES (+ optional routes) → `score_batch.py` → labels + standard
candidate shards.

**Loop → trace:** `oracle_calls.csv`, `acquisition_timings.csv`, `phase_timings.csv`.

Same contract, two selector implementations today: `hub_batching` / `best_candidate` in
`select_acquisition.py`; the `retrosynthesis` selector (planned) wraps `validation/lsdflow/eval/`
(`network.py` route-merge + `sparrow.py` MILP).

---

## 5. Extension seams (each a small, named change — do NOT build ahead of need)

- **New oracle** → `GlueOracle` subclass in `glue/oracles/` + one line in `score_batch.py::ORACLES`
  + per-target hit-bar/direction in the config. (MD, co-folding, a learned classifier.)
- **New reaction-GFN proposer** → a per-env worker emitting §4 files + a thin `al_loop.py`; flow-hub
  and best-candidate work for free.
- **New non-reaction proposer** (S3-GFN, REINVENT, VAE-BO) → a per-env worker emitting a SMILES pool
  (`records.csv`, `has_route=0`) + a thin loop; **only** best-candidate + retrosynthesis-batching apply
  (no flow field).
- **New batching paradigm** → a new selector behind the §2/§4 contract (`select_acquisition.py`
  `--paradigm ...`, or a sibling script). The loop is unchanged.
- **New proxy `M`** → a `CachedProxyBase` with `fit()`; wire in the config.
- **New `U(h)` / hub-scoring / molecule-selection function** → register in
  `glue/samplers/lsdflow/hub/ucb.py` (`REWARD_FNS` / `UNCERTAINTY_FNS`) / `hub/registry.py` /
  `child_select.py`.

---

## 6. Current state (built / in-progress / planned)

- **Built + validated:** the four axes for reaction-GFNs; flow-hub-batching + best-candidate acquisition
  (in-env `glue/samplers/lsdflow/acquisition.py`; cross-env `validation/lsdflow/select_acquisition.py`);
  the oracle bridge (4 docking oracles + mock); the trace + curve; RGFN in-env AL run (6TD3, live);
  the SCENT selector validated on real sEH enumeration.
- **In progress:** the SCENT cross-env AL loop wiring (warm-start + in-process enumerate → selector →
  dock); the uniform per-env worker contract across all four generators (matrix16).
- **Planned (architected-for, not built):** the retrosynthesis-batching selector (wraps the SPARROW
  evaluator); non-reaction proposers (S3-GFN, REINVENT, VAE-BO) as AL arms; MD/co-folding oracles;
  the information-per-oracle-call metric for the retrosynthesis paradigm.

---

## 7. Design decisions (resolved) + one deferred question

**Q1 — Loop-code dedup vs env isolation → RESOLVED: shared *contracts*, thin loops, reuse env-safe
functions.** A single shared loop *class* is impossible across the conda envs (a forced one would be
less honest than the real design), so for a top-tier venue the generalizability is carried by (a) the
documented **contracts** (§4: the worker output files, the oracle bridge, the selector, the trace
schema) and (b) thin per-env loops that **reuse env-safe importable functions** rather than duplicating
— e.g. the SCENT loop imports `scent_worker`'s vendored `extract_flow_records`/enumerator in-process
(its `rgfn` imports live inside `main()`, so the module imports cleanly in any env). The paper shows
"one contract set + N thin adapters," reproducible and legibly modular.

**Q3 — Hub-vs-retrosynthesis comparability → RESOLVED: fixed-reaction-budget campaigns, cumulative
modes.** The two paradigms are compared on the **shared cost currency = bench reactions**, not oracle
calls. Each runs its **own AL campaign** under a **constant per-cycle reaction budget (~100)**: dock
*every* candidate that fits within that budget, over **5 AL cycles**, and track the **cumulative
distinct modes discovered** as the campaigns progress. The headline is of the form *"our AL campaign
discovered 764 modes with hub-batching but only 212 with retrosynthesis-batching at a constant
100-reaction budget across 5 cycles"* — the reaction-cheap shared scaffolds of forward construction
converting the same synthesis budget into far more diverse hits than post-hoc route recovery can.
(NOTE: the current RGFN/SCENT *generator-comparison* runs use a mode-count budget + the oracle-call
Fig.7 axis; the **reaction-budget + cumulative-modes** framing is the *paradigm-comparison* readout, to
be wired with the retrosynthesis-batching selector. Revisit whether the generator runs should also
report the reaction-budget axis for one consistent headline.)

**Q2 — Retrosynthesis-batching's `U(h)`-analogue → DEFERRED** (researcher: revisit later). Flow-hub's
exploration signal is `U(h)` = variance of hub-flow estimates; retrosynthesis-batching needs its own
(proxy-error-reduction, route-novelty, or pure-exploit) — decided when that selector is built.
