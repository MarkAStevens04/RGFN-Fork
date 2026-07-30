# LSD-Flow library-efficiency benchmark — implementation plan

**For:** the coding agent (has the repo). **Not to be implemented beyond what each task states.**
**Prime directives:** *simplicity* and *modularity*. Every new piece is a drop-in behind a stable
interface; nothing edits `rgfn/` or `configs/` (except `configs/glue/`); `validation/` may import
`glue/`/`rgfn/` but never the reverse; heavy third-party code installs via `external/setup_*.sh` and
crosses conda-env boundaries by subprocess, never by import.

---

## Background & objective (read this first — it is not in the repo)

**What this project is.** LSD-Flow is a research fork of **RGFN** (Reaction-GFlowNet). An RGFN builds
molecules by composing chemical *reactions* over a fixed building-block set, so every molecule it
proposes is synthesizable *by construction* (it comes with a route), and it samples proportional to
reward — many diverse good molecules, not one optimum. **LSD-Flow** is a post-hoc capability on top of
a trained reaction-GFN: read the learned flow field to find high-traffic pre-terminal **"hub"**
intermediates, synthesize a hub once, then diversify it with many cheap last-step reactions. This is
*late-stage diversification* recovered from the flow — hence the name.

**The scientific claim we are testing.** A good screening library is two things at once: *near in
synthesis space* (its molecules share scaffolds/routes, so they are cheap to co-synthesize) and *far
in chemical space* (structurally diverse, so it actually covers ground). The single number that
captures this tension is **reactions per mode** — total distinct bench reactions divided by the number
of diverse, high-reward molecules obtained. The thesis: because a reaction-GFN's molecules are built
from shared reaction steps, it can hit a far lower reactions-per-mode *at matched diversity and reward*
than any other way of assembling a library — **even when the competing approaches are handed
state-of-the-art batch-retrosynthesis planners (SPARROW, MultiAiZ)** to find shared intermediates
after the fact. Forward construction beats post-hoc recovery. **This benchmark is the evidence for
that claim**, targeting a top-tier ML conference.

**Why S3-GFN is the marquee baseline (T3.1–T3.2).** A recent paper (S3-GFN, arXiv 2602.04119) argues
you *don't* need a reaction-based decision process — a plain SMILES generator with a synthesizability
penalty matches reaction-based models on *per-molecule* synthesizability. That is true but beside the
point: per-molecule synthesizability is not what sets a lab's cost — *library* co-synthesis is, and the
shared-hub structure that makes a library cheap only exists in a reaction-grounded model. S3-GFN emits
molecules with no shared-route structure, so batching them requires recovering routes post-hoc. The
headline figure shows S3-GFN + best planner still loses to a reaction-GFN + hub-batching on
reactions-per-mode. Everything in the plan is built so this one comparison is airtight and fair.

**Terminology (used precisely throughout — do not conflate the two "libraries").**
- **Chemistry library** = the *fixed* set of reaction templates + building blocks (`glue_standard_v1`,
  418 fragments + 112 templates). It never varies within a run. It is the vocabulary, not the product.
- **Library** (the thing being *built* and measured) = the set of molecules a scientist commits to
  synthesizing. Its size *is* its **mode count**. A **mode** = a molecule whose reward clears a
  per-target gate (sEH: `> 7`, higher-is-better) *and* that is Tanimoto-`< τ` (Morgan r=3) from every
  molecule already in the library. So the library is a curated, diverse, high-reward set, grown
  greedily one molecule at a time with that filter checked on every addition.

**The two stopping conditions (the "two companies" — the core experimental setup).** Both want the
same thing: many modes, few reactions. They differ only in when they stop:
- **Fixed-reaction budget (the HEADLINE):** "I will run 100 reactions — get me as many modes as
  possible." Sweep the diversity filter τ from 0.3 (strict — modes must be very dissimilar, so fewer
  fit) to 0.9 (lax — near-duplicates count, so more fit); plot modes achieved, per method.
- **Fixed-mode target:** "I need 300 modes — get them in as few reactions as possible." Same τ sweep;
  plot reactions required, per method.

**The two selection strategies (what a chemist actually does with a pool of candidates).**
- **best-candidate** (the naive baseline): sort the pool by reward, add the best one that passes the
  filter, repeat. Available to *every* generator.
- **hub-batching** (LSD-Flow): pick high-flow hubs (plus high-value shared fragments for SCENT), take a
  few filter-passing representative children of each hub, move to the next hub. Requires a
  reaction-grounded flow field, so it only applies to reaction-GFNs.

**One subtlety that shapes the architecture.** A strategy's output is *whatever it selected* — never
assume it is the reward-optimal or reaction-minimal subset. hub-batching may pick molecules that are
not the strongest binders and not a perfectly cost-minimal set; that is fine and expected. The
**evaluator scores the set the strategy actually chose.** This is why selection and evaluation are kept
as independent stages (§0): the strategy *tries* to be cheap using its own logic; the evaluator
(SPARROW/MultiAiZ, or count-once for the internal check) *independently* prices the result. When the
strategy's own cheap-ness (count-once) and the independent price (SPARROW) agree, the result is
trustworthy; the reconciliation gate (T1.5) enforces exactly this before any comparison is plotted.

---

## 0. The one-paragraph mental model

Three **orthogonal, pluggable stages** joined by two contracts:

```
   Generator ──pool──▶  Selection strategy ──ordered library──▶  Evaluator ──▶ (reactions, timing)
  (produces a set of      (best-candidate | hub-batching;          (count-once | SPARROW |
   scored molecules,       emits an ORDER, not a fixed set)         MultiAiZ→SPARROW; returns a
   + routes if any)                                                 global reaction count per set)
```

The **frontier driver** sweeps the filter cutoff τ, asks a strategy for an ordering, evaluates
**snapshots** of that ordering at a schedule of sizes, and reads the two company stopping-conditions
off the resulting (reactions, modes) curve. Selection cost-model and evaluation cost-model are
deliberately decoupled: the strategy *tries* to be cheap; the evaluator *independently* scores it.

**Definitions (parameterized, never hard-coded):** a *mode* = a molecule with reward past the
per-target gate (sEH: `> 7`, higher-is-better) that is Tanimoto-`< τ` (Morgan r=3) from every
already-selected member. τ sweeps 0.3→0.9; the mode-definition τ and the frontier-sweep τ are the
same knob. Reward gate value **and direction** live in a per-target config (`> 7` for sEH; docking
targets negate/transform — see §6).

---

## 1. Contracts to define first (Phase 0 — scaffolding)

These are the seams. Get them right and every later task is a small file.

### T0.1 — `Evaluator` protocol + `LibrarySet`/`EvaluationResult`
`validation/lsdflow/eval/base.py` (new).

```python
# spec, not implementation
@dataclass
class LibrarySet:
    smiles: list[str]                 # canonical
    rewards: dict[str, float]
    routes: dict[str, Route] | None   # step schema of glue/active_learning/route.py; None ⇒ route-less
    provenance: dict                  # generator, strategy, τ, seed, target

@dataclass
class EvaluationResult:
    total_reactions: int
    per_tool: dict[str, int]          # {"aizynth_routing": .., "sparrow_mip": ..} — always populated for debugging
    timing_s: dict[str, float]        # per stage; feeds the compute frontier
    per_molecule: dict[str, float] | None   # optional attribution

class Evaluator(Protocol):
    def score(self, library: LibrarySet) -> EvaluationResult: ...
```

**Acceptance:** interface imports clean; a no-op stub returns a well-formed `EvaluationResult`.

### T0.2 — `CountOnceEvaluator` (wraps what exists)
`validation/lsdflow/eval/count_once.py`. Thin adapter over the existing count-once cost
(`validation/lsdflow/metrics/cost/` + `validation/harness/cost.py::PathCostProxy`). Route-less
libraries ⇒ `total_reactions = None` (count-once is only defined on recorded routes).
**Acceptance (regression pin):** on a committed SCENT sEH snapshot, reproduces the current campaign
numbers (naive 2.73 / free-frag 1.22 rxn/mode) exactly. This pins the refactor.

### T0.3 — Make the frontier evaluator-agnostic
Extend `experiments/lsd_hubs/campaign/sweep_campaign.py`: add `--evaluator {count_once,sparrow,
multiaiz}`; the sweep calls an **injected** `Evaluator` on ordering snapshots instead of hard-coding
count-once. Do **not** fork a parallel frontier generator — reuse the existing plotting/readout and
the `pareto.{csv,png}` / `fixed_modes.{csv,png}` / `budget_efficiency.{csv,png}` outputs.
**Acceptance:** with `--evaluator count_once`, output plots/CSVs match the committed ones within
tolerance (bit-for-bit if the snapshot schedule is set to the current one).

> **Reconcile with the agent before writing T0.3:** the exact column schemas of `pareto.csv`,
> `fixed_modes.csv`, `budget_efficiency.csv` (independent variable + x/y). The plan assumes:
> `pareto.csv` = (τ, modes_at_100rxn) per strategy [HEADLINE]; `fixed_modes.csv` =
> (τ, reactions_for_300_modes); `budget_efficiency.csv` = (n_selected, cumulative_reactions).
> If the real columns differ, keep the driver's read-time slice logic and just remap names.

### Snapshot-and-read-time-slice logic (the frontier heart)
For each τ: strategy → ordering of length ≥ N_max; evaluate the chosen `Evaluator` on prefixes at a
size schedule (geometric or every-k) → stepwise (modes, reactions, timing) curve. Then:
- **Company A / headline (fixed reactions):** largest `modes` with `cumulative_reactions ≤ 100`.
- **Company B (fixed modes):** `cumulative_reactions` at `modes == 300`.

Budgets/targets are read-time constants (`--rxn-budget 100`, `--mode-target 300`), never selection
stop-conditions.

---

## 2. SPARROW native-chemistry evaluator (Phase 1 — MVP core)

MVP path is **AiZynth (per-molecule routes) → merge/dedup → SPARROW (batch MILP) → count**.
MultiAiZ is a later drop-in upgrade to the routing stage (§4), and must *earn* its slot by beating
AiZynth+SPARROW on shared-intermediate recovery — itself a reportable result.

### T1.1 — SPARROW env
`external/setup_sparrow.sh` (own conda env; github.com/coleygroup/sparrow). Smoke-test the MILP on a
toy 3-target network. **Acceptance:** SPARROW solves the toy and returns a route selection.

### T1.2 — Route-merge converter
`validation/lsdflow/eval/network.py`. Merge per-molecule routes (`routes.jsonl` step schema:
`reaction_idx`, `reaction_smarts`, `reactant`, `fragments`, `product`) into **one** reaction network,
dedup nodes by canonical SMILES, flag starting materials, attach glue-library fragment prices where
present (else default), attach per-target rewards. Emit SPARROW's input schema.
**Acceptance:** on a hand-built 3-molecule set sharing one intermediate, the merged network contains
that intermediate as a **single** node; a set with no sharing yields no spurious merges.

### T1.3 — Route-finding for route-less molecules
Wrap the existing, already-parameterized `validation/harness/synthesizability.py` (AiZynth;
`--config/--stock/--expansion/--filter` all selectable). Molecules with `has_route=0` (FragGFN,
S3-GFN, any SMILES generator) get an AiZynth route → feed T1.2; unsolved molecules are flagged and
excluded with a logged count (report solve-rate as a fairness stat).
**Acceptance:** a route-less molecule returns either a route or an explicit `unsolved`.

### T1.4 — `SparrowEvaluator`
`validation/lsdflow/eval/sparrow.py` (validation-side) + `validation/lsdflow/adapters/workers/
sparrow_worker.py` (runs in the SPARROW env). The evaluator: routed molecules use their routes;
route-less go through T1.3; T1.2 merges; subprocess to `sparrow_worker` (mirror the `scent_worker`
cross-env pattern; cross via `scripts/score_batch.py`/`ingest_candidates.py` conventions); parse the
MILP result into `EvaluationResult` with `per_tool` and `timing_s` populated.
**Acceptance:** returns `total_reactions` for a mixed (routed + route-less) library; timing attributed
to `aizynth_routing` vs `sparrow_mip`.

### T1.5 — UNIT-RECONCILIATION GATE ⛔ (blocks Phase 2)
Before any cross-method plot: run a **glue-constrained** SPARROW/count comparison on a reaction-GFN
library and confirm SPARROW's reaction unit agrees with count-once within a stated tolerance. (Native
SPARROW re-routes off the glue chemistry, so this reconciliation must use the glue-constrained AiZynth
config — see §5; it is a *config/data* task, not new code.)
**Acceptance:** report `|sparrow_glue − count_once| / count_once`; if above tolerance, **fix the
reaction-unit definition here**, not after the frontier is built. This is the promised cross-check,
promoted to a gate.

---

## 3. MVP end-to-end (Phase 2 — sEH × SCENT)

### T2.1 — Headline run
Frontier with `--evaluator sparrow`, native chemistry, existing SCENT sEH pool, **both** strategies.
Hub-batching uses the benchmark default **pre-select-K, `--prebuild-k 100 --child-policy free_frag`**
(set explicitly — the code default is naive; do not inherit it). Produce headline fixed-reaction
pareto + fixed-mode + cumulative plots.
**Acceptance:** best-candidate and hub-batching render on shared axes across τ∈[0.3,0.9]; hub-batching
dominates or the gap is quantified per τ.

### T2.2 — Compute frontier ("where does the time go")
Second frontier: reactions-saved vs compute-spent, with per-stage timing attributed —
`generation`, `child_enumeration` (`enum_timings.json`), `hub_selection` (`pick_hubs_timing.json`),
`library_selection`, `route_finding`, `sparrow_mip`. Emit a stacked per-method timing bar (longest
stage legible) and a **reactions-per-compute-hour** scalar (leave a `lab_hours_per_reaction`
multiplier hook = 1.0 for now; §8).
**Acceptance:** timing plot identifies each method's dominant stage; reactions/compute-hr emitted per
method. Include the baselines' own planning compute (SPARROW MILP, later MultiAiZ search) so the
comparison is honest end-to-end.

---

## 4. First external baseline + generality (Phase 3–4; 3.x is the marquee, 4.x parallelizable)

### T3.1 — S3-GFN generator (biggest single lift)
`external/setup_s3gfn.sh`; wire the **sEH oracle into S3-GFN's own training loop** (SMILES GFlowNet,
soft-synthesizability reg; arXiv 2602.04119). Emit a pool in the candidate-dataset format
(`manifest.json`+`candidates.csv`; `has_route=0`).
**Acceptance:** trains on sEH; emits a ≥95%-synthesizable SMILES pool ingested via
`scripts/ingest_candidates.py`.

### T3.2 — S3-GFN onto the frontier
S3-GFN pool → **best-candidate only** (no flow field ⇒ no hub-batching) → `SparrowEvaluator` (routes
found via T1.3). Matched reward-gate, τ-sweep, seed.
**Acceptance:** S3-GFN curve appears on the headline axes beside SCENT best-candidate and
hub-batching. This is the "MDP-necessary-for-library-economics" figure.

### T4.1 — MultiAiZ upgrade
`external/setup_multiaiz.sh` + `validation/lsdflow/eval/multiaiz.py` +
`.../workers/multiaiz_worker.py`: MultiAiZ discovers routes/shared intermediates (iterative stock
augmentation) → feeds the same SPARROW stage. Swaps into `--evaluator multiaiz` with **no
frontier-driver change**.
**Acceptance:** produces a frontier; report whether it lowers reactions vs AiZynth+SPARROW (does
smarter discovery help, or does SPARROW's MILP already capture the sharing?).

### T4.2 — Primitive generality
LSD-Flow hub-batching across **RGFN + RxnFlow** on sEH (adapters exist / stubbed → finish).
**Acceptance:** three reaction-GFN curves; RxnFlow's smaller payoff (weaker flow concentration) is
reported as an interpretable data point, not hidden.

### T4.3 — Target generality
Extend to DRD2 / 6TD3 / ClpP via config only (per-target gate + direction). Note: expensive docking
rewards *shrink* hub-batching's relative compute overhead (scoring dominates) — showcase this.
**Acceptance:** one non-sEH target runs end-to-end with only config changes.

### T4.4 — "What we leave on the table" ablation
Run SPARROW/MultiAiZ on LSD-Flow's **own** molecules vs its native hub-batching.
**Acceptance:** a single number for the gap (flow field near-optimal ⇒ strong; gap ⇒ honestly
quantified, still likely beats baselines).

### T4.5 — Replicates
3 seeds per (method × target); CIs/error bands on the headline. Start at 1, expand.
**Acceptance:** headline frontier with error bands.

---

## 5. The glue-constrained cross-check (data/config, not code)

AiZynth is already parameterized, so this is artifacts, not new plumbing: add
`data/models/aizynthfinder/config_glue.yml` with `stock` = the `glue_standard_v1` building blocks
(+ optional glutarimide-aware expansion policy), pass `--config/--stock`. Used by T1.5 (gate) and
available for the "what would reaction-GFN molecules score on native vs glue chemistry" curiosity run.
Sourcing/building the custom stock + policy artifacts is the only real effort.

---

## 6. Config surface (keep everything swappable)

One target/run config object carrying: `reward_gate` (value + direction), `tau_sweep` (0.3–0.9,
step), `rxn_budget` (100), `mode_target` (300), `child_policy`/`prebuild_k`, `evaluator`,
`chemistry` (native | glue), `snapshot_schedule`, `seeds`. No magic numbers in code. sEH ships as the
reference config.

---

## 7. Extension seams — architected-for, **do NOT build now**

Each is a named drop-in so a reviewer-ask is a small PR, not a redesign:
- **New generator** (REINVENT, GraphGA, **SyntheMol** — the non-GFN reaction-aware cell that would
  isolate *flow* from mere reaction-grounding): new pool adapter → best-candidate → existing evaluator.
- **New selection heuristic:** new `CampaignStrategy` (e.g., SPARROW-guided selection).
- **New cost model:** new `Evaluator` behind the T0.1 protocol.
- **New diversity x-axis** (measured mean-pairwise-distance or #Circles instead of filter-τ): new
  frontier x-column; the read-time slice logic is unchanged.
- **New chemistry:** swap planner config (§5).
- **Lab-hours conversion:** replace the `lab_hours_per_reaction` hook (§8).

State these in the repo's README as "supported extensions"; leave them unimplemented.

---

## 8. The cost-translation hook (retroactive)

Emit **reactions-per-compute-hour** now with a `lab_hours_per_reaction = 1.0` multiplier stub. When
order-of-magnitude bench figures exist later, one config change turns the compute frontier into
"~X CPU-hours of one-time enumeration buys ~N fewer bench reactions ≈ Y days / $Z saved" — the
concrete form of the time-is-the-real-cost thesis. Do not hard-code any lab-cost number.

---

## 9. Critical path & parallelism

**Serial spine:** T0.1 → T0.2 → T0.3 → T1.1 → T1.2 → T1.3 → T1.4 → **T1.5 (gate)** → T2.1 → T2.2 →
T3.1 → T3.2 (MVP complete = SCENT hub-batching vs SCENT best-candidate vs S3-GFN on sEH, native
SPARROW, headline + compute frontiers).
**Parallel once the gate clears:** T4.1 (MultiAiZ), T4.2 (RGFN/RxnFlow), T4.3 (targets),
T4.4 (ablation), T4.5 (seeds), T5 (glue artifacts).

## 10. Guardrails checklist (per task)
- [ ] No edits to `rgfn/` or `configs/` except `configs/glue/`.
- [ ] `validation/` never imported by the pipeline; heavy tools in own conda env, crossed by subprocess.
- [ ] `per_tool` + `timing_s` always populated (debuggability is a requirement, not a nicety).
- [ ] Budgets/targets applied at read-time; selection emits an ordering, not a fixed set.
- [ ] Ignore the stale `glue/chemistry/__init__.py` "non-functional stubs" docstring.
- [ ] pre-select-K set explicitly for benchmark runs.

---

# 11. EXTENSION (post-Phase-3): information-theoretic AL — the BALD / BatchGFN family

> **⛔ DO NOT BUILD until the Phase-3 spine (T3.2) is complete.** This section is a *plan*, written
> so it can be picked up cold. Everything §11 is **live active-learning** work, not fixed-pool
> benchmarking — it is the AL-loop counterpart of §0–§9 and its natural sibling doc is
> `docs/AL_PIPELINE_ARCHITECTURE.md`. It lives here because that is where the implementable
> T-task convention lives. **When implementation starts**, add cross-refs: a row in
> `AL_PIPELINE_ARCHITECTURE.md` §2's batching-paradigm table (information-theoretic batching) and a
> resolution note on its §7 **Q2** (retrosynthesis-batching's `U(h)`-analogue) — §11's ensemble
> posterior is the principled answer that question was deferred waiting for.

## 11.1 Objective and the claim

**What we are benchmarking against.** [`malik2023batchgfn`] (BatchGFN, ICML 2023 SPIGM workshop)
does pool-based active learning by training a GFlowNet whose *state* is a partially-built **batch**
and whose actions **add a pool point**, with reward `exp(I(y_{1:B}; θ)/T)` — the BatchBALD **joint
mutual information (JMI)**, computed in closed form under an exact GP: `½·log|I + σ⁻²K_B|`.
Its successor [`zhang2025baldgfn`] (BALD-GFlowNet, arXiv:2509.00704) makes acquisition *generative* —
a GFlowNet is trained to **produce** informative molecules rather than select them, so acquisition
cost stops scaling with pool size — and evaluates on JAK2 virtual screening (Enamine REAL).
**BALD-GFlowNet is the stronger comparison target** (it is the one actually run on molecules);
BatchGFN is the origin of the batch objective.

> **Read the two papers before building — they do NOT use the same objective.** BatchGFN uses
> BatchBALD's **joint** MI (batch-aware, submodular). BALD-GFlowNet uses **single-point BALD**
> `I(y; ω | x, D)` over an ensemble, *not* joint MI — so it inherits BALD's batch-redundancy
> weakness and offsets it with GFlowNet diversity plus a **multiplicative** composite reward
> `MI · TPSA · QED · SAS · Rings`. Our arm ladder covers both: `bald` (single-point, their setting)
> and `batchbald` (joint, BatchGFN's objective).

**These are not competitors on our axis, and the plan must say so plainly.** They optimize
information per *label*; a synthesis campaign is bound by information per *bench reaction*. Neither
has a cost model — every pool point costs the same to label. The claim §11 exists to support:

> BatchGFN-family methods optimize information per *label*. In a synthesis campaign the binding
> constraint is information per *reaction*. On matched oracle budget, JMI-optimal batches cost N×
> more bench reactions than hub-batching for comparable information.

**The second, equally important objective — and the reason this is instrumentation, not just a
baseline.** JMI is computed for **every arm's batch, every round**, including `random`. That turns
"are LSD-Flow's proposed batches actually information-rich?" into a measured number rather than an
argument, and it enables the study that validates the metric itself:

> **Does JMI predict campaign efficiency?** Rounds-to-N-modes is noisy; JMI is a per-round scalar.
> Do they agree — i.e. is JMI a usable *leading indicator* of how long a campaign takes to deliver
> N modes? A negative answer is publishable and is the honest thing to check before leaning on it.

## 11.2 Decisions already made (do not re-litigate)

| # | Decision | Rationale |
|---|---|---|
| 1 | **Live AL loop only.** No fixed-pool/post-hoc variant. | A log-det computed on a delivered library with no model being updated is a *kernel diversity* measure, not information. Rejected deliberately. |
| 2 | **Generator = SCENT. Target = 6TD3 differential.** | SCENT is the paper's spine; 6TD3 has the calibrated hit bar (−1.5), the config, and post-fix checkpoints (§11.3). |
| 3 | **`M` becomes a K-member deep ensemble.** | The loop's proxy is a single deterministic MPNN with *no* uncertainty — JMI/BALD is undefined against it. Ensembling `M` itself means the information we measure is information about the model we are actually training, and the generative arm (T6.7) needs uncertainty inside the reward path anyway. |
| 4 | **Both expected and realized information.** | Expected IG (pre-label, against the posterior on `D_{i-1}`) is the acquisition-time quantity and the *predictor* for §11.1's second study. Realized contraction (post-label) is the check. Their gap is a reportable calibration result. |
| 5 | **Arms at both levels**: selection (`bald`, `batchbald`) then generative (`bald_gfn`). | Selection-level arms are a prerequisite for the generative one (same posterior, same information module) and give a cheap intermediate checkpoint. |
| 6 | **Greedy BatchBALD, not a subset-GFlowNet.** | BatchGFN's own §4.2–4.3 reports it is *"on par with BatchBALD"* — its contribution is *amortizing* the greedy objective, not improving it, so greedy BatchBALD **upper-bounds** it. State that in one cited sentence rather than building a GFlowNet over subsets that cannot exceed it. |
| 7 | **Primary information baseline is reward-weighted (composite).** | Pure BALD is exploration-only; hub-batching is reward-gated. Comparing them head-to-head on hits beats a strawman. Direct precedent: BALD-GFlowNet's own reward is the **multiplicative** `MI · TPSA · QED · SAS · Rings`. Mirror that form (MI × a quality term); report pure BALD as a reference point. |
| 8 | **Downstream held-out metric is primary; JMI is the mechanism check.** | BatchBALD maximizes JMI by construction, so a JMI-primary headline can only ever report a ratio. Held-out AUROC/BEDROC is task-level and rigged for neither side. This finally builds `LSD_FLOW_PROPOSAL.md` §11's planned-never-built Information metric. |
| 9 | **Accept the ensemble tax and report it.** | Hub-batching scores the most molecules of any arm (measured: **20,000 reward-gen calls/round**), so K=5 makes that ~100k proxy evaluations/round. `reward_gen_s` is already a published timing component; the tax falls hardest on the arm we advocate, which makes reporting it a credibility asset. **All arms must be re-run under the ensemble `M` — existing single-`M` results are not comparable.** |
| 10 | **Mode (AL definition)**: a molecule with a **real oracle label** clearing the calibrated bar (6TD3: `≤ −1.5`), Tanimoto `< 0.5` (Morgan r=3/2048) from every mode already counted, **cumulative over `D`**. | Oracle truth, not proxy prediction; applies identically to every arm including `random`; matches the existing mode machinery. Report the paper-comparable 0.7 cutoff alongside where cheap. |
| 11 | **Pilot before committing compute.** | 1 seed × all arms at full length, then size seeds/rounds/N from measured JMI variance (T6.9). |

## 11.3 Current state — verified on disk, not inferred from docs

Checked 2026-07-30. **`AL_PIPELINE_ARCHITECTURE.md` §6 calling the SCENT AL loop "in progress" is
stale** — it has run to completion:

- `validation/generators/scent/al_loop.py` implements `_ARMS = ("policy","random","hub_batching","best_candidate")`
  with warm-start + `ScentHubAcquisition`; `validation/configs/scent_seh_lsdflow{,_smoke}.gin` exist.
- **Completed runs**: `$SCRATCH/rgfn_runs/experiments/active_learning/scent_seh_lsdflow/{hub_batching,best_candidate,random}_seed42/`
  — 5 rounds × 100 oracle calls, all three arms, full `oracle_calls.csv` + per-round `dataset_round_*.csv`.
- **6TD3 prerequisites present**: `$SCRATCH/.../experiments/fixed_reward/scent_6td3_5k/seed{42,43,44,999}/train/checkpoints/guidance_models.pt`
  — post-fix sidecars, so `P_B` is exactly recoverable for warm-start (Logs/024).
- **The chosen cell needs one config**: `validation/configs/scent_6td3_lsdflow.gin`. By inspection
  `scent_seh_lsdflow.gin` is a **4-line overlay** on `scent_seh.gin`; the 6TD3 equivalent is the same
  overlay on the existing `scent_6td3.gin`. Config bring-up, **not new code**.

**⚠ Open dependency (researcher-owned, not this plan's work).** In those seed-42 sEH runs the
`random` arm **beats both learned arms on top-k** (−10.65 vs −9.06 hub_batching / −9.19
best_candidate; lower better). Partly not apples-to-apples — the learned arms enforce a Tanimoto-0.5
diversity filter and a hit bar, so top-k is structurally unfair to them — and it is n=1 seed with
`warm_start_policy=False` at 500 iters/round. A molecular-weight artifact was **ruled out** (random's
molecules are *smaller*: MW 557 vs 598/575). The researcher is investigating a possible AL-loop bug.
**§11 assumes that resolves.** Note that §11's instrumentation is itself an independent diagnostic
for it: if the learned arms' batches do not score as more informative either, that is evidence about
the loop rather than about the metric.

## 11.4 Tasks

### T6.1 — Give `M` a posterior: `EnsembleProxy`
`glue/proxies/ensemble_proxy.py` (shared math, production side) + the mirrored SCENT-side class in
`validation/generators/scent/proxy.py` (SCENT cannot `import glue` — its package is also named
`rgfn`; this file **already** deliberately duplicates `LearnedGlueProxy` for that reason, so extend
the existing pattern, do not invent a new one). K members, different seeds + bootstrap resamples,
`fit()` trains all K; `predict` returns the ensemble mean (this **is** the new `M`); new
`predict_members(smiles) -> (K, N)` exposes the raw member predictions. **K = 5** unless the pilot
says otherwise; note in the code that ensemble MI estimates are sensitive at small K.
**Acceptance:** K-member fit on the 6TD3 seed CSV; `predict_members` returns `(K, N)`; ensemble-mean
predictions match the single-model distribution; per-round fit wall-clock measured and reported
against GFN-training wall-clock in the same round.

### T6.2 — The information math: `glue/metrics/information.py`
A **pure** module in the shape of `glue/metrics/lsdflow_flow.py` / `uncertainty.py` — takes a `(K, N)`
member-prediction matrix, returns floats; knows nothing about RGFN internals or RDKit.
- `bald(preds) -> per-molecule MI`
- `joint_mutual_information(preds_batch, sigma2) -> ½log|I + σ⁻²Σ|` under the Gaussian/rank-K
  approximation to the ensemble posterior
- `greedy_batchbald(preds_pool, B) -> indices` — the cost-benefit greedy, (1−1/e) on a monotone
  submodular objective
- `realized_contraction(...)` — posterior entropy change after labels arrive

> **Placement note.** `LSD_FLOW_PROPOSAL.md` §3b sketched a single `validation/lsdflow/metrics/information.py`.
> Split it: **acquisition-time math lives in `glue/`** (the arms import it — one-way rule), the
> **downstream held-out evaluation** stays validation-side (T6.8).

**Acceptance:** rank-K covariance matches a dense reference on a synthetic case; JMI of B *identical*
molecules ≈ JMI of one (the redundancy penalty works); JMI monotone non-decreasing in B.

### T6.3 — Instrument **every** arm (the measurement, decoupled from acquisition) ★
The piece §11.1's second objective rests on. Per round, for the batch each arm *actually selected*,
compute and log **expected IG** (against the posterior on `D_{i-1}`, before labelling) and **realized
contraction** (after labels). Add `jmi_expected`, `jmi_realized`, `bald_mean`, `bald_max` to
`AcquisitionTrace.COLUMNS`; both loops (`glue/active_learning/loop.py`,
`validation/generators/scent/al_loop.py`) already forward an accounting object per round, so this is
**additive and arm-agnostic** — no arm-specific branching.
**Acceptance:** every arm **including `random` and `policy`** emits finite JMI every round;
`oracle_calls.csv` carries the columns; `validation/harness/acquisition_curve.py` plots them.

### T6.4 — Cross-env contract: ship the K member predictions
`enum_children.json` children and `records.csv` gain `preds` (K floats) beside the scalar `reward`.
**This is what makes BatchBALD computable across the conda boundary**: the ensemble covariance is
**rank-K and exactly reconstructible** from the `(K, N)` matrix, so we never ship an N×N matrix
(K=5, N=20,000 → 100k floats, trivial). Backwards compatible: absent `preds` ⇒ information arms
unavailable, existing arms byte-identical.
**Acceptance:** `scent_worker` emits `preds`; `select_acquisition.py` reconstructs the covariance and
matches an in-env reference computation bit-for-bit on a committed fixture.

### T6.5 — Selection-level arms: `bald`, `batchbald`
New entries in `_ARMS` (both loops) + `--arm` in `validation/lsdflow/select_acquisition.py`. The GFN
trains as usual; only batch selection changes. **Pool parity is load-bearing** — give these arms the
*same* candidate set the comparison arm sees, and state which in the run card (sampled terminals for
parity with `best_candidate`; the enumerated hub children for the strictest same-pool comparison).
Primary variant **reward-weighted** (decision 7); pure BALD/BatchBALD as reference.
**Acceptance:** both arms return a full-size batch; greedy BatchBALD's selected JMI ≥ BALD-top-B's on
the same pool (the batch-aware term earns its keep); a fixture where BatchBALD refuses near-duplicates
that BALD takes.

### T6.6 — Hybrid arm + the `U(h)` autopsy
Register `ensemble_bald` in `UNCERTAINTY_FNS` (`glue/samplers/lsdflow/hub/ucb.py`) so
`UcbHubStrategy` can score `z(reward) + λ·z(BALD(h))` instead of `z(reward) + λ·z(U(h))` — "LSD-Flow
structure, their exploration signal", a one-line registration plus a scorer.
**Plus the diagnostic:** per-hub scatter + Spearman ρ of `U(h)` against ensemble BALD, per round.
This answers *with a number* whether flow-variance `U(h)` was ever measuring epistemic uncertainty —
the open question left by the flow-consistency finding that `U(h)` is dominated by `Var[log P_F]`
(policy peakedness).
**Acceptance:** arm runs end-to-end; committed scatter + ρ per round; `hub_batching` with
`uncertainty_fn='flow_variance'` is **byte-identical** to today (no regression).

### T6.7 — Generative arm `bald_gfn` (the real BALD-GFlowNet) — gated on T6.5
Replace the GFN's training reward with the composite information reward, so the model learns to
*produce* high-information molecules. The ensemble must be callable per-molecule every training step;
the seam is a `Reward`/proxy wrapper, **not** a loop change.
**Acceptance:** trains without collapse; its pool's mean BALD exceeds the reward-trained policy's;
per-iteration cost measured against the K=1 baseline. **Caveat to carry into the write-up:** this arm
optimizes a different objective, so it is not comparable to the others on reward — compare it on
information and on modes.

### T6.8 — Downstream information metric (validation side)
`validation/lsdflow/metrics/information.py`: after each round, evaluate the refit `M` on held-out
**known actives + property-matched decoys** — AUROC, BEDROC, regression error, calibration
(over/under-estimation). This is `LSD_FLOW_PROPOSAL.md` §11's Information metric, finally built.
Sets exist and are wired by config: 6TD3 `experiments/oracle_validation/docking_6td3/`
(**160 knowns + 248 decoys**), ClpP (183+183), sEH (2,315+2,315 proxy / 1,000+1,000 docking).
**State the power limitation:** 6TD3's set gives wide AUROC CIs and an underpowered BEDROC at small
α — if a second target is run, sEH is the well-powered one.
**Acceptance:** per-round held-out metrics CSV per arm; curves render alongside the oracle-call curve.

### T6.9 — Pilot (sizing, not evidence)
Build `validation/configs/scent_6td3_lsdflow.gin` (the 4-line overlay, §11.3), warm-start from
`scent_6td3_5k/seed42`. Run **1 seed × all arms at full length**. Measure: round-to-round JMI
variance, whether and when N modes is reached, per-round wall-clock **with** the ensemble tax.
**Acceptance:** all arms complete; a written sizing recommendation (seeds × arms × rounds, and the
value of N) recorded in the Logs entry **before** the full campaign is submitted.

### T6.10 — Full campaign + the two headline analyses
1. **JMI per arm per round** — are LSD-Flow's batches information-rich, measured not asserted.
2. **The predictive study** — does cumulative/mean JMI rank-correlate with rounds-to-N-modes?
   Report ρ with CIs **and the campaign count n stated in the caption**; with a realistic arm × seed
   grid n is small, so the honest readout may be "directionally consistent, n=18", not a hard claim.
**Acceptance:** both figures; an explicit verdict on whether JMI is a usable leading indicator,
including if the verdict is negative.

### T6.11 — Side validation on the completed sEH runs (NOT a gate)
Offline, zero GPU: refit the ensemble on each stored `dataset_round_*.csv` (= `D_{i-1}`) from the
existing seed-42 `scent_seh_lsdflow` runs and score each arm's round-*i* batch.
**Acceptance:** a JMI-per-arm table over those runs, labelled explicitly as an **offline** posterior
(not the one that was live in the loop) and reported as a consistency check, never as evidence.

## 11.5 Compute envelope

The binding cost is **GFN training, not docking** — with the GPU oracle, docking is <2% of the loop
(Logs/014). Anchor: the RGFN LSD-Flow AL config runs 150 iters/round at ~13 s/iter × 10 rounds ≈
**~6 GPU-h/campaign**; the SCENT×6TD3 figure is T6.9's job to measure (its sEH sibling uses 500
iters/round × 5 rounds). Consequence: **more seeds means more GFN training** — a cheaper oracle buys
almost nothing. Budget the ensemble tax on top: K× the per-round proxy fit, and K× on reward-gen
child scoring (hub_batching: 20,000 → ~100,000 proxy evaluations/round at K=5).

## 11.6 Stated assumptions and risks
- **AL-loop correctness** (§11.3) is assumed resolved. If it is not, §11's numbers inherit it.
- **K=5** ensemble; MI estimates are sensitive at small K. Pilot may revise.
- **Gaussian / rank-K approximation** to a NN ensemble's joint posterior. A Tanimoto-GP cross-check
  (exact JMI, matching BatchGFN's own setting) is the named fallback if a reviewer challenges that
  the arm ranking is an artifact of the posterior approximation — **architected-for, not built**.
- **6TD3's held-out set is thin** (160/248) — see T6.8.
- **All arms re-run under ensemble `M`**; pre-ensemble AL results are not comparable.

## 11.7 Guardrails (in addition to §10)
- [ ] Acquisition-time information math in `glue/`; held-out evaluation in `validation/`. One-way rule holds.
- [ ] JMI logged for **every** arm, including `random` — measurement is decoupled from acquisition.
- [ ] Cross-env boundary carries `(K, N)` member predictions, never a dense covariance.
- [ ] Existing arms byte-identical when the new knobs are left at their defaults.
- [ ] Greedy BatchBALD's status as an **upper bound** on BatchGFN is stated with a citation, not implied.
- [ ] Cross-refs added to `AL_PIPELINE_ARCHITECTURE.md` §2 (paradigm table) and §7-Q2 when work starts.
- [ ] `malik2023batchgfn` (Malik, Lahlou, Jesson, Jain, Malkin, Deleu, Bengio & Gal, arXiv:2306.15058)
      and `zhang2025baldgfn` (arXiv:2509.00704) added to `Logs/references/references.bib` + its README.
