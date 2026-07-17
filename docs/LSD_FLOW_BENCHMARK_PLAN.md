# LSD-Flow library-efficiency benchmark — implementation plan

**For:** the coding agent (has the repo). **Not to be implemented beyond what each task states.**
**Prime directives:** *simplicity* and *modularity*. Every new piece is a drop-in behind a stable
interface; nothing edits `rgfn/` or `configs/` (except `configs/glue/`); `validation/` may import
`glue/`/`rgfn/` but never the reverse; heavy third-party code installs via `external/setup_*.sh` and
crosses conda-env boundaries by subprocess, never by import.

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
