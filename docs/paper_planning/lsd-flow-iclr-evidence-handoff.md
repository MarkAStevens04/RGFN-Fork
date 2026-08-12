# LSD-Flow → ICLR 2027: evidence handoff

**Written:** 2026-08-12. **For:** the agent who will draft the paper. **Status of this file:** it is
the *inventory and narrative plan*, not the paper. It does not contain paper prose on purpose.

**Read in this order:** this file → `docs/paper_planning/lsd-flow-publication-strategy.md` (reviewer
defence, theory budget) → `docs/RESEARCH_CONTEXT.md` §"How library cost is measured" + "Key
terminology" (definitions; **read these before touching any number**) → the individual `Logs/NNN_*.md`
entries cited per result below.

**Deadlines (verified 2026-08-12):** ICLR 2027 abstract **Sept 19, 2026**, full paper **Sept 24,
2026**. ~6 weeks. Everything in §8 is triaged against that.

**Locked decisions** (from the researcher, 2026-08-12 — do not re-litigate):

| Decision | Value |
|---|---|
| Venue | ICLR 2027 main track; Nature Computational Science as the adaptation target |
| Framing | **Metric is the axis, method is the mechanism** — lead with library economics, then show the flow field is what delivers it |
| Molecular-glue oracle (Logs 001–018) | **Not a section.** 6TD3 is one of four reward targets. Keep the NCS hook (§10) so it can be promoted to an application contribution without restructuring |
| Negative results | **Front and centre**, as a severe-testing section (§6) |
| Active learning / `U(h)` acquisition | **Out of scope for this paper.** The `U(h)` *diagnostic* stays (it is a finding); the AL loop, BALD/BatchGFN arms, and `docs/LSD_FLOW_BENCHMARK_PLAN.md` §11 do not appear |

---

## 1. The claim architecture

One sentence, three sub-claims. Every experiment below is filed under exactly one sub-claim. If a
result does not fit one of these, it belongs in the appendix or nowhere.

> **Claim.** Per-molecule synthesizability is not what sets the cost of a screening library —
> *co-synthesis* is; a reaction-grounded GFlowNet's trained flow field already identifies which
> intermediates to batch around, and no amount of post-hoc retrosynthetic planning over a
> route-less generator's output recovers what forward construction gives for free.

- **C1 — the axis.** Library cost is *reactions per distinct high-reward molecule* ("reactions/mode"),
  not per-molecule route-solvability. → §3 (definitions), §5.1, §6.2, §6.4.
- **C2 — the mechanism.** The flow field, read post-hoc with no retraining, selects the intermediates
  that make a library cheap; ordering, filters, and the concentration behaviour of a cost-only
  optimizer all show the *spreading across intermediates* is the load-bearing part. → §5.3, §5.4.
- **C3 — the generality.** The effect is a property of reaction-grounded generation, not of one model,
  one target, one scoring function, or one threshold: 4 generators × 4 targets, surrogate **and** real
  GPU docking, 380/380 threshold-grid cells, 42/42 docking gate points. → §5.2.

**The ICLR hook.** `[kim2026s3gfn]` (S3-GFN) argues a reaction MDP is unnecessary — a plain SMILES
generator with a synthesizability penalty matches reaction-based models on *per-molecule*
synthesizability. That is true and beside the point. This paper is the direct answer: **the decision
process matters at the library level**, and we measure it against S3-GFN handed the strongest
available post-hoc planning stack. Frame the paper as engaging that claim, not as proposing a
heuristic. (Titles in that spirit: "Reaction MDPs earn their keep at the library level";
"Synthesizability is per-molecule, cost is per-library".)

**Naming — open decision.** "LSD-Flow" = late-stage diversification. The pun is load-bearing for the
construct-validity argument (§6.4) but the acronym reads as the drug on first contact. Decide before
drafting; if it changes, the mechanism name ("hub-batching") is unaffected.

---

## 2. What the method actually is (be precise; reviewers will check)

Implementation, in dependency order. Paths are repo-relative to
`/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

1. **Sample.** 30,000 trajectories from a trained, frozen reaction-GFlowNet checkpoint. No retraining,
   no reward modification. Cross-conda-env by subprocess: `validation/lsdflow/adapters/workers/{rgfn,scent,rxnflow,fraggfn}_worker.py`.
2. **Recover state flow at pre-terminal states.** Detailed balance (`[malkin2022trajectorybalance]`
   Eq. 7) rearranged: `F(h) = F(x)·P_B(h|x)/P_F(x|h)` with `F(x) = R(x)/P_F(stop|x)`. All in log space.
   `glue/metrics/lsdflow_flow.py`. **The shipped hub estimator is the max over *sampled* children**
   (`pick_hubs`), which is the near-unbiased one — see §6.3, this matters and is easy to misreport.
3. **Rank and select 200 hubs** by that estimate. `glue/samplers/lsdflow/hub/` (6 strategies + registry).
4. **Enumerate exhaustively** every one-reaction child of each selected hub and score it with the
   target's reward. `glue/samplers/lsdflow/rgfn_enumerate.py` + per-env workers.
5. **Walk hubs in rank order, building a library greedily.** Per hub, a *child policy* picks which
   children to take (`glue/samplers/lsdflow/child_select.py`: `reward` = naive; `free_frag` = keep only
   children whose added fragment is already stocked); a *mode selector* accepts a child only if it
   clears the reward gate and is Tanimoto-`< τ` from every already-accepted member
   (`glue/samplers/lsdflow/mode_select.py`). `pre-select-K` pre-stocks the top-K highest-fan-out
   fragments up front and charges for them (Logs/037).
6. **Price with one cost model applied identically to both strategies** — *count-once*: each molecule's
   assembly couplings charged once, each distinct promoted library fragment built once (closure under
   nesting). `glue/samplers/lsdflow/campaign.py` docstring is the authority;
   `validation/lsdflow/metrics/cost/dynamic_amortization.py` is the model.
7. **Baseline (`best_candidate`).** Sort the generator's own sampled molecules by reward, greedily
   accept modes, and — this is the fairness fix that killed the original inflated result — *credit it
   for hubs its picks accidentally share* (`MostSharedAssignment`, a greedy upper bound on realistic
   savings). Logs/033.

**Two things to state in the method section because they are unusual and defensible:**
- The strategy's output is *whatever it selected*; the evaluator prices it independently. Selection
  cost model and evaluation cost model are deliberately decoupled, and the reconciliation gate
  (§5.1) is what licenses comparing them.
- FragGFN is a **cost-model control, never a peer**: its "reactions" are fragment attachments, not
  synthesis steps, and its molecules carry `has_route=0`. Every table that includes it must say so.

---

## 3. Definitions that must be copied exactly, not paraphrased

Source of truth: `docs/RESEARCH_CONTEXT.md` §"Key terminology". Getting any of these subtly wrong is
the single most likely way to produce a wrong paper.

| Term | Definition |
|---|---|
| **Chemistry library** | The *fixed vocabulary*: `glue_standard_v1` = 418 building blocks + 112 reaction templates (SCENT's SMALL library). Never varies within a run. |
| **Library** | The thing being built and measured: the set of molecules a chemist commits to synthesizing. Its size **is** its mode count. |
| **Mode** | One molecule that (a) clears a per-target reward gate **and** (b) is Tanimoto-`< τ` (Morgan r=3 / 2048 bits) from every mode already accepted, grown greedily best-reward-first. Default τ = 0.5. |
| **reactions/mode** | count-once total reactions ÷ modes. Lower is better. **Primary metric.** |
| **count-once** | Our cost model. Assembly couplings once + each distinct promoted fragment built once. **Undefined on route-less generators** — this is why any cross-generator comparison must go through SPARROW. |
| **Hub** | A pre-terminal intermediate several finished molecules descend from. Build once, branch off it. |
| **Native vs from-scratch** | *native* = the recipe our generator already used (free). *from-scratch* = a planner re-deriving a recipe for a finished molecule. **Which one you price with can reverse a head-to-head** (Logs/047, and §5.1). |
| **SPARROW-Verifier (SV)** | SPARROW as auditor: given a library we already chose + its recipes, find the cheapest way to make *all* of it (`constrain_all_targets=True`, reward weight 0). We *expect* agreement. |
| **SPARROW-Batching (SB)** | SPARROW as competitor: given a pool + a reaction budget, choose *which* molecules to make. **No diversity term.** Always name the pool with it (`BC-SB`, `S3GFN + SB`). |
| **The two stopping conditions** | Fixed-reaction budget ("100 reactions — how many modes?") and fixed-mode target ("300 modes — how many reactions?"). Both read off the same ordering at read time; neither is a selection stop condition. |

**Per-target reward gates** (value *and direction*): sEH `> 7.0` (calibrated useful range 5–6, Logs/034);
DRD2 `> 0.5`; ClpP `≤ −8.0` kcal/mol raw Vina (AUROC 0.895, receptor human ClpP 7UVU, Logs/045);
6TD3 `≤ −2.0` on the Tier2−Tier1 differential (**provisional, no AUROC calibration** — say so).

**Trap that has already burned us twice:** for docking cells, `reward` is the RAW energy the bar is
measured in (lower better); `log_reward` is `beta·clip(...)` and is never negative. Gating on
`log_reward` qualifies nothing, silently. `log_reward` is also not comparable across generators.

---

## 4. What exists, what is running, what does not exist

**Trained generators.** 4 generators (RGFN, SCENT, RxnFlow, FragGFN) × 4 targets (sEH, DRD2, ClpP,
6TD3), 5,000 iterations, reaction cap 4, shared `glue_standard_v1` library. Logs/030.

**LSD-Flow evaluation cells.**

| Half | Cells | Status |
|---|---|---|
| Surrogate (sEH, DRD2) | 8 of 8 | **Complete.** Logs/050 + Logs/055 |
| Real GPU docking (ClpP, 6TD3) | 6 of 8 | **Complete.** Logs/058. Missing: `rgfn_6td3`, and `rgfn_clpp` |
| Two-knob surfaces (9 bars × 13 cutoffs) | 4 generators, sEH only | **Complete.** Logs/055 |
| External head-to-head (vs S3-GFN) | sEH only, seed 42 | **Complete.** Logs/056 |
| Non-flow competitor arms (BC-SB, BC-Enum-SB) | sEH | **Complete.** Logs/059 |

**In flight right now** (check before quoting; ask the researcher for current state):
- `rgfn_clpp` **seed 43** docking cell on Trillium — was 6/16 slices, 78/200 hubs, 62,578 children at
  last report. Completing it makes **ClpP the first target with all four generators**. Note the seed
  mismatch (43 vs 42 elsewhere) must appear in the caption.
- `rgfn_6td3` seed-42 training chain on Balam (was 2730/5000).
- `scent_clpp` re-enumeration at `ENUM_MAX=20000` to repair the child-truncation described in §6.5.

**Does not exist and will not without new compute:** S3-GFN on any target but sEH (needs retraining,
not re-planning); error bars on the competitor's side; more than one seed on any docking cell;
hub-depth ablation.

---

## 5. The positive results, by sub-claim

Every number below is traceable. Cite the log entry, not this file.

### 5.1 The external head-to-head — the headline (C1)

**Logs/056.** sEH, τ=0.5, gate 7.0, deliverable = 100 modes, both sides priced by SPARROW.

| pipeline | reactions for 100 modes | rxn/mode |
|---|---|---|
| S3-GFN(500) → MultiAiZ → **SPARROW selects** *(as actually run)* | **411** | 4.11 |
| S3-GFN(500) → from-scratch AiZynth → diversity-aware greedy | 269 | 2.69 |
| S3-GFN(500) → MultiAiZ → diversity-aware greedy *(**their best**)* | **235** | 2.35 |
| LSD-Flow, routes **re-derived from scratch** | 305 | 3.05 |
| **LSD-Flow native, free-frag K=0** | **131** | 1.31 |
| LSD-Flow native, free-frag K=20 | 135 | 1.35 |

→ **3.13×** as-run; **1.79×** against the competitor's strongest configuration. **Lead with 1.79×.**
Read the other way: at 131 reactions the competitor delivers 32 modes to our 100.

**The reversal is the story, not an embarrassment.** Under from-scratch re-derivation the ordering
flips — 305 (ours) vs 269 (theirs), i.e. we *lose*, which is what Logs/041 and Logs/048 reported. The
cause is entirely who does the retrosynthesis: AiZynth solved only 48.7% of our molecules because our
418 blocks are not in its catalogue, versus ~90% for ZINC-native S3-GFN (Logs/047; adding our blocks
to stock lifts us to 73.8%, ceiling ~92%). A chemist would never re-derive routes a generator already
handed them. Present both regimes and explain the flip — this is a construct-validity argument, and
`paper_pipeline_headline` already draws it that way.

**The competitor is not weak, and say so:** S3-GFN pool n=2,000, median surrogate score **8.03** vs
our 7.61, **99.7%** above the gate vs our 82.9%, route-solvable **95.6%** under its own catalogue.
Neither molecule quality nor planner failure explains the gap.

**Asymmetries — report, do not equalize:** candidate pool the selector saw 500 vs 26,069; surrogate
calls ~500 vs 155,764; route-planning compute 2.25 h (16 s/molecule, linear) vs **0**.

**The honest trade** (put this in the main text): S3-GFN is ~4× more mode-dense per candidate (~230
candidates contain 100 distinct molecules vs our ~1,010) while our molecules are ~2× cheaper each.
**We win on lab reactions and lose on candidates-you-must-score.**

**The cost model is audited, twice.** count-once vs an independent SPARROW MILP optimum on the same
molecules and same native routes (Logs/042 gate, Logs/049 + its 2026-08-04 update):

| target | strategy | count-once | SPARROW optimum | gap |
|---|---|---|---|---|
| sEH | hub-batching | 288 | 279 | 3.12% |
| sEH | best-candidate | 314 | 299 | 4.78% |
| DRD2 | hub-batching | 117 | 117 | **0.00%** |
| DRD2 | best-candidate | 364 | 346 | 4.95% |

→ **hub-batching sits 0.0–3.1% off provably optimal; the baseline 4.8–5.0%.** Our selection is
*closer* to what a global optimizer achieves than the baseline is, so essentially none of the
advantage is an accounting artifact. Also: count-once charges reactions we genuinely spent, so where
the two disagree our published numbers are **conservative**.

### 5.2 Generality (C3)

**a) Surrogate matrix, 8/8 cells** (Logs/050, Logs/055; 300-mode budget, τ=0.5):

| cell | best-candidate rxn/mode | hub rxn/mode | edge | hubs used |
|---|---|---|---|---|
| `rgfn_seh` | 4.000 (only **9** modes) | 2.993 (135 modes) | 1.34× | 92 |
| `scent_seh` | 3.383 | 1.303 | 2.60× | 57 |
| `rxnflow_seh` | 3.000 (83 modes) | 2.446 (186) | 1.23× | 135 |
| `fraggfn_seh` | 4.997 | 1.333 | 3.75× | 25 |
| `rgfn_drd2` | 3.843 | 1.197 | 3.21× | 24 |
| `scent_drd2` | 3.673 | 1.160 | 3.17× | 28 |
| `rxnflow_drd2` | 2.935 (277) | 1.833 | 1.60× | 126 |
| `fraggfn_drd2` | 4.680 | 1.173 | 3.99× | 13 |

Apples-to-apples (naive child policy for everyone, no `free_frag`/pre-select-K, which only SCENT can
run): SCENT 1.22×/1.38×, RxnFlow 1.23×/1.60×, RGFN 1.34×/3.21×.

**b) Real GPU docking, 6/6 cells** (Logs/058). This is the result that closes the biggest gap between
claim and evidence — every prior number came from a millisecond surrogate.

| cell | bar | hub rxn/mode | best-cand rxn/mode | edge | best docked |
|---|---|---|---|---|---|
| `rxnflow_clpp` | −8.0 | 1.223 | 2.977 | 2.43× | −11.6 |
| `scent_clpp` | −8.0 | 1.290 | 3.420 | 2.65× | −14.2 |
| `fraggfn_clpp` | −8.0 | 1.093 | 4.840 | 4.43× | −13.5 |
| `rxnflow_6td3` | −2.0 | 1.193 | 2.977 | 2.50× | −6.224 |
| `scent_6td3` | −2.0 | 1.157 | 3.320 | 2.87× | −7.398 |
| `fraggfn_6td3` | −2.0 | 1.533 | 5.000 | 3.26× | −7.041 |

min 2.43× / **median 2.76×** / max 4.43×. **All twelve arms reached the full 300-mode budget**, so no
comparison is flattered by one side running out of material. Scale: **2,022,682 docked children**,
**412.5 GPU-hours**. Threshold sweep: 7 gates × 6 cells = 42 points, hub-batching leads at every one;
**35 of 42 are strictly like-for-like** (both arms full budget) — all 7 exceptions are ClpP at bars
stricter than calibrated, and in 3 of them only the *baseline* falls short, which favours us.
Cross-generator magnitudes are **not** comparable (the baseline's own cost differs by generator).

**c) The advantage does not depend on where the thresholds sit.** Logs/052 (1 generator, 206/206
comparable cells, 1.59–3.02×) generalized by Logs/055 to all four:

| cell | comparable cells | hub-batching cheaper | best cell | at (5.0, 0.5) |
|---|---|---|---|---|
| `rgfn_seh` | 71 | **71/71** | 3.81× | 2.37× |
| `scent_seh` | 109 | **109/109** | 3.25× | 3.19× |
| `rxnflow_seh` | 85 | **85/85** | 2.79× | 2.27× |
| `fraggfn_seh` | 115 | **115/115** | 4.93× | 4.20× |

**380 of 380 comparable cells, no crossover anywhere on either knob, for any generator.**
Pool-limited cells are hatched and excluded, never counted as wins.

**d) The headline bar was distorting the comparison — disclose this.** At the calibrated bar 5.0 the
edge grows (RGFN 1.34→2.32×, RxnFlow 1.23→2.32×, SCENT 2.60→3.10×, FragGFN 3.75→4.21×) *and* all four
cells reach 300/300 modes for both strategies, so every number is measured on the same library. At
7.0, two cells are pool-starved. Between-generator spread narrows from 3.0× to 1.8×: **most of the
apparent difference between models was the bar, not the model.** Choose the headline bar deliberately
and say why. Supporting mechanism: the baseline's *cost* is bar-invariant (`rgfn_seh` 4.000→3.920);
only its *yield* collapses (9 → 300 modes).

### 5.3 The mechanism is the flow field (C2)

**Hub-ordering ablation, six arms, everything but the ordering held fixed** (Logs/053, SCENT/sEH):
- Reverse the sort (lowest flow first) → the method **cannot build the library at all** (279/300
  modes, all 200 hubs exhausted) while burning the most compute.
- Random order completes but costs **1.53×** the reactions.
- The three arms that all draw on *good* hubs — flow over everything, flow over the reward-filtered
  pool, and ordering by best-molecule score — land within **9%** of each other.
- **Honest claim: flow finds the right *neighbourhood*, not the exact rank.** State it that way.
- **The two-axis result is the sharpest.** On (bench reactions, enumeration compute) every frontier
  point is flow- or reward-informed, and both no-signal controls are **strictly dominated**:
  flow-first 357 rxn / 5,648 s; best-candidate-order 389 rxn / 2,481 s (9% more reactions, **2.3×
  less compute**); random beaten on *both* axes; reverse-flow beaten by everything.
- **Why the best-candidate ordering does well, and it favours us:** a GFlowNet routes trajectories
  through high-flow intermediates, so parents of good sampled molecules *are* high-flow hubs.
  Measured: that arm never reads flow, yet its 200 hubs sit at median flow rank **546 of 20,874**
  (top 2.6%), 63% inside the top 1,000; random sits at 11,043 against a uniform null of 10,437.

**BC-SB / BC-Enum-SB: what happens if you just run a cost optimizer over our own molecules**
(Logs/059, sEH, 300-reaction budget). This is the ablation that answers "your generator makes the
molecules — why do we need your selection?"

| arm | candidates chosen | **distinct molecules** | reactions/candidate |
|---|---|---|---|
| hub-batching (ours) | 228 | **228** | 1.32 |
| BC-SB (n=3 seeds) | 266–279 | **47.7 ± 1.5** | ~1.08 |
| BC-Enum-SB (n=1) | 296 | **14** | 1.01 |

At 100 reactions: 82 / 12 / 2 distinct. **We lose on cost per molecule and must say so** — the claim
is specifically cost per *distinct* molecule. The mechanism is measured, not argued: at 100 reactions
BC-Enum-SB drew from **3** intermediates with **78% from one**, and its reactions/candidate is pinned
at ≈1.02 across every budget. Once an intermediate is built, each further child costs ~1 marginal
reaction, so a cost-only optimizer takes as many as it can from one place. **Enumeration alone is not
what makes reaction-grounded generation good for batch synthesis — spreading the selection across
many intermediates is, and that is what the flow ordering supplies.** BC-Enum-SB is the strongest
non-flow control we can construct (64 hubs from a reward walk, 131,474 enumerated children, 55%
overlap with the flow-ranked hub set); the solver handled the entire 21,001-molecule pool, so no
ceiling excuse is available.

### 5.4 Both halves of the metric are load-bearing, and they cost *us* more than the baseline

**Logs/054** (the numerator defence the strategy notes flagged as missing):
- Drop the reward gate → hub-batching looks **13% cheaper** while **91%** of the delivered library
  falls below the bar.
- Drop the diversity filter → looks **12% cheaper** while the 300-molecule library collapses to **39**
  genuinely distinct molecules.
- Priced per molecule worth keeping, those "savings" cost **9.8×**, **6.8×**, and **29×** with both off.
- A score-blind clustering agrees and is slightly harsher (144 families vs the 153 our own count
  reports) — so this is not an artifact of counting best-first.
- **The failure has a recognisable shape:** with either filter off, hub-batching stops walking the
  ranking and collapses onto a handful of shallow scaffolds (43 hubs → 5 → 2), and the typical library
  member drops from two nested reactions to one. **That is the depth-0 catalog degenerate optimum
  appearing empirically** — connect it to the construct-validity ladder in §6.4.
- The baseline is *exactly* indifferent to the reward bar (identical to 4 d.p. from "off" to 8.0), so
  the reward filter's entire effect falls on the method that could otherwise exploit its absence.

**Supporting: the metric is not riding a diversity artifact** (Logs/051). Raising the reward bar does
make survivors more alike, perfectly monotonically — but the effect is negligible below bar ~7 (5→6
costs 0.2–0.7% of distinct molecules; 5→7 costs 2.3–4.2%) and only turns steep past it (~42–44% loss
by 8.0). All headline comparisons sit at 5/6/7, inside the flat regime. The scaffold cross-check
*disagrees in sign* for the enumerated library, and the enumerated library is **more** internally
diverse than the model's own sample at every bar below ~7.6 — the opposite of the intuition that
children of 200 shared scaffolds must be redundant. Two independent measurements (Logs/051, Logs/052)
agree on where the useful operating range ends.

**Supporting: compute is the price, and it is measured, not modelled** (Logs/039, Logs/044, Logs/055,
Logs/058). Hub-batching costs ~150× more compute than best-candidate (~170 s vs ~1 s at τ=0.5) to
save ~113 bench reactions. **The component split flips with reward cost, which is a finding:** for the
millisecond surrogate, enumeration is 64–79% and reward-gen only ~5% (RxnFlow is the extreme —
**97.8%** of its bill is backward-policy flow extraction, a quantity its own hub ranking is
insensitive to); for real docking, reward-gen is **87.5–99.7%**. So **the more expensive the oracle,
the smaller hub-batching's relative overhead** — a point worth making explicitly for the docking half.
Also from Logs/041: when a lab prices a library with a from-scratch planner, **route search is the
entire cost** (12,000–13,000 s) and the set-cover MILP is **0.06 s** — a reaction-grounded generator
sidesteps almost all of it.

**Supporting: the batches are chemically sensible, not one giant step** (Logs/037, Logs/038). Batch
sizes are moderate and even (largest 65, Gini 0.37–0.44); pre-select-K is a clean consolidation dial
(K=16 is a small *net* win — 364 vs 367 reactions *and* 21% fewer oracle calls; K=100 → 1.41 rxn/mode
at 73% fewer calls); the pre-selected fragments are recognisable universal building blocks
(bromo-aryl amides/amines with azetidine/pyrrolidine), each usable in 130–175 of 200 hubs. Fan-out is
the load-bearing signal.

---

## 6. The severe-testing section (front and centre, per the locked decision)

Frame as: *we tried to kill the claim in six independent ways; here is what survived and what did
not.* Two of these overturned our own prior conclusions, and reporting them is the credibility asset.

### 6.1 Our own earlier cost model inflated the result, and we fixed it
Logs/029 claimed hub-batching "roughly halves" cost (1,479 → 820 reactions, ~1.8×). Logs/033 found
**most of that was a fragment double-count** in the baseline (530 of 550 reactions saved came from the
fix alone) plus a missing credit for accidental hub sharing. On the corrected single count-once model
applied to both strategies the naive-policy edge is **~1.13×**, not ~1.8×. Everything published since
uses the corrected model, and the large modern edges (2.4–4.4×) come from the *child policy* and the
calibrated bar, not from the accounting. **Never cite 1,479 vs 820 or the ~2× figure.**

### 6.2 The pricing regime can reverse the headline
See §5.1. Under from-scratch re-derivation we lose (305 vs 269). We report both regimes and explain
the cause (catalogue coverage, Logs/047). A reviewer who discovers this independently sinks the paper;
a paper that leads with it is much harder to attack.

### 6.3 `U(h)` does not measure what we designed it to measure
Logs/050, on the exhaustive enumeration. The per-child flow estimates for a single hub span **65 nats**
(min 48.5 / median 68.2 / max 113.5 over 1,799 children) where detailed balance says they should be
identical. The variance decomposition shows why: `Var[log P_F]` = 213.6 vs `Var[log F(x)]` = 23.4 vs
`Var[log P_B]` = 0.36 for RxnFlow — **`U(h)` is dominated by forward-policy peakedness, not epistemic
uncertainty about molecule quality**. The sampled version we would compute in a live loop is 40–55×
smaller than the enumerated value and undefined for most hubs. This killed the planned
uncertainty-driven acquisition before it was built. Two positive findings ride along: conservation
picks the weighting (`F(h)` is the **P_F-weighted** mean, which is why the unweighted mean is biased
+6.35 nats while the single-sampled-child estimator we ship is nearly unbiased, +0.43 / +0.28 nats);
and a hub's flow is carried by only **~18–23 effective children out of 1,161–2,195** (Kish), i.e.
hub-batching harvests children the policy itself would rarely sample.
**But the shipped estimator is too noisy to *rank*** — its own error spread (p5–p95 ≈ 4.8 nats) is as
large as the entire across-hub spread of true log-flow (4.9–6.5 nats), giving ρ ≈ 0.6. That is the
quantitative explanation for §5.3's "neighbourhood, not rank" and for Logs/025's puzzling near-tie
between flow ranking and a trivial control.

### 6.4 The extraction validates, but the trained models violate their own conservation law
Positive: summing the model's own forward probabilities over our enumerated children recovers
**0.999 / 1.000** of the available mass (0/200 hubs above 1, 0/200 below 0.01) — independent
confirmation that the enumeration is exhaustive *and* that we read the policy correctly.
Negative: the dimensionless identity `R(h)/(R(h)+N(h)) = P_F(stop|h)`, with both sides measured
independently, is violated at hubs by **−6.05 nats (≈430×) for RxnFlow and −14.03 nats (≈1.2×10⁶) for
SCENT** — both models terminate at hubs far more often than their reward justifies, and **the
violation is larger in the model whose backward policy is trained and exactly recovered**, so it is a
property of trained models, not of one model's approximations. The violation lives in the *stop*
channel, which the shipped estimator never touches for cap-truncated children — which is how that
estimator can be unbiased while conservation fails. Also: RxnFlow's learned `log Z` = 53.07 is *below*
its own maximum single-molecule reward (63.69), i.e. internally inconsistent and unusable as an anchor.
**This is genuinely interesting to an ICLR audience** — it is a measurement about what trained
GFlowNets actually satisfy at interior states, on two independent models, and it is not in the
literature. Consider promoting it from caveat to result.

### 6.5 One enumeration is not exhaustive, and the bias runs against us
Logs/058 audit. Hub coverage is exact in all six docking cells (200/200, zero duplicates, no partial
hub possible by construction). But `ENUM_MAX=4000` binds on SCENT, whose promoted library gives it far
more reachable last-step reactions: `scent_clpp` has 25 capped hubs retaining only **~10%** of
reachable forward mass (median 0.1039 vs 0.9945 uncapped), `scent_6td3` 51 capped at ~93%. Capped hubs
sit inside the walk (including rank 1). **Direction of the bias is toward understating our result** —
fewer children per scaffold means more hubs needed for 300 modes, i.e. higher reactions/mode — so
2.65× and 2.87× are conservative. This is reasoning from mechanism, not measurement; the re-enumeration
at cap 20,000 is in flight. It also forces an **interpretation correction to Logs/050**: its
low-forward-mass tail was attributed to genuine stop-probability at hubs the policy likes to terminate
at; it is the 4000-child cap. `scent_seh` (57 capped) and `scent_drd2` (31) carry the same truncation.

### 6.6 One prior experiment came out inconclusive and must not be cited
Logs/048 Exp A (fragments-only chemistry homogenization) is **inconclusive** and is not a
synthesizability claim. What it *did* establish is methodological: a fragments-only transfer cannot
homogenize a template-based synthesizability signal, so "give the baseline the same blocks" is not a
well-posed fairness test for S3-GFN (it *is* for AiZynth's broad USPTO templates, which is why
Logs/047 worked). Exp C is the solid result: MultiAiZ pulls our from-scratch cost 2.74 → ~1.85
reactions/mode but still cannot match native routes (1.22), and hub-batching stays ~1.5× cheaper than
best-candidate under *every* pricing regime.

### 6.7 Own the scaffold-collapse effect (framing, not a new result)
The advantage narrows as τ tightens. This is mechanistically predictable — hub children share
substructure by construction — and must be framed as *"batching around shared intermediates trades
structural diversity for reaction economy; we quantify the trade and report the regime where it pays"*,
never as *"we win everywhere, though margins narrow"*. The surfaces sweep τ 0.30–0.90 with **no
crossover anywhere**, so report that range explicitly rather than letting a reviewer suspect the sweep
was truncated before one.

**The construct-validity paragraph — take it, it is free and it is the strongest paragraph available.**
Walking the metric's low end recovers established medicinal chemistry at every depth: depth 0 =
fragment/catalog screening; depth 1 with diverse purchasable SMs = parallel & combinatorial synthesis;
the hub limit = late-stage diversification. A metric whose optimum recovers practices chemists adopted
for reasons entirely independent of this paper is measuring something real. Note the naming coherence
explicitly (the method is named for the practice its constrained optimum recovers). And keep the
distinction the strategy notes insist on: **depth-0 catalog picking is the true degenerate optimum**
(name it openly as the metric's failure mode; the reward gate is what closes it, per §5.4), while
**one hub with maximally diverse children is the *constrained* optimum** — the best achievable point
conditional on actually synthesizing toward a target. Do not call the latter degenerate.

---

## 7. Proposed figure set

Committed assets are real paths; verify before use. Every one has a `.pdf` beside the `.png` unless
noted.

| # | Figure | Source | Status |
|---|---|---|---|
| **F1** | **Headline**: same 100-mode deliverable, two pricing regimes, LSD-Flow vs S3-GFN+MultiAiZ+SPARROW; panel B = SPARROW's selections are less diverse than its pool | `experiments/lsd_hubs/campaign/results/paper_pipeline_headline/pipeline_headline.{png,pdf,csv}` | **Publication-ready.** Needs: our 3-seed band (123.3 ± 2.1), and a note that the competitor is n=1 |
| **F2** | **Generality**: reactions/mode, hub vs best-candidate, per generator × target | `experiments/lsd_hubs/matrix16/results/generality_panel/generality_panel.{png,pdf,csv}` | **Stale** — shows FragGFN cap-9; regenerate with cap-6 and add the 6 docking cells |
| **F3** | **Real docking**: 6-cell head-to-head + the 42-point gate sweep | `matrix16/results/<cell>/summary.json`, `matrix16/results/gate_curve/<cell>/gate_curve.{json,png}` | Data complete; **panel figure not yet drawn** |
| **F4** | **Two-knob surfaces**, 4 generators, hatched pool-limited cells, ★ at the operating point | `matrix16/results/surface/<cell>/surface.{png,pdf,csv}` | **Publication-ready** (4 separate figures; consider a 4-panel composite) |
| **F5** | **Mechanism**: hub-ordering ablation on the two cost axes — every frontier point flow/reward-informed, both no-signal controls dominated | `experiments/lsd_hubs/hub_order/results/comparison/{cost_pareto,cost_vs_cutoff,compute_vs_cutoff,hub_rank_distribution}.png` | **Publication-ready** |
| **F6** | **Cost-only optimizers concentrate**: distinct-molecule counts for hub-batching vs BC-SB vs BC-Enum-SB + the intermediate-concentration table | Logs/059 tables; `$SCRATCH/rgfn_runs/lsdflow_sparrow/bc_sb/**` | **Not yet drawn** |
| **F7** | **Filters are load-bearing**: factorial ablation, cost per *keeper* | `experiments/lsd_hubs/filter_ablation/results/scent_seh/filter_ablation{,_factorial}.{png,pdf}` | **Publication-ready** |
| **F8** | **Compute frontier**: reactions saved vs compute spent, per-stage attribution; the surrogate-vs-docking flip | `campaign/results/scent_seh_sparrow_headline/compute_frontier.png`; `matrix16/results/<cell>/compute_time.{png,csv}` | Ready; needs the docking/surrogate contrast drawn as one panel |
| **A1** | Chemistry gallery: closest still-distinct pair at each τ, with route trees — makes τ interpretable to a chemist | `campaign/results/scent_seh_1kx200/diversity_pairs/{gallery,route_trees}.png` + `schemes/scheme_cut*.png` | Ready. **Strong appendix asset**, possibly main text for NCS |
| **A2** | Per-hub batch-size distribution (rebuts "one giant synthesis step") | `campaign/results/scent_seh_batch_dist/{batch_distribution,batch_ranksize}.png` | Ready |
| **A3** | Reward-cutoff vs intrinsic diversity | `experiments/lsd_hubs/reward_diversity/results/` | Ready |
| **A4** | Cross-generator τ sweep at both bars, incl. the FragGFN control | `matrix16/results/tau_curve_all/{seh,seh_gate5,seh_with_fraggfn_control}/` | Ready |
| **A5** | Severe-test tables: forward-mass closure, estimator bias/ρ, variance decomposition, conservation violation | Logs/050 §Results | Tables exist in the log; **not yet figures** |

---

## 8. Gaps, triaged against Sept 24

**Must have — the paper is attackable without these**

1. **A second seed on the docking cells.** Every docking ratio is n=1 seed, n=1 checkpoint. The sEH
   native side already has 3 seeds (123.3 ± 2.1 reactions for 100 modes). Cheapest partial fix:
   report the 3-seed sEH band prominently and state the docking cells as single-seed in every caption.
2. **Finish `rgfn_clpp` seed 43** (in flight) → ClpP becomes the first target with all four generators.
   Caption the seed mismatch.
3. **Regenerate F2** with the cap-6 FragGFN model and the docking cells folded in.
4. **`fromer2024sparrow` is missing from `Logs/references/references.bib`** despite being cited as a
   key in Logs/042/049/056/059. Add it (and check the MultiAiZ / S3-GFN entries render).
5. **Draw F3 and F6** — the two most load-bearing results with no figure.
6. **Decide and document the headline reward bar** (5.0 vs 7.0) with the §5.2d reason stated.

**High value and feasible in 6 weeks**

7. **Run the competitor head-to-head on a docking target** (ClpP or 6TD3). All machinery exists: pool →
   MultiAiZ discovery (~16 s/molecule, so ~2.25 h for 500) → SPARROW selection → mode count. This is
   the single highest-value remaining run: it turns the external comparison from "one surrogate target"
   into "surrogate + real docking", and per Logs/058's own next-steps it is the named gap.
8. **Mode-definition robustness**: re-count modes with Butina / sphere-exclusion clustering and a
   second fingerprint. Pure CPU. Logs/054 already did the score-blind clustering cross-check (144 vs
   153 families), so this is an extension, not a new build. The strategy notes call raw max-Tanimoto
   mode counting fragile in this subfield — pre-empt it.
9. **Normalize mode counts for library size** where compared methods emit different numbers of
   molecules (strategy notes §2.3; Logs/059's raw-count caveat is the same issue).
10. **Replicate BC-Enum-SB** (n=1 today). Logs/059 says the expensive part is done and the sweep takes
    minutes — bookkeeping, not compute.
11. **Vendor-price translation** of reactions/mode (catalog reagent counts → plate counts → dollars).
    Cheap, and a metric that converts to money is much harder to dismiss as invented-to-win. The
    `lab_hours_per_reaction = 1.0` hook already exists.

**Known holes to declare in Limitations rather than fix**

12. **Competitor error bars.** S3-GFN is one training run and one planning run; ours has three seeds.
    The ratio has error bars on one side only. Replicating needs retraining, not re-planning.
13. **S3-GFN on a second target** — same reason.
14. **Hub-depth ablation** (why pre-terminal only? why not depth-2?). Named in the strategy notes as a
    free criticism. A full version needs new enumeration (expensive). **Cheap substitute available:**
    report the depth distribution of *selected* hubs (Logs/053 measured flow preferring shallow hubs,
    11 depth-0 / 150 depth-1) and argue the pre-terminal restriction is exactly what makes the final
    step cheap; state the deeper-hub question as future work.
15. **FragGFN's cost is not on a comparable footing** (attachments ≠ synthesis steps). Fixable by
    routing its molecules through retrosynthesis; until then it is labelled a control everywhere.
16. **The 6TD3 bar is provisional** (no AUROC calibration, unlike ClpP's −8.0).
17. **`scent_clpp` child truncation** (§6.5) — re-enumeration in flight; if it does not land, report
    the audit and the bias direction.
18. **Route-choice asymmetry.** SPARROW gets ~16 alternative recipes per molecule on the competitor's
    side and **1.07** on ours (only 6.3% of our molecules can be made more than one way), because
    `routes.json` keeps one recipe per product. The optimizer has *more* freedom on the competitor's
    side, so this cuts against us — say so. Fixing it needs multi-trajectory logging at sample time.

---

## 9. Reviewer objections and the current answer

| Objection | Answer | Where |
|---|---|---|
| "You compared against your own ablations." | Full external pipeline: S3-GFN + MultiAiZ + SPARROW, competitor's own catalogue, optimizer allowed to select. 1.79–3.13×. | §5.1 |
| "Your generator makes the molecules; why do we need your selection?" | BC-SB and BC-Enum-SB, both given our molecules (and our enumerated children); 5× and 16× fewer distinct molecules at matched budget, with the concentration mechanism measured. | §5.3 |
| "reactions/mode is your own metric and you win on it." | Numerator *and* denominator defended: filter ablation (dropping either flatters us and costs 9.8×/6.8× per keeper), score-blind clustering agrees, two-knob surface has no crossover in 380/380 cells, reward–diversity coupling measured and negligible in the operating range, construct-validity ladder recovers established practice. | §5.4, §6.7 |
| "Post-hoc heuristic on existing models — thin contribution." | Works on 4 independently trained backbones with no retraining (that is the *point*); theory makes the extraction principled (§11); and the severe tests are real findings about trained flow fields. | §6.4, §11 |
| "Surrogate reward, not real physics." | 6 real-GPU-docking cells, 2 protein systems, 3 generators, 2.02M docked children, 42/42 gate points. | §5.2b |
| "Your cost accounting is self-serving." | Independent SPARROW MILP optimum: we are 0.0–3.1% off optimal, the baseline 4.8–5.0%. Where the models disagree, ours is conservative. | §5.1 |
| "Hub definition is arbitrary." | Ordering ablation done (reverse breaks it, random costs 1.53×, good orderings agree within 9%, both no-signal controls dominated). Depth ablation **open** — see gap 14. | §5.3, §8 |
| "The oracle is meaningless, so diversity does not matter." | sEH surrogate enrichment measured honestly (AUROC 0.76 vs random molecules, **0.68** vs property-matched decoys; "modes ≥ 8.0" is *not* an activity claim) — which is precisely why broad diversity is the mechanism for finding hits under an unreliable oracle. ClpP is calibrated (AUROC 0.895). | Logs/034, Logs/045 |
| "Why not just a bigger pool for SPARROW?" | It solved the entire 21,001-molecule pool; no ceiling was reached. MILP is superlinear (~1 s at 500 targets, >110 s at 2,000). | Logs/059 |

---

## 10. The Nature Computational Science adaptation hook

Do not restructure for it now; keep these seams so the switch is additive:

- **Promote the glue-oracle work to a contribution.** Logs/001–018 contain a validated, novel oracle:
  the two-tier **neosubstrate differential** (Tier2 − Tier1 on the same docked pose), which isolates
  the ligand arm's contribution once the recruited partner is present. On 6TD3/CR8-cyclin K it gives
  85.6% vs 7.3% separation of known glues from warhead-matched decoys (+78 pp), AUROC **0.946**,
  survives MW-matching (0.95 → 0.87) where absolute scores collapse (Vina Tier 1 → 0.38), and the
  CNN-vs-Vina pose-selection ablation is done (0.946 → 0.795 under Vina selection). The CRBN/5HXB
  ceiling (−3 pp) is a *structural* negative result worth reporting: docking cannot see PPI-driven
  neosubstrate recognition.
- **Keep the depth-ladder / construct-validity paragraph and the chemistry gallery (A1)** — both read
  better to a chemistry audience than to an ML one.
- **The dollar/plate translation (gap 11)** is the natural NCS headline unit.
- **Say plainly** what is and is not validated: everything is in silico; ground truth is docking, not
  affinity.

---

## 11. Theory footprint (keep under half a page of main text)

From `lsd-flow-publication-strategy.md`, unchanged and still the plan:

- **Lemma (not Theorem)** — the flow identity `F(h) = F(x)·P_B(h|x)/P_F(x|h)` is detailed balance at
  edge h→x, rearranged. Calling it a Theorem reads as overclaiming.
- **Corollary (highest-value item, not yet written)** — if the log-space detailed-balance residual is
  bounded by ε on every edge, per-child estimates of `log F(h)` agree within 2ε and the induced hub
  ranking is exact whenever the true log-flow gap exceeds 2ε. This links reported training loss to
  whether the hub ranking can be trusted, and pre-empts "your GFlowNet is approximate, so why believe
  the hubs?" **Note it must be written against what §6.3 measured** — the residual is *not* small in
  our models, so the honest statement is the ranking-error bound, which is exactly why the claim is
  "neighbourhood, not rank".
- **Reward-mass interpretation** — state that state flow equals the reward mass of terminal descendants
  weighted by backward reachability, so "sort hubs by flow" is "sort hubs by captured reward mass".
  Cheapest theory-to-value conversion available.
- **Proposition** — hub selection under a reaction budget is budgeted maximum coverage (NP-hard); the
  objective is monotone submodular, so cost-benefit greedy attains (1 − 1/e). **Verify before
  claiming:** (a) submodularity of the *implemented* objective (`campaign.py`) — a terminal reachable
  from two selected hubs is counted once, so marginal gain should diminish, but confirm on the code;
  (b) that the implemented walk *is* cost-benefit greedy. If there is a mismatch, state the gap.
  Reviewers forgive a stated gap; they do not forgive a guarantee that silently does not apply.
- **Do not** include a bare NP-hardness theorem with no guarantee attached.

---

## 12. Numbers and claims that must NOT be cited

| Do not cite | Because | Cite instead |
|---|---|---|
| 1,479 vs 820 reactions; "hub-batching halves cost"; ~1.8–2× | Fragment double-count in the baseline (Logs/029, Logs/031) | Corrected count-once: ~1.13× naive; 2.4–4.4× with child policy + calibrated bar (Logs/033, 055, 058) |
| Logs/028 cost numbers | Superseded nested model | Logs/033 |
| FragGFN cap-9 rows (`fraggfn_cap9_*`) | Mis-set fragment cap collapses DRD2 (Logs/046) | cap-6 cells |
| Logs/048 Exp A as a synthesizability result | Inconclusive | Exp C; and Logs/047 for the stock-mismatch result |
| Logs/050's "SCENT's low-mass tail is genuine stop-probability" | It is the 4000-child enumeration cap (Logs/058 audit) | Logs/058 §Enumeration-completeness audit |
| Logs/041/048 "S3-GFN 34–35 modes vs our 20–27" as the current head-to-head | From-scratch pricing regime only | Logs/056, with **both** regimes shown |
| `log_reward` compared across generators | Each is its own `beta·clip(...)` with a different fitted clip | raw `reward` only |
| "modes ≥ 8.0 on the sEH surrogate" as an activity claim | Proxy AUROC vs property-matched decoys is 0.68; 0/2,315 ChEMBL actives reach it | Logs/034; treat the surrogate as a scale, not activity |
| Logs/025's near-tie between flow ranking and the `parent_of_topk` control as evidence flow does not matter | Explained by estimator noise (§6.3) and refuted by Logs/053 | Logs/053 |

---

## 13. Placeholder convention

Use `⟦PENDING: <what> — <source that will fill it>⟧` inline, and keep a running list at the top of the
draft. Currently expected substitutions:

| Placeholder | Will come from |
|---|---|
| `rgfn_clpp` edge + "all four generators on ClpP" | in-flight seed-43 docking cell |
| `rgfn_6td3` edge | in-flight training chain |
| `scent_clpp` / `scent_6td3` edges at `ENUM_MAX=20000` | in-flight re-enumeration; current 2.65×/2.87× are conservative lower bounds |
| Second-seed bands on docking ratios | not yet scheduled |
| Docking-target external head-to-head | gap 7 |
| Butina/second-fingerprint mode counts | gap 8 |
| Vendor-price conversion | gap 11 |
| ε-robustness corollary statement | §11, to be written |

---

## 14. Where everything lives

- **Experiment logs:** `Logs/NNN_*.md`, indexed with verdicts in `docs/RESEARCH_CONTEXT.md`
  §"Experiment log index". The index verdict lines are the fastest way to find the entry behind a number.
- **Method code:** `glue/samplers/lsdflow/`, `glue/metrics/{lsdflow_flow,uncertainty}.py`.
- **Evaluation code:** `validation/lsdflow/` (adapters, cost models, diversity, SPARROW/MultiAiZ
  evaluators), `experiments/lsd_hubs/` (per-analysis harnesses + committed results).
- **Committed results/figures:** `experiments/lsd_hubs/*/results/**` (small artifacts only).
- **Run artifacts:** `/scratch/markymoo/rgfn_runs/**` (checkpoints, samples, enumerations — not in git).
- **Reviewer-defence notes:** `docs/paper_planning/lsd-flow-publication-strategy.md`.
- **Do not read as current:** `docs/RESEARCH_CONTEXT.md` §"Paper target" (predates the ICLR decision and
  the LSD-Flow pivot) and `docs/AL_PIPELINE_ARCHITECTURE.md` §6 ("in progress" — the SCENT AL loop has
  run to completion). `docs/LSD_FLOW_BENCHMARK_PLAN.md` §11 is explicitly out of scope for this paper.
- **Repo working practice:** this is a shared working tree with other agents. Never `git add -A`; isolate
  hunks (`git diff -U3 <file>` → `git apply --cached` → `git commit --no-verify`). See
  `/home/markymoo/agent_comms/README.md`.
