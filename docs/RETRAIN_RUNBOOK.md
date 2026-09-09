# `benchmark_v2` — specification for the clean re-run

**What this is.** The plan for regenerating the whole benchmark as one internally consistent set of
experiments, in one new tree (`experiments/benchmark_v2/`), under standards that did not exist when
the current results were produced. It is **not** a patch list. Where it is unclear whether a result
can be carried forward, the default is to re-run it.

**Why it exists.** The current results are correct but were assembled over three months during which
the taxonomy, the training budget, the hit gates, the pipeline shape and the route contract all
changed. Nothing is *wrong*; several things are no longer *commensurable*, and a journal reviewer
reads a benchmark's internal consistency as a proxy for its trustworthiness.

**When it runs.** After the workshop submission. It is deliberately not on a deadline: the point is a
clean foundation, not speed. Large GPU-hour totals are acceptable; avoidable ones are not.

**What it is for.** A backup body of evidence if the workshop/ICLR route does not land, and the
substrate for the journal (NCS) submission — including a chemist-facing route dataset that the
current results cannot support.

**Read first:** `docs/RESEARCH_CONTEXT.md` (definitions, the two pools, the logging spec) and
`experiments/benchmark_v2/README.md` (the operational twin of this document — what lives where).

---

## 0. What changed, and therefore what cannot be carried forward

Six decisions post-date most of the results on disk. Each one is a reason a cell has to be rebuilt,
and together they are why this is a re-run rather than a repair.

| # | Change | Date | Consequence |
|---|---|---|---|
| 1 | **Generator taxonomy.** FragGFN is a *non-reaction* GFlowNet and belongs with S3-GFN, not with our reaction-grounded generators | 2026-08-28 | The "16-cell matrix" is dissolved. Hub-batching is built on **three** generators (RGFN, RxnFlow, SCENT); FragGFN moves to the competitor block and needs the full route-less pipeline it has never had |
| 2 | **Training budget standardised** to the PMO convention (~10,000 oracle calls), with each generator at **its own paper's batch size** | 2026-08-21 → 08-28 | Our three generators ran at 320,000–640,000 calls, i.e. **32–64× the competitors**. Every one of our trained cells is superseded. FragGFN's old runs are ~30× over budget (`a327c3a`) |
| 3 | **Hit gates re-derived** on one rule: the score at which **5% of that target's property-matched decoys pass** | 2026-08-21 | sEH 7.0→**5.68**, DRD2 0.5→**0.345**, ClpP −8.0→**−9.10**, 6TD3-B **6.718**. Free for our side (gate is applied post-hoc, on CPU); **not** free for competitors — it changes pool composition, and therefore retrosynthesis and selection |
| 4 | **A fourth competitor stage: upsample-and-filter** (`upsample_to_modes.py`, `dd8f1a9`) | 2026-08-28 | A competitor "pool" is no longer a slice of a fixed 2,000-molecule sample. `pool-limited` used to conflate *the generator cannot* with *we did not ask for enough* |
| 5 | **Route contract enforced at write time**; RGFN/RxnFlow route emission implemented | 2026-08-24 | A fresh sample now produces `routes.json`. The route dataset can go from **5 cell-seeds to matrix-wide** as a by-product of re-running — see §6 |
| 6 | **6TD3-B** replaces the exploitable Tier2−Tier1 differential as the CDK12–DDB1 reward — reward *and* gate are `cnn_vs` (CNNscore × CNNaffinity) at **6.718** | 2026-08-21, settled 08-28 | Every 6TD3 cell of every generator is superseded: the old libraries do not survive re-gating. Oracle wiring in progress; see §7.1 for the threshold and the evidence |

### The one thing that survives unchanged

**Oracle calibration and hardware measurements are properties of oracles, molecule sets and GPUs, not
of any generator run.** They are copied, not re-run: the four oracle validations (Logs/034, 045, 066,
069), the pose-selection ablation (008), docking throughput and batch-size tuning (036, 057),
determinism findings (`PYTHONHASHSEED`, REINVENT nondeterminism, DRD2 env-invariance), the
baseline-config audit, TANGO's inventory and `num_top_results` measurements, and the AiZynth
stock-mismatch results (047, 048).

---

## 1. The grid

**Authoritative list: `experiments/benchmark_v2/grid.csv`** (108 training cells). This table is the
summary; the CSV is what drivers read.

|  | **GFlowNet** | **not a GFlowNet** |
|---|---|---|
| **reaction-grounded** | **RGFN, RxnFlow, SCENT** — hub-batching applies | SynFormer |
| **not reaction-grounded** | FragGFN, S3-GFN | REINVENT, Saturn, TANGO |

9 generators × 4 targets × 3 seeds (42/43/44) = **108 training cells**; 81 in phase 1, 27 in phase 2.

**Phase 1 — sEH, DRD2, ClpP.** Everything. **Phase 2 — 6TD3-B.** Reward and gate are settled
(§7.1); it starts once the oracle is wired, and is explicitly lower priority than the other three
targets, for every generator.

### Two arms, both defined on the oracle-call axis

Never on the step axis. Three generators have three different per-step call counts (RGFN 100,
SCENT 64, RxnFlow 64 at the authors' `num_from_policy`), and replay buffers make the arithmetic
unsettleable. **The trace counter decides, not multiplication.**

| arm | budget | who | purpose |
|---|---|---|---|
| **A** | **10,000 oracle calls** | all 9 generators | the PMO convention the competitors' own papers use |
| **B** | **320,000 oracle calls** | the 3 reaction-GFNs | **SCENT's own published protocol** — 64 forward trajectories × 5,000 iterations |

**Each exhibit uses the arm that makes it valid (decided 2026-09-08). Arm B's downstream is NOT
paused.**

| exhibit | arm | why |
|---|---|---|
| **external head-to-head** (us vs 6 competitors) | **A** | budget parity is what makes a cross-generator claim fair |
| **internal matrix** (hub-batching vs best-candidate) | **B** | within-generator on an identical pool, so parity was never needed — and at arm A one of our three generators is not itself (§7.1, Logs/078) |
| **compute↔reactions tradeoff** (new) | **A vs B**, our generators | prices GPU-hours against bench reactions: *"X more GPU-hours buys Y fewer reactions"*. Carries the composability control below |
| **route dataset** | **B** | SCENT promotes nothing at arm A, so its arm-A routes bottom out at stock and lose the nested promoted-fragment case the schema is built around |

**Why this is not "short vs long" but two literatures.** SCENT's paper contains the word "oracle"
**zero** times in 23 pages — it normalises on iterations, and explicitly matched RGFN ("we changed the
number of sampled forward trajectories to 64") and SynFlowNet ("64 forward trajectories and 5000
iterations") to its own 64 × 5,000 = 320,000. RGFN normalises on *training time*. S3-GFN is the PMO
paper ("oracle budget is limited to 10K"). So the reaction-GFN and PMO literatures normalise **32×
apart**, and this benchmark is the first thing to span both. Report under both and say so.

**Arm B is not available to the competitors, on a measured basis.** SynFormer trains at 10,048 calls
in 19.30 / 20.70 / 20.17 h (`timing.json`, `total_run_s` 69,489 / 74,502 / 72,605), so 320,000 would
be ~615 h/cell and ~5,530 GPU-h for its nine cells. Saturn and TANGO are the same shape. An
iteration-matched external comparison is therefore *unavailable*, not merely expensive.

**One training run yields both arms.** Train to 320,000 calls and checkpoint on the way past 10,000.
That *is* the continuation, on one trajectory, at zero extra training compute — and it sidesteps
reproducibility entirely, which matters because the docking reward is genuinely stochastic, so two
runs at the same seed would **not** agree on ClpP or 6TD3-B.

### The composability control: run HB-Enum-SB at BOTH arms

**This is a control, not an exhibit.** It is not a study of whether SPARROW beats greedy; it exists
so the arm-A external result and the arm-B internal matrix can be *discussed together* without an
unstated assumption between them.

**The risk it closes.** The two exhibits are not composed arithmetically, but the paper narrates them
together, and a reader will infer that the internal advantage explains the external win. That
inference needs the selector effect to be budget-independent, and nobody has checked. There is reason
to doubt it: the greedy→SB uplift depends on how much route sharing a pool offers, and a model
trained 32× longer converges somewhere with different sharing. The competitor side already shows the
uplift is strongly generator-dependent (REINVENT 1.86–2.57×, S3-GFN 1.01–1.65×) and that it
**reverses the ranking** — greedy favours S3-GFN 4 of 4 target×pool combos, SB favours REINVENT 3 of 4
— while `CLAUDE.md` designates SB to carry the headline.

**It is nearly free, because the experiment already exists.** `submit_hb_enum_sb.sh` hands SPARROW
hub-batching's own enumerated candidates — same optimizer, same gate, same budget, only the chooser
differs. That IS the greedy-vs-SB ratio on our pools; Logs/062 measured it at **2.27× (n=3, per-seed
1.67 / 2.03 / 3.13)**. The marginal ask is to run it at both arms rather than one.

**RUN IT ON DRD2, NOT sEH — the MILP does not converge on our sEH pools.** On the corrected uncapped
enumerations (123k–150k above gate) a 12 h cap and `gapRel` 1e-3 both returned TimeLimit, and the
capped answers do not trend (12 / 27 / 37 distinct across three pool sizes, no ordering), so sEH is
reportable only as a lower bound. DRD2 converged on all six budget points (Logs/062, jobs
74727/74728) — but note it was capped at `--top-n 50000` to match its competitor rows, so the DRD2
MILP saw ~⅓ the variables. Expect arm A to converge more readily than arm B, since a
less-trained model yields a smaller eligible pool (B measured 7,111 hubs at arm A against v1's
20,874) — which is the wrong direction for a paired comparison, so **check `time_capped` on every row
and do not form a ratio across one converged and one truncated arm.**

**What it can and cannot say.** It measures budget-sensitivity of the selector effect on *our* pools.
It cannot test the cross-generator ranking's budget-stability — that would need the competitors at
arm B, which is unavailable (§1). Converting an unstated assumption into a measured one is the whole
of the win.

### Resume, per generator — and why SCENT's arm B must not requeue

Verified from the checkpoint code 2026-09-07 (agent A), not from comments.

| generator | what the checkpoint holds | resumable? |
|---|---|---|
| **RxnFlow** | model + **both** optimizers + **both** LR schedulers + step (`al_loop.save_checkpoint`) | **yes**, fully |
| **RGFN** | model + optimizer + lr_scheduler + metrics + replay buffer | **yes, but only after a cache strip** |
| **SCENT** | the same five — and **not the dynamic fragment library** | **partially, and it silently corrupts** |

**A second, independent reason SCENT's arm B must not requeue (added 2026-09-08).** The RNG is not
checkpointed, so a resumed segment samples differently and promotes different fragments. Since
`7134642` a requeued run is *correct* — the library, its costs, the path-cost cache and the recipe
routes all restore — but it is **not reproducible**: it will not promote what an uninterrupted run
would. Do not relax the one-walltime rule on the grounds that the library is now safe; that fixes
correctness, not reproducibility.

**RGFN needs a cache strip.** Its forward policy carries lazily-populated `*_cache` buffers that a
freshly constructed model does not have, so a strict `load_state_dict` against a raw checkpoint
*fails* (Logs/021). The production chain works around it at resume time
(`experiments/fixed_reward/scale5k/submit_rgfn.sh` strips every `model` key ending in `_cache`), which
means **a raw RGFN checkpoint is not by itself resumable**. The caches are 78% of the file — 404 MB →
90 MB on `rgfn_seh_5k/seed42`. The arm-A checkpoint is therefore written **already stripped**; it is
the one checkpoint whose entire purpose is to be resumed from.

**SCENT's dynamic library is not in the checkpoint, and the damage is worse than a smaller library.**
`external/scent/rgfn/trainer/trainer.py:281-287` saves `model / optimizer / lr_scheduler / metrics /
replay_buffer`; the promoted-fragment library is only ever serialized to
`additional_fragments/fragments_<N>.json` for analysis and is never restored. So a resumed SCENT run
restarts with an **empty** library. v1 shows this happened: `scent_6td3_5k/seed42` progressed
400/800/**400**/800 (two resets, final 800) and `scent_clpp_5k/seed43,44` 400/**400**/800/1200
(one reset, final 1200), against a clean 1,600 for the cells that ran straight through — so the final
library size in v1 is a function of **requeue timing**, not of the science.

The second half is the nastier one, and it is silent. `FragmentOneHotEmbedding.weights` is
pre-allocated to `418 + max_additional` rows and `_get_embeddings()` returns `weights[:
current_fragments]` — **indexed by position**, while `on_update_fragments_library` updates only the
*count*. After a reset, a re-promoted fragment therefore inherits the **trained embedding row of the
slot's previous occupant**. Nothing crashes; the content-based `all_fingerprints` rebuilds correctly,
so only the one-hot half carries stale identity.

**Three consequences for this campaign:**

1. **Train one process straight through to arm B, checkpointing at arm A on the way past.** Already
   the plan above — but it is now load-bearing for a second reason, not just continuation semantics.
2. **"Train the cheap 10,000-call arm everywhere, extend later" is available for RGFN and RxnFlow
   only.** For SCENT the extension would not continue the same library trajectory.
3. **A SCENT arm-B run must finish inside one walltime.** Arm B is 320,000 calls = 5,000 iterations,
   SLURM `compute` caps at 3 days, and v1 proves SCENT 5k does *not* fit on at least some targets. A
   requeued SCENT arm-B cell is confounded and must be discarded, not repaired.

**FIXED 2026-09-07 — the library now persists across a requeue** (`validation/generators/scent/
library_io.py`, a sidecar beside `last_gfn.pt`, on the `guidance_io.py` precedent; the clone stays
untouched). The July audit declined this because it changes training behaviour mid-campaign; on a
clean slate that objection expires. Four things worth carrying, because they change what the fix is:

* **The trained rows were never lost, only made invisible.** `weights` is an `nn.Parameter`, so
  `model.load_state_dict` restores all 4,418 rows; it is `current_fragments` resetting to 418 that
  hides everything past it. So this restores *bookkeeping*, not a second weight sidecar.
* **It restores SCENT's design rather than deviating from it.** The one-hot table is sized
  `418 + num_additions × n_new_fragments` = **4,418** — built for a complete, never-reset run. The
  reset is an artifact of our 3-day requeue.
* **Three pieces of state, not two** — the ordered fragment list, the embedding rows, *and* the
  path-cost cache entry per fragment. The third was found by crashing into it: `assign_costs` only
  fills `molecule_num_reaction_to_cost` for states reached through a reaction action, so once the
  restore puts promoted fragments back in the action space the policy can select one as a *starting*
  block and hit a `None` the clone does not guard (it guards the identical lookup two lines away
  with `or inf`, so this is an upstream gap we compensate for, not a choice). The restore re-seeds
  the true cost, never `inf` — that value is what the next promotion records for the fragment.
* **A resumed run with the library restored but the cache cold CRASHES.** v1 never hit this only
  because the library was empty on resume — *the reset was accidentally what kept the crash away.*
  Anyone who fixes the reset without the cache seeding gets a run that dies at the first promotion
  after a resume. It is also the sharpest argument for the verification bar below: a count check
  never even runs, because the process is already dead.

**Verification: `validation/generators/scent/verify_library_recovery.py`.** Uninterrupted vs
killed-and-resumed at the same iteration, compared on a sha1 of the *ordered* `chosen_smiles` **and**
a sha1 of `weights[:current_fragments]` — a count check passes on precisely the bug being fixed —
plus a blind attribute-by-attribute diff of the library, the cost proxy and the replay buffer, so the
restore inventory is shown complete rather than merely sufficient for the cases we hit.

⛔ **v2's SCENT is deliberately NOT bit-comparable to v1's.** This changes training behaviour. Do not
diff a v2 SCENT cell against its v1 counterpart and report a regression.

### What "arm A" precisely means

Two details that a later reader will otherwise simplify away, both in
`validation/generators/_trace.py`:

* **The checkpoint fires at the first iteration BOUNDARY at or after 10,000 calls**, not at the
  10,000th row. A GFlowNet scores a whole minibatch inside one iteration, so the 10,000th call lands
  mid-iteration where no coherent optimizer/replay state exists to check-point. Overshoot is bounded
  by one batch (64–100 molecules, ≤1%), and `arm_a.json` records **both** numbers:
  `crossed_at_n_scored` is the truth, `saved_at_iteration` is where the weights are.
* **`step` is stamped from `on_start_sampling`, the budget checked from `on_end_sampling`.** Reward
  evaluation happens *during* sampling, so reading the step at the end would label every row with the
  previous iteration — off by one for the entire file.
* **`phase` is load-bearing.** RGFN and SCENT score the final candidate batch through the *same*
  proxy the training loop uses, so those rows are flipped to `phase="eval"` before sampling. Without
  it a smoke measured 600 training calls followed by 185 scoring calls, indistinguishable — and every
  budget or modes-vs-calls reading would count the second set. **Filter to `phase == "train"` for any
  budget claim** — and count ROWS, never the cumulative `n_scored`, which absorbs eval calls as it
  goes (measured on `s3gfn_drd2/42`: the same file reads 12,048 / 11,048 / 10,048 depending on which
  you take).
* **SCENT's periodic validation was inflating its own training count 2.84×**, and this is a measured
  figure rather than a rounding concern. Its config sets `valid_sampler = RandomSampler` with
  `valid_n_trajectories = 1000`, so every validation pass scores a thousand molecules through the
  same proxy — and they land labelled `train`, because `TrainingHooksMixin` has no validation hook.
  Per-iteration train rows in a 6-iteration smoke read 63/94/96/95/96/**1088**. The arm-A budget now
  gates on training-phase rows only, and `attach_proxy_trace` wraps the trainer's own `valid_step` to
  flip the phase for its duration; after the fix that iteration reads **96**.

### Batch sizes: each paper's own

| generator | batch | source |
|---|---|---|
| RGFN | 100/step | `configs/rgfn_base.gin` `train_forward_n_trajectories` (upstream, pristine) |
| SCENT | 64/step | `configs/scent_base.gin` (clone default) |
| RxnFlow | 64/step | `external/RxnFlow/src/gflownet/algo/config.py:187` `num_from_policy: int = 64`. The runner never overrides it, so it already inherits the authors' value |

**CORRECTION (2026-09-07, agent A).** An earlier draft of this runbook said RxnFlow ran at 2× its
authors' batch and had to be fixed. **That was wrong, and no config change is needed.** The `128` is
`reward.batch_size` in `rxnflow_*_5k.yaml`, which is passed to `SEHFrozenReward(batch_size=...)` —
the sEH proxy's *scoring* batch, a pure throughput knob with no effect on the training budget. The
*policy* batch is `algo.num_from_policy`, which `run_rxnflow_fixed.py` never sets, so it inherits the
authors' 64. FragGFN's own configs already record the same conclusion for the shared Recursion
codebase ("The batch was already right"), and `validation/generators/fraggfn/task.py:104` pins it
explicitly. Two different knobs, one name — check which one a number refers to before calling it an
error. The step count still follows from the budget, not the other way round.

---

## 2. Standards — the things that make cells comparable

Every one of these is a property a cell must have, checkable, not a convention to remember.

### 2.1 Hit gates — 5% FPR, and do not round

| target | column | gate | TPR | FPR | enrichment | AUROC |
|---|---|---|---|---|---|---|
| sEH | proxy value | **5.680** | 13% | 5.0% | 2.5× | 0.676 |
| DRD2 | activity probability | **0.345** | 74% | 5.0% | 14.9× | 0.949 |
| ClpP | raw Vina | **−9.100** | 47% | 5.0% | 9.5× | 0.895 |
| 6TD3-B | **`cnn_vs`** (CNNscore × CNNaffinity) | **6.718** | 78% | 4.4% | 17.8× | 0.917 |

Source of truth `experiments/lsd_hubs/matrix16/targets.py`; re-derive with
`experiments/oracle_validation/calibrate_gates.py` (verified 2026-08-28 to reproduce all four
exactly). **These are empirical grid points — rounding breaks the exact-FPR property they are defined
by.** Resolve a gate by importing `targets.py`, the way `upsample_to_modes.gate_for()` does; never
hardcode one, and never give one a default (§7 lists the places that still do).

**Carry this caveat into the paper:** sEH cannot reach a good operating point at *any* threshold
(AUROC 0.676, enrichment 2.4–2.7× across the whole range). That does not invalidate the benchmark —
all arms share the gate — but "sEH modes" is a weaker claim than "DRD2 modes", and the two must never
be averaged into one number without saying so.

### 2.2 Hub-batching configuration — one algorithm across all three generators

**`--pool all --child-policy free_frag --prebuild-k 0`.**

#### `--pool all` — no reward pre-filter on the hub set (decided 2026-08-28)

v1 ranked flow only over the parents of the top-`TOPK` candidates by reward (`TOPK=1000` in
production). That is a **reward pre-filter**, and `pick_hubs.py`'s own docstring describes it as
legacy continuity with the original Logs/028 recipe, not as a justified choice — `--pool all` was
added specifically so "does the flow signal buy anything?" is answerable.

**It is free to remove.** The flow estimate is read straight off the record's log-terms
(`logR + logP_B − logP_F(move) − logP_F(stop)`) — no model, no scoring. Measured on `scent_seh`
seed 42: **0.17 s over 575 filtered hubs, 0.17 s over all 20,874.** Enumeration cost is set by
`--n-hubs`, identical either way.

**It changes 60% of the walked hub set and ~1 mode of outcome.** Overlap between the two
flow-descending orders is **40% across the first 5/10/20/40 hubs** (46% at 200); three of the
unfiltered top-10 hubs are absent from the filtered list entirely. Yet Logs/053 measured, on
identical everything else:

| arm | pool | modes @ R=100 | reactions | rxn/mode | reward-gen calls | compute |
|---|---|---|---|---|---|---|
| `incumbent` | reward-filtered | 72 | 365 | 1.217 | 241,158 | 5,685 s |
| **`flow_top`** | **all** | **73** | **357** | **1.190** | 244,685 | 5,648 s |

Swap 60% of the hubs, move one mode. That is "flow finds the neighbourhood, not the rank" arriving
from a third direction, and it is the reason the switch is safe.

**Why do it:** it removes a confound from our own ordering ablation — with the pre-filter, "flow
ordering wins" is measured on a pool that was *itself reward-selected*, so a reviewer can say the
reward filter did the selecting and flow only sorted the survivors. It also deletes an arbitrary knob
(`TOPK`) from the method description.

**⚠ The consequence that must be reported, not just accepted: depth-0 exposure rises 4×.**
`pick_hubs.py` applies **no depth filter**, so `--pool all` admits bought building blocks that the
reward pre-filter happened to exclude — measured on the same cell, depth distribution of the top-200:

| pool | depth 0 | 1 | 2 | 3 | **depth-0 in the first 40 (the part actually walked)** |
|---|---|---|---|---|---|
| reward-filtered | 2 | 104 | 84 | 10 | **2** |
| `all` | **11** | 150 | 37 | 2 | **8** |

A depth-0 hub is legitimately priced at **zero** reactions — a chemist buys it
(`shallow_couplings(depth=0, promoted=()) == 0`) — and since `cc25046` the competitor gets catalogue
starting materials too, so this is not an unfair advantage. But "a bought scaffold costs 0 reactions
and still carries thousands of children, so it ranks near the top by flow" (`cc25046`), and the paper
itself names depth-0 catalogue picking as the metric's **degenerate optimum**. Leaning harder on it
without saying so would be the single most attackable move in the benchmark. So:

1. **Every cell reports the depth distribution of the hubs it walked AND the share of delivered modes
   that came off depth-0 hubs.** This is a new required output — v1's committed curves record
   `source_hub` but the hub-ordering arms kept only plots, so the delivered library's depth mix
   **cannot be recovered for the v1 comparison** and is unmeasured today.
2. **Run `--min-hub-depth 1` as a labelled sensitivity arm.** It is exactly the "is the win late-stage
   diversification or catalogue picking?" question a reviewer asks, it is cheap (same enumeration
   cache, different `hubs.csv`), and it matches the AL acquisition path's own default — which uses
   `min_hub_depth=1` precisely because depth-0 is "huge fan-out, zero amortization — not the 'build
   once, diversify' signal" (Logs/025). See §7.5: the campaign's `pick_hubs.py` has no such knob while
   the acquisition path does, which is an inconsistency between two implementations of the same idea.

**`--n-hubs` becomes the method's single width knob.** Under the pre-filter, `TOPK=1000` capped
eligibility at 575, so `--n-hubs 200` was a soft cap. Under `--pool all` it is the only thing between
the walk and 20,874 hubs. Report it beside the competitor's Stage-2 cap (§2.3).

#### `--child-policy free_frag --prebuild-k 0`

These are two knobs, not one. `free_frag` is the *filter*: keep only children whose last step attaches
an already-available fragment, so each kept child costs exactly one marginal reaction. `prebuild-k` is
the *stock*: pre-pay to synthesize the top-K promoted fragments up front, charged before the first
mode (`glue/samplers/lsdflow/campaign.py:451`).

- `free_frag` is **inert** on RGFN and RxnFlow (no dynamic library ⇒ it keeps everything ⇒ identical
  to `reward`), so switching it on universally costs nothing and removes a per-generator special case.
- `K=0` because Logs/049 traced the entire 12.69% count-once-vs-SPARROW gap on DRD2 to pre-select-K's
  up-front reactions (~17 of 26 never used by a 100-mode selection). **At K=0 the gap is exactly
  0.00%.** The cost-model audit is the credibility anchor for every number in the paper; do not
  hand it a 12% discrepancy that has a known cause.
- `K=20` remains available as a **labelled compute-saving variation** (Logs/037: 2–6× fewer
  oracle/enumeration calls at roughly constant reactions/mode). Revisit if SCENT runs short of oracle
  calls. It is never the core method.

### 2.3 Budgets: there are two, and only one is equalised

| budget | what it covers | treatment |
|---|---|---|
| **training** | oracle calls consumed while learning | **equalised at 10,000** (arm A) |
| **inference / selection** | Stage-2 upsampling (competitors, up to 50k distinct); hub enumeration (ours, up to ~700k children) | **measured and reported per arm, never equalised** |

Equalising the second would force the fixed-*mode* readout this project demoted to secondary. The
asymmetry is not directional — the calls make the competitor look expensive while the resulting pool
makes it stronger — so report it and say so. Our side's width knobs (`n_hubs=200`,
`n_traj=30,000`) belong in the same table as the competitor's Stage-2 cap, so a reader can see both.

**The competitor matrix is NOT a clean 10,000, and must not be labelled one.** Stage 2 equalises
POOLS (500 modes), not oracle calls, so each cell carries a measured surcharge on top of arm-A
training — REINVENT/Saturn/TANGO 12,018–12,048, FragGFN 12,047–14,046, **S3-GFN 10,048–18,048**. A
1.80× spread, 122,411 calls over 45 cells, concentrated on S3-GFN. Label it *"arm-A training plus a
measured Stage-2 surcharge of 2,000–8,000"* and state the spread rather than averaging it away.

**But the asymmetry runs toward us, not away.** Our own inference-time spend is 33,099–69,896
reward-gen calls per v1 campaign cell and reaches **791,503** on a budget-scale cell — **50–100×** the
competitors' entire Stage-2 cost. The inference-time axis is uncontrolled for everyone and we are by
far its heavier user. Equalising it is not the fix: that would force the fixed-MODE readout this
project demoted to secondary. Report it; do not equalise it. One within-field reading does survive:
S3-GFN's edge over Saturn and TANGO is partly bought with up to 1.8× their oracle calls.

### 2.4 Reporting conventions (unchanged, restated so a cell can be checked against them)

- **Primary readout: modes at a fixed 100-reaction budget.** Emit 50/100/150/200/300 for the
  reaction-GFNs (free — same ordering, CPU re-read) and headline 100. Multi-budget is **not**
  available on the competitor side without re-running retrosynthesis and re-solving SPARROW; out of
  scope.
- **Every cell carries its stop reason:** `budget-binding` (the only like-for-like case),
  `pool-exhausted`, `mode-capped`, `solver-truncated` (SB arm only), and — new from Stage 2 —
  `target-reached` / `stalled` / `cap`. Outside the budget-binding regime quote `cost_kept_rxns`,
  not `used_rxns`; the gap between them *is* the exhaustion detector.
- **Seeds 42/43/44**, `PYTHONHASHSEED=0` exported on every sample (load-bearing: `--seed` alone gave
  377 vs 387 routes; with it, 730/730 byte-identical).
- **Two pools per competitor cell** (naive / pruned), each read two ways (reactions/candidate,
  reactions/mode). Run `mode_saturation.py` first — cells whose two pools coincide need only one.

---

## 3. The two stage graphs

They are different. The runbook holds both; do not let a driver assume symmetry.

```
REACTION-GFNs (RGFN, RxnFlow, SCENT) — hub-batching
  train ──▶ sample ──▶ pick_hubs ──▶ enumerate ──▶ campaign
    │         │                        │             │
    │  ckpt@10k calls          children +        hub_batching vs
    │  + trace.csv             reactions         best_candidate
    │  + recipes (SCENT)       + rewards         at R = 50…300
    └─ ① routes.json prefix   └─ ② children[].reaction    ③ recipes expand promoted fragments

COMPETITORS (FragGFN, S3-GFN, SynFormer, REINVENT, Saturn, TANGO)
  train ──▶ upsample&filter ──▶ retrosynthesis ──▶ selection
    │            │                    │                │
    │      Stage 2: sample until      MultiAiZ, or      SPARROW MILP
    │      500 diverse modes;         native routes     + diversity-aware
    │      harvests trace FIRST       (SynFormer,       greedy
    │      as a free pool             TANGO ARM 2)
    └─ trace.csv is REQUIRED for the free pool
```

**Stage 2 does not apply to the reaction-GFNs.** Our pool is the enumerated hub neighbourhood, not a
sampled pool; there is nothing to upsample. **SynFormer is exempt from Stage 2 and that is a
finding** — its candidates are a slice of an accumulated GA population, so getting more molecules
means more *training*. Generate-then-sample methods can be upsampled; a genetic algorithm cannot.
Those cells stay pool-limited by construction.

**TANGO sits on two axes that do not coincide.** On *construction mechanism* it groups with Saturn and
REINVENT (Saturn's generator with a constrained-synthesizability reward; syntheseus decomposes
post-hoc, so synthesizability is reward-level, not construction-level). On *who ships routes* it groups
with SynFormer. **The 2×2 is on the first axis.** A table that implies the two coincide reads as an
error. TANGO uses **ARM 2** (re-run syntheseus over the emitted pool, ~3.5 h/cell), never ARM 1 —
ARM 1's coverage correlates with the generator's own collapse and covered 7 of 218 molecules on the
pruned sEH pool.

---

## 4. Generated vs copied — one tree, one ledger

**There is no `copied/` subdirectory.** `benchmark_v2` is a single source of truth with a uniform
layout; an agent reading it never has to ask which run a file came from. Provenance lives in **one
machine-readable ledger**, `experiments/benchmark_v2/PROVENANCE.csv`:

```
stage,generator,target,seed,arm,origin,source_path,source_md5,date,operator,note
```

`origin ∈ {generated, copied}` and nothing else.

### What is planned as copied (36 of 108 training cells)

| generator | targets | why it copies |
|---|---|---|
| S3-GFN | seh, drd2, clpp | corrected to ~10k calls 2026-08-21; `aux_coefficient` fixed to the authors' 0.001 |
| REINVENT | seh, drd2, clpp | corrected to batch 64 × 157 steps = 10,048 |
| Saturn | seh, drd2, clpp | config verbatim from the authors' `constrained_synthesizability/experiment.json` |
| TANGO | seh, drd2, clpp | same, plus the TANGO oracle |

Everything else generates: all three reaction-GFNs (budget change), FragGFN (~30× over budget),
SynFormer (1 of 9 exists, blocked), and every 6TD3-B cell (never run for any competitor).

### Copying is at the TRAINING stage only

**A copied training run does not carry its pools forward.** The gate change moves pool composition,
and therefore retrosynthesis and selection. Measured across all 37 trained competitor cells
(naive top-500):

- **27 of 37 unchanged** — pools *and* cached MultiAiZ routes stay valid; re-run selection only.
- **10 change**, and they cluster: `reinvent/clpp` all three seeds (587→122, 647→158, 511→92),
  `s3gfn/clpp` all three (263→41, 243→30, 302→47), `reinvent/seh/s44` (464→1202),
  `s3gfn/seh/s43` (65→476, verified independently), `s3gfn/seh/s44` (168→807),
  `s3gfn/drd2/s44` (97→186).
- Saturn and TANGO ClpP are untouched at 1800+, so **the ClpP damage is specific to REINVENT and
  S3-GFN.** Those six cells go pool-exhausted at R=100 and must be flagged, not quoted. Their
  survivors are a *subset* of already-routed molecules, so re-gating them is a filter, not a re-route.

⟦OPEN — not measured: whether **pruned** pools change more widely than naive ones. Plausible (the
sphere-exclusion scan reaches deeper below the gate) but unverified. Do not state it as established.⟧

### A copy MUST include `trace.csv`

Stage 2 harvests the training trace as a free pool of already-scored molecules at zero marginal
oracle cost — `saturn_clpp/s42` reaches 500 modes from its history alone, turning 28 GPU-h into 0. A
copy that brings the checkpoint but not the trace silently throws that away. `trace.csv` is a required
column in the ledger, not an optional extra.

---

## 5. Order of operations, per cell

Cells run concurrently and independently. Nothing waits for a phase to complete across all cells.

1. **Verify the plan.** Read the cell's row in `grid.csv`. If `train_plan=copy`, check the source
   still exists and its md5 matches; write the ledger row *before* copying.
2. **Train** to arm B's budget, with `_trace.py` writing continuously, checkpointing when the trace's
   `n_scored` crosses **10,000** (arm A) and at the end (arm B). SCENT trains with `--log-recipes`
   (default on since 2026-07-29 — confirm in the run's own config, do not assume).
3. **Freeze the training artifacts.** `chmod -R a-w` the cell's `train/` directory once step 4 passes.
   This is the structural fix for the hazard in §8.1 — every later stage writes to its own directory,
   so "do not re-invoke a runner against this cell" is enforced by the filesystem rather than
   remembered.
4. **Verify the training stage** (§6). A cell that fails here is re-trained now, at the cost of one
   run. A cell that fails here and is discovered in three weeks costs the same run plus everything
   built on it.
5. **Back up** to `/project/def-naeilum/.../RGFN_LSD_MarkStevens/backups` — per cell, per stage,
   as soon as that stage passes verification. Copy-verify-delete, never `mv`. Login node only
   (`/project` is not mounted on compute nodes).
6. **Downstream**, per the cell's pipeline in §3. Ours: sample → pick_hubs → enumerate → campaign.
   Competitors: Stage 2 → retrosynthesis → selection, both pools.
7. **Verify each stage** (§6) and write its ledger row before moving on.
8. **Record the delta.** A re-run cell's numbers *will* move. Diff against the corresponding v1 result
   and record the change rather than quietly replacing it.

---

## 6. Verification — every item is a failure that actually occurred

A cell is not accepted until all of these pass. Each corresponds to a silent failure that produced a
confident wrong answer rather than an error.

### 6.1 Routes were emitted (①)
```
python -c "import json;print(len(json.load(open('<cell>/sample/routes.json'))))"
```
Non-zero for rgfn / rxnflow / scent. `_routes.py` validates at write time and emits
`route_status.json` — check that file exists. *Why:* two workers hardcoded `json.dump({})`; a run
wrote a well-formed empty file, exited 0, and was promoted. 36 of 40 cell-seeds were in that state,
and downstream SPARROW priced the **empty library** as trivially `Optimal` at zero cost.

### 6.2 Every enumerated child carries its reaction (②)
Fraction of `children[].reaction` non-empty must be **1.00**. Anything strictly between 0 and 1 is
*more* dangerous than 0: a partial artifact prices a mixture — some children at their true molecule,
the rest at their hub — and still returns `Optimal`. Measured on merged artifacts: 56.6%, 24.6%,
**99.1%**. A 99.1% file is exactly the one whose warning gets scrolled past.

### 6.3 Recipes exist AND belong to this run (③)
Two parts. Part 1: does the snapshot have `smiles_to_route` at all? Part 2: **does it cover the
fragments THIS RUN used?** Part 2 is the one that was missing. Coverage of the snapshot's own
`chosen_smiles` is the wrong question — a snapshot from a *different* model is internally complete, so
it reads **100%** while only **53%** of the run's fragments are expandable (549 of 1,169 silently
*bought* rather than *built*).

### 6.4 One command answers 6.1–6.3
```
python experiments/lsd_hubs/matrix16/check_route_readiness.py
```
**In `benchmark_v2` this is a hard gate, not a report.** A cell ships only if a chemist could act on
every molecule in it. That is what turns the route dataset from 5 cell-seeds into matrix-wide
coverage — the emitters exist now, so the coverage is a by-product of re-running, provided nothing
accepts a cell that fails.

### 6.5 The trace is present, continuous, and the checkpoint matches it
`trace.csv` exists, its `n_scored` reaches the arm's budget, `phase == "train"` rows are separable
(some generators score outside the training loop — S3-GFN's `evaluate()` scores 1,000 molecules, and
counting those inflates the budget), and the arm-A checkpoint sits at the row where `n_scored` first
crosses 10,000.

### 6.6 Determinism
`PYTHONHASHSEED=0` exported. Without it a sample is a one-of-a-kind artifact recoverable only from
backup.

### 6.7 Do not trust a solver's status
CBC reports `Optimal` when it has merely run out of time (PuLP's `sol_status` does not distinguish).
Detect by wall-clock, not status. A capped row is a **lower bound on the competitor**, i.e. it
flatters us, and cannot carry a ratio.

### 6.8 Freeze inputs for any comparative arm
Any experiment claiming two arms saw "the same candidates" reads from a frozen snapshot with the
source md5 recorded — and the snapshot must include `meta.json` and `compositions.json`, because the
provenance and coverage guards resolve them *relative to their inputs*. A snapshot omitting them turns
both checks into silent no-ops, which is what happened on the first attempt.

### 6.9 The one-line rule
Every failure above was an **existence check where the real question was a match or a content check**.
The file was present, well-formed, and wrong. When adding a guard, ask what it would take for the
check to pass on bad data — and check *that* instead.

---

## 7. Build items — before any cell launches

### 7.1 6TD3-B: reward and gate are both `cnn_vs` at 6.718 (settled 2026-08-28)

**Reward and gate: `cnn_vs` — gnina's virtual-screening score, CNNscore × CNNaffinity — higher is
better, gate 6.718.** Another agent is wiring the oracle; this section records the threshold and why
it is the right one.

The problem it solves. Three candidate signals were compared on one docking pass —
`dock_6td3_matched_74500` + `dock_6td3_scentsample_2452248`: 160 real glues, 160 property-matched
decoys, and 400 of our **known reward-exploiting** candidates from the old differential. Each at its
own 5%-FPR gate (recomputed independently 2026-08-28):

| signal | gate | AUROC glues vs decoys | **AUROC glues vs OUR exploiters** | glues | decoys | **ours** |
|---|---|---|---|---|---|---|
| `ddb1_dvina` *(old reward)* | −2.0 | 0.688 | 0.327 | 66.9% | 31.2% | **75.5%** |
| `cnnaff_t2` | 7.970 | 0.804 | **0.521 — chance** | 37.5% | 5.0% | **35.8%** |
| `cnnsc_t2` | 0.9066 | 0.923 | 0.965 | 78.8% | 5.0% | 1.0% |
| **`cnn_vs`** | **6.718** | **0.917** | **0.946** | **78.1%** | **4.4%** | **3.5%** |

Each single column failed one of the two jobs. `cnnaff_t2` is an unbounded pK estimate, so it makes a
good *reward* — but it scores our known exploiters at **chance** (0.521; they clear its gate at 35.8%
against real glues' 37.5%), so it cannot serve as the *gate*. `cnnsc_t2` is the sharp discriminator
but a bounded [0,1] pose-quality probability whose real-glue p90 is 0.987, leaving little headroom
once a pose is good.

**`cnn_vs` does both, and it forecloses the exploit structurally rather than merely failing to
observe it.** Because CNNscore enters multiplicatively, a molecule with poor pose confidence cannot
buy its way past the gate with affinity: at our exploiters' median CNNscore of 0.334, reaching 6.718
would need CNNaffinity ≈ **20**, against a real-glue range of roughly 5.6–9.4. The old reward's
failure mode — make Tier 1 worse rather than Tier 2 better, with no requirement that the pose be
physical — has no analogue here.

It also has the reward shape the training run needs. Our candidates span 0.22–8.37 against the glues'
0.72–8.71, with our median at 2.63 against the glues' 7.59 — ample gradient, no saturation. And
climbing it drags pose confidence up with it, which is exactly what `cnnaff_t2` failed to do:

| our candidates, ranked by | median `cnn_vs` | median `cnnsc_t2` | median `cnnaff_t2` |
|---|---|---|---|
| top 10 by `cnn_vs` | 7.190 | **0.834** | 8.757 |
| top 50 by `cnn_vs` | 6.166 | 0.764 | 8.308 |
| all 400 | 2.634 | 0.334 | 7.767 |
| *(top 10 by `cnnaff_t2`, for contrast)* | — | *0.627* | *9.032* |
| **real glues** | **7.592** | **0.975** | **7.824** |

**One consequence to keep.** Reward and gate are now the same column again, so the standing
exploitation check must score against something that is **not a component of the reward**. Use
`vina_t2` — a different scoring function entirely — alongside the two CNN components, which remain
separately emitted so a passing molecule can be decomposed into *why* it passed. Note our old
candidates already dock *better* than real glues on raw `vina_t2` (median −11.13 vs −10.15), so "beats
real glues on Vina but not on CNN VS" is itself the diagnostic signature.

**Verify the column at wiring time.** The current result CSVs carry `cnnsc_t2`, `cnnaff_t2`,
`cnnaff_t1` and `ddb1_dcnnaff` — there is **no `cnn_vs` column yet**. gnina emits `CNN_VS` natively as
exactly this product, and reconstructing it as `cnnsc_t2 × cnnaff_t2` reproduces the 5%-FPR point to
6.7134 against the adopted 6.718, so the two agree. Confirm which one the oracle captures, and keep
both components in the output either way — not recording a cheap column is what made Logs/069
expensive.

**Generalisation still open (§11).** sEH, DRD2 and ClpP all use the same column for reward and gate,
so none has an independent exploitation check. 6TD3 is the only target where gaming was caught — not
because it is the only one being gamed, but because it is the only one where a second signal happened
to be recorded.

### 7.2 Wire `_trace.py` into the three reaction-GFNs

`validation/generators/_trace.py` exists and is wired into the four competitor runners. It is **not**
wired into `run_scent_fixed.py`, `run_rxnflow_fixed.py`, `run_fraggfn_fixed.py`, or the RGFN path —
they emit no `trace.csv` and no `timing.json`. Without it: no modes-vs-oracle-calls curve on our side,
no per-phase wall-clock, and **no way to place the arm-A checkpoint exactly**. This blocks §1.

Free side effect: the trace's first rows *are* the untrained-policy samples, so the t=0 baseline for a
learning curve costs nothing extra.

### 7.3 Close the stale-gate and unknown-target defaults

- ~~`submit_competitor_routes.sh` still carries pre-standard gates as shell defaults.~~
  **STRUCK 2026-09-08 — this was already done and the item was written from a stale read.** Line 88
  of that script reads `from targets import get_target` and line 90 calls it, on both this branch
  and the competitor branch. Flagged by the Standard Pipeline session and verified. The remaining
  bullets below were NOT re-checked at the same time and should be treated as still open.
- `upsample_to_modes.py`'s `--target` choices are `sorted(DEFAULT_CAP)` = clpp/drd2/seh, and
  `DEFAULT_CAP` has no `6td3b` entry.
- About eight hardcoded `("6td3","clpp")` membership tests and `_HIGHER_IS_BETTER_BY_REWARD` dicts in
  the sample/enumerate path — `scent_worker.py:67,696,793`, `rgfn_adapter.py:63`,
  `rxnflow_worker.py:98`, `fraggfn_worker.py:120`, `scent_adapter.py:43,87`,
  `greedy_oracle/analyze_greedy.py:49` — where `6td3b` falls through a `.get(name, True)` default and
  lands on the **correct answer by luck** (it is higher-is-better, unlike both existing docking
  targets). Add it explicitly and make unknown targets fail loudly.

### 7.4 ~~Correct RxnFlow's batch size~~ — WITHDRAWN, no change needed

Verified 2026-09-07 (agent A): RxnFlow already runs at the authors' `num_from_policy = 64`. The
`128` that prompted this item is `reward.batch_size`, the sEH proxy's scoring batch. See §1.

### 7.5 `pick_hubs.py` — pass `--pool all`, and give it the depth knob it lacks

- **The v2 driver must pass `--pool all`** (§2.2). v1's `submit_cell.sh` does not, and **must not be
  edited** — jobs are in the queue, bash resumes a script at a byte offset, and SLURM runs a submit
  script as of its start. Leave v1 alone; put the flag in the v2 driver.
- **Add a depth band to `pick_hubs.py`.** It has no depth filter at all, while the AL acquisition path
  defaults to `min_hub_depth=1 / max_hub_depth=3` for reasons that apply equally here (depth 0 = huge
  fan-out, zero amortization; depth 4 = at the reaction cap, `P_B` unrecoverable). Two
  implementations of "pick hubs" should not disagree about what a hub is. The default stays
  unfiltered — the sensitivity arm needs the knob, not a new default.
- **Emit the depth mix.** The per-cell campaign output must carry the walked hubs' depth distribution
  and the share of delivered modes sitting on depth-0 hubs. `source_hub` is already recorded; joining
  it to depth is all that is missing.
- **Two incidental bugs, both of which vanish under `--pool all`:** `pick_hubs.py`'s own
  `--top-k-candidates` default is **100** (yielding 64 hubs, never reaching `--n-hubs 200`) while
  `submit_cell.sh` passes 1000 — so invoking the script directly runs a silently different method.
  And `submit_cell.sh`'s Knobs comment says `TOPK (100)` on line 18 while line 35 sets 1000.

### 7.6 A `manifest.py` for `benchmark_v2`

Spanning both pipelines, resolving a cell from `grid.csv` + `targets.py` + live filesystem status, the
way `matrix16/manifest.py` does for one. Deliberately not written yet — a bad duplicate is worse than
none.

---

## 8. Hazards

### 8.1 Re-invoking a generator runner OVERWRITES that cell's outputs

`trace.csv`, `candidates.csv`, `pairs.csv`, `timing.json`, `run_config.yaml`. This cost
`s3gfn_seh/seed43`'s entire training history (unrecoverable) and `s3gfn_seh/seed42`'s budget-faithful
2,000-molecule `candidates.csv` (overwritten at 20,000 rows). Mitigated in `3281bce` (TraceWriter
rotates) and `dd8f1a9` (Stage 2 snapshots `fixed_reward/` → `fixed_reward.budget_faithful/`).

**This threatens copy-forward directly:** a copied training run destroyed by a later runner
invocation destroys exactly what was copied. The structural fix is §5 step 3 — freeze `train/`
read-only after verification. Do not rely on remembering.

### 8.2 Fields whose names lie — check the writer before you quote the reader

Two near-misses on 2026-09-08 were the **measurement being right and the reading being wrong**, and
neither would have been caught by re-running anything: a cumulative field summed as if incremental,
and `19:18` read as minutes when it was hours. Both were caught only because the wrong number looked
surprising. The unsurprising instances of this class are still out there.

The defence is not vigilance, it is a lookup. Every field below has produced a wrong number in this
project at least once:

| field | reads as | actually is |
|---|---|---|
| `asked` (`upsample_log.json`) | this round's request | **cumulative** across rounds — summing it double-counts (24,000 reported for a cell whose distinct set grew by 1,216) |
| `used_rxns` | what the selection cost | **inflates** outside the budget-binding regime; quote `cost_kept_rxns` (65 molecules priced at 247 read as 300→387) |
| `n_modes` | modes delivered | modes **requested**; `n_targets_priced` is delivered (89 vs 100 on native routes) |
| `n_modes_kept` | same as `n_modes` | post-filter count — a co-agent lost a result to the difference |
| `n_scored` (`trace.csv`) | training oracle calls | **all phases**, eval interleaved; count `phase == "train"` ROWS, and note that `max(n_scored)` over filtered rows is still contaminated |
| `total_modes` | a total | capped at the 500-molecule prefix |
| `sample_s` / any timing component | 0.0 when absent | **absent ≠ zero** — a missing component must be omitted from the total, never written as 0.0, or an untimed stage becomes a free one |
| durations in prose | `20:10` = 20 min 10 s | on this cluster it is as likely **20 h 10 min**. Always write `19.30 h` |

**The rule:** read the field's writer before quoting its reader. The names lie by omission, and a
plausible reading of a plausible number is exactly what no re-run will catch.

### 8.3 Never edit a running bash script

bash resumes at a **byte offset**; an edit mid-job runs garbage hours later (this killed job 74318
five hours in, *after* its work had succeeded). The chain scripts snapshot their callee to `/tmp` for
this reason. Copy that pattern; do not reinvent it.

### 8.4 Shared scratch is rewritten by other agents

On 2026-08-19 all three `scent_seh` enumerations were re-run mid-experiment (a correct fix) *after* a
comparison arm had read the old ones — silently turning a same-chemistry comparison into a
cross-chemistry one, with both runs reporting success. The pools differed by 6× and nothing in the
outputs showed it. See §6.8.

### 8.5 Environment traps that a login smoke cannot catch

`$HOME` is read-only on compute nodes — export `TRITON_CACHE_DIR`, `MPLCONFIGDIR`, `XDG_CACHE_HOME`,
`HF_HOME`, `TORCH_HOME`, `SYNTHESEUS_CACHE_DIR` to `$SCRATCH` in every submit script. Any job that
docks must `source ~/bin/rgfn-smoke-env.sh` — omitting it leaves QuickVina2-GPU with three unresolved
boost libraries, which surfaces as all-`nan` and reads exactly like a degraded GPU.

### 8.6 Do not submit a cell another chain has already claimed

Chains claim cells before any file appears. Grep every queued job's `CELLS=` line first, and prefer
`scontrol hold` over cancel.

---

## 9. What gets rebuilt downstream

Re-running a cell invalidates everything derived from it. This is the dependency map, so nothing is
quoted from a mixed vintage.

| exhibit | depends on | status in v2 |
|---|---|---|
| Reaction-budget readout (the headline) | every reaction-GFN cell's campaign curves | rebuild; **note the headline moves 2.93× → 2.73×** once FragGFN leaves (verified 2026-08-28: 3 reaction-GFNs, n=23 comparable at R=100, range 1.67–3.56) |
| External head-to-head | competitor pools + routes + selection | rebuild — and on the **primary** axis. The committed figure is still on the secondary axis (reactions-for-100-modes) |
| Diversity-aware SPARROW comparison | our enumerations + competitor selection | rebuild |
| Ordering floor + ceiling | per-cell enumerations | rebuild; robust to the taxonomy change (median recovery 94.6% → 94.0%). **The claim gets stronger under `--pool all`**: v1 measured flow ordering over a reward-pre-filtered pool, so "flow does the selecting" was confounded. Re-run the arms on the unfiltered pool |
| Filter ablation | one cell's enumeration | rebuild; currently `scent_seh` seed 42 only |
| Two-knob surfaces | per-cell enumerations | rebuild at **R=100**; currently budget 300, sEH only, seed 42 |
| Cost-model audit vs SPARROW MILP | selected libraries | rebuild — CPU-cheap, and it is the credibility anchor |
| Compute-time accounting | per-stage timings | rebuild; `rgfn_6td3`'s v1 attribution is broken (all 184,069 s in `unattributed_s`) |
| Route dataset | ① + ② + ③ on every cell | **new capability** — matrix-wide instead of 5 cell-seeds, arm A only |
| Oracle calibration, hardware benchmarks | oracles + molecule sets + GPUs | **copy** (§0) |

---

## 10. Harvest contract to the publication repo

`AC-MedChem-SDL/RGFN-LSD` reads this tree through `tools/harvest/*.py`, already parameterised by
`--matrix-root` / `--campaign-root` / `--research-root`, so retargeting is a flag change. Each exhibit
must record, in `benchmark_v2/results/<exhibit>/PROVENANCE.md`: the script that produced it, the
artifact path it lands in, and the `reproduce/` script that consumes it. Note
`artifacts/reaction_budget/seed42/*_6td3/` in the publication repo is invalidated by the 6TD3-B swap.

Figures are build products and are never committed. Run `tools/check_identity.py --all` before
committing there — the default scans only *tracked* files, so a new file is invisible to it.

---

## 11. Open questions

1. **A second untrained signal for sEH / DRD2 / ClpP** (§7.1). 6TD3-B is settled, but its resolution
   made reward and gate the same column again, so even there the standing exploitation check needs a
   signal that is not a component of `cnn_vs` — `vina_t2` is the candidate. The other three targets
   have no such signal at all. Free for the docking targets; ~100 molecules/cell for sEH.
2. **How much of the delivered library sits on depth-0 (bought) hubs?** (§2.2) — unmeasured, and
   `--pool all` raises depth-0 exposure 4× in the walked prefix. v1's hub-ordering arms kept only
   plots, not the per-step curves, so the v1 baseline for this cannot be recovered. Answer it in the
   arm-A pilot, alongside the `--min-hub-depth 1` sensitivity arm.
3. **Do pruned pools change more widely than naive ones under the new gates?** (§4) — unmeasured.
4. **Arm B's downstream** — paused by decision; revisit only if arm A trains badly.
5. **SynFormer** — 1 of 9 cells; blocked on a worker-pool memory leak. If it stays blocked, the
   reaction-grounded / not-a-GFlowNet quadrant has one entrant and thin coverage.
