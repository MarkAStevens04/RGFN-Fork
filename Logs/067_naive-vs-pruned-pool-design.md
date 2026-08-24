# Competitor pipeline — two ways to pick the 500 molecules we price

> ⚠ **CORRECTION (2026-08-24). The headline number in this entry measured our own budget error, not
> a property of the baselines, and must not be quoted.** Every result below was computed from
> REINVENT running at **12.4x its authors' oracle budget** (batch 128 x 1000 steps = 124,587 distinct
> molecules, against the ~10,000 its paper uses; audit in docs/RESEARCH_CONTEXT.md, "Baseline configs
> audited against their authors' defaults"). Over-training collapsed REINVENT into a narrow
> high-reward band, and the "pruning is worth 2.3-2.9x" result is that collapse.
>
> Re-measured at the corrected budget, on the same cell and seeds:
>
> | REINVENT sEH | above gate | naive modes | pruned modes | lift | naive mean sim |
> |---|---|---|---|---|---|
> | old (124,587 calls) | 1,860 | 201 | 500 | **2.5x** | 0.40 |
> | new (10,048 calls)  | 464-728 | 369-417 | 417-500 | **1.0-1.36x** | 0.17-0.22 |
>
> On seed 44 the two pools are *identical*: its whole above-gate set is smaller than 500.
>
> **What survives.** The two-pool DESIGN survives, and so does the reason for it — but the finding is
> narrower and sharper than this entry claims. Pruning matters exactly where a generator genuinely
> collapses, and at their own authors' budgets only the Saturn-family entrants do: Saturn 13.7x (sEH)
> / 5.2x (DRD2) / 4.0x (ClpP), TANGO 7.4x / 2.7x / 3.3x, against REINVENT ~1.2x and S3-GFN 1.00x on
> sEH and ClpP. So the naive pool is the honest default, and it under-reports only for a
> mode-collapsed generator. Saturn's own numbers in this entry were never affected — it was already
> at its authors' budget and was never re-run.
>
> **What does not survive.** The "our advantage is ~3.7x naive / ~1.6x pruned" framing, and the claim
> that the pruned baseline overtakes our from-scratch arm. Both rest on the inflated REINVENT.
> Superseded by the re-measurement in progress; see the entry that follows this one.

**Date:** 2026-08-19, ~5pm (design + pool diagnostics); routed cost results added 2026-08-20, ~11am

## Question

When we take 500 molecules from a generator to cost out a synthesis campaign, does it matter whether
we take the 500 highest-scoring ones or the 500 that are actually different from each other?

## Context & Summary

**Context.** Every competitor cell in this benchmark works the same way: take a generator's output,
thin it to a 500-molecule pool, plan routes for the whole pool with MultiAiZ, then ask what a budget
of 100 reactions can buy. Until today "thin to 500" always meant *the top 500 by score*. Entry `062`
found that SPARROW's own diversity mechanism has edge effects, which raised the question of how much
of a generator's measured cost depends on how we picked its pool in the first place. The question had
teeth because of Saturn: its top-500 pool contains only **18 distinct molecules** on sEH — meaning 482
of the 500 are near-duplicates of one of the other 18 — and on that basis Saturn had been written off
as unpriceable, since planning routes for a pool that holds 18 deliverables measures almost nothing.

**Summary.** We now build the pool two ways and run *both* through the identical downstream pipeline.
The **naive pool** is the top 500 by score, with whatever diversity the generator's own machinery
happens to produce — the pipeline as a chemist would run it if they trusted the tool. The **pruned
pool** is 500 molecules guaranteed to be different from one another, found by walking further down
the score ranking and skipping anything too similar to something already taken. Each pool reports two
costs — reactions per molecule bought, and reactions per genuinely distinct molecule bought — plus the
average chemical similarity within the selection. This entry covers the design, the pool diagnostics,
the four infrastructure defects the design exposed, and the routed cost results from the overnight
campaign.

## Answer

**The pool construction, not the generator, decides whether a cell is measurable at all.** Saturn's
apparent mode collapse is not a failure of its search: the very same 2,000 molecules that yield 18
distinct ones at the top of the ranking contain **338** distinct ones overall, and its training
history contains 1,251. The collapse is concentrated in the highest-scoring band — the pressure that
makes Saturn efficient at finding high scores also makes its best molecules nearly all one family.
Pruning recovers those 338 from the samples we already had, without changing Saturn's configuration,
sampling more, or spending more route planning. Saturn is priceable after all.

That also means neither pool alone tells the truth. Reporting only the naive pool would say "Saturn
cannot produce diverse molecules", which is false. Reporting only the pruned pool would hide a real
effect that a chemist taking the top 500 would actually hit. Both numbers are needed to say either.

**And the choice of pool changes the paper's headline, not just a baseline's footnote.** Pruning is
worth 2.3-2.9x to the competitor on the same generator and seed. Measured against the naive pool our
advantage reads about 3.7x; measured against the pruned pool — the baseline's best case, built with
our own diversity rule — it is about 1.6x. On the harder footing where both sides re-discover their
routes from scratch, the competitor's pruned pool overtakes our hub-batching arm outright. That is a
number a reviewer can derive from our own tables, so we report it rather than wait to be asked.

## Relevance to our Publication

Two ICLR reviewer objections cancel each other, and running both pools is what closes them together.
Reporting only the naive pool invites "you never let the baseline be diverse"; reporting only the
pruned pool invites "you did the hard part for the baseline by hand". Running both, with the same rule
applied to every generator including ours, answers each with the other. It also converts our single
weakest cell from an exclusion into a datapoint: an excluded baseline reads as a baseline that was not
given a fair run, and we no longer have to make that argument. Separately, it supplies a pipeline
correctness check with real teeth — on a pruned pool the two cost metrics are computed by different
code paths and must agree, so a disagreement invalidates every reactions-per-mode figure in the paper.

## Next Experiments

**Refining for publication.** Three seeds of both pools on both targets is the full deliverable;
sEH seeds 42/43 are in for REINVENT and one seed for Saturn, and the gaps are listed under "Still
open". The comparison that now needs the most care is the one this entry surfaced: our advantage is
~3.7x against the naive pool and ~1.6x against the pruned one, and on the from-scratch footing the
pruned baseline wins outright — so the paper needs to fix which footing is the headline and say why,
rather than letting a reviewer pick. Reviewers will also want the price of pruning stated rather than
assumed:
MultiAiZ plans a whole batch at once and converges near-duplicates onto shared intermediates, so a
deliberately diverse pool may genuinely cost more per molecule, and our own side never pays that cost
because our routes already exist. The REINVENT sEH cell is running both pools specifically to isolate
that effect. One asymmetry needs stating in the paper: a pruned pool short of 500 (Saturn's 338) is
reported at its true size, not padded.

**Next steps in project.** Extend both pools across the remaining cells and seeds, and settle whether
Saturn's Beam Enumeration question still matters now that the pruned pool makes Saturn measurable
without departing from its authors' configuration.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/lsd_hubs/campaign/mode_saturation.py` — counts how many distinct molecules a pool
  contains as a function of pool size. Rewritten here: it used to be a go/no-go gate that refused any
  pool short of a 100-mode target, which under the current reaction-budget readout would delete
  legitimate datapoints. It now predicts the stop reason and aborts only below `--min-modes`.
- `./experiments/lsd_hubs/campaign/build_s3gfn_pools.py` — builds the pool handed to MultiAiZ. Gained
  `--pruned` / `--cutoff`, and now emits a short pruned pool at its true size instead of skipping it.
- `./experiments/lsd_hubs/campaign/submit_competitor_routes.sh` — the per-cell gate → pool → MultiAiZ →
  select pipeline. Gained `POOL={naive,pruned}`, gate flag passthrough, and resolution of the actual
  emitted pool directory (a pruned pool may be `_N338` when `_N500` was requested).
- `./experiments/lsd_hubs/campaign/submit_sb_readout.sh` — **new.** Runs only the SPARROW-Batching
  solve over pools whose routes are already cached, so a night of readouts is one job. Exists because
  these solves cannot run on a login node (see Method 5).
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — unchanged here; the source of the two
  cost metrics and the similarity columns.
- `./validation/lsdflow/metrics/diversity.py` — unchanged; the canonical mode metric (Morgan r=3/2048,
  greedy sphere exclusion, best-score-first) used by every number in this entry.

**Datasets**

- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/{saturn,reinvent,s3gfn}_{seh,drd2}/seed*/fixed_reward/candidates/candidates.csv`
  — each generator's pool source: 2,000 molecules freshly sampled from its *trained final policy*.
  This is the protocol behind all sixteen matrix cells, which is why both pool variants draw from it.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/saturn_seh/seed42/oracle_history.csv` — the
  10,020 molecules Saturn scored *during training*. Used only as a diagnostic, deliberately not as a
  pool source: it is a different object, unavailable in comparable form for our own generators.

**Results**

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/{reinvent_seh_seed42_pruned_N500,saturn_seh_seed42_pruned_N338}/`
  — the two pruned pools built here, each with `pool.smi`, `pool_scores.csv`, and (pruned only) a
  `pool_meta.json` recording requested vs written size, scan depth, and the `pool_limited` flag.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/*_saturation/summary.json` — per-cell pool
  diagnostics including the predicted stop reason.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/cr_rnv_pruned-74439.out` — REINVENT sEH seed 42, pruned pool (running).

## Relevant Versions

`2b7c39c` is HEAD. The files in this entry are **not yet committed** (branch `Hub-Analysis`):

```
 M CLAUDE.md
 M docs/RESEARCH_CONTEXT.md
 M experiments/lsd_hubs/campaign/build_s3gfn_pools.py
 M experiments/lsd_hubs/campaign/mode_saturation.py
 M experiments/lsd_hubs/campaign/submit_competitor_routes.sh
 M experiments/lsd_hubs/campaign/submit_competitor_routes_chain.sh
?? experiments/lsd_hubs/campaign/submit_sb_readout.sh
```

[TODO — add commit hash after pushing]

## Relevant Resources

**Sources**

- `docs/RESEARCH_CONTEXT.md` § "The two pools and the two numbers (decided 2026-08-19)" — the design
  this entry implements, including the pool-source decision and the correctness check.
- `CLAUDE.md` § "THE BENCHMARK'S PRIMARY READOUT IS A FIXED REACTION BUDGET" — the readout convention
  that made the old saturation gate wrong.
- Entry `062` — SPARROW's diversity mechanism and its edge effects; the origin of the question.
- Entry `056` — the source of the regression pin used to verify the rewritten gate.

**Packages**

- RDKit (Morgan fingerprints, Bemis-Murcko scaffolds) — via `validation/lsdflow/metrics/diversity.py`.
- PuLP + CBC (the SPARROW-Batching MILP) — invoked by `sparrow_select_frontier.py`, in the `sparrow`
  conda env.

## Method

1. **Established what `candidates.csv` actually is**, since the pool-source question turned on it.
   `run_reinvent_fixed.py:301-306` draws a fresh sample from the trained agent; `run_saturn_fixed.py:285`
   passes `n_target=n_samples`. It is a sample from the final policy, not a slice of training history —
   so both pool variants draw from it, uniformly across every generator.

2. **Measured both pool ceilings for Saturn sEH seed 42**, counting distinct molecules with the
   canonical metric at gate 7.0 / τ 0.5 over (a) `candidates.csv` and (b) `oracle_history.csv`. The
   history's raw score lives in `glue_surrogate_raw_values`; its `reward` column is Saturn's shaped
   0-1 transform, and reading that column instead returns zero molecules above a gate of 7.0.

3. **Built the pruned pools** with the new `--pruned` flag and confirmed the construction does what it
   claims — re-running the mode metric over the emitted REINVENT pool returns 500 modes from 500
   molecules.

4. **Verified the naive path was untouched**, via `build_s3gfn_pools.py --verify` against the existing
   `reinvent_seh_seed42_N500` pool: identical set, identical scores.

5. **Attempted the SPARROW-Batching readout for REINVENT sEH seed 42 on the login node** and found it
   cannot run there. CBC held 99.8% CPU for 16 minutes on R=50 alone without returning, against a login
   `ulimit -t` of 3600 s — so the process would have been killed by SIGXCPU at roughly one hour, with
   no CSV written. Killed and moved to batch as `submit_sb_readout.sh`.

6. **Re-ran the regression pin** on the rewritten gate (`mode_saturation.py` against the S3-GFN sEH
   seed-42 pool) to confirm the mode metric had not moved.

## Results

**Pool ceilings** (sEH, gate 7.0, τ = 0.5, seed 42). "Modes" = mutually distinct molecules under the
canonical metric.

| entrant | above gate | modes in naive top-500 | modes available in whole set | pruned pool built |
|---|---|---|---|---|
| REINVENT | 1,860 | 201 | 963 | 500 (scan depth 1,127; score 8.357 → 7.761) |
| S3-GFN | 500 | 206 | 732 | 500 |
| Saturn | 1,911 | **18** | **338** | 338 — short of 500, flagged `pool_limited` |

**Saturn sEH seed 42, pool source comparison.** The two sources agree almost exactly on the naive
number, which is why the pool-source choice does not affect any reported result:

| source | distinct molecules | above gate 7.0 | total modes | modes in top-500 |
|---|---|---|---|---|
| `candidates.csv` (final-policy samples) | 2,000 | 1,911 | 338 | 18 |
| `oracle_history.csv` (training) | 10,020 | 8,747 | 1,251 | 16 |

Overlap between the two sets is **0 molecules**, as expected for a fresh sample from a trained policy.
Reaching 100 modes takes a scan depth of 1,291 candidates from the samples, or 3,089 from the history.

**Gate behaviour after the rewrite** (predicted stop reason at a 100-reaction budget):

| cell | pool | modes in pool to be built | prediction |
|---|---|---|---|
| Saturn sEH 42 | naive | 18 | pool-exhausted expected |
| Saturn sEH 42 | pruned | 338 | budget-binding possible |
| REINVENT sEH 42 | naive | 201 | budget-binding possible |
| REINVENT sEH 42 | pruned | 500 | budget-binding possible |

Saturn's naive cell is now *run and flagged* rather than excluded; under the old gate it exited 2.

**Regression pin held exactly.** S3-GFN sEH seed 42: 206 modes at n=500, mode rate 0.412, 100 modes
first reached at 250 candidates — bit-identical to entry `056`.

**MILP tractability is pool-dependent, not budget-dependent.** This qualifies a claim recorded in
`CLAUDE.md`, which had generalised from S3-GFN alone:

| pool | targets | compound nodes | reaction nodes | intermediates | R=50 solve |
|---|---|---|---|---|---|
| S3-GFN sEH | 451 | 4,500 | 7,783 | 2,943 | `Optimal`, 6-57 s |
| REINVENT sEH | 459 | 7,256 | 10,762 | **5,597** | no return after 16 min CPU |

Nearly the same target count, ~1.9× the intermediates. A time-capped row is a lower bound on the
competitor — i.e. it flatters us — so `time_capped` must be read before any SB row is called optimal.

**S3-GFN's primary readout already exists** and needed no recomputation (all `Optimal`, all spending
the full budget): 24 modes at R=100 (seed 42), 37 (seed 43), 32 (seed 44). REINVENT has no SB result
at any budget yet, which is the gap the queued jobs fill.

### Routed cost results (added 2026-08-20 after the overnight jobs)

**Modes at the primary readout — 100 reactions, sEH, gate 7.0, tau 0.5.** `C` marks a time-capped
solve, which is a LOWER bound on the competitor. Every row spent the full 100 reactions.

| generator | seed | pool | modes | rxn/candidate | rxn/mode | mean pairwise sim | |
|---|---|---|---|---|---|---|---|
| S3-GFN | 42 | naive | 24 | 1.12 | 4.17 | — | |
| S3-GFN | 43 | naive | 37 | 1.11 | 1.27 | 0.257 | |
| S3-GFN | 44 | naive | 32 | 1.15 | 1.34 | 0.274 | |
| REINVENT | 42 | naive | 16 | 1.79 | 2.56 | 0.400 | C |
| REINVENT | 43 | naive | 22 | 1.82 | 2.50 | 0.322 | C |
| REINVENT | 44 | naive | 23 | 2.04 | 2.52 | 0.355 | C |
| **REINVENT** | **42** | **pruned** | **46** | 2.17 | 2.17 | 0.229 | C |
| **REINVENT** | **43** | **pruned** | **51** | 1.96 | 1.96 | 0.222 | |
| Saturn | 43 | naive | **3** | 1.32 | 6.67 | 0.500 | |
| **Saturn** | **42** | **pruned** | **38** | 2.63 | 2.63 | 0.239 | C |

**Pruning is worth 2.3-2.9x to the competitor**, paired on the same generator and seed: REINVENT
seed 42 goes 16 → 46 modes (2.88x), seed 43 goes 22 → 51 (2.32x). Mean pairwise similarity separates
cleanly by pool — naive 0.257-0.500, pruned 0.222-0.239 — so the pruned pool is measurably more
diverse and not merely relabelled. Saturn's naive seed 43 is the extreme: **3 modes** at similarity
0.500, against 38 on the pruned pool.

**The correctness check passed, and its failure mode turned out to be informative.** On a pruned pool
`n_selected == n_modes_kept` in every row, and the two independently-computed cost metrics agree to
the digit **at every budget-binding point** (gap exactly 0 at R=50/100/150/200 on both pruned cells).
They diverge only past pool exhaustion, where SPARROW's reported `used_rxns` keeps climbing while the
true cost of the identical selected set stays flat — Saturn holds all 65 of its routed molecules from
R=300 on, priced at **247** reactions, while `used_rxns` reads 300 / 322 / 357 / 388 / 387 / 387 across
R=300...1000. Up to **140 reactions of slack**, and non-monotonic. So `used_rxns` is the right
stop-reason diagnostic but the wrong cost number outside the budget-binding regime, where it
overstates the competitor's spend.

**Routability is generator-specific, and it is the pruned pool's real cost.** REINVENT's pruned pool
routes as well as its naive one (90.6% vs 91.8-95.2%), because it only had to scan 1,127 of 1,860
candidates to find 500 distinct ones. Saturn's collapses to **19.2%** (65 of 338) against 55.2% naive,
because reaching 338 distinct molecules meant scanning its entire above-gate set down to a score of
7.017. Pruning is close to free where diversity is plentiful and expensive where it is scarce.

**Where our own side sits, on two different footings.** Both at gate 7.0, tau 0.5, R=100:

| arm | modes | cost model |
|---|---|---|
| Ours — hub-batching, native routes, free-frag | **82** | count-once, our routes already exist |
| Ours — best-candidate baseline, same footing | 29 | count-once |
| Competitor best — REINVENT pruned + SPARROW | 46-51 | MultiAiZ from scratch, priced |
| Ours — hub-batching through the SPARROW evaluator, `from_scratch` | 31 | routes re-discovered, hub structure dissolved |
| Ours — best-candidate, `from_scratch` | 26 | as above |

**The pruned pool moves the competitor from behind both of our numbers to between them.** Against the
naive pool our headline advantage read ~3.7x (82 vs 22); against the pruned pool it is **~1.6x**
(82 vs 51). And on the chemistry-homogenized footing, where both sides re-discover routes from
scratch, the competitor's pruned pool (51) now **beats** our hub-batching arm (31) — a comparison a
reviewer can compute from our own tables, so it has to be ours to state first. The two footings are
different cost models and must not be mixed in one figure; the native-route asymmetry is a reported
result, not a hidden subsidy, but the size of the advantage now depends on which footing is quoted.

**Still open.** Saturn's naive seed-42 cell and REINVENT's pruned seed 44 lost their route discovery
to walltime; Saturn's pruned seeds 43/44 are unrun; no DRD2 pruned pools exist. Saturn's pruned rows
and REINVENT's naive rows are time-capped lower bounds and want a longer `--max-seconds` before being
quoted as optima.

**Jobs queued/running at time of writing:** 74439 (REINVENT sEH 42 pruned, routing), 74457 (Saturn sEH
42 pruned), 74458 (Saturn sEH 42 naive), 74459 (Saturn naive, seeds 43/44 + DRD2 42), 74460 (REINVENT
sEH pruned, seeds 43/44), 74461/74462 (REINVENT SB readout at R=100 on cached routes).
