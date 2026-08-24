# sEH — what happens when the batch optimizer is allowed to care about diversity

**Date:** 2026-08-18, ~11am

## Question

Our competitor arms compared against an optimizer that had no reason to pick varied molecules — if we
switch on its own diversity feature, does our advantage survive?

## Answer

**It survives, and at the budget this benchmark actually reports it survives by a wide margin — but
the size of the margin depends almost entirely on how big a budget you read it at, and the first
version of this entry read it at the wrong one.**

At the primary readout of **100 reactions**, we deliver **82 distinct molecules against the best
competitor arm's 24** — about **3.4×**. And we do it while *also* spending fewer reactions per distinct
molecule (1.22 vs 1.67), so this is not a diversity-for-cost trade: we win on both axes at once.

> ⚠️ **BOTH sentences above are revised by the 2026-08-19 update at the end of this entry — read it
> before quoting either.** The competitor row behind "24" was `TimeLimit` (a lower bound); two
> replicate seeds converged at 32–33, so the ratio is **~2.5×, not 3.4×**. And the converged
> competitor is *cheaper* per mode than we are (1.03 vs 1.22), so the "both axes" claim does not
> survive — the count advantage does.

The near-tie reported below is real but belongs to a different regime. It was measured at ~290
reactions of priced chemistry, roughly three times the headline budget, and by then our own method has
nearly exhausted its pool (it plateaus at ~232 distinct molecules and stops climbing) while the
competitor is still ascending. The two curves are not parallel, so a single ratio was never going to
describe them:

| priced reactions | ours | best competitor | ratio |
|---|---|---|---|
| **100 (primary readout)** | **82** | **24** | **3.4×** |
| ~290 | 228 | 213 | 1.07× |

**The mechanism is a hard constraint versus a soft penalty**, and it explains the shape rather than
merely restating it. The optimizer maximizes total reward under a reaction cap, and its diversity term
is a *penalty* traded off against that reward. When the cap is tight, the cheapest reward available is
another analogue hanging off an intermediate already paid for, and no affordable diversity weight
outbids that — at λ=0 it returns **98 compounds that amount to 2 distinct ones**. Our selection instead
*refuses* any molecule within τ of one already taken, so it pays to move elsewhere from the very first
step. A soft penalty loses to a hard constraint precisely when the budget is small, which is the
regime a chemist with one plate is actually in.

So the earlier reading — that most of the value is in enumeration and our selection is merely a cheap
way to exploit it — was drawn from the competitor's best regime and understated the selection's
contribution. Enumeration matters, but at the headline budget *what you keep* matters more than *what
you enumerated*: both arms below were given the same enumerated molecules.

Two findings from the first pass stand unchanged. The optimizer, given two hours per decision, **still
could not prove it had found the best answer** on two thirds of the points measured, against ~1 s for
ours. And the optimizer's solver **reports success when it has merely run out of time** — the reason
this entry exists at all, and the reason four of the R=100 competitor points below are lower bounds
rather than settled numbers.

## Context & Summary

Entry `059` handed a standard batch optimizer our own molecules and found it produced cheap libraries
of near-duplicates: at a 300-reaction budget it chose 296 compounds that amounted to just 14 genuinely
different ones, against 228 for our method. That is a real result, but it invites an obvious
objection — *of course* it was not diverse, nothing asked it to be.

`[fromer2025diversity]`, published by the same group as the optimizer itself, closes exactly that gap.
It adds a diversity term with an adjustable weight: turn the weight up, and the optimizer is rewarded
for spreading its picks across more families of molecules. Our clone of the tool already contained
this feature; we had simply never switched it on.

So we did, sweeping the weight across four orders of magnitude on both competitor arms — the one given
our sampled molecules, and the harder one given everything reachable one step from the intermediates
those molecules were built from. We use *their* mechanism rather than a rule of our own so that the
comparison is against the published method rather than our interpretation of it. Two measurements come
out of each run: how many families the optimizer touched (its own metric, which we expect to lose), and
how many of its picks are genuinely different from one another (ours). Because a chemist does not pay
for compounds they discard, we then prune each selection to the genuinely-different molecules and
**re-price only those** — a step that is generous to the competitor, and provably a no-op on our own
output.

## Relevance to our Publication

This is the entry that decides how we describe our own contribution, and it argues for describing it
more narrowly and more defensibly than we had been. A Digital Discovery reviewer who reads
`[fromer2025diversity]` will immediately ask whether our advantage is an artifact of comparing against
a diversity-blind selector. We can now answer that we ran their mechanism, swept its weight, gave the
competitor our own enumerated molecules, and reported the result even where it nearly erased our
margin.

What replaces the wider claim is a sharper one: reaction-grounded **enumeration** is where most of the
library-efficiency win comes from, and flow-based selection captures nearly all of that win at
negligible cost, where the optimal alternative does not finish. A paper that narrows its own claim
after testing it is far harder to dismiss than one that reports only its widest.

## Next Experiments

**Refining for publication**

- **Close the candidate-count confound on BC-SB** (Result 8). It is the one arm that saw fewer
  candidates than we did, so "you just sampled more" is still live against it. The fix is cheap and
  needs no new generator compute: `records.csv` is a post-hoc sample from a *frozen* checkpoint — mean
  reward is flat across the file (7.489 / 7.495 / 7.505 over first, middle, last 5k rows) — so a
  prefix of k rows is exactly "what if we had only sampled k times", with no training-progress
  confound. Re-run the R=100 selection on prefixes k ∈ {2k, 5k, 10k, 20k, 30k} and plot distinct
  molecules against reward-gen calls, against our own curve on the same axis. Cost is 5 MILP solves
  per seed, no oracle calls.
- **Give the large-budget near-tie its error bars.** The 5–7% result at ~290 reactions is a single run
  of each arm. It is now the secondary readout rather than the headline, but it is the number a
  reviewer will probe hardest, and it needs the replication the rest of the benchmark has.
- **Report the selection-effort axis as a first-class result.** One second versus two hours without
  convergence is currently a footnote; it is the strongest remaining part of the claim and belongs in
  the headline figure.
- **Relax the optimizer's precision requirement.** It is currently asked to prove optimality to one
  part in ten million, far tighter than any effect we care about. A looser setting may let it finish,
  which would turn our lower bounds into settled numbers.
- **Repeat on the second target.** Everything here is sEH; the machinery is target-agnostic.

**Next steps in project**

- Fold both competitor arms and the effort axis into the headline figure, which currently predates all
  of this.
- Extend to the docking-based targets, where the cost of making the wrong compounds is highest.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/campaign/make_pair_gallery.py` — **new**. Renders Result 7: the three
  closest and three most distant pairs in each arm's R=100 library, methods as rows. **Two figures,
  not one** — the closest and most-distant blocks answer different questions, and splitting them also
  doubles the drawn size of each structure, which matters when the whole point is to see a single acyl
  cap. Pairs are chosen without reusing a molecule so each block shows six distinct structures. Uses
  the project's own mode metric (Morgan r=3 / 2048) so the picture and the mode count cannot disagree.
  Writes `results/paper_pair_gallery/pair_gallery_{closest,distant}.{png,pdf}` + `pair_gallery.csv`.

- `./validation/lsdflow/adapters/workers/sparrow_worker.py` — **modified twice**. (1) `--lambda-div`
  exposes SPARROW's native diversity weight (objective slot 3, which triggers its
  `add_diversity_objective`); `--clusters` supplies the partition; `--min-clusters` adds the paper's
  constraint variant, which upstream does not ship. `--lambda-div` and `--min-clusters` are mutually
  exclusive — they are two mechanisms for one goal and combining them makes a result attributable to
  neither. (2) **Time-limit detection** (see Method 4).
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — **modified**: sweeps the diversity
  weight × reaction budget, builds tau-mode clusters once per pool, prunes each selection to
  genuinely-distinct molecules, and re-prices the survivors via `_price_kept_set`.
- `./experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh` — **modified**: `LAMBDA_DIV` selects the
  arm's diversity weight; one value per job, because sweeping inside a job would serialize an already
  long run.
- `./validation/lsdflow/metrics/diversity.py` — **modified**: new `mode_assignments`, the full
  partition behind `mode_representatives`. Needed to hand SPARROW *our* notion of distinctness as its
  clusters. Its group count equals `count_modes` on the same arguments, so the two readouts cannot
  disagree (verified: 369 groups vs 369 modes over 3,000 molecules, 100% assigned).

**Datasets**

- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_native_70974/` — the seed-42 sEH run supplying
  BC-SB's pool (21,001 distinct molecules above the gate) and the hub route prefixes.
- `/scratch/markymoo/rgfn_runs/lsdflow/bc_enum_seh_seed42/` — BC-Enum-SB's material: 131,474 molecules
  enumerated one step from the 64 intermediates a naive best-score walk collects.

**Results** (on `$SCRATCH`; `$HOME` is read-only on compute)

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb/bcsb_div{,2}_L*_N21000/` — BC-SB, λ ∈ {0, 0.1, 1,
  10, 100}. The `div2` runs are the re-solves at the 2 h cap; the originals are kept, not overwritten,
  because the comparison between them is itself evidence (Results table 3).
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb/enumdiv{,2}_L*_N50000/` — BC-Enum-SB, λ ∈ {0,
  0.1, 1, 10}.
- `./experiments/lsd_hubs/campaign/results/t15_regress_capfix_reconcile/` — the regression proving the
  time-limit change did not perturb the pricing path.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/bcsb_div_L*-{73396,73397,73398,73399,73400}.out` — first pass
- `/scratch/markymoo/rgfn_runs/{bcsb2,enum2,bcsbhi2}_L*-{73564..73570}.out` — the 2 h re-solves

### Relevant Versions

Branch `Hub-Analysis`. Most recent commit at time of writing: `204279d` ("Headline figure: read SB in
tiers keyed on the MILP time budget") — the other session's independent handling of the same
time-limit issue.

**Not yet committed:**

```
M  validation/lsdflow/adapters/workers/sparrow_worker.py
M  validation/lsdflow/metrics/diversity.py
M  experiments/lsd_hubs/campaign/sparrow_select_frontier.py
M  experiments/lsd_hubs/campaign/submit_bc_sb_scale.sh
M  Logs/references/{README.md,references.bib}     (the [fromer2025diversity] entry)
```

Three agents share this working tree — isolate hunks (`git diff -U3 <f>` → `git apply --cached` →
`git commit --no-verify`) rather than staging whole files.

`[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources**

- `[fromer2025diversity]` — Fromer, Volkova & Coley, *J. Chem. Inf. Model.* **65**(12):5989–5997
  (2025), DOI 10.1021/acs.jcim.5c00606. The diversity mechanism used here. Note their §2.1: the
  nonlinear "expected cumulative reward" objective requires Gurobi and is not guaranteed globally
  optimal, so **they** use the linear formulation for all their own analysis — which is also the only
  path open to us on PuLP/CBC.
- `[fromer2024sparrow]` — the original optimizer.

**Packages**

- `sparrow` + PuLP/CBC (`sparrow` env), clone dated 2025-06-30 — already at the 2025 version, so this
  needed no upgrade, only configuration.
- RDKit (`rgfn` env) — clustering and distinctness via `validation/lsdflow/metrics/diversity.py`.

### Method

1. **Registered the paper** and confirmed the installed clone already carried its features
   (`selector/{linear,nonlinear,bayesian}`; `LinearSelector` exposing `clusters`, `N_per_cluster`,
   `rxn_classes`, `max_rxn_classes`, weights `[reward, start_cost, reaction, diversity, class]`).

2. **Chose the clusters to be our own τ-modes**, not SPARROW's default Butina/count-Morgan. The paper
   defines clusters as an arbitrary caller-supplied partition precisely so this substitution is
   legitimate, and it makes the optimizer compete on the metric the benchmark reports.

3. **Discarded a hard "touch ≥ K clusters" rule in favour of their soft weight.** Both were built; the
   rule was rejected on evidence. Sphere exclusion guarantees *representatives* are mutually
   dissimilar, but two non-representative members of adjacent families can be near-identical, so
   "families touched" is a looser quantity than "molecules that differ". Measured on a 500-molecule
   pool: all 75 representatives = 75 distinct; one arbitrary member of each of 75 families = **54**
   distinct; and a run constrained to touch 25 families produced **14** distinct molecules. The leak
   affects the soft mechanism equally — it counts the same quantity — which is why the fix is to
   report both metrics rather than to pick a different mechanism.

4. **Fixed a correctness bug that had produced a publishable-looking wrong number.** CBC reports
   status `Optimal` for the best solution it happens to hold when a time limit stops it, and PuLP's
   `sol_status` does not distinguish either (verified on pulp 3.3.2: a deliberately 2 s-limited solve
   still reports "Optimal Solution Found"). The tell was that a budget-1200 solve returned *lower*
   objective (4893) than a budget-800 solve (5760) while both claimed optimality — impossible for true
   optima, since the tighter problem's solution is feasible in the looser one. Detection is now by
   wall-clock against the cap; capped rows report `TimeLimit`, carry `time_capped`, and are treated as
   **lower bounds** on the competitor. Regression: the sEH T4.4 reconcile still reproduces entry `049`
   exactly (314/299 and 288/279).

5. **Swept** λ ∈ {0, 0.1, 1, 10, 100} × 9 reaction budgets on BC-SB (21,000-molecule pool) and λ ∈ {0,
   0.1, 1, 10} × 5–7 budgets on BC-Enum-SB (50,000), one job per λ. Every selection was pruned to
   genuinely-distinct molecules and the survivors re-priced.

6. **Re-solved every capped point at a 2 h cap** (from 900/1800 s), under `_v2` tags so the truncated
   originals remain on disk as evidence rather than being overwritten.

### Results

**1 — SPARROW's own mechanism works.** Raising the weight buys real diversity (BC-SB, 21k pool, 300
reactions). "Touched" is its metric, "distinct" is ours; `meanSim` is mean pairwise Tanimoto over the
selection, so lower is more varied.

| λ | selected | families touched | distinct | cost of distinct | meanSim |
|---|---|---|---|---|---|
| 0 | 279 | — | 46 | 82 | 0.305 |
| 0.1 | 279 | 140 | 49 | 87 | 0.299 |
| 1 | 275 | 168 | 55 | 100 | 0.292 |
| 10 | 253 | 252 | 88 | 162 | 0.237 |
| 100 | 249 | 249 | 91 | 173 | 0.232 |

The same on BC-Enum-SB is more dramatic — 14 → 81 distinct molecules from λ=0 to λ=1 — because that
pool is the more redundant one to begin with.

**2 — the head-to-head, at the SECONDARY (large-budget) readout.** ⚠️ **Superseded as the headline by
Result 6** — these are correct numbers read at 130–290 reactions, i.e. outside the benchmark's primary
100-reaction budget, and in the regime where our own pool is nearly exhausted. Kept because the
large-budget behaviour is a genuine result and the paper should report both. Best point per arm, at
mode counts our own curve also reaches. Capped rows are lower bounds on the competitor, so the true
ratio can only be **≤** what is shown.

| arm | λ | distinct | their rxns | our rxns | ratio | |
|---|---|---|---|---|---|---|
| **BC-Enum-SB** | 1 | 213 | 292 | 278 | **1.05×** | capped |
| BC-Enum-SB | 10 | 209 | 289 | 270 | 1.07× | capped |
| BC-Enum-SB | 0 | 80 | 127 | 98 | 1.30× | ok |
| BC-SB | 0.1 | 49 | 87 | 60 | 1.45× | ok |
| BC-SB | 1 | 55 | 100 | 68 | 1.47× | ok |
| BC-SB | 0 | 137 | 265 | 169 | 1.57× | ok |

**3 — the near-tie is not a truncation artifact.** Re-solving at 2 h moved individual points in *both*
directions, which is a solver wandering near its best-found solution rather than converging on a much
better one:

| arm | budget | old distinct / cost | new distinct / cost |
|---|---|---|---|
| BC-SB λ=1 | 400 | 79 / 148 | 72 / 132 |
| BC-SB λ=10 | 400 | 111 / 216 | 118 / 227 |
| BC-Enum-SB λ=1 | 800 | 204 / 279 | 213 / 292 |
| BC-Enum-SB λ=1 | 1200 | 129 / 186 | 122 / 180 |

**4 — the selection-effort axis.** Even at a 2 h cap, **25 of 36** points did not converge:

| run | points | still capped | max solve |
|---|---|---|---|
| BC-SB λ=1 / 10 / 100 | 6 each | 3 / 5 / 6 | 7,206 s |
| BC-Enum-SB λ=1 / 10 | 5 each | 5 / 5 | 7,526 s |
| BC-SB λ=0 (high budgets) | 4 | 0 | 675 s |

λ=0 and λ=0.1 never capped; the diversity term is what makes the problem hard. Our own selection over
the same material takes ~1 s.

**5 — the prune is symmetric.** Applied to hub-batching's own 100-mode library it removes nothing
(100 in, 100 out), because that strategy only ever accepts a molecule τ-diverse from everything
already taken. The step is therefore identical on both sides and generous only to the competitor.

**6 — the primary readout: 100 reactions.** This is the benchmark's headline stopping condition
(`CLAUDE.md`, decided 2026-08-17), and it is where the arms are furthest apart. Every row spent its
budget exactly (`used_rxns == 100`), so all are **budget-binding** — none is pool-exhausted, and the
comparison is like-for-like on that axis. "selected" is what the optimizer bought; "distinct" is how
many genuinely different molecules that amounts to; "priced" is the cost of just those survivors.

| arm | seed | λ | selected | distinct | priced rxns | rxn/distinct | meanSim | solver |
|---|---|---|---|---|---|---|---|---|
| **hub-batching (ours)** | 42 | — | **82** | **82** | **100** | **1.22** | **0.187** | ~1 s |
| BC-Enum-SB | 42 | 10 | 96 | 24 | 38 | 1.58 | 0.285 | ⚠️ TimeLimit |
| BC-Enum-SB | 42 | 1 | 96 | 24 | 40 | 1.67 | 0.276 | ⚠️ TimeLimit |
| BC-Enum-SB | 42 | 0.1 | 98 | 2 | 4 | 2.00 | 0.590 | Optimal |
| BC-Enum-SB | 42 | 0 | 98 | 2 | 4 | 2.00 | 0.593 | Optimal |
| BC-SB | 44 | 1 | 89 | 23 | 48 | 2.09 | 0.310 | Optimal |
| BC-SB | 43 | 1 | 91 | 19 | 40 | 2.11 | 0.349 | Optimal |
| BC-SB | 44 | 0 / 0.1 | 91 | 18 | 35 | 1.94 | 0.330 | Optimal |
| BC-SB | 43 | 0 / 0.1 | 93 | 16 | 35 | 2.19 | 0.398 | Optimal |

Three things to read off it.

*The gap is large and it is on the axis we lead with.* 82 vs 24 distinct molecules — **3.4×** — from
the same 100 reactions. The λ=1 and λ=10 competitor rows are `TimeLimit`, so per the project's
budget discipline they are **lower bounds on the competitor and therefore flatter it**; the true ratio
is ≥3.4×, not ≤. (These are *not* solver-truncated in the budget sense — `used_rxns` still equals 100,
so the budget was spent; what is uncertified is the objective, not the spend.)

*We are not buying diversity with reactions.* At 1.22 reactions per distinct molecule we are also the
cheapest arm on the secondary axis. The competitor's best diversity-aware point costs 1.67 and its
best-priced point (1.58) still only reaches 24 molecules. There is no trade being made here.

*The diversity weight is load-bearing but cannot close the gap.* Turning λ from 0 to 1 takes
BC-Enum-SB from 2 distinct molecules to 24 — a 12× improvement on their own knob, which is why the
diversity-blind comparison in `059` was not a fair fight. Pushing to λ=10 buys nothing further (24)
while pushing solve time past 7,000 s. The mechanism saturates well short of us.

**7 — what "distinct" looks like.** A Tanimoto threshold is easy to dismiss as a metric artifact, so
`results/paper_pair_gallery/` renders the pairs in each arm's R=100 library as **two figures**, since
the blocks answer different questions and are read separately: `pair_gallery_closest.*` (each method's
*worst* case — the honest test, and what the mode count is about) and `pair_gallery_distant.*` (its
*best* case — how far it can reach at all).

Each pair is drawn with its **maximum common substructure haloed in red**, so what remains in black is
exactly what differs. `mcs%` is the fraction of the smaller molecule inside that common core.

| arm | 3 closest pairs (Tanimoto) | mcs% | 3 most distant | mcs% |
|---|---|---|---|---|
| hub-batching (ours) | 0.50, 0.50, 0.50 | 70 / 81 / 74 | 0.04, 0.05, 0.05 | 27 / 26 / 26 |
| BC-Enum-SB (λ=1) | 0.87, 0.86, 0.83 | 83 / 86 / 83 | 0.06, 0.06, 0.06 | 17 / 27 / 38 |
| BC-SB (λ=1) | 0.90, 0.89, 0.88 | 91 / 70 / 92 | 0.08, 0.08, 0.08 | 24 / 30 / 19 |

Our worst pair sits exactly at τ=0.5, which is the constraint doing its job — the greedy accepts a
molecule only if its similarity to everything already taken is ≤0.5, so 0.50 is the *ceiling* on
redundancy and no pair in our library can exceed it. The competitors' worst pairs are at 0.87–0.90, and the
structures show what that means. All three of `BC-SB`'s closest pairs are **saturated-ring homologues
of one another** — identical across the entire benzyl-tetrazole-cyclopropyl-indole scaffold, differing
only in the size of one saturated N-heterocycle:

| pair | T | what differs |
|---|---|---|
| 1 | 0.90 | pyrrolidine → azetidine (one CH₂) |
| 2 | 0.89 | pyrrolidine → azetidine, both N-quinolinyl |
| 3 | 0.88 | terminal piperidine → azetidine |

Those homologues are separate purchasable building blocks feeding the *same* route, so the shared
intermediate is the whole rest of the molecule and the marginal price of the second one is a single
reaction. That is the count-once cost model working exactly as designed
and against the chemist's interest — a molecule sharing an intermediate you have already paid for is
nearly free, so under a tight cap the optimizer keeps buying neighbours. Both arms reach comparably
distant pairs at their extremes (0.04–0.08); the difference is entirely in the redundancy they
tolerate, which is what the mode count measures and the reaction count does not.

**Report `mcs%` as illustration, not as evidence.** It is a noisier discriminator than Tanimoto and
the two arms *overlap* on it — one BC-SB pair shares 70%, the same as our least-shared pair, because
`completeRingsOnly` MCS undercounts a common core when a shared ring is fused differently in the two
molecules. What the picture shows is not that we share *less* of the atom count but that the parts we
do not share are whole distinct fragments (a benzyl against an indol-7-ylamine) rather than a
terminating cap. The claim rests on the Tanimoto column and the mode count; `mcs%` only makes the
red/black rendering interpretable.

**8 — the candidate-count confound, quantified.** The obvious objection is that we win by looking at
more molecules. The pool costs say otherwise for the arm that matters:

| arm | reward-gen calls to build its pool | distinct candidates | above gate → selector | distinct @ R=100 |
|---|---|---|---|---|
| hub-batching (ours) | 74,606 *(cumulative, at the step it reaches 100 rxns)* | — | — | 82 |
| BC-Enum-SB | 131,474 enumerated (64 hubs) | 104,072 | 31,399 | 24 |
| BC-SB | 29,997 sampled | 26,069–27,133 | 21,000 (top-N cap) | 19–23 |

**BC-Enum-SB saw 1.8× more candidates than we did and still lost by 3.4×**, so for that arm — the
harder one, and the one built from our own enumeration — the confound is closed outright. BC-SB saw
0.4× as many, so for that arm it remains open and is the subject of the next experiment.

A within-method control bounds how much a candidate-count correction could possibly buy. Four
configurations of the same generator and seed, **all at the same reward gate of 7.0** (this matters —
an earlier draft of this table compared a gate-5.0 run against a gate-7.0 one, which is not a control
at all, and separately mislabelled the varied knob as `child_policy` when all four are `free_frag`;
what actually varies is `prebuild_k`, i.e. how aggressively children are pre-filtered before scoring):

| config | reward-gen calls @ 100 rxns | distinct @ 100 rxns |
|---|---|---|
| `reward` policy, `prebuild_k=0` (`scent_seh_1kx200`) | 2,549 | 34 |
| `free_frag`, `prebuild_k=20` (`scent_seh_thr7`) | 37,774 | 70 |
| `free_frag`, no prebuild-K (`scent_seh_freefrag`) | 74,606 | 82 |
| `free_frag`, no prebuild-K (`scent_seh_1kx200_freefrag`) | 92,304 | 80 |

**The return on oracle calls is severely diminishing and flattens out entirely above ~75k.** A 29×
increase in calls (2,549 → 74,606) buys 2.4× the molecules (34 → 82); doubling from 37,774 buys +17%;
and the largest budget on record (92,304) returns *fewer* molecules than a smaller one, which is run
noise swamping any remaining gain. At that elasticity, raising BC-SB from 29,997 calls to our 74,606
(2.5×) could not plausibly move it from ~23 to 82. The advantage is not a sampling-budget artifact.

---

## Update 2026-08-19 — replicates land, and they move the number

Three jobs' worth of overnight compute (74093-74100) replicate both competitor arms and close the
candidate-count confound. Two of the three results below **revise claims made above**, so read this
section as authoritative where it conflicts.

**9 — the headline drops from 3.4× to ~2.5×, because seed 42's competitor was a LOWER BOUND.** The
3.4× above rests on BC-Enum-SB scoring 24 distinct molecules at R=100 — a row flagged `TimeLimit`,
i.e. CBC never proved optimality. Both replicate seeds converged (`Optimal`) and landed *higher*:

| arm | seed | λ | selected | distinct | priced rxns | rxn/mode | solver |
|---|---|---|---|---|---|---|---|
| **hub-batching (ours)** | 42 | — | **82** | **82** | **100** | **1.22** | ~1 s |
| BC-Enum-SB | 43 | 1 | 99 | **33** | 34 | 1.03 | Optimal |
| BC-Enum-SB | 44 | 1 | 99 | **32** | 33 | 1.03 | Optimal |
| BC-Enum-SB | 42 | 1 | 96 | 24 | 40 | 1.67 | ⚠️ TimeLimit |
| BC-Enum-SB | 43 | 0 | 99 | 24 | 25 | 1.04 | Optimal |
| BC-Enum-SB | 44 | 0 | 99 | 26 | 27 | 1.04 | Optimal |
| BC-SB | 44 | 1 | 89 | 23 | 48 | 2.09 | Optimal |
| BC-SB | 42 | 1 | 92 | 20 | 35 | 1.75 | Optimal |
| BC-SB | 43 | 1 | 91 | 19 | 40 | 2.11 | Optimal |

This is exactly the failure mode the entry was written to catch, arriving from the other direction:
a capped row is a lower bound on the competitor, so it **flatters us**, and here it flattered us by
about a third. **Quote ~2.5×, not 3.4×.**

Two honesty notes on that ratio. It is **cross-seed** — our R=100 number at gate 7.0 exists only for
seed 42 (n=1) while the competitor now has three seeds, because the seed-43/44 reconcile runs
produced 300-mode reconciliations rather than curves. And the converged competitor is now **cheaper
per mode than we are** (1.03 vs our 1.22), reversing the "we win on both axes" claim in the Answer
above: at λ=1 with a converged solve, BC-Enum-SB buys its 33 molecules for 34 reactions. The
surviving claim is the one about *count*, which is the axis we lead with — but the cost axis is now
theirs, and saying so is cheaper than being caught.

**10 — the candidate-count confound is CLOSED, and my smoke-test reading of it was wrong.** The
prefix sweep (jobs 74093/74094) re-runs BC-SB's selection on the first k sampling events, k ∈ {2k,
5k, 10k, 20k, 30k}. Because `records.csv` is a post-hoc sample from a frozen checkpoint — mean reward
flat at 7.489 / 7.495 / 7.505 across the file — a prefix is exactly "what if we had sampled k times",
with no training-progress confound and no new oracle calls.

| k events | pool > gate | distinct @ R=100, seed 43 | seed 44 |
|---|---|---|---|
| 2,000 | ~1,650 | 43 | 40 |
| 5,000 | ~4,000 | 52 | 40 |
| 10,000 | ~7,800 | 50 | 43 |
| 20,000 | ~15,000 | 48 | 52 |
| 30,000 | ~21,700 | 58 | 53 |

**A 15× increase in sampling buys BC-SB about +35% distinct molecules** (43→58, 40→53), and the curve
is noisy and near-flat from k=5,000 onward — seed 43 actually dips at k=10k and 20k before recovering.
Extrapolating that elasticity, no achievable sampling budget takes BC-SB from ~55 to our 82. The
objection "you only win because you looked at more molecules" is answered for **both** arms now:
BC-Enum-SB saw 1.8× more candidates than us and lost outright, and BC-SB cannot close the gap by
sampling harder.

An earlier reading of the 2-minute smoke — that *more* sampling made BC-SB *worse* — was an artifact
of comparing an uncapped prefix run against the `--top-n 21000`-capped run in Result 6, i.e. two
different pool constructions. The full sweep, uncapped throughout, shows a weak positive trend. The
inverted-confound claim was wrong and is withdrawn.

**Figure — the sample-efficiency curve.**
`results/paper_sample_efficiency/sample_efficiency.{png,pdf,csv}`, from
`experiments/lsd_hubs/campaign/plot_sample_efficiency.py`. Distinct molecules at R=100 against the
oracle evaluations spent building the pool, log x, with BOTH competitor arms on one axis.

The BC-SB prefix sweep is drawn as joined lines because it IS a controlled sweep (one knob, k,
moves); our own points and the BC-Enum-SB points are drawn UNJOINED because they vary configuration
or seed rather than a single swept knob, and connecting them would assert a curve we did not
measure. That distinction is the figure's main honesty guard.

**BC-Enum-SB is the decisive point on this plot, and it makes the argument without extrapolation.**
Its pool is built by ENUMERATING off hubs rather than by sampling, so its oracle cost is the
enumeration itself — and that cost lands to the RIGHT of our own spend:

| seed | children enumerated (= oracle calls) | distinct @ R=100 | solver |
|---|---|---|---|
| 42 | 131,474 (1.8× ours) | 24 | ⚠️ TimeLimit — lower bound |
| 43 | 170,724 (2.3× ours) | 33 | Optimal |
| 44 | 175,345 (2.4× ours) | 32 | Optimal |
| **ours** | **74,606** | **82** | ~1 s |

The competitor spends **1.8–2.4× our oracle budget and delivers under half the molecules**. Where the
BC-SB sweep answers the candidate-count objection by extrapolating a flat trend, this answers it with
a measured point that is already past us on the x-axis. The seed-42 row is capped and therefore a
lower bound, i.e. it flatters the competitor's best case, not ours.

**11 — where the replicates leave each arm.** λ=1 is worth 1.3–1.4× on BC-Enum-SB even with a
converged solve (24→33 on seed 43, 26→32 on seed 44), so the diversity mechanism is load-bearing and
the `059` comparison against a diversity-blind selector really was unfair. It still saturates well
short of us.

---

## ⛔ WITHDRAWN 2026-08-20 (later the same day) — Results 13-15 were measured on MIXED enumerations

**Do not quote the HB-Enum-SB numbers, the 1.67× or the 2.27× below.** Re-running against frozen
inputs; this banner is removed when they land.

**What happened.** All three `scent_seh` enumerations were re-run on 2026-08-19 between 21:13 and
22:06 to fix a **per-hub enumeration cap that had been truncating children**. The fix is correct and
those are the good files — but the rewrite landed *after* the HB-Enum-SB jobs had already read the
old ones, while our own seed-curve jobs ran the next morning against the new ones. So the arms in
Result 15 were priced on **different candidate sets**, which is precisely the confound that comparison
existed to remove.

The size of the drift is not marginal — the same three cells at gate 7.0:

| seed | pool the HB job used | pool in the current file |
|---|---|---|
| 42 | 21,073 | 119,866 |
| 43 | 21,581 | 125,414 |
| 44 | 27,250 | 152,328 |

**Nothing here was detectable from the outputs.** Both runs completed, reported `Optimal`, and
produced plausible libraries. The only tell was an input mtime, which nothing checked. The lesson is
narrow and worth stating: *a comparison that claims "the same candidates" must read from a frozen
copy*, because the shared scratch tree is written by other agents mid-experiment. Re-runs now read
`/scratch/.../lsdflow_sparrow/_enum_snapshot_20260820/`, with source md5s recorded.

**What is NOT affected, checked rather than assumed.** The BC-Enum-SB sources
(`bc_enum_seh_seed42`, `t45_seh_seed43/44`) were untouched (mtimes 08-05/08-06) and show no cap
signature, so Results 1-12 stand — including the **2.51×** within-seed headline. Entry `065`'s matrix
curves were all regenerated 08-20 11:00-11:02, i.e. *after* the enum fix, so they are on good inputs
(their numbers did shift slightly; see that entry).

---

## Update 2026-08-23 — HB-Enum-SB does not converge on the corrected pools; the arm is reported as a BOUND

The 2026-08-19 enumeration cap-fix multiplied the qualifying pools ~6× (21k → 123k-150k above gate),
and SPARROW's MILP does not solve at that size. Every attempt at R=100 on sEH:

| run | pool | selected | distinct | solve | status |
|---|---|---|---|---|---|
| seed 42 (pre-fix enum) | 21,073 | 98 | 42 | 405 s | Optimal |
| seed 43 (pre-fix enum) | 21,581 | 94 | 37 | 3,308 s | Optimal |
| seed 44 (pre-fix enum) | 27,250 | 97 | 23 | 1,146 s | Optimal |
| seed 43, `--top-n 50000` | 50,000 | 89 | 12 | 7,589 s | ⚠️ TimeLimit |
| seed 44, `--top-n 50000` | 50,000 | 87 | 39 | 7,322 s | ⚠️ TimeLimit |
| seed 43, uncapped, 12 h | 123,298 | 88 | 27 | 43,080 s | ⚠️ TimeLimit |
| seed 44, uncapped, 12 h | 149,604 | 93 | 25 | 43,012 s | ⚠️ TimeLimit |

**Raising the cap 2 h → 12 h and relaxing `gapRel` to 1e-3 did not achieve convergence**, and the
capped answers do not trend toward the converged ones — seed 43 reads 12, 27 and 37 distinct on three
pool sizes with no ordering. So the capped numbers are not "nearly right"; they are lower bounds whose
distance from the optimum is unknown and evidently large.

**Consequence for the paper: on the corrected enumerations, HB-Enum-SB is reportable only as a bound
on the competitor, and a bound in the direction that FLATTERS US.** It cannot carry a ratio. The
converged rows all used the pre-fix (truncated) enumerations, so they are not usable either — the same
mixing that Result 15's withdrawal was about.

This is not a defect in the arm; it is the measurement CLAUDE.md already anticipates
("solver-truncated (SB arm only) — this is COMPUTE-limited and is a completely different claim"). The
honest framing is that the optimizer cannot solve the selection problem at the scale our enumeration
produces, which is itself a result: our greedy returns its answer in ~1 s on the same pool.

**Results 13-15 stay withdrawn.** The 2.51× BC-Enum-SB headline is unaffected — different arm,
different pools, all converged.

---

## Update 2026-08-20 — our side reaches n=3, and a new arm returns a NULL

Jobs 74441/74442 (our seeds 43/44) and 74444-74446 (the new HB-Enum-SB arm) completed. All five
converged; no capped rows on our side and none on HB-Enum-SB. Costs below are `cost_kept_rxns`, not
`used_rxns`, per the pool-exhaustion rule in `CLAUDE.md`.

**12 — the cross-seed asymmetry is gone, and the ratio survives it.** Our arm had been n=1 (seed 42's
82) against a competitor at n=3 — the weakest link in the headline. The two missing curves exist now:

| seed | ours | BC-Enum-SB | ours / BC |
|---|---|---|---|
| 42 | 82 | 24 ⚠️ capped | 3.42 |
| 43 | 81 | 33 | **2.45** |
| 44 | 82 | 32 | **2.56** |

**Ours: mean 81.7, sd 0.6, CV 0.7%** — strikingly stable, and far tighter than the 7.0% median seed
CV entry `065` measured on the reaction axis for the matrix as a whole. On the two seeds where the
competitor *converged*, the within-seed ratio is **2.51×**. That is the same ~2.5× the 2026-08-19
update arrived at by cross-seed inference, now obtained properly, and it should be the quoted number.
Seed 42's 3.42× remains excluded: its competitor row is a lower bound.

**13 — HB-Enum-SB: SPARROW does NOT do better on flow-derived candidates.** The arm exists to answer
one question — *does the optimizer do better on candidates enumerated from high-flow hubs than on
candidates from reward-picked hubs?* Same optimizer, same budget, same gate, same pool cap; only the
candidates' origin differs. All three seeds solved to `Optimal`.

| seed | HB-Enum-SB (flow hubs) | BC-Enum-SB (reward hubs) |
|---|---|---|
| 42 | 42 | 24 ⚠️ capped |
| 43 | 37 | 33 |
| 44 | **23** | **32** |

On the two seeds where both converged: **HB 30.0 vs BC 32.5.** No advantage — marginally worse, and
**the direction flips between seeds** (43: HB +12%; 44: HB −28%). The expectation going in was a
modest gain, on the reasoning that reward-picked hubs are themselves already fairly high-flow
(entry `053`: flow picks the neighbourhood, not the rank). The measurement is weaker than that
expectation: at this sample size there is no detectable difference at all.

**This is "no evidence of a difference", NOT "no difference", and two things forbid the stronger
claim.** First, **HB-Enum-SB is noisy where we are not** — CV **29.0%** (23–42) against our 0.7%. That
is itself a finding: SPARROW's selection is far more sensitive to which candidate set it is handed
than our greedy is, so a ±10% effect is simply not resolvable at n=3. Second, a **live confound**:
these enumerations used `n_hubs=200 / top_k=1000` while BC-Enum-SB's used `64 / 100`, so the arms
differ in the WIDTH of the net as well as in how hubs were ranked, and a 200-vs-64 hub difference
could easily swamp a small flow effect. Isolating it needs a 64-hub *flow* enumeration, which does
not exist.

**14 — the cleanest selection-only contrast** ⚠️ **(n=1 — SUPERSEDED by Result 15, which turns this
into a same-enum band at n=3; read that instead. Seed 42 is the LOWEST of the three points, so the
deflationary interpretation below is withdrawn there.)** Every ratio above
compares arms built on *different* candidate sets. For seed 42 one comparison avoids that: our arm and
HB-Enum-SB on the **same enumeration at the same gate** (`matrix16/scent_seh`, gate 7.0):

| arm, same enum + same gate | distinct @ R=100 |
|---|---|
| hub-batching (`scent_seh_thr7`) | **70** |
| HB-Enum-SB | **42** |
| ratio | **1.67×** |

That is the honest "same molecules, different chooser" number, and it sits well below the 2.51× we
quote against BC-Enum-SB. **The implication is uncomfortable and belongs in the paper rather than in a
footnote: a real share of the 2.51× comes from candidate ORIGIN, not from selection.** Selection is
worth ~1.67× on identical candidates; the rest is what enumerating off our hubs buys.

Two limits on that number, both real. It is **one seed** — the matrix cells for seeds 43/44 sit at the
5.0 headline bar, so no same-enum gate-7.0 comparison exists there. And our own two gate-7.0 seed-42
numbers (82 from `scent_seh_freefrag`, 70 from `scent_seh_thr7`) differ by a `prebuild_k` of 0 vs 20
on the same gate, so "our arm" is not a single fixed configuration either. Neither undermines the
direction; both cap how hard the 1.67× can be pushed.

**15 — the band lands, and it REVISES Result 14 upward.** Jobs 74482-74485 ran our arm on the
*matrix16* enumerations at gate 7.0 for seeds 43/44, at both prebuild-K settings, so the selection-only
contrast is now same-enum, same-gate and within-seed at n=3:

| seed | ours (K=0) | ours (K=20) | HB-Enum-SB | ratio K=0 | ratio K=20 |
|---|---|---|---|---|---|
| 42 | 82 † | 70 | 42 | 1.95 † | **1.67** |
| 43 | 79 | 75 | 37 | 2.14 | **2.03** |
| 44 | 83 | 72 | 23 | 3.61 | **3.13** |

† seed 42's K=0 number comes from a *different* enumeration (`campaign_enum_seh_70363`), so only its
K=20 row is same-enum. Seeds 43/44 are same-enum at both settings.

**Selection alone is worth 2.27× (K=20, n=3, sd 0.76), not 1.67×.** The 1.67× reported in Result 14
was seed 42 — **the lowest of the three**, and reading a single seed as the value was exactly the
error this entry has now made twice. On the same-enum K=0 rows (seeds 43/44 only) it is 2.87×.

**So the deflationary reading in Result 14 is withdrawn.** With selection worth ~2.3× and the headline
against BC-Enum-SB at 2.51×, candidate origin accounts for far less of the advantage than that
paragraph claimed — most of it *is* selection. Result 14's numbers stand as reported; its
interpretation does not.

**What actually drives the spread is the competitor, not us.** Our arm is stable across seeds
(79/83 at K=0, 70/72/75 at K=20); HB-Enum-SB is 42/37/23. Seed 44's 23 is what produces the 3.13×,
and it is the same CV=29% instability flagged in Result 13. The ratio's spread is a property of how
sensitive SPARROW is to its candidate set, and should be described that way rather than as uncertainty
in our own arm.

A secondary observation worth keeping: on the same enumeration, `prebuild_k=20` costs us modes at
R=100 (79→75 on seed 43, 83→72 on seed 44). Pre-selecting K fragments buys cheaper oracle calls
(Logs/037) but is not free on the reaction axis.
