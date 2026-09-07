# All entrants / sEH+DRD2+ClpP — two measurement defects in the Stage-4 frontier, and what they were hiding

**Date:** 2026-09-06, ~11pm

## Question

When we report "100 reactions buys you N distinct high-reward molecules", is N a property of
the generator being measured, or of our own measuring tools?

## Context & Summary

The benchmark's headline number is a fixed reaction budget: spend 100 reactions, count the distinct
high-scoring molecules you get. Two selectors produce that number for each cell, and both of them are
the competitor at its strongest — a diversity-aware greedy selector that prices a fixed shopping
list, and SPARROW-Batching, the published tool's own optimiser, which chooses the shopping list
itself under a shared-intermediate objective.

Entry [078] established that routability, not reward or diversity, is what bounds this benchmark.
This entry looks one stage later, at the pricing itself, and finds that two of the numbers we were
about to publish were set by our tooling rather than by the generators.

The first is a solver problem. SPARROW-Batching's optimiser is given a wall-clock limit, and a run
that hits it returns whatever it happens to be holding, flagged `TimeLimit`. Our own convention
(CLAUDE.md) says such a row is a *lower bound on the competitor* and cannot carry a ratio. Fourteen
of 87 rows at the 100-reaction readout were in that state — and they are not spread evenly.

The second is a resolution problem. The greedy arm is priced on the *mode* axis: we ask for M
molecules and learn what they cost. The headline is on the *reaction* axis: we fix 100 reactions and
ask how many molecules fit. Converting one into the other means taking the largest rung that fits
inside the budget — so the answer is quantised to however finely the rungs were spaced, and whatever
budget is left unspent at that rung is a silent understatement.

We measured how large both effects are, then fixed both without re-running any chemistry: the routes
were already cached, so only the selection maths was redone.

## Answer

Both defects were real, and neither changed a scientific conclusion in the direction we feared —
but one of them had put a headline comparison on ground we could not have defended.

On sEH, every truncated row belonged to REINVENT and every S3-GFN row was certified, so the
comparison read as an S3-GFN win on one pool while the *fully certified* other pool showed the
opposite. Re-solving all three of those cells returned **the identical answers, now certified** —
80, 70 and 92 molecules, unchanged, with `TimeLimit` after ~1800 seconds becoming `Optimal` after
under ten. The cap had been hiding correct numbers, not wrong ones. The fix was not more compute: it
was asking the solver for a 1%-optimal answer instead of a seven-decimal-place one, on a quantity we
report in whole molecules.

That leaves the disagreement between the two pools standing as a real result rather than a solver
artefact, and it has a mundane cause worth stating plainly. The two pools are built differently — one
takes the 500 highest-scoring molecules, the other takes 500 that are deliberately unlike each other
— and generators differ enormously in how many of their own top 500 are already unlike each other:
about 40-58% for S3-GFN against 66-87% for REINVENT, and as low as 2-4% for Saturn and TANGO. The
pool that holds more distinct molecules then yields proportionally more distinct products, and
checking that arithmetic against the measured gains accounts for most of the effect. So the pool
comparison is not revealing a subtle interaction; it is revealing how much of each generator's
best-scoring output is the same molecule over again. That is still the thing worth reporting, because
it decides which generator looks better, and it is invisible if only one pool is shown.

The ladder defect turned out to be worse than pure understatement. Across the campaign the median
affected cell was leaving a third of its budget unspent, ten cells had no readable number at all, and
re-pricing 58 of them raised the reported total on the 18 that moved by 38%. But the gains were
**uneven** — some cells rose a fifth, others three-fifths — and on ClpP that unevenness had erased a
result outright: S3-GFN and REINVENT both reported 25 molecules on the old rungs, a perfect tie,
where a finer ladder shows 40 against 30 on all three seeds with no overlap between them. The worst
case was SynFormer, understated by nearly a factor of two on one cell because its own ladder skipped
from 25 straight to 50. None of these are generator effects; they are the spacing of a list we chose.

One more thing surfaced that the first version of this entry got wrong. We had described the
naive-versus-pruned split as *the* disagreement in the benchmark. It is one of two. Which generator
leads also depends on **which selector** does the choosing: the greedy selector chases diversity,
while SPARROW-Batching optimises for shared synthetic steps, and how much a generator benefits from
that is a stable property of its chemistry — REINVENT's molecules share intermediates about 1.6 times
more usefully than S3-GFN's, on both pools and all three targets. That is enough to reverse the
naive-pool ranking between the two selectors. So the honest presentation is a two-by-two, with a
measured mechanism on each axis, rather than a single number with a robustness check beside it.

## Relevance to our Publication

This is the ICLR submission's headline readout, and both defects are exactly what a competent
reviewer checks. Quoting `TimeLimit` rows as if they were optima would mean reporting a competitor
number we knew was understated — and the asymmetry was the worst possible shape, with the truncation
falling entirely on the competitor in the comparison we most want to make. Fixing it *before*
submission converts a soft claim into a certified one; the alternative was a reviewer with the same
solver and an afternoon discovering it for us. The ladder fix matters for a different reason: it
raises numbers for every entrant including our own, so it is a fairness correction rather than a
favourable one, and it makes the reaction-axis readout mean what the paper says it means.

## Next Experiments

**Refining for publication**
- Certify the eleven capped rows outside sEH the same way. At 5-10 s each this is nearly free, and
  every certified row is one more cell that can carry a ratio. Note in any table that certification
  is now at a 1% gap rather than SPARROW's 1e-7.
- Present the benchmark as a **two-by-two** — {naive, pruned} × {greedy, SPARROW-Batching} — with the
  mechanism named on each axis (reward-ranking redundancy; route sharing). Three of the four
  quadrants favour S3-GFN on current data and one favours REINVENT, and showing a single quadrant
  would let us choose our own winner.
- Re-check that no table mixes ladder resolutions. After the queued pass every cell should be on
  `2,5,…,90,100,125,150`; a band mixing a 25-spaced cell with a 5-spaced one reads as a seed effect,
  which has cost this project a result before.
- The ladder is still 25-spaced ABOVE 100 (100→125→150). No current cell lands there, but a cell
  delivering ~110 modes would report 100, so the top of the ladder needs the same treatment before
  any high-delivery cell is quoted.

**Next steps in project**
- Complete the Stage-3 route discovery still outstanding, then re-price the whole matrix once from a
  single tooling version so every cell in the paper is priced identically.
- Check whether the per-generator redundancy ordering (Saturn/TANGO 2-4%, S3-GFN 39-58%, REINVENT
  59-87%, FragGFN 87-100%) predicts anything the benchmark does not already measure, or whether it is
  simply a restatement of the pool sizes.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork` (worktree `.claude/worktrees/fraggfn-stage2`).

**Scripts**
- `./experiments/lsd_hubs/campaign/submit_reprice_cached.sh` — re-prices a cell from cached routes,
  changing exactly one solver knob. `ARM=sb` re-solves a time-capped budget row; `ARM=greedy`
  re-prices on a dense mode ladder. Written for this experiment.
- `./experiments/lsd_hubs/campaign/_resolve_gate.py` — prints `<target> <gate> higher|lower` from
  `matrix16/targets.py`, so a launcher never defaults the 5%-FPR bar.
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — the frontier itself; called
  directly here rather than through the chain.
- `./experiments/lsd_hubs/campaign/submit_competitor_routes.sh` — the normal chain entry point,
  deliberately BYPASSED (see Method step 2).

**Results**
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/<tag>_select_N500/select_frontier.csv` — the
  SPARROW-Batching arm, carrying `milp_status`, `time_capped`, `solve_s`. Untouched by this work.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/<tag>_select_N500_longsolve/` — re-solved
  R=100 rows, written to a separate directory so the original ten-budget CSV survives.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/<tag>_greedy_N500/greedy_frontier.csv` — the
  greedy arm, re-priced IN PLACE (the dense ladder is a strict superset of the coarse one).
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/greedy_csv_backup_20260906.tar.gz` — 330 greedy CSVs
  and summaries backed up before the in-place re-price.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/ls_rvseh{42,43,44}-757{88,89,90}.out` — the three re-solves. Each
  prints the capped baseline it is testing beside the new row, so the comparison is in the log.
- `/scratch/markymoo/rgfn_runs/smk_reprice-75791.out` — the dense-ladder smoke.
- `/scratch/markymoo/rgfn_runs/dense_{n,p}-757{92,93}.out` — the 58-cell dense pass.
- `/scratch/markymoo/rgfn_runs/smk_syn-75804.out` — the SynFormer smoke that FAILED in 8 s and named
  both launcher bugs. Kept deliberately: it is the evidence that refusing on a cache miss is what
  made two silent-failure bugs loud.
- `/scratch/markymoo/rgfn_runs/fine_syn_n-75806.out` — SynFormer's fine-ladder pass.

## Relevant Versions

```
e8d35a7 Stage 2 named the REINVENT env `reinvent`; it is `reinvent4`, and nothing had ever exercised it
61c5e55 `eligible` was strict where the gate is inclusive, so a cell logged more modes than eligible molecules
6a043c2 REFACTOR_LOG: what changed in the Stage-2/3 work, and the seven things left open
```

`submit_reprice_cached.sh` and `_resolve_gate.py` were added in `187fd5a` ("Every truncated solve
at the headline readout was the competitor's, and the ladder was understating everyone"), pushed to
`origin/worktree-fraggfn-stage2`.

A version check was run specifically for this experiment: `sparrow_select_frontier.py` last changed
in `c802026` (2026-08-27 11:52, a greedy-arm fix), and **all 96 campaign greedy results postdate it**
(earliest 2026-08-28). The re-price therefore changes the ladder and nothing else. The pre-fix
results on disk are older non-campaign directories that carry no `_stage2` tag.

## Relevant Resources

**Packages**
- CBC via SPARROW's MILP layer — the solver whose `TimeLimit` status this entry is about.
  SPARROW hardcodes a relative gap of 1e-7; `sparrow_select_frontier.py --gap-rel` overrides it.

## Method

1. **Audit.** Read `select_frontier.csv` for every campaign cell, taking the `budget_rxns == 100`
   row, and classified it by `time_capped` / `milp_status`.
2. **Bypassed the chain deliberately.** Re-pricing through `submit_competitor_routes_chain.sh` would
   have (a) re-run ~3.7 h/cell of MultiAiZ discovery for a solve costing seconds, (b) overwritten the
   ten-budget CSV with a one-row file, and (c) **rebuilt the pool** — and `build_s3gfn_pools.py` has
   since gained the NaN-gate fix, so a rebuilt pool is no longer the pool the cached routes were
   planned for. Reading the cached `pool_scores.csv` and `multiaiz_routes.json` directly is what
   makes this a re-price rather than a re-run. The script refuses to run if that cache is absent,
   rather than silently rediscovering.
3. **Re-solved the capped rows** with `--budgets 100 --gap-rel 0.01 --max-seconds 21600`, into a
   separate `_longsolve` directory. Jobs 75788/75789/75790.
4. **Quantified the ladder defect** by taking, for each greedy cell, the largest rung with
   `used_rxns <= 100` and recording the unspent budget at that rung.
5. **Backed up 330 greedy CSVs**, then smoke-tested the dense ladder on one cell on the `debug`
   partition (job 75791, 73 s) before batching. Jobs 75792 (25 naive cells) / 75793 (33 pruned).
6. **Ruled out a code-version confound before touching anything.** `git log` on
   `sparrow_select_frontier.py` gives `c802026` (2026-08-27 11:52) as the last greedy-arm change;
   every campaign greedy result postdates it, so the re-price moves the ladder alone. The pre-fix
   results on disk are older non-campaign directories carrying no `_stage2` tag.
7. **Audited pool drift** against the backup tarball's mtimes: a cell is ladder-only iff its cached
   `pool_scores.csv` is OLDER than its original greedy CSV. 91 of 96 qualify; the 5 that do not are
   SynFormer cells whose pools were rebuilt the same night by the SB backfill (75771/75772).
8. **Isolated the NaN-gate pool effect from the ladder effect.** 75771/75772 re-ran SynFormer's
   greedy on the SAME coarse ladder against the rebuilt pools, which is a pool-only comparison; all
   six pruned cells came back identical, so the pool change contributes nothing to the deltas above.
9. **Re-measured residual quantisation** where each cell actually lands, then re-priced the two
   residual classes on `2,5,…,90,100,125,150`: SynFormer via the new `ROUTE_SOURCE=external` path
   (jobs 75806/75807) and the ladder-gap cells via the chain path (75801/75803).

## Results

**Capped rows at the 100-reaction readout: 14 of 87.** Distribution on sEH, which is where the
headline S3-GFN-vs-REINVENT comparison lives (modes kept at R=100):

| pool | S3-GFN s42/43/44 | status | REINVENT s42/43/44 | status |
|---|---|---|---|---|
| naive | 56 / 38 / 49 | Optimal (2–3 s) | 61 / 47 / 83 | Optimal (2–7 s) |
| pruned | 98 / 100 / 100 | Optimal (~1.6 s) | 80 / 70 / 92 | **TimeLimit (1802 s)** |

The two pools point in opposite directions, and the pool on which S3-GFN leads is precisely the one
where the competitor's solver was truncated. S3-GFN's pruned rows sit at the arithmetic ceiling —
100 reactions for 100 modes is 1.0 reaction/mode and cannot be beaten — so the whole question was
whether REINVENT reaches that ceiling too when allowed to finish.

**Re-solve, REINVENT sEH pruned, all three seeds (jobs 75788/75789/75790):**

| seed | modes before | status before | solve before | modes after | status after | solve after |
|---|---|---|---|---|---|---|
| 42 | 80 | TimeLimit | 1802.36 s | **80** | Optimal | **9.74 s** |
| 43 | 70 | TimeLimit | 1802.12 s | **70** | Optimal | **8.29 s** |
| 44 | 92 | TimeLimit | 1801.71 s | **92** | Optimal | **5.65 s** |

**Every value is unchanged.** All three answers were already correct; only their certification was
missing. Total solve time after the gap change was 23.7 s, replacing 5,406 s of timed-out solving.
Note the new status is optimal *to a 1% relative gap*, against SPARROW's default 1e-7 for the
previously-certified rows — ±1 mode at these counts, which does not disturb any comparison here, but
the asymmetry should be stated wherever these rows are quoted.

**The sEH comparison, now fully certified on both sides:**

| pool | S3-GFN 42/43/44 | REINVENT 42/43/44 | leader | bands |
|---|---|---|---|---|
| naive | 56 / 38 / 49 | 61 / 47 / 83 | REINVENT, all three seeds | overlapping (56 > 47) |
| pruned | 98 / 100 / 100 | 80 / 70 / 92 | S3-GFN, all three seeds | **disjoint** (98 > 92) |

The direction flip is REAL, not a solver artefact — it survived certification of all three
truncated rows with unchanged values.

**Why the pools disagree: redundancy in the reward-ranked prefix.** The naive pool is top-500 by
reward; the pruned pool is top-500 mutually distinct. The fraction of the naive pool that also
appears in the pruned pool turns out to equal `modes_in_largest_prefix / 500` **exactly** in every
cell checked — i.e. it is simply how many of the reward-ranked top-500 are mutually dissimilar, a
quantity `mode_saturation.py` already records. It is not a new measurement, and it is not a Tanimoto
redundancy score:

| cell | naive ∩ pruned | SB modes naive → pruned | gain |
|---|---|---|---|
| S3-GFN 42 | 58% | 56 → 98 | 1.75× |
| S3-GFN 43 | 40% | 38 → 100 | 2.63× |
| S3-GFN 44 | 39% | 49 → 100 | 2.04× |
| REINVENT 42 | 77% | 61 → 80 | 1.31× |
| REINVENT 43 | 66% | 47 → 70 | 1.49× |
| REINVENT 44 | 87% | 83 → 92 | 1.11× |

The gain from pruning tracks the overlap almost monotonically across all six cells: the lowest
overlap (S3-GFN 43, 40%) gives the largest gain (2.63×) and the highest (REINVENT 44, 87%) the
smallest (1.11×).

**But the mechanism is largely ARITHMETIC, and must be reported as such.** The naive pool's mode
ceiling is `modes_in_largest_prefix`; the pruned pool's is `min(500, modes_available_whole_set)`.
Comparing SPARROW's pruning gain against that ratio over 15 uncapped cells with ≥20 modes gives a
**median deviation of 5%**, nine of them within ±0.05:

| cell | modes in top-500 | SB gain | mode-count ratio | difference |
|---|---|---|---|---|
| REINVENT sEH 42 | 385 | 1.31× | 1.30× | +0.01 |
| REINVENT sEH 43 | 331 | 1.49× | 1.51× | −0.02 |
| S3-GFN sEH 42 | 292 | 1.75× | 1.71× | +0.04 |
| S3-GFN sEH 43 | 199 | 2.63× | 2.51× | +0.12 |
| S3-GFN sEH 44 | 195 | 2.04× | 2.56× | **−0.52** |
| S3-GFN ClpP 42/43/44 | 111 / 89 / 116 | 1.42 / 1.33 / 1.48× | 1.00× each | **+0.33…+0.48** |

So the pruned pool wins mostly because it *contains more distinct molecules*, and SPARROW returns
proportionally more modes. That is close to a definition, not a discovery, and the entry should not
be read as evidence that pruning interacts cleverly with route sharing. The two exceptions are the
informative ones: S3-GFN sEH 44 under-delivers against its own headroom, and the three S3-GFN ClpP
cells gain 33-48% where the ratio predicts nothing — those pools hold only 89-116 distinct molecules
in total, so there the gain comes from *composition* rather than count.

**The non-tautological part is the per-generator redundancy itself**, which varies enormously and
consistently — the share of a reward-ranked top-500 that is mutually dissimilar:

| generator | modes inside its own top-500 |
|---|---|
| Saturn / TANGO (sEH) | 10-19 of 500 (2-4%) |
| S3-GFN | 195-292 (39-58%) |
| REINVENT | 296-436 (59-87%) |
| FragGFN | 434-499 (87-100%) |

That ordering is a real property of the generators and it is what decides which pool favours whom.
FragGFN's near-total overlap is **not** a headroom artefact: it has 731-2,372 modes available in its
whole above-gate set, so a different 500 was freely available and the diversity filter simply did not
want one. 35 of 42 cells had >600 modes available and are informative on this point; the 7 that did
not (REINVENT ClpP ×3, S3-GFN ClpP ×3, S3-GFN sEH 43) are near-exhaustive and must not be read as
diversity claims.

This also refines a note carried in `sparrow_select_frontier.py --gap-rel`'s own help text, which
recorded that gap relaxation "did NOT achieve convergence alone". On this instance it did, and the
6 h wall-clock was never approached.

**Ladder quantisation, 96 cells with a greedy arm:**

| | cells |
|---|---|
| dense ladder already | 36 |
| coarse ladder (25/50/75/…) | **59** |
| coarse, of which no rung fits R=100 at all | **10** |

Over the 49 coarse cells that do produce a number, the budget left unspent at the quoted rung has
**median 33 of 100 reactions**, maximum 52, with 26 cells above 30. The ten with no readable number
are Saturn and TANGO cells whose first rung of 25 modes already costs more than 100 reactions.

**Dense-ladder smoke, REINVENT sEH seed 42, naive (job 75791, 73 s):**

| ladder | best rung fitting R=100 | reactions used | unspent |
|---|---|---|---|
| coarse | 25 modes | 77 | 23 |
| dense | **30 modes** | 94 | 6 |

Every shared rung reproduced exactly (25 modes → 77 reactions on both), and all 14 rungs returned
`Optimal`, so the dense re-price is a strict refinement rather than a different measurement.

**Full dense re-price, 58 cells (jobs 75792 naive / 75793 pruned, 18 min and 30 min):**

| outcome | cells |
|---|---|
| gained modes | 18 |
| recovered — previously NO readable R=100 number | 1 (`saturn_clpp_seed42`, none → 20 @ 98) |
| unchanged | 6 |
| skipped / failed | **0** |

Summed over the 18 gainers the reported modes go **450 → 620 (1.38×)**. The gains are **uneven**,
which is the part that matters: some cells move 25→30 (+20%) and others 25→40 (+60%), so the coarse
ladder was distorting cross-generator comparison and not merely depressing every cell alike.

**The coarse ladder had ERASED a result on ClpP.** Naive pool, greedy arm, modes at R=100:

| generator | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| S3-GFN | 40 @ 92 | 40 @ 96 | 40 @ 84 |
| REINVENT | 30 @ 97 | 30 @ 73 | 30 @ 86 |

Disjoint bands across three seeds, S3-GFN 1.33× ahead. **On the coarse ladder both read 25/25/25 — a
perfect tie.** Nothing about the generators changed; the first rung was simply too far apart to see
the difference.

**A SECOND axis of disagreement, which the first version of this entry missed.** The naive-pool
result depends on the SELECTOR too, not only on the pool:

| | greedy | SPARROW-Batching |
|---|---|---|
| **naive** | S3-GFN 40/50/75 vs REINVENT 30/30/30 | REINVENT 61/47/83 vs S3-GFN 56/38/49 |
| **pruned** | (fine-ladder pass still queued) | S3-GFN 98/100/100 vs REINVENT 80/70/92 |

Greedy chases diversity; SPARROW-Batching optimises for SHARED SYNTHETIC INTERMEDIATES. The uplift
from one to the other therefore measures how much route sharing a pool actually offers, and it is a
stable per-generator property (capped and coarse rows excluded):

| generator | naive uplift | pruned uplift |
|---|---|---|
| REINVENT | 1.86× (n=7) | **2.57×** (n=6) |
| S3-GFN | 1.01× (n=6) | 1.65× (n=7) |
| FragGFN | 1.20× (n=9) | 1.23× (n=8) |
| Saturn | 0.17× (n=7) | 0.36× (n=6) |
| TANGO | 0.27× (n=7) | 0.22× (n=5) |

REINVENT's pool batches ~1.6× better than S3-GFN's under SPARROW's objective, consistently on both
pools and all three targets — which is what lets it overtake on naive/SB while trailing on
naive/greedy. Saturn and TANGO falling BELOW 1.0× is not a defect: on a collapsed pool SPARROW
pursues its own objective and does not chase modes, which is exactly why both arms are reported.

**SynFormer was the worst-quantised entrant, and its own ladder was the cause.**
`submit_native_routes.sh` defaults to `5,10,15,20,25,50,75,100,125,150`, which jumps 25→50, so a cell
needing ~45 modes had to stop at 25. On a finer ladder (job 75806, 6 cells in 2 min 42 s):

| cell | coarse | fine |
|---|---|---|
| clpp 42 | 25 @ 52 | **45** @ 93 |
| clpp 43 | 25 @ 57 | **45** @ 99 |
| drd2 42 | 75 @ 76 | **90** @ 93 |
| drd2 43 | 75 @ 91 | **80** @ 98 |
| clpp 44 / drd2 44 | 50 @ 96 / 75 @ 97 | unchanged |

**The NaN-gate pool rebuild moved NOTHING, measured rather than assumed.** Jobs 75771/75772 rebuilt
SynFormer's pools with the fixed `build_s3gfn_pools.py` and re-ran greedy on the SAME coarse ladder,
which is a clean pool-only comparison. All six pruned cells are **identical** to their 2026-09-05
values (25@52, 25@57, 50@96, 75@76, 75@91, 75@97). So the ladder gets full credit for the changes
above, and the ladder/pool confound that applied to 5 of 96 cells is answered, not merely flagged.
A pool-drift audit against the pre-re-price backup put the other 91 cells firmly in the ladder-only
category (pool `pool_scores.csv` older than the original greedy CSV).

**Residual quantisation after the dense pass**, from re-reading where each cell actually lands:

| | cells |
|---|---|
| resolved acceptably (≤15 rxn unspent, or next rung ≤5 away) | 77 |
| still coarse WHERE THEY LAND | 18 |
| no readable R=100 number | 0 real cells (1 smoke-test artefact) |

The 18 split into two causes: **6 SynFormer cells never re-priced at all** (native routes, a
different launcher, so the chain-derived lists never included them — 24-48 reactions unspent, the
worst residuals), and **12 cells limited by the dense ladder's own spacing**, which jumps
30→40→50→60→75→100 and so leaves 18-28 reactions unspent for a cell landing at 30. Both are being
re-priced on `2,5,…,90,100,125,150` (jobs 75801/75803/75806/75807).

**Two launcher bugs, both caught by the REFUSAL guard rather than by a wrong number** (job 75804,
failed in 8 s):

1. `TAG_SUFFIX=${TAG_SUFFIX:-_stage2}` rewrote an explicitly-empty suffix back to `_stage2`. Route-
   carrying generators never had a Stage 2, so the cell looked for `synformer_clpp_seed42_stage2_N500`,
   which has never existed. `${TAG_SUFFIX-_stage2}` — no colon — distinguishes empty from unset.
2. `FR_ROOT` was missing the `experiments/` path segment.

Both are silent-failure shaped: a launcher that rediscovered on a cache miss would have spent hours
and produced a plausible frontier for the wrong pool. Refusing on a missing cache is what turned
them into an 8-second error naming both bad paths.

**A HAZARD THIS ENTRY CREATED, and how it will be closed.** Writing re-solved rows to a separate
`<tag>_select_N500_longsolve/` directory protected the original ten-budget CSV, but it leaves TWO
sources of truth for the primary readout: `<tag>_select_N500/select_frontier.csv` still carries the
stale `TimeLimit` row at R=100, and only the `_longsolve` sibling has the certified one. This is not
hypothetical — the audit script written for this very entry globbed `*_select_N*`, matched both
directories, and re-reported the three already-certified sEH rows as still capped. Any reader that
does not know to prefer `_longsolve` will quote the lower bound.

The fix is to MERGE the certified R=100 rows back into the main CSV once the remaining certification
jobs land (75810/75811), so there is one file per cell, with the originals backed up first. Two
details matter when doing it:
  * `solve_s` dropping from ~1800 to <10 alongside `milp_status=Optimal` is the only in-file trace
    that a row was re-solved; the RELAXED GAP is not recorded anywhere in the schema. A `gap_rel`
    column appended at the END is safe for both `DictReader` and the positional `awk` readers used
    throughout this campaign, and is the honest way to carry "certified to 1% here, 1e-7 there".
  * A later SB re-run through the chain will overwrite a merged row with a fresh 1e-7 solve, which
    may cap again. That is correct behaviour, not a regression, but it means the merge is not
    permanent and the `gap_rel` column is what makes the difference visible.
