# All entrants / sEH+DRD2+ClpP — two measurement defects in the Stage-4 frontier, and what they were hiding

**Date:** 2026-09-06, ~11pm

> **Operational summary for anyone comparing against these numbers:**
> [`experiments/lsd_hubs/campaign/COMPETITOR_MATRIX.md`](../experiments/lsd_hubs/campaign/COMPETITOR_MATRIX.md)
> — where the results live, the quote rules, how to re-audit, and the traps that already cost
> time. This entry is the narrative; that file is what you need before quoting anything.

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
  mechanism named on each axis (reward-ranking redundancy; route sharing). Showing a single quadrant
  would let us choose our own winner. **The SELECTOR is the dominant axis, not the pool:** across sEH
  and ClpP the greedy arm favours S3-GFN in 4 of 4 pool×target combinations while SPARROW-Batching
  favours REINVENT in 3 of 4 (S3-GFN holds only sEH/pruned). An earlier draft of this entry said
  "three of four quadrants favour S3-GFN", which was true of sEH alone and does not generalise.
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
- `./experiments/lsd_hubs/campaign/COMPETITOR_MATRIX.md` — the handoff doc: result paths, quote
  rules, re-audit commands, and the four traps. Start here, not with this log.
- `./experiments/lsd_hubs/campaign/audit_matrix_coverage.py` — completeness across all 108 cells.
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

**THE COMPLETED TWO-BY-TWO**, modes at R=100, seeds 42/43/44, after both the certification and the
fine-ladder passes. DRD2 is omitted: S3-GFN routes almost nothing there ([078] traced it to an amine
absent from our ZINC stock), so the row cannot be compared.

| target | GREEDY naive | GREEDY pruned | SB naive | SB pruned |
|---|---|---|---|---|
| **sEH** S3-GFN | 45/50/75 | 40/60/70 | 56/38/49 | **98/100/100** |
| **sEH** REINVENT | 30/30/35 | 30/30/35 | **61/47/83** | 80/70/92 |
| **ClpP** S3-GFN | 40/40/40 | 40/40/45 | 45/43/42 | 64/57/62 |
| **ClpP** REINVENT | 30/35/30 | 30/35/30 | **81\*/69/71\*** | **86/74\*/72\*** |

`*` = still time-capped. Those rows are LOWER bounds, so certifying them can only raise REINVENT —
the direction of every REINVENT win above is therefore safe, and only its margin is uncertain.

**The selector is the dominant axis, and this corrects an earlier claim in this entry.** The greedy
arm favours S3-GFN in 4 of 4 pool×target combinations, with disjoint three-seed bands in all four.
SPARROW-Batching favours REINVENT in 3 of 4 — every combination except sEH/pruned, which is exactly
the one where S3-GFN reaches the 1.0 rxn/mode ceiling and cannot be beaten. An earlier version of
this entry reported "three of four quadrants favour S3-GFN"; that was read off sEH alone, and adding
ClpP reverses the summary. The pool axis matters, but it moves margins; the selector axis moves the
winner.

That is consistent with the uplift table rather than a separate fact: REINVENT gains 1.86×/2.57×
from greedy→SB against S3-GFN's 1.01×/1.65×, and a ~1.6× relative advantage is more than enough to
overturn the greedy ordering wherever S3-GFN is not already at the arithmetic ceiling. The
uncomfortable reading for us is that the arm CLAUDE.md designates to carry the headline — SB, being
the competitor's own optimiser and the stronger arm — is the arm on which the competitor usually
wins. That has to be stated plainly in the paper rather than resolved by choosing the greedy arm.

**Fine-ladder pass, second round (jobs 75801 naive / 75803 pruned, 5 min / 7 min):** all 12 cells
gained, uniformly +5 or +10 modes, closing 18-28 reactions of unspent budget down to 5-16. Examples:
`s3gfn_seh_seed44_pruned` 60→70, `s3gfn_clpp_seed44_pruned` 40→45, `reinvent_clpp_seed44_pruned`
25→30. No cell lost modes, consistent with a strictly finer superset of the previous rungs.

**Certifying the remaining capped rows (jobs 75810 naive / 75811 pruned): 11 of 12 succeeded, and
THREE CHANGED VALUE — which qualifies the "the cap was hiding a correct number" reading above.**

| cell | capped | certified @ gap 1e-2 | solve |
|---|---|---|---|
| reinvent clpp 42 naive | 81 | **81** | 1801 s → 3.2 s |
| reinvent clpp 44 naive | 71 | **71** | 1801 s → 2.3 s |
| reinvent clpp 43 pruned | 74 | **74** | 1801 s → 3.0 s |
| reinvent clpp 44 pruned | 72 | **72** | 1801 s → 2.6 s |
| saturn drd2 43 pruned | 9 | **9** | 1801 s → 2.4 s |
| saturn drd2 44 pruned | 1 | **1** | 1802 s → 1.9 s |
| tango drd2 42 pruned | 1 | **1** | 1802 s → 4.7 s |
| tango seh 43 pruned | 5 | **5** | 1802 s → 1404 s |
| saturn seh 42 pruned | 4 | **3** ↓ | 1802 s → 22.8 s |
| tango clpp 43 pruned | 14 | **13** ↓ | 1803 s → 8.2 s |
| tango clpp 44 pruned | 8 | **9** ↑ | 1802 s → 140 s |
| saturn seh 44 naive | 4 | 5, **STILL TimeLimit** | 1802 s → 1802 s |

**A certified value BELOW a capped one is not a contradiction, it is the price of the relaxed gap.**
`Optimal` at `--gap-rel 1e-2` means within 1% of the bound, and on a cell holding 3-14 modes one
percent IS about one mode — so on collapsed pools the relaxation costs ~1 mode, a ~25% RELATIVE
error where on REINVENT's 70-92-mode rows the same 1% is ±1 and immaterial. The capped incumbent and
the 1%-certified answer are both valid lower bounds; the larger of the two is the best available.
Those three cells are being re-solved at `--gap-rel 1e-6` (job 75814), which they can afford — they
converged in 2-140 s at 1e-2.

Two rows stay flagged rather than fixed. `saturn_seh_seed44` (naive) hit the wall even at 1e-2, so it
remains **solver-truncated** and its 5 modes is a lower bound. `tango_seh_seed43_pruned` needed
1,404 s of its 1,800 s at 1e-2, so it is certified but only barely, and a tighter gap may not close.

**The generalisation to carry forward:** gap relaxation is a safe lever for the LARGE cells, which is
where the caps actually hurt (REINVENT's ClpP and sEH rows all certified unchanged in ~3 s). It is
NOT safe as a blanket setting for small-count cells, where the tolerance is comparable to the
quantity being measured. Pick the gap against the mode count, not once for the campaign.

**A TIGHTER GAP DOES NOT RESCUE THOSE THREE CELLS, so they are not certifiable at any affordable
tolerance (job 75814, `--gap-rel 1e-6 --max-seconds 3600`):**

| cell | original (1e-7, capped) | 1e-2 | 1e-6 | best lower bound |
|---|---|---|---|---|
| saturn seh 42 pruned | 4 | 3 `Optimal` | 3, **TimeLimit** 3603 s | **4** |
| tango clpp 43 pruned | 14 | 13 `Optimal` | 13, **TimeLimit** 3602 s | **14** |
| tango clpp 44 pruned | 8 | 9 `Optimal` | **10**, TimeLimit 3602 s | **10** |

All three ran the full hour and capped. The `Optimal` they returned at 1e-2 in 8-140 s was the
tolerance letting CBC stop early, not the problem being easy. So these cells — plus
`saturn_seh_seed44` naive — must be reported as **solver-truncated lower bounds**, taking the largest
incumbent across solves. That makes the tally **11 certified of the original 15 capped rows, 4 still
truncated**, not 12 of 15.

**A MECHANISM I PROPOSED AND THEN REFUTED.** The obvious explanation is that these cells have big
route networks despite small mode counts. They do not:

| | modes | solve | reaction nodes | targets |
|---|---|---|---|---|
| HARD tango clpp 43 pruned | 13 | 3602 s | 9,457 | 446 |
| HARD saturn seh 42 pruned | 3 | 3603 s | **4,404** | 200 |
| easy reinvent clpp 43 pruned | 74 | **3 s** | **4,710** | 269 |
| easy reinvent seh 44 pruned | 92 | **6 s** | 7,262 | 376 |
| easy saturn drd2 43 pruned | 9 | **2 s** | 3,494 | 228 |

`saturn_seh_42_pruned` resists certification with a SMALLER network than `reinvent_clpp_43_pruned`,
which certifies in three seconds; and `saturn_drd2_43_pruned` holds only 9 modes on a 3,494-node
network and solves in two. Neither mode count nor network size predicts which cells resist. The
mode-to-target ratio is suggestive (1.3-3.1% for the hard ones against 24-37% for the easy) but
`saturn_drd2_43_pruned` sits at 3.9% and is easy, so that does not hold either.

**This is left as an open observation, not a mechanism.** What is actionable does not depend on
explaining it: check `time_capped` on every row before quoting it, take the largest incumbent across
solves as the lower bound, and do not assume a small pool implies a cheap solve. An earlier version
of this entry advised "pick the gap against the mode count" — that is too neat. The gap governs how
early CBC may stop; whether a cell certifies at all is a property of the instance that we cannot
currently predict from anything we record.

**FINAL STATE.**

*Ladder.* Every campaign cell is now on the fine ladder and **99 of 99 resolve at R=100** (the one
"no readable number" is a leftover smoke-test directory, `fraggfn_seh_seed42_stage2smoke50`, not a
cell). Getting there took four rounds, because two jobs reverted finished work: job 75769's SB
backfill overwrote `tango_clpp_seed44`'s dense row with a coarse one, and 75770 did the same to
`reinvent_drd2_seed42_pruned`. Both were found by auditing first rungs across the tree, not from any
log. `RUN_GREEDY=0` now exists so an SB-only backfill leaves the greedy arm alone.

*Certification.* `merge_longsolve_rows.py --apply` merged **11 cells** back into their main CSVs and
**refused 4** that are still capped, keeping `select_frontier.csv.pre_merge` backups. Merged files
carry a new final column: `gap_rel = 0.01` on the re-solved R=100 row and `1e-07` on the nine
untouched budget points, so the two tolerances stay distinguishable in the file itself rather than in
a reader's memory.

*The four that cannot be certified*, with the best lower bound across all three gaps tried. Recorded
in `results/UNCERTIFIED_R100_BOUNDS.json` (written by `write_uncertified_bounds.py`) because the
merge cannot touch a capped row and two of these main CSVs **understate the competitor**, which
flatters us:

| cell | 1e-7 | 1e-2 | 1e-6 | best bound | in main CSV |
|---|---|---|---|---|---|
| saturn seh 42 pruned | 4 | 3 | 3 | **4** | 4 ok |
| tango clpp 43 pruned | 14 | 13 | 13 | **14** | 14 ok |
| tango clpp 44 pruned | 8 | 9 | 10 | **10** | 8 — **understated by 2** |
| saturn seh 44 naive | 4 | 5 | — | **5** | 4 — **understated by 1** |

*DRD2 is no longer empty on the S3-GFN side, and it needs no further runs.* A claim made earlier in
this session — that seed 42 was a "pre-fix leftover" needing the same `_stage2fix` treatment as
seeds 43/44 — was **wrong**, and the Stage-2 logs say so plainly: seed 42 records
`stop_reason: target-reached` with 500 modes from 5,740 eligible molecules, while 43 and 44 record
`stalled` at 422 and 168 modes from 2,463 and 378. The tag marks **which cells stalled**, not two
pipelines, and there are no separate fix directories — all three consumed the same
`stage2_candidates.csv`. So 47/99/90 (SB, pruned) is a legitimate three-seed band.

What the band shows is large within-generator variance, driven by ROUTABILITY rather than by the
pipeline: seed 42's route artifact is 286 KB against seed 43's 2.3 MB (~10% vs ~83% routed). [075]
traced DRD2's routing failures to an amine absent from our ZINC stock, so seed 42 plausibly made more
of that chemistry. **Quote the DRD2 band with its spread, never a mean.**

**COVERAGE AUDIT: 98 of 108 cells complete** (6 generators x 3 targets x 3 seeds x 2 pools, where
complete = pool + routes + greedy row at R=100 + SB row at R=100). Written as
`audit_matrix_coverage.py` so it can be re-run rather than reconstructed.

The ten incomplete cells are not ten gaps:

| what | cells | status |
|---|---|---|
| SynFormer sEH | 6 | blocked on Stage-1 training (jobs 75753-55, ~13.5 h left) |
| `fraggfn:drd2:44` pruned | 1 | in progress, job 75770 |
| `tango:seh:44` naive | 1 | was GATED at `MIN_MODES=10` with 8 modes in its top-500; queued with `MIN_MODES=1` (job 75823) so it becomes a flagged datapoint instead of an exclusion |
| `s3gfn:drd2:42` naive, `saturn:seh:42` naive | 2 | **not gaps — results.** See below |

One trap the audit itself hit: its first version hard-coded `_select_N500` and reported
`synformer_drd2_seed44_pruned` as missing its SB arm. That cell is pool-limited and writes `_N495`,
where it holds 97 modes at R=100, `Optimal`. Glob the size, never assume it — the same lesson the
campaign already learned on the greedy directories.

**A REWARD-RANKED POOL CAN BE 100% UNSYNTHESIZABLE.** Those last two cells have 2-byte
`multiaiz_routes.json` files and empty result directories, which looks exactly like a crashed
discovery. It was not — `discovery_timing.json` says the work happened:

| cell (naive pool) | targets | discovery | s/target | rc | routes |
|---|---|---|---|---|---|
| s3gfn drd2 42 | 500 | 17,650 s (4.9 h) | 35.3 | 0 | **0 routed** |
| saturn seh 42 | 500 | 36,978 s (10.3 h) | 74.0 | 0 | **0 routed** |
| s3gfn drd2 42 **pruned** | 500 | 18,672 s | 37.3 | 0 | 292 KB -> 47 modes at R=100 |

Both ran to completion at per-target rates inside the normal 16-89 s band ([078]), so this is a
measured zero and not a missing measurement — the check that separates the two is
`discovery_timing.json`, not the file size. The same cell's DIVERSE 500 routes fine. So on these
cells the top-500-by-reward is entirely unmakeable while a diversity-forced 500 drawn from the same
above-gate set prices 47 modes.

That is the strongest available form of the naive-pool pathology and it points the same way as
[078]'s mechanism (routable molecules are flatter and greasier; the sp3-rich drug-like majority
fails retrosynthesis): reward here correlates with the unroutable direction strongly enough to zero
out the entire reward-ranked prefix. It is also the cleanest argument for reporting the pruned pool
rather than the naive one — not because it flatters anyone, but because the naive pool can contain
nothing a chemist could make.

**A COARSE LADDER CAN PRODUCE AN EMPTY FILE, NOT JUST A LOW NUMBER — a distinct failure mode.**
When job 75770 created `fraggfn_drd2_seed44_stage2_pruned`'s greedy arm it used the launcher's
default ladder, whose first rung is 25. That pool routes **20 of 500 molecules (4.0%, 175 route
entries)**, so every rung was skipped and `greedy_frontier.csv` was written containing **only its
header**. The cell had a perfectly good SB row the whole time (20 modes, `Optimal`) and no greedy row
at all. On the fine ladder it reads **20 modes @ 88 rxn** (job 75827, 20 s).

Greedy and SB both deliver exactly 20 here because 20 is everything that routed — a `pool-exhausted`
cell in the CLAUDE.md taxonomy, where `cost_kept_rxns` and not `used_rxns` is the honest cost.

**The detector for this is different from the coarse-ladder detector, and mine was blind to it.** The
clobber audit flags a cell whose first rung is >5; a CSV with NO rungs passes that test silently
because there is nothing to compare. The committed `audit_matrix_coverage.py` catches it only because
it requires a READABLE R=100 row rather than the file's existence — which is the right predicate for
any completeness check over this tree. An "exists and is non-empty" check would have passed a
header-only file.

**CLOSED OUT: 100 of 108 cells complete, 0 on a coarse ladder, 101 resolving at R=100.** The eight
incomplete are the six SynFormer sEH cells (Stage-1 training) and the two zero-route results above,
which are findings rather than gaps. Four cells carry an uncertifiable SB row, recorded with their
best bounds in `results/UNCERTIFIED_R100_BOUNDS.json`.

**`tango:seh:44` naive, the cell that was gated, is now a datapoint — and it refutes a prediction I
made when queueing it.** I expected an 8-mode pool to route ~0, as `saturn:seh:42` naive had. It
routed **305 of 500 (61.0%)**. Routability and mode-diversity are independent: those 305 molecules
collapse to just **5 distinct modes**, so the cell reads **5 modes @ 37 rxn** with 63 reactions
unspent — `pool-exhausted`, where CLAUDE.md says to quote `cost_kept_rxns` (37) and never read the
unspent budget as a choice. Its SB arm gives 3 modes. Worth the 7.5 h precisely because the guess was
wrong.

**Both launcher ladder defaults are now the fine ladder** (`submit_competitor_routes.sh` was
25,50,75,…; `submit_native_routes.sh` was 5,10,…,25,50,…). Leaving them coarse would have
re-introduced every problem in this entry on the next cell submitted, and the second header-only CSV
of the night was caused by exactly that — I un-gated `tango:seh:44` with `MIN_MODES=1` but did not
pass `MODE_POINTS`, and a cell gated FOR HAVING FEW MODES is by definition one the coarse ladder
cannot price. The two knobs belong together, which is why the default now removes the choice.

**Three distinct ways the tooling produced a confident wrong answer tonight**, all found by auditing
artifacts across the whole tree and none visible in a job log, every one of which reported success:

| failure | what it looked like | the detector |
|---|---|---|
| stale primary from the `_longsolve` split | a certified row existed, but the main CSV still held the capped one | prefer `_longsolve`, then MERGE so there is one file |
| silent revert by a two-arm launcher | two successful jobs, the later one undoing the earlier | the first ladder rung in the CSV |
| header-only CSV | file exists, non-empty, no data rows | require a READABLE R=100 row, not existence |

**CORRECTION: every SynFormer greedy figure in this entry above is the REQUEST, not the DELIVERY.**
`greedy_frontier.csv` records `n_modes` (asked) and `n_targets_priced` (delivered), and [078]
established that they are equal on multiaiz cells and diverge on SynFormer's native routes. Re-checked
after tonight's re-pricing, because the re-price changed which rung each cell lands on and the rule
could have stopped holding: **88 cells still have request == delivery, and all 17 divergent cells are
SynFormer's.** So the rule stands and the correction is scoped to one entrant.

Restated at the R=100 readout, delivery first:

| cell | delivered @ rxn | asked | delivery |
|---|---|---|---|
| seh 42 (both pools) | **41** @ 97 | 55 | 75% |
| seh 43 (both pools) | **40** @ 96 | 50 | 80% |
| drd2 42 | **66** @ 93 | 90 | 73% |
| drd2 43 | **69** @ 98 | 80 | 86% |
| drd2 44 | **66** @ 97 | 75 | 88% |
| clpp 42 | **39** @ 93 | 45 | 87% |
| clpp 43 | **42** @ 99 | 45 | 93% |
| clpp 44 | **41** @ 96 | 50 | 82% |

The ladder improvements reported earlier are still real — `synformer_clpp_seed42` genuinely moved
from a rung at 52 reactions to one at 93 — but the headline number for that cell is **39 delivered**,
not the 45 requested. The mechanism is [078]'s: MultiAiZ offers many routes per molecule while native
routes offer one, so under the `count` objective the solver declines targets it cannot make cheaper.

**A SECOND SYNFORMER PROPERTY, and it is not a bug.** Its naive and pruned greedy frontiers are
**identical at every shared rung** — for `synformer_seh_seed42` the two CSVs agree exactly from 2
through 90 modes, and the pruned file simply continues to 150 where the naive pool runs out. So the
naive pool's 90 modes ARE the pruned pool's first 90, in the same order: reward ranking and diversity
selection agree perfectly at the top for this generator. That is the extreme end of the redundancy
spectrum measured earlier (Saturn/TANGO 2-4% of a top-500 mutually dissimilar, FragGFN 87-100%,
SynFormer effectively 100% agreement in ordering). Their SB rows still differ sharply (21 vs 57 on
seh 42), because SPARROW optimises over the whole pool rather than a reward-ordered prefix.

**SynFormer's sEH output is mode-limited:** 2,000 candidates, 100% above the 5.68 gate, but only
**284/290 mutually dissimilar molecules** in the whole set (90 and 88 inside the top-500). Its pruned
pools are therefore pool-limited by construction — `_N284`, `_N290` — and its routed fraction is
100% by definition, since it carries its own routes. High reward, narrow chemistry.

**MATRIX COMPLETE: 106 of 108 cells.** Zero missing pools, zero missing routes, zero cells on a
coarse ladder, 107 resolving at R=100. The two incomplete cells are `s3gfn:drd2:42/naive` and
`saturn:seh:42/naive` — the zero-route findings, which are results and not gaps.

**SynFormer sEH, the last six cells** (jobs 75921/75922, ~34 min each — no MultiAiZ, since SynFormer
carries its own routes). Quoted as DELIVERED targets, with the request in brackets:

| seed | above gate | modes available | in top-500 | greedy @ R=100 | SB naive | SB pruned |
|---|---|---|---|---|---|---|
| 42 | 2000 (100%) | 284 | 90 | **41** @ 97 (asked 55) | 21 (cost 39) | **57** (cost 95) |
| 43 | 2000 (100%) | 290 | 88 | **40** @ 96 (asked 50) | 16 (cost 28) | **55** (cost 98) |
| 44 | 2000 (100%) | 318 | 77 | **41** @ 98 (asked 50) | 19 (cost 35) | **60** (cost 98) |

The training was verified before anything downstream ran, because the earlier SynFormer sEH attempt
reported success while producing unscored molecules: 2,000 rows per seed, **0 NaN**, ~1,990 distinct
score values (so the reward varied and the fork was not poisoned), all above the 5.68 gate, and a
2,000-line `routes.jsonl`. Checking the exit code would not have distinguished this from that
failure.

Two things about these cells to carry into the write-up. Their **naive SB rows are far below their
pruned ones** (21/16/19 against 57/55/60) — the largest pool effect of any entrant, and the same
direction as the redundancy story: a reward-ranked 500 drawn from only 284-318 distinct molecules is
mostly duplicates, and SPARROW cannot turn duplicates into modes. And SynFormer's **greedy exceeds
its naive SB** (41 vs 21), the sub-1.0 uplift pattern also seen on Saturn and TANGO: on a pool this
redundant, SPARROW's shared-intermediate objective does not chase mode count.
