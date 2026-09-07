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
opposite. Re-solving the first of those cells returned **the identical answer, now certified**: 92
molecules either way, but `TimeLimit` after 1801 seconds became `Optimal` after 5.65. The cap had
been hiding a correct number, not a wrong one. The fix was not more compute — it was asking the
solver for a 1%-optimal answer instead of a seven-decimal-place one, on a quantity we report in whole
molecules.

The ladder defect was pure understatement and cost us real numbers: the first cell re-priced went
from 25 molecules to 30 at the same 100-reaction budget, purely because the old rungs forced it to
stop with 23 of its 100 reactions unspent. Across the campaign the median coarse cell was leaving a
third of the budget on the table, and ten cells had no readable number at all.

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
- Finish certifying the remaining capped rows and re-state every affected comparison from certified
  numbers only, noting that certification is now at a 1% gap rather than SPARROW's default.
- Reconcile the naive-vs-pruned direction flip on sEH once all four sEH rows are certified. If it
  survives certification it is a finding about pool construction and belongs in the paper; if it
  does not, it was a solver artefact and must not be quoted.
- Re-check that no table mixes a dense-ladder cell with a coarse one — mixing resolutions inside a
  seed band reads as a seed effect, which has cost this project a result before.

**Next steps in project**
- Complete the Stage-3 route discovery still outstanding, then re-price the whole matrix once from a
  single tooling version so every cell in the paper is priced identically.

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
- `/scratch/markymoo/rgfn_runs/ls_rvseh44-75790.out` — the decisive re-solve.
- `/scratch/markymoo/rgfn_runs/smk_reprice-75791.out` — the dense-ladder smoke.

## Relevant Versions

```
e8d35a7 Stage 2 named the REINVENT env `reinvent`; it is `reinvent4`, and nothing had ever exercised it
61c5e55 `eligible` was strict where the gate is inclusive, so a cell logged more modes than eligible molecules
6a043c2 REFACTOR_LOG: what changed in the Stage-2/3 work, and the seven things left open
```

`submit_reprice_cached.sh` and `_resolve_gate.py` are staged but not yet committed at the time of
writing — [TODO — add commit hash once committed].

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

**Re-solve, REINVENT sEH seed 44, pruned (job 75790):**

| | modes kept | cost (kept) | status | solve |
|---|---|---|---|---|
| before | 92 | 100 | TimeLimit | 1801.71 s |
| after | **92** | 99 | **Optimal** | **5.65 s** |

The answer was already correct; only its certification was missing. Note the new status is optimal
*to a 1% relative gap*, against SPARROW's default 1e-7 for the previously-certified rows — a 1% gap
on 92 modes is ±1 mode and does not disturb the 100-vs-92 comparison, but the asymmetry should be
stated wherever these rows are quoted. Seeds 42 and 43 were still queued at the time of writing
[TODO — add their re-solved rows].

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
