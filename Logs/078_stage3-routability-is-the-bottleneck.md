# Stage 3 — the molecules survive the gate and then die at retrosynthesis

**Date:** 2026-08-31, ~10am (campaign started 2026-08-28)

## Question

Once a generator has been given a full pool of 500 distinct high-scoring molecules, how many of them
can a retrosynthesis planner actually make — and does that depend more on the generator or on the
target?

## Context & Summary

Stage 2 ([077]) equalised the entrants: every cell now enters Stage 3 with a pool of 500 molecules
holding ~500 distinct modes, so nobody is disadvantaged by pool size. Stage 3 then asks the planner
(MultiAiZ, ZINC stock) for routes and prices the result two ways — a diversity-aware greedy and
SPARROW-Batching. Entry 075 showed that pool size, not pipeline quality, had been driving earlier
comparisons; this stage is where the pools finally meet a synthesis planner on equal terms.

The campaign is **~13% complete** (10 of 76 route runs) and stalled on cluster contention, so this
entry records what is already settled and — importantly — which of its claims are still confounded.
Three findings are solid enough to act on, two are qualified, and one is a pipeline defect that
would have mislabelled published numbers.

## Answer

**Routability, not diversity or reward, is what bounds this benchmark.** Every cell entered Stage 3
with ~500 modes; the ones that come out useful are the ones whose molecules a planner can make.
FragGFN loses ~90–97% of its pool at this step on every target tested, and the survivors are the
flat all-aromatic molecules whose fragment joints happen to coincide with cross-coupling templates —
its route steps form aryl–aryl bonds 46–56% of the time against S3-GFN's 2%. That is a mechanism,
not just an observation: a generator that joins fragments makes bonds that reaction templates do not
make, except where the joint happens to be a biaryl.

But the effect is **not purely a generator property**. S3-GFN routes at 70–79% on sEH and collapses
to 10% on DRD2 — a target where entry 075 already traced 0/500 routing to an amine absent from our
catalogue. So the reaction-aware advantage is measurable where stock is not the binding constraint,
and DRD2 is confounded.

## Relevance to our Publication

The paper argues that composing reactions beats composing fragments. This is the most direct
evidence yet, and it is mechanistic rather than assertive: same pool size, same gate, same catalogue,
same planner, and a 10–14× difference in how much of the pool survives — with a measured reason
(bond types) rather than a plausible one. It also supplies the honest limitation a reviewer will
look for: on one of three targets our own catalogue, not the generator, is what fails.

## Next Experiments

**Refining for publication.** Finish the seed grid — one seed carries most of these numbers, and
S3-GFN's own sEH seeds already differ 1.5× on cost (222 vs 145 reactions for 100 modes), so
single-seed claims are not safe. Get ClpP's S3-GFN cells, which break the sEH/DRD2 tie: if S3-GFN
routes well on ClpP while FragGFN stays at 10%, the generator effect stands on two targets of three.
Re-run the frontier for pool-limited cells on the finer mode ladder (cheap — routes are cached).

**Next steps in project.** Resolve whether the greedy arm should FORCE all N targets rather than let
SPARROW decline them (see Results); this changes what "modes at 100 reactions" means for every
route-carrying entrant. Then the competitor comparison itself — REINVENT, Saturn and TANGO have not
started.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork` (worktree `.claude/worktrees/fraggfn-stage2`,
branch `worktree-fraggfn-stage2`).

**Scripts**
- `./experiments/lsd_hubs/campaign/submit_competitor_routes.sh` — Stage 3 for ROUTE-LESS entrants:
  saturation gate → pool → MultiAiZ discovery → both frontiers. `CANDS` is now an override so it can
  consume a Stage-2 pool.
- `./experiments/lsd_hubs/campaign/submit_competitor_routes_chain.sh` — runs the above for many
  cells; `USE_STAGE2=1` derives each cell's Stage-2 candidates, `REPO_DIR` selects which tree's cell
  script is snapshotted.
- `./experiments/lsd_hubs/campaign/submit_native_routes.sh` — NEW. Stage 3 for a generator that
  already carries routes (SynFormer): no MultiAiZ, prices via `--route-source external`.
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — both selectors; emits
  `n_modes` (requested) and `n_targets_priced` (delivered), which are NOT always equal.

**Results**
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/<tag>_N<N>/` — `pool.smi`,
  `pool_scores.csv`, `multiaiz_routes.json`, `discovery_timing.json`.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/<tag>_greedy_N<N>/greedy_frontier.csv` — the
  frontier, plus `network_m<M>/network_stats.json` per mode point.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/s3_fg_*-{75169..75173}.out` (timed out at 20 h, 1 cell each),
  `s3_s3gfn_pruned-75190.out`, `nr_smoke_sf42-75300.out`.

## Relevant Versions

```
7fdd4c7 Stage 3 for route-carrying generators, and the pricing asymmetry it exposes
abe06e1 CFG_FORCE, to ask whether a `sampler-capped` cell is genuinely saturated
5587e77 A capped sampler is not an exhausted generator
32e6ffd The chain ran the WRONG cell script and built _stage2-tagged pools out of Stage-1 data
be428d7 The routes chain can drive Stage-2 pools: USE_STAGE2=1
5c6437d Stage 3 can read a Stage-2 pool: CANDS becomes an override
```

Branch `worktree-fraggfn-stage2`, not merged.

## Relevant Resources

**Sources** — `[koziarski2024rgfn]` (reaction composition); entry 075 (DRD2 catalogue coverage);
entry 077 (the Stage-2 pools these runs consume); `docs/LSD_FLOW_BENCHMARK_PLAN.md` §0.

**Packages** — AiZynthFinder via `submit_multiaiz_discover.sh` (stock=zinc, uspto expansion,
n_iters=5); SPARROW `LinearSelector` via `validation/lsdflow/adapters/workers/sparrow_worker.py`;
RDKit for the structural analysis.

## Method

1. Stage-2 pools → `build_s3gfn_pools.py` (top-N by reward, gate from `matrix16/targets.py`), both
   naive and pruned constructions, N=500.
2. `submit_multiaiz_discover.sh` for route-less entrants; native `routes.jsonl` for SynFormer.
3. `sparrow_select_frontier.py` twice per cell (greedy, then SPARROW-Batching).
4. Post-hoc analysis (login node, CPU): routed-vs-unrouted structural comparison over the pooled
   seed-42 FragGFN cells; per-step bond-formation classification from the route JSON; naive-vs-pruned
   pool overlap; `network_stats.json` inspection to separate MILP choice from route coverage.

## Results

### Routability, by cell (pool = 500 in every row)

| cell | routed | frontier (modes@reactions) |
|---|---|---|
| S3-GFN sEH s42 | 351 (70.2%) | 25@55 · 50@112 · 75@154 · 100@222 · 125@267 · 150@319 |
| S3-GFN sEH s44 | 394 (78.8%) | 25@33 · 50@69 · 75@101 · 100@145 · 125@181 · 150@218 |
| S3-GFN DRD2 s42 | 50 (10.0%) | 25@57 · 50@115 |
| FragGFN sEH s42 | 35 (7.0%) | 25@101 |
| FragGFN sEH s42 pruned | 28 (5.6%) | 25@103 |
| FragGFN ClpP s42 | 52 (10.4%) | 25@120 · 50@237 |
| FragGFN DRD2 s42 | 32 (6.4%) | 25@117 |
| FragGFN DRD2 s43 | 12 (2.4%) | none — below the ladder |

For scale, the pre-Stage-2 pools of other entrants route at 52–96% (REINVENT sEH 92–95%, ClpP 52–63%;
S3-GFN sEH 90–96%).

### Why FragGFN's molecules fail — and why the survivors are the wrong ones

Pooled over its three seed-42 cells, 119 routed vs 1,381 unrouted:

| property | routed | unrouted |
|---|---|---|
| biaryl bonds | 3 | 1 |
| aromatic rings | 5 | 3 |
| Fsp3 | 0.16 | 0.33 |
| heteroatoms | 3 | 5 |
| logP | 7.24 | 5.88 |

The molecules that route are the flat, greasy, all-aromatic ones; the more sp3- and heteroatom-rich
(more drug-like) majority fails. Measured on the routes themselves, **46% / 56% / 19%** of FragGFN's
route steps (sEH / ClpP / DRD2) form an aryl–aryl bond, against **2%** for S3-GFN on either sEH seed.

### A DEFECT THAT WOULD HAVE BEEN PUBLISHED: request vs delivery

`greedy_frontier.csv` records `n_modes` (requested) and `n_targets_priced` (delivered) and computes
`rxn_per_mode = used_rxns / n_modes`. On every multiaiz cell these are equal, so the distinction is
invisible. On SynFormer's native routes they diverge systematically — 13/15, 43/50, **89/100**,
112/125, 133/150 — so the row "100 modes at 198 reactions" delivers **89** modes at 198 (2.22
rxn/mode, not the recorded 1.98).

Cause established, not assumed: NOT stereo collapse (1 molecule in 500) and NOT route coverage
(`network_stats` for m=100 reads `n_targets: 100`, `n_routeless_targets: 0`, `n_dropped_steps: 0`).
The MILP had all 100 and chose 89. The reason is structural — MultiAiZ emits MANY candidate routes
per molecule (1,422 reaction nodes / 217 shared compounds for 100 S3-GFN targets) while native routes
give exactly ONE (232 / 55), so under the `count` objective the solver has no cheaper alternative for
an expensive target and declines it. **Quote `n_targets_priced`, never `n_modes`.**

### Measured discovery cost (the chains were sized wrong)

6–12 h per N=500 cell, not the 3.67 h of the reference pool used for sizing: FragGFN 9.4–12.4 h
(68–89 s/target), Saturn 5.9–12.0 h, REINVENT ClpP 6.0–6.7 h. FragGFN is both the least routable AND
the slowest — unroutable molecules cost more planner time, not less. Five chains timed out at 20 h
having completed one cell each; discovery is idempotent, so resubmission resumes.

### Naive vs pruned pools overlap enormously for some generators

Shared molecules between the two constructions: FragGFN DRD2 **100%** (499/500), FragGFN sEH 87%,
REINVENT 40–90%, S3-GFN 10–69%, Saturn **4–19%**, TANGO sEH 5%. Where the above-gate set is already
almost all-distinct, pruning is a no-op and the second MultiAiZ run is an exact duplicate. This
doubles as a free redundancy measure of each generator — Saturn's 4–19% is what its known mode
collapse predicts.

### WHAT IS NOT ESTABLISHED

- **DRD2 is confounded.** S3-GFN routes at 10% there, so the FragGFN gap narrows to 1.6–4× on that
  target against 10–14× on sEH. Entry 075 traced DRD2's routing failure to catalogue coverage (an
  amine absent from stock), so the binding constraint there may be ours, not the generator's.
- **One seed carries most rows.** S3-GFN's own sEH seeds differ 1.5× on cost for the same 100 modes.
- **The DRD2 aryl–aryl figure (19%) does not fit the mechanism** the other two targets support.
- **REINVENT, Saturn and TANGO have not run at all** — the competitor comparison this campaign exists
  for is 0/58 complete, blocked on cluster fairshare (measured `EffectvUsage` 0.279 against
  `NormShares` 0.023, i.e. ~12× our entitlement) after five of nine nodes were reserved for another
  group.
- A pool-limited cell delivering <25 modes produced NO frontier row at all until the ladder was
  changed; `submit_native_routes.sh` starts at 5, the route-less path still starts at 25.

### One Stage-2 result worth keeping

`s3gfn_drd2_seed42` routed **0/500** from its budget-faithful pool in entry 075. From the Stage-2
upsampled pool it routes **50/500** and produces real frontier rows. Upsampling recovered a cell that
previously yielded nothing.
