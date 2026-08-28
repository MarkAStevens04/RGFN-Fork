# S3-GFN / sEH + DRD2 + ClpP — greedy vs SPARROW on one retrosynthesis, and why S3-GFN's molecules are cheap

**Date:** 2026-08-28, ~12pm

## Question

When one generator's molecules are routed once and then handed to two different shopping strategies
under a budget of 100 reactions, which strategy delivers more distinct high-scoring molecules — and
is the winner's advantage real chemistry or an artefact of how we measured it?

## Context & Summary

**Context.** The workshop claim we are trying to support is that our hub-batching approach delivers
more distinct high-scoring molecules per unit of synthesis effort than the standard pipeline a
chemist would otherwise run: take a generator's output, plan routes for it, then let a solver pick
what to make. Hub-batching reaches **82** distinct molecules for 100 reactions on sEH. To claim we
beat the standard pipeline, the standard pipeline has to be given a fair chance — and on sEH it was
not. S3-GFN, the non-reaction baseline, produced only **65** distinct above-bar molecules from its
2,000-molecule sample on seed 43, and after the diversity filter that is **28** genuinely different
ones. A solver choosing from 28 cannot possibly reach 82, so winning against it would have shown
nothing about *choosing* at all.

**Summary.** We did three things. First, we let S3-GFN draw far more molecules from the model it had
already trained — no extra training, the same frozen model — until it could supply a full pool.
Second, we ran both shopping strategies over the identical set of planned routes for every S3-GFN
cell, and built a table that reports four numbers per cell so a small pool can never masquerade as a
large one. Third, when the enlarged sEH result came back at a suspiciously round "100 molecules for
exactly 100 reactions", we opened up the individual molecules and their recipes to see whether the
number was real.

## Answer

The round number is real, and what it reveals is more interesting than the comparison it came from:
**most of what S3-GFN makes is a single reaction away from something you can buy.** On the enlarged
sEH pool, 98 of the 100 selected molecules are one step from purchasable material, every one of the
144 starting materials involved is genuinely in the catalogue, and all 100 clear the quality bar and
are genuinely different from one another. One reaction per molecule is the cheapest anything can
possibly be, so the solver did not out-plan anyone — it was handed a shopping list. This is a
property of the generator, not of the solver: across every pool we have routed, **S3-GFN's molecules
are one-step 39–80% of the time, REINVENT's 9–34%, and Saturn's 0–16%.** The comparison that matters
for the paper is therefore not "which shopping strategy wins" in general, but "which wins when the
molecules are not already trivial to make" — and the data to test that is already on disk.

Separately, giving S3-GFN a bigger pool changed its answer completely rather than marginally: the
same cell went from 25 molecules delivered to 100. Any earlier comparison against the small pool was
measuring the pool, not the pipeline.

## Relevance to our Publication

The workshop submission (GEM / ICLR MLDD tier) rests on out-performing the standard pipeline, and
the first question a reviewer asks about a favourable comparison is whether the baseline was given a
real chance. Three things in this entry pre-empt that: the baseline now gets the largest pool its own
trained model can supply, it gets both shopping strategies rather than only the one that suits us,
and where it wins we say so — on the enlarged sEH pool the standard pipeline reaches 100 distinct
molecules against hub-batching's 82. Reporting that, together with the reason (its molecules are
one step from the catalogue), is a far stronger position than a table where the baseline quietly
never had enough molecules to compete. It also sharpens the claim we can actually defend: our
advantage should be argued where synthesis is non-trivial, which is precisely where a
reaction-grounded generator is supposed to matter.

## Next Experiments

**Refining for publication.** The enlarged pools cost extra scoring calls — 10,271 / 41,346 / 10,342
for seeds 42 / 43 / 44 — which is up to four times the model's own training budget. That number has
to appear beside any result taken from these pools, and reviewers will reasonably ask whether
hub-batching's 82 came from a comparable amount of sampling; if it did not, the asymmetry belongs in
the same sentence. The three seeds also differ by a factor of four in how hard they were to enlarge,
so per-seed reporting rather than an average. Finally, the molecules selected here are small
(median 22 heavy atoms) and one building block appears in 37 of the 100 — worth stating plainly
rather than letting a reader assume a diverse, elaborate library.

**Next steps in project.** Test the hypothesis this entry raises: run the same two-strategy
comparison on Saturn's pools, where almost nothing is a single step from the catalogue, and see
whether the advantage reverses. That needs no new generation — Saturn is routed on seven cells
already. The remaining generators (TANGO, SynFormer) are being handled separately.

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**
- `experiments/lsd_hubs/campaign/s3gfn_sample_more.py` — loads a finished S3-GFN run's frozen
  checkpoint and draws additional samples until N molecules clear the quality bar, scoring each and
  writing `pairs.csv` + `bigsample_meta.json`, then ingesting a standard candidate dataset. Takes no
  gradient steps; the training budget is untouched.
- `experiments/lsd_hubs/campaign/submit_s3gfn_bigsample.sh` — per-cell chain: enlarge the pool,
  ingest, plan routes with MultiAiZ, then run both selection arms on both pool constructions at a
  5-mode ladder.
- `experiments/lsd_hubs/campaign/s3gfn_selection_table.py` — reads both arms off the same routes at
  a fixed reaction budget and prints four numbers per cell (delivered, makeable, pool size,
  reactions used) plus a derived stop reason. Discovers cells from disk.
- `experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — the two selection arms themselves.
- `experiments/lsd_hubs/campaign/build_s3gfn_pools.py` — builds the naive (top-N by score) and
  pruned (top-N mutually dissimilar, Morgan r=3/2048, tau=0.5) pools.

**Models**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed{42,43,44}/s3gfn_seh/s3gfn_seh-seed{N}_model.pt`
  — the trained S3-GFN policies, loaded frozen for the enlarged draw. Seed 43's is at step 156, the
  end of its 10,048-call training budget.
- `data/models/aizynthfinder/{uspto_model.onnx, uspto_templates.csv.gz, zinc_stock.hdf5}` — the
  retrosynthesis model and the purchasable catalogue every route must terminate in.

**Datasets**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed{N}_bigsample/pairs.csv` —
  every molecule drawn in the enlarged sample with its score. Kept separate from the
  budget-faithful run so the two can never be confused.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed{N}_bigsample/bigsample_meta.json`
  — the scoring cost of the enlarged draw; required alongside any number taken from these pools.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_seed43_big_pruned_N270/multiaiz_routes.json`
  — the planned routes for the enlarged pruned pool; the source for the one-step analysis.

**Results**
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_selection_R100.csv` — the full table.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed43_big_pruned_select_N500/milp_L0_R100.json`
  — the 100 selected molecules, the solver status, and the total reaction count.

**Job Logs**
- `/scratch/markymoo/rgfn_runs/s3_bigsample-75098.out` — the seed-43 enlarged draw.
- `/scratch/markymoo/rgfn_runs/s3_big_seh43-75101.out` — seed 43 end to end.
- Jobs 75099 / 75100 failed at the ingest step and were recovered by hand; 75123 / 75124 re-run them.

## Relevant Versions

`71fd9e5` — "Give S3-GFN a pool SPARROW could actually win from: 65 above-gate molecules become 649"
(the enlarged-sampling script and submit chain).

Not yet committed: the `s3gfn_selection_table.py` cell-discovery fix, and the two `s3gfn_sample_more.py`
fixes described in Method step 2. **[TODO — add commit hash after pushing]**

## Relevant Resources

**Sources**
- AiZynthFinder / MultiAiZ set-based retrosynthesis — the route planner (`[genheden2020aizynthfinder]`).
- SPARROW — the reaction-budget selection solver (`[fromer2024sparrow]`).
- S3-GFN — the non-reaction baseline generator.

**Packages**
- `aizynthfinder` (own `aizynth` conda env) — route planning, and the stock membership check.
- `rdkit` — canonicalisation, Morgan fingerprints, molecular properties.
- `pulp` / CBC — the selection solver, via `sparrow_select_frontier.py`.
- `torch` + `transformers` (own `s3gfn` conda env) — loading and sampling the frozen policy.

## Method

1. **Enlarge the pool.** For each sEH seed, load the trained checkpoint frozen and draw samples in
   rounds of 4,000 unique valid molecules, scoring each against the sEH proxy, until 500 clear the
   7.0 bar or 40,000 molecules have been scored:
   `python experiments/lsd_hubs/campaign/s3gfn_sample_more.py --run-dir <run> --target-above-gate 500 --gate 7.0 --max-samples 40000`
2. **Two defects fixed during this step.** The seed was being read from `run_config.yaml`, which
   says `seed: 42` for all three cells because the original runs passed `--seed N` on the command
   line and the override was never written back — seed 44's pool was ingested labelled seed 42. It
   is now taken from the run directory name. Separately the ingest subprocess needs the CUDA
   libraries on `LD_LIBRARY_PATH` (dgl/graphbolt) and is now run through `~/bin/rgfn-smoke-env.sh`;
   without it jobs 75099/75100 died *after* their sampling had succeeded.
3. **Plan routes and run both arms** for both pool constructions, at a 5-mode ladder:
   `CELLS="s3gfn_seh/seed43" sbatch experiments/lsd_hubs/campaign/submit_s3gfn_bigsample.sh`
4. **Read both arms at the fixed budget:**
   `python experiments/lsd_hubs/campaign/s3gfn_selection_table.py --budget 100 --out .../s3gfn_selection_R100.csv`
5. **Audit the 100-for-100 result.** Take the selected molecules from `milp_L0_R100.json`; count the
   cheapest route length for each from `multiaiz_routes.json`; confirm every selected molecule is
   above the bar and mutually dissimilar at tau=0.5; and test every starting material of every
   one-step route for membership in `zinc_stock.hdf5` under the `aizynth` env.
6. **Count one-step molecules across every routed pool** to see whether the effect is specific to
   S3-GFN.

## Results

**Enlarging the pool (frozen policy, no training).**

| seed | scored | distinct above bar | modes (tau=0.5) | rounds | wall |
|---|---|---|---|---|---|
| 42 | 10,271 | 522 (from 154) | 239 (from 89) | 2 | 2.9 min |
| 43 | 41,346 | 649 (from 65) | 270 (from 28) | 4 | 13.4 min |
| 44 | 10,342 | 586 (from 168) | 209 (from 60) | 2 | 2.6 min |

The share of samples clearing the bar decays with sampling — 3.3 / 2.8 / 2.2 / 1.8 / 1.6% across
seed 43's rounds — but the *mode rate* barely moves (seed 42 0.578 → 0.458; seed 43 0.431 → 0.416;
seed 44 0.357 → 0.357), so the extra molecules are about as diverse as the originals.

**Seed 43, before and after enlargement, at 100 reactions.**

| pool | molecules | greedy | SPARROW |
|---|---|---|---|
| original naive | 65 | 25 @ 41 rxn | 25 @ 80 rxn |
| original pruned | 28 | 25 @ 40 rxn | 25 @ 49 rxn |
| enlarged naive | 500 | 55 @ 99 rxn | 46 @ 100 rxn |
| enlarged pruned | 270 | 60 @ 100 rxn | **100 @ 100 rxn** |

**Audit of the 100-for-100 cell** (`s3gfn_seh_seed43_big_pruned`, solver Optimal, not time-capped):

| check | result |
|---|---|
| molecules selected | 100 |
| above the 7.0 bar | 100/100 (min 7.063, median 7.414, max 8.156) |
| mutually dissimilar at tau=0.5 | 100/100 |
| cheapest route = 1 reaction | 98/100 (remaining 2 need 2 reactions) |
| starting materials in the purchasable catalogue | **144/144 (100%)** |
| median size of selected molecules | 291 Da, 22 heavy atoms |
| most-reused starting material | 5-aminoindane, in 37 of 100 |

Across the whole enlarged pruned pool, 112 of 214 routed molecules have a one-reaction route.

**One-step share by generator** (median over its routed pools; n = number of pools):

| generator | one-step share |
|---|---|
| S3-GFN | 39–80% |
| REINVENT | 9–34% |
| Saturn | 0–16% |

By target the medians are ClpP 34% (n=16), sEH 19% (n=20), DRD2 14% (n=17) — a much smaller spread
than the between-generator one, which is why the effect is attributed to the generator.

**Full S3-GFN table at 100 reactions** (delivered / reactions used; `pool` = molecules routed,
`makeable` = largest mode count actually priceable):

| cell | pool | makeable | greedy | SPARROW | stop reason |
|---|---|---|---|---|---|
| seh_42 | 154 | 50 | 50 @ 86 | 50 @ 100 | budget-binding |
| seh_42_pruned | 89 | 60 | 55 @ 96 | 62 @ 100 | budget-binding |
| seh_43 | 65 | 25 | 25 @ 41 | 25 @ 80 | pool-exhausted |
| seh_43_pruned | 28 | 25 | 25 @ 40 | 25 @ 49 | pool-exhausted |
| **seh_43_big** | 500 | 150 | 55 @ 99 | 46 @ 100 | redundant |
| **seh_43_big_pruned** | 270 | 150 | 60 @ 100 | 100 @ 100 | budget-binding |
| seh_44 | 168 | 50 | 50 @ 76 | 30 @ 100 | redundant |
| seh_44_pruned | 60 | 55 | 55 @ 85 | 57 @ 98 | budget-binding |
| drd2_42 | 500 | — | — | — | unroutable (see below) |
| drd2_42_pruned | 174 | 8 | 8 @ 16 | 8 @ 18 | pool-exhausted |
| drd2_43 | 500 | 45 | 45 @ 88 | 15 @ 100 | redundant |
| drd2_43_pruned | 119 | 85 | 55 @ 95 | 70 @ 100 | budget-binding |
| drd2_44 | 101 | — | — | 17 @ 37 | pool-exhausted |
| drd2_44_pruned | 28 | 15 | 15 @ 24 | 16 @ 32 | pool-exhausted |
| clpp_42 | 339 | 125 | 75 @ 96 | 60 @ 100 | budget-binding |
| clpp_42_pruned | 211 | 125 | 75 @ 97 | 100 @ 100 | budget-binding |
| clpp_43 | 314 | 125 | 25 @ 53 | 70 @ 100 | budget-binding |
| clpp_43_pruned | 218 | 125 | 45 @ 95 | 93 @ 100 | budget-binding |
| clpp_44 | 368 | 150 | 50 @ 71 | 61 @ 100 | budget-binding |
| clpp_44_pruned | 233 | 150 | 70 @ 100 | 100 @ 100 | budget-binding |

`redundant` means the solver spent the whole budget but bought near-duplicates: on `drd2_43` it
selected 100 molecules that collapse to 15 distinct ones (mean pairwise similarity 0.48). That is
the diversity blind spot already measured in entry 056 (27% distinct vs the pool's own 41%),
appearing here in a sharper form.

**`s3gfn_drd2_seed42` cannot be routed at all**: 0 of 500 molecules routed after 5.03 h, and the
pruned pool managed 8 of 174. Reproduced standalone — the planner explores 580–696 nodes per
molecule and solves none, because 79% of that pool is amides of
(1-propylpyrrolidin-2-yl)methanamine, and that amine is not in our catalogue while the acid half is.
A stock-coverage result, not a pipeline failure, and not worth re-running.
