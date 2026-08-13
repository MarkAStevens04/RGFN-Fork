# sEH — giving the competitor the same error bars we give ourselves

**Date:** 2026-08-13, ~12pm

**STATUS: STUB — jobs launched and running. Results sections marked `[TODO]`.**

## Question

Our headline comparison rests on a single run of the competing method. If we ran it again from a
different starting point, would it land in the same place?

## Context & Summary

Entry [056] is the comparison the paper leads with: to deliver the same 100-family library, the
competitor pipeline needs about three times more chemistry than ours. Entry [059] then closed the
sharper loophole — handing a batch optimizer *our own* molecules — and did it across three independent
runs, so that result carries a spread.

The headline itself does not. Our side of it has been run three times and lands within a couple of
reactions each time; the competitor's side is one training run followed by one route-planning run. A
reviewer is entitled to ask whether the gap we report is a property of the two methods or an accident
of the one run we happened to do, and right now we cannot answer. This is the last arm in the whole
benchmark without a replicate, and it is the arm the headline number comes from.

So we are running the competitor twice more, from two fresh starting points. Each repeat is the whole
pipeline, not a re-solve: the generator is retrained, its molecules are re-planned by the route
planner, and the batch selection is redone. That matters because the planner works on a *set* of
molecules at once — it looks for chemistry the molecules can share — so a molecule's route depends on
what else was planned alongside it. Reusing the first run's routes would understate the variation we
are trying to measure.

Both selection procedures are repeated for each new run: the optimizer choosing for itself, and the
stronger diversity-aware alternative that entry [056] added precisely because it could only make our
number worse.

## Answer

`[TODO — after jobs 73365 / 73367 land]`

## Relevance to our Publication

This closes the last "n=1" in the benchmark, and it closes it on the number the paper leads with. The
project's own figure plan (`docs/paper_planning/lsd-flow-iclr-evidence-handoff.md`, row F1) already
lists the missing competitor replicate as the one outstanding item on an otherwise
publication-ready headline figure. For **ICLR**, a headline ratio quoted from a single run of the
baseline is the kind of thing a reviewer can dismiss without engaging with the method at all; for the
**Digital Discovery / Nature Computational Science** version the same objection arrives as a request
for error bars on the main claim.

The result can only move against us or leave us unchanged, which is the reason to run it. If the
competitor's spread is small, the headline stands with a band on both sides. If it is large, we learn
that before a reviewer does, and the honest claim becomes a range rather than a point.

## Next Experiments

**Refining for publication**

- Decide, once the spread is known, whether the quoted headline moves to the 3-seed mean on both
  sides. Our own side already has three seeds (124 ± 6 reactions, SPARROW-priced); the figure
  currently quotes seed 42's 131 — the *worst* of the three — so the number in the draft is
  conservative by construction. Adopting means on both sides is a deliberate claim change.
- Give our own side a replicated *budget curve*, not just a replicated 100-mode point. Seeds 43/44
  exist only as reconciled endpoints, so panel C of the headline figure is still single-seed on our
  arm.

**Next steps in project**

- Repeat the same replication on the second surrogate target (DRD2), where our replicates already
  exist.
- Extend to the docking-based targets, which remain one seed per cell throughout entries [058]/[060].

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/campaign/build_s3gfn_pools.py` — **new**: builds the nested top-N candidate
  pool that route discovery is keyed to (dedup by SMILES, ties broken by first appearance so the
  output is a deterministic function of the input file). Its `--verify` mode reproduces all four
  seed-42 reference pools exactly — identical order, membership and scores — which is what licenses
  using it to build the replicate pools.
- `./experiments/lsd_hubs/campaign/submit_s3gfn_replicate_routes.sh` — **new**: everything downstream
  of one training run (pool → MultiAiZ discovery → both frontiers), with seed 42's exact budget
  ladder and greedy mode points pinned so the replicate curves are read at the same x-positions.
  Invokes `submit_multiaiz_discover.sh` as a *subroutine* rather than copying it, so discovery
  parameters cannot drift between seeds.
- `./experiments/lsd_hubs/campaign/submit_s3gfn_seh.sh` — reused unchanged; only `SEED` and `OUT_DIR`
  are overridden.
- `./experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh` — reused unchanged to **rebuild** the
  training-time synthesizability env, which had been lost (see Method 1).
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — unmodified here; run twice per seed
  (`--selection sparrow`, then `--selection greedy`).
- `./experiments/lsd_hubs/campaign/make_pipeline_headline.py` — **modified**: consumes the replicates
  when they appear (see Method 4) and gained the Logs/059 panel.

**Models**

- `external/s3gfn/data/envs/zincfrag_hb105/{building_block.smi,template.txt}` — the training-time
  synthesizability signal: ZINCFrag's 178,622 public fragments (stereo-stripped canonical, for
  exact-string indexing) plus the 105-template `hb.txt` set. **Rebuilt on 2026-08-13**; the choice of
  this env over the paper's Enamine-based one is documented in `build_s3gfn_retro_env.sh`.
- `data/models/aizynthfinder/` — the route planner's USPTO expansion/filter models and the plain-ZINC
  stock. Plain ZINC by design (decision 2026-08-04): S3-GFN is ZINC-native, so it is the catalogue a
  chemist would actually stock for it.

**Datasets**

- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/71007/` — the seed-42 training run
  from entry [056], the reference every replicate is compared against.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed4{3,4}/` — the replicate
  training runs (jobs 73364 / 73366).
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_seed4{3,4}_N500/` — each
  replicate's own pool and its own routes artifact. Separate per seed because MultiAiZ is set-based:
  the cache key is the POOL, never the molecule.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh_smoke/seed43_smoke/` — the
  10-step smoke that caught the missing env; kept as the evidence for Method 1.

**Results**

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed4{3,4}_select_N500/` — SB.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed4{3,4}_greedy_N500/` — greedy.
- `./experiments/lsd_hubs/campaign/results/paper_pipeline_headline/` — the figure the replicates feed,
  plus `panel_c_our_molecules.csv`.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/s3gfn_seh_s4{3,4}-{73364,73366}.{out,err}` — training
- `/scratch/markymoo/rgfn_runs/s3gfn_rep_routes_s4{3,4}-{73365,73367}.{out,err}` — routes + frontiers
- `/scratch/markymoo/rgfn_runs/s3gfn_smoke43-73361.err` — the silent failure of Method 1, kept
- `/scratch/markymoo/rgfn_runs/s3gfn_smoke43-73363.out` — the passing smoke after the rebuild

### Relevant Versions

Branch `Hub-Analysis`. This entry's code is committed:

```
5a866d9  Headline figure: add the SPARROW-Batching arms on our own molecules (Logs/059)
5784961  Headline figure: pick up the competitor's replicates automatically (F1's open item)
8e0e4ca  Competitor replicates: one script for everything downstream of an S3-GFN run
```

Note this repo is a **shared working tree with two other agents**; the three commits above stage
explicit paths only. Another agent's uncommitted `--min-clusters` work in
`sparrow_select_frontier.py` was left untouched and does not affect these runs (the flag is not
passed).

### Relevant Resources

**Sources**

- `[kim2026s3gfn]` — S3-GFN, the non-reaction baseline being replicated.
- `[fromer2024sparrow]` — SPARROW, used here in SELECTION mode (SPARROW-Batching).
- MultiAiZ / AiZynthFinder — the convergent route planner (entry [043] for the bring-up).

**Packages**

- `s3gfn` env — GP-MolFormer + soft-synthesizability training
- `aizynth` env — MultiAiZ route discovery
- `sparrow` env + PuLP/CBC — the MILP
- `rgfn` env — pool building, mode counting (`validation/lsdflow/metrics/diversity.py`, Morgan r=3 /
  2048, greedy sphere exclusion at Tanimoto ≤ 0.5, best-reward-first)

### Method

1. **Rebuilt the training-time synthesizability env, which was gone.**
   `external/s3gfn/data/envs/zincfrag_hb105/` did not exist — lost when the s3gfn clone was recreated
   on 2026-07-30. A 10-step smoke on `-p debug` (job 73361) exposed it, and exposed it in the worst
   possible form: SLURM reported **COMPLETED with exit code 0** while the job had died on the missing
   `template.txt`. That is the same silent-success mode entry [059] Method 5 records, in a new place.
   Rebuilt with `build_s3gfn_retro_env.sh` (re-downloads ZINCFrag; 200,000 → 178,622 unique
   stereo-stripped blocks; in-stock blocks score 1.0 as its own assertion requires).
   **Checked faithful, not merely present:** the re-smoke's `sampled_synth_ratio` came in at
   1.6–6.3%, matching this env's documented 4.7% starting point on the GP-MolFormer prior — so
   training begins from the same signal seed 42 did, rather than from a different env that happens to
   load. Re-smoke (job 73363) passed end to end: 11:25, no traceback, 100 candidates ingested.

2. **Verified the pool rule before using it.** `build_s3gfn_pools.py --verify` against all four
   seed-42 reference pools (N = 50/100/250/500): identical order, identical membership, identical
   scores. The seed-42 `candidates.csv` happens to be already distinct (2,000 rows → 2,000
   molecules), so dedup is a no-op there — which is exactly why it is written down rather than relied
   on, since the same rule is used for pools built from `records.csv` where it is not.

3. **Launched two chains** (`--dependency=afterok`, so a failed training cannot feed a frontier):

   ```
   seed 43:  73364  train (6 h)  ->  73365  pool + MultiAiZ + SB frontier + greedy frontier (5 h)
   seed 44:  73366  train (6 h)  ->  73367  same
   ```

   Wall times sized from seed 42's measured 4:14:42 training and 2.25 h discovery. Budgets
   `50,100,150,200,300,400,500,600,800,1000` and mode points `25,50,75,100,125,150` are seed 42's own
   ladders, read out of its `*_frontier_summary.json` rather than left to script defaults.

4. **Wired the figure to consume the result before it existed**, and dry-ran that path against
   synthetic stand-ins rather than waiting hours to discover a bug in it. The dry run found one: the
   two competitor arms are swept along **different axes** — SB sweeps the reaction budget
   (`used_rxns == budget`) while the greedy sweeps *mode points* and asks SPARROW what they cost.
   Averaging the greedy arm over reactions intersects on `used_rxns`, which no two seeds share,
   silently producing an empty curve and a missing readout. Fixed to average over whichever axis the
   experiment held fixed; stand-ins deleted afterwards.

5. **Recorded in the shared-agent mailbox** (`/home/markymoo/agent_comms/`): claim
   `claims/s3gfn_seh_replicates`, status in `msg/balam-b2.md`.

### Results

`[TODO — jobs running. To fill in: per-seed reactions to reach 100 modes for both selectors, the
3-seed mean ± sd on the competitor side, and whether the headline ratio (currently 3.13× against a
single competitor run) and the conservative ratio (1.79×) survive with bands on both sides.]`

**Already established, for comparison when the replicates land:**

| arm | reactions for 100 modes | n |
|---|---|---|
| ours, hub-batching (SPARROW-priced) | 119 / 123 / 131 → **124.3 ± 6.1** | 3 |
| competitor, MultiAiZ + SPARROW-Batching | 411 | 1 → **3 pending** |
| competitor, MultiAiZ + diversity-aware greedy (their best) | 235 | 1 → **3 pending** |

Rebuilt-env fidelity check (Method 1), seed 43, 10 steps:

| quantity | this run | documented for `zincfrag_hb105` |
|---|---|---|
| `sampled_synth_ratio`, early steps | 1.6–6.3% | 4.7% on the GP-MolFormer prior |
| unique stereo-stripped blocks | 178,622 | ~178k |
| in-stock block score | 1.0 (5/5) | 1.0 required by the build assertion |
