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

**Partial — the competitor's strongest configuration is now replicated, and the single run we had been
comparing against was its luckiest.** Across three independent runs it needs 264 ± 25 reactions to
deliver the 100-family library, where the one run we had reported 235. The comparison we describe as
our conservative one therefore moves from 1.79× to **2.02×** in our favour. The claim survives
replication and gets slightly stronger, which is the outcome worth having; the spread is about 10% of
the mean, so it should not be quoted more tightly than that.

The other selection procedure is being re-measured, for two reasons that are themselves the most
important findings here, and both share a shape: **a wrong number that looked like a result.** First,
a co-author changed what the measurement *means* while these jobs sat in the queue, so the replicates
quietly computed a different quantity than the run they were meant to be compared against, and the
numbers looked like a large seed effect rather than a bug (Method 6). Second, the optimizer's own
solver reports success when it has merely run out of time, so a third of the competitor's curve was
never actually solved (Method 7).

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

**Version-pinning artifacts** (Method 6 — these exist because the frontier script changed mid-flight)

- `/scratch/markymoo/rgfn_runs/pinned_frontier_head/` — a frozen copy of the **committed**
  `sparrow_select_frontier.py` (sha256 `f428aa59863694a9`), at the directory depth the script's
  `parents[3]` requires, with `validation` symlinked to the live repo. Deliberately outside the repo.
- `/scratch/markymoo/rgfn_runs/rerun_sb_pinned.sh` — re-runs only the SB frontier against that pinned
  copy, reusing the cached MultiAiZ routes.

**Results**

- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed4{3,4}_select_N500/` — SB, from
  the **pinned** script. These are the comparable ones.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed4{3,4}_greedy_N500/` — greedy,
  from the original chain; valid, because that code path is byte-identical across the two versions.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/s3gfn_seh_seed4{3,4}_select_N500_LAMBDADIV_not_comparable/`
  — the first, non-comparable SB run. **Kept on purpose:** it is a co-author's newer
  diversity-constrained metric applied to two fresh MultiAiZ route sets, so it is data they may want;
  it simply cannot be averaged with seed 42.
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

6. **Discovered the SB arm had measured a different quantity, and repaired it.** This is the entry's
   most transferable finding, so it is recorded in full.

   *The failure.* `sparrow_select_frontier.py` lives in a working tree shared by three agents. At
   12:37 on 2026-08-13 — **after** these jobs were queued and before they reached their frontier step
   — a co-author rewrote its SB measurement: `n_modes` (modes among the SELECTED set, via
   `count_modes`) became `n_modes_kept` (a pruned subset priced by a separate `_price_kept_set` solve),
   `mode_rate` was redefined from `n_modes/n_selected` to `len(kept)/n_selected`, and `lambda_div` /
   `clusters_touched` / `cost_kept_rxns` / `mean_pairwise_sim` were added. A SLURM job runs the script
   as of the moment it *starts*, so both replicates ran that version.

   *Why it was dangerous.* Exit 0, `milp_status Optimal` on every row, plausible numbers. Seed 43
   reported **149 modes at a 300-reaction budget** where seed 42 reports ~70. Read naively that is a
   2× seed effect that would have destroyed the headline. It is a definition change. Entry [059]
   Method 5 records "an infrastructure failure reported as a scientific limit"; this is the same
   pathology one level up — **a co-author's definition change reported as a seed effect.**

   *The repair, and how the pinned version was proven rather than assumed.* Extracted the committed
   version (`git show HEAD:…`, sha256 `f428aa59863694a9`, the file as of `8c2fe63`) into
   `/scratch/markymoo/rgfn_runs/pinned_frontier_head/`, mirroring `experiments/lsd_hubs/campaign/`
   because the script derives `REPO = Path(__file__).resolve().parents[3]`, with `validation`
   symlinked back to the live repo — outside the repo, so it carries none of the in-repo symlink
   hazard. Then **re-derived seed 42's own frontier with it** at the three budgets whose solves
   finish well under the 600 s MILP cap (R = 50/100/1000): identical to the archived CSV on every
   measured column (`used_rxns`, `n_selected`, `n_modes`, `mode_rate`, `rxn_per_selected`,
   `rxn_per_mode`, `total_reward`, `milp_status`), and `routed=478 (95.6%)` reproduced exactly.
   Only the sub-cap budgets are used for this check, because a time-limited solve's incumbent is not
   guaranteed reproducible across machine load. Note this run imported the **live**
   `metrics/diversity.py` and `sparrow_worker.py`, which the same co-author had also modified — so
   the exact reproduction is simultaneously evidence that those two edits are additive and do not
   move the measurement.

   *Scope of the re-run, minimised by diffing rather than guessing.* `_greedy_frontier` is
   **byte-identical** between the two versions (3,077 chars each), so the greedy results stood and
   were not re-run. MultiAiZ discovery does not depend on the frontier script at all, so the expensive
   artifacts (2.64 h / 2.55 h) were reused from cache. Only the SB arm was re-run
   (`/scratch/markymoo/rgfn_runs/rerun_sb_pinned.sh`, ~50 min/seed). The co-author's outputs were
   **preserved, not deleted** — renamed to `*_select_N500_LAMBDADIV_not_comparable/`, since they are
   their new metric applied to two fresh route sets.

7. **The optimizer's solver reports success when it has only run out of time.** Independently of
   Method 6, the same co-author added time-limit detection to `sparrow_worker.py`, having found that
   **CBC reports status "Optimal" for whatever incumbent it holds when a time limit stops it**, and
   PuLP's `sol_status` does not distinguish (verified on pulp 3.3.2: a 2 s-limited solve still says
   "Optimal Solution Found"). Their tell was a looser reaction budget returning a *worse* objective
   than a tighter one while both claimed optimality, which is impossible for true optima.

   Running the replicates through the corrected worker exposed the consequence: under the 600 s MILP
   cap that seed 42 used, **6 of seed 43's 10 SB rows are genuinely `TimeLimit`** (R = 150/300/400/
   500/600/800), and by the same token 6 of the *archived seed-42* rows carry an `Optimal` label they
   did not earn — R = 150/200/300 have `solve_s` of 602.57/602.65/602.79.

   **What this does and does not mean.** A truncated solve returns a feasible selection, so it is a
   real, achievable library. But SPARROW maximizes *reward*, and the mode count is measured afterwards
   — so a truncated incumbent's mode count is **not** a bound on the optimum's in either direction; it
   is a noisy sample. Measured noise is a few modes (R=400 gave 107 modes in one 600 s run and 104 in
   another). The correct description is therefore *non-reproducible*, not *biased in a known
   direction*, and an earlier draft of this entry and of the mailbox note said "understates SPARROW,
   biasing in our favour" — that was wrong and is corrected here.

   **Why it still had to be re-run.** Seed 42's 411-reaction readout interpolates between R=400
   (98 modes, `solve_s` 225.66) and R=500 (117, 205.40) — both terminated on their own well inside the
   cap, so the headline itself rests on true optima. Seed 43's readout falls between R=300 and R=400,
   **both truncated**, so its value would carry solver noise rather than being a measurement. Jobs
   73607/73608/73609 re-solve budgets 50,100,200,300,400,500,1000 for all three seeds at
   `--max-seconds 1800`. Seed 42 is included deliberately: a longer cap can only find an
   equal-or-better solution, so one cap across all three seeds is more comparable than mixing, and
   seed 42's readout rows staying at 98/117 is the check that this reasoning holds. **These jobs were
   still queued when this entry was last updated — Balam had 45 pending jobs — so the SB band is not
   yet reportable.**

   *Prophylactic.* `make_pipeline_headline.py::_assert_comparable_schema` now refuses any frontier CSV
   without an `n_modes` column, naming the version problem and the fix instead of averaging across two
   definitions (commit `578a87e`; tested against the preserved drifted file). Balam also forces
   `--gpus-per-node=1` on every job, so the CPU-only re-run had to request a GPU it does not use, and
   `-p debug` is `MaxJobsPU=1, MaxSubmitPU=1` — one debug job at a time, contrary to a note claiming
   the limit does not apply there, **and the limit is per UNIX user**, so a co-author's debug job
   blocks yours (their `rnv_seh42`/73606 did, twice).

   *A second prophylactic, added because the first guard exposed a worse bug of my own.* Dropping
   non-optimal rows leaves a seed's curve sparse, and `_mean_curve` averages seeds on their **shared**
   x-values — so a sparse replicate does not merely add noise, it deletes the dense seed's points too.
   With seed 43 reduced to {50,100,200,1000}, the mean curve lost every row in between and the
   100-mode readout interpolated across an 800-reaction gap to **467 instead of 411, moving the
   headline from 3.13× to 3.56× in our favour.** `_admissible_seeds` now excludes any seed whose
   100-mode bracket exceeds `MAX_READOUT_BRACKET` (250 reactions) from the average entirely, printing
   why, and `_seed_readouts` refuses such a seed's readout. With seed 43 excluded the headline returns
   to 3.13× at n=1, which is the honest state until the re-solves land.

### Results

**Reactions to deliver the 100-mode library.** τ = 0.5, reward gate 7.0, all arms SPARROW-priced.

| arm | per seed (42 / 43 / 44) | mean ± sd | n |
|---|---|---|---|
| ours, hub-batching | 131 / 123 / 119 | **124.3 ± 6.1** | 3 |
| competitor, MultiAiZ + diversity-aware greedy (**their best**) | 235 / 280 / 278 | **264.3 ± 25.4** | 3 |
| competitor, MultiAiZ + SPARROW-Batching | 411 / `[TODO]` / `[TODO]` | `[TODO]` | 1 → 3 running |

**Ratios.** The figure deliberately quotes our **seed-42** readout of 131, the worst of our three
seeds, so every ratio below is the conservative form:

| comparison | before (competitor n=1) | now (competitor n=3) |
|---|---|---|
| conservative — vs their strongest configuration | 1.79× | **2.02×** |
| same, against our 3-seed mean 124.3 instead of 131 | 1.89× | 2.13× |
| headline — vs MultiAiZ + SPARROW-Batching | 3.13× | `[TODO]` |

**Seed 42 was the competitor's best run of the three on its strongest configuration** (235 vs 278/280),
so the number the draft had been quoting was favourable to them. Replication moves the conservative
claim in our favour and puts an error bar on it. The sd is 9.6% of the mean, and n = 3.

**Per-seed pipeline agreement** — evidence the replicates are the same experiment, not three different
ones:

| quantity | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| training wall-clock | 4:14:42 | 4:09:19 | 4:24:19 |
| best pool score | 8.4839 | 8.4602 | `see summary` |
| `final/synth_ratio` | — | 0.875 | 0.880 |
| `final/avg_reward` | — | 0.990 | 1.005 |
| MultiAiZ discovery | 8,085 s (2.25 h) | 9,507 s (2.64 h) | 9,192 s (2.55 h) |
| pool routed by MultiAiZ | 478 / 500 (95.6%) | 454 / 500 (90.8%) | `[TODO]` |

Discovery ran longer for the replicates because both shared a node; it is wall-clock, not work.

Rebuilt-env fidelity check (Method 1), seed 43, 10 steps:

| quantity | this run | documented for `zincfrag_hb105` |
|---|---|---|
| `sampled_synth_ratio`, early steps | 1.6–6.3% | 4.7% on the GP-MolFormer prior |
| unique stereo-stripped blocks | 178,622 | ~178k |
| in-stock block score | 1.0 (5/5) | 1.0 required by the build assertion |

Pinned-version verification (Method 6), seed 42 re-derived at the sub-cap budgets:

| budget | archived (`n_modes`) | pinned re-run | every measured column identical |
|---|---|---|---|
| 50 | 16 | 16 | yes |
| 100 | 24 | 24 | yes |
| 1000 | 192 | 192 | yes |

Rebuilt-env fidelity check (Method 1), seed 43, 10 steps:

| quantity | this run | documented for `zincfrag_hb105` |
|---|---|---|
| `sampled_synth_ratio`, early steps | 1.6–6.3% | 4.7% on the GP-MolFormer prior |
| unique stereo-stripped blocks | 178,622 | ~178k |
| in-stock block score | 1.0 (5/5) | 1.0 required by the build assertion |
