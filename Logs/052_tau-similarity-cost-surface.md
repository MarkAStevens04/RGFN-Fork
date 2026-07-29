# SCENT / sEH — does the batching advantage survive everywhere on both quality knobs, at a fixed bench budget?

**Date:** 2026-07-29, ~10am

## Question

If we give a chemist a fixed budget of 300 reactions, does building shared scaffolds and diversifying
them ("hub batching") still deliver more distinct, high-scoring molecules than simply making the best
molecules one at a time — no matter how strictly we define "high-scoring" and "distinct"?

## Context & Summary

Our headline cost number is **reactions per distinct useful molecule**, and it depends on two knobs we
set by hand: the **quality bar** a molecule must clear to count, and the **similarity cutoff** below
which two molecules count as the same thing. Every result so far walks one knob with the other pinned —
entry `035` and the gate curves walk the quality bar; the Pareto sweeps walk the similarity cutoff — so a
reader has to take on faith that the advantage doesn't evaporate in some corner we never looked at. Our
publication notes make this the single largest threat to acceptance: a ratio metric that the authors
also happen to win on invites the suspicion that the knobs were chosen to win.

Entry `051` cleared the ground on the pool side: raising the quality bar barely changes how diverse the
molecule pools themselves are (under 5% up to a bar of 7), so the bar is a legitimate control rather
than a hidden diversity dial. This entry does the cost side. We fix the budget at 300 reactions and run
both strategies over a full **17 × 13 grid** — quality bars from 4 to 8 in steps of 0.25, similarity
cutoffs from 0.3 to 0.9 in steps of 0.05 — giving 221 measured cells per strategy, and report how many
reactions each spends per distinct molecule it delivers. Cells where a strategy runs out of library
before spending the budget are flagged and hatched rather than counted as wins, since a shorter run
isn't comparable to a full one.

## Answer

**Hub batching is cheaper in every single one of the 206 comparable cells — 1.6× to 3.0× — so the
advantage is not an artifact of where we set either knob.** There is no corner of the grid where the
naive strategy wins, and no crossover inside the measured range. The advantage is largest when either
bar is loose (3.0× at the lowest quality bar) and narrows as either is tightened (1.6× at the highest),
which is exactly the mechanism working as designed: batching buys reaction economy by building molecules
that share a scaffold, so it gives up ground as you demand more structural distinctness.

**The two strategies fail for different reasons, and that turns out to be the most informative part.**
The naive strategy's cost is *completely* insensitive to the quality bar — identical to four decimal
places across bars 4 through 7.75 — and moves only with the distinctness cutoff. That makes sense once
stated: it makes the best-scoring molecules first, and 300 reactions never gets it past the very top of
the pool, so a bar below that top slice never binds. Hub batching, in contrast, responds to both knobs,
and in the tightest corner (top ~7% of cells) it runs *out of library* before it can spend the budget:
13 of the 15 unusable cells are hub batching running dry, because it may only draw on the children of
200 pre-selected scaffolds. Those cells are flagged, not counted as wins.

**The place where hub batching starts to struggle is the same place entry `051` found the molecule pools
collapsing.** Its cost climbs above a bar of ~7 and the library exhausts past ~7.5 — and independently,
`051` measured the pools losing 40%+ of their distinct molecules over exactly that range. Two different
measurements agree on where the useful operating range ends, which is a stronger statement than either
alone.

## Relevance to our Publication

Our publication notes call the metric defense the single largest threat to acceptance: we propose
reactions/mode *and* win on it, and reviewers are trained to suspect exactly that shape. The answer they
ask for is that the result survives the knobs being moved — and this is that answer in one figure, 206
cells, both knobs at once, rather than two separate one-dimensional sweeps a reader has to mentally
compose. It directly supports the "rank invariance" claim §2.3 asks for and gives §2.5's τ axis its
companion, and because it reports the narrowing honestly it lets us make the §3 framing argument in our
own words — *batching around shared intermediates trades structural diversity for reaction economy, and
here is the regime where that trade pays* — instead of hoping nobody asks why the margin shrinks.

It also hands us a defensible operating range for the paper: the comparison is like-with-like for bars
≤7 and cutoffs ≥0.4, and outside that the limiting factor is the 200-hub library size, not the method.
Stating that boundary ourselves is worth more than an unblemished plot.

## Next Experiments

**Refining for publication**

- **Test whether the pool-limited corner is a library-size artifact.** Hub batching runs dry at strict
  cutoffs because it only has 200 scaffolds' children to draw on. Entry `031` already showed scaling
  50→200 hubs changes the picture; re-running the strict-cutoff column with more hubs would show whether
  that corner is a property of the method or just of how much we enumerated.
- **Repeat the surface on a second generator/target.** Entry `050` has sampled + enumerated pools for
  RxnFlow and for DRD2, so the same driver answers whether the shape (naive strategy flat in the quality
  bar, batching responsive to both knobs) is general or SCENT-specific.
- **Add the numerator guard.** This surface, like every reactions/mode result we have, says nothing
  about whether the cheaper libraries are made of *worse* molecules. §2.4 asks for reward and
  synthetic-depth distributions alongside the ratio; the per-cell data already carries best/median
  reward, so this is a plotting job, not a new run.

**Next steps in project**

- **Quote the operating range wherever the ratio appears**, and mark the pool-limited region on the
  existing gate curves and Pareto panels the same way this figure does.
- **Fold the two-knob view into how we report new targets.** For the docking rewards (6TD3/ClpP) the
  bar was calibrated separately (entry `045`) and neither knob has been swept; running this surface
  there before quoting a docking cost claim would avoid re-litigating the same question per target.

# Re-creation

## Relevant Files

Repo root `./`; scratch paths absolute.

**Scripts**

- `./experiments/lsd_hubs/campaign/tau_similarity_surface.py` — the driver. Loads the candidate pool,
  the enumerated hubs and the cost table **once**, then runs both selection strategies on every
  (τ, cutoff) cell in-process. Reuses `run_campaign.py`'s own `_load_candidates` /
  `_load_enumerated_hubs` / `build_strategy`, so a cell is by construction the same computation as a
  `run_campaign.py` invocation at that (τ, cutoff) — no second implementation of the cost model.
  `--plot-only` redraws the figure from the committed CSV.
- `./experiments/lsd_hubs/campaign/submit_tau_similarity_surface.sh` — SLURM wrapper (`debug`
  partition, CPU-only). Writes to `$SCRATCH` because `$HOME` is read-only on Balam compute nodes; the
  four small artifacts are copied into the repo afterwards.
- `./glue/samplers/lsdflow/mode_select.py` — the production mode selector each cell runs. Its
  per-candidate diversity test was switched to one `BulkTanimotoSimilarity` call for this experiment
  (3.5–6.1× faster on a best-candidate run, accepted curves byte-identical; verified in entry `051`).
  Without it the grid would not fit in a debug-partition slot.

**Datasets** — the SCENT × sEH campaign anchor (checkpoint
`/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt`):

- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/{records.csv,compositions.json}` — the 30k
  sampled trajectories (26,069 unique molecules) = the pool **best-candidate** selects from, plus each
  molecule's promoted-fragment composition, which is what lets the cost model charge shared parts once.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json` — exhaustive
  one-reaction children of the 200 ranked hubs = the library **hub-batching** selects from. Rewards are
  already computed in this file, so the grid re-scores rather than re-generating (no GPU, no oracle).
- `.../fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` — the frozen
  promoted-fragment snapshot **with recipes**, required by the count-once cost model to charge each
  distinct fragment's own build exactly once (closure under nesting). Same snapshot the committed
  campaign scripts default to, so these numbers sit on the published cost model.

**Results**

- `./experiments/lsd_hubs/campaign/results/scent_seh_surface/surface.csv` — one row per
  (strategy, τ, cutoff): modes reached, reactions used, reactions/mode, reward-generation calls,
  stop reason, `pool_limited`.
- `.../surface.json` — full run config (inputs, grid, budget, child policy, pre-select-K) + a summary:
  comparable-vs-pool-limited cell counts, how many cells hub-batching wins, the extreme cells, and the
  value at the headline operating point.
- `.../surface.{png,pdf}` — the three panels (each strategy's cost surface on a shared scale, then the
  advantage ratio).

**Job Logs**

- `/scratch/markymoo/rgfn_runs/tau_sim_surface-71896.{out,err}` — the grid run.
- `/scratch/markymoo/rgfn_runs/tau_sim_surface-71885.{out,err}` — the first attempt, dead in 15 s with
  `PermissionError` writing into the repo (read-only `$HOME` on compute); kept as the trace of why the
  submit script writes to `$SCRATCH`.

## Relevant Versions

Branch `Hub-Analysis`, last commit `1e82730`. Not yet committed: the driver + submit script + results,
this log, entry `051` and its sub-dir, the `validation/lsdflow/metrics/diversity.py` additions and the
`glue/samplers/lsdflow/mode_select.py` speedup.

`[TODO — add commit hash after pushing]`

## Relevant Resources

**Sources**

- `docs/paper_planning/lsd-flow-publication-strategy.md` §2.2–2.5 — the threat model this addresses
  (reactions/mode is a ratio, gameable from the denominator side) and the instruction to present the
  similarity sweep and the τ sweep as one framework rather than two bolted-on defenses. §3 adds the
  framing rule: report the crossover, don't stop the sweep before it.
- Logs/051 — the pool-side prerequisite: τ barely moves the pools' own diversity below ~7.
- Logs/033 — the count-once cost model (each synthesis step charged exactly once, fragments included),
  which is what makes a cell's reactions/mode fair to both strategies.
- Logs/037 — `free_frag` + pre-select-K, the within-hub child policy used here.
- Logs/035, `matrix16/gate_curve.py` — the one-dimensional slices this generalizes.

**Packages**

- RDKit — ECFP + Tanimoto inside `glue/samplers/lsdflow/mode_select.py`.
- matplotlib — the surface figure (`tau_similarity_surface.py::plot`).
- No torch/dgl/gin, no GPU: the grid re-scores cached rewards.

## Method

1. **Sized the grid against a measured cell cost.** One cell (both strategies, 300-reaction budget) was
   timed on the login node at ~4.5 s after the `mode_select` speedup (before it, `best_candidate` alone
   took 28–44 s). 221 cells ≈ 17–20 min, past the 15-minute interactive guideline → SLURM `debug`.

2. **Ran the grid** (job 71896, `debug` partition, CPU-only):

   ```bash
   sbatch experiments/lsd_hubs/campaign/submit_tau_similarity_surface.sh
   # = tau_similarity_surface.py --taus 4:8:0.25 --similarities 0.3:0.9:0.05 \
   #     --budget-reactions 300 --child-policy free_frag --prebuild-k 20 \
   #     --analysis-dir .../scent_seh_70189 --enum-children .../campaign_enum_seh_70363/enum_children.json \
   #     --snapshot .../fragments_4000.json --tag scent_seh_surface --out-dir $SCRATCH/...
   ```

   17 quality bars × 13 similarity cutoffs × 2 strategies = 442 selection runs. The pre-select-K
   fragment ranking depends on τ (not on the cutoff), so it is computed once per τ row and reused across
   that row's 13 cells.

3. **Read each cell at the budget, not from the run totals.** A run stopped by a reaction budget
   overshoots by the mode that crosses the line, so the readout is the last accepted point with
   `cum_reactions <= 300`: `reactions/mode = reactions_used / modes_reached`. Cells whose stop reason
   isn't the budget (the strategy ran out of library first) are marked `pool_limited` and excluded from
   the win/loss tallies, because a shorter run is not comparable to a full one.

4. **Copied the four small artifacts** from `$SCRATCH` into
   `experiments/lsd_hubs/campaign/results/scent_seh_surface/` for committing.

## Results

Grid: 17 reward bars (τ = 4.0 → 8.0, step 0.25) × 13 similarity cutoffs (0.30 → 0.90, step 0.05) = 221
cells × 2 strategies = **442 selection runs in 639 s** (10.6 min) on one `debug` node, CPU only. Pools
loaded once (4.7 s): 26,069 candidates, 200 enumerated hubs, 1,600 promoted fragments with recipes.
`child_policy=free_frag`, `prebuild_k=20`, `rank_by=build_score` — the current headline configuration.

**Headline tally:** 206/221 cells comparable (15 pool-limited); **hub batching cheaper in 206/206**.

| cell | hub batching | best candidate | advantage | modes at 300 rxn (hub / best) |
|---|---|---|---|---|
| headline operating point (τ=7.0, cutoff 0.5) | 1.2245 | 3.1263 | **2.55×** | 245 / 95 |
| best cell (τ=4.0, cutoff 0.40) | 1.1494 | 3.4767 | **3.02×** | 261 / 86 |
| worst comparable cell (τ=8.0, cutoff 0.55) | 1.9351 | 3.0825 | **1.59×** | 154 / 97 |

**Advantage along the diversity cutoff** (mean over that column's comparable τ rows) — peaks mid-range
and falls off at both ends:

| cutoff | 0.90 | 0.80 | 0.70 | 0.60 | 0.50 | 0.45 | 0.40 | 0.35 | 0.30 |
|---|---|---|---|---|---|---|---|---|---|
| mean advantage | 2.29× | 2.26× | 2.43× | 2.49× | 2.64× | **2.70×** | 2.69× | 2.51× | 2.20× |
| comparable rows | 17 | 17 | 17 | 17 | 16 | 16 | 15 | 13 | 10 |

**Advantage along the reward bar** (mean over that row's comparable cutoffs) — monotone decline, and the
hub-batching cost itself starts climbing in the same range:

| τ | 4.0 | 5.0 | 6.0 | 6.5 | 7.0 | 7.25 | 7.5 | 7.75 | 8.0 |
|---|---|---|---|---|---|---|---|---|---|
| mean advantage | 2.61× | 2.58× | 2.52× | 2.50× | 2.38× | 2.32× | 2.14× | 1.98× | **1.88×** |
| hub r/m range | 1.08–1.58 | 1.08–1.65 | 1.08–1.85 | 1.08–1.64 | 1.09–1.93 | 1.09–1.73 | 1.13–2.11 | 1.17–2.04 | 1.23–1.94 |
| comparable cutoffs | 13 | 13 | 13 | 12 | 12 | 11 | 11 | 10 | 8 |

**Best-candidate is exactly invariant in the reward bar.** Holding the cutoff fixed, its reactions/mode
is identical across every τ from 4.0 to 7.75 (spread **0.00** at cutoffs 0.9 / 0.7 / 0.5 / 0.3), and only
τ=8.0 moves it. It varies only with the cutoff: 2.48 (0.90) → 3.70 (0.30), spread 1.22. Mechanism: it
consumes the pool best-reward-first, and a 300-reaction budget never reaches below the top reward slice,
so any bar under that slice cannot bind. This is the exact-limit version of the "best-candidate is nearly
gate-invariant" observation in entries `035`/`050`.

**The 15 pool-limited cells** are one contiguous corner — strict cutoff × high bar — and are
overwhelmingly hub batching running out of library (13 of 15; best-candidate joins it in only 3 cells,
all at τ ≥ 7.75 with cutoff ≤ 0.35):

| τ | pool-limited cutoffs |
|---|---|
| 6.5, 6.75, 7.0 | 0.30 |
| 7.25, 7.5 | 0.30, 0.35 |
| 7.75 | 0.30 (both strategies), 0.35, 0.40 |
| 8.0 | 0.30 (both), 0.35 (both), 0.40, 0.45, 0.50 |

**Cross-check against the published point.** At the operating point this surface gives hub 1.2245 vs
best 3.1263 reactions/mode. That is the *fixed-300-reaction* readout, so it is not the same number as the
committed `matrix16/results/scent_seh/summary.json` (1.303 vs 3.383), which is measured at a fixed
**300-mode** target; the two agree in ratio (2.55× vs 2.60×) and in ordering, as they should for the same
run under two different budget conventions.
