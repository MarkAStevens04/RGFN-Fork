# sEH + DRD2 + ClpP — what the reaction budget measures once trivially-makeable molecules are excluded

**Date:** 2026-09-04, ~8pm

## Question

When a chemist demands molecules that cannot simply be bought and stuck together in one step, does
our reaction-based method deliver more distinct high-scoring molecules per reaction than the standard
generate-then-plan-then-select pipeline?

## Context & Summary

**Context.** Entry [075] found that the competitor pipeline reaches 100 distinct molecules for 100
reactions on sEH — and then explained why: 98 of the 100 molecules it picked are a *single* reaction
away from something you can buy out of a 17.4-million-compound catalogue. One reaction per molecule
is the cheapest anything can possibly be, so the competitor had not out-planned anyone; it had hit
the floor of the measurement. A number that cannot go lower also cannot tell two methods apart. That
left an obvious question the headline can no longer answer, and it is the one a medicinal chemist
would actually ask: what happens when the deliverable has to be a molecule you must *build*?

**Summary.** We added one constraint — every delivered molecule must take at least *N* reactions to
make from purchasable material — and swept *N* from 1 to 5 on both sides of the comparison, holding
everything else fixed. On the competitor's side this re-runs its solver over a filtered pool of the
same pre-planned routes; on ours it filters the enumerated candidates before selection. Because
shrinking a pool is itself a handicap, we also ran a control that shrinks the pool by the same amount
*without* removing shallow molecules, to check which of the two was doing the work. Separately we
started a "catalogue ladder": the same molecules routed against progressively smaller purchasable
catalogues, to measure how much of the competitor's advantage is catalogue access rather than
chemistry. One earlier result is retracted here.

## Answer

**Forbidding one-step molecules changes the ordering, and it does so for a legible reason: when the
competitor cannot buy its core it pays for one every time, while we pay once and reuse it.** Across
three training runs on sEH our method holds 94-96 molecules at a budget of 100 reactions when
one-step molecules are allowed and 93-94 when they are not, whereas the competitor falls from 98-100
to 72-86. The entry originally reported a third act that is not in our favour — demand molecules
five reactions deep and our method delivers nothing at all, on every seed. **THAT WAS AN ARTIFACT OF
THE AXIS AND IS RETRACTED** (2026-09-06, job 75749): "depth" was counted from our own 418 blocks on
our side and from ZINC's 17.4M compounds on the competitor's. Re-measured with BOTH sides counted
from ZINC, the same hub-batching arm delivers **73-86 molecules at depth >= 5**, and the advantage
GROWS monotonically with depth instead of collapsing — 1.28x at >= 2 rising to 1.92-2.26x at >= 5.
A molecule four reactions from our blocks is often further from ZINC, because ZINC does not stock our
intermediates; counting from our own library made our own molecules look shallow and emptied the pool
exactly where we in fact win most. The mismatched-axis figure would have shown us collapsing in our
best regime. See "The ZINC-axis re-measurement" under Results.

The control matters as much as the result. Halving the competitor's pool costs it only 1-6% of its
molecules, while removing the shallow ones from a pool of *identical size* costs 14-27% — so the
effect is genuinely about how deep the molecules are, not about giving the solver less to choose
from. That was the failure mode we most needed to rule out, and it is ruled out.

**One caveat is load-bearing and no figure should omit it:** "depth" is not the same quantity on the
two sides. Ours counts reactions from our own 418-block library; the competitor's counts reactions
from a 17.4-million-compound catalogue that contains most of our blocks and far more. The same
molecule therefore scores *shallower* on their axis, so the constraint bites them harder than us at
every setting. The per-method curves are meaningful; a crossing point read off the two is not, and
must not be quoted until both are measured against the same catalogue.

**And one earlier number from this session is withdrawn.** A first pass at this experiment used an
older, superseded set of candidate pools and reported a much larger advantage. It was measuring the
pool, not the constraint. The numbers here supersede it.

## Relevance to our Publication

The workshop/ICLR submission rests on out-performing the standard pipeline on reactions per distinct
molecule, and entry [075] showed that on two of three targets the standard pipeline now matches or
beats us — by finding molecules so simple that the metric bottoms out. A reviewer who notices that
will ask whether the benchmark measures library economics or merely molecular simplicity. This entry
supplies the answer we can actually defend: state the regime. Where the deliverable is trivially
purchasable-plus-one-step, buy it — the competitor wins and we say so; where it must be built, our
economy degrades more slowly; and where it must be built very deep, *we cannot compete at all*
because of our own reaction cap. Reporting the third regime unprompted is worth more than the second,
because it is the limitation a reviewer would otherwise find. The pool-size control and the retraction
are both here for the same reason: the result only means something if the obvious alternative
explanations were tested and named.

## Next Experiments

**Refining for publication.**
- **Put both sides on the same catalogue before drawing a two-line figure.** Routing our own
  delivered molecules against the same 17.4-million-compound catalogue the competitor uses would make
  the depth axis mean one thing. Measured throughput on this session's jobs is 11.1 s per molecule on
  one core, the selection is iterative (route the chosen set, drop what fails, route the
  replacements), so ~1,000-2,000 molecules per training run is the real requirement rather than the
  whole enumerated pool — under an hour of compute for all three runs if parallelised across the 32
  cores a debug node allocates. Until then, present the two curves separately.
- **Finish the catalogue ladder** (in flight, below). It measures the same asymmetry from the other
  direction and either corroborates the depth result or undermines it.
- **Re-run everything at the current hit thresholds.** Every number here uses the superseded bars; a
  publishable version needs the planned retraining first, then this experiment on top.

**Next steps in project.**
- Extend the depth constraint to DRD2 and ClpP on our side (only the competitor's side exists for
  those), so the regime claim rests on three systems rather than one.
- ~~Decide whether the reaction cap that empties our pool at depth 5 is worth relaxing~~ **moot**: the
  pool never empties once depth is measured from a shared catalogue. The reaction cap of 4 bounds
  depth in OUR units, not in the competitor's.
- Replicate the ZINC-axis measurement on seeds 43/44 (queued). It is currently one seed, and it is
  now the headline figure rather than a supporting one.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**
- `./experiments/lsd_hubs/campaign/depth_pruned_frontier.py` — **new.** Filters a competitor pool to
  molecules whose cheapest cached route needs ≥K reactions, writes a filtered `pool_scores.csv` into a
  **new** directory, and calls `sparrow_select_frontier.py` unchanged. Emits `funnel.json` recording
  every stage from sampling to selection. Also carries the `--size-match` control arm (keep N routed
  molecules chosen *without* regard to depth). It re-solves rather than trims, because trimming a
  solved SPARROW selection is not exact (CLAUDE.md) and would return a lower bound that flatters us.
  Refuses to write into the pool directory, and aborts on a gate mismatch between the pool and the
  requested gate.
- `./experiments/lsd_hubs/campaign/run_campaign.py` — **modified.** Added `--min-synth-depth`
  (default 0 = off; every existing result bit-for-bit unchanged). Filters BOTH strategy pools on the
  same quantity: best-candidate on `Candidate.num_reactions` (already SCENT's fully-nested count),
  hub-batching on `hub.depth + 1 + shared_build_cost(child.added_promoted)`. Hubs left with no
  surviving child are dropped. `glue/` is untouched — the existing `mode_selector_factory` seam cannot
  carry this because `ModeSelector.accept(smiles, reward)` never sees depth.
- `./experiments/lsd_hubs/campaign/aiz_stock_ladder.py` — **new.** Routes one molecule list against
  several purchasable catalogues at an identical search budget; per-molecule solve rate + cheapest
  route length. Plain AiZynthFinder, not MultiAiZ: solvability is a per-molecule question and answering
  it per molecule is what lets a rung fit a 2 h debug slot. Stock-major order with per-line flushing,
  so a walltime kill leaves complete rates for the early rungs; resumes by skipping
  `(stock, molecule)` pairs already in the JSONL.
- `./experiments/lsd_hubs/campaign/submit_aiz_ladder.sh` — **new.** One cell per job, `debug`
  partition, 2 h. `--gpus-per-node=1` on a CPU job is deliberate (Balam refuses CPU-only jobs at
  submit; both existing AiZynth/MultiAiZ submit scripts do the same).
- `./experiments/lsd_hubs/campaign/sparrow_select_frontier.py` — unchanged; the source of every
  competitor number here.

**Models / catalogues**
- `data/models/aizynthfinder/zinc_stock.hdf5` — 17,422,831 InChIKeys. The competitor's purchasable
  catalogue, and rung 1 of the ladder. Never modified; each new rung is a separate file and key.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/rgfn_smalllib_stock_flat.hdf5` — the reaction-GFN's
  418 blocks as 456 stereo+flat InChIKeys (built for entry [047]). The catalogue our own cost model
  treats as free, and the smallest substantive rung.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/ladder/zincfrag_stock.hdf5` — 178,598 keys from
  ZINCFrag, the public fragment catalogue S3-GFN's synthesizability signal was trained against.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/ladder/zinc_rand{418,178k}_stock.hdf5` — random draws
  from ZINC matched key-for-key to the two curated rungs. **These are the controls that make the
  ladder interpretable:** a small catalogue can hurt because it is small or because it is the wrong
  chemistry, and the gap between a curated rung and its random twin separates the two.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh_5k/seed{42,43,44}/additional_fragments/fragments_4000.json`
  — the SCENT promoted-fragment snapshots supplying nested build costs. **These carry
  `smiles_to_min_num_reactions` but NOT `smiles_to_route`**, so the cost table reports
  `recipes=False` and falls back to a non-nested unit cost. That is the same fallback every existing
  number for these cells was computed under, so it is consistent rather than a new error — but the
  depth figures inherit the approximation.

**Datasets**
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_{seh_seed42,seh_seed44,drd2_seed42}_stage2_pruned_N500/`
  — the competitor pools. **Stage 2** (`upsample_to_modes.py`, 2026-08-29): sampled until the pool
  holds 500 τ-distinct modes, at the current 5%-FPR gates. These supersede the `_big` pools
  (`s3gfn_sample_more.py`, 2026-08-28) which stopped at 500 molecules *above the gate* and reached
  only 209-270 modes, at the older bars.
- `/scratch/markymoo/rgfn_runs/lsdflow/matrix16{,_seed43,_seed44}/scent_seh/enum/enum_children.json`
  — our side's enumerated candidates (644,759 / 702,574 / 791,503 children over 200 hubs each). The
  expensive part; the depth sweep re-reads it rather than re-enumerating.

**Results**
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/*_depth{1..5}_select/` — competitor, one dir
  per (cell, depth), each with `funnel.json` + `select_frontier.csv`.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/*_size{226,203}_{topn1,random1,random2}_select/`
  — the size-matched control arms.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/hb_depth_seed{42,43,44}/d{1..5}/` — our side.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results/*_big_pruned*_depth2_select/` — the
  **retracted** first pass. Kept deliberately: the numbers exist on disk and would otherwise look
  quotable. Every `funnel.json` carries a `gate_provenance` field saying so.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/ladder/*_ladder.jsonl` — the catalogue ladder,
  in flight.

**Job Logs** — `/scratch/markymoo/rgfn_runs/aiz_ladder-75660.{out,err}` (ladder cell 1). Everything
else ran on the login node.

## Relevant Versions

Branch `Hub-Analysis`, at `6b9b92b` ("run_campaign: dump the accepted SMILES beside the summary").

**Not yet committed.** Files to commit:
- `experiments/lsd_hubs/campaign/depth_pruned_frontier.py` (new)
- `experiments/lsd_hubs/campaign/aiz_stock_ladder.py` (new)
- `experiments/lsd_hubs/campaign/submit_aiz_ladder.sh` (new)
- `experiments/lsd_hubs/campaign/run_campaign.py` (modified — `--min-synth-depth`)
- this log + the `docs/RESEARCH_CONTEXT.md` index row

`[TODO — add commit hash after pushing]`

## Relevant Resources

**Sources**
- Entry [075] — the one-step finding this entry follows from, and the source of the competitor's
  unfiltered numbers.
- Entry [067] — the naive/pruned pool design; its own headline is already retracted for the same
  class of error (a superseded budget), which is why the retraction below is stated rather than
  quietly dropped.
- Entry [047] — the stock-mismatch result and `build_block_stock.py`, the template for every ladder
  rung.
- Entry [048] — Exp C, MultiAiZ + ZINC pricing of our own molecules (1.85 rxn/mode); the existing
  half of the "put both sides on one catalogue" fix.
- Entry [033] — the count-once cost model whose `shallow_couplings` term the depth computation
  inverts.
- SPARROW `[fromer2024sparrow]`; MultiAiZ `[ianez2026multiaiz]`; AiZynthFinder `[genheden2020aizynthfinder]`.

**Packages**
- `aizynthfinder` 4.4.1 (own `aizynth` env) — the ladder and all route planning.
- `pulp` / CBC (own `sparrow` env) — the selection MILP, via `sparrow_select_frontier.py`.
- `rdkit` (`rgfn` env) — canonicalisation, InChIKeys, Morgan fingerprints.

## Method

1. **Competitor depth sweep.** Per cell and K ∈ {1..5}:
   `python experiments/lsd_hubs/campaign/depth_pruned_frontier.py --pool-dir <pool> --out-dir <res>_depth<K>_select --gate <g> --higher-is-better <b> --min-steps <K> --budgets 100 --max-seconds 600`
2. **Size-matched control.** Same script, `--size-match <N> --size-match-mode {topn,random} --size-match-seed {1,2}`, with N set to the depth≥2 pool size for that cell (226 for sEH s42, 203 for s44).
3. **Our depth sweep.** Per seed and K ∈ {1..5}:
   `python experiments/lsd_hubs/campaign/run_campaign.py --analysis-dir <sample> --enum-children <enum> --snapshot <fragments_4000.json> --reward-threshold 5.68 --higher-is-better true --similarity 0.5 --budget-reactions 100 --budget-modes 300 --child-policy free_frag --min-synth-depth <K> --tag hb_d<K> --out-dir <res>/d<K>`
4. **Ladder stocks.** Built ZINCFrag + two random-ZINC twins into
   `/scratch/.../lsdflow_sparrow/ladder/`, unioned with the existing `zinc` and `rgfnlib` keys into
   one `config_ladder.yml`. 100 molecules sampled per cell with a fixed seed — a *random* draw, not
   the top-N, because the top of the reward ranking is where trivially-makeable molecules concentrate
   ([075]).
5. **Ladder run.** `CELL=<cell> sbatch experiments/lsd_hubs/campaign/submit_aiz_ladder.sh`, rungs
   `zinc,rgfnlib,zincfrag,zinc_rand418`, budget iter=100 / time=60 / depth=6 (entry [047]'s
   production budget, so these rates sit beside its 48.7% → 73.8% ladder).

## Results

### The depth front — sEH, both sides at gate 5.68, budget 100 reactions

Distinct molecules delivered. Ours is three training runs; the competitor is two (only two sEH cells
have stage-2 pools with routes).

| min depth | ours s42 | ours s43 | ours s44 | competitor s42 | competitor s44 |
|---|---|---|---|---|---|
| ≥1 | 96 | 96 | 95 | 98 | 100 |
| ≥2 | 94 | 93 | 94 | 72 | 86 |
| ≥3 | 71 | 75 | 75 | 60 ⚠ | 77 |
| ≥4 | 58 | 59 | 54 | 53 | 53 |
| ≥5 | **0** | **0** | **0** | 38 | 24 |

⚠ competitor s42 at ≥3 returned `TimeLimit` — a lower bound on the competitor, i.e. it flatters us.
Every other cell returned `Optimal`, not time-capped.

Reactions per delivered molecule over the same sweep (ours s42 / competitor s42):
1.07 / 1.00 → 1.08 / 1.39 → 1.31 / 1.67 → 1.48 / 1.89 → — / 2.63.

Candidate pool at each rung — ours (enumerated children, s42): 644,759 / 643,441 / 580,612 /
449,972 / **0**. Competitor (routed molecules, s42): 351 / 226 / 162 / 116 / 81.

**Best-candidate is an internal control and it is flat:** 27-28 molecules at ~3.2-3.4 reactions each
at *every* depth, because its 25,918-candidate pool barely shrinks. The filter is not starving pools
indiscriminately.

**Why the constraint barely binds on us**, from the enumeration (s42): 2 hubs at depth 0 carrying
11,628 children, 86 at depth 1 (476,084), 95 at depth 2 (130,935), 17 at depth 3 (26,112) — and
**73.8% of all children attach a promoted fragment**, which must itself be built. So a child of even a
purchasable hub usually reaches depth ≥2 on our axis. This is the mechanism behind the axis-mismatch
caveat, not a separate effect.

### The size-matched control — is it depth, or is it less choice?

| arm | s42 (pool 351 → 226) | s44 (pool 394 → 203) |
|---|---|---|
| full routed pool | 98 | 100 |
| size-matched, top-N by reward | 94 | 98 |
| size-matched, random draw 1 | 94 | 99 |
| size-matched, random draw 2 | 92 | 98 |
| **depth ≥2, same size** | **72** | **86** |

Halving the pool costs 4-6% (s42) and 1-2% (s44); depth-filtering to the identical size costs 27% and
14%. The depth effect is 5-15× the size effect. All rows `Optimal`, none capped. The size-matched
controls also sit at 1.01-1.07 reactions per molecule — still on the floor of the metric — so pool
size does not lift the competitor off it and forbidding one-step molecules does.

### DRD2 — a different failure, and it is about the catalogue

`s3gfn_drd2_seed42_stage2_pruned`, gate 0.345: 5,740 above-gate → 730 τ-distinct → 500 written →
**50 routed (10%)** → 28 at depth ≥2.

| min depth | pool | delivered | reactions used | rxn/mode |
|---|---|---|---|---|
| ≥1 | 50 | 47 | 100 | 2.13 |
| ≥2 | 28 | 28 | 93 | 3.32 |
| ≥3 | 16 | 16 | 70 | 4.38 |
| ≥4 | 9 | 9 | 50 | 5.56 |
| ≥5 | 7 | 7 | 43 | 6.14 |

**Every rung from ≥2 is pool-exhausted** — the arm took everything available and left the budget
unspent, so per CLAUDE.md these are flagged datapoints, not cost comparisons. The binding constraint
is that 90% of the pool cannot be routed against our referee's catalogue at all, reproducing [075]'s
unroutable seed-42 finding at the current protocol. No matched hub-batching side was run for DRD2.

### Reagent structure of the delivered libraries

Distinct purchased compounds across the selection's shortest routes (competitor, R=100):

| cell | delivered | distinct reagents | used exactly once | reagents/molecule |
|---|---|---|---|---|
| sEH s43 `_big`, unfiltered | 100 | 132 | 112 | 1.32 |
| sEH s43 `_big`, depth ≥2 | 62 | 98 | 72 | 1.58 |
| ClpP s42, unfiltered | 100 | **165** | **135** | 1.65 |
| ClpP s42, depth ≥2 | 33 | 81 | 72 | 2.45 |
| DRD2 s43, unfiltered | 70 | 100 | 94 | 1.43 |

ClpP unfiltered is the extreme: 100 molecules made in 100 one-step reactions from 165 purchased
compounds, 135 of them used once, most-reused appearing 4 times — no shared structure at all. For
comparison our sEH library at the same budget is 80 molecules off **18 built hubs** (reuse 14/10/9/7/…).

**A measurement note.** The MILP artifacts carry an `n_starting_materials` field reading 339-1,833.
**Do not use it.** The objective is `weights=[1,0,0,0,0]`, so purchases cost nothing and the solver
turns those variables on freely — it tracks ~43% of all compound nodes regardless of cell. The table
above is computed independently from the routes.

### The retraction

A first pass ran the depth filter on the **`_big`** pools (2026-08-28, gate 7.0 on sEH) and reported
67 / 62 / 70 molecules at depth ≥2 against our 80 — a 1.19-1.29× advantage, and ClpP and DRD2
flipping too. Those pools are superseded by stage 2 (2026-08-29), which targets 500 τ-distinct modes
rather than 500 above-gate molecules and uses the current gates. On the current pools the same
constraint yields 72 and 86 against our 93-94 — a much smaller margin. **The `_big` numbers must not
be quoted.** They are retained on disk with `gate_provenance` set, because deleting them would make
the same error easy to repeat.

### Gate provenance — the whole entry is a diagnostic

Everything above uses **superseded hit thresholds**. The sEH front is internally gate-matched (5.68
on both sides), and DRD2 is at 0.345, but the wider matrix results these connect to were computed at
sEH 5.0 / DRD2 0.5 / ClpP −8.0, and the `_big` cells at sEH 7.0 / ClpP −8.0. A publishable version
requires the planned retraining at the current bars first, then this experiment on top. **No number
here belongs in a figure.**

### Catalogue ladder — in flight

Job 75660 (`debug`, balam001, 32 CPUs allocated, script single-threaded), cell 1 of 3. At the time of
writing the ZINC rung reads **53/74 solved = 72%**, against MultiAiZ's independently measured 70.2%
routed on the same pool — two different planners agreeing, which validates the instrument before the
interesting rungs run. Measured throughput **11.1 s/molecule**.

**Two defects were caught by a 3-molecule smoke before this reached the queue**, both of which would
have produced a silently wrong ladder:
1. `finder.search_stats` has **no** `is_solved` key (it carries `time`, `iterations`,
   `first_solution_time`, `first_solution_iteration`, `returned_first`), so reading one there returns
   `None` and marks *every* molecule unsolved — an all-zero ladder that looks like a finding.
2. `routes.reaction_trees` includes **unsolved partial** trees, whose reaction count is not a route
   length — so the first version reported a depth for molecules the catalogue cannot make.
Both fixed by using `ReactionTree.is_solved` ("all leaf nodes are in stock"). Post-fix the smoke gives
`solved: true, min_steps: 1/2/5` against ZINC and `solved: false, min_steps: null, n_routes: 5`
against the 418 blocks.

### Known imprecision

`run_campaign.py` prints `POOL COLLAPSED` when *either* pool empties. On seed 43 at depth ≥5 the
best-candidate pool reached 0 while hub-batching still held 262 children over 94 hubs; both delivered
0 molecules at the budget, so the reported result is right, but the message overstates what emptied.

### The ZINC-axis re-measurement (job 75749, 2026-09-06) — supersedes the axis caveat above

Hub-batching's delivered molecules routed against ZINC with plain AiZynthFinder, so BOTH sides'
depth is "reactions from ZINC-purchasable material". sEH seed 42, gate 5.68, R=100. Two readings of
the unroutable molecules, because dropping them removes exactly the molecules FURTHEST from the
competitor's catalogue while counting them as deep does the opposite; the truth is between.

| min ZINC depth | ours, `drop` | rxn/mode | ours, `deep` | rxn/mode | competitor | ratio |
|---|---|---|---|---|---|---|
| >= 1 | 94 | 1.08 | 96 | 1.07 | 98 | 0.96-0.98x |
| >= 2 | 92 | 1.14 | 94 | 1.08 | 72 | **1.28-1.31x** |
| >= 3 | 86 | 1.23 | 91 | 1.12 | 60 ⚠capped | 1.43-1.52x |
| >= 4 | 82 | 1.33 | 89 | 1.15 | 53 | **1.55-1.68x** |
| >= 5 | 73 | 1.44 | 86 | 1.17 | 38 | **1.92-2.26x** |

best-candidate holds flat at 26-28 modes at every rung — the internal control that says the response
is to the constraint, not to pool starvation.

**Convergence status is part of the number.** The driver re-runs selection, routes whatever it chose,
drops what comes back shallow or unroutable, and repeats. All five `deep` rungs CONVERGED (1/2/11/32/9
rounds). All five `drop` rungs hit the 40-round cap with 2/3/10/17/42 molecules still unjudged, so the
`drop` column is an UPPER bound — tight at depth 1, loosest at depth 5, which is also where the band
is widest (73 vs 86) because 39% of our molecules are unroutable by plain AiZynth. Total routing:
11,167 molecules, far above the 1,000-2,000 estimated when the loop was designed, because each
rejection pulls a fresh unjudged candidate into the walk.

Planner asymmetry, accepted deliberately and reported: the competitor's pool was routed with MultiAiZ
(5 cycles, set-based), ours with plain AiZynthFinder. The weaker planner fails molecules MultiAiZ
would solve, so our pool is understated relative to theirs.

### Catalogue ladder — the finding is CURATION, not size

Completed 2026-09-06, three cells, 100 molecules each at iter=100/time=60/depth=6.

| catalogue | keys | sEH s42 | sEH s44 | DRD2 s42 |
|---|---|---|---|---|
| ZINC | 17,422,831 | 71% | 70% | **16%** |
| ZINCFrag | 178,598 | 51% | 52% | **39%** |
| random ZINC, 178,598 keys | 178,598 | — | **0%** | **0%** |
| our 418 blocks (`rgfnlib`) | 456 | 0% | 0% | 0% |
| random ZINC, 456 keys | 456 | 0% | — | — |

**A random 178,598-compound slice of ZINC routes NOTHING; a curated building-block set of exactly the
same size routes half.** The competitor's advantage is not that its catalogue is large, it is that its
catalogue is building blocks. This retires the "~40,000x asymmetry" framing — the real comparison is a
curated 178k block set against our curated 456, and at 456 keys nothing routes regardless of curation,
so both scale and curation are necessary. Median route depth is 2.0 at both ZINC and ZINCFrag, so
shrinking the catalogue 100x removes molecules wholesale rather than lengthening routes.

**Two caveats.** (i) `zinc` and `zincfrag` are NOT nested in our implementation — `zinc_stock.hdf5` is
17.4M InChIKeys from one ZINC snapshot, `zincfrag_stock.hdf5` is built from ZINCFrag's fragment SMILES,
and many of the latter are absent from the former. DRD2 inverts the ordering (39% vs 16%) for exactly
this reason. Do not describe ZINC as a superset. (ii) 0% from our blocks is NOT "our library cannot
make these molecules" — AiZynth searches with USPTO templates while our generator uses its own 112,
which is the template/block incompatibility of entry [048]'s Exp A. The defensible claim is narrower:
a standard planner cannot rediscover routes to S3-GFN's molecules from a 456-compound catalogue.

### A pre-existing bug found by a cross-check: NaN-scored molecules entered pools

`build_s3gfn_pools.py::load_ranked` phrased its gate as "skip if it fails":
`if (val < gate) if higher_is_better else (val > gate): continue`. Every comparison against NaN is
False, so an UNSCORED molecule fell through the gate and entered the pool as if it had passed
(`float("nan")` also parses cleanly, so the `except` never fired). Surfaced because the new
`--catalogue-distinct` guard cross-checks the pool against an independently built block-similarity
cache, and the two disagreed by exactly the NaN rows.

**20 of 176 production pools are contaminated.** Worst: `s3gfn_clpp_seed42/43/44_stage2_N500` at
**408/422/403 NaN of 500** — those pools are ~82% unscored molecules. Also `reinvent_clpp_*` at 27-45
each, and single digits in `saturn_clpp`, `tango_clpp` and `s3gfn_seh_*_big`. Docking targets are hit
hardest because a docking failure writes NaN while a surrogate almost always returns a number.

**Scope of the damage.** Every OTHER gate site phrases the test positively and rejects NaN correctly:
`sparrow_select_frontier.passes_gate`, `mode_saturation.py:104`, and
`validation/lsdflow/metrics/diversity._passes_gate` (which carries an explicit `reward != reward`
guard). `upsample_to_modes.py` counts modes through that function and so inherits it. So selection
results and mode counts are NOT corrupted — but reported POOL SIZES are overstated, MultiAiZ routing
effort was spent on molecules that should never have been routed, and worst, **pool-exhausted flags
get misattributed**: a ClpP cell that reads as generator collapse may have had a pool that was 82%
unscored. Fixed by the co-agent as `3a66acf` (canonical); this session's identical working-tree edit
is not committed.

**A deeper problem underneath it, handed to the pipeline session.** The SOURCE candidate files are
~48-50% NaN: `s3gfn_clpp_seed42/43/44` stage-2 candidates carry **exactly 8,000** unscored rows each
(of 16,095 / 16,747 / 16,521). Exactly 8,000, three times, across seeds with different row totals, is
not a failure rate — 8,000 = 2 x 4,000 and 4,000 is `s3gfn_sample_more.py`'s `--round-size` default,
so the signature is two entire scoring rounds returning nothing. Both `score` and `raw_score` are NaN
for all 8,000, consistent with the docker never producing a pose rather than a scoring bug. If the
docker never started, those molecules were never scored and the Stage-2 oracle-call surcharge for
every ClpP cell is OVERSTATED by 8,000 rather than merely low-yield. Diagnostics and the
`rgfn-smoke-env` hypothesis were passed to the Standard Pipeline session; unresolved here.

### Catalogue-distinct ("c-mode") sweep — first numbers

A stricter mode definition needing no retrosynthesis: a molecule must be Tanimoto-< tau from every
accepted mode AND from every purchasable building block (ZINCFrag + our 418). ONE knob moves both
tests. sEH seed 42, gate 5.68, R=100.

| tau | hub-batching | rxn/mode | best-candidate | competitor | rxn/mode |
|---|---|---|---|---|---|
| 0.5 | **95** | 1.07 | 27 | pending | |
| 0.4 | **78** | 1.39 | 25 | pending | |
| 0.3 | **51** | 2.16 | 25 | **49** (pool 90) | 2.04 |

**At tau=0.3 the two methods converge — 51 vs 49, and their rxn/mode is marginally better than ours.**
Our arm goes pool-exhausted at 128 modes from 391,876 candidate children while theirs held 90; two
pools differing by four orders of magnitude land within 4%, which says the reaction budget is no
longer the binding constraint at that cutoff — mutual dissimilarity is. Hubs used move the opposite
way to intuition (19 -> 91 -> 97 as tau tightens), which is why rxn/mode doubles.

Pool availability, measured from 8,922 above-gate molecules after an enlarged draw: tau=0.5 gives
7,246 block-passers and a full 500-mode pool; tau=0.4 gives 2,415 and a full 500; **tau=0.3 gives 180
block-passers and only 90 mutually distinct — pool-limited even after 5.8x the sampling.**

**Two of my own claims about this sweep were tested and withdrawn.** (i) "The block test is a fixed
bar so more sampling helps linearly" — false; the tau=0.3 yield fell from 7.8% to 2.0% between draws.
(ii) The follow-up story, "S3-GFN's extra draws are increasingly catalogue-like" — also false: median
block-similarity is FLAT across reward deciles (0.425 -> 0.444 best-to-worst, pass-at-0.3 wandering
1.2-2.6% with no trend), and the two draws are near-identical in the middle (median block-sim 0.440 vs
0.441, median reward 6.37 vs 6.31, pass-at-0.4 28.8% vs 25.2%, 934 molecules shared). They differ only
in the far-left tail: pass-at-0.3 7.5% vs 2.8%. The real finding is a caveat — **at tau=0.3 the
qualifying pool is scarce enough that its size swings ~2.7x between two draws from the same frozen
checkpoint**, so that rung is both pool-limited and draw-sensitive and must not carry a conclusion
alone. tau=0.5 and 0.4 are stable across draws.

### Operational notes worth reusing

- **`conda activate rgfn` is not enough in a batch job.** `run_campaign.py` imports `glue` -> dgl,
  which needs the torch-bundled CUDA libs on `LD_LIBRARY_PATH`; without them the import dies with
  `Cannot load Graphbolt C++ library`. Job 75746 failed all ten configurations in 52 s this way, and
  **login-node testing cannot catch it** because the helper is always sourced by hand there. Use
  `source ~/bin/rgfn-smoke-env.sh`, and verify a real job gets PAST the import rather than trusting
  the submit receipt.
- **Forking a worker pool after a heavy import hangs.** Job bf5n600kr: SIGTERM at 30 min, zero output,
  zero cache rows. Precompute in a clean process; readers never fork.
- **Long login-node work dies silently.** A `nohup` block-sim run died at 5,000 records with the log
  still showing a healthy ETA. `setsid nohup ... < /dev/null` survives, but SLURM is the right answer.
- **8 workers x 64 OpenBLAS threads exhausts the login node's thread limit.** Pin
  `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- **Do not pipe or filter the first run of a new script.** Two results in this session were lost to my
  own greps (`"  "` matched my own output lines) and one traceback to stdout buffering under SIGTERM.
