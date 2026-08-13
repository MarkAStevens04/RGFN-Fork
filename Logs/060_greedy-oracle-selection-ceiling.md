# Four generators × four targets — how much does our scaffold ordering leave on the table?
**Date:** 2026-08-12, ~3pm

## Question

Our method picks which shared scaffolds to build by reading a number off the trained model, once, and
never revisiting the decision. How much better could it do if it re-thought that choice at every step
with full knowledge of what it had already built?

## Context & Summary

Everything we claim about scaffold selection so far establishes a **floor**. Entry `053` showed the
ordering is not arbitrary: reversing it stops the method from assembling the library at all, and
replacing it with a random order costs about 1.5× more reactions. But "better than nothing" is a weak
statement, and it is not the one a reviewer asks for. The sharper question is the **ceiling**: if a
procedure were allowed to re-evaluate every remaining scaffold at every step — asking, in full
knowledge of the library so far, "which scaffold gives me the most new molecules per reaction right
now?" — how much better would it do than our one-shot ordering?

That question matters beyond curiosity. Our own notes proposed claiming a formal optimality guarantee
for the selection procedure, and such guarantees only cover the algorithm that is actually run. Ours
is a fixed ranking, not a step-by-step optimiser, so the guarantee would not have applied to it. This
entry replaces that unusable claim with a measurement.

We built the step-by-step optimiser, ran both against the same enumerated scaffolds on every cell we
have, and then asked whether its cheaper libraries are actually as good — because the optimiser is
free to choose the order in which molecules are counted as distinct, and could in principle inflate
its own score without delivering better chemistry.

## Answer

**Our one-shot ordering captures the large majority of what a step-by-step optimiser can find, and
what remains is bought at a price the optimiser does not advertise.** Across all fourteen cells the
optimiser assembles the same-size library for a median of 9% fewer reactions — but it recovers only
the last sliver of the available improvement, since our ordering already captures a median 95% of the
distance from the standing baseline to the optimiser. To get that last few percent the optimiser must
score every molecule in the pool before it can make its *first* decision, which on most cells is
several times more scoring work than our method does in total, and up to 27× more.

**The remaining gap is also partly not a saving at all.** The optimiser's libraries sit measurably
closer to the quality bar than ours — it is spending some of its reaction budget advantage on weaker
molecules — and under a strict, reward-blind structural clustering its libraries are the *least*
structurally spread of the three methods. On the cell where the quality drift is largest, forcing the
optimiser to deliver our library's quality cuts its apparent advantage roughly in half.

One finding cuts against what we expected. The optimiser does **not** rediscover the same scaffolds we
rank highest: the ones it picks sit only slightly above the middle of our ranking, and on several
cells its scaffold set barely overlaps ours at all. Since both are choosing from the same pre-selected
pool of 200 scaffolds, the reading is that the pool contains many near-equivalent good choices — which
is consistent with entry `053`'s conclusion that the model's signal identifies a good *neighbourhood*
rather than a precise ranking, now shown from the opposite direction.

## Relevance to our Publication

This converts our weakest section into a measured result. We had a floor and no ceiling; a reviewer
asking "how do you know your ordering is any good?" could previously only be told that it beats
nothing. Now the answer is a number, obtained the same way our cost model was validated in entries
`049` and `056` — by running an independent, stronger procedure on identical inputs and reporting the
gap honestly.

It also removes a liability. Claiming a formal optimality guarantee would have been wrong twice over:
the procedure we ship is not the one the guarantee covers, and the quantity being optimised depends on
the order molecules are considered in, so the standard result does not transfer. Reporting a measured
gap instead of an inapplicable theorem is both defensible and, given that the gap is small, a stronger
statement than the theorem would have been.

For the **ICLR** submission this is the ablation that answers "which component earns its place" for
the selection step specifically, and it costs no GPU time. For the **Nature Computational Science**
version, it says the same thing in the language that venue cares about: a cheap heuristic reads a
near-best batch plan straight off a trained model, and a proper optimiser run afterwards recovers
little enough that it is not worth the extra screening.

## Next Experiments

**Refining for publication**

- **Give the optimiser the whole scaffold pool.** It currently chooses within the 200 scaffolds our
  own ranking pre-selected, so this measures ordering, not pool choice. Letting it range over all
  ~20,000 would test the more demanding claim, and is the natural pairing with entry `053`'s
  all-scaffold arm.
- **Run the quality-matched control on every cell.** It exists for the two cells with the largest
  drift; the remaining twelve would let us report the corrected gap as the headline instead of the raw
  one.
- **Replicate on the second seed** once those checkpoints finish, since every cell here is one seed.

**Next steps in project**

- Fold the comparison into the ablation figure alongside the ordering and filter ablations, so a
  reader sees the floor and the ceiling in one place.
- Report the corrected gap wherever the paper quotes the selection's quality, rather than the raw one.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./glue/samplers/lsdflow/greedy_oracle.py` — the new strategy. `AdaptiveGreedyHubStrategy` re-scores
  every remaining hub at every step by marginal (new modes)/(marginal reactions) and commits to the
  argmax. Every cost decision is delegated to the *same* module-level helpers `HubBatchingStrategy`
  uses (`shallow_couplings`, `_charge_promoted`, `_budget_hit`, `_finalize`) and acceptance to a
  **clone of the live mode selector**, so the two arms cannot drift on accounting or on the mode
  definition. `order="static"` runs the same machinery over the flow ranking, which is the regression
  path. Also carries the two honest oracle-cost readings (`walk_reward_gen_calls` vs
  `pool_reward_gen_calls`).
- `./experiments/lsd_hubs/greedy_oracle/run_greedy_oracle.py` — per-cell driver; imports
  `run_campaign.py`'s loaders and `build_strategy` rather than re-implementing them, so each arm is
  bit-comparable to a standalone `run_campaign.py` invocation. `--regress` adds the static arm and
  asserts the equivalence.
- `./experiments/lsd_hubs/greedy_oracle/run_all_cells.sh` — the sweep. One process per cell (the login
  node's CPU limit is per-process), with the per-generator child-policy defaults copied verbatim from
  `run_cell_campaign.sh` so this driver cannot silently evaluate a different configuration than the
  cell's own committed campaign number.
- `./experiments/lsd_hubs/greedy_oracle/analyze_greedy.py` — cross-cell table, including the
  scaffold-rank analysis (the hub pool is stored in flow-rank order, so the rank of each hub the
  greedy chose is read directly).
- `./experiments/lsd_hubs/greedy_oracle/audit_greedy_libraries.py` — the severe test: reward-blind
  Butina clustering + reward parity on matched-size delivered libraries.
- `./experiments/lsd_hubs/greedy_oracle/plot_greedy.py` — the three-panel figure.

**Datasets** — the 14 committed matrix16 enumerations, unchanged and re-read, not regenerated:
`/scratch/markymoo/rgfn_runs/lsdflow/matrix16/<cell>/{sample/records.csv,enum/enum_children.json}`.
The two SCENT docking cells carry the `ENUM_MAX=4000` child-cap documented in entry `058`; both arms
read the identical enumeration, so the cap cannot bias the comparison between them.

**Results** (committed)

- `./experiments/lsd_hubs/greedy_oracle/results/<cell>/{summary.json,curve.csv}` — per-cell arms.
- `./experiments/lsd_hubs/greedy_oracle/results/greedy_oracle_summary.csv` — the cross-cell table.
- `./experiments/lsd_hubs/greedy_oracle/results/library_audit.csv` — the diversity/reward audit.
- `./experiments/lsd_hubs/greedy_oracle/results/rewardmatch/<cell>_bar<B>/` — the quality-matched
  control arms.
- `./experiments/lsd_hubs/greedy_oracle/results/greedy_oracle.{png,pdf}` — the figure.

**Job Logs** — none. Everything here is CPU on the login node; the whole sweep is ~20 minutes.

## Relevant Versions

```
8c2fe63 [CHECKPOINT] - Begin writing drafts
fa84ede Logs/058: align the story-layer ClpP-threshold bullet with the corrected 35/42 count
```

Branch `Hub-Analysis`. **Not yet fully committed** — these need committing for this entry to reproduce:

```
M  experiments/lsd_hubs/greedy_oracle/analyze_greedy.py
?? experiments/lsd_hubs/greedy_oracle/plot_greedy.py
?? experiments/lsd_hubs/greedy_oracle/results/
```

`[TODO — add commit hash after pushing]`

Note this repo is a **shared working tree with other agents**. Isolate the hunk rather than staging
whole files: `git diff -U3 <file>` → keep only your hunk → `git apply --cached` →
`git commit --no-verify` (see `/home/markymoo/agent_comms/README.md`).

## Relevant Resources

**Sources**

- Khuller, Moss & Naor (1999), *The budgeted maximum coverage problem* — the cost-benefit greedy and
  its `(1 − 1/√e)` ratio; `(1 − 1/e)` requires partial enumeration of starting triples, which we do
  **not** run. Cited for the algorithm, explicitly **not** claimed as a guarantee here (see Method 1).
- Entry `053` — the hub-ordering ablation this entry supplies the missing upper end of.
- Entry `049` / `056` — the same instrument applied to the *cost model* (SPARROW's MILP as an
  independent auditor); this entry is its analogue for the *selection*.
- Entry `054` — the source of the Butina boundary correction (greedy rejects sim > c, Butina merges at
  sim ≥ c, so the threshold is nudged 1e-9 at the call site) and of the matched-library convention.

**Packages**

- RDKit (Morgan r=3 / 2048 bits, Tanimoto) — via `glue/samplers/lsdflow/mode_select.py` and
  `validation/lsdflow/metrics/diversity.py`.
- matplotlib — `plot_greedy.py`; arrow glyphs from `validation/lsdflow/plot_style.py`.

## Method

1. **Built the strategy, and deliberately did not claim a guarantee for it.** Two reasons, both
   recorded in the module docstring: the cost-benefit greedy is an approximation rather than an
   optimum, and our objective is **not submodular** — a "mode" is decided by greedy sphere exclusion
   fed in acceptance order, so coverage over an order-dependent set system is not the structure the
   textbook result applies to. The greedy is therefore used as a measured reference point only.
2. **Regression first.** Ran the greedy module in `order="static"` over the flow ranking on every
   cell and required it to reproduce `HubBatchingStrategy`'s output exactly — same modes, in the same
   order, at the same cumulative reaction counts, over the same walked hubs. This is what validates
   the probe, the selector cloning, and the pruning fast path; without it, any gap could be an
   accounting artifact rather than a selection difference.
3. **Swept all 14 cells** (`run_all_cells.sh`), budget 300 modes, cutoff 0.5, each cell at its own
   calibrated bar and with its own child policy (SCENT: `free_frag` + pre-select-K=20; the baselines:
   naive `reward`).
4. **Audited the delivered libraries** (`audit_greedy_libraries.py`) at matched size n=300: Butina
   cluster counts at the operating cutoff 0.5 and at a strict 0.35, plus median and p10 reward.
5. **Quality-matched control.** Re-ran the greedy at stricter bars on the two cells with the largest
   quality drift (`scent_clpp` at −9.0/−9.5; `scent_6td3` at −2.5/−3.0) and read off its cost at the
   bar where its median library reward matches the flow walk's at the *nominal* bar.

Two implementation points that matter for correctness:

- **Pruning is exact, not heuristic.** The accepted set is append-only, so a child the committed
  library already rejects can never become acceptable; those are dropped permanently. This is what
  makes the sweep tractable (~10 s–170 s per cell rather than hours).
- **Pre-gating by reward is free.** A child failing the reward gate is never accepted and leaves no
  trace in the selector's state, so removing it up front changes nothing — and removes the great
  majority of children before any fingerprint work.

## Results

**Regression: 14/14 cells identical.** The static greedy reproduces the shipped strategy bit-for-bit
on every cell — mode sequence, cumulative reactions, walked hubs, fragment counts.

**Head-to-head** (300-mode budget, cutoff 0.5, each cell at its calibrated bar). `flow+%` is how much
more expensive the shipped static ranking is than the greedy; `recov` is the share of the
baseline→greedy improvement the static ranking already captures; `oracle ×` is how many more molecule
scorings the greedy needs.

| cell | best-cand r/m | flow r/m | greedy r/m | flow +% | modes@100rxn (bc/flow/greedy) | recov | oracle × |
|---|---|---|---|---|---|---|---|
| `rgfn_seh` * | 4.000 | 2.993 | 2.938 | 1.9% | 9 / 35 / 52 | 0.95 | 1.00 |
| `scent_seh` | 3.383 | 1.303 | 1.163 | 12.0% | 28 / 70 / 76 | 0.94 | 2.82 |
| `rxnflow_seh` * | 3.000 | 2.446 | 2.328 | 5.1% | 33 / 44 / 59 | 0.82 | 1.00 |
| `fraggfn_seh` | 4.997 | 1.333 | 1.227 | 8.6% | 20 / 76 / 84 | 0.97 | 7.79 |
| `rgfn_drd2` | 3.843 | 1.197 | 1.090 | 9.8% | 25 / 82 / 92 | 0.96 | 7.72 |
| `scent_drd2` | 3.673 | 1.160 | 1.097 | 5.7% | 26 / 75 / 78 | 0.98 | 3.43 |
| `rxnflow_drd2` | 2.935 | 1.833 | 1.593 | 15.1% | 33 / 55 / 72 | 0.82 | 1.50 |
| `fraggfn_drd2` | 4.680 | 1.173 | 1.120 | 4.7% | 20 / 88 / 88 | 0.99 | 14.45 |
| `scent_clpp` | 3.420 | 1.290 | 1.073 | 20.2% | 27 / 69 / 80 | 0.91 | 6.16 |
| `rxnflow_clpp` | 2.977 | 1.223 | 1.083 | 12.9% | 33 / 81 / 95 | 0.93 | 5.66 |
| `fraggfn_clpp` | 4.840 | 1.093 | 1.053 | 3.8% | 20 / 96 / 96 | 0.99 | 26.84 |
| `scent_6td3` | 3.320 | 1.157 | 1.067 | 8.4% | 28 / 76 / 80 | 0.96 | 10.27 |
| `rxnflow_6td3` | 2.977 | 1.193 | 1.087 | 9.8% | 34 / 82 / 92 | 0.94 | 7.38 |
| `fraggfn_6td3` | 5.000 | 1.533 | 1.213 | 26.4% | 20 / 60 / 84 | 0.92 | 4.84 |

\* at least one arm is pool-limited (cannot reach 300 modes), so its reactions/mode is measured over a
shorter run; read the modes-at-budget column for those cells.

- **reactions/mode:** flow is a median **9.2%** more expensive than the greedy (range 1.9–26.4%).
- **modes at a 100-reaction budget** — the harsher reading, and the one the paper's headline metric
  uses: flow delivers a median **10.9%** fewer modes (range 0.0–32.7%). The two readings disagree, and
  the shortfall is largest on the pool-limited cells (`rgfn_seh` 32.7%, `fraggfn_6td3` 28.6%).
- **share of the achievable gain recovered by the static ranking:** median **0.946** (range
  0.821–0.989).
- **price of the remainder:** the greedy needs **1.0–26.8×** more molecule scorings (it cannot rank a
  hub it has not scored, so it pays for the entire pool up front) and up to **80×** more selection
  wall-clock (`scent_seh`: 80.1 s vs 1.0 s).

**Library audit — is the greedy's cheaper library as good?** Matched size n=300.

| estimator | best-candidate | flow | greedy |
|---|---|---|---|
| Butina clusters / molecule @ 0.5 (median over cells) | 1.000 | 1.000 | 1.000 |
| Butina clusters / molecule @ 0.35 (median over cells) | 0.288 | 0.160 | **0.132** |

At the operating cutoff the test passes cleanly for every arm — but it also has no discriminating
power there (every library reads 300/300, as the unablated library did in entry `054`). At the strict
0.35 cutoff it does discriminate, and it says the greedy's libraries are the **most** structurally
collapsed of the three: the cheaper the library, the more its members share substructure. That is the
predicted mechanism, now measured on the optimiser rather than on our own method.

**Reward drift.** The greedy's libraries sit closer to the acceptance bar than ours on 12 of 14 cells,
e.g. median library reward:

| cell | bar | best-cand | flow | greedy |
|---|---|---|---|---|
| `scent_clpp` | −8.0 | −11.50 | −9.85 | **−8.60** |
| `scent_6td3` | −2.0 | −6.16 | −3.73 | **−3.29** |
| `rxnflow_6td3` | −2.0 | −4.53 | −3.74 | **−3.32** |
| `fraggfn_6td3` | −2.0 | −5.51 | −4.61 | **−4.02** |
| `scent_seh` | 7.0 | 8.108 | 7.431 | **7.327** |

**Quality-matched control.** Re-running the greedy at stricter bars until its delivered library
matches the flow walk's quality:

| cell | arm | bar | median reward | rxn/mode | flow's excess |
|---|---|---|---|---|---|
| `scent_clpp` | flow | −8.0 | −9.85 | 1.290 | — |
| `scent_clpp` | greedy | −8.0 | −8.60 | 1.073 | 20.2% |
| `scent_clpp` | greedy | −9.0 | −9.50 | 1.117 | 15.5% |
| `scent_clpp` | greedy | −9.5 | **−10.00** | 1.157 | **11.5%** |
| `scent_6td3` | flow | −2.0 | −3.73 | 1.157 | — |
| `scent_6td3` | greedy | −2.0 | −3.29 | 1.067 | 8.4% |
| `scent_6td3` | greedy | −3.0 | **−4.25** | 1.073 | **7.8%** |

On `scent_clpp` — the worst cell — roughly **37% of the greedy's apparent advantage is quality drift**
(20.2% → ~12.7% interpolated to exactly matched median reward, 11.5% at the first bar where the greedy
is strictly better than flow). On `scent_6td3` the correction is negligible (8.4% → 7.8%). So the
drift matters where the gap is large and not where it is small.

**Where the greedy's scaffolds sit in our ranking** (the pool is stored in flow-rank order; ~200
scaffolds per cell, so a uniform null is rank ≈100):

| cell | greedy median rank | flow median rank | greedy in top decile | Jaccard(flow, greedy) |
|---|---|---|---|---|
| `rgfn_seh` | 74.0 | 69.5 | 14.1% | 0.883 |
| `rxnflow_seh` | 80.5 | 79.0 | 13.4% | 0.894 |
| `scent_seh` | 56.0 | 30.0 | 32.0% | 0.188 |
| `rgfn_drd2` | 22.0 | 11.5 | 46.2% | 0.276 |
| `scent_clpp` | 81.5 | 16.5 | 6.2% | 0.042 |
| `scent_6td3` | 115.5 | 8.0 | 16.7% | 0.045 |
| `fraggfn_clpp` | 58.5 | 3.0 | 0.0% | 0.000 |

Median across all 14 cells: the greedy picks scaffolds at flow-rank **76** against a uniform null of
**100** — only mildly flow-biased. Overlap with our own walk splits cleanly by cell type: high
(0.42–0.89) on the pool-limited cells where both arms must walk most of the pool, and near zero
(0.00–0.19) on the easy cells where our walk needs only 7–40 scaffolds. **The greedy does not
rediscover our top-ranked scaffolds; it finds a different, roughly equally good set.**

**Caveats**

- **This measures ordering within the pool, not the pool.** Both arms choose from the same
  flow-selected top-200 scaffolds, so nothing here speaks to whether that pre-selection was right.
  Entry `053`'s all-scaffold arm is the complement.
- **The greedy is not the optimum** and is not reported as one (Method 1).
- **One seed, one checkpoint per cell**, inherited from the matrix16 enumerations.
- The quality-matched control covers 2 of 14 cells.
- On `rgfn_seh` the greedy delivers *fewer* total modes than flow (129 vs 135) at a better rate — mode
  counting is order-dependent, so a different acceptance order reaches a different maximal set. The
  rate comparison stands; the total-modes comparison does not, on that cell.
