# SCENT / sEH — what does each half of our "useful molecule" definition actually buy?

**Date:** 2026-07-30, ~11am

## Question

We only count a molecule as useful if it both scores above a quality bar and looks different from
everything we have already kept — so what happens to the library we deliver, and to the cost we report
for it, if we drop one of those two requirements?

## Context & Summary

Our headline number is **reactions per useful molecule**. Its denominator is a definition we chose: a
molecule counts only if it (a) scores above a bar and (b) is chemically dissimilar from every molecule
already in the library. Because we propose that metric *and* win on it, our publication notes call the
metric defence the single largest threat to acceptance, and they name the two ways a number of that
shape can be gamed from the denominator side. Each of our two requirements blocks one of them: the
quality bar stops a strategy looking cheap by making weak molecules, and the dissimilarity requirement
stops it looking cheap by making the same molecule three hundred times.

Entry `051` showed the quality bar is a legitimate control (it barely changes how diverse the molecule
pools themselves are below a bar of ~7), and entry `052` showed our advantage survives moving both
knobs across a 221-cell grid. Neither asks what happens when a requirement is removed *entirely*. Entry
`026` counted how many molecules each filter throws away, but not what the surviving library is made
of. This entry closes that gap, and it is the ablation a reviewer asks for when they see two
hand-chosen filters in a metric definition.

We take the SCENT model trained on sEH — the checkpoint anchoring the whole cost campaign — and run the
two library-building strategies with each requirement relaxed, one at a time, all the way to removed.
Removing a requirement is just the permissive end of its own dial, so instead of asserting an ablation
we **scan** each dial from "off" through its normal setting to strict. Every arm builds a library of
exactly the same size (300 molecules), so the comparison is not confounded by how many molecules each
one happens to produce. For each library we then ask three things: what it cost, how good its molecules
score, and how chemically distinct they actually are. Finally we re-apply the *original* definition to
every delivered library and divide — giving the honest cost per molecule you would actually keep.

## Answer

**Both requirements are load-bearing, and — the part that matters for how we report this — removing
either one hurts our own method more than it hurts the baseline we compare against.** Dropping the
quality bar makes hub-batching's cost look 13% better while 91% of the library it delivers falls below
the bar; dropping the dissimilarity requirement makes it look 12% better while the 300-molecule library
collapses to 39 genuinely distinct molecules. Priced per molecule actually worth keeping, those two
"savings" cost 9.8× and 6.8× respectively, and 29× with both removed. So the filters are not
conveniences that flatter our numbers: they are what makes the advantage we report real, and a
strategy that skipped them would be visibly worse, not better.

**A clustering that never looks at the score agrees**, which matters because our usual way of counting
distinct molecules walks the set best-scoring-first — a procedure ordered by the very quantity under
test. Counting families by chemical neighbourhood instead, with the score taken out entirely, gives the
same verdict and is if anything slightly harsher: the diversity-ablated library holds 144 families
rather than the 153 our own count reports, against 300 for the unablated one. The two filters also turn
out to defend near-orthogonal things — with the quality bar removed, both counts still say 300 distinct
molecules, so that ablation costs quality and not diversity.

**The failure has a recognisable shape rather than being diffuse noise.** With either requirement
removed, hub-batching stops walking the hub ranking and collapses onto a handful of shallow scaffolds
— 43 hubs at the normal setting, 5 with either filter off, 2 with both — and the typical library member
drops from two nested reactions to one. That is our publication notes' depth-0 catalog degenerate
optimum appearing empirically: with nothing forcing quality, the cheapest way to make three hundred
things is to decorate the shallowest scaffold available.

**The baseline strategy is exactly indifferent to the quality bar**, reproducing to four decimal places
across every bar from "off" to 8.0 — it works down a list sorted by score, so a bar it never reaches
cannot bind. That is the exact-limit version of the near-invariance entries `035`/`050` reported and
the invariance `052` measured, and it means the reward filter's whole effect in our comparison falls on
the method that could otherwise exploit its absence.

## Relevance to our Publication

This is the §2.4 numerator guard, which our publication notes have listed as missing since they were
written: reward *and* synthetic-depth distributions reported alongside the cost ratio, showing the
reaction savings are not bought with molecular quality. It also converts §2.5's degenerate-optimum
discussion from an argument into a measurement — the depth-0 exploit is not hypothetical, it is what
our own method does when the bar is removed, and we can now say so with the numbers and state that
conditioning on a reward threshold is what closes it.

For Digital Discovery (or an ML4Science workshop), the specific reviewer question this answers is "you
defined the denominator, so why should I believe the definition isn't chosen to win?" The strongest
available reply is not that the filters are standard — it is that removing them *inverts our own
result*: with no quality bar, hub-batching's apparent 2.88× advantage becomes a 3.9× disadvantage per
qualified molecule. An ablation that hurts the authors is much harder to dismiss than one that doesn't.
Together with `051` (the bar doesn't move the pools) and `052` (the advantage holds across both knobs),
the metric defence is now three measurements rather than three assertions.

## Next Experiments

**Refining for publication**

- **Repeat on a second model and target.** One checkpoint, one reward — the same limitation `051` and
  `052` carry. The matrix16 cells (entry `050`) already hold sampled + enumerated pools for RxnFlow and
  for DRD2, and the driver takes them as arguments, so this is a re-run rather than new work.
- **Add the reward-blind-ordering arm.** We removed the reward *gate* but kept best-reward-first
  ordering, because that ordering is the strategy rather than the filter. A third arm that also drops
  the ordering would separate "what the gate buys" from "what reward information buys at all" — and it
  is the arm that would move best-candidate, which is otherwise inert here.
- **Do it on a docking reward before quoting a docking cost claim.** 6TD3/ClpP bars were calibrated
  separately (entry `045`) and neither knob has been ablated there; the driver already supports
  lower-is-better rewards.

**Next steps in project**

- **Report reactions per qualified mode next to reactions/mode wherever the ratio appears.** For an
  unablated run the two are identical by construction, so it costs nothing to state, and it makes the
  metric visibly un-gameable rather than requiring the reader to trust the definition.
- **Reuse the ablated selector as a control arm in the active-learning loop.** `RewardOnlyModeSelector`
  is production code in `glue/`, and the AL acquisition takes the same selector seam, so the "what if
  we didn't filter" arm is available there without new machinery.

# Re-creation

## Relevant Files

Repo root `./`; scratch paths absolute.

**Scripts**

- `./experiments/lsd_hubs/filter_ablation/filter_ablation_scan.py` — the whole analysis: builds the
  cross-shaped cell design, runs both strategies per cell through `run_campaign.py`'s own
  `build_strategy` (so a cell is the same computation as a `run_campaign` invocation), computes the
  cost/reward/diversity/depth readouts, and writes CSV + JSON + the figure. `--plot-only` redraws.
- `./experiments/lsd_hubs/filter_ablation/README.md` — what the sub-dir measures, the matched-N design
  constraint, the two ablation semantics, and how to re-run it.
- `./glue/samplers/lsdflow/mode_select.py` — **production** mode selectors. Adds
  `RewardOnlyModeSelector` (the diversity ablation: reward gate + exact-duplicate suppression, no
  similarity test) and factors the shared reward gate into a module-level `passes_reward_gate` so both
  selectors provably apply the identical, NaN-safe, orientation-aware rule.
  `DiverseThresholdModeSelector` is behaviourally unchanged.
- `./experiments/lsd_hubs/campaign/run_campaign.py` — `build_strategy` gains an optional
  `mode_selector_factory` passthrough (the seam both strategy classes have always exposed but no
  driver had used). Default `None` → the canonical selector, so every existing caller is unchanged;
  verified byte-identical below.

**Datasets** — the SCENT × sEH campaign anchor (checkpoint
`/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/train/checkpoints/last_gfn.pt`),
the same three inputs entries `051`/`052` use:

- `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/{records.csv,compositions.json}` — 29,997
  sampled trajectories → 26,069 unique molecules (rewards 0.425–8.404, no NaN) = the pool
  **best-candidate** selects from, plus each molecule's promoted-fragment composition so shared parts
  can be charged once.
- `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json` — exhaustive
  one-reaction children of the 200 ranked hubs (828,448 children, rewards 0.000–8.404, no NaN, hub
  depths 0–3) = the library **hub-batching** selects from. Rewards are precomputed, so the scan
  re-scores rather than re-generates.
- `.../fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json` — the frozen
  promoted-fragment snapshot with recipes (1,600 fragments), required by the count-once cost model.

**Results**

- `./experiments/lsd_hubs/filter_ablation/results/scent_seh/filter_ablation_results.csv` — one row per
  (cell, strategy): cost, reward distribution, the diversity readouts (greedy **and** reward-blind Butina
  at both cutoffs), depth, qualified modes, duplicates suppressed, and both budget conventions. The
  `_results` suffix is load-bearing: the repo gitignores `*.csv` except `experiments/**/*_results.csv`,
  so a plain `filter_ablation.csv` would be silently left out of the commit.
- `.../filter_ablation.json` — run config + the 2×2 factorial, the pool-limited cells, and the
  unablated self-check.
- `.../filter_ablation.{png,pdf}` — the scan figure: 2 ablations (rows) × 4 readouts (columns).
- `.../filter_ablation_factorial.{png,pdf}` — the headline figure: the four arms side by side, showing
  how much of each delivered 300-molecule library is really distinct molecules — counted greedily
  (reward-ordered) and by Butina (reward-blind) at both cutoffs — and how much of it qualifies.

**Job Logs**

None — the scan ran interactively on the Balam login node in 34 s (pure CPU, no GPU/model/oracle, well
inside both the 15-minute interactive guideline and the login node's CPU-time limit).

## Relevant Versions

Branch `Hub-Analysis`, last commit `7d99b27` ("Diversity vs Rewards vs rxns/mode"). The files this entry
adds/changes are **not yet committed**:

- new: `experiments/lsd_hubs/filter_ablation/` (driver, README, `results/scent_seh/*`)
- new: `Logs/054_filter-ablation-reward-vs-diversity.md`
- modified: `glue/samplers/lsdflow/mode_select.py`, `experiments/lsd_hubs/campaign/run_campaign.py`,
  `experiments/lsd_hubs/README.md`, `docs/RESEARCH_CONTEXT.md`, `docs/REFACTOR_LOG.md`

Numbered `054`, not `053`: `053` is the hub-ordering ablation living on branch
`worktree-hub-order-ablation` (worktree `.claude/worktrees/hub-order-ablation`), which has not merged
into `Hub-Analysis` yet.

`[TODO — add commit hash after pushing]`

## Relevant Resources

**Sources**

- `docs/paper_planning/lsd-flow-publication-strategy.md` §2.4 (numerator defence — "currently missing":
  report reward and synthetic-depth distributions alongside the ratio) and §2.5 (the depth-0 degenerate
  optimum, and conditioning on a reward threshold as the one intervention that closes it).
- Logs/026 — the per-hub filter funnel: how many children each filter *kills* (reward gate dominates,
  dedup gentle). This entry is its complement: what the delivered library is made of.
- Logs/051 — the pool-side prerequisite (τ barely moves pool diversity below ~7) and the matched-N
  design constraint reused here.
- Logs/052 — the two-knob cost surface; source of the τ-invariance of best-candidate and of the
  operating-point numbers cross-checked below.
- Logs/033 — the count-once cost model, unchanged by this ablation (only the selection changes).
- Logs/034 — the sEH-proxy calibration that makes 5/6/7 the meaningful bars to report fractions over.
- Logs/037 — `free_frag` + pre-select-K, the within-hub child policy used throughout.
- `[bengio2021gflownet]` / `[koziarski2024rgfn]` — origin of the two-part mode definition being ablated.

**Packages**

- RDKit — ECFP (Morgan r=3/2048, no features/chirality), `BulkTanimotoSimilarity`, Butina clustering,
  `MurckoScaffold`, and canonical SMILES for duplicate suppression — via
  `glue/samplers/lsdflow/mode_select.py` and `validation/lsdflow/metrics/diversity.py`.
- matplotlib — the figure, in `filter_ablation_scan.py::plot`.
- No torch/dgl/gin, no GPU: the driver reads persisted pools and cached rewards only.

## Method

1. **Built the ablation through the existing seam, not a second code path.** Both strategy classes have
   always accepted a `mode_selector_factory`; no driver had ever passed one. Added the passthrough to
   `build_strategy` and a `RewardOnlyModeSelector` to the production selector module, so the ablation
   changes *only* the mode definition and leaves the count-once cost accounting untouched.

2. **Fixed the two ablation semantics deliberately.** *No diversity filter* keeps exact-duplicate
   suppression (on canonical SMILES) — the knob under test is the similarity criterion, not
   deduplication, and a library counting one molecule 300 times would be a strawman; the suppressed
   count is reported per cell. *No reward filter* drops the hard gate but keeps best-reward-first
   feeding, since that ordering is the strategy rather than the filter. Both selectors reject
   unparseable SMILES, so the arms differ only in the test under study.

3. **Verified the seam changes nothing when unused.** Re-ran `run_campaign.py` at the operating point
   after the edit and diffed against the pre-edit run: `summary.json` identical and both
   `curve_*.csv` byte-identical. Import-checked all ten `build_strategy` callers (`sweep_campaign`,
   `tau_similarity_surface`, `preselect_sweep`, `batch_size_distribution`, `dump_frontier_smiles`,
   `reconcile_t15`, `s3gfn_frontier`, `matrix16/gate_curve`, `matrix16/tau_curve_all_generators`, and
   `run_campaign` itself) — every one passes keyword arguments after `comps`, so a defaulted keyword
   is invisible to them. Unit-checked the new
   selector: reward-gate parity with the existing selector across `{7.0, None, −1.5}` × `{8.0, 7.0,
   6.9, NaN}` and both orientations; accepts a near-identical molecule; rejects a re-spelled duplicate,
   a below-gate molecule, an unparseable string, and an empty string.

4. **Confirmed the pools contain no NaN rewards** (29,997 records, 828,448 enum children), so the
   ungated arm cannot admit an undefined score — the one case where dropping the gate would change what
   a NaN means.

5. **Ran the scan** on the Balam login node:

   ```bash
   source ~/bin/rgfn-smoke-env.sh
   python experiments/lsd_hubs/filter_ablation/filter_ablation_scan.py \
       --analysis-dir  /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189 \
       --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
       --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
       --tag scent_seh --out-dir /scratch/markymoo/rgfn_runs/lsdflow/filter_ablation/scent_seh
   ```

   Defaults: τ ∈ {off, 4.0→8.0 by 0.5} at cutoff 0.5; cutoff ∈ {off, 0.3→0.9 by 0.1} at τ=7.0; plus the
   both-off corner = **18 cells × 2 strategies = 36 selection runs**, `--child-policy free_frag`
   `--prebuild-k 20` `--rank-by build_score` (the headline configuration), each run to a **300-mode**
   target. Artifacts copied from `$SCRATCH` into the repo afterwards.

6. **Read every cell two ways.** Primary: run to the 300-mode target, so all arms deliver the same
   library size and the quality readouts are comparable (the saturation problem `051` pins down).
   Secondary: the fixed-300-reaction prefix of the same curve (Case 1), where N is *not* matched and is
   reported per row. Cells that exhausted their library before the target are flagged `pool_limited`,
   drawn hollow, and excluded from the matched-N claims.

7. **Re-imposed the canonical definition on every delivered library** (reward ≥ 7.0 **and** greedy
   sphere exclusion at Tanimoto 0.5, best-reward-first) to get `qualified_modes`, and divided the
   reactions actually spent by it. The driver self-checks that the unablated library re-qualifies in
   full (300/300 for both strategies) — if it did not, the ablation seam would have perturbed the
   canonical path.

8. **Counted distinctness a second, reward-blind way, and aligned the two predicates first.** Butina
   clustering picks centres by neighbourhood density and never reads the reward, so it is the honest
   cross-check for an ablation of the reward filter (the same reason entry `051` runs it). One
   correction was needed before the two counts could be compared: greedy sphere exclusion rejects a
   molecule when similarity **> c**, while Butina treats two molecules as neighbours at distance
   **≤ 1 − c**, i.e. similarity **≥ c** — the rules disagree on exactly the boundary, and a library
   *built* at cutoff c piles pairs onto that boundary. Measured directly on the unablated library:
   **36 of its 44,850 pairs sit at exactly Tanimoto 0.500 and none exceed it**, so an unaligned Butina
   returned 266 of 300 and read as a diversity deficit that is not there. Nudging Butina's threshold by
   1e-9 (shrinking the neighbour radius just inside the boundary) makes its predicate "similarity > c"
   too, and the unablated library correctly reads 300/300. The shared metric in
   `validation/lsdflow/metrics/diversity.py` was **not** changed — the epsilon is applied at this
   driver's call site, so entry `051`'s numbers stand exactly as published.

## Results

**Scan cost:** 36 selection runs in **34 s** on the Balam login node (pools load once, 4.7 s: 26,069
candidates, 200 hubs, 1,600 promoted fragments with recipes); 3,535 distinct molecules fingerprinted
across the scan.

**Cross-check against the published operating point.** The Case-1 (fixed 300-reaction) readout of the
unablated cell gives hub **1.2245** vs best **3.1263** reactions/mode — identical to entry `052`'s
headline cell. The 300-mode convention used here gives 1.2167 vs 3.0967 on the same run.

### The 2×2 factorial at the operating point (τ=7.0, cutoff=0.5)

Every arm delivers a **300-molecule library**. "qualified" = members surviving the canonical definition
(reward ≥ 7.0 and mutual Tanimoto ≤ 0.5). "apparent gain" is how much *cheaper* reactions/mode looks
than the unablated arm; "true cost" is the ratio of reactions per qualified mode against the same
baseline.

"distinct" counts how many molecules the 300 really are, **two ways at each cutoff**: greedy sphere
exclusion fed best-reward-first (the paper's definition) / Taylor-Butina, which picks cluster centres
by neighbourhood density and **never reads the reward**. The pair is the load-bearing part of an
ablation whose subject *is* the reward filter — a diversity number whose own procedure is ordered by
reward could not settle the question.

**Hub batching**

| arm | rxn/mode | apparent gain | qualified | rxn/qualified | true cost | median R | frac ≥ 7 | R p10 | mean pairwise | distinct @0.7 greedy / **Butina** | distinct @0.5 greedy / **Butina** | median depth | hubs used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both filters on | 1.2167 | — | **300** (100%) | 1.217 | 1.00× | 7.300 | 1.000 | 7.038 | 0.1606 | 300 / **300** | 300 / **300** | 2 | 43 |
| reward filter off | 1.0767 | 1.13× | **27** (9%) | 11.963 | **9.83×** | 5.470 | 0.090 | 1.794 | 0.1370 | 300 / **300** | 300 / **300** | 1 | 5 |
| diversity filter off | 1.0833 | 1.12× | **39** (13%) | 8.333 | **6.85×** | 7.369 | 1.000 | 7.091 | 0.2977 | 153 / **144** | 39 / **31** | 2 | 5 |
| both off | 1.0700 | 1.14× | **9** (3%) | 35.667 | **29.31×** | 4.721 | 0.140 | 1.331 | 0.1899 | 263 / **257** | 117 / **104** | 1 | 2 |

**Best candidate**

| arm | rxn/mode | apparent gain | qualified | rxn/qualified | true cost | median R | frac ≥ 7 | mean pairwise | distinct @0.7 greedy / **Butina** | distinct @0.5 greedy / **Butina** | median depth | hubs used |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| both filters on | 3.0967 | — | **300** (100%) | 3.097 | 1.00× | 8.103 | 1.000 | 0.2022 | 300 / **300** | 300 / **300** | 4 | 254 |
| reward filter off | 3.0967 | 1.00× | **300** (100%) | 3.097 | 1.00× | 8.103 | 1.000 | 0.2022 | 300 / **300** | 300 / **300** | 4 | 254 |
| diversity filter off | 2.4300 | 1.27× | **50** (17%) | 14.580 | **4.71×** | 8.244 | 1.000 | 0.2650 | 164 / **155** | 50 / **43** | 4 | 164 |
| both off | 2.4300 | 1.27× | **50** (17%) | 14.580 | **4.71×** | 8.244 | 1.000 | 0.2650 | 164 / **155** | 50 / **43** | 4 | 164 |

**The reward-blind count agrees with the reward-ordered one everywhere, and is slightly stricter** —
within 6% at cutoff 0.7 (153 vs 144; 164 vs 155) and 11–21% at 0.5 (39 vs 31; 50 vs 43) — the same
"Butina is uniformly a little stricter" relationship entry `051` found on the pools. So the collapse is
a property of the molecules, not of feeding a greedy selector in reward order. It also cuts the other
way, in our favour: with the reward gate off, **both** counts still say 300 distinct molecules — the
gate's removal costs quality, not diversity, and the two filters are close to orthogonal in what they
defend.

Best-candidate's reward-off column is identical to its baseline in **every** readout — not approximately,
exactly — and its both-off column is identical to its diversity-off column, i.e. the gate is completely
inert for it at this budget.

### What the ablations do to the comparison between strategies

| arm | advantage on reactions/mode | advantage per QUALIFIED mode |
|---|---|---|
| both filters on | 2.545× | 2.545× |
| reward filter off | 2.876× | **0.259×** (hub-batching 3.9× *worse*) |
| diversity filter off | 2.243× | 1.750× |
| both off | 2.271× | **0.409×** |

Removing the reward gate makes hub-batching look *better* on the reported metric and inverts its real
standing; the same removal leaves best-candidate untouched.

### Fixed-300-reaction view (Case 1; library size differs per arm, so it is stated)

| strategy | arm | "modes" at 300 rxn | of which qualified | rxn / qualified |
|---|---|---|---|---|
| hub batching | both on | 245 | 245 | 1.2245 |
| hub batching | reward off | **277** | 27 | 11.111 |
| hub batching | diversity off | 275 | 39 | 7.692 |
| hub batching | both off | 279 | 9 | 33.333 |
| best candidate | both on | 95 | 95 | 3.1263 |
| best candidate | reward off | 95 | 95 | 3.1263 |
| best candidate | diversity off | 119 | 28 | 10.643 |
| best candidate | both off | 119 | 28 | 10.643 |

At a fixed bench budget, the ablated arms deliver *more* library members (277 vs 245) — which is exactly
the exploit the filters exist to block.

### The reward-bar scan (cutoff fixed at 0.5), hub batching

| τ | off | 4.0 | 5.0 | 6.0 | 6.5 | **7.0** | 7.5 | 8.0 |
|---|---|---|---|---|---|---|---|---|
| reactions/mode | 1.0767 | 1.0833 | 1.0933 | 1.1233 | 1.1433 | **1.2167** | 1.5100 | 2.1923 † |
| median reward | 5.470 | 5.599 | 6.288 | 6.774 | 7.052 | **7.300** | 7.761 | 8.104 † |
| reward p10 | 1.794 | 4.330 | 5.253 | 6.153 | 6.593 | **7.038** | 7.554 | 8.022 † |
| fraction ≥ 7 | 0.090 | 0.153 | 0.243 | 0.377 | 0.553 | **1.000** | 1.000 | 1.000 † |
| mean pairwise | 0.1370 | 0.1631 | 0.1597 | 0.1557 | 0.1583 | **0.1606** | 0.1742 | 0.2031 † |
| qualified modes | 27 | 46 | 73 | 113 | 166 | **300** | 300 | 130 † |
| rxn / qualified | 11.963 | 7.065 | 4.493 | 2.982 | 2.066 | **1.217** | 1.510 | 2.192 † |
| hubs used | 5 | 7 | 10 | 17 | 22 | **43** | 105 | 96 † |

† τ=8.0 is **pool-limited**: hub-batching exhausted its 200-hub library at 130 modes, so that column is
measured over a shorter, smaller library and is not matched-N. It is the same strict corner entry `052`
hatched. Best-candidate is flat at 3.0967 / median 8.103 / 300 qualified across this entire row.

Library self-similarity is flat (0.153–0.163) from τ=4 to τ=7 and rises only at 7.5–8.0 — the delivered
libraries reproduce, independently, the flat-then-collapse shape entry `051` measured on the pools.

### The diversity-cutoff scan (τ fixed at 7.0), hub batching

| cutoff | off | 0.9 | 0.8 | 0.7 | 0.6 | **0.5** | 0.4 | 0.3 |
|---|---|---|---|---|---|---|---|---|
| reactions/mode | 1.0833 | 1.0833 | 1.0833 | 1.1000 | 1.1367 | **1.2167** | 1.6133 | 2.3469 † |
| mean pairwise | 0.2977 | 0.2967 | 0.2958 | 0.2108 | 0.1854 | **0.1606** | 0.1504 | 0.1417 † |
| modes @0.7 | 153 | 154 | 164 | 300 | 300 | **300** | 300 | 98 † |
| modes @0.5 (= qualified) | 39 | 39 | 39 | 80 | 121 | **300** | 300 | 98 † |
| rxn / qualified | 8.333 | 8.333 | 8.333 | 4.125 | 2.818 | **1.217** | 1.613 | 2.347 † |
| median reward | 7.369 | 7.361 | 7.295 | 7.288 | 7.349 | **7.300** | 7.314 | 7.329 † |
| duplicates suppressed | 67 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| hubs used | 5 | 5 | 5 | 10 | 18 | **43** | 129 | 86 † |

(`duplicates suppressed` is *not applicable* wherever the Tanimoto test is on — it subsumes exact
duplicates — so the CSV leaves those blank rather than writing a zero that would read as a measurement.
Best-candidate's cutoff-off cell reports a genuine **0**: its pool is deduplicated by molecule upstream,
so no repeat can arise.)

† cutoff 0.3 is **pool-limited** (98 modes). Median library reward is flat across the whole row
including "off" (7.29–7.37), i.e. the diversity requirement neither buys nor costs reward — the two
filters are close to orthogonal in what they protect.

Even with exact duplicates still suppressed (67 of them were, at cutoff off), the delivered
300-molecule library contains only **39** molecules distinct at Tanimoto 0.5 and **153** at 0.7.

### Two caveats on reading these numbers

1. **The minimum of "reactions per qualified mode" sits at the operating point partly by construction**,
   because qualification is *defined* at τ=7.0 / cutoff 0.5. What is not automatic — and is the actual
   finding — is the **magnitude** of the penalty (6.8–29×), the fact that it falls almost entirely on
   hub-batching, and that the apparent saving is only 12–14% in the first place.
2. **One checkpoint, one reward, one draw**, the same limitation entries `051` and `052` carry. The
   `pool_limited` cells (hub-batching at τ=8.0 and at cutoff 0.3) are flagged in the CSV/JSON and drawn
   hollow, and no matched-N claim rests on them.
