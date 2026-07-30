# `filter_ablation/` — what does each half of the mode definition buy?

**One question:** our cost metric counts a molecule only if it (a) clears a reward bar and (b) is
Tanimoto-dissimilar from everything already kept. Drop one of those two filters — what happens to the
library we deliver, and to the cost we report for it?

**Why it belongs here.** `reactions/mode` is a ratio we also happen to win on, which is exactly the
shape reviewers are trained to distrust. `docs/paper_planning/lsd-flow-publication-strategy.md`
§2.4–2.5 names the two ways a ratio like this can be gamed from the denominator side, and each of our
two filters blocks one of them:

| filter | the exploit it blocks |
|---|---|
| reward gate (`reward >= τ`) | "cheap because the molecules are weak" — in the limit, §2.5's depth-0 catalog optimum: buy N diverse catalog compounds, spend zero reactions |
| diversity filter (Tanimoto `<= cutoff`) | "cheap because it's the same molecule 300 times" — one hub, one scaffold, 300 near-copies |

Logs/026 ([`../dropoff/`](../dropoff/)) already measured how many children each filter *kills* per hub
(the reward gate dominates the funnel; the dedup is gentle). This sub-dir measures the other half of
the question — what the delivered **library** is actually made of when a filter is removed — so the
inclusion of both filters is *demonstrated* rather than asserted. Logs/054.

## What it measures

Removing a filter is the permissive endpoint of that filter's own knob, so the ablation is **scanned**,
not asserted: two 1-D scans crossing at the headline operating point (τ=7.0, cutoff=0.5), plus the
both-off corner.

```
reward-filter scan:     τ      ∈ {off, 4.0 … 8.0}   at cutoff = 0.5
diversity-filter scan:  cutoff ∈ {off, 0.9 … 0.3}   at τ = 7.0
corner:                 both off
```

Each cell runs **both** selection strategies (`HubBatchingStrategy` / `BestCandidateStrategy`, the same
`build_strategy` `run_campaign.py` uses, so a cell is the same computation as a `run_campaign`
invocation) and reports four things about the library it delivered:

| readout | what it says | direction |
|---|---|---|
| **reactions per mode** | the headline cost, exactly as the campaign counts it — what relaxing a filter *appears* to buy | lower = cheaper |
| **library reward** (median, p10, fraction over 5/6/7) | the quality the reward gate protects (§2.4's numerator guard) | higher = better |
| **library self-similarity** (mean pairwise Tanimoto) plus **how many distinct molecules the library really is**, counted two ways at each cutoff — greedy sphere exclusion (reward-ordered, the paper's definition) and Butina (**reward-blind**) — with Murcko scaffolds as a fingerprint-independent third view | the quality the diversity filter protects | lower self-similarity / more distinct = more diverse |
| **reactions per QUALIFIED mode** | the honest cost: re-impose the canonical definition (reward ≥ 7 **and** mutual Tanimoto ≤ 0.5) on whatever was delivered, and divide by what survives | lower = cheaper |

Nested synthetic depth rides along (§2.4 asks for depth as well as reward): SCENT's fully nested
`num_reactions` for a sampled candidate, `hub.depth + 1` for an enumerated child — the same scale, so
the two strategies are comparable.

**The design constraint that makes it valid:** mode and scaffold counts saturate with set size, so every
arm is run to the **same 300-mode target** (`--budget-modes`) rather than a fixed reaction budget —
matched N by construction, the same precaution Logs/051 takes by subsampling. The fixed-reaction view
(Case 1) is read off the same curve prefix and reported alongside, where N is *not* matched and is
stated per row. Cells that run out of library before reaching the target are flagged `pool_limited`,
drawn hollow, and must not be compared with the rest.

**Two ablation semantics, both deliberate, both reported:**

- *no diversity filter* → `RewardOnlyModeSelector` (`glue/samplers/lsdflow/mode_select.py`): keeps
  everything clearing the gate, but never the same canonical SMILES twice. The knob under test is the
  **similarity** criterion, not deduplication — a library that counted one molecule 300 times would be
  a strawman. `duplicates_suppressed` reports how often that mattered.
- *no reward filter* → `reward_threshold=None`. The strategies still feed candidates
  best-reward-first: that ordering is the *strategy*, not the filter.

Both are reached through the strategies' `mode_selector_factory` seam, so the campaign's cost
accounting is untouched and the unablated cell stays bit-identical to a plain `run_campaign.py` run
(the driver self-checks this: every member of the unablated library must re-qualify).

**Why the reward-blind count is reported next to ours.** Our mode count is greedy sphere exclusion fed
*best-reward-first* — order-dependent, and here the order is the very variable one arm ablates. Butina
picks centres by neighbourhood density and never sees the reward, so it settles "did the library really
collapse?" without that circularity. One alignment is needed before the two are comparable: greedy
rejects `similarity > c`, Butina merges at `similarity >= c`, so they disagree on exactly the boundary —
and a library *built* at cutoff `c` piles pairs onto it (36 pairs sit at exactly 0.500 in the unablated
library, none above). The driver nudges Butina's threshold by `1e-9` so both use `> c`; without it the
unablated library reads 266 of 300 and looks deficient when it is not. The shared metric in
`validation/lsdflow/metrics/diversity.py` is untouched — the epsilon lives at this driver's call site,
so Logs/051's published numbers are unaffected.

## Run

Pure CPU — re-scores the cached enumeration (rewards already computed), no GPU, model, or oracle.
The whole scan is ~35 s on a login node.

```bash
source ~/bin/rgfn-smoke-env.sh
python experiments/lsd_hubs/filter_ablation/filter_ablation_scan.py \
    --analysis-dir  /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189 \
    --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
    --snapshot /scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
    --tag scent_seh
```

`--plot-only` redraws the figure from the committed CSV. For a lower-is-better reward (docking
differentials) pass `--higher-is-better false` and docking-scale `--taus` / `--qualify-tau`.

## Files

- `filter_ablation_scan.py` — the driver (knob scans, readouts, figure).
- `results/<tag>/filter_ablation_results.csv` — one row per (cell, strategy): cost, reward, diversity,
  depth, qualified modes, both budget conventions. (The `_results` suffix is what makes it committable:
  the repo gitignores `*.csv` except `experiments/**/*_results.csv`.)
- `results/<tag>/filter_ablation.json` — run config + the 2×2 factorial summary, the pool-limited
  cells, and the unablated self-check.
- `results/<tag>/filter_ablation.{png,pdf}` — the scan figure (2 ablations × 4 readouts).
- `results/<tag>/filter_ablation_factorial.{png,pdf}` — the headline figure: the four arms side by
  side — of the molecules each library hands over, how many are distinct (greedy **and** Butina, at
  both cutoffs) and how many qualify.

## Finding (SCENT × sEH, 200-hub enumeration)

Both filters are load-bearing, and **removing either one costs hub-batching more than it costs the
baseline** — dropping the reward gate makes hub-batching look 1.13× cheaper while 91% of its library
falls below the bar, and dropping the diversity filter makes it look 1.12× cheaper while the
300-molecule library collapses to 39 distinct molecules (**31 by the reward-blind count**, so it is not
an artifact of counting in reward order). Per molecule actually worth keeping, the two ablations cost
9.8× and 6.8× respectively (29× with both off). The two filters defend near-orthogonal things: with the
reward gate off the library is still 300/300 distinct by **both** counts, and with the diversity filter
off the median reward is unchanged. Best-candidate is *exactly* invariant to the reward gate at this
budget (every readout identical from τ=off to τ=8), the exact-limit version of the τ-invariance in
Logs/052. Full numbers in Logs/054.
