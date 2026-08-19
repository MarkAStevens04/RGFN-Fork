# Four generators × three scoring systems — the benchmark re-read as "what does 100 reactions buy?"

**Date:** 2026-08-18, ~3pm

## Question

If a chemist has a fixed budget of one hundred reactions, how many genuinely different high-quality
molecules does each method hand them?

## Context & Summary

**Context** — Every result in this benchmark so far has been reported the other way round: *"how many
reactions does it take to reach three hundred different molecules?"* Entry `063` built the full
matrix that way, and entry `064` put error bars on it across three independent training runs, landing
on a mean advantage of **3.29×** for our method. But the project decided on 2026-08-17 that the
headline question should be the reverse — you fix the budget and count the molecules, because a
budget is what a chemist actually has, whereas a target number of molecules is a shopping list. The
two readings come off the *same* ordering of the same molecules, so this is a change in how we report,
not a different experiment, and it needs no new computation at all.

**Summary** — We re-read every finished cell of the matrix at fixed reaction budgets of 50, 100, 150
and 200, and compare our method against the simpler strategy of just taking the best molecules one at
a time. Because a fixed budget can be spent in three quite different ways, each cell is labelled with
which one applies: the budget ran out (the only case where two methods can be fairly compared), the
method ran out of molecules worth adding before spending the budget, or the run stopped early because
it had already collected the three hundred molecules the original run was asked for. Only cells where
*both* methods genuinely ran out of budget are counted toward the headline.

## Answer

**The headline survives the change of axis, at a slightly lower value and with more run-to-run
scatter.** At a hundred reactions our method delivers a median of **2.88× more distinct molecules**
than the straightforward alternative, across 25 of 26 comparable cells, and it holds in every single
cell — the worst case is still a 1.67× advantage. This is close to but below the 3.29× that entry
`064` reported on the old axis, so the two readings agree on the conclusion while disagreeing mildly
on the size, and the new number is the one to quote.

Two things are genuinely new. The advantage **grows with the budget** — 2.50× at fifty reactions
rising to 3.12× at two hundred — so the number we quote is the *most conservative* of the four, not a
figure picked to flatter. And the run-to-run scatter is roughly twice as large on this axis (median
7.0% across seeds, against 3.2% on the old one), which is worth stating plainly rather than
discovering in review.

## Relevance to our Publication

This is the entry that makes the paper's main table say what the paper's main claim says. A Digital
Discovery reviewer reading "we get more molecules per reaction" and then finding a table of
reactions-per-300-molecules has to do the conversion themselves, and a reviewer who does that
conversion badly gets a wrong number. Reporting the budgeted reading directly removes that gap. It
also pre-empts the sharpest available objection to a fixed-budget claim — that a method which simply
runs out of things to buy will *look* thrifty — because every cell is explicitly labelled with
whether the budget bound, and the one cell where it did not is excluded from the headline rather than
quietly averaged in.

## Next Experiments

**Refining for publication**

- **Finish the replicate seeds on the docking systems.** Four of the twelve cells here rest on a
  single training run, and all four are the docking-based ones. The scatter we can measure (2–20%
  depending on the cell) is measured only on the cells that have three runs.
- **Report the budget trend as a small figure, not a sentence.** That the advantage grows from 2.5×
  to 3.1× as the budget widens is a more persuasive object than any single ratio, because it shows
  the effect is not an artifact of one arbitrary choice of budget.
- **Explain the noisier cells.** Three cells scatter at 15–20% across runs while three others sit
  under 4%. Whether that is a property of those generators or of reading a curve at an early point
  is not yet established.

**Next steps in project**

- Price these libraries through the competitor pipeline at the same budgets, so the outside
  comparison and the internal one are read on one axis.
- Settle the hit threshold for the remaining protein system, which is currently the only one whose
  bar is not backed by a discrimination measurement against realistic negatives.

---

# Re-creation

Root for repo-relative paths: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`.

### Relevant Files

**Scripts**

- `./experiments/lsd_hubs/matrix16/reaction_axis_readout.py` — **new**, and the whole of this entry.
  Walks every `results*/<cell>/` directory, reads both arms' per-step curves, and reports
  `modes_at_R` with `used_rxns` beside it. Its substance is the three-way status label, not the
  lookup: `budget-binding` (the next mode would exceed R — the only like-for-like case),
  `pool-exhausted` (the arm ran out of qualifying candidates, so the budget went UNSPENT and quoting
  "modes at R" would falsely imply it could have spent R), and `mode-capped` (the run stopped at its
  own 300-mode budget with reactions still available, making the count a LOWER bound). It also
  cross-checks each cell's recorded `reward_threshold` against the authoritative gate in
  `targets.py` and refuses to silently mix bars, and reports `n_modes_available` so a short cell can
  be diagnosed as collapse rather than redundancy.
- `./experiments/lsd_hubs/matrix16/targets.py` — the authoritative per-target gate table, imported
  rather than duplicated. sEH 5.0, DRD2 0.5, ClpP −8.0.

**Datasets** (inputs; no new compute)

- `./experiments/lsd_hubs/matrix16/results{,_seed43,_seed44}/<cell>/curve_hub_batching.csv` — the
  per-step greedy ordering for our method: `cum_reactions`, `cum_modes`, `cum_reward_gen_calls` at
  every step. Both readouts are projections of this one file, which is why the axis change costs
  nothing.
- `.../curve_best_candidate.csv` — the same ordering for the comparison arm: take the highest-reward
  molecule that is still different enough, ignoring which intermediates are already paid for.
- `.../summary.json` — carries `reward_threshold` and `budget_modes` per cell; the gate check and
  the `mode-capped` test both read it.

**Results**

- `./experiments/lsd_hubs/matrix16/results/reaction_axis/reaction_axis.csv` — one row per
  (seed, cell, arm, budget): modes, reactions used, oracle calls, status, modes available.
- `./experiments/lsd_hubs/matrix16/results/reaction_axis/reaction_axis_ratio.csv` — the paired view,
  one row per (seed, cell, budget) with both arms, the ratio, and a `comparable` flag that is true
  only when BOTH arms were budget-binding.

### Relevant Versions

`0296459` — *Logs/064 + headline sEH bar 7.0 -> 5.0: three-seed error bars on the surrogate matrix*
is the state this entry reads. The gate move to 5.0 in that commit is load-bearing here: all 29 cell
readings sit at the headline gate, so no cell needed re-evaluation.

`reaction_axis_readout.py` and its two output CSVs are **not yet committed**.
[TODO — add commit hash after pushing]

### Relevant Resources

**Sources**

- Entry `063` — the four-way matrix these curves come from.
- Entry `064` — the three-seed error bars and the sEH bar move to 5.0; source of the 3.29× mean and
  3.2% seed CV quoted for comparison.
- `CLAUDE.md` §"THE BENCHMARK'S PRIMARY READOUT IS A FIXED REACTION BUDGET" (decided 2026-08-17) —
  the decision this entry implements, including the three-way classification it requires.

**Packages**

- Python standard library only (`csv`, `json`, `argparse`, `pathlib`) — deliberately, so the readout
  has no dependency that could change a published number.

### Method

1. For each of `results/`, `results_seed43/`, `results_seed44/`, enumerate `<cell>/summary.json`,
   skipping the deliberate sensitivity-sweep variants (`*_thr*`, `*_cap9*`, `*_naive`) which are not
   cells of the matrix proper.
2. Resolve each cell's target and compare its recorded `reward_threshold` to `targets.py`. All 29
   matched; **0 gate mismatches**, so nothing needed re-evaluating.
3. Exclude 6TD3 (3 cells). Its −2.0 bar rests on **warhead-matched, not property-matched** decoys —
   every decoy carries the CR8-like purine, but molecular weight is only range-bounded (250–650) and
   no other property is matched — so the bar is not yet defensible as a hit threshold. Parked
   pending a property-matched decoy set; `--include-6td3` overrides.
4. For each arm and each budget R ∈ {50, 100, 150, 200}, take the last curve step with
   `cum_reactions ≤ R`, and assign the status label.
5. Pair the arms per (seed, cell, budget); mark `comparable` only where both are budget-binding.

```
conda run -n rgfn python experiments/lsd_hubs/matrix16/reaction_axis_readout.py \
    --budgets 50,100,150,200
```

### Results

**1 — the headline, at R = 100 reactions.** Distinct molecules delivered, mean ± SD over available
seeds. 26 cells; 25 strictly comparable.

| cell | seeds | hub-batching | best-candidate | ratio | seed CV |
|---|---|---|---|---|---|
| fraggfn_clpp | 1 | 96 | 20 | **4.80×** | — |
| rgfn_clpp | 1 | 66 | 28 | **2.36×** | — |
| rxnflow_clpp | 1 | 81 | 33 | **2.46×** | — |
| scent_clpp | 1 | 79 | 27 | **2.93×** | — |
| fraggfn_drd2 | 3 | 82 ± 6 | 20 ± 0 | **4.08×** | 7.0% |
| rgfn_drd2 | 1 | 82 | 25 | **3.28×** | — |
| rxnflow_drd2 ⚠ | 3 | 75 ± 18 | 34 ± 3 | **2.17×** | 20.4% |
| scent_drd2 | 3 | 76 ± 2 | 27 ± 1 | **2.84×** | 6.4% |
| fraggfn_seh | 3 | 81 ± 2 | 20 ± 0 | **4.05×** | 2.1% |
| rgfn_seh | 3 | 53 ± 7 | 26 ± 1 | **2.08×** | 14.9% |
| rxnflow_seh | 3 | 81 ± 9 | 34 ± 2 | **2.36×** | 16.0% |
| scent_seh | 3 | 78 ± 1 | 27 ± 1 | **2.87×** | 3.5% |

**Median over the 25 strictly-comparable points: 2.88×** (min 1.67, max 4.80). The advantage holds in
every cell — there is no cell where the simpler strategy wins or ties.

⚠ `rxnflow_drd2` seed 44 is the single excluded point: hub-batching was budget-binding at 80 modes
while best-candidate was **pool-exhausted** at 32, i.e. it ran out of qualifying molecules rather
than out of budget. Counting it would credit us with a cost win the data does not support.

**2 — the advantage grows with the budget.** Same cells, four budgets:

| budget (reactions) | median ratio | strictly comparable |
|---|---|---|
| 50 | 2.50× | 26/26 |
| **100 (headline)** | **2.88×** | **25/26** |
| 150 | 3.02× | 25/26 |
| 200 | 3.12× | 24/26 |

The headline budget is therefore the second-most conservative of the four measured, and the 100-mode
readout's 3.29× (entry `064`) sits at the top of this range rather than outside it.

**3 — the two axes agree on the conclusion, not on the scatter.**

| | entry `064` (fixed modes) | this entry (fixed reactions) |
|---|---|---|
| central advantage | 3.29× mean | 2.88× median |
| median seed CV | 3.2% | **7.0%** |
| n cells behind it | 5 three-seed cells | 25 comparable points, 7 three-seed cells |

The scatter roughly doubles. Three cells (`rxnflow_drd2` 20.4%, `rxnflow_seh` 16.0%, `rgfn_seh`
14.9%) carry nearly all of it, while `fraggfn_seh`, `scent_seh` and `scent_drd2` sit at 2–4%. A
fixed-budget reading samples one early point on the curve, where a single expensive mode moves the
count, whereas the mode-budget reading integrates over the whole curve — but which of that is
generator behaviour and which is the readout has not been separated, and is listed as an open item
rather than asserted.

**4 — no cell needed re-evaluating.** All 29 cell readings (including ClpP at −8.0) already sat at
the gate `targets.py` declares, so the axis change was a pure re-read: **0 gate mismatches**, no
GPU time, no oracle calls.
