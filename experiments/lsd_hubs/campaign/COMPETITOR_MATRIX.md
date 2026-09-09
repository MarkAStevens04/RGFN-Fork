# The competitor matrix — where the numbers are, and how to quote them

**State as of 2026-09-08: 106 of 108 cells complete.** Merged to `Hub-Analysis` at `8ca3cd9`
(originated on `worktree-fraggfn-stage2`). Full narrative in [`Logs/079`](../../../Logs/079_reprice-capped-rows-and-coarse-ladder.md);
this file is the operational summary for anyone comparing against these numbers.

This is the **competitor** half of the benchmark: 6 generators × 3 targets × 3 seeds × 2 pools.
Our own arm (hub-batching / LSD-Flow) is NOT here — it lives under
`$SCRATCH/rgfn_runs/lsdflow_sparrow/bc_sb*` and was not audited in this pass, so do not assume the
two halves share conventions until you have checked.

| | |
|---|---|
| generators | s3gfn, reinvent, saturn, tango, fraggfn, synformer |
| targets | seh, drd2, clpp — **6TD3 is deliberately paused** (`docs/RESEARCH_CONTEXT.md:669`) |
| seeds | 42, 43, 44 |
| pools | naive (top-500 by reward) and pruned (500 mutually distinct) |
| arms | greedy (prices a fixed list) and SPARROW-Batching (chooses the list) |

---

## Re-audit before trusting anything

```bash
python experiments/lsd_hubs/campaign/audit_matrix_coverage.py
```

Prints completeness per stage (pool / routes / greedy@R=100 / SB@R=100). It requires a **readable
R=100 row**, not a file that exists — a coarse ladder can write a CSV containing only its header, and
an "exists and non-empty" check passes that.

---

## Quote rules — each of these has already produced a wrong number

1. **`n_targets_priced`, not `n_modes`.** `n_modes` is the REQUEST, `n_targets_priced` the DELIVERY.
   They are equal on all 88 multiaiz cells and diverge on **all 19 SynFormer cells**, where delivery
   is 73-93% of request. Quoting the request overstates SynFormer by up to 27%.
2. **`cost_kept_rxns`, not `used_rxns`, outside the budget-binding regime.** For a `pool-exhausted`
   cell the unspent budget is not a choice, and the SB arm's `used_rxns` inflates. See CLAUDE.md.
3. **Never quote a `time_capped` row as an optimum.** It is a LOWER bound, so it flatters us. Four
   cells cannot be certified at any gap tried (1e-7, 1e-2, 1e-6); their best incumbents are in
   `$SCRATCH/rgfn_runs/lsdflow_sparrow/results/UNCERTIFIED_R100_BOUNDS.json`. Two of those main CSVs
   UNDERSTATE the competitor — `tango_clpp_44_pruned` holds 8 where 10 is measured,
   `saturn_seh_44` holds 4 where 5 is.
4. **Check the `gap_rel` column.** Re-solved R=100 rows carry `gap_rel = 0.01`; every other row is
   SPARROW's `1e-07`. On 70-92-mode rows a 1% gap is ±1 and immaterial; on a 3-14-mode cell it is
   about a quarter of the answer, and there a relaxed-gap `Optimal` can be BELOW a capped incumbent.
5. **Never mix the two axes** (modes-at-100-reactions vs reactions-for-100-modes) or two ladder
   resolutions in one table. A co-agent already lost a result to the second.

---

## What budget are these cells at? NOT 10,000 oracle calls

Stage 2 equalises **pools** (500 modes), not oracle calls, so the oracle-call axis is uncontrolled —
and uncontrolled in a generator-correlated way. Measured from each cell's
`$SCRATCH/rgfn_runs/stage2/<cell>/upsample_log.json` (`newly_sampled`, on top of Stage 1's 10,048):

| generator | total oracle calls |
|---|---|
| reinvent / saturn / tango | 12,018 - 12,048 |
| fraggfn | 12,047 - 14,046 |
| s3gfn | 10,048 - 18,048 |

Spread **1.80x** (10,048 - 18,048); total Stage-2 surcharge across 45 cells **122,411 calls**. The
largest surcharges are all S3-GFN's — `seh 43` +8,000, `drd2 42` +7,308, and the three ClpP cells
+6,300-6,500 each, all of which STALLED and still did not reach 500 modes.

**So these are not "10,000 oracle call" results and must not be placed in a column that claims to
be.** The honest label is *"arm-A training plus a measured Stage-2 surcharge of 2,000-8,000"*, with
the 1.80x spread and its S3-GFN concentration stated rather than averaged away.

**This is a LABELLING obligation, not a fairness hole — an earlier draft of this file had the
direction backwards.** Within the competitor field the surcharge does tilt things: S3-GFN's edge over
Saturn and TANGO is partly bought with up to 1.8x their oracle calls, and that is worth stating. But
the inference-time oracle axis is uncontrolled for *everyone*, and **we are by far its heavier
user** — our own campaign's per-cell `cum_reward_gen_calls` runs to **447,000-791,503** in the
budget-scale runs (`results/budget_scale_ours_*/curve_hub_batching.csv`), i.e. 50-100x the
competitors' entire Stage-2 surcharge. Equalising this axis is not the fix; the project deliberately
demoted the fixed-MODE readout that equalising would force. Measure it and report it per arm.

**Do not compute the surcharge from `rounds[].asked`.** That field is the CUMULATIVE escalating
request per round (4000, then 8000, then 12000), so summing it double-counts — it reports 24,000 for
a cell whose distinct set grew by 1,216. Use the top-level `newly_sampled`; `free_from_training` and
`distinct_scored` are the other honest fields.

**Consequence for any budget-matched comparison:** the redundancy and route-sharing mechanisms below
are properties of the CONVERGED pool, so both are budget-dependent in principle. Nothing here has
been measured at a second budget. If you are comparing against a run at a different oracle budget,
measure the greedy→SB uplift on both rather than assuming it transfers — it is free, since both arms
already run for every cell.

---

## Reading the results tree

Root: `$SCRATCH/rgfn_runs/lsdflow_sparrow/`

| path | holds |
|---|---|
| `multiaiz_pools/<tag>_N*/` | `pool.smi`, `pool_scores.csv`, `multiaiz_routes.json`, `discovery_timing.json` |
| `results/<tag>_saturation/summary.json` | the pre-flight: `n_distinct_above_gate`, `modes_in_largest_prefix` (top-500 prefix, capped at 500 — NOT a headroom measure), `modes_available_whole_set` (what the pruned pool draws from) |
| `results/<tag>_greedy_N*/greedy_frontier.csv` | greedy arm, priced on the MODE axis |
| `results/<tag>_select_N*/select_frontier.csv` | SPARROW arm, priced on the REACTION axis |
| `results/UNCERTIFIED_R100_BOUNDS.json` | the four cells that cannot be certified |
| `greedy_csv_backup_20260906.tar.gz` | 330 greedy CSVs as they were before the re-pricing |

**`<tag>` = `<gen>_<target>_seed<N><suffix>[_pruned]`.** The suffix is `_stage2` for most cells, `""`
for SynFormer (no Stage 2), and `_stage2fix` for s3gfn drd2 seeds 43/44 — which marks **which cells
stalled in Stage 2**, not a second pipeline. Seed 42 reads `target-reached` and needed no re-run.

**Glob the size, never assume `_N500`.** A pool-limited cell writes `_N<actual>`
(`synformer_drd2_seed44_pruned` is `_N495`), and a hard-coded `_N500` reports a complete cell as
missing.

---

## The two cells that are not gaps

`s3gfn:drd2:42/naive` and `saturn:seh:42/naive` have 2-byte route files and empty result directories.
That is a **measured zero, not a crash**: `discovery_timing.json` records 500 targets over 4.9 h and
10.3 h at 35 and 74 s/target with `returncode 0`. Their reward-ranked top-500 is entirely
unsynthesizable while the same cell's diverse 500 prices 47 modes. Report them; do not re-run them.

---

## What the matrix says, in one table

Modes at R=100, seeds 42/43/44, delivery-quoted. DRD2 omitted — S3-GFN routes almost nothing there
([075] traced it to an amine absent from our ZINC stock).

| target | GREEDY naive | GREEDY pruned | SB naive | SB pruned |
|---|---|---|---|---|
| **sEH** S3-GFN | 45/50/75 | 40/60/70 | 56/38/49 | **98/100/100** |
| **sEH** REINVENT | 30/30/35 | 30/30/35 | **61/47/83** | 80/70/92 |
| **ClpP** S3-GFN | 40/40/40 | 40/40/45 | 45/43/42 | 64/57/62 |
| **ClpP** REINVENT | 30/35/30 | 30/35/30 | **81\*/69/71\*** | **86/74\*/72\*** |

`*` = still capped, i.e. a lower bound, so certifying can only raise REINVENT.

**The SELECTOR decides the winner, not the pool.** Greedy favours S3-GFN in 4 of 4 pool×target
combinations with disjoint three-seed bands; SPARROW favours REINVENT in 3 of 4 — all but sEH/pruned,
which is exactly where S3-GFN sits at the 1.0 rxn/mode arithmetic ceiling and cannot be beaten. So
the arm CLAUDE.md designates to carry the headline is the arm the competitor usually wins on. Say it
plainly rather than quietly preferring greedy.

Two measured mechanisms sit under that:
* **pool axis — redundancy.** Share of a reward-ranked top-500 that is mutually dissimilar:
  Saturn/TANGO 2-4%, S3-GFN 39-58%, REINVENT 59-87%, FragGFN 87-100%. The pruning gain is close to
  the ratio of distinct molecules in the two pools (median deviation 5%), so it is largely
  arithmetic, not a subtle interaction.
* **selector axis — route sharing.** Mean greedy→SB uplift: REINVENT 1.86×/2.57× (naive/pruned),
  S3-GFN 1.01×/1.65×, FragGFN ~1.2×, Saturn 0.17×/0.36×, TANGO 0.27×/0.22×. Below 1.0× is not a
  defect: on a collapsed pool SPARROW pursues its own objective and does not chase mode count.

---

## Traps that already cost time here

| trap | looked like | detector |
|---|---|---|
| coarse mode ladder | a low number, or a **header-only CSV** when the pool cannot reach the first rung | first rung in the CSV; require a readable R=100 row |
| two-arm launcher | an SB-only backfill silently re-priced greedy back to coarse **hours later**; both jobs logged success | audit first rungs tree-wide, not job logs. `RUN_GREEDY=0` now prevents it |
| `_longsolve` split | a certified row existed but the main CSV still held the capped one — my own audit requoted the stale value | merged via `merge_longsolve_rows.py`; prefer the merged file |
| pool rebuild mid-flight | the chain and the native launcher REBUILD pools, and `build_s3gfn_pools.py` has since gained the NaN gate | re-price with `submit_reprice_cached.sh`, which reads cached artifacts and refuses if they are absent |

---

## Tools added in this pass

| script | does |
|---|---|
| `audit_matrix_coverage.py` | completeness per stage across all 108 cells |
| `submit_reprice_cached.sh` | re-price from CACHED routes, changing one solver knob. `ARM=sb` re-solves a capped budget row, `ARM=greedy` re-prices on a ladder; `ROUTE_SOURCE=external` for SynFormer. Refuses rather than rediscovering |
| `merge_longsolve_rows.py` | merges certified R=100 rows into the main CSV and appends `gap_rel`. Refuses a still-capped row, and a changed mode count without `--allow-change` |
| `write_uncertified_bounds.py` | regenerates `UNCERTIFIED_R100_BOUNDS.json` |
| `_resolve_gate.py` | prints `<target> <gate> higher\|lower` from `matrix16/targets.py` so no launcher defaults the 5%-FPR bar |

Both launchers now default to the fine ladder
(`2,5,…,90,100,125,150`). The ladder is still 25-spaced **above 100**; no cell lands there today, but
a cell delivering ~110 modes would report 100.
