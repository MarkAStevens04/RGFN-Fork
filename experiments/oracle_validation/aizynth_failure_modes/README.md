# AiZynthFinder failure-mode diagnosis (SCENT sEH)

Read-only investigation of **why ~51% of SCENT sEH molecules "fail" from-scratch AiZynth**
(ZINC/USPTO), the confound behind the from-scratch-SPARROW headline in `Logs/041`. Full write-up:
**`Logs/047_aizynth-failure-stock-mismatch.md`**. Runs against the from-scratch route cache
(`Logs/041`, job 70976) — no benchmark code changed, molecules routed read-only.

## Scripts (run in the `aizynth` env)

| Script | What it does |
|---|---|
| `aiz_failure_diag.py` | Routes a sample of failed molecules at current vs 10× budget; records the per-molecule failure-mode signatures (iteration cap hit? best-route fraction-in-stock? depth? template occurrence?) that separate budget-exhaustion from policy-OOD. |
| `aiz_residual_diag.py` | Isolates the residual cause: re-routes with filter OFF and depth 12, and dumps every unsolved route's not-in-stock ("open") leaves with heavy-atom counts. |
| `build_block_stock.py` | Builds a NEW AiZynth stock from the 418 `glue_standard_v1` blocks (stereo + flat InChIKeys) + a config that unions it with ZINC. **ZINC hdf5 untouched.** |
| `aiz_stock_diag.py` | 2×2 {zinc, zinc+blocks} × {current, high budget} on the failed sample — how many failures the reaction-GFN's own blocks rescue. |
| `aiz_fullpool.py` | Headline: full 4,749-molecule mode-union solve rate with `zinc+blocks` at production budget, plus a zinc-only control. |

## Reproduce

```bash
conda activate aizynth
D=/scratch/markymoo/rgfn_runs/lsdflow_sparrow          # cache + stock live here (large)
# 1. build the block stock + union config (ZINC left alone)
python build_block_stock.py $D/rgfn_smalllib_stock_flat.hdf5 $D/config_rgfnlib_flat.yml
# 2. headline full-pool run (compute/debug, ~105 min over 32 cores)
AIZ_CONFIG=$D/config_rgfnlib_flat.yml \
  python aiz_fullpool.py $D/scent_rawpool/full_union.smi $D/scent_rawpool/control300.smi out.json 32
```

`full_union.smi` = the 4,749 canonical SMILES keys of the from-scratch timed cache
(`$D/scent_seh/routecache_zinc_uspto_timed.json`); `failed50.smi` = an even sample of its 2,435
unsolved entries.

## Result

Failures are **not** OOD (0/50) and **not** unsynthesizable — they're a **stock/terminal-form
mismatch**: AiZynth reaches the reaction-GFN's own scaffolds (THIQ/proline/azetidine) in a different
oxidation/protection/stereo form, and ZINC doesn't stock the reactive blocks. Adding the 418 blocks
to stock lifts full-pool solvability **48.7% → 73.8% (+25 pts)** at production budget.
