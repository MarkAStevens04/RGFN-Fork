# SCENT sEH — why AiZynthFinder "fails" half the molecules (a stock/chemistry mismatch, not bad molecules)

**Date:** 2026-07-24, ~11am

## Question

When a standard retrosynthesis planner says it *can't* find a route to half of our generator's
molecules, is that because the molecules are genuinely hard to make — or because the planner is
looking for the wrong starting materials?

## Context & Summary

Our neutral referee for the library-efficiency benchmark re-plans a route for every molecule with
AiZynthFinder against the ZINC catalogue and then prices it (entry [041]). That referee only finds a
route for **48.7%** of the SCENT sEH molecules, which makes the "from-scratch" comparison look bad
for the reaction-grounded generator and leaves the fixed-modes plot half-empty. A separate check had
already shown the generator itself is fine — under the SCENT paper's own protocol (top-100 molecules
by reward, no diversity filter) our model recovers their published **0.773** route-success — so the
low pool-wide number had to be coming from *how AiZynth is being run*, not from the molecules. This
entry opens the hood: we re-routed a sample of the "failures" while recording, for each one, the
tell-tale signs that distinguish the four ways an AiZynth search can come up empty (ran out of search
budget, the templates propose nothing sensible, the catalogue is too small, or a feasibility filter
pruned the one good step). Then we tested the leading explanation directly by adding the generator's
own **418 building blocks** to the catalogue — as a **new** stock library, leaving the ZINC catalogue
untouched — and re-measured the whole pool.

## Answer

The "failures" are almost entirely a **stock/representation mismatch, not unsynthesizable or exotic
molecules.** None of the sampled failures were out-of-distribution: AiZynth's own templates
disconnect them most of the way (past three-quarters of the pieces already purchasable) using common,
well-precedented chemistry. What it gets stuck on is a single leftover fragment that is one of the
**reaction-GFN's own building blocks reached in a slightly different form** (a different oxidation or
protecting-group state, or the mirror-image stereochemistry), plus a tail of specialized reagents the
ZINC catalogue simply doesn't list. When we tell AiZynth those 418 blocks are purchasable, pool-wide
route-success jumps from **48.7% to 73.8%** — half the failures resolve immediately, with no change
to the search effort or the planner's chemistry knowledge. So the molecules *are* makeable from the
generator's blocks; the referee just wasn't allowed to stop at them.

## Relevance to our Publication

The reactions-per-mode thesis invites two obvious reviewer objections, and this entry answers both
with numbers. (1) *"Your reaction-GFN just makes less-synthesizable molecules, so the comparison is
rigged in its favour."* — No: ~74% are route-solvable from the generator's own blocks, and the
shortfall from 100% is dominated by conservative search budget and catalogue gaps, not molecule
quality. (2) *"The from-scratch-SPARROW penalty you report ([041], 2.74 vs 3.66 reactions/mode) is a
real synthesizability cost."* — It is substantially a **chemistry-mismatch artifact**: from-scratch
AiZynth couldn't route ~51% of the pool against ZINC at all, and the routes it did find used generic
literature chemistry rather than the generator's efficient block-based routes. This is the empirical
backbone for reporting *native-route* pricing (which the reaction-GFN uniquely provides) as the
honest comparison, and it directly motivates the chemistry-homogenization experiments.

## Next Experiments

**Refining for publication**
- Report the full-pool number at generous search budget too (`zinc+blocks` at 10× iterations),
  giving an upper bound to sit alongside the 73.8% production-budget figure (the 50-sample already
  reached 39/50 = 78% there).
- Show the same stock effect for the other three generators (RGFN/RxnFlow/FragGFN), to confirm the
  from-scratch confound is not specific to SCENT.
- ~~A cleaner residual taxonomy figure~~ **DONE 2026-07-27** (`residual_taxonomy.py`, table + figure
  above): 51% of the residual leaves are the reaction-GFN's own core in another terminal form, 42%
  standard reagents/protecting groups, only 7% unclassified.

**Next steps in project**
- Fold this into the benchmark's framing: present native-route pricing as primary and treat
  from-scratch SPARROW as the deliberately-unfair floor, with this entry quantifying the unfairness.
- Proceed with the chemistry-homogenization plan (Exp A/B/C): the simplest, most direct version of
  the same argument is now in hand, so those training runs become confirmation rather than discovery.

# Re-creation

### Relevant Files

Root: `./experiments/oracle_validation/aizynth_failure_modes/` unless noted. Diagnostics run
read-only against the from-scratch route cache; no benchmark code was modified.

**Scripts**
- `aiz_failure_diag.py` — routes a sample of failures at current (iter=100/time=60/depth=6) vs high
  (iter=1000/time=300/depth=9) budget; per molecule records iterations run, whether the iteration cap
  was hit, and the best partial route's fraction-in-stock / depth / template-occurrence — the
  signatures that separate budget-exhaustion from policy-OOD.
- `aiz_residual_diag.py` — isolates the residual cause: re-routes the sample with the feasibility
  filter OFF and at depth 12, and dumps each unsolved route's not-in-stock ("open") leaves + their
  heavy-atom counts.
- `build_block_stock.py` — builds a NEW AiZynth stock hdf5 from the 418 `glue_standard_v1` blocks
  (both stereo and stereo-stripped InChIKeys) and a config unioning it with ZINC as key `rgfnlib`;
  the pristine ZINC hdf5 is never touched.
- `aiz_stock_diag.py` — 2×2 of {zinc, zinc+blocks} × {current, high budget} on the failed sample.
- `aiz_fullpool.py` — headline: full 4,749-molecule mode-union route-success with `zinc+blocks` at
  production budget, plus a zinc-only control sample; writes incrementally (wall-clock safe).
- `residual_taxonomy.py` — rule-based classification of the residual open leaves (block-stereo-only /
  block-core-other-form / partial-assembly / tiny-reagent / reagent-or-PG / other) + the two-panel
  figure; emits `results/residual_taxonomy.{png,pdf,csv}`. Rules are deliberately conservative
  against our own claim, and one SMARTS bug was caught here: `O=C1c2ccccc2C1=O` is a 4-membered ring,
  so phthalimide needs `O=C1[#7]C(=O)c2ccccc12`.
- `README.md` — directory guide + reproduce steps.

**Datasets / inputs**
- `./data/libraries/glue_standard_v1/fragments.csv` — the 418 reaction-GFN "small" building blocks
  (identical set to `external/scent/data/small/fragments.txt`); the blocks added to stock.
- `./data/models/aizynthfinder/config.yml` + `zinc_stock.hdf5` — the pristine referee config/stock
  (17.4M ZINC InChIKeys); used unmodified as the baseline and as the base for the union config.
- `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/scent_seh/routecache_zinc_uspto_timed.json` — the
  from-scratch timed route cache (entry [041], job 70976): 4,749 molecules, 2,314 solved (48.7%). Its
  keys are the mode-union; the failed sample is drawn from its 2,435 unsolved entries.

**Results (scratch)** — root `/scratch/markymoo/rgfn_runs/lsdflow_sparrow/`
- `scent_rawpool/failed50.smi`, `full_union.smi` (4,749), `control300.smi` — input molecule lists.
- `rgfn_smalllib_stock_flat.hdf5` (+ `config_rgfnlib_flat.yml`) — stereo-agnostic block stock/config
  (456 InChIKeys); `rgfn_smalllib_stock.hdf5` (+ `config_rgfnlib.yml`) — stereo-retained variant.
- `scent_rawpool/aiz_{failure,residual,stock,stock_flat,fullpool}_diag*.json` — per-run outputs.

**Job Logs** — `/scratch/markymoo/rgfn_runs/aiz_*-{71427,71428,71430,71432,71436}.{out,err}`.

### Relevant Versions

Committed on branch `Hub-Analysis` as **`e226ed1`** ("Figuring out where synthesizability is
going") — `Logs/047_*.md`, `docs/RESEARCH_CONTEXT.md`, and the five files under
`experiments/oracle_validation/aizynth_failure_modes/`. The stock hdf5s / SMILES lists / per-run JSON
are large scratch artifacts, regenerated by `build_block_stock.py` + the cache, and are intentionally
not committed.

### Relevant Resources

**Sources**
- Genheden et al., *AiZynthFinder* (J. Cheminform. 2020) and *AiZynthFinder 4.0* (2024) — MCTS
  search, `is_solved` = all leaves in stock within `max_transforms`, the scorers/filters and default
  knobs (`iteration_limit=100`, `time_limit=120`, `max_transforms=6`, `C=1.4`, QuickKerasFilter
  `filter_cutoff=0.05`) used to design the diagnostic. `[genheden2020aizynth]`
- Gaiński et al., *SCENT* — the source of the 418-block "small" library and the published
  top-100 AiZynth route-success (0.773 ± 0.066) this pool is measured against. `[gainski2025scent]`
- Entry [041] (from-scratch SPARROW headline + the 48.7% coverage this entry explains); entry [018]
  (the AiZynth route-success metric); entry [040] (ZINCFrag chemistry-alignment decision).

**Packages**
- AiZynthFinder 4.4.1 in the `aizynth` conda env (`external/setup_aizynthfinder.sh`) — used by all
  five scripts.
- RDKit — InChIKey computation / stereo stripping in `build_block_stock.py` and the leaf/block match.

### Method

All runs on Balam (CPU-only; 32 cores/node), `aizynth` env. Molecules routed read-only.

1. **Failure-mode signatures** — `aiz_failure_diag.py` on `failed50.smi` (50 evenly-sampled unsolved
   cache entries) at current vs high budget → `aiz_failure_diag.json` (job 71427, debug).
2. **Residual cause** — `aiz_residual_diag.py` on the same 50 at high budget with filter-on,
   filter-off, and depth-12 arms; dumps open leaves → `aiz_residual_diag.json` (job 71428).
3. **Leaf-vs-block match** — offline set-membership of the unsolved open leaves against the 418
   blocks, with and without stereo (no AiZynth; RDKit InChIKey).
4. **Add the blocks to stock** — `build_block_stock.py` → `rgfn_smalllib_stock_flat.hdf5` +
   `config_rgfnlib_flat.yml` (ZINC untouched); `aiz_stock_diag.py` 2×2 on the 50, stereo-retained
   (71430) and stereo-agnostic (71432).
5. **Full-pool headline** — `aiz_fullpool.py` over all 4,749 union molecules with `zinc+blocks` at
   production budget + a 300-molecule zinc-only control → `aiz_fullpool.json` (job 71436, debug,
   ~1 h; incremental JSONL). Compared to the cached ZINC baseline (48.7%).

### Results

**Failure-mode signatures — 50 sampled failures (job 71427).** None are out-of-distribution.

| budget | solved | hit iteration cap | ran-dry & frac<0.5 (OOD) | unsolved best-route frac-in-stock |
|---|---|---|---|---|
| current (iter=100/60s/d6) | 0/50 | 41/50 | **0/50** | 8 at 0.5–0.75, 42 at ≥0.75 |
| high (iter=1000/300s/d9) | 9/50 | 1/50 | **0/50** | 41 at ≥0.75 |

Budget alone recovers 9/50. The other 41 run dry at ≥0.75 in-stock — nearly complete, one open leaf.

**Residual cause — same 50 at high budget (job 71428).** Failures have ~1 open leaf (41/42 exactly
one); leaf heavy-atom count median 14–16 (only ~16% ≤8 HA).

| arm | solved | Δ vs filter-on |
|---|---|---|
| filter on (uspto) | 8/50 | — |
| filter off | 10/50 | +2 (filter pruned) |
| depth 12 | 11/50 | +3 (residual depth) |

**Leaf-vs-block match (offline).** Block heavy-atom count: mean 9.3, median 10, max 17. Of the 43
residual open leaves: **0/43** match a block with stereo, **4/43** stereo-stripped. Recurring
residuals are the reaction-GFN's own scaffolds in a different form — tetrahydroisoquinoline (THIQ)
`…C1Cc2ccccc2CN1` as the N-carboxylated acid (×6), methyl ester (×4), free acid (×3, flat → matches
block only stereo-stripped), aldehyde — plus specialized reagents (2,4-dibromobutanoate ×4, B2pin2,
Wittig phosphonium, Boc/SEM/trityl-protected, `C#N` ×4). The block library contains the THIQ (1),
azetidine (4) and pyrrolidine (9) scaffolds — with defined stereo — confirming a terminal-form, not a
scaffold, mismatch.

**Add the 418 blocks to stock — 50 failures (jobs 71430 stereo-retained, 71432 stereo-agnostic).**

| budget | ZINC only | ZINC+blocks (stereo) | ZINC+blocks (stereo-agnostic) |
|---|---|---|---|
| current | 0/50 | 6/50 | **24/50** |
| high | 9/50 | 22/50 | **39/50** |

**Residual taxonomy (follow-up, 2026-07-27).** The 43 open leaves of the 42 high-budget failures,
classified by rule (`experiments/oracle_validation/aizynth_failure_modes/residual_taxonomy.py`;
figure + CSV in its `results/`):

| category | leaves | share |
|---|---|---|
| block core, other form (THIQ/proline/azetidine as ester, aldehyde, N-carbamate) | 14 | 33% |
| block, stereo only (exact flat match to a SMALL block) | 4 | 9% |
| partial assembly of block cores (two cores already coupled) | 4 | 9% |
| **→ the reaction-GFN's own core** | **22** | **51%** |
| reagent / protecting group (poly-halide ester, Boc, SEM, trityl, phthalimide, pinacol boronate, phosphonium) | 13 | 30% |
| tiny reagent (HCN, methanol) | 5 | 12% |
| **→ standard reagent / protecting group** | **18** | **42%** |
| other | 3 | 7% |

Only **7%** resist classification, and the 43 leaves collapse to just **26 distinct species** (top 10 =
63%) — so a handful of stock entries account for most of the residual. The rules are deliberately
conservative against our own claim (N-benzyl/N-acyl proline esters fall to "other" because
pyrrolidine is not a distinctive core), so 51% is a floor.

**Full-pool headline — all 4,749 mode-union molecules, production budget (job 71436).**

| stock | route-success |
|---|---|
| ZINC baseline (cache) | 2314/4749 = **48.7%** |
| ZINC control (this run, 300) | 139/300 = 46.3% (reproduces baseline within sampling) |
| **ZINC + 418 reaction-GFN blocks** | **3506/4749 = 73.8%** (**+25.1 pts**) |

~1,192 of the 2,435 failures (49%) become solvable purely by adding the generator's own blocks to the
catalogue — matching the 50-sample projection (24/50), with no change to search budget or chemistry.
