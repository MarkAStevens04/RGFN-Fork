# ClpP docking — hit-threshold calibration

What docking score should count as a "hit" for the **ClpP** system in the 4×4 matrix
([030]) and the LSD-Flow analysis? The provisional bar was `raw_score < -2.0`, which
~100 % of every generator clears (raw ClpP docking is −5..−14 kcal/mol) — so it doesn't
discriminate. This experiment sets a real bar from known binders vs. matched decoys, the
docking analogue of `../seh_proxy/`.

Full write-up: **`Logs/045_clpp-docking-threshold-calibration.md`**.

## Headline result

Real QuickVina2-GPU docking separates known human-ClpP binders from property-matched,
drug-like decoys **well**:

- actives dock at median **−8.90** kcal/mol, decoys at **−7.00**.
- **AUROC 0.895** (MW-matched **0.897** — genuine binding signal, not a size confound).
- **Recommended ClpP hit bar: `raw_score <= -8.0 kcal/mol`** (Youden-optimal: keeps 88 %
  of true binders, rejects 77 % of decoys). Stricter variants: −9.0 (95 % decoy
  specificity), −9.78 (99 %).

Adopting −8.0 turns a non-discriminating gate into a discriminating one — where −2.0
passed 99.5–100 % of every generator, −8.0 spreads them: SCENT 99 % · RGFN 73 % ·
FragGFN 34 % · RxnFlow 7.5 %.

## The receptor (important)

The matrix docks against **human mitochondrial ClpP** — PDB **7UVU** (ClpP + the
imipridone activator TR-107), `data/targets/ClpP.pdbqt`, box centre
`[-38.127, 45.671, -20.898]` / size `[17,17,17]`. Identified from the pdbqt's ordered
residues (chains B/C **58–249**): only human ClpP (277 aa, transit peptide 1–56 cleaved)
can occupy that range — bacterial ClpP (≤207 aa) cannot. So actives come from ChEMBL
target **CHEMBL4523305** (Homo sapiens). Its binders are ~93 % **EC50 activators**
(imipridone/ONC201 class) that bind the same apical pocket as 7UVU's TR-107 — the site
the docking box covers — so they are the right positives for this docking site.

## Convention

Vina energy in kcal/mol, **lower = better binder**. The threshold is on the **raw** Vina
value (`candidates.csv` `raw_score` / snapshot `term_raw_score`), NOT the `score` column
(which is `ReLU(-raw)`, higher-better). The matrix16 `targets.py` ClpP gate is
`higher_is_better=False`, `mode_reward_threshold=-8.0`.

## Files

**Scripts** (`rgfn` env; `source ~/bin/rgfn-smoke-env.sh` first)
- `fetch_clpp_actives.py` — human ClpP actives from ChEMBL (CHEMBL4523305; EC50/AC50/
  Potency/IC50/Ki/Kd) → `data/validation-molecules/ClpP_actives_chembl.csv`.
- `make_matched_decoys.py` — DUD-E property-matched decoys (reuses `../seh_proxy/
  background_pool_chembl.csv`) → `data/validation-molecules/ClpP_decoys_matched.csv`.
- `dock_sets.py` — dock one set with `DockingClpPOracle` → `<set>_docking_results.csv`.
- `benchmark_clpp_docking.py` — ROC/AUROC, Youden + specificity cutoffs, enrichment →
  `summary_docking.json`, `discrimination_docking.png`.
- `submit_dock_clpp.sh` — single-GPU `debug` sbatch (docks both sets + benchmark).

**Committed inputs / outputs**
- `../../../data/validation-molecules/ClpP_actives_chembl.csv` — 183 actives.
- `../../../data/validation-molecules/ClpP_decoys_matched.csv` — 183 matched decoys.
- `{actives,matched_decoys}_docking_results.csv`, `summary_docking.json`,
  `discrimination_docking.png`.

## Reproduce

```bash
source ~/bin/rgfn-smoke-env.sh
cd <repo root>
python experiments/oracle_validation/docking_clpp/fetch_clpp_actives.py --min-pchembl 5
python experiments/oracle_validation/docking_clpp/make_matched_decoys.py
# docking (GPU): a free debug slot -> sbatch submit_dock_clpp.sh; else on a login A100:
python experiments/oracle_validation/docking_clpp/dock_sets.py --set actives
python experiments/oracle_validation/docking_clpp/dock_sets.py --set matched
python experiments/oracle_validation/docking_clpp/benchmark_clpp_docking.py
```

ChEMBL pulls need internet (cached afterward); docking ~0.3 s/mol on one A100 (~2 min
for both sets of ~183).
