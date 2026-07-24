# ClpP docking — mode-threshold calibration (known binders vs matched decoys)

**Date:** 2026-07-23, ~3pm

## Question

What docking score should count as a "hit" for the ClpP system — i.e., what Vina
cutoff best tells real ClpP binders apart from look-alike molecules that don't bind?

## Context & Summary

The publication-scale 4×4 matrix ([030]) scores one of its four systems, **ClpP**, by
straight molecular docking. To count how many *distinct good molecules* ("modes") each
generator finds, the pipeline needs a score bar a molecule must clear to be called a hit.
For ClpP that bar was a **provisional −2.0 kcal/mol** — a placeholder never grounded in
data. A diagnostic sweep of the finished matrix (this session) showed why that is a
problem: raw ClpP docking energies run −5 to −14 kcal/mol, so **~100 % of every
generator's molecules clear −2.0** and the "mode" count can't tell a strong ClpP
generator from a weak one. The other three systems have calibrated bars (sEH 7.0, DRD2
0.5, 6TD3 −2.0 = the known-glue range), so ClpP was the odd one out.

This experiment sets a real bar the same way [034] did for the sEH proxy: pull known
ClpP binders from ChEMBL as positives, build drug-like look-alike molecules matched to
their physical properties as negatives (a standard "DUD-E" decoy set), dock both with
the **exact** engine the matrix trains on, and read off the docking score that best
separates the two groups. That score becomes the ClpP hit bar.

One thing had to be nailed down first: *which* ClpP. The receptor file ships without
provenance, but its ordered residues span chains B/C **58–249** — a range only **human**
mitochondrial ClpP (277 aa, its 1–56 transit peptide cleaved) can occupy; bacterial
ClpP (≤207 aa) cannot reach residue 249. That plus the RGFN paper's own target list
pins it to **human ClpP, PDB 7UVU** (co-crystallized with the activator TR-107), so the
positives are human-ClpP binders from ChEMBL target CHEMBL4523305.

## Answer

Real docking separates known human-ClpP binders from property-matched decoys **well** —
AUROC **0.895** (and 0.897 after matching molecular weight, so it's genuine binding
signal, not a size artifact). The score that best splits the two groups is **−8.0
kcal/mol** (keeps 88 % of true binders while rejecting 77 % of decoys). Adopting −8.0 as
the ClpP hit bar turns a useless gate into a discriminating one: where the old −2.0 bar
passed 99.5–100 % of *every* generator, −8.0 now spreads them — SCENT 99 %, RGFN 73 %,
FragGFN 34 %, RxnFlow 7.5 % — which is exactly the kind of separation a benchmark needs.

## Relevance to our Publication

The four-system matrix is a headline comparison (RGFN vs. baselines across targets). A
reviewer who sees a "modes found" bar chart will immediately ask what defines a mode —
and "everything clears the bar" is indefensible for one of the four systems. This entry
replaces the ClpP placeholder with a data-grounded, reproducible threshold backed by real
inhibitors and a standard decoy control (AUROC 0.895), so the ClpP column of the matrix
measures something real and is defensible on the same footing as sEH ([034]).

## Next Experiments

**Refining for publication.**
- Report a threshold *sensitivity* band, not a single number: −8.0 (Youden, balanced),
  −9.0 (95 % decoy-specificity), −9.78 (99 %). If the generator ranking is stable across
  these bars, the comparison is threshold-robust (mirrors [035] for the hub campaign).
- Optionally overlay the generated-molecule docking distribution against the actives to
  show where each generator's output falls relative to real binders.

**Next steps in project.**
- Apply the same calibration recipe to the **6TD3** differential bar if a reviewer
  questions the −2.0 known-glue value there.
- When the LSD-Flow docking cells (6TD3/ClpP) are activated (they are deferred in the
  current build), make the enumeration reward-generator + hit gate compare the **raw
  Vina value** (lower-is-better), consistent with this −8.0 — *not* the `score`=ReLU(−raw)
  reward column.

# Re-creation

## Relevant Files

Root: `experiments/oracle_validation/docking_clpp/` unless noted.

**Scripts** (run in the `rgfn` env; `source ~/bin/rgfn-smoke-env.sh` first)
- `fetch_clpp_actives.py` — pulls human-ClpP binders from ChEMBL target **CHEMBL4523305**
  via the REST API; dedup + salt-strip + canonicalize. Keeps EC50/AC50/Potency/IC50/Ki/Kd
  (ClpP actives are ~93 % EC50 imipridone *activators*, not inhibitors → keeping only
  IC50/Ki/Kd would drop them). → `data/validation-molecules/ClpP_actives_chembl.csv`.
- `make_matched_decoys.py` — DUD-E-style negatives: per active, nearest-in-property
  (MW/logP/HBD/HBA/RotB/charge) drug-like ChEMBL molecule that is topologically distinct
  (ECFP4 Tanimoto < 0.35 to every active). Reuses the shared background pool
  `../seh_proxy/background_pool_chembl.csv` (~30k, target-agnostic). →
  `data/validation-molecules/ClpP_decoys_matched.csv`.
- `dock_sets.py` — docks one set with `glue.oracles.DockingClpPOracle` (QuickVina2-GPU vs
  ClpP, `docking_batch_size=200`), the same engine/box the matrix ClpP cell trains on. →
  `<set>_docking_results.csv`.
- `benchmark_clpp_docking.py` — ROC/PR-AUC, MW-matched AUROC, the Youden-optimal +
  high-specificity cutoffs, and an enrichment sweep. → `summary_docking.json`,
  `discrimination_docking.png`.
- `submit_dock_clpp.sh` — single-GPU `debug` sbatch that docks both sets + runs the
  benchmark (unused this run — the debug slot was occupied, so docking ran on the login
  A100; kept for reproducibility).

**Datasets**
- `./data/validation-molecules/ClpP_actives_chembl.csv` — 183 unique human-ClpP binders
  (pChEMBL ≥ 5; potency + MW). The curated positives.
- `./data/validation-molecules/ClpP_decoys_matched.csv` — 183 property-matched drug-like
  ChEMBL decoys (+ MW). The negatives.
- `./data/targets/ClpP.pdbqt` — the docked receptor: human ClpP (7UVU-derived), chains
  B/C res 58–249. Box centre `[-38.127, 45.671, -20.898]`, size `[17,17,17]` (upstream
  `RECEPTOR_CENTERS['ClpP']`).

**Results**
- `actives_docking_results.csv`, `matched_decoys_docking_results.csv` — per-molecule ClpP
  Vina energies (nan = failed to dock).
- `summary_docking.json` — all metrics (AUROC, cutoffs, enrichment sweep).
- `discrimination_docking.png` — score distributions, ROC, Vina-vs-MW, enrichment.

**Threshold wiring**
- `experiments/lsd_hubs/matrix16/targets.py` (worktree `matrix16-lsdflow`) — ClpP
  `mode_reward_threshold` −2.0 → **−8.0**, `threshold_variants=[-8.0, -9.0]`, with the
  raw-vs-score caveat recorded in `reward_note`.

**Job Logs**
- Docking + benchmark console: login A100, 2026-07-23 14:58–15:00 (both sets in ~2 min).

## Relevant Versions

New files on branch `Hub-Analysis` (scripts + `data/validation-molecules/ClpP_*`,
results); `targets.py` edit on worktree branch `matrix16-lsdflow`. **Not yet committed.**
`[TODO — add commit hash after pushing]`. Files to commit: everything under
`experiments/oracle_validation/docking_clpp/`, the two `data/validation-molecules/ClpP_*.csv`,
and the worktree `experiments/lsd_hubs/matrix16/targets.py`.

## Relevant Resources

**Sources**
- **PDB 7UVU** — human mitochondrial ClpP in complex with the activator TR-107
  (identifies the receptor's organism + the apical binding site the docking box covers).
- **ChEMBL CHEMBL4523305** — *Homo sapiens* ATP-dependent Clp protease proteolytic
  subunit, mitochondrial (the actives' target). https://www.ebi.ac.uk/chembl/
- **DUD-E** (Mysinger et al. 2012) — property-matched decoy methodology.
- `[koziarski2024rgfn]` — ClpP is one of RGFN's benchmark docking targets.
- Method mirrors entry [034] (sEH proxy hit-vs-decoy discrimination).

**Packages**
- `requests` (ChEMBL REST), `rdkit` (clean/featurize/Tanimoto), `numpy`/`scikit-learn`
  1.8.0 (ROC/AUROC in `benchmark_clpp_docking.py`), `matplotlib`.
- QuickVina2-GPU-2.1 + Meeko/OpenBabel via `glue.oracles.DockingClpPOracle`
  (`glue/oracles/docking_seh_oracle.py`).

## Method

1. **Identify the receptor.** Read the ordered residue range of `data/targets/ClpP.pdbqt`
   (chains B/C, res 58–249) → only human ClpP (277 aa) fits; cross-checked against the
   RGFN paper's target list and PDB 7UVU. → ChEMBL target CHEMBL4523305.
2. **Actives.** `python fetch_clpp_actives.py --min-pchembl 5` → 183 unique binders
   (174 EC50 / 6 Kd / 3 IC50; MW median 422).
3. **Decoys.** `python make_matched_decoys.py` → 183 property-matched decoys (MW/logP/
   HBA/RotB medians match the actives; Tanimoto < 0.35 to all actives).
4. **Dock both sets** on one A100 (login), `docking_batch_size=200`:
   `python dock_sets.py --set actives` ; `python dock_sets.py --set matched`.
5. **Discrimination + cutoff:** `python benchmark_clpp_docking.py`.
6. **Wire the bar:** set ClpP `mode_reward_threshold=-8.0` in the matrix16 `targets.py`.

## Results

Docking success: actives **161/183** docked (22 conformer/prep failures), decoys
**163/183**.

| set | n docked | Vina median [p25,p75] | min |
|---|---|---|---|
| actives (real ClpP binders) | 161 | **−8.90** [−9.80, −8.30] | −12.00 |
| matched decoys | 163 | **−7.00** [−7.90, −6.50] | −10.10 |

**Discrimination:** AUROC **0.895**, PR-AUC 0.879, **MW-matched AUROC 0.897** (signal is
not a size confound).

**Cutoffs** (call a hit if Vina ≤ cutoff):

| cutoff | rule | TPR (actives kept) | FPR (decoys passed) | enrichment |
|---|---|---|---|---|
| **−8.00** | **Youden / F1 / balanced-acc optimum** | **88.2 %** | **23.3 %** | 3.8× |
| −8.60 | 90 % decoy-specificity | 61.5 % | 11.0 % | 5.6× |
| −9.00 | 95 % decoy-specificity | 48.4 % | 6.1 % | 7.9× |
| −9.78 | 99 % decoy-specificity | 28.6 % | 1.2 % | 23.3× |

**Payoff — ClpP mode fractions, old vs new bar** (seed 42; rgfn from its in-progress
training snapshot):

| cell | raw Vina median | modes @ −2.0 (old) | modes @ **−8.0 (new)** |
|---|---|---|---|
| scent_clpp | −11.60 | 100 % | **99.0 %** |
| rgfn_clpp | −9.00 | 99.8 % | **73.4 %** |
| fraggfn_clpp | −7.40 | 99.5 % | **33.5 %** |
| rxnflow_clpp | −6.80 | 100 % | **7.5 %** |

The old bar gave no separation; −8.0 recovers the full generator ranking.
