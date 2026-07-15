# sEH proxy — hit-vs-decoy discrimination benchmark

Does the pretrained **sEH proxy** (`SehMoleculeProxy`, the MPNN from
`[bengio2021gflownet]` that RGFN uses as its sEH benchmark reward) actually
separate real sEH inhibitors from random molecules — and what proxy score should
we treat as a "potential hit"? This is the proxy analogue of the docking-oracle
discrimination studies in `docking_6td3/` / `docking_crbn/`.

Full write-up: **`Logs/034_seh-proxy-hit-decoy-discrimination.md`**.

## Headline result

The proxy carries **real but modest, size-independent** signal, and its strength
depends heavily on the negative set:

- vs **random RGFN-policy decoys** — AUROC **0.756** (MW-matched 0.740).
- vs **property-matched (DUD-E-style) drug-like decoys** — AUROC **0.676**
  (MW-matched 0.677).

The drop from 0.76→0.68 shows a good part of the apparent signal was the RGFN
decoys being distributionally odd (large poly-triazoles); against realistic
drug-like look-alikes the proxy separates only weakly, but still above chance
(so it *does* read something binding-relevant beyond gross properties).

Its **absolute scale is not an activity scale**: of 2,315 real ChEMBL sEH
inhibitors, **0 score ≥ 8.0** and only 9 score ≥ 7.0 — the "hit"/"mode" thresholds
the project uses (`8.0` = `TanimotoSimilarityModes.proxy_term_threshold` in
`configs/rgfn_seh_proxy.gin`; `7.0` = entry `028`'s sEH hit bar) — even though
trained RGFN reaches mean 7.0 (entry `019`). Within the actives, proxy score
barely tracks measured potency (Pearson r=0.22).

**Enrichment sweep** (actives-pass-rate ÷ decoys-pass-rate) — enrichment collapses
against the property-matched decoys:

| cutoff | actives kept | rgfn decoys pass / EF | matched decoys pass / EF |
|--------|-------------|-----------------------|--------------------------|
| ≥ 5.0  | 34.5 %      | 6.8 % / **5.0×**      | 15.8 % / **2.2×**        |
| ≥ 6.0  | 7.0 %       | 1.0 % / **7.0×**      | 2.6 % / **2.7×**         |
| ≥ 7.0  | 0.4 %       | 0 % / (empty)         | 0.09 % / 4.5×            |
| ≥ 8.0  | 0 %         | 0 %                   | 0 %                      |

⇒ Treat **~5** as a "promising / enriched-vs-random" bar and **~6** for higher
precision, but expect only **~2–3× enrichment over realistic drug-like molecules**
(not the 5–7× the RGFN-decoy comparison suggests). **7–8 is the proxy-optimized
regime, not a real-activity threshold**, and "modes ≥ 8.0" should be read as a
benchmark diversity count, not an activity claim.

## Real docking beats the proxy (`dock_sets.py` + `benchmark_seh_docking.py`)

Running the same discrimination analysis with **actual QuickVina2-GPU docking against
sEH** (~1,000/set subsample) separates the molecules **better than the proxy on both
negatives**, and on an interpretable kcal/mol scale (actives dock at median
**−9.70**, vs matched −8.50, rgfn −7.40):

| decoy set | **docking** AUROC | proxy AUROC | docking MW-matched |
|-----------|-------------------|-------------|--------------------|
| rgfn      | **0.864**         | 0.756       | 0.780              |
| matched   | **0.730**         | 0.676       | 0.740              |

Enrichment (keep vina ≤ cutoff): at ≤ −9.0, actives 72% vs rgfn 14% / matched 35%
(EF 5.1× / 2.1×); at ≤ −10.0, actives 38% vs 4.4% / 11.8% (EF 8.7× / 3.2×). Docking
is still only moderate against the property-matched decoys (0.73), and it agrees
with the proxy on the actives only weakly (Spearman 0.42) — the proxy is a loose
surrogate. Docking cost ~1 s GPU/molecule (vs the proxy's instant forward pass).
See `discrimination_docking.png` / `summary_docking.json`.

## Files

**Scripts** (run in the `rgfn` env; `source ~/bin/rgfn-smoke-env.sh` first)
- `fetch_seh_actives.py` — pull known sEH inhibitors from ChEMBL (target
  **CHEMBL2409**, human EPHX2) via the REST API; dedup + salt-strip + canonicalize.
  → `data/validation-molecules/sEH_actives_chembl.csv` (the curated positives).
- `make_decoys.py` — sample random molecules from the **untrained RGFN policy**
  (same sampler as the seed set of entry `010` / random arm of entry `023`).
  → `seed_decoys_rgfn.csv` (the "rgfn" negative control).
- `make_matched_decoys.py` — **DUD-E-style** decoys: fetch a drug-like ChEMBL
  background pool, then per active pick the nearest-in-property (MW/logP/HBD/HBA/
  RotB/charge) unused molecule that is topologically distinct (ECFP4 Tanimoto <
  0.35 to every active). → `data/validation-molecules/sEH_decoys_matched.csv`.
- `benchmark_seh_proxy.py` — score actives + **every available decoy set** with
  `SehMoleculeProxy`; compute per-set ROC/PR, cutoffs, enrichment, MW-matched
  control, and the actives-only potency correlation.
- `dock_sets.py` — dock one set per process (subsampled) with `DockingSEHOracle`
  (QuickVina2-GPU vs sEH) → `<set>_docking_results.csv`.
- `benchmark_seh_docking.py` — docking analogue of `benchmark_seh_proxy.py` (Vina,
  lower = better) + docking-vs-proxy AUROC comparison and score correlation.

**Committed inputs / outputs**
- `../../../data/validation-molecules/sEH_actives_chembl.csv` — 2,315 unique
  actives, pChEMBL ≥ 6, with potency + MW.
- `seed_decoys_rgfn.csv` — 2,500 random RGFN-policy decoys (+ MW).
- `../../../data/validation-molecules/sEH_decoys_matched.csv` — 2,315 property-
  matched drug-like ChEMBL decoys (+ MW).
- `actives_results.csv`, `decoys_results.csv`, `matched_decoys_results.csv` —
  per-molecule proxy scores.
- `{actives,rgfn_decoys,matched_decoys}_docking_results.csv` — per-molecule sEH
  Vina energies (~1,000/set subsample); `summary_docking.json` +
  `discrimination_docking.png` — docking metrics + figure.
- `background_pool_chembl.csv` — ~30k drug-like ChEMBL background (cache for the
  matcher; git-ignored intermediate, regenerable).
- `summary.json` — all metrics (AUROC, PR-AUC, MW-matched AUROC, cutoff/enrichment
  sweep, existing-threshold evaluation, potency correlation).
- `discrimination.png` — score distributions, ROC, score-vs-MW, MW-matched panel.

## Reproduce

```bash
source ~/bin/rgfn-smoke-env.sh                 # rgfn env + dgl CUDA libs (login node)
cd <repo root>
python experiments/oracle_validation/seh_proxy/fetch_seh_actives.py --min-pchembl 6
python experiments/oracle_validation/seh_proxy/make_decoys.py --n 2500 \
    --root-dir "$SCRATCH/rgfn_runs/seh_proxy_bench"   # sampler run dir off-repo
python experiments/oracle_validation/seh_proxy/make_matched_decoys.py   # ChEMBL background + match
python experiments/oracle_validation/seh_proxy/benchmark_seh_proxy.py
# --- docking head-to-head (GPU; one process per set to respect the CPU cap) ---
for s in actives rgfn matched; do
    python experiments/oracle_validation/seh_proxy/dock_sets.py --set $s --n 1000
done
python experiments/oracle_validation/seh_proxy/benchmark_seh_docking.py
```

RGFN-decoy sampling uses an A100 (~7 min for 2,500 molecules); the matched-decoy
build fetches ~30k ChEMBL molecules then does the Tanimoto/property matching
(~5 min); fetch + scoring are CPU and finish in a couple of minutes each. Both
ChEMBL pulls need internet (cached afterward).
