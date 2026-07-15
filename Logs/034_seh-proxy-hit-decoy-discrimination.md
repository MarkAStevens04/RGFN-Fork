# sEH — does the pretrained proxy separate known inhibitors from random molecules?
**Date:** 2026-07-14, ~3pm

### Question

Can the pretrained sEH proxy tell real, experimentally-confirmed sEH inhibitors apart from randomly generated molecules — and if so, what score should we treat as the line between a "hit" and a "miss"?

### Context & Summary

**Context.** The sEH proxy (`SehMoleculeProxy`) is the field-standard RGFN benchmark reward: a pretrained neural network from the original GFlowNet paper (`[bengio2021gflownet]`) that predicts how well a molecule binds soluble epoxide hydrolase, reused unchanged by RGFN (`[koziarski2024rgfn]`) and by our fixed-reward pipeline (entry `019`/`020`). We *use* this score as a reward and even classify molecules as "hits"/"modes" against fixed thresholds — `rgfn_seh_proxy.gin` counts a molecule as a mode at proxy ≥ **8.0**, and entry `028` used **7.0** as the sEH hit bar — but we have never checked, on real data, whether those thresholds actually correspond to being an sEH binder. Every glue-side oracle in this project (entries `002`–`008`) was validated exactly this way: does the scorer separate known actives from decoys? This entry does the same acid test for the sEH proxy itself.

**Summary.** We pulled a large batch of known sEH inhibitors from ChEMBL (target CHEMBL2409, human sEH / gene EPHX2), kept the potent, confirmed binders, and scored them with the exact `SehMoleculeProxy` used as the training reward. We compared them against **two** negative-control sets: (1) **randomly generated molecules** from the untrained RGFN policy — the "random synthesizable molecule" baseline the project uses elsewhere (seed set of entry `010`, random arm of entry `023`); and (2) **property-matched (DUD-E-style) decoys** — drug-like ChEMBL molecules matched to the actives on molecular weight / logP / H-bond donors / H-bond acceptors / rotatable bonds / charge but built on a different scaffold (fingerprint similarity < 0.35 to every active), so the scorer cannot win on gross properties. We then measured how well the proxy score separates actives from each negative (ROC/PR), derived an empirical cutoff for calling a molecule a potential hit, checked those cutoffs against the 7.0/8.0 values already in our configs, and repeated the separation on a molecular-weight-matched subsample (the entry `007` control) to confirm the proxy is reading binding, not just size. Finally we ran the **same discrimination analysis with the real QuickVina2-GPU docking oracle** (entry `010`) on ~1,000-molecule subsamples of each set, to see whether actual docking separates hits from decoys better than the surrogate — the head-to-head this entry's earlier draft flagged as the next step.

### Answer

The proxy carries a **real but modest** signal whose strength depends on how realistic the decoys are. Against random RGFN-policy molecules it ranks known inhibitors above decoys with AUROC 0.76 (0.74 molecular-weight-matched, so not just size); against **property-matched drug-like decoys** it drops to 0.68 — meaning a good part of that 0.76 was the RGFN molecules being distributionally odd, and against realistic look-alikes the proxy separates only weakly, though still above chance (it does read *something* binding-relevant beyond gross properties). Its **absolute scale is not an activity scale**: real potent inhibitors top out around a proxy score of 7.7 and *none* of 2,315 reach the 8.0 "hit/mode" threshold we use in configs (only 9 reach 7.0), even though a trained RGFN routinely reaches 7–8, and within the real inhibitors the score barely tracks measured potency (r ≈ 0.22). So 7.0/8.0 are the *proxy-optimized* regime, not a real-activity threshold; the empirically meaningful "potential hit" bar is far lower — roughly **5–6** — but it buys only ~2–3× enrichment over realistic drug-like molecules (not the 5–7× the random-decoy comparison implies). Practically, "modes ≥ 8.0" is a benchmark diversity count, not a claim that those molecules resemble real hits.

**Real docking beats the surrogate — the follow-up analysis.** Running the identical comparison with actual QuickVina2-GPU docking against sEH (the real oracle, entry `010`) separates the same molecules better on *both* negatives — AUROC **0.86 vs 0.76** against random molecules and **0.73 vs 0.68** against the property-matched decoys — and, unlike the proxy, on an **interpretable scale**: real inhibitors dock at a median **−9.7 kcal/mol**, so chemically sensible hit cutoffs (≤ −9 / ≤ −10) actually apply. Docking is still only moderate against the hard property-matched decoys (0.73), and it agrees with the proxy on the actives only weakly (Spearman 0.42) — confirming the proxy is a rough stand-in for docking, not a tight one. So docking improves discriminating power and gives back a real energy scale, at the cost of ~1 s of GPU time per molecule (vs the proxy's instant forward pass).

### Relevance to our Publication

Our evaluation plan (Objective 5 in `RESEARCH_CONTEXT.md`) promises a **recovery-of-known-actives / retrospective-enrichment** result and an **anti-gaming** check that the in-loop reward tracks real activity. Reviewers at Digital Discovery / JCIM will ask whether the reward the generator optimizes is meaningful at all — a proxy that scores random molecules as highly as real inhibitors would make every downstream sEH number suspect. This entry gives a direct, quantitative answer on the field-standard benchmark target: the proxy *does* enrich real inhibitors over random molecules (so it is not noise), but its high-reward region is *not* where real inhibitors live, which reframes how we must report "modes above 8.0" and turns the 7.0/8.0 thresholds from unexamined convention into a documented calibration caveat. The docking follow-up strengthens the story with a clean methods result reviewers will want: swapping the surrogate proxy for the real docking oracle measurably improves retrospective enrichment (AUROC 0.76→0.86 / 0.68→0.73) and restores an interpretable binding scale — evidence for reporting sEH enrichment against docking, and for the multi-fidelity direction (docking/MD as the expensive `O`, Objective 3) rather than trusting the surrogate's scale.

### Next Experiments

**Refining for publication**
- **DONE this entry:** the property/topology-matched (DUD-E-style) decoy control — AUROC drops 0.76→0.68 and enrichment collapses from 5–7× to 2–3×, showing much of the random-decoy separation was distributional. A future variant could use ZINC (rather than ChEMBL) as the DUD-E background for full DUD-E fidelity.
- **DONE this entry:** the docking head-to-head — real QuickVina2-GPU docking beats the proxy on both negatives (AUROC 0.86 vs 0.76 random; 0.73 vs 0.68 matched) and restores an interpretable kcal/mol scale (actives median −9.7). Worth extending to the full 2,315/2,500/2,315 sets (this ran ~1,000/set subsamples) and to ≥3 docking replicas per molecule to quantify QuickVina2-GPU's stochastic noise.
- Report the proxy-vs-potency correlation and the docking-beats-proxy AUROC gap as explicit anti-gaming figures, and re-audit any sEH result that quotes "modes ≥ 8.0" with the calibration caveat.

**Next steps in project**
- Even real docking is only moderate against property-matched decoys (0.73) and only weakly tracks potency — motivating a **higher-fidelity oracle** (co-folding / Boltz-2 / short MD, Objective 3/5) for the top-k, and using docking (not the raw proxy) as the reference when reporting sEH enrichment.
- Decide whether to keep using raw proxy ≥ 8.0 as the sEH hit/mode threshold anywhere it implies activity, or switch reported sEH "hit" counts to the enrichment-calibrated ~5/~6 bars (proxy) / ≤ −9 kcal/mol (docking) documented here.
- Run the identical benchmark on the **DRD2** proxy (`glue/.../drd2.py`, used in the stdlib four-way) to check whether the scale-miscalibration is sEH-specific or general to the pretrained surrogate rewards.

# Re-creation

### Relevant Files

Root: project repository `RGFN-Fork/`.

**Scripts**
- `./experiments/oracle_validation/seh_proxy/fetch_seh_actives.py` — pulls known sEH inhibitors from the ChEMBL REST API (target CHEMBL2409 = human bifunctional epoxide hydrolase 2 / EPHX2), keeps IC50/Ki/Kd with pChEMBL ≥ cutoff, dedups to one row per compound (best potency), strips salts / takes the largest organic fragment, uncharges, canonicalizes with RDKit. Uses only `requests` + `rdkit`.
- `./experiments/oracle_validation/seh_proxy/make_decoys.py` — samples unique valid terminal SMILES from the **untrained** RGFN forward policy (built from `configs/rgfn_seh_proxy.gin`), mirroring `make_seh_seed._sample_unique`; drops any collision with the actives; writes SMILES + MW. Trainer is used only as a sampler (no GFN training). The "rgfn" negative.
- `./experiments/oracle_validation/seh_proxy/make_matched_decoys.py` — DUD-E-style decoys: fetches a drug-like ChEMBL background pool (molecule endpoint, MW-binned, `only=`-projected, cached), cleans/canonicalizes as the actives, gates to ECFP4 Tanimoto < 0.35 to every active, then per active greedily picks the nearest-in-property (MW/logP/HBD/HBA/RotB/charge; hard gates |ΔMW| ≤ 40 Da + equal charge) unused molecule. The "matched" negative.
- `./experiments/oracle_validation/seh_proxy/benchmark_seh_proxy.py` — scores actives + **every available decoy set** with `SehMoleculeProxy` via `proxy.model.compute_scores` (byte-identical to the in-training reward path); per decoy set computes ROC-AUC / PR-AUC, Youden + fixed-specificity cutoffs, an enrichment sweep, evaluation of the 7.0/8.0 thresholds, and an MW-matched re-analysis (entry `007` control); plus the actives-only proxy-vs-pChEMBL correlation; emits the CSVs, `summary.json`, and `discrimination.png`.
- `./experiments/oracle_validation/seh_proxy/dock_sets.py` — docks **one** set per process (subsampled to `--n`) with `glue.oracles.DockingSEHOracle` (QuickVina2-GPU vs sEH); one-set-per-process keeps each run under the login node's 3600 s CPU cap. Writes `<set>_docking_results.csv`.
- `./experiments/oracle_validation/seh_proxy/benchmark_seh_docking.py` — the docking analogue of `benchmark_seh_proxy.py`: same ROC/PR/MW-matched/enrichment analysis on Vina energies (lower = better; oriented as `-vina`), plus the docking-vs-proxy AUROC comparison and proxy↔docking score correlation on actives; emits `summary_docking.json` + `discrimination_docking.png`.
- `./experiments/oracle_validation/seh_proxy/README.md` — directory guide + reproduce steps.

**Models**
- `./rgfn/gfns/reaction_gfn/proxies/cache/bengio2021flow_proxy.pkl.gz` — the pretrained MPNN weights (`SehMoleculeProxy`); the exact reward under test. Originates from `GFNOrg/gflownet` (`[bengio2021gflownet]`).

**Datasets**
- `./data/validation-molecules/sEH_actives_chembl.csv` — 2,315 unique curated sEH actives (canonical SMILES, ChEMBL id, pChEMBL, potency, MW); pChEMBL ≥ 6 (≤ 1 µM). The KNOWN+ positives for the benchmark.
- `./experiments/oracle_validation/seh_proxy/seed_decoys_rgfn.csv` — 2,500 random molecules from the untrained RGFN policy (SMILES + MW). The "rgfn" negative control.
- `./data/validation-molecules/sEH_decoys_matched.csv` — 2,315 DUD-E-style property-matched drug-like ChEMBL decoys (SMILES + MW); topologically distinct from all actives. The "matched" negative control.
- `./experiments/oracle_validation/seh_proxy/background_pool_chembl.csv` — ~30k drug-like ChEMBL molecules (MW 180–700) the matcher draws from; git-ignored intermediate, regenerable.

**Results — proxy**
- `./experiments/oracle_validation/seh_proxy/actives_results.csv`, `decoys_results.csv`, `matched_decoys_results.csv` — per-molecule proxy scores (+ MW; + pChEMBL for actives).
- `./experiments/oracle_validation/seh_proxy/summary.json` — all proxy metrics (per decoy set).
- `./experiments/oracle_validation/seh_proxy/discrimination.png` — 4-panel proxy figure (score distributions for both decoy sets, ROC per set, score-vs-MW, enrichment-vs-cutoff per set).

**Results — docking**
- `./experiments/oracle_validation/seh_proxy/{actives,rgfn_decoys,matched_decoys}_docking_results.csv` — per-molecule sEH Vina energies (subsampled ~1,000/set; nan = failed to dock).
- `./experiments/oracle_validation/seh_proxy/summary_docking.json` — docking metrics + proxy-AUROC comparison + docking↔proxy correlation.
- `./experiments/oracle_validation/seh_proxy/discrimination_docking.png` — 4-panel docking figure (Vina distributions, ROC with proxy AUROC in-legend, Vina-vs-MW, enrichment-vs-Vina-cutoff).

**Job Logs**
- `/scratch/markymoo/rgfn_runs/seh_proxy_bench/{make_decoys,make_matched_decoys,dock_all}.log` — sampling / matching / docking logs (Balam scratch; not in repo).

### Relevant Versions

New/modified files to commit: `Logs/034_seh-proxy-hit-decoy-discrimination.md`, `data/validation-molecules/sEH_actives_chembl.csv`, `data/validation-molecules/sEH_decoys_matched.csv`, the `experiments/oracle_validation/seh_proxy/` dir (6 scripts + README + `seed_decoys_rgfn.csv` + the proxy outputs `{actives,decoys,matched_decoys}_results.csv` + `summary.json` + `discrimination.png` + the docking outputs `{actives,rgfn_decoys,matched_decoys}_docking_results.csv` + `summary_docking.json` + `discrimination_docking.png`; the `background_pool_chembl.csv` cache is git-ignored), and the `RESEARCH_CONTEXT.md` index row.

[TODO — add commit hash after pushing]

### Relevant Resources

**Sources**
- ChEMBL v34 REST API — target **CHEMBL2409** "Bifunctional epoxide hydrolase 2" (human, gene EPHX2, UniProt **P34913**); endpoint `https://www.ebi.ac.uk/chembl/api/data/activity`. 2,895 activity rows at pChEMBL ≥ 6 → 2,315 unique canonical actives.
- `[bengio2021gflownet]` — origin of the pretrained sEH proxy (the MPNN + `best_params.pkl.gz` weights); active-learning/mode framing.
- `[koziarski2024rgfn]` — RGFN; `configs/rgfn_seh_proxy.gin` (β=8, `proxy_term_threshold` 8.0) is the benchmark setup this proxy anchors.
- Prior in-project references: entry `007` (MW-matched-control method), entry `010` (untrained-policy sampling for the sEH seed), entry `019` (RGFN fixed-reward sEH proxy reward mean 7.0/max 8.3), entry `028` (7.0 sEH hit bar).

**Packages**
- `requests` — ChEMBL REST pull (`fetch_seh_actives.py`).
- RDKit (incl. `rdMolStandardize`) — salt-strip / uncharge / canonicalize; MW (`fetch_seh_actives.py`, `benchmark_seh_proxy.py`).
- PyTorch + torch-geometric — `MPNNet` / `SehMoleculeProxy` forward pass (`rgfn/gfns/reaction_gfn/proxies/seh_proxy.py`).
- dgl — pulled in by the RGFN policy import in `make_decoys.py` (needs the smoke-env CUDA libs).
- QuickVina2-GPU-2.1 (via upstream `DockingMoleculeProxy` / `glue.oracles.DockingSEHOracle`) + Meeko/openbabel/RDKit conformer prep — `dock_sets.py`; receptor `data/targets/sEH.pdbqt`, box from `docking_proxy.RECEPTOR_CENTERS['sEH']`.
- scikit-learn (`roc_auc_score`, `average_precision_score`, `roc_curve`), SciPy (`pearsonr`, `spearmanr`), NumPy, pandas, matplotlib — `benchmark_seh_proxy.py`, `benchmark_seh_docking.py`.

### Method

All on balam-login01 (A100), `rgfn` conda env, after `source ~/bin/rgfn-smoke-env.sh`.

1. **Fetch actives:** `python experiments/oracle_validation/seh_proxy/fetch_seh_actives.py --min-pchembl 6` → paged 2,895 ChEMBL activity rows for CHEMBL2409, deduped/cleaned to 2,315 unique canonical actives → `data/validation-molecules/sEH_actives_chembl.csv`.
2. **Generate RGFN decoys:** `python experiments/oracle_validation/seh_proxy/make_decoys.py --n 2500 --oversample 3.0 --root-dir "$SCRATCH/rgfn_runs/seh_proxy_bench"` → 2,500 unique valid SMILES from the untrained RGFN policy (350 fragments / 66 reactions), ~7 min on one A100 → `seed_decoys_rgfn.csv`.
3. **Build matched decoys:** `python experiments/oracle_validation/seh_proxy/make_matched_decoys.py` → fetched 29,824 drug-like ChEMBL molecules (MW 180–700), 27,082 eligible after the Tanimoto < 0.35 gate, selected 2,315 property-matched decoys (one per active) → `data/validation-molecules/sEH_decoys_matched.csv`. Achieved match: actives vs decoys MW 425/425, logP 4.10/4.13, HBA 4/4, RotB 5/6.
4. **Score + analyze (proxy):** `python experiments/oracle_validation/seh_proxy/benchmark_seh_proxy.py` → scored actives + both decoy sets with `SehMoleculeProxy` (0 unscorable), computed all metrics, wrote `actives_results.csv`, `decoys_results.csv`, `matched_decoys_results.csv`, `summary.json`, `discrimination.png`.
5. **Dock (real oracle):** three sequential login-node processes (one per set, fresh CPU budget each), after `source ~/bin/rgfn-smoke-env.sh`: `python experiments/oracle_validation/seh_proxy/dock_sets.py --set {actives,rgfn,matched} --n 1000 --seed 0`. QuickVina2-GPU vs sEH on one A100; ~0.3–0.6 s/mol; 958/872/932 of 1,000 docked ok. Smoke first confirmed aspirin −6.2 / caffeine −6.5 / invalid → nan.
6. **Analyze docking:** `python experiments/oracle_validation/seh_proxy/benchmark_seh_docking.py` → per-set ROC/PR/MW-matched/enrichment on Vina, proxy-AUROC comparison, docking↔proxy correlation; wrote the `*_docking_results.csv` (already written by step 5), `summary_docking.json`, `discrimination_docking.png`.

### Results

Positives: 2,315 unique ChEMBL sEH actives (pChEMBL ≥ 6, median 8.08, MW median 425, proxy median 4.61 [4.02–5.27], proxy max 7.67). Negatives: 2,500 untrained-RGFN-policy molecules ("rgfn"; MW median 523, proxy median 3.73) and 2,315 property-matched drug-like ChEMBL decoys ("matched"; MW median 425, proxy median 4.01). All molecules scored (none dropped as unscorable).

**Separation depends on the negative set.**

| Metric | vs rgfn decoys | vs matched decoys |
|---|---|---|
| ROC-AUC | **0.756** | **0.676** |
| PR-AUC (average precision) | 0.740 | 0.650 |
| MW-matched ROC-AUC | 0.740 | 0.677 |

**Cutoff / enrichment sweep** (enrichment factor EF = actives-pass-rate ÷ decoys-pass-rate):

| Cutoff | Actives kept | rgfn decoys pass / EF | matched decoys pass / EF |
|---|---|---|---|
| ≥ 4.0 | 75.7 % | 37.8 % / 2.0× | 50.2 % / 1.5× |
| ≥ 5.0 | 34.5 % | 6.8 % / **5.0×** | 15.8 % / **2.2×** |
| ≥ 6.0 | 7.0 % | 1.0 % / **7.0×** | 2.6 % / **2.7×** |
| ≥ 6.5 | 1.9 % | 0.16 % / 11.9× | 0.73 % / 2.6× |
| **≥ 7.0** | **0.4 % (9)** | **0 %** | 0.09 % / 4.5× |
| **≥ 8.0** | **0 %** | **0 %** | **0 %** |

The property-matched decoys are a much harder negative: matched to the actives on MW/logP/HBD/HBA/RotB/charge, they sit in the same proxy-score regime, so enrichment at any cutoff is roughly a third of what the RGFN decoys imply (~2–3× vs 5–7×). AUROC still exceeds 0.5, so the proxy reads some binding-relevant signal beyond gross properties.

**Existing project thresholds fail as activity cutoffs:** 0/2,315 actives reach 8.0 (`TanimotoSimilarityModes.proxy_term_threshold`), 9/2,315 reach 7.0 (entry `028` bar) — yet trained RGFN reaches proxy mean 7.0 / max 8.3 (entry `019`). **Proxy-vs-measured-potency among actives:** Pearson r = 0.219, Spearman ρ = 0.233 (n = 2,315) — the score barely tracks pIC50 even within true binders. The highest-scoring RGFN decoys are poly-triazole/tetrazole oligomers (proxy ≈ 6.9), above 99.6 % of real inhibitors.

**Real docking vs the proxy** (QuickVina2-GPU, ~1,000/set subsample: 958 actives / 872 rgfn / 932 matched docked ok). Actives dock at Vina median **−9.70** kcal/mol [−10.20, −8.80], vs rgfn decoys −7.40 and matched decoys −8.50.

| Decoy set | Docking AUROC | Proxy AUROC | Docking MW-matched AUROC |
|---|---|---|---|
| rgfn | **0.864** | 0.756 | 0.780 |
| matched | **0.730** | 0.676 | 0.740 |

Docking beats the proxy on both negatives. Enrichment sweep (keep vina ≤ cutoff): at **≤ −9.0**, 72 % of actives kept vs 14.1 % rgfn / 34.9 % matched decoys (EF 5.1× / 2.1×); at **≤ −10.0**, 37.8 % actives vs 4.4 % / 11.8 % (EF 8.7× / 3.2×). **Docking↔proxy agreement on the actives is weak** (Pearson 0.333, Spearman 0.422, n = 958) — the proxy is a loose surrogate for docking, not a tight one. Note docking is stochastic and this used n_conformers = 1, single replica; the rgfn AUROC drops 0.864→0.780 under MW-matching (part of that separation was the RGFN decoys' large size), while the already-matched decoys are unchanged (0.730/0.740).
