# FragGFN–DRD2 — recovering the paper's ~0.9 reward (the fragment-cap fix)

**Date:** 2026-07-23, ~4:30pm

## Question

Why does our FragGFN reproduction score ~0 on DRD2 when the RGFN paper reports it reaches
~0.9 — and what one setting recovers the paper's result?

## Context & Summary

In the publication-scale 4×4 matrix ([030]), the FragGFN generator on the DRD2 target
**collapses**: mean DRD2 activity ~0.03, zero "hit" molecules, while the other three
generators reach 0.9+. Entry [019] diagnosed this as "size/chemotype drift" — FragGFN
builds oversized molecules (MW ~664) that the DRD2 model, which rewards a narrow
*drug-sized* chemotype, scores near zero — but two attempted fixes (a fixed reward
temperature, then the exploration-annealed temperature) both flatlined, and the cause was
never tied to a concrete config setting. It was flagged as possibly an intrinsic limit of
fragment assembly. Critically, this *diverges from the RGFN paper itself*, which reports
FragGFN optimizes DRD2 fine.

This entry closes that gap by comparing our config against the paper's actual recipe
(Appendix B.2) line by line, finding the single setting that differs, and testing whether
changing it recovers the paper's result.

## Answer

The difference is one number: the **maximum fragment count**. The paper caps FragGFN at
**6 fragments** ("*a maximum fragment count of 6 as opposed to RGFN's 5*"); our matrix used
FragGFN's native default of **9**. Everything else already matched the paper — the reward
transform (`exp(β·proxy)`), the fixed β=48, the 72-fragment bengio2021flow library, and the
TDC DRD2 oracle (all verified in code). Those three extra fragments are exactly what
produced the MW ~664 oversized molecules. Setting `max_frags=6` **recovers the paper**: a
short 1,500-step run reaches mean DRD2 **0.908** (median 0.982), **93%** of samples are
hits, and median molecular weight drops from 664 to **385** (drug-sized). This also explains
why the earlier temperature fixes failed — the exponent/temperature was never the difference,
the molecule-size cap was, so more exploration at max-9 just produced more oversized
molecules. It is not a fragment-assembly limitation; it was a mis-set cap.

## Relevance to our Publication

FragGFN is a headline baseline in the four-way comparison. A blank/failed DRD2 cell — one
that *contradicts the very paper we cite for the baseline* — is a direct reviewer liability.
This entry makes the FragGFN-DRD2 result match the published baseline and, more importantly,
identifies that our whole FragGFN column ran at `max_frags=9` rather than the paper's 6,
which is a fidelity issue reviewers comparing to the RGFN paper would catch.

## Next Experiments

**Refining for publication.**
- **Decide the FragGFN fragment cap for the whole matrix.** All four FragGFN cells
  (sEH/6TD3/ClpP too) ran at `max_frags=9`, not the paper's 6. It only *broke* DRD2 (its
  reward punishes size) — sEH's large lipophilic pocket happens to reward big molecules, so
  those cells "worked" but are still off-spec. For paper-faithful, internally-consistent
  FragGFN, re-run all four at `max_frags=6` (a methodology call — pending).
- Confirm the full 5,000-step DRD2 run lands near the smoke's 0.9 (job 71433).

**Next steps in project.**
- Fold the recovered DRD2 cell into the LSD-Flow analysis in place of the failed one.

# Re-creation

## Relevant Files

- `validation/configs/fraggfn_drd2_maxfrag6.yaml` — the recovery config: identical to
  `fraggfn_drd2_fixed_5k.yaml` except `gflownet.max_nodes` 9 → **6** (the paper's FGFN cap).
- `experiments/fixed_reward/fraggfn_drd2/submit_fraggfn_drd2_maxfrag6.sh` — full 5,000-step
  retrain (compute, 1 GPU) → `experiments/fixed_reward/fraggfn_drd2_maxfrag6/seed42`.
- `validation/generators/fraggfn/run_fraggfn_fixed.py` — runner; reads `gflownet.max_nodes`
  into `cfg.algo.max_nodes`, which `FragMolBuildingEnvContext(max_frags=...)` consumes.
- `validation/generators/fraggfn/fixed_reward.py` — `DRD2FrozenReward.reward()` =
  `exp(clip(v))`, β applied by the task → `exp(β·v)` (matches the paper; verified).
- `validation/generators/fraggfn/task.py` — `FragGFNTrainer` (bengio2021flow fragments,
  constant-β temperature).
- `oracle/drd2_current.pkl` — cached TDC DRD2 SVM (reproduces `tdc.Oracle("DRD2")`), the reward.

**Results (smoke)**
- `/scratch/markymoo/rgfn_runs/fraggfn_maxfrag6_smoke/.../fixed_reward/candidates/candidates.csv`
  — 300 sampled molecules + DRD2 scores.

**Job Logs**
- Smoke: login A100, 2026-07-23 16:12–16:38 (1,500 steps). Full run: job **71433** (queued).

## Relevant Versions

New/edited files on branch `Hub-Analysis`, **not yet committed**. `[TODO — add commit hash
after pushing]`. Files: `validation/configs/fraggfn_drd2_maxfrag6.yaml`,
`experiments/fixed_reward/fraggfn_drd2/submit_fraggfn_drd2_maxfrag6.sh`.

## Relevant Resources

**Sources**
- `[koziarski2024rgfn]` — RGFN paper, **Appendix B.2**: FGFN max fragment count = 6;
  reward `R(x)=exp(β·score)`, β=48 for DRD2; batch 100, 4,000 steps; DRD2 proxy = TDC SVM
  (ECFP6, Gaussian kernel = Olivecrona 2017). arXiv:2406.08506.
- `recursionpharma/gflownet` — the FragGFN implementation (`FragMolBuildingEnvContext`,
  `bengio2021flow.FRAGMENTS`).
- Entry [019] — prior FragGFN-DRD2 diagnosis (size drift; two failed temperature fixes).

**Packages**
- `gflownet` (recursion) FragGFN; `rdkit`; `scikit-learn` (cached DRD2 SVM); torch (cu118).

## Method

1. **Diff vs paper.** Extracted the paper's FragGFN/DRD2 recipe (App. B.2) and compared to
   our config + code: reward transform, β, fragment library, DRD2 oracle all match; only
   `max_frags` differs (paper 6, ours 9).
2. **Config.** Wrote `fraggfn_drd2_maxfrag6.yaml` = `fraggfn_drd2_fixed_5k.yaml` with
   `max_nodes: 6`.
3. **Smoke.** `run_fraggfn_fixed.py --cfg fraggfn_drd2_maxfrag6.yaml --n-train-steps 1500
   --n-samples 300` on a login A100 (fraggfn env); scored the 300 samples with the DRD2 oracle.
4. **Full run.** `sbatch submit_fraggfn_drd2_maxfrag6.sh` (5,000 steps) → job 71433.

## Results

**DRD2 activity of sampled molecules** (higher = better; hit bar 0.5):

| run | max_frags | steps | DRD2 mean | median | modes (>0.5) | MW median |
|---|---|---|---|---|---|---|
| matrix cell (failed) | 9 | 5,000 | 0.03 | 0.029 | 0 / 1000 | 664 |
| constant-β (job earlier, [019]) | 9 | 5,000 | 0.035 | — | 0 / 1000 | ~680 |
| uniform-β ([019], job 69690) | 9 | 5,000 | 0.036 | — | 1 / 1000 | — |
| **max_frags=6 smoke (this entry)** | **6** | **1,500** | **0.908** | **0.982** | **280 / 300 (93%)** | **385** |
| RGFN paper (reported) | 6 | 4,000 | ~0.9 | — | — | — |

The `max_frags=6` run reaches the paper's ~0.9 in fewer steps (1,500 vs the paper's 4,000
and our failed runs' 5,000), and the molecular weight drop (664 → 385) confirms the size
mechanism. Full 5,000-step production run: job 71433 (pending).
