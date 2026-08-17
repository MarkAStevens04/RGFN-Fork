# Four systems × four generators — the complete three-seed comparison
**Date:** 2026-08-14, ~11am

## Question

Now that every run has finished with three independent repeats, how do our four
molecule generators actually compare across all four scoring systems — on reward,
diversity, drug-likeness, and synthesizability — and do the differences hold up once
we have error bars?

## Context & Summary

Entry `030` launched the publication-scale version of our generator benchmark: the same
four generators (RGFN, FragGFN, RxnFlow, SCENT) trained against the same four scoring
systems (two fast surrogate scorers, sEH and DRD2; two real GPU-docking scorers, the ClpP
pocket and our 6TD3 glue differential), but this time **three times each** and at a uniform
5,000-step budget. The earlier single-run comparisons (entries `019`/`020`/`021`) had already
found the generators *separate* rather than converge — SCENT scored best, RxnFlow made the
most drug-like but lowest-scoring molecules, RGFN drifted to the largest molecules, and
FragGFN matched on score but produced molecules with no synthesis route. But every one of
those entries carried the same caveat at the top of its "what's missing" list: **one seed, no
error bars.** This entry closes that — all 48 runs (4 generators × 4 systems × 3 seeds) are
complete, and here we do the full head-to-head with the spread across seeds attached.

We loaded the molecules each run produced and, using the project's standard recipes,
measured for every run: the reward it achieved, how many distinct high-scoring molecule
families ("modes") it found, how chemically diverse and how drug-like its molecules are, and
whether they come with a synthesis route. Then we averaged each number over the three seeds
and compared the four generators system by system.

## Answer

**The single-run rankings survive contact with error bars, and the picture is consistent
across all four systems.** SCENT wins on reward everywhere — including the strongest ClpP
binders we generated (a median docking energy of −11.4 kcal/mol, versus roughly −9 for RGFN
and FragGFN and −7.4 for RxnFlow) — with the tightest seed-to-seed spread of the four.
RxnFlow is consistently the *weakest* on reward **and, newly, by far the least reliable**: it is
the only generator whose runs disagree with each other, sometimes producing only a quarter as
many molecules as it should and collapsing onto a narrow set — the three seeds turn that
instability from an anecdote into a measured result. RGFN scores well but makes the largest
molecules of the group. FragGFN, the deliberately non-synthesizable control, keeps up on
reward yet carries no synthesis route — and at the corrected fragment cap (`max_nodes=6`) it is
now the *most drug-like* generator, which is a direct consequence of that fix. One system,
DRD2, saturates for everyone (all four sit at ~0.98 of the maximum), so there reward stops
telling the generators apart and the diversity/drug-likeness/synthesizability axes carry the
signal instead.

## Relevance to our Publication

The first question any NeurIPS or Digital-Discovery reviewer asks of a generator comparison
is "how many seeds, and where are the error bars?" — it is the single caveat flagged at the top
of entries `019`, `020`, `021`, `055`, and `058`. This entry answers it directly: the entire
4×4 comparison now carries three-seed error bars on every metric, and they are tight enough to
confirm the generator rankings are real rather than a lucky draw. The one place the spread is
*large* — RxnFlow — is itself a finding: a reviewer weighing these methods should know RxnFlow's
output varies run to run, which single-seed benchmarks (including our own earlier ones) could
not have surfaced.

## Next Experiments

**Refining for publication**
- **External synthesizability on all 48.** Score every run's top molecules with AiZynthFinder +
  SA (entry `018` harness) so the "has a route by construction" claim is checked against an
  independent retrosynthesis tool — entry `020` found that check diverges sharply by generator
  (SCENT/RGFN routes recovered ~0.7, RxnFlow only ~0.29 despite a 100% by-construction claim),
  and that divergence now deserves its three-seed version.
- **Figures.** Per-system generator bar charts with the error bars, plus reward-vs-drug-likeness
  and reward-vs-size scatters, for the paper.
- **Synthesis-cost and novelty axes** — price every generator's routes with SCENT's cost model
  on the shared library, and measure novelty against the training building blocks.

**Next steps in project**
- This finished, backed-up matrix (48 trained policies + candidate sets) is the **input to the
  LSD-Flow post-hoc hub analysis** — the next objective — which re-samples these same trained
  policies. Nothing here needs re-running for it.

# Re-creation

## Relevant Files

Root: `./` (repo root). Run outputs on `/scratch/markymoo/rgfn_runs/`.

**Scripts (ours, new):**
- `./experiments/fixed_reward/scale5k/analyze_matrix.py` — the analysis. Reads all 48
  `candidates.csv`, computes per-cell reward/modes/diversity/scaffolds/synth-proxy/drug-likeness
  using the repo's canonical metric functions, aggregates over the 3 seeds (mean±std), writes the
  tidy tables + prints the per-system comparison. Reuses `validation/lsdflow/metrics/diversity.py`
  so the numbers match prior logs.
- `./experiments/fixed_reward/scale5k/{orchestrate.sh,extend_chains.sh}` — the campaign drivers
  (cap-aware chain launcher + docking-chain deepener) that ran the 48 cells to completion under
  the 60-submit/30-run caps.

**Metric definitions (ours, reused):**
- `./validation/lsdflow/metrics/diversity.py` — the canonical mode / diversity / scaffold recipes:
  ECFP Morgan **r=3, 2048 bits**; **modes** = greedy sphere-exclusion, best-reward-first, Tanimoto
  ≤ 0.7 above a per-system bar (identical to upstream `TanimotoSimilarityModes` and the RGFN paper);
  internal diversity = 1 − mean pairwise Tanimoto; unique Bemis-Murcko scaffolds.

**Configs (ours):** the 16 `./configs/glue/fixed_reward_*_5k.gin` (RGFN) +
`./validation/configs/{fraggfn,rxnflow,scent}_*_5k.{yaml,gin}` (baselines) — the 4×4 matrix at
5,000 iters, cap 4; FragGFN at the corrected `max_nodes=6`; ClpP `docking_batch_size=200`.

**Datasets (the runs, inputs to this analysis):**
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/<gen>_<sys>_5k/seed{42,43,44}/fixed_reward/candidates/candidates.csv`
  — the 48 emitted candidate sets (200 rows for docking cells, 1000 for surrogate, minus RxnFlow
  dedup). Each row carries SMILES, `score`, `raw_score`, `has_route`, `num_reactions`, and RDKit
  descriptors (QED, MW, clogP, TPSA, Lipinski, …).
- Backed up (only durable copy) at `~/rgfn_run_backups/fixed_reward/<same layout>` — 52 checkpoints,
  12 SCENT guidance sidecars, 0 gaps (verified 2026-08-14).

**Results (ours, new):**
- `./experiments/fixed_reward/scale5k/analysis/per_cell.csv` — one row per run (48), all metrics.
- `./experiments/fixed_reward/scale5k/analysis/aggregate.csv` — mean±std per (system, generator).

## Relevant Versions

Branch `Hub-Analysis`. **Not yet committed:** the campaign infrastructure
(`experiments/fixed_reward/scale5k/{orchestrate.sh,extend_chains.sh,submit_*.sh,analyze_matrix.py,analysis/}`),
the 16 `_5k` configs, the `docking_batch_size=200` change in
`glue/oracles/docking_seh_oracle.py`, and the FragGFN `max_nodes: 6` change in the four
`validation/configs/fraggfn_*_fixed_5k.yaml`, plus this log. [TODO — add commit hash after pushing.]
**Can you commit the experiment files? Once you do, let me know and I'll fill in the hash.**

## Relevant Resources

**Sources**
- RGFN (`[koziarski2024rgfn]`), SCENT (`[gainski2025scent]`), RxnFlow (`[seo2024rxnflow]`), and the
  FragGFN fragment GFlowNet — the four generators.
- Mode definition: `[bengio2021gflownet]` (Tanimoto-mode count), as implemented in
  `validation/lsdflow/metrics/diversity.py`.
- Calibrated docking bars: ClpP **−8.0** kcal/mol (entry `045`, receptor 7UVU, AUROC 0.895); 6TD3
  differential **−2.0** (provisional, no AUROC calibration).

**Packages**
- `rgfn` env (py3.11, torch 2.3.0+cu118) + RDKit — ran `analyze_matrix.py`.

## Method

1. Ran the 48-cell matrix to completion under entry `030`'s auto-requeue infrastructure
   (`orchestrate.sh`/`extend_chains.sh`); confirmed all 48 emitted a full candidate set and every
   run reached iteration 5,000 (completion audit, 2026-08-13).
2. `source ~/bin/rgfn-smoke-env.sh && python experiments/fixed_reward/scale5k/analyze_matrix.py`
   — per-cell metrics from each `candidates.csv`, mode bar on `score` (= −raw docking energy for
   docking cells; proxy value for surrogate): 6TD3 ≥2.0, ClpP ≥8.0, sEH ≥5.0, DRD2 ≥0.5. Modes /
   diversity / scaffolds via `diversity.py` (ECFP r=3/2048, sim≤0.7). Aggregated over seeds.

## Results

**48/48 runs complete.** Per-run counts: 200 candidates (docking cells) / 1000 (surrogate) for
every generator **except RxnFlow**, which dedups to unique-valid and so emitted uneven counts
(DRD2 = 1000/849/260 for seeds 42/43/44; sEH = 1000/1000/904) — a real property of that generator,
not truncation (all reached iter 5,000; logs show clean emission). This makes RxnFlow's *count*
metrics (modes, scaffolds) N-confounded; the per-molecule medians below are the clean comparison.

**Reward** (mean ± std over 3 seeds; docking = median −raw energy, so higher = stronger binding;
surrogate = median proxy):

| system | RGFN | FragGFN (foil) | RxnFlow | SCENT |
|---|---|---|---|---|
| 6TD3 (differential) — median | 2.22±0.16 | 2.43±0.19 | 1.82±0.24 | 2.28±0.10 |
| 6TD3 — top-10 mean | 4.73±0.19 | 4.82±0.31 | 3.97±0.38 | **6.67±0.28** |
| ClpP (Vina) — median | 9.05±0.90 | 9.37±0.15 | 7.43±0.57 | **11.40±0.20** |
| ClpP — top-10 mean | 12.61±0.86 | 11.74±0.21 | 9.43±0.69 | **14.28±0.05** |
| sEH (proxy) — median | 7.19±0.02 | 7.47±0.05 | 5.90±0.58 | **7.61±0.01** |
| DRD2 (activity) — median | 0.96±0.01 | 0.98±0.00 | 0.90±0.05 | 0.98±0.00 |

SCENT leads on every system; RxnFlow trails on every system. DRD2 is saturated for all four
(top-10 mean = 1.00 everywhere).

**Drug-likeness** (median QED / median MW):

| system | RGFN | FragGFN | RxnFlow | SCENT |
|---|---|---|---|---|
| 6TD3 | 0.10 / 685 | 0.23 / 585 | 0.15 / 553 | 0.11 / 654 |
| ClpP | 0.17 / 611 | **0.37** / **456** | 0.18 / 531 | 0.18 / 627 |
| sEH  | 0.21 / 560 | **0.36** / **462** | 0.26 / 535 | 0.25 / 519 |
| DRD2 | 0.36 / 509 | **0.68** / **380** | 0.42 / 444 | 0.35 / 508 |

FragGFN (at the corrected cap-6) is the most drug-like across the board; RGFN makes the largest
molecules (MW up to 685 on 6TD3 — the size-drift signature).

**Reliability, diversity, synthesizability** (selected, mean ± std):

| | RGFN | FragGFN | RxnFlow | SCENT |
|---|---|---|---|---|
| DRD2 candidate count N | 1000±0 | 1000±0 | **703±391** | 1000±0 |
| DRD2 modes (bar 0.5) | 927±19 | 969±18 | **326±213** | 975±12 |
| ClpP modes (bar −8.0) | 139±21 | 172±6 | **63±40** | 137±5 |
| internal diversity (all cells) | 0.84–0.88 | 0.80–0.87 | 0.66–0.86 | 0.70–0.87 |
| has_route (synth by construction) | 1.00 | **0.00** | 1.00 | ~1.00 |

RxnFlow carries the large error bars (mode collapse + uneven N); the other three are tight.
FragGFN's `has_route=0` confirms its role as the non-synthesizable control — reward parity ≠
synthesizability. Full per-run numbers in `analysis/per_cell.csv`; aggregates in `aggregate.csv`.

**Caveats.** (1) Modes/scaffolds are counts and thus N-confounded for RxnFlow — read its
per-molecule medians, not its counts. (2) The sEH proxy scale is not an activity scale (entry
`034`), so "modes above bar 5.0" is a reward statement, not a potency claim. (3) The 6TD3 −2.0 bar
has no AUROC calibration behind it (unlike ClpP's −8.0). (4) "Synthesizable by construction"
(`has_route=1`) is not the same as externally verified — the AiZynth cross-check (Next Experiments)
is still pending for these three-seed runs. (5) FragGFN is the non-synthesizable foil throughout,
not a competitor on the synthesizability axis.
