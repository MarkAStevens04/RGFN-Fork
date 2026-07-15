# SCENT — hub-batching vs best-candidate at relaxed "mode" thresholds (5 and 6)
**Date:** 2026-07-14, ~5pm

## Question

If we call a molecule a "hit" at the more realistic sEH-proxy score of 5 or 6 instead of the strict 7
we used before, does the hub-batching-vs-best-candidate comparison — the Pareto front and the
synthesis-cost curves — actually change?

## Context & Summary

The hub-batching campaign (entries `029`/`033`) counts a molecule as a diverse "hit" (a *mode*) only if
its sEH-proxy reward clears a fixed bar, which those entries set at **7.0**. Entry `034` then showed that
7.0 is the wrong bar: scoring 2,315 real, experimentally-confirmed ChEMBL sEH inhibitors with the exact
same proxy, **none reach 8.0 and only 9 reach 7.0** — real inhibitors top out around 7.7, and the
empirically meaningful "potential hit" line is far lower, roughly **5–6**. So the numbers in `029`/`033`
were computed against a bar that would reject essentially every known drug. This entry re-runs the whole
campaign at the calibrated bars of **5 and 6** to check whether the headline conclusion (hub-batching is
modestly cheaper per hit, paid for with a large enumeration-scoring bill) survives a more permissive,
better-grounded definition of "hit."

Because the strategies only re-select over molecules that were **already enumerated and scored** (the
cached 50-hub and 200-hub enumerations from `029`/`031`, plus the 30k sampled candidate pool), lowering
the threshold is a pure-CPU re-run — no GPU, no re-scoring. We regenerated the head-to-head point (cutoff
0.5) and both diversity sweeps (Pareto = modes at a fixed 100-reaction budget; cost = reactions to reach
300 modes) at thresholds 5 and 6, kept them in clearly-separated result dirs alongside the threshold-7
baseline, and drew a three-threshold overlay so the comparison is one figure.

## Answer

**Relaxing the mode threshold from 7 to 5 or 6 does not change the conclusion — the synthesis-cost
comparison is threshold-robust.** best-candidate is *byte-identical* at every diversity cutoff and every
threshold, because its top-300 diverse modes are all high-reward molecules anyway, so a lower bar never
admits anything new into the winning set. hub-batching's cost is likewise unchanged from mid cutoffs
(≈0.55) upward, and diverges only at the *strictest* diversity, where admitting weaker hits lets it reach
the 300-mode target a bit more cheaply (200-hub run, cutoff 0.30: **882 → 847 → 823** reactions as the
bar drops 7 → 6 → 5). The one genuine effect is on the *enumeration* axis, not synthesis: at cutoff 0.5 a
**single** hub now supplies all 300 modes instead of two, roughly **halving** hub-batching's reward-gen
bill (**18,856 → 9,428** calls), at the cost of a slightly weaker library (median mode reward 7.48 →
7.35). In short, the "hub-batching is ~1.13× cheaper on synthesis, and its real price is enumeration
calls" story from `033` holds under the calibrated hit definition — and the enumeration price actually
*drops* when the bar is set where real actives live.

## Relevance to our Publication

Entry `034` will force a reviewer at Digital Discovery / JCIM to ask the obvious follow-up: if 7.0 isn't a
real activity threshold, does the hub-batching result depend on that arbitrary choice? This entry answers
"no" directly — the reactions-per-mode comparison is invariant to the hit bar across the entire calibrated
5–7 range, so the contribution is not an artifact of a mis-set threshold. It also sharpens the
enumeration-cost narrative that `033` identified as the true cost of hub-batching: setting the bar where
real hits are found makes hub-batching *cheaper to run*, because one hub's neighborhood now yields enough
diverse hits without recruiting more scaffolds — exactly the efficiency axis the active-learning loop
(Objective 1's oracle-efficiency curve) has to balance.

## Next Experiments

**Refining for publication**
- **DRD2 through the same threshold check** — repeat this robustness pass on the second target so the
  invariance isn't presented as sEH-specific (ties into `033`/`034`'s DRD2 to-dos).
- **Probe the strict-cutoff floor at low thresholds** — the 50-hub run still can't reach 300 modes at
  cutoff 0.30 even at bar 5 (a scaffold-diversity floor, not a reward floor), and `031` found neither
  strategy reaches 300 below ~0.30. Sweeping cutoffs < 0.30 at bar 5 would confirm the floor is chemistry,
  not the bar.
- **An even lower sanity bar (~4)** as an extreme to confirm best-candidate stays invariant and to see
  where hub-batching's median-reward degradation becomes unacceptable.

**Next steps in project**
- Wire both strategies into the **active-learning loop** on the calibrated bar (they're already swappable)
  and measure the oracle-efficiency curve — the halved enumeration bill at the realistic threshold is
  directly what that curve rewards.
- Move from the raw proxy to a **docking-based hit definition** (≤ −9 kcal/mol, the interpretable scale
  `034` recovered) in a future campaign variant, since docking separated actives from decoys better than
  the proxy.

# Re-creation

## Relevant Files

Root: `./` (repo root).

**Analysis drivers (ours — `experiments/lsd_hubs/campaign/`):**
- `run_campaign.py` — single head-to-head point (both strategies, cutoff 0.5). **Unchanged**; the mode
  bar is the existing `--reward-threshold` CLI argument, so thresholds 5/6 need no code change.
- `sweep_campaign.py` — the diversity sweeps (Pareto + cost + budget-efficiency). **Unchanged**; same
  `--reward-threshold` argument. All outputs land in `results/<tag>/`, so a distinct `--tag` keeps each
  threshold's artifacts separate.
- `compare_thresholds.py` — **new this entry.** Reads the *committed* per-threshold sweep CSVs
  (`fixed_modes.csv`, `pareto.csv`) for thresholds 7/6/5 and overlays them (strategy = colour, threshold
  = linestyle) into `results/threshold_comparison/`. Pure CSV+matplotlib, no recompute.
- `README.md` — updated with a "Threshold variants (Logs/035)" subsection documenting the new result
  dirs and the overlay script.

**Strategy code (ours — `glue/`):** `glue/samplers/lsdflow/campaign.py` +
`glue/samplers/lsdflow/mode_select.py` — **unchanged.** The threshold flows into
`DiverseThresholdModeSelector`; the count-once cost model (`033`) is untouched.

**Results (committed, ours):** under `experiments/lsd_hubs/campaign/results/`:
- `scent_seh_thr5/`, `scent_seh_thr6/` — 50-hub head-to-head + sweep at bars 5 and 6
  (`summary.json`, `curve_*.csv/png`, `pareto.*`, `fixed_modes.*`, `budget_efficiency.*`, `sweep_summary.json`).
- `scent_seh_1kx200_thr5/`, `scent_seh_1kx200_thr6/` — 200-hub sweep at bars 5 and 6 (sweep artifacts only).
- `threshold_comparison/` — the four overlay figures (`cost_{50hub,200hub}.png`, `pareto_{50hub,200hub}.png`).
- Baselines (bar 7) are the pre-existing `results/scent_seh/` (50-hub) and `results/scent_seh_1kx200/`
  (200-hub) from `029`/`031`/`033` — **left untouched**; their `summary.json`/`sweep_summary.json` record
  `"reward_threshold": 7.0`.

**Inputs (on `$SCRATCH`, unchanged from `029`/`031`/`033`):**
- Sampled candidate pool: `/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/` (`records.csv` = 29,997
  candidates; `compositions.json`).
- 50-hub enumeration: `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70295/enum_children.json`
  (50 hubs, 148,153 scored children).
- 200-hub enumeration: `/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json`
  (200 hubs).
- Recipe snapshot (nested cost table):
  `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json`.

## Relevant Versions

Branch `Hub-Analysis`. New/modified: `Logs/035_hub-batching-mode-threshold-robustness.md`,
`experiments/lsd_hubs/campaign/compare_thresholds.py`, the four new
`experiments/lsd_hubs/campaign/results/scent_seh{,_1kx200}_thr{5,6}/` dirs +
`results/threshold_comparison/`, the campaign `README.md` threshold-variants note, and the
`RESEARCH_CONTEXT.md` index row. **Not yet committed.** [TODO — add commit hash after committing.]

## Relevant Resources

**Sources** — entry `034` (the sEH-proxy calibration that motivates the 5/6 bars: real inhibitors top out
~7.7, empirically useful cutoffs ~5–6), `033` (the fair count-once cost model these runs use), `029` (the
original comparison + 50-hub enumeration job 70295), `031` (the 200-hub scale-up, job 70363).
`[bengio2021gflownet]` (the pretrained sEH proxy + modes/top-k framing), `[gainski2025scent]` (SCENT
dynamic library).

**Packages** — `rgfn` conda env (campaign + analysis, CPU-only for this entry); RDKit (Murcko scaffold
counts, ECFP diversity in `mode_select`); matplotlib (`sweep_campaign.py`, `compare_thresholds.py`).

## Method

All on balam-login01 (A100 not used — pure CPU), `rgfn` env, after `source ~/bin/rgfn-smoke-env.sh` and
`export PYTHONPATH=$(pwd)` from the repo root. Inputs abbreviated `ANALYSIS` / `ENUM50` / `ENUM200` /
`SNAP` (paths above).

1. **Confirmed the cache spans the full reward range** (so bars 5/6 are computable without re-scoring):
   `records.csv` — 99.4 % ≥ 5, 97.6 % ≥ 6, 82.9 % ≥ 7; 50-hub enum children — 87.9 % ≥ 5, 63.6 % ≥ 6,
   26.3 % ≥ 7.
2. **50-hub head-to-head (cutoff 0.5):**
   `python experiments/lsd_hubs/campaign/run_campaign.py --analysis-dir $ANALYSIS --enum-children $ENUM50 --snapshot $SNAP --reward-threshold {5.0,6.0} --tag scent_seh_thr{5,6}`.
3. **50-hub sweep (0.30→0.90):**
   `python experiments/lsd_hubs/campaign/sweep_campaign.py --analysis-dir $ANALYSIS --enum-children $ENUM50 --snapshot $SNAP --reward-threshold {5.0,6.0} --tag scent_seh_thr{5,6}` (~65–75 s each).
4. **200-hub sweep (0.30→0.90):** same with `--enum-children $ENUM200 --tag scent_seh_1kx200_thr{5,6}`
   (~70 s each).
5. **Overlay:** `python experiments/lsd_hubs/campaign/compare_thresholds.py` → the four
   `results/threshold_comparison/` figures.

## Results

**Head-to-head (sEH, 50-hub, diversity cutoff 0.5) — hub-batching across the bar:**

| metric | bar 7 (`029`/`033`) | bar 6 | bar 5 |
|---|---|---|---|
| reactions for 300 modes | 819 | 825 | 825 |
| reward-gen (enumeration) calls | 18,856 | **9,428** | **9,428** |
| distinct hubs used | 2 | **1** | **1** |
| median / best mode reward | 7.48 / 8.40 | 7.35 / 8.38 | 7.35 / 8.38 |
| modes at 100-reaction budget | 33 | 33 | 33 |

best-candidate at cutoff 0.5 is **929 reactions / 29 modes@100rxn at all three bars** (identical). So the
head-to-head edge stays **~1.13×** (929 vs ~822); the only movement is hub-batching's enumeration bill
halving and its median mode softening.

**Cost to reach 300 modes vs diversity cutoff (reactions; "X" = can't reach 300):**

| cutoff | 50-hub HUB 7/6/5 | 50-hub BEST 7/6/5 | 200-hub HUB 7/6/5 | 200-hub BEST 7/6/5 |
|---|---|---|---|---|
| 0.30 | X / X / X | 1123 (all) | **882 / 847 / 823** | 1123 (all) |
| 0.35 | 807 / 803 / 798 | 1080 (all) | 812 / 815 / 809 | 1080 (all) |
| 0.50 | 819 / 825 / 825 | 929 (all) | 816 / 819 / 819 | 929 (all) |
| 0.70 | 706 (all) | 793 (all) | 770 (all) | 793 (all) |
| 0.90 | 644 (all) | 728 (all) | 822 (all) | 728 (all) |

- **best-candidate is identical across all three thresholds at every cutoff and both enum sizes** (the
  reward bar never binds — its 300 diverse winners are all high-reward). This is the built-in control.
- **hub-batching converges to the bar-7 curve from cutoff ~0.55 upward**; the threshold only bites at
  strict diversity. The clearest signal: 200-hub, cutoff 0.30 falls **882 → 847 → 823** as the bar drops
  — admitting weaker hits lets fewer/cheaper hub recruitments cover the 300-mode target.
- **The 50-hub cutoff-0.30 ceiling is a scaffold-diversity floor, not a reward-bar limit:** it can't reach
  300 mutually-Tanimoto-<0.30 modes even at bar 5, because the 50 enumerated hubs' children decorate too
  few cores (consistent with `031`'s chemistry floor). Lowering the bar doesn't create new scaffolds.

**Pareto — modes at a 100-reaction budget vs cutoff:** essentially threshold-invariant. best-candidate is
identical at every cutoff/bar (25→34 modes, 50-hub and 200-hub alike); hub-batching is identical except a
single point (200-hub, cutoff 0.30: 33 modes at bar 7 → 34 at bars 6/5).

**Overlay figures** (`results/threshold_comparison/`): in the cost plots the three best-candidate lines
(red) are perfectly superimposed — one visible curve — and the hub-batching lines (blue) overlap
everywhere except the strict-cutoff fan-out on the 200-hub panel, making the threshold-robustness visible
at a glance.
