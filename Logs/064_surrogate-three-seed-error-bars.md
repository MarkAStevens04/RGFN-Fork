# sEH + DRD2 — how much of our advantage is luck? Three seeds per cell, and a new headline bar
**Date:** 2026-08-18, ~2pm

## Question

If we retrain each generator from scratch with a different random start, does our library-building
advantage stay the same size?

## Context & Summary

Every reactions-per-mode number we have published so far came from a **single** trained model per
cell. Entry `055` swept the fast-surrogate half of the benchmark exhaustively and found hub-batching
cheaper in 380 of 380 settings; entry `058` did the same against real GPU docking and found it cheaper
in all six cells. Both entries put the identical caveat at the top of their own "what reviewers will
still want" list: **one seed, no error bars.** A reviewer cannot tell a 3× effect from a 3× fluke
without knowing how much the number moves when you change nothing but the random seed.

The models to answer that already existed. The generator benchmark (entry `063`) trained all four
generators against all four scoring systems **three times each**, so 16 more fully-trained surrogate
checkpoints were sitting on disk — meaning this experiment needed no training at all, only the
enumeration and selection stages. That is the cheap half of the matrix: measured at 74 GPU-hours per
seed against 588 for a docking seed.

This entry enumerates and evaluates all eight surrogate cells at seeds 43 and 44, puts a spread on
every ratio, and — prompted by what the three seeds revealed — **moves the headline sEH hit bar from
7.0 to 5.0**.

## Answer

The advantage is stable: across the five cells with three complete seeds and both strategies reaching
the full library budget, the median seed-to-seed variation is **3.2%** on a mean advantage of
**3.29×**. That is the same order as the ±2% we measured for docking-oracle noise in entry `058`, so
both known sources of run-to-run variation are small compared with the effect they might have
explained away.

Two things came out of the seed data beyond the error bar itself. The old **bar of 7.0 was not
measuring what we thought**: at 7.0 an arm falls short of the library budget in three of eight cells on
*every* seed, so those ratios were computed over starved libraries and understated us — moving to the
calibrated bar of 5.0 lifts RGFN-sEH from 1.34× to **2.32×** and makes the comparison honest. And an
apparent seed *ordering* — seed 42 beating 43 beating 44 — turned out to be an artifact of our own
tooling rather than anything about the models.

## Relevance to our Publication

This closes the last item that appeared on both entry `055`'s and entry `058`'s own list of gaps, and
it is the one a **Digital Discovery** reviewer is most likely to raise first: a methods paper claiming
a 2–4× improvement needs to show the number does not move that much on its own. We can now quote every
surrogate cell as mean ± spread over three independent training runs.

It also removes a soft spot in how we were quoting sEH. Choosing 5.0 is defensible from the
calibration work (entries `034`/`051`) rather than convenient, and 7.0 stays reported as the
paper-comparable variant, so we are not accused of having picked the bar that flatters us.

## Next Experiments

**Refining for publication**

- **Finish the two RxnFlow cells honestly.** They are the only cells still pool-limited at bar 5.0, and
  their hub pools differ by seed (200/177/105 hubs), so part of their larger spread is library size
  rather than method. Worth reporting separately rather than folding into a median.
- **The same treatment for the docking half.** Seed 43 is now running for the two cheap ClpP cells;
  the expensive SCENT cells are the open question.
- **`scent_6td3` still carries the truncation `scent_clpp` had** — 51 capped hubs. Uncapping moved
  `scent_clpp` by 10.6%, so this is the largest remaining "our number is conservative" caveat.

**Next steps in project**

- Price these libraries through the competitor pipeline (entry `056`) so the surrogate cells get the
  same head-to-head cost comparison, now with error bars on our side of it.
- Feed the hub selections into the active-learning loop, which is what the selection machinery is for.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/lsd_hubs/matrix16/manifest_seed43.csv`, `manifest_seed44.csv` — the per-seed cell
  manifests, read via `MATRIX16_MANIFEST`. Previously held docking rows only; this entry added the
  eight surrogate rows to each.
- `./experiments/lsd_hubs/matrix16/submit_cell.sh` — the surrogate sample→pick_hubs→enumerate launcher.
- `./experiments/lsd_hubs/matrix16/harvest_surrogate.sh` — polls for finished surrogate enumerations and
  runs the campaign. Readiness is **hub coverage against `hubs.csv`**; see Results for why an
  existence test was not enough.
- `./experiments/lsd_hubs/matrix16/gate_curve.py` — the in-process bar sweep, run at each target's own
  declared `threshold_variants` (sEH 5/6/7, DRD2 0.5/0.7/0.9).
- `./experiments/lsd_hubs/matrix16/targets.py` — where the headline bar lives; the sEH change and its
  justification are recorded in the comment beside it.

**Models** — 16 checkpoints, all pre-existing from entry `063`'s training campaign:
`/scratch/markymoo/rgfn_runs/experiments/fixed_reward/{rgfn,scent}_{seh,drd2}_5k/seed{43,44}/train/checkpoints/last_gfn.pt`
and `.../{rxnflow,fraggfn}_{seh,drd2}_5k/seed{43,44}/checkpoints/last_gfn.pt`. All read 4999–5000
iterations. SCENT additionally needs its `guidance_models.pt` sidecar (entry `024`) and its
`additional_fragments/fragments_*.json` promoted-fragment snapshot, without which its nested cost model
silently changes.

`fraggfn_drd2` needed care: the seed-42 manifest row points at a `maxfrag6/<timestamp>` path because
the cap-9 model mode-collapsed (entry `046`). Each run's own `run_config.yaml` confirms
`max_nodes: 6` for `fraggfn_{seh,drd2}` on all three seeds, so the regular `_5k/seedN` path is the same
model generation and the three seeds are comparable.

**Results** (committed)

- `./experiments/lsd_hubs/matrix16/results_seed{43,44}/<cell>/summary.json` — per-cell head-to-head.
- `./experiments/lsd_hubs/matrix16/results{,_seed43,_seed44}/gate_curve/<cell>/` — the bar sweeps.

**Job Logs** — `/scratch/markymoo/rgfn_runs/{s43,s44}_surr_*-*.out` (the 16 enumerations),
`harvest_s{43,44}.log`, `reharvest.log`, `bar5_drd2.log`.

## Relevant Versions

```
1ef9d9f  Surrogate seeds 43/44: there is NO seed trend — the apparent one was my harvester bug
09698e7  harvest_surrogate: readiness must be hub COVERAGE, not file existence
b706304  Queue all three surrogate seeds: 16 cell-enumerations + a surrogate harvester
```

The bar-5.0 switch in `targets.py`, the recomputed sEH campaigns, and the corrected DRD2 sweeps are
`cb85434`.

## Relevant Resources

**Sources** — the sEH proxy is the frozen Bengio-2021 model (`[bengio2021gflownet]`); its calibration
against known actives is entry `034` (AUROC 0.76/0.68; 0 of 2,315 real actives reach 8.0) and the
diversity-vs-bar coupling is entry `051`. DRD2 is the PyTDC oracle, env-invariant to 12 d.p.

**Packages** — RDKit (Morgan r=3 fingerprints, Tanimoto) via
`glue/samplers/lsdflow/mode_select.py`; the count-once cost model in
`glue/samplers/lsdflow/campaign.py` (entry `033`).

## Method

1. **Add the surrogate rows** to `manifest_seed43.csv` / `manifest_seed44.csv`, re-pointing each
   seed-42 checkpoint path at the corresponding seed, and verify every checkpoint exists and reports
   4999–5000 iterations (`manifest.py --scan-epochs`).
2. **Smoke the one redirected row** (`fraggfn_drd2` seed 43, 400 trajectories / 4 hubs) before
   committing 16 jobs to the queue.
3. **Enumerate** all 16 cells (`submit_cell.sh`), with walltime per generator taken from the measured
   seed-42 enumeration cost rather than a flat guess — RGFN's 31.5 GPU-h enumeration is a single
   unsliceable job and needed the 3-day maximum, FragGFN's 0.3 GPU-h needed 6 hours.
4. **Campaign each cell** once its enumeration covers every hub, then **sweep bars** at each target's
   declared variants.
5. **Move the sEH headline bar** to 5.0 in `targets.py` and recompute all 12 sEH campaigns.

## Results

**The headline, at the new bars** (sEH 5.0, DRD2 0.5). Budget 300 modes, similarity 0.5, Morgan r=3.
"all 300" means both strategies reached the full budget on **every** seed:

| cell | bar | s42 | s43 | s44 | mean | sd | CV | all 300 |
|---|---|---|---|---|---|---|---|---|
| `fraggfn_seh` | 5.0 | 4.21× | 4.07× | 4.07× | **4.12×** | 0.078 | **1.9%** | yes |
| `fraggfn_drd2` | 0.5 | 3.99× | 4.06× | 3.79× | **3.95×** | 0.138 | 3.5% | yes |
| `scent_drd2` | 0.5 | 3.17× | 3.35× | 3.20× | **3.24×** | 0.094 | 2.9% | yes |
| `scent_seh` | 5.0 | 3.10× | 2.92× | 2.94× | **2.99×** | 0.095 | 3.2% | yes |
| `rgfn_seh` | 5.0 | 2.32× | 2.15× | 2.05× | **2.17×** | 0.140 | 6.4% | yes |
| `rxnflow_seh` | 5.0 | 2.31× | 2.31× | 1.80× | 2.14× | 0.297 | 13.9% | no |
| `rxnflow_drd2` | 0.5 | 1.60× | 1.96× | 1.70× | 1.75× | 0.187 | 10.7% | no |
| `rgfn_drd2` | 0.5 | 3.21× | — | — | (n=1) | | | yes |

Across the **five clean cells**: median CV **3.2%**, mean advantage **3.29×**.

**Why the bar moved.** At the old bar of 7.0 the same five cells read 1.28× / 2.60× / 3.24× / 3.60× /
3.95×, and three of the eight cells had an arm short of 300 modes on *every* seed. The clearest case is
`rgfn_seh`: at 7.0 it reads **1.34× with best-candidate assembling only 9 modes**, at 5.0 it reads
**2.32× with both arms at 300**. Entry `055` observed this on one seed; three seeds show it is
systematic rather than seed luck, which is what settled the switch. 7.0 stays in
`threshold_variants`, so the paper-comparable number remains available.

**There is no seed ordering, and the apparent one was our tooling.** The first pass showed seed 42
beating 43 beating 44 across several cells — implausible for random seeds, and false. Three checks
located the cause:

1. *The models are equivalent.* `scent_seh` sample rewards are mean 7.486 / 7.498 / 7.472 and 83.4 /
   83.7 / 83.2 % above the gate for seeds 42/43/44 — indistinguishable — yet the campaigns reported
   300 vs 106 vs 86 modes. `rxnflow_seh`'s seed 44 is by a wide margin the **best** model of its three
   (mean 7.824 vs 5.641 / 5.358), the opposite of the claimed order.
2. *No code drift.* Re-running the seed-42 `scent_seh` campaign with current code reproduced the
   entry-`055` numbers bit-for-bit (1.303 r/m, 300 modes, 57 hubs), ruling out the mid-flight-edit
   failure mode of entry `062`.
3. *The fingerprint.* Seeds 43/44 walked exactly **10 hubs** and stopped `pool_exhausted`, where seed
   42 walked 57 and stopped on the mode budget. Ten is a `PartialFlusher` boundary.

`harvest_surrogate.sh` tested readiness with `[ -s enum_children.json ]`. But every worker flushes a
**partial** enumeration every 10 hubs — deliberate walltime insurance — so the file appears minutes
into a 15-hour enumeration and grows. The harvester campaigned each cell on whatever had been flushed,
then marked it done. Fourteen of sixteen campaigns were affected; the two exceptions were the FragGFN
cells, whose enumerations take 28 minutes and therefore finished before the first poll — which is why
FragGFN was the only cell showing no trend, a control hiding in plain sight. Seed 43 merely tended to
be a little further along than seed 44 whenever the poll landed, which manufactured the ordering.

Readiness is now hub coverage against `hubs.csv`, the same standard `merge_docking_slices.sh` already
enforced for docking cells, verified to refuse a 70/200 cell and accept a 200/200 one. All 14 campaigns
were recomputed from complete enumerations. Nothing incorrect reached git — the affected result
directories were all untracked when the problem was found. **This is the third time in this project
that "ready" meant less than it sounded** (`docking_wired` in entry `057`, the stale epoch sidecar, and
now this), each time a cheap proxy standing in for the property actually wanted.

**Two caveats that are properties of the cells, not bugs.** `rxnflow_drd2`'s hub pool differs by seed —
**200 / 177 / 105** hubs — because `pick_hubs` is capped by the number of distinct parents among the
top-1000 candidates (entry `031`), so part of its 10.7% spread is library size rather than method. And
both RxnFlow cells remain pool-limited even at bar 5.0, so their ratios are not strictly like-for-like
and are excluded from the median above.

**Cost.** 113 GPU-hours for all 16 enumerations, against a measured 74 GPU-h per seed for the
enumeration itself; the remainder is the RGFN cells, which dominate at 31.5 and 23.9 GPU-h each while
FragGFN's are 0.3. `rgfn_drd2` seeds 43/44 were still enumerating at the time of writing (100/200 and
70/200 hubs), so that cell remains n=1.

**Caveats**

- Five of eight cells have a three-seed error bar; `rgfn_drd2` is n=1 and the two RxnFlow cells are
  n=3 but not budget-comparable.
- Three seeds give a spread, not a confidence interval. The CVs quoted are descriptive.
- FragGFN remains a cost-model **control** throughout (its attachments are not synthesis steps), even
  though it posts the highest ratios.
- The sEH bar move is a reporting decision, not a re-measurement: the same enumerations are re-scored.
