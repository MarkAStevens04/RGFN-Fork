# Four generators × sEH — completing the surrogate matrix, recalibrating the hit bar, and mapping the advantage over both knobs

**Date:** 2026-08-04, ~11am

## Question

Does building a shared scaffold once and diversifying it beat assembling molecules independently for
*every* generator we benchmark — and does that answer survive wherever we set the two thresholds that
define a "useful, distinct molecule"?

## Context & Summary

Entry `050` built the 16-cell evaluation pipeline and ran six of the eight fast-reward cells, finding
hub-batching cheaper in all of them. Two gaps remained. First, two cells were simply missing: both RGFN
cells had never run, and the FragGFN sEH model had not finished training. Second, and more subtly, entry
`050`'s own threshold sweep hinted that the quality bar we quote our headline numbers at may be set too
high — entry `034` had already shown that bar (7.0 on the sEH scoring model) sits in a range real
inhibitors barely reach, and entry `051` showed the molecule pools start behaving differently past
about 7. Entry `052` then mapped the cost over *both* thresholds at once — quality bar against
distinctness requirement — producing a surface rather than two lines, but only for one generator
(SCENT). So the obvious question was left open: is the *shape* of that surface a property of the
method, or of SCENT?

This entry closes all three. The two missing cells were run to completion; every sEH cell was then
re-scored at the lower, better-calibrated quality bars; and the two-threshold surface was measured for
all four generators on identical grids.

## Answer

Hub-batching is cheaper than the baseline in **every single comparable cell of every surface we
measured** — 380 of 380 across four generators — so the advantage is a property of the selection
strategy, not of any one model or of where the two thresholds sit. More importantly, the quality bar we
had been quoting at was distorting the comparison: at the lower calibrated bar the advantage roughly
*doubles* for two of the four generators, and the four generators stop looking very different from each
other. Most of the apparent spread between models in our headline table was an artifact of a bar so
high that the baseline could not assemble a full library to be measured against.

## Relevance to our Publication

This turns the central claim from "holds on the cells we ran" into "holds on every cell, at every
threshold setting we tested." **Digital Discovery** or **JCIM** reviewers will press on exactly two
things here — whether the result is cherry-picked to one model, and whether it depends on arbitrary
cutoffs — and the four-generator surface answers both in one figure. The recalibration matters for a
different reason: it means we must choose our headline bar deliberately and say why, because the choice
changes two of the four headline numbers by nearly a factor of two. Quoting the high bar without saying
that the baseline is starved there would be the kind of thing a careful reviewer catches.

## Next Experiments

**Refining for publication**
- Additional random seeds per cell, so the per-cell margins carry error bars. Every number here is a
  single training run.
- Decide and document the headline quality bar. The lower bar is the defensible one (the baseline can
  fill a full library there); the higher bar becomes the stress test.
- Prune the superseded FragGFN rows from the threshold table before it goes in the paper — they come
  from a model we replaced.
- Convert FragGFN's numbers onto a comparable footing. Its "reactions" are fragment attachments rather
  than synthesis steps, so its cost is not directly comparable to the three reaction-based generators;
  running the proposed molecules through retrosynthesis would fix this.

**Next steps in project**
- Extend to the two protein-docking targets, where scoring each enumerated molecule needs the slow
  docking oracle rather than a fast scoring model. All 16 trained models are now available for this.
- Settle what the scaffold-uncertainty signal should measure before building the uncertainty-driven
  scaffold selection that entry `050` showed the current definition cannot support.

---

# Re-creation

## Relevant Files

Root: `./` (repo root `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`).

**Scripts — orchestration (`experiments/lsd_hubs/matrix16/`):**
- `submit_cell.sh` — one cell's GPU pipeline (sample → pick_hubs → enumerate). Now auto-skips a stage
  whose output already exists (`RESUME=0` forces a redo) so a requeue cannot silently repeat a finished
  30k-trajectory sample — the single most expensive accidental redo available here.
- `submit_timing.sh` / `merge_timings.sh` — re-enumerate an already-finished cell with the timers live,
  in disjoint round-robin hub slices, then union the slices. Used for the two RxnFlow cells, whose
  serial re-run would have been ~9 h. `merge_timings.sh` refuses to publish unless every enumerated hub
  has a timing row, because a walked-but-untimed hub contributes 0 s and would under-report silently.
- `gate_sweep.sh` — re-scores both strategies at each target's threshold variants over the cached
  enumeration. Used for the 5/6/7 sweep.
- `surface_all_generators.sh` — **new here.** Thin driver that runs `campaign/tau_similarity_surface.py`
  (entry `052`, the other agent's) once per generator with manifest-resolved paths and our
  per-generator child-policy convention. The surface maths, pool-limited hatching and plotting are NOT
  duplicated. One process per cell on purpose: the login node's CPU limit is per-process.
- `tau_curve_all_generators.py` — cross-generator diversity sweep (modes at a fixed reaction budget vs
  distinctness cutoff). Rejects a cell whose enumeration covers <90% of its own `hubs.csv`.
- `run_cell_campaign.sh` / `manifest.py` / `manifest.csv` / `targets.py` — per-cell readout and the
  single source of truth for paths and per-target thresholds.

**Scripts — shared:**
- `experiments/lsd_hubs/campaign/tau_similarity_surface.py` — the surface driver from entry `052`. One
  fix applied here: its figure subtitle was the literal string "SCENT sEH surrogate", which silently
  mislabelled all four of the new surfaces; it now derives the label from the tag.
- `validation/lsdflow/adapters/workers/rgfn_worker.py` — RGFN's per-env worker. Its enumeration is a
  single opaque call into `glue/samplers/lsdflow/rgfn_enumerate`, so its compute-time split is not
  observable from the worker; it records an exact per-hub **total** as `unattributed_s` with
  `component_split: "lumped"` rather than fabricating a breakdown.
- `scripts/backup_scratch_critical.sh` — **new here.** Copies the irreplaceable scratch artifacts to
  `$HOME` (not purge-eligible). Tier 1 = checkpoints + SCENT backward-policy sidecars + candidates;
  tier 2 = samples and enumerations.

**Models:** `$SCRATCH/rgfn_runs/experiments/fixed_reward/<cell>/seed42/**/checkpoints/last_gfn.pt`, one
per cell. `fraggfn_seh_5k/seed42` is the newly-trained **cap-6** model (job 71742); the superseded cap-9
run is kept beside it as `seed42_maxnodes9` and must not be confused with it — the pairing was verified
three ways before use (see Method 4).

**Results (committed):** `experiments/lsd_hubs/matrix16/results/`
- `<cell>/{summary.json,curve.png,compute_time.png,compute_time.csv}` — all 8 surrogate cells now have
  both figures; before this entry only 3 had measured compute time.
- `<cell>_thr{5,6,7}/` + `gate_sweep/summary.csv` — the 20-point threshold table.
- `surface/<cell>/{surface.csv,surface.json,surface.png,surface.pdf}` — the four two-knob surfaces.
- `tau_curve_all/{seh,seh_gate5,seh_with_fraggfn_control}/` — cross-generator diversity sweeps at the
  high bar, the calibrated bar, and with the non-reaction control included.

**Job Logs:** `/scratch/markymoo/rgfn_runs/m16_{rgfn_seh-71981,rgfn_drd2-71982,fraggfn_seh-72178}.{out,err}`,
`m16_t_rxnflow_{seh,drd2}{1,2,3}-718{62..67}.out`.

## Relevant Versions

```
1701374 Logs/055: four-generator two-knob surfaces -- 380/380 comparable cells favour hub-batching
02626ff sEH at the calibrated bar 5.0: the edge grows AND the comparison becomes fair
b57b6b0 matrix16: fraggfn_seh cap-6 closes the surrogate matrix -- 8 of 8 cells
03e391b matrix16: both RGFN cells land + RxnFlow measured timings -> 7 of 8 surrogate cells
0aa7de2 Add scratch-purge backup for the irreplaceable artifacts
```

Branch `Hub-Analysis`, all committed. Mapping of content to commit:

| content | commit |
|---|---|
| the four two-knob surfaces + `surface_all_generators.sh` + this log + the index row + the `tau_similarity_surface.py` subtitle fix | `1701374` |
| the 5/6/7 threshold sweep (20 points) + the bar-5 cross-generator panel | `02626ff` |
| `fraggfn_seh` cap-6 cell + the 4-way control panel | `b57b6b0` |
| both RGFN cells + merged RxnFlow timings + the 3-generator panel | `03e391b` |
| `scripts/backup_scratch_critical.sh` | `0aa7de2` |

Earlier supporting commits in this line: `4c7824b` (horizontal-bar direction marker), `dd9fce3`
(diagonal Pareto marker), `88b2e94` (compute-time instrumentation ported to all four workers),
`bbc16f7` (`submit_timing.sh`/`merge_timings.sh`).

## Relevant Resources

**Sources**
- Entry `034` — calibration of the sEH scoring model against real actives; the origin of the claim that
  bar 7.0 is optimistic and ~5–6 is the empirically useful range.
- Entry `051` — the molecule pools' intrinsic distinctness is flat below bar ~7 and collapses above it.
- Entry `052` — the single-generator two-knob surface this entry generalises, and the source of the
  pool-limited hatching convention.
- Entry `046` — the FragGFN fragment-cap fix (9 → 6) whose sEH model is used here for the first time.
- `malkin2022trajectorybalance` — the balance conditions underlying the flow terms the workers extract.

**Packages**
- RDKit (fingerprints + distinctness) — `glue/samplers/lsdflow/mode_select.py`
- PyTorch + each generator's own framework, one conda env per generator —
  `validation/lsdflow/adapters/workers/<gen>_worker.py`
- SLURM (Balam) — `experiments/lsd_hubs/matrix16/submit_*.sh`

## Method

1. **Repaired two dead queued jobs.** Jobs 71766/71767 (both RGFN cells) had been submitted from a git
   worktree that was deleted after a merge, so their working directory no longer existed and both would
   have failed at startup after days in the queue. 71767 had a second, independent fault: it was
   submitted enumerate-only, but that cell had no sampled pool at all. Cancelled both and resubmitted
   from the real repository with correct stages:
   ```
   N_TRAJ=30000 N_HUBS=200 STAGE=enum sbatch -J m16_rgfn_seh  --time=3-00:00:00 .../submit_cell.sh rgfn seh
   N_TRAJ=30000 N_HUBS=200 STAGE=all  sbatch -J m16_rgfn_drd2 --time=3-00:00:00 .../submit_cell.sh rgfn drd2
   ```
   → 71981 (31.6 h) and 71982 (26.2 h), both 200/200 hubs.
2. **Smoke-tested the one never-executed code path** before leaving it to run unattended: RGFN's
   `--mode enumerate` had never been run since the compute-time instrumentation was added.
3. **Collected RxnFlow's measured compute** on cells enumerated before the instrumentation existed:
   `submit_timing.sh rxnflow {seh,drd2} <slice> 3` (jobs 71862–71867, 1.6–3.0 h each), then
   `merge_timings.sh rxnflow {seh,drd2}` → 200/200 hubs each.
4. **Verified the FragGFN cap-6 config/checkpoint pairing three ways** before launching, because a
   mismatch here passes every smoke test while producing wrong science: the config declares
   `max_nodes: 6`; the run's own `run_config.yaml` snapshot records `6`; and the molecules' size
   distribution confirms it (median 35 heavy atoms vs 52 for the archived cap-9 run). `it = 5000`
   confirmed training finished rather than leaving a stub.
5. **Ran the final cell:** `N_TRAJ=30000 N_HUBS=200 STAGE=all sbatch -p debug --time=02:00:00
   .../submit_cell.sh fraggfn seh` → job 72178, 29 min.
6. **Per-cell readout** for all changed cells: `run_cell_campaign.sh <gen> <target>` (CPU, login node).
7. **Threshold sweep** over the cached enumerations, both strategies, all four sEH cells:
   `bash gate_sweep.sh rgfn_seh scent_seh rxnflow_seh fraggfn_seh` → 3 min 07 s, 20 cell×bar points.
8. **Two-knob surfaces**, identical grids for all four generators (9 quality bars × 13 distinctness
   cutoffs = 117 cells × 2 strategies, fixed 300-reaction budget):
   `bash surface_all_generators.sh seh` → 140–346 s per cell.
9. **Cross-generator diversity sweeps** at both bars:
   `tau_curve_all_generators.py` and `... --gate 5.0`, plus `--include-fraggfn` for the control panel.

## Results

### The eight fast-reward cells (quality bar 7.0 for sEH, 0.5 for DRD2; 300-mode budget)

| cell | baseline modes | baseline rxn/mode | hub modes | hub rxn/mode | ratio | hubs used | measured compute |
|---|---|---|---|---|---|---|---|
| `rgfn_seh` | **9** | 4.000 | 135 | 2.993 | 1.34× | 92 | 113,586 s |
| `scent_seh` | 300 | 3.383 | 300 | 1.303 | 2.60× | 57 | 3,483 s |
| `rxnflow_seh` | 83 | 3.000 | 186 | 2.446 | 1.23× | 135 | 30,021 s |
| `fraggfn_seh` | 300 | 4.997 | 300 | 1.333 | 3.75× | 25 | 147 s |
| `rgfn_drd2` | 300 | 3.843 | 300 | 1.197 | 3.21× | 24 | 6,632 s |
| `scent_drd2` | 300 | 3.673 | 300 | 1.160 | 3.17× | 28 | 2,586 s |
| `rxnflow_drd2` | 277 | 2.935 | 300 | 1.833 | 1.60× | 126 | 11,767 s |
| `fraggfn_drd2` | 300 | 4.680 | 300 | 1.173 | 3.99× | 13 | 71 s |

Hub-batching wins all eight. Note `rgfn_seh` and `rxnflow_seh`: the baseline reaches only 9 and 83
modes, so its reactions/mode is measured over a fraction of the target library and the ratio understates
hub-batching badly — at bar 7.0 `rgfn_seh` hub-batching delivers **15× more modes**, which the 1.34×
cost ratio hides entirely.

### Recalibrating the quality bar (sEH, 300-mode budget, cutoff 0.5)

| cell | ratio @ 7.0 | @ 6.0 | ratio @ 5.0 | change 7→5 |
|---|---|---|---|---|
| `rgfn_seh` | 1.336× | 1.876× | **2.324×** | +74% |
| `rxnflow_seh` | 1.226× | 2.067× | **2.315×** | +89% |
| `scent_seh` | 2.596× | 2.986× | **3.095×** | +19% |
| `fraggfn_seh` | 3.749× | 4.073× | **4.210×** | +12% |

At bar 5.0 **all four cells reach 300/300 modes for both strategies**, so every number is measured on the
same library; at 7.0 two are pool-starved. Consequently the between-generator spread narrows from
1.23–3.75× (a 3.0× ratio) at bar 7.0 to 2.32–4.21× (1.8×) at bar 5.0 — most of the apparent difference
between models was the bar, not the model.

Supporting detail: `rgfn_seh`'s baseline reactions/mode barely moves between the bars (4.000 → 3.920)
while its mode count goes 9 → 300. The baseline's **cost** is bar-invariant; only its **yield**
collapses. This is the exact-limit behaviour entries `052` and `054` found for SCENT, now on three more
cells.

### The two-knob surfaces (9 bars × 13 cutoffs, fixed 300-reaction budget)

| cell | comparable cells | pool-limited | hub-batching cheaper | best cell | at (7.0, 0.5) | at (5.0, 0.5) |
|---|---|---|---|---|---|---|
| `rgfn_seh` | 71 | 46 | **71/71** | 3.81× @ bar 4.5, cut 0.9 | pool-limited | 2.37× |
| `scent_seh` | 109 | 8 | **109/109** | 3.25× @ bar 4.0, cut 0.45 | 2.70× | 3.19× |
| `rxnflow_seh` | 85 | 32 | **85/85** | 2.79× @ bar 5.5, cut 0.85 | pool-limited | 2.27× |
| `fraggfn_seh` | 115 | 2 | **115/115** | 4.93× @ bar 6.0, cut 0.75 | 3.73× | 4.20× |

**380 of 380 comparable cells favour hub-batching, with no crossover anywhere on either knob for any
generator** — the generalisation of entry `052`'s 206/206 single-generator result. Pool-limited cells are
hatched and excluded from the tallies, never counted as wins.

The count of pool-limited cells is itself informative and orders the generators the same way the
headline table does: FragGFN 2, SCENT 8, RxnFlow 32, RGFN 46. For RGFN the entire region above bar ≈6.3
is pool-limited, including the (7.0, 0.5) operating point our headline number is quoted at — which is a
direct, independent statement that bar 7.0 is the wrong place to quote that cell.

### Cross-generator diversity sweep (modes at a 100-reaction budget)

At bar 7.0, hub-batching starts at 78–96 modes and decays with the distinctness requirement while the
baseline is flat and low (5–36) for every generator. At bar 5.0 the same shape holds with every curve
lifted — e.g. SCENT holds 62 modes at cutoff 0.3 versus 37 at bar 7.0, and RxnFlow no longer flattens.
FragGFN posts the single highest value of any generator (96 at cutoff 0.9) but is a **cost-model
control**: its "reactions" are fragment attachments, not synthesis steps, so the shared axis flatters it.
The useful reading is that the qualitative shape reproduces even for the non-reaction control, which
places the effect in the selection strategy rather than in reaction chemistry.

### Measured compute (per-component, all 200 hubs)

| cell | enumeration | reward scoring | flow extraction | total | per child |
|---|---|---|---|---|---|
| `rgfn_seh` | — lumped — | — | — | 113,558 s | 584 ms |
| `rxnflow_seh` | 392 s | 282 s | **29,345 s (97.8%)** | 30,019 s | 129 ms |
| `scent_seh` | 7,825 s (78.8%) | 555 s | 1,554 s | 9,934 s | 23 ms |
| `fraggfn_seh` | 651 s | 330 s | 108 s | 1,089 s | 3.6 ms |

RxnFlow's 97.8% flow-extraction share confirms a 3-hub pilot that predicted 97.9%. Nearly all of its
enumeration cost is its backward-policy computation — the term entry `050` showed its scaffold *ranking*
is insensitive to (87.5% identical scaffolds with the term ablated). Keeping it is the right call for a
clean story, but it is worth stating that this generator spends almost all its enumeration budget on a
quantity that does not change its answer.

### Caveats

- One training run per cell; no error bars.
- FragGFN is a cost-model control throughout (attachments ≠ synthesis steps).
- `rgfn_seh`'s compute is a per-hub total with no component split, by construction.
- The superseded cap-9 FragGFN rows are still present in `gate_sweep/summary.csv`.
- Surfaces are sEH only; the DRD2 half and both docking targets are not yet measured.
