# ClpP + 6TD3 — the hub-batching advantage measured against real docking, on every cell we can run
**Date:** 2026-08-11, ~11am

## Question

Does our library-building advantage survive when the molecules are scored by real GPU docking instead
of the fast stand-in we have used for every result so far?

## Context & Summary

Everything we have claimed about hub-batching so far rests on a **surrogate** score — a fast neural
predictor for the sEH protein that returns a number in milliseconds. Entry `055` swept that surrogate
exhaustively and found hub-batching cheaper in **380 of 380** comparable settings across all four
generators. That is a strong result with one obvious hole: a surrogate is a smooth, cheap function,
and a reviewer's first question is whether the advantage is an artifact of scoring molecules with a
model rather than with the physics-based docking a drug project would actually use. Docking is roughly
a hundred thousand times slower per molecule, which is why the surrogate existed in the first place.

Entry `057` sized that bill and reported a coverage problem: only two of the four generators had
docking wired at all, so the docking half of the matrix looked like a two-generator comparison.
We then wired the remaining two, which turned it into a three-generator comparison (the fourth,
RGFN, is blocked on unfinished training rather than on missing code).

This entry runs the whole thing: three generators × two docking targets = six cells, each one
enumerating the one-reaction children of 200 shared scaffolds and **docking every child** (exhaustively
for the two baselines; the two SCENT cells hit a deliberate per-scaffold cap — see the audit below).
Two protein systems are involved — **ClpP**, where the score is a single binding energy, and
**6TD3**, where it is the *difference* between two docking runs, which is how we isolate the
cooperative "glue" effect we actually care about. For each cell we compare our method against the
same baseline as always (take the generator's best molecules and build them individually) and then
re-score the whole comparison at seven different hit thresholds, to check the answer does not depend
on where we drew the line.

## Answer

The advantage survives contact with real docking on both protein systems and all three generators:
hub-batching builds a 300-molecule diverse library for **2.43× to 4.43× fewer reactions** (median
2.76×), and it leads at **every one of the 42 hit thresholds tested**. Critically, both methods
reached the full 300-molecule target in all six cells, so no comparison is flattered by one side
running out of material — the weakness that made entry `055`'s original threshold choice misleading.
The surrogate result was therefore not an artifact of cheap scoring.

The one thing this entry does **not** support is comparing the *size* of the advantage between
generators, because the baseline's own cost differs a lot between them; the larger ratios come partly
from a worse baseline rather than better batching.

## Relevance to our Publication

This closes the single biggest gap between our claims and our evidence, and it targets a specific
reviewer objection for **Digital Discovery** (our primary venue): that a methods result demonstrated
only against a learned surrogate has not been shown to hold for a real scoring function. We can now
state the advantage on physics-based docking, on two protein systems, for three independent
generators, with the threshold-dependence mapped rather than asserted.

It also strengthens the "2+ validated systems" requirement the project tracks for the **JCIM** stretch
target: the advantage is now measured on both ClpP and the 6TD3 glue differential, not just on sEH.

## Next Experiments

**Refining for publication**

- **Error bars.** Every cell is a single seed and a single trained checkpoint. Reviewers will want at
  least a second seed per cell before we quote a median ratio.
- **Finish the fourth generator.** RGFN's two docking cells are stuck at 55% and 71% of their training
  target. Completing them makes the docking matrix four generators wide, matching the surrogate half.
- **Decide the ClpP threshold in the paper's own terms.** The sweep shows the two strictest ClpP
  settings starve one generator's material while a third generator is unaffected by the threshold
  entirely — worth stating explicitly rather than quoting a single number.

**Next steps in project**

- Price these libraries through the competitor pipeline (entry `056`) so the docking cells get the
  same head-to-head cost comparison the sEH cell already has.
- Feed the docking-scored hubs into the active-learning loop, which is what the whole selection
  machinery is ultimately for.

---

# Re-creation

## Relevant Files

Root: `/home/markymoo/projects/RGFN_Fork/RGFN-Fork`

**Scripts**

- `./experiments/lsd_hubs/matrix16/autoharvest.sh` — the automation that produced this entry: polls all
  docking cells and, when a cell's slices cover every hub, runs merge → campaign → gate sweep → backup
  once per cell. Polls the **filesystem**, not SLURM, because the slices ran on two unfederated
  schedulers (Balam and Trillium) that share `/scratch`; a Balam job cannot depend on a Trillium job id.
- `./experiments/lsd_hubs/matrix16/submit_docking_cell.sh` — per-slice enumeration launcher: persistent
  docking server, live pose probe through the socket, round-robin hub slicing, `HUBS_FILE` override.
- `./experiments/lsd_hubs/matrix16/merge_docking_slices.sh` — unions a cell's slices; **refuses** unless
  every hub in `hubs.csv` is present exactly once. Used as the readiness test by `autoharvest.sh`, so
  "complete" has exactly one definition.
- `./experiments/lsd_hubs/matrix16/run_cell_campaign.sh` — count-once campaign readout per cell; applies
  the per-generator child policy (`free_frag` + pre-select-K=20 for SCENT, naive `reward` otherwise).
- `./experiments/lsd_hubs/matrix16/gate_curve.py` — post-hoc threshold sweep; loads one enumeration once
  and loops gates in-process, reusing `run_campaign`'s own `build_strategy`/`run_timed` so each point is
  bit-identical to a standalone `run_campaign.py --reward-threshold` invocation.
- `./validation/lsdflow/adapters/workers/_docking.py` — the two-column contract seam (see Method 1).
- `./scripts/backup_watchdog.sh` — bounded periodic backup loop; `crontab` is denied for this account on
  `balam-login01`, so periodic work runs as a `flock`'d `setsid nohup` loop.

**Models**

- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/{rxnflow,scent,fraggfn}_{clpp,6td3}_5k/seed42/train/checkpoints/last_gfn.pt`
  — the six trained generators, all at the 5,000-iteration target (SCENT reports 4999, indexing from 0).
  Each was trained **against the same docking oracle it is evaluated with**, which is what makes the
  recovered flow terms comparable to the sampled DAG.
- `/scratch/markymoo/rgfn_runs/experiments/fixed_reward/scent_*_5k/seed42/additional_fragments/fragments_4000.json`
  — SCENT's promoted-fragment snapshot; required for its nested cost model. Without it the cost model
  silently falls back to a different accounting and the numbers stop being comparable.

**Results** (committed)

- `./experiments/lsd_hubs/matrix16/results/<cell>/summary.json` — per-cell head-to-head + measured compute.
- `./experiments/lsd_hubs/matrix16/results/gate_curve/<cell>/gate_curve.{json,png}` — 7-point threshold sweeps.

**Job Logs**

- `/scratch/markymoo/rgfn_runs/autoharvest.log` — the harvest ledger (215 rounds, 6 cells promoted).
- `/scratch/markymoo/rgfn_runs/m16_dock_*-{72493..72663}.{out,err}` — Balam slices. Trillium slices ran
  under that cluster's own ids (`7213xx`, entry `057`).

## Relevant Versions

```
42eeb26  Docking matrix COMPLETE: hub-batching wins all 6 cells, 2.43-4.43x (median 2.76x)
bf494c0  First real-docking LSD-Flow results: hub-batching wins 2.4-4.4x on ClpP and 6TD3
b36c3d5  Weekend automation: auto-harvest finished docking cells + periodic backup
```

The `best_reward` direction fix in `glue/samplers/lsdflow/campaign.py` and the per-generator default fix
in `gate_curve.py` are `9ca6916`; the six `summary.json` files were
regenerated after the former.

## Relevant Resources

**Sources**

- ClpP receptor: human ClpP **7UVU**; threshold calibrated to **−8.0** kcal/mol at AUROC 0.895 (entry `045`).
- 6TD3 (CR8 / cyclin K) glue system; the score is the Tier2−Tier1 differential, provisional bar **−2.0**
  (entries `002`, `011`, `013`).

**Packages**

- QuickVina2-GPU (pose search) + gnina (CNN pose rescoring) — via `glue/oracles/docking_server.py` and
  `glue/oracles/docking_seh_oracle.py`.
- RDKit (Morgan r=3 fingerprints, Tanimoto) — `glue/samplers/lsdflow/mode_select.py`.

## Method

1. **Wire docking for the two unwired generators.** Each generator runs in its own conda env and cannot
   co-import the oracle, so each worker scores children through its own `DockingBridgeReward` over a
   persistent AF_UNIX socket (`RGFN_DOCK_SOCKET`). One dock feeds **two** record columns and conflating
   them is the trap this seam exists to prevent: `reward` is the **RAW** energy the calibrated bar is
   measured in (lower-is-better), while `log_reward` is `beta * clip(training transform)`, which is
   `ReLU(−raw)`-based and therefore never negative. Gating on the latter would qualify **nothing**, with
   no error. A missing socket is a hard failure rather than a slow success.
2. **Smoke each cell before launching** (`smoke_cell.sh`, ~200 molecules, isolated scratch tag): asserts
   oracle failure rate, that `reward` is raw-negative, and the two-column relation with **both** `beta`
   and `clip` *fitted from the data* — necessary because SCENT does not clip while FragGFN clips at 10.
3. **Enumerate each cell** as 6–12 disjoint round-robin hub slices across Balam and Trillium
   (`submit_docking_cell.sh`), 200 hubs per cell, every one-reaction child docked.
4. **Merge, evaluate, sweep** — `autoharvest.sh` detected completion from the artifacts and ran
   `merge_docking_slices.sh` → `run_cell_campaign.sh` → `gate_curve.py` per cell.
5. **Threshold sweep**: 7 gates per cell (ClpP −11.0→−8.0, 6TD3 −4.0→−1.0), re-scoring already-docked
   children — no re-enumeration, no GPU.

## Results

**Head-to-head at each cell's calibrated bar.** Budget = 300 modes; similarity cutoff 0.5; Morgan r=3.
`stop_reason` was `modes` for all twelve arms, i.e. every arm reached the full budget.

| cell | bar | hub r/m | hubs walked | best-cand r/m | edge | best docked |
|---|---|---|---|---|---|---|
| `rxnflow_clpp` | −8.0 | 1.223 | 34 | 2.977 | **2.43×** | −11.6 |
| `scent_clpp` | −8.0 | 1.290 | 34 | 3.420 | **2.65×** | −14.2 |
| `fraggfn_clpp` | −8.0 | 1.093 | 7 | 4.840 | **4.43×** | −13.5 |
| `rxnflow_6td3` | −2.0 | 1.193 | 29 | 2.977 | **2.50×** | −6.224 |
| `scent_6td3` | −2.0 | 1.157 | 17 | 3.320 | **2.87×** | −7.398 |
| `fraggfn_6td3` | −2.0 | 1.533 | 40 | 5.000 | **3.26×** | −7.041 |

min 2.43× / **median 2.76×** / max 4.43×. Scale: **2,022,682 docked children** over 1,200 hub
enumerations.

**Threshold robustness** (7 gates × 6 cells = 42 points). Hub-batching leads at every point. The edge
grows monotonically as the bar loosens, e.g.:

| cell | strictest gate | edge | bar gate | edge | loosest gate | edge |
|---|---|---|---|---|---|---|
| `scent_clpp` | −11.0 | 1.63× (280 modes) | −8.0 | 2.65× | — | — |
| `rxnflow_clpp` | −10.5 | 1.18× (30 modes) | −8.0 | 2.43× | — | — |
| `fraggfn_clpp` | −11.0 | 4.03× | −8.0 | 4.43× | — | — |
| `scent_6td3` | −4.0 | 2.69× | −2.0 | 2.87× | −1.0 | 2.95× |
| `fraggfn_6td3` | −4.0 | 2.78× | −2.0 | 3.26× | −1.0 | 3.32× |
| `rxnflow_6td3` | −4.0 | 2.06× | −2.0 | 2.50× | −1.0 | 2.54× |

Only the two strictest **ClpP** gates are pool-limited (`scent_clpp` 280/300 modes at −11.0;
`rxnflow_clpp` 30/300 at −10.5), which is what makes **−8.0** the defensible ClpP bar rather than a
convenient one. `fraggfn_clpp` reaches 300 modes at every gate from −11.0 to −8.0, i.e. the ClpP
threshold is effectively **non-binding** for that cell — consistent with its 91.5% smoke-scale
qualification rate. Each cell's bar-point reproduces its campaign number exactly (e.g. `scent_clpp`
1.290 r/m → 2.651×), the bit-identity `gate_curve.py` exists to provide.

**Measured compute** (live in-run wall-clock, per component, not modeled):

| cell | enumeration_s | reward_gen_s | flow_extract_s | total | dock share | hub-walk share |
|---|---|---|---|---|---|---|
| `rxnflow_clpp` | 339 | 129,908 | 18,288 | 41.3 h | 87.5% | 6.95 h |
| `scent_clpp` | 8,746 | 360,907 | 1,478 | 103.1 h | 97.2% | 18.53 h |
| `fraggfn_clpp` | 495 | 93,518 | 87 | 26.1 h | 99.4% | 0.96 h |
| `rxnflow_6td3` | 360 | 165,693 | 21,287 | 52.0 h | 88.4% | 6.98 h |
| `scent_6td3` | 7,088 | 505,838 | 2,187 | 143.1 h | 98.2% | 13.67 h |
| `fraggfn_6td3` | 481 | 168,249 | 77 | 46.9 h | 99.7% | 9.54 h |

**412.5 GPU-hours** of enumeration total. Docking dominates everywhere (87.5–99.7%), unlike the
surrogate cells where flow extraction was up to 97.8% of the bill (entry `055`, `rxnflow_seh`). The
`hub-walk share` is the compute hub-batching actually consumed to build its library (0.96–18.53 h),
the rest being enumeration of hubs it never walked.

**Operational findings**

- **8 of 12 `scent_clpp` slices hit the 8 h walltime** at 147/200 hubs. The per-10-hub partial flush
  turned that into a top-up rather than a total loss: a missing-hubs file plus 8 fresh slices
  (72656–72663) finished the remaining 53 hubs without re-docking anything already paid for. Sizing
  from the raw oracle rate (0.398 s/mol × 1,629 children ≈ 11 min/hub) was **wrong by 2.3×** — the
  cells actually ran ~25 min/hub, the cross-env bridge being the difference. Size future slices from
  observed per-hub time.
- **Editing a script does not reach an already-running loop.** A `--gates` fix committed at 11:13 on
  Aug 7 never took effect for cells harvested at 20:21, 03:06 and 05:37, because bash had parsed
  `harvest_one` into memory when the loop started the previous evening. All six sweeps had to be re-run
  by hand. Restart the loop to apply a fix.
- **Three bugs the sweep path hid**, each found only by executing it: (1) `--gates "-11,..."` — argparse
  reads a value beginning with `-` as the next option, so the sweep needs `--gates=VALUE`; (2)
  `gate_curve.py` imports `glue` and therefore needs the `rgfn` env, not the `base` env the loop runs
  in; (3) it defaulted to child policy `reward`/K=0/no-snapshot for **every** generator, which is right
  for the baselines but wrong for SCENT — a mismatch that does not crash, it silently yields a sweep
  incomparable to the cell's own campaign. Now resolved per generator, mirroring `run_cell_campaign.sh`.
- **`best_reward` was direction-blind** (`campaign.py` used an unconditional `max()`), so every docking
  cell reported its *worst* qualifying molecule as "best" — `rxnflow_clpp` showed −8.0, exactly the gate
  bar, beside a median of −8.9. Fixed and the six summaries regenerated; every other field is
  bit-identical, which also re-confirms the campaign is deterministic. Cost metrics and mode counts were
  never affected.

**Enumeration-completeness audit** (run because `scent_clpp` lost 8 of 12 slices to walltime, so
"200/200 hubs present" needed to be distinguished from "every hub's children are complete"):

*Hub coverage is exact in all six cells.* Each merged `enum_children.json` hub set is **set-equal to
its `hubs.csv`** (200/200), with **zero duplicates** and zero hubs appearing in more than one slice —
so the `of12` + `of8` union for `scent_clpp` was genuinely disjoint and the merge's keep-first rule
never had to arbitrate. **No truncated hub is possible by construction:** the worker calls
`enumerate_terminal_children` for a hub, appends it with its complete child list, and only then calls
`flusher.maybe()`, so a mid-hub kill drops that hub **entirely** rather than persisting a partial one.
Empirically no cell has an anomalously small hub (0 hubs below 10% of the cell's median child count).

*Per-hub child enumeration is NOT exhaustive for the two SCENT cells.* `ENUM_MAX=4000` binds only on
SCENT, whose dynamic promoted library gives it far more reachable last-step reactions than the
baselines (max children: rxnflow 2000/2099, fraggfn 2100/1680 — all well clear of the cap):

| cell | hubs at the 4000 cap | median `Σ_x P_F(x\|h)`, uncapped | median, CAPPED |
|---|---|---|---|
| `scent_clpp` | 25 (12 stereo-distinct) | 0.9945 | **0.1039** |
| `scent_6td3` | 51 | 0.9977 | **0.9288** |
| `rxnflow_clpp` | 0 | 0.9430 | — |

The forward-mass sum is the same normalization readout entry `050` used, and it closes to ~1.0 on
uncapped hubs, which validates the measurement. So `scent_clpp`'s capped hubs retain only **~10% of
the reachable forward-probability mass** (severe), while `scent_6td3`'s retain ~93% (mild). The cap
truncates by **depth-first action-index order**, i.e. a deterministic but arbitrary subset — not the
top-scoring children and not a random sample. Capped hubs are also **inside the walk**: 5 of
`scent_clpp`'s 37 walked hubs (including rank 1) and 7 of `scent_6td3`'s 17 (including ranks 0 and 1).

*Direction of the bias is toward understating our result.* Fewer enumerated children per scaffold means
fewer modes extractable per scaffold build, so the walk needs MORE hubs to reach 300 modes, which
raises reactions/mode. The two SCENT edges (2.65×, 2.87×) are therefore conservative. This is reasoning
from the mechanism, not a measurement — quantifying it needs a re-enumeration at a higher cap.

*This is pre-existing, not docking-specific.* `scent_seh` (57 capped hubs, median capped mass 0.5555 vs
0.9998 uncapped) and `scent_drd2` (31) carry the same truncation, so it is a property of every SCENT
cell in entries `050`/`055`. **Entry `050` needs an interpretation correction:** its stated criteria
(0/200 hubs exceed 1, 0/200 below 0.01) are both confirmed by this measurement, but it attributed
SCENT's low-mass tail — "SCENT's 5th percentile of 0.402 is genuine stop-probability mass at hubs the
policy likes to terminate at" — to policy stopping. The tail is the 4000-child cap: capped hubs sit at
median 0.556 while uncapped sit at 0.9998. Its broader "independently confirms the enumeration is
exhaustive" therefore holds for the 143 uncapped `scent_seh` hubs and not for the 57 capped ones.

**Caveats**

- **One seed, one checkpoint per cell.** No error bars on any ratio.
- **The two SCENT cells are not child-exhaustive** (see the audit above); `scent_clpp` in particular
  loses ~90% of forward mass on 25 of its 200 hubs, 5 of them inside the walk.
- **Cross-generator magnitudes are not comparable.** Best-candidate's own cost differs (FragGFN
  4.84–5.00 r/m vs RxnFlow 2.977), so FragGFN's larger ratio is partly a worse denominator. FragGFN also
  remains a **cost-model control** throughout — its attachments are not synthesis steps.
- **`log_reward` is not comparable across generators** either: each is its own `beta * clip(...)` with a
  different fitted clip (SCENT does not clip and reached 56.8; FragGFN clips at 10). Only the raw
  `reward` column is cross-generator comparable.
- **Three generators, not four.** RGFN's docking cells are undertrained (6TD3 2730/5000, ClpP
  3570/5000) and excluded by the readiness gate; `rgfn_clpp` seed43/44 are complete at 4999 but are a
  different seed from the rest of the row.
- The 6TD3 bar remains **provisional**; unlike ClpP's −8.0 it has no AUROC calibration behind it.
