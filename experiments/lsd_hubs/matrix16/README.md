# `matrix16/` — LSD-Flow evaluation across the full 4×4 generator × target matrix

Runs the LSD-Flow hub pipeline (**sample → pick hubs → enumerate → campaign**) across all
**16 cells** of the publication-scale matrix (Logs/030): 4 generators
(**RGFN / SCENT / FragGFN / RxnFlow**) × 4 targets (**sEH / DRD2 / 6TD3 / ClpP**), each a trained
5,000-iter fixed-reward model on the shared SMALL library (`glue_standard_v1`).

This dir is the **manifest-driven orchestration layer** only. The science lives in the packages it
calls: hub/flow primitives in `glue/samplers/lsdflow/` + `glue/metrics/`; adapters, DAG, diversity,
count-once cost in `validation/lsdflow/`; the strategy comparison in
`experiments/lsd_hubs/campaign/`. `matrix16/` adds no new science — it just resolves every cell's
spec and fans the existing per-cell pipeline across all 16.

> **Status (2026-07-27): 6/8 surrogate cells complete; branch `matrix16-lsdflow` is fully committed and
> READY TO MERGE into `Hub-Analysis` — not yet merged.**
> → **Read [`MERGE_NOTES.md`](MERGE_NOTES.md) first.** It carries the commit-by-commit map, the single
> (trivial, identical-fix) merge conflict + resolution, live run state with job IDs, the three caveats
> that must travel with the numbers, and the open threads. Docking cells (6TD3/ClpP) are
> wired-but-deferred (they need the GPU-docking enumeration path).

---

## The three files that define the matrix

| file | role |
|---|---|
| `manifest.csv` | **Static per-cell spec** — the human-editable source of truth: `generator, target, seed, config, checkpoint, guidance_sidecar` for all 16 cells. |
| `targets.py` | **Per-target science** — reward gate (value + direction) + enumeration cost class (`surrogate`/`docking`) for each system. The one place `--reward-threshold`/`higher_is_better` come from. |
| `manifest.py` | **The loader** — joins `manifest.csv` + `targets.py`, resolves all paths, and annotates **live** filesystem status (checkpoint present? SCENT sidecar? #candidates emitted?). Every driver imports this. `python manifest.py` prints the live status table. |

Adding a target = one `targets.py` entry + 4 `manifest.csv` rows. Adding a seed = 16 rows. Nothing
else changes.

---

## The per-cell pipeline (identical for every generator)

The existing SCENT campaign pipeline, generalized. `pick_hubs.py` and `run_campaign.py`/
`sweep_campaign.py` are already **model-agnostic** (they read `records.csv` / `enum_children.json`);
only **sample** and **enumerate** are per-model, and each generator emits the *same artifact set* so
everything downstream is uniform.

```
 checkpoint ──sample──▶ records.csv          ──pick_hubs──▶ hubs.csv ──enumerate──▶ enum_children.json
 (per cell)            compositions.json                   (model-agnostic)        enumerated_records.csv
                       routes.json                          top-N by F_hat          enum_timings.json
                       visit_counts.json                                                    │
                       meta.json                                                            ▼
                                                              run_campaign.py / sweep_campaign.py
                                                              (count-once now; SPARROW later, other agent)
```

**Artifact contract (what every worker/adapter must emit):**
- *sample mode* → `records.csv` (the §2 flow log-terms per `hub→child` transition),
  `compositions.json` (promoted-fragment composition — empty for non-SCENT), `routes.json`
  (per-product synthesis route — may be empty), `visit_counts.json`, `meta.json`.
- *enumerate mode* → `enum_children.json` (`{"hubs":[{hub_key,hub_input,depth,uncertainty,
  n_effective,children:[{smiles,reward,added_promoted,reaction}]}]}` — the campaign's `EnumeratedHub`
  schema), `enumerated_records.csv`, `enum_per_hub.json`, `enum_timings.json`.

Per-generator sample/enumerate mechanism:
| generator | env | mechanism |
|---|---|---|
| RGFN | `rgfn` (in-process) | `RGFNAdapter` (`rgfn_extract` / `rgfn_enumerate`) |
| SCENT | `scent` (subprocess) | `scent_worker.py --mode {sample,enumerate}` (P_B from `guidance_models.pt` sidecar) |
| FragGFN | `fraggfn` (subprocess) | `fraggfn_worker.py` *(new; P_B is uniform — control model, non-reaction moves)* |
| RxnFlow | `rxnflow` (subprocess) | `rxnflow_worker.py` *(new; P_B heuristic — U(h) unreliable, flag it)* |

All four surface through `validation/lsdflow/adapters/get_adapter(<gen>)` and the same
`submit_cell.sh <gen> <target>` launcher.

---

## Parameters (smoke vs. real run)

Two knobs are **CLI/env parameters** — low while debugging, exhaustive for the headline run:

| knob | smoke (debug) | real run (after all 16 train) | where |
|---|---|---|---|
| `n_trajectories` | **10,000** | **30,000** (matches the SCENT sEH anchor / Logs/031) | `submit_cell.sh` arg / `N_TRAJ` |
| `enumerate_top_hubs` | **50** | **200** (Logs/031 scaled setting; contains top-50 as a subset) | `submit_cell.sh` arg / `N_HUBS` |

A 200-hub enumeration is a superset of 50, so the real run's analysis can report both. RGFN stdlib
sampling is CPU-slow (Logs/020) — its 30k run wants a compute node.

---

## Organization / where outputs land

- **Heavy scratch artifacts** (sampled DAG + enumeration; git-ignored, not in repo):
  `$SCRATCH/rgfn_runs/lsdflow/matrix16/<gen>_<target>/{sample,enum}/`
  (override root with `MATRIX16_SCRATCH`).
- **Committed small results** (campaign readouts, summaries, plots):
  `experiments/lsd_hubs/matrix16/results/<gen>_<target>/`.

One `<gen>_<target>` tag threads through scratch, results, and logs.

---

## Build status (what runs today — 2026-07-20)

**All four generators run the full pipeline (sample → enumerate → campaign) end-to-end** — validated
on sEH via the launcher (`submit_cell.sh` + `run_cell_campaign.sh`) at smoke scale (60 traj / 2-3 hubs):

| generator | sample (flow extract) | enumerate (`enum_children.json`) | full cell |
|---|---|---|---|
| **RGFN** | ✅ `rgfn_worker.py` | ✅ native reaction hubs (`hub_state_from_smiles`) | ✅ validated |
| **SCENT** | ✅ `scent_worker.py` (pre-existing) | ✅ native, frozen 2018-frag library | ✅ validated |
| **FragGFN** | ✅ `fraggfn_worker.py` | ✅ persist `hub_graphs.pkl` + DFS AddNode/SetEdgeAttr | ✅ validated |
| **RxnFlow** | ✅ `rxnflow_worker.py` | ✅ `MolGraph(smi)` + protocol×block, fresh retro P_B | ✅ validated |

Enumerate hub-state acquisition differs per model (the one genuine divergence): RGFN/SCENT/RxnFlow
rebuild the hub state from its SMILES natively; **FragGFN persists `hub_graphs.pkl` during sampling**
(`obj_to_graph` mis-decomposes ~6% of hubs) and the enumerate stage reloads it via `--sample-dir`.

**Two documented caveats (FragGFN is the control, as intended):**
- FragGFN's *enumerated* move (3 actions) is a different conditional probability than its *sampled*
  move (~9-17 actions, deferred attachments) — so FragGFN enumerated F̂ is NOT merged/compared with
  its sampled F̂. Fine for the campaign (uses `enum_children` alone) and within-set U(h).
- FragGFN's count-once cost charges fragment-attachments as "reactions" (an approximation); the
  meaningful FragGFN library cost comes from post-hoc retrosynthesis, not this number.

**Child-policy note:** `--child-policy free_frag` + pre-select-K (the comparison hero) operate on
SCENT's *promoted-fragment* metadata, so they meaningfully apply to **SCENT**; for the baselines
(no dynamic library) pre-select-K pre-builds 0 fragments and naive `reward` hub-batching is the
natural comparison. Both run without error on all four.

- ✅ `manifest.csv` / `targets.py` / `manifest.py` — all 16 cells resolve; live status + `--emit`/`--list`.
- ✅ `_artifacts.py` — shared stdlib artifact writer (records/enum_children/uncertainty), format-locked
  to `scent_worker`.
- ✅ `link_worktree_data.sh` — symlinks gitignored `data/` payloads into the worktree (required —
  `git worktree` omits gitignored files the pipeline reads).
- ✅ `submit_cell.sh` / `launch_surrogates.sh` / `run_cell_campaign.sh` — parameterized launchers
  (`N_TRAJ` / `N_HUBS` / `STAGE` / `TIME`).
- ✅ **6/8 surrogate cells COMPLETE at the real scale** (30k trajectories / 200 hubs) with committed
  campaign readouts in `results/<cell>/`: `scent_{seh,drd2}`, `rxnflow_{seh,drd2}`, `fraggfn_{seh,drd2}`.
- ⏳ **The 2 RGFN cells** — jobs **71766** (`rgfn_seh`, enum-only on its saved sample) and **71767**
  (`rgfn_drd2`, full) queued with 3-day walltimes. RGFN's enumerate is the slow one (two 24h timeouts →
  hence the per-hub progress logging + partial flush). Run `run_cell_campaign.sh rgfn {seh,drd2}` after.
- ⏳ FragGFN/RxnFlow LSD-Flow *adapters* (`get_adapter`) — the matrix calls workers directly, so these
  are only needed for the harness severe-test path; deferred by design.

**Live status any time:** `python experiments/lsd_hubs/matrix16/manifest.py`.
Docking cells (6TD3/ClpP) remain `run_stage=deferred` and are auto-excluded by `launch_surrogates.sh`.

### Analysis tools (all pure-CPU / login-safe, per cell)

| script | question |
|---|---|
| `gate_sweep.sh` | both strategies re-scored at each target's `threshold_variants` over the cached enumeration → `results/<cell>_thr<gate>/` + `results/gate_sweep/summary.csv` |
| `hub_rank_sensitivity.py` | does the (heuristic, for RxnFlow) `P_B` drive hub selection? |
| `flow_consistency.py` | enumeration-derived `F_true(h)` vs per-child `F̂`; `Σ_x P_F(x\|h)` normalization; variance decomposition; sampled-vs-enumerated `U(h)` |
| `hub_flow_estimators.py` | four `F(h)` estimators scored on bias **and** rank correlation vs `F_true`; well-conditioned flow-conservation test (needs `<gen>_worker --mode probe_hubs`) |

Findings live in `results/gate_sweep/*.json` and the commit messages; the headline caveats are in
[`MERGE_NOTES.md`](MERGE_NOTES.md) §5 and the key ones are summarized below.

## The enumerate question — RESOLVED (option 2: exhaustive for all four)

The project lead chose exhaustive enumeration for FragGFN/RxnFlow (no sampled-neighborhood
shortcut). Built + validated:

- **RGFN / SCENT** — native reaction state-machines; `hub_state_from_smiles` + forward enumeration.
- **RxnFlow** — Markovian, so the hub state rebuilds directly from SMILES (`MolGraph(smi)`); enumerate
  all applicable protocol×block reactions, score at `sampling_ratio=1.0`, P_B via a *fresh* retro
  analyzer per child with the reverse action injected (else P_B collapses to 0). Reconstructed
  forward terms are bit-identical to sampling.
- **FragGFN** — the trained policy defers attachment points, so `obj_to_graph`-from-SMILES
  mis-decomposes ~6% of hubs; instead the sample stage **persists `hub_graphs.pkl`** and enumerate
  reloads the exact hub graph, then DFS `AddNode → SetEdgeAttr×2 → Stop` over the fragment vocabulary.

The mapping + validation was done by two per-env agents (validated prototypes in the scratchpad); the
recipes are captured in each worker's module docstring.

---

## Coordination

This work is on the `matrix16-lsdflow` worktree; a concurrent agent owns the **evaluator axis**
(`sweep_campaign.py --evaluator`, `validation/lsdflow/eval/*` SPARROW, S3-GFN — see
`docs/LSD_FLOW_BENCHMARK_PLAN.md`). The clean handoff point is the **enumeration artifact set**
above: matrix16 produces it per cell; the evaluator (count-once now, SPARROW later) consumes it.
matrix16 does not touch the evaluator files.
