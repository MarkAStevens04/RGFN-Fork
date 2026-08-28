# `benchmark_v2` — the clean benchmark tree

**This directory is the source of truth.** Everything in it is trusted and internally consistent: one
training budget, one gate rule, one hub-batching configuration, one set of seeds. An agent reading a
file here never has to ask which run it came from or which conventions were live when it was written.

**The plan that governs this tree is [`docs/RETRAIN_RUNBOOK.md`](../../docs/RETRAIN_RUNBOOK.md).**
Read it before adding anything. This README is the operational half: what lives where, and how to
tell whether a cell is done.

**The v1 trees are not touched.** `experiments/lsd_hubs/matrix16/results*/` and
`experiments/lsd_hubs/campaign/results/` stay exactly as they are. They become the frozen record of
the workshop-tier submission as a *consequence* of everything new landing here — not because anything
was locked.

---

## The two files that define the campaign

| file | what it is |
|---|---|
| **`grid.csv`** | the authoritative cell list — 108 training cells (9 generators × 4 targets × 3 seeds), with each cell's class, role, pipeline, phase, arms and planned origin. Drivers read this, not a hand-maintained list |
| **`PROVENANCE.csv`** | the ledger — one row per (stage, cell, arm) recording whether that artifact was **generated** here or **copied** from a v1 run, with source path, md5 and date |

`PROVENANCE.csv` is the *only* place origin is recorded. There is deliberately no `copied/`
subdirectory: splitting the tree by provenance would force every reader to think about it, which is
the opposite of what this directory is for.

```
stage,generator,target,seed,arm,origin,source_path,source_md5,date,operator,note
```

`origin` is `generated` or `copied`. Nothing else.

---

## Layout

```
benchmark_v2/
  grid.csv              the cell list          (authoritative)
  PROVENANCE.csv        the origin ledger      (append-only)
  train/                stage 1 — checkpoints, trace.csv, timing.json, recipes
                        READ-ONLY once a cell passes verification (see below)
  pools/                stage 2 — competitor pools (upsample & filter); naive + pruned
  routes/               stage 3 — retrosynthesis (MultiAiZ) or native routes
  selection/            stage 4 — SPARROW MILP + diversity-aware greedy
  campaign/             reaction-GFN side — sample -> pick_hubs -> enumerate -> campaign
  ablations/            ordering, filters, greedy ceiling, tau surfaces, parallel groups
  results/              committed small CSV/JSON per exhibit, each with PROVENANCE.md
  tools/                drivers and verifiers specific to this campaign
```

Heavy artifacts mirror to `$SCRATCH/rgfn_runs/v2/` with the same shape. Small committed results live
here.

---

## The rule that keeps this tree trustworthy

**`train/` is frozen read-only (`chmod -R a-w`) the moment a cell passes verification, and every
later stage writes to its own directory.**

This is not tidiness. Re-invoking any generator runner *overwrites* that cell's `trace.csv`,
`candidates.csv`, `pairs.csv`, `timing.json` and `run_config.yaml` — which already destroyed
`s3gfn_seh/seed43`'s entire training history, unrecoverably. Freezing turns "remember not to re-invoke
a runner against a copied cell" from a discipline into a filesystem property.

A copied training run **must** bring its `trace.csv`, not just its checkpoint: stage 2 harvests the
trace as a free pool of already-scored molecules, and `saturn_clpp/s42` reaches its 500-mode target
from history alone — 28 GPU-hours to 0.

---

## Is a cell done?

A cell is accepted only when every check in `RETRAIN_RUNBOOK.md` §6 passes. The gating one:

```bash
python experiments/lsd_hubs/matrix16/check_route_readiness.py
```

In this tree that is a **hard gate**, not a report: a cell ships only if a chemist could act on every
molecule in it. That is what makes the route dataset matrix-wide rather than the 5 cell-seeds v1
could support.

Also required per cell: `trace.csv` present and continuous, its `n_scored` reaching the arm's budget,
and the arm-A checkpoint sitting at the row where `n_scored` first crosses 10,000.

---

## Per-cell backup

As soon as a stage passes verification, back that stage up to
`/project/def-naeilum/.../RGFN_LSD_MarkStevens/backups`, **per cell, per stage** — not per phase.
Cells run concurrently at different stages, so waiting for a phase boundary means waiting a long time
and probably forgetting. Copy-verify-delete, never `mv`. Login node only; `/project` is not mounted on
compute nodes.

---

## Quick facts a driver needs

| | |
|---|---|
| seeds | 42, 43, 44 — always with `PYTHONHASHSEED=0` exported |
| arm A (headline) | 10,000 oracle calls, all 9 generators |
| arm B (secondary) | 320,000 oracle calls, the 3 reaction-GFNs only; downstream paused |
| batch size | each paper's own — RGFN 100, SCENT 64, RxnFlow 64 (**not** 128) |
| gates | resolve by importing `matrix16/targets.py`; never hardcode, never default |
| hub-batching | `--child-policy free_frag --prebuild-k 0`, all three reaction-GFNs |
| primary readout | modes at **100 reactions** (emit 50/100/150/200/300 on the reaction-GFN side) |
| phase 1 | sEH, DRD2, ClpP — 81 cells |
| phase 2 | 6TD3-B — 27 cells; reward *and* gate are `cnn_vs` at **6.718** (runbook §7.1), oracle wiring in progress |
