# `validation/lsdflow/` — LSD-Flow analysis world

The validation-axis half of **LSD-Flow** (post-hoc hub selection for batched late-stage
diversification). The design spec is `docs/LSD_FLOW_PROPOSAL.md` — read it first. This README
is the map of what lives here and how the two axes split.

## The split (proposal §3)

LSD-Flow is deliberately spread across the repo's two axes:

- **Production side — `glue/`** holds the acquisition *primitives* (the same category as
  `glue/samplers/` + `glue/metrics/`), which the active-learning loop imports directly:
  - `glue/metrics/lsdflow_flow.py` — the §2 flow recovery `F_hat(h;x)` + visitation estimate,
    log-space.
  - `glue/metrics/uncertainty.py` — `U(h)`, the flow-matching residual.
  - `glue/samplers/lsdflow/` — hub + molecule selection strategies (+ registries),
    `LiteHubDAG` (the lightweight in-loop DAG), `rgfn_extract` (rgfn-native flow extraction,
    shared with the RGFN adapter here), and `LSDFlowAcquisition` (the AL-facing entry point).
- **Validation side — `validation/lsdflow/` (here)** holds everything analytical/comparative.
  It imports the primitives from `glue/` (allowed by the one-way rule) and is **never**
  imported back by the pipeline.

## Layout

```
validation/lsdflow/
  adapters/        # the §4b per-model contract
    base.py          GFNAdapter ABC + FlowSample (canonical schema)
    rgfn_adapter.py  in-process RGFN anchor (the only one wired in v1)
    registry.py      name -> adapter; declares the cross-env targets + build status
    workers/         per-env subprocess workers (SCENT/FragGFN/RxnFlow) — phase 3-4
  dag/             # the §6 canonical DAG
    node.py          canonical node identity (stereo-stripped cross-model key)
    graph.py         HubDAG: rich, networkx-backed, persisted; conforms to the §6 duck type
    build.py         FlowSample -> HubDAG
  metrics/
    diversity.py     Butina modes (ECFP4, cutoff 0.65) + Bemis-Murcko scaffolds (§11)
    cost/            reactions-per-mode (PRIMARY, §11); amortization-ratio declared for phase 2
  analysis/        # severe tests, hub-coincidence, pareto — phases 5 & 3 (not yet built)
  harness/
    config.py        LSDFlowRunConfig (one model x reward run spec)
    run.py           the vertical-slice driver (sample -> DAG -> rank -> acquire -> persist)
    matrix.py        the full model x reward x strategy sweep — phase 2 (not yet built)
  results/         # committed small artifacts (DAG summaries, acquisition tables, plots)
```

## Running the RGFN anchor (phase-1 vertical slice)

GFN **inference only** — no docking — so it runs on a Balam/Trillium login node. Prefix with
the smoke env (per `CLAUDE.md`) so dgl's CUDA libs are on `LD_LIBRARY_PATH`:

```bash
source ~/bin/rgfn-smoke-env.sh
python -m validation.lsdflow.harness.run \
    --checkpoint /scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-59-53/train/checkpoints/last_gfn.pt \
    --config-path configs/glue/fixed_reward_seh_proxy_stdlib.gin \
    --n-trajectories 10000
```

> **Anchor checkpoint provenance (verify before using any checkpoint).** `seh_proxy_stdlib/`
> holds three timestamped dirs; only **`2026-07-02_14-59-53`** is the completed 5,001-iter run
> (job 69616, Logs/020 — `candidates.csv` median sEH 7.263 / max 8.354). `2026-07-02_13-45-03`
> (epoch 30) and `2026-07-02_14-18-34` (epoch 40) are cancelled early attempts (69613/69615) —
> do **not** run flow analysis on them (unconverged flow field). Check
> `torch.load(ckpt)['metrics']['epoch']` before trusting a checkpoint.

Writes the persisted DAG (`records.csv`, `hub_summary.csv`, `meta.json`, `graph.gpickle`) plus
`report.json` + `acquisitions.csv` to `--out-dir`.

## Build status (proposal §10)

- **Done (phase 1-2 code):** RGFN adapter, canonical DAG, flow recovery + `U(h)`, all hub +
  molecule strategy registries, reactions-per-mode cost + Butina diversity, the harness driver,
  `LSDFlowAcquisition` wired for the AL path, and the exhaustive **`enumerate_children`** path
  (`rgfn_enumerate.py` + `--enumerate-top-hubs` / `--from-records`) — validated: a depth-0 hub
  enumerates to 498 one-reaction children (sampling saw 8), 100% sampled-child recovery
  (Logs/025 addendum). Enumeration is exact for fragment hubs; stereo-bearing hubs need a fresh
  stereo-keyed DAG (now persisted).
- **Next:** the matrix driver over RGFN's four fixed rewards; loop integration of
  `LSDFlowAcquisition` (a small, tracked `glue/active_learning/loop.py` change — the current
  loop has no pluggable-sampler hook, contrary to the proposal §4a note); SCENT adapter +
  hub-coincidence study once the ModuleList-patched retrain lands (§9); FragGFN + RxnFlow
  workers; the severe-test suite + Pareto front.
