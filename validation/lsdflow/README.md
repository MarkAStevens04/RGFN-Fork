# `validation/lsdflow/` — LSD-Flow analysis world

The validation-axis half of **LSD-Flow** (post-hoc hub extraction for batched late-stage
diversification). The scientific framing lives in `docs/LSD_FLOW_PROPOSAL.md`; **this README is the
authoritative map of what the code actually does today.**

The work runs as **two stages** joined by a persisted, on-disk flow-record DAG:

1. **Sampling / flow extraction** — a per-model adapter samples a trained reaction GFN, recovers the
   §2 flow (`log F_hat` + `U(h)`), and persists the flow-record DAG (`records.csv`,
   `compositions.json`, optional exhaustive `enumerated_records.csv`). Driver:
   `harness/run.py`.
2. **The library-cost campaign** (the current headline) — consumes that persisted DAG and compares
   two selection strategies (**hub-batching** vs **best-candidate**) on the **count-once** synthesis
   cost. This lives on the experiments axis in `experiments/lsd_hubs/campaign/` and imports the
   `glue/` primitives; see its README.

## The split (proposal §3)

LSD-Flow is deliberately spread across the repo's two axes:

- **Production side — `glue/`** holds the reusable primitives (same category as `glue/samplers/` +
  `glue/metrics/`); `validation/` and the campaign import these, never the reverse:
  - `glue/metrics/lsdflow_flow.py` — §2 flow recovery `log F_hat(h;x)` + the reward-free visitation
    estimate, in log space.
  - `glue/metrics/uncertainty.py` — `U(h)`, the flow-matching residual (variance of per-child
    log-flow estimates).
  - `glue/samplers/lsdflow/` — the flow-record schema (`records.py::FlowRecord`), the lightweight
    in-env aggregation (`dag.py::LiteHubDAG`), the rgfn-native extraction/enumeration
    (`rgfn_extract.py` / `rgfn_enumerate.py`, shared with the RGFN adapter here), the hub-selection
    strategies + registry (`hub/`), and the **campaign selection strategies** the cost comparison
    runs (`campaign.py::BestCandidateStrategy` / `HubBatchingStrategy`, with `child_select.py`
    within-hub policies + `mode_select.py` diversity acceptance).
- **Validation side — `validation/lsdflow/` (here)** holds the analysis machinery. It imports the
  primitives from `glue/` and is **never** imported back by the pipeline.

## Layout

```
validation/lsdflow/
  adapters/        # the §4b per-model sampling contract
    base.py          GFNAdapter ABC + FlowSample (canonical schema)
    rgfn_adapter.py  in-process RGFN sampler (uses glue rgfn_extract/enumerate)
    scent_adapter.py in-process client that shells to the scent-env worker
    registry.py      name -> adapter (rgfn + scent live)
    workers/
      scent_worker.py  runs in the `scent` env; emits records.csv/compositions.json/enum_children.json
  dag/             # the §6 canonical DAG
    node.py          canonical node identity (stereo-stripped cross-model key)
    graph.py         HubDAG: rich, networkx-backed, persisted (records.csv/graph.gpickle)
    build.py         FlowSample -> HubDAG
  metrics/
    diversity.py     paper-comparable modes (Morgan r=3/2048, Tanimoto 0.7) + Bemis-Murcko scaffolds
    cost/
      dynamic_amortization.py  the count-once synthesis cost: FragmentCostTable (each distinct
                               promoted fragment built once, closure under nesting) + snapshot loader
      compute_time.py          measured per-hub wall-clock accounting (EnumTimings, account_strategy)
  harness/
    config.py        LSDFlowRunConfig (one model x reward run spec)
    run.py           the sampling driver (sample -> build DAG -> persist -> optional enumeration)
  results/         # committed small artifacts (DAG summaries, enumeration stats)
```

The heavy count-once **campaign** that consumes this DAG (hub-batching vs best-candidate,
reactions/mode, compute-time) lives in `experiments/lsd_hubs/campaign/`, not here — the harness only
produces the flow field + neighborhoods; the cost comparison is downstream.

## Running the sampling stage

GFN **inference only** — no docking — so it runs on a Balam/Trillium login node (a large sample or
enumeration wants a compute node; see the submit scripts). Prefix with the smoke env (per
`CLAUDE.md`) so dgl's CUDA libs are on `LD_LIBRARY_PATH`:

```bash
source ~/bin/rgfn-smoke-env.sh
# RGFN (in-process)
python -m validation.lsdflow.harness.run \
    --checkpoint /scratch/markymoo/rgfn_runs/experiments/fixed_reward/seh_proxy_stdlib/2026-07-02_14-59-53/train/checkpoints/last_gfn.pt \
    --config-path configs/glue/fixed_reward_seh_proxy_stdlib.gin \
    --n-trajectories 10000
# SCENT (cross-env; the scent-env worker is spawned by the adapter) — compute node:
sbatch validation/lsdflow/submit_scent_seh.sh
```

> **Checkpoint-provenance caution (verify before using any checkpoint).** Scratch run dirs hold
> cancelled early-iteration stubs next to the completed run — e.g. `seh_proxy_stdlib/` holds three
> timestamped dirs but only **`2026-07-02_14-59-53`** is the completed 5,001-iter model. Check
> `torch.load(ckpt)['metrics']['epoch']` (and that `candidates.csv` exists) before trusting a
> checkpoint — flow analysis on an unconverged field is meaningless. See the `verify-checkpoint-trained`
> memory.

Writes the persisted DAG (`records.csv`, `compositions.json`, `hub_summary.csv`, `meta.json`,
`graph.gpickle`) plus `report.json` and — with `--enumerate-top-hubs N` — `enumeration.json` +
`enumerated_records.csv` to `--out-dir`.

## Status

- **Live:** RGFN (in-process) and SCENT (cross-env) sampling + flow extraction; exhaustive
  `enumerate_children` for both; the count-once cost model + measured compute-time accounting; the
  hub-batching-vs-best-candidate campaign on SCENT sEH (`experiments/lsd_hubs/campaign/`, Logs
  028–039). The SCENT sampling run that anchors the current campaign is `scent_seh_70189`.
- **Planned (not built):** an AL-facing acquisition entry point that consumes hubs as a
  batch-selection sampler (needs a small `glue/active_learning/loop.py` change — the loop has no
  pluggable-sampler hook yet); FragGFN + RxnFlow adapters; the severe-test suite and the
  RGFN-vs-SCENT hub-coincidence study (`docs/LSD_FLOW_PROPOSAL.md` §7/§8); the library-efficiency
  benchmark in `docs/LSD_FLOW_BENCHMARK_PLAN.md`.
