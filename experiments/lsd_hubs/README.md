# `experiments/lsd_hubs/` — LSD-Flow hub analyses

Post-hoc studies of the **hub** structure and quality latent in a trained reaction-GFlowNet's
flow field — the analysis layer of LSD-Flow (`docs/LSD_FLOW_PROPOSAL.md`; built in Logs/025).

**Modular by design: one sub-directory per analysis type.** Each is self-contained (its script +
`README` + small committed `*_results.csv` / `*_summary.json`), and **reuses** the canonical
primitives rather than reimplementing them:

- flow recovery / `U(h)` / hub + molecule strategies / acquisition → `glue/samplers/lsdflow/`
- adapters / DAG / **mode + diversity + cost metrics** / harness → `validation/lsdflow/`

Reusable logic graduates into those packages; only the one-off analysis harness lives here. The
inputs are the persisted DAGs a harness run leaves on `$SCRATCH`
(`.../lsdflow/<run>/{records,enumerated_records}.csv`), so these analyses re-run cheaply without
re-sampling or re-enumerating.

## Analyses

| sub-dir | question |
|---|---|
| [`dropoff/`](dropoff/) | Per-hub **filter funnel**: how many one-reaction children survive each stage — raw → binding gate → Tanimoto-dissimilar modes — and which filter dominates per hub. |

(More to come as we address the hub-quality question — expect additional sub-dirs here.)
