# `experiments/lsd_hubs/` — LSD-Flow hub analyses

Post-hoc studies of the **hub** structure and quality latent in a trained reaction-GFlowNet's
flow field — the analysis layer of LSD-Flow (`docs/LSD_FLOW_PROPOSAL.md`). The headline is the
**library-cost campaign** (`campaign/`): hub-batching vs best-candidate on the count-once synthesis
cost.

**Modular by design: one sub-directory per analysis type.** Each is self-contained (its scripts +
`README` + small committed `*_results.csv` / `*.json`), and **reuses** the canonical primitives
rather than reimplementing them:

- flow recovery / `U(h)` / hub-selection strategies / the campaign selection strategies
  (`BestCandidateStrategy` / `HubBatchingStrategy` + within-hub `child_select` + `mode_select`)
  → `glue/samplers/lsdflow/` + `glue/metrics/`
- adapters / rich DAG / diversity (paper-comparable modes) / the **count-once** synthesis-cost model
  + measured compute-time accounting / sampling harness → `validation/lsdflow/`

Reusable logic graduates into those packages; only the one-off analysis harness lives here. The
inputs are the persisted flow-record DAGs a sampling run leaves on `$SCRATCH`
(`.../lsdflow/<run>/{records,enumerated_records}.csv` + `compositions.json`) plus the campaign's own
cross-env enumeration (`enum_children.json`), so these analyses re-run cheaply on CPU without
re-sampling or re-enumerating.

## Analyses

| sub-dir | question |
|---|---|
| [`campaign/`](campaign/) | **Hub-batching vs best-candidate under a budget** — two swappable selection strategies for building a diverse library of hits (modes), scored on the count-once synthesis cost (reactions/mode), reward-gen (≈ oracle) calls, and measured compute time. The main comparison; Logs 028–039. |
| [`reward_diversity/`](reward_diversity/) | **Is intrinsic diversity coupled to the reward cutoff?** A pool diagnostic (no strategies, no cost model): sweep the reward bar over the sampled and enumerated molecule sets and measure how similar the survivors are, at a fixed subsample size. The precondition for reading any threshold-conditioned cost comparison; Logs/051. |
| [`dropoff/`](dropoff/) | Per-hub **filter funnel**: how many one-reaction children survive each stage — raw → binding gate → Tanimoto-dissimilar modes — and which filter dominates per hub. |

(More sub-dirs as we address further hub-quality questions.)
