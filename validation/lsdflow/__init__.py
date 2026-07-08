"""LSD-Flow validation/analysis world — ``docs/LSD_FLOW_PROPOSAL.md`` §3b.

Self-contained comparative machinery for post-hoc hub analysis over trained reaction
GFlowNets: per-model adapters (cross-env), the canonical DAG, the severe-test suite, the
RGFN-vs-SCENT hub-coincidence study, and the cost/diversity metrics + harness. It imports the
acquisition primitives from ``glue`` (allowed by the one-way rule) and is **never** imported
back by the production pipeline.
"""
