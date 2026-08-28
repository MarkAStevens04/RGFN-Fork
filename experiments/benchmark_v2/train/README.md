# stage 1 — train

One run per cell to arm B's budget (320,000 oracle calls), with a checkpoint written when
`trace.csv`'s `n_scored` first crosses **10,000** (arm A, the headline). Both arms come from one
trajectory; there is no second training run.

Holds: checkpoints, `trace.csv`, `timing.json`, milestone checkpoints, and — for SCENT only —
the promoted-fragment recipes (`fragments_N.json` with `smiles_to_route`), which exist ONLY during
training and cannot be recovered afterwards.

**Read-only once verified.** Re-invoking a generator runner overwrites `trace.csv`,
`candidates.csv`, `pairs.csv`, `timing.json` and `run_config.yaml`. See ../README.md.
