# stage 2 — upsample & filter (COMPETITORS ONLY)

`upsample_to_modes.py`: keep sampling the trained checkpoint until the pool holds 500 diverse modes,
or it stalls, or it caps. Harvests the training trace FIRST as a free pool — already-scored molecules,
zero marginal oracle calls.

Two pool variants per cell: **naive** (top 500 by reward) and **pruned** (500 guaranteed-distinct).
Run `mode_saturation.py` first — cells whose two pools coincide need only one.

**Does not apply to the reaction-GFNs** (their pool is the enumerated hub neighbourhood).
**SynFormer is exempt and that is a finding** — its candidates are a slice of an accumulated GA
population, so more molecules means more training.

Stop reasons are reported explicitly and support different claims: `target-reached` (quote freely),
`stalled` (a claim about the GENERATOR), `cap` (a claim about OUR budget — never present a capped
cell as evidence the generator cannot do better).
