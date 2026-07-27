# Does S3-GFN's retro signal accept the reaction-GFN's blocks? (Exp A falsification — Logs/048)

Read-only probes of S3-GFN's own `SynthesizabilityEvaluator` (the object its training loop calls) that
overturned Exp A's initial interpretation. Exp A gave S3-GFN the reaction-GFN's 418 SMALL blocks but
S3-GFN's own `hb105` templates (fragments-only, by design) and scored `synth_ratio = 0.0`. These
probes show that is a **template–block incompatibility**, not evidence about molecule synthesizability.

| Script | Question | Finding |
|---|---|---|
| `probe_retro_env_coverage.py` | Can `hb105` decompose molecules provably built FROM the SMALL blocks? | SCENT mols 0/30 under `small_hb105` **and** 0/30 under `zincfrag_hb105` (178k blocks), while S3-GFN's own mols score 93% under `zincfrag_hb105` → templates work, but can't reverse SCENT chemistry |
| `probe_depth_and_steps.py` | Is it the step budget, or is the env broken? | Raw SMALL blocks score **100%** (env fine); `max_steps` 3→6 changes nothing; SCENT mols ~0% at every `hub_depth` |
| `probe_small_block_usage.py` | Are the SMALL blocks inert in Exp B? | **No** — 14/59 molecules have ≥1 route using a SMALL block; 6.5% of block-uses (3,666/56,098) are SMALL blocks |

Run in the `s3gfn` env from the repo root, e.g.:
```bash
conda run -n s3gfn env PYTHONPATH=external/s3gfn/src python probe_retro_env_coverage.py
```
Blocks live on `RxnAction._block` (not on leaf `.smi`) — an early version of the usage probe walked
leaf SMILES, found zero leaves, and looked like "0/58 use SMALL blocks"; that was a probe bug.
