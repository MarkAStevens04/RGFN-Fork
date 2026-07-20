"""S3-GFN entrant adapter (`[kim2026s3gfn]`) — the marquee NON-reaction baseline for the LSD-Flow
library-efficiency benchmark (docs/LSD_FLOW_BENCHMARK_PLAN.md T3.1–T3.2).

A thin adapter over the upstream S3-GFN (installed via ``external/setup_s3gfn.sh`` into its own
``s3gfn`` conda env; a SMILES/sequence GFlowNet post-training GP-MolFormer with soft synthesizability).
Like the RxnFlow/FragGFN adapters, the heavy code is NOT vendored; this package holds only the wiring:

- ``fixed_reward.py`` — swappable reward generators (``SEHFrozenReward`` / ``DRD2FrozenReward`` /
  ``DockingBridgeReward``) sharing one interface, injected in place of S3-GFN's native ``get_scores``
  so it optimizes the SAME oracle as the reaction-GFN entrants (sEH implemented + a verification case;
  DRD2/docking interface-ready).
- ``run_s3gfn_fixed.py`` — (planned) one-shot fixed-reward training driver that wires the chosen
  reward provider into S3-GFN's ``SynthSmilesTrainer`` and emits a candidate pool (``has_route=0``)
  for the frontier's best-candidate arm.

Runs in the ``s3gfn`` env; the pool is emitted in the standard candidate-dataset format and re-scored
across the env boundary (``scripts/ingest_candidates.py`` / ``score_batch.py`` under ``rgfn``) for the
per-target mode gate — the same two-env pattern as the other baselines.
"""
