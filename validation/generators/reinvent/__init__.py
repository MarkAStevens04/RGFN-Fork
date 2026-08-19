"""REINVENT 4 entrant adapter (`[loeffler2024reinvent4]`) — a ROUTE-LESS SMILES baseline for the
LSD-Flow library-efficiency benchmark, in the same class as S3-GFN.

A thin adapter over upstream REINVENT 4 (installed by ``external/setup_reinvent.sh`` into its own
``reinvent4`` conda env; a ChEMBL-pretrained SMILES RNN fine-tuned by reinforcement learning). The
heavy code is not vendored; this package holds only the wiring:

- ``fixed_reward.py`` — the frozen reward providers (``SEHFrozenReward`` / ``DRD2FrozenReward``), a
  per-adapter copy of the same classes the RxnFlow/FragGFN/S3-GFN adapters use, so REINVENT
  optimizes the identical function every other entrant does.
- ``plugins/reinvent_plugins/components/comp_glue_surrogate.py`` — the REINVENT scoring component
  that exposes those providers. It lives here rather than in the clone because REINVENT discovers
  components through a **namespace package** on ``sys.path``, which is the upstream-supported
  extension point; the clone stays pristine. See that file before adding any ``__init__.py`` under
  ``plugins/`` — doing so breaks discovery.
- ``run_reinvent_fixed.py`` — the one-shot driver: RL against the frozen reward, then sample the
  candidate pool from the TRAINED agent (matching ``run_s3gfn_fixed.py``'s protocol rather than
  harvesting the RL trace), then emit a standard candidate dataset with ``has_route=0``.

Runs in the ``reinvent4`` env; the pool crosses to ``rgfn`` by subprocess
(``scripts/ingest_candidates.py``) — the same two-env pattern as the other baselines.
"""
