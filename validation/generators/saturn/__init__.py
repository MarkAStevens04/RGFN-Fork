"""Saturn entrant adapter (`[guo2026saturn]`) — a ROUTE-LESS SMILES baseline for the LSD-Flow
library-efficiency benchmark, and the host for the TANGO arms (`[guo2026tango]`, Phase 3).

A thin adapter over upstream Saturn (installed by ``external/setup_saturn.sh`` into its own prefix
env; a Mamba SMILES language model trained with Augmented Memory for sample efficiency). The heavy
code is not vendored; this package holds only the wiring:

- ``fixed_reward.py`` — the frozen reward providers, **byte-for-byte** the REINVENT adapter's copy
  (verified by AST comparison), so the two entrants provably optimize the same function.
- ``oracle_component.py`` — the Saturn ``OracleComponent`` exposing those providers.
- ``_stubs.py`` — stubs the heavy deps of Saturn oracles we never call, so its eagerly-importing
  oracle registry can load without them.
- ``run_saturn_fixed.py`` — the one-shot driver: RL against the frozen reward until the oracle budget
  is spent, then sample the pool from the TRAINED agent, then emit a candidate dataset
  (``has_route=0``).

Runs in the ``saturn`` prefix env; the pool crosses to ``rgfn`` by subprocess
(``scripts/ingest_candidates.py``) — the same two-env pattern as every other baseline.
"""
