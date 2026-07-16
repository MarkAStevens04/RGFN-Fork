"""Cost accounting for LSD-Flow (proposal §11; Logs/028, Logs/039).

Two axes:
  * **Synthesis (bench) cost** — :class:`FragmentCostTable` (recursive per-promoted-fragment unit
    cost from the logged synthesis routes, closure under nesting) + the snapshot loader. Reported as
    reactions/mode.
  * **Measured compute time** (Logs/039) — :class:`EnumTimings` (per-hub enumeration / reward-gen /
    flow-extract wall-clock from the worker's ``enum_timings.json``) + :func:`account_strategy` /
    :func:`head_to_head`, which attribute those measured times over each strategy's hub walk.

The budget-greedy hub-batching-vs-best-candidate comparison that consumes both lives in the AL-ready
``glue.samplers.lsdflow.campaign`` and the ``experiments/lsd_hubs/campaign/`` analysis.
"""

from validation.lsdflow.metrics.cost.compute_time import (  # noqa: F401
    ComputeTimeBreakdown,
    EnumTimings,
    account_strategy,
    head_to_head,
)
from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: F401
    FragmentCostTable,
    load_cost_table_from_snapshot,
)
