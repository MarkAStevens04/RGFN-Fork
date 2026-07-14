"""Cost accounting for LSD-Flow (proposal §11; Logs/028).

Nested dynamic-fragment cost: :class:`FragmentCostTable` (recursive per-promoted-fragment unit
cost from the logged synthesis routes, closure under nesting) + the snapshot loader. The
budget-greedy hub-batching-vs-best-candidate comparison that consumes this lives in the AL-ready
``glue.samplers.lsdflow.campaign`` and the ``experiments/lsd_hubs/campaign/`` analysis.
"""

from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: F401
    FragmentCostTable,
    load_cost_table_from_snapshot,
)
