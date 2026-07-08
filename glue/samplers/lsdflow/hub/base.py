"""``HubSelectionStrategy`` ABC — rank a HubDAG's hubs, best-first (proposal §3a).

A hub-selection strategy answers "which pre-terminal states are the best batchable
neighborhoods?" It is a batch-selection strategy in the same sense as ``glue/samplers/``,
which is why it lives here. Strategies are **protocol-pure**: they read only the duck-typed
fields declared in :mod:`glue.samplers.lsdflow.records` (so they work identically on the
lightweight AL-path DAG and the validation rich DAG) and defer all flow/uncertainty math to
:mod:`glue.metrics`. A concrete strategy implements :meth:`score` (higher = better); the base
handles gating + sorting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

_NEG_INF = float("-inf")


class HubSelectionStrategy(ABC):
    """Rank hubs by a scalar score (higher is better), after eligibility gating."""

    name: str = "base"

    def __init__(self, min_children: int = 1, min_effective_n: int = 1):
        """
        Args:
            min_children: drop hubs with fewer than this many distinct terminal children —
                a 1-child hub is not a *diversifying* neighborhood (§1).
            min_effective_n: drop hubs with fewer than this many finite per-child flow
                estimates (§7 severe-test #5: estimates from too few children are noise).
        """
        self.min_children = min_children
        self.min_effective_n = min_effective_n

    @abstractmethod
    def score(self, hub, dag) -> float:
        """Scalar hub score; higher = more preferred. ``nan`` sorts last."""

    def is_eligible(self, hub, dag) -> bool:
        finite = [v for v in hub.child_log_flows() if v == v]
        return len(hub.children) >= self.min_children and len(finite) >= self.min_effective_n

    def rank(self, dag) -> List:
        """All eligible hubs, best-first (``nan`` scores demoted to the tail)."""
        eligible = [h for h in dag.hubs_iter() if self.is_eligible(h, dag)]

        def _key(hub):
            s = self.score(hub, dag)
            return _NEG_INF if s != s else s

        return sorted(eligible, key=_key, reverse=True)

    def select(self, dag, n_hubs: int) -> List:
        """The top ``n_hubs`` hubs under :meth:`rank`."""
        return self.rank(dag)[: max(0, n_hubs)]
