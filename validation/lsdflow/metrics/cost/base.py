"""``CostModel`` ABC — synthesis cost of a batch (``docs/LSD_FLOW_PROPOSAL.md`` §11).

A cost model turns "how was this batch built?" into a number, so hub batches (build a parent
once, run ``k`` final reactions) can be compared against independent top-``k`` (build each
molecule from scratch). The PRIMARY metric is reactions-per-mode
(:class:`~validation.lsdflow.metrics.cost.reactions_per_mode.ReactionsPerModeCost`); the
per-mode division lets diversity and cost be traded off on one axis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class CostModel(ABC):
    name: str = "base"

    @abstractmethod
    def batch_reactions(self, hub_depth: int, k: int) -> float:
        """Reactions to build a ``k``-product batch that shares one hub of depth ``hub_depth``."""

    @abstractmethod
    def independent_reactions(self, child_depths: Sequence[int]) -> float:
        """Reactions to build the same ``k`` products independently (no shared parent)."""

    def per_mode(self, total_reactions: float, n_modes: int) -> float:
        """``total_reactions / n_modes`` — lower is better (§11). ``inf`` for zero modes."""
        return total_reactions / n_modes if n_modes > 0 else float("inf")
