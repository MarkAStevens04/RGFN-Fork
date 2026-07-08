"""``MoleculeSelectionStrategy`` ABC — pick a product batch from one hub (proposal §3a).

Given a selected hub, choose up to ``k`` of its terminal children to synthesize. The three
concrete strategies span the reward/diversity trade-off: reward-greedy, on-policy
(trajectory-balance) weighted, and uniform (maximally diverse). Strategies are protocol-pure
(they read only ``child.reward`` / ``child.log_flow`` / ``child.key``) and deterministic given
a seeded ``rng``.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional


class MoleculeSelectionStrategy(ABC):
    """Select up to ``k`` products from a hub's terminal children."""

    name: str = "base"

    @abstractmethod
    def select(
        self,
        hub,
        k: int,
        *,
        higher_is_better: bool = True,
        rng: Optional[random.Random] = None,
    ) -> List:
        """Return up to ``k`` of ``hub.children`` (:class:`ChildEstimate`-like objects)."""
