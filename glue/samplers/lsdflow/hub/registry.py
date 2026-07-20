"""Name -> hub-selection-strategy registry (proposal §1: "adding a new strategy must be
trivial").

The harness and configs refer to strategies by their short ``name``; ``glue`` code and gin
configs can also reference the classes directly (each is ``@gin.configurable``). To add a
strategy: implement it under this package, decorate with ``@gin.configurable``, and add it to
``_STRATEGIES`` below (and make sure it is imported on the ``glue.registry`` path).
"""

from __future__ import annotations

from typing import Dict, List, Type

from glue.samplers.lsdflow.hub.base import HubSelectionStrategy
from glue.samplers.lsdflow.hub.highest_flow import HighestFlowStrategy
from glue.samplers.lsdflow.hub.highest_terminating_flow import (
    HighestTerminatingFlowStrategy,
)
from glue.samplers.lsdflow.hub.highest_visitation import HighestVisitationStrategy
from glue.samplers.lsdflow.hub.lowest_uncertainty import LowestUncertaintyStrategy
from glue.samplers.lsdflow.hub.most_modes import MostModesStrategy
from glue.samplers.lsdflow.hub.parent_of_topk import ParentOfTopKStrategy
from glue.samplers.lsdflow.hub.ucb import UcbHubStrategy

_STRATEGIES: Dict[str, Type[HubSelectionStrategy]] = {
    cls.name: cls
    for cls in (
        HighestTerminatingFlowStrategy,
        HighestFlowStrategy,
        MostModesStrategy,
        ParentOfTopKStrategy,
        HighestVisitationStrategy,
        LowestUncertaintyStrategy,
        UcbHubStrategy,
    )
}


def available() -> List[str]:
    return sorted(_STRATEGIES)


def register(cls: Type[HubSelectionStrategy]) -> Type[HubSelectionStrategy]:
    _STRATEGIES[cls.name] = cls
    return cls


def get_hub_strategy(name: str, **kwargs) -> HubSelectionStrategy:
    try:
        cls = _STRATEGIES[name]
    except KeyError:
        raise KeyError(f"Unknown hub strategy {name!r}. Available: {available()}") from None
    return cls(**kwargs)
