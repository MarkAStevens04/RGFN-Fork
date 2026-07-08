"""Name -> cost-model registry (``docs/LSD_FLOW_PROPOSAL.md`` §11).

PRIMARY (wired): ``reactions_per_mode``. SECONDARY (phase 2): ``amortization_ratio`` — the
SCENT-comparable metric using SCENT's own yield/reactant-cost tables from the priced
``glue_standard_v1`` library; declared here, built when the SCENT comparison phase begins.
"""

from __future__ import annotations

from typing import Dict, List, Type

from validation.lsdflow.metrics.cost.base import CostModel
from validation.lsdflow.metrics.cost.reactions_per_mode import ReactionsPerModeCost

_COST_MODELS: Dict[str, Type[CostModel]] = {c.name: c for c in (ReactionsPerModeCost,)}
_PLANNED = {"amortization_ratio": "phase 2 — SCENT-comparable, needs the priced library (§11)"}


def available() -> List[str]:
    return sorted(_COST_MODELS)


def get_cost_model(name: str, **kwargs) -> CostModel:
    if name in _COST_MODELS:
        return _COST_MODELS[name](**kwargs)
    if name in _PLANNED:
        raise NotImplementedError(f"Cost model {name!r} not built: {_PLANNED[name]}")
    raise KeyError(f"Unknown cost model {name!r}. Available: {available()}")
