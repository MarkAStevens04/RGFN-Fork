"""Build a :class:`~validation.lsdflow.dag.graph.HubDAG` from a model's flow sample.

Thin, model-agnostic: any adapter's :class:`~validation.lsdflow.adapters.base.FlowSample`
becomes a rich DAG here (``docs/LSD_FLOW_PROPOSAL.md`` §6). The per-model work (composing the
§2 log-terms) already happened in the adapter; this only aggregates.
"""

from __future__ import annotations

from typing import Optional

from validation.lsdflow.adapters.base import FlowSample
from validation.lsdflow.dag.graph import HubDAG


def build_hub_dag(sample: FlowSample, run_id: Optional[str] = None) -> HubDAG:
    return HubDAG(sample, run_id=run_id)
