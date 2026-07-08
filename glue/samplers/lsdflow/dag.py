"""``LiteHubDAG`` — the lightweight, in-env hub aggregation for the AL path.

``docs/LSD_FLOW_PROPOSAL.md`` §3b/§4a: the active-learning acquisition builds a *lightweight*
HubDAG from the trajectories the loop already sampled (no adapter, single env). This is that
object. It is a pure aggregation over :class:`~glue.samplers.lsdflow.records.FlowRecord`s —
no ``networkx``, no persistence, no RDKit — so it is cheap to build every AL round.

The heavier cross-env / persisted / analysis DAG lives in ``validation/lsdflow/dag`` and
exposes the same protocol (§6), so the selection strategies work on either unchanged.

Aggregation choices (documented because they affect ``U(h)`` — §2):
  * **children are deduplicated by their canonical (stereo-stripped) key.** A hub's terminal
    children are its *distinct high-reward products* (§1); counting the same product twice
    (it was resampled) would artificially shrink ``U(h)``. The first estimate seen for a
    given ``(hub, child)`` is kept; the observation multiplicity is retained separately for
    the reward-free visitation estimate.
  * **hub depth is the minimum** ``num_reactions`` observed for that hub across records — the
    cheapest known route to build it (used by the depth-parity severe test and the
    reactions-per-mode cost).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from glue.metrics import lsdflow_flow, uncertainty
from glue.samplers.lsdflow.records import FlowRecord


class ChildEstimate:
    """One terminal child ``x`` of a hub ``h`` with its recovered ``log F_hat(h; x)``."""

    __slots__ = (
        "key",
        "stereo_key",
        "reward",
        "log_reward",
        "log_pf_move",
        "log_pb_move",
        "log_pf_stop",
        "log_flow",
        "obs_count",
    )

    def __init__(self, record: FlowRecord):
        self.key = record.child_key
        self.stereo_key = record.child_stereo_key
        self.reward = record.reward
        self.log_reward = record.log_reward
        self.log_pf_move = record.log_pf_move
        self.log_pb_move = record.log_pb_move
        self.log_pf_stop = record.log_pf_stop
        self.log_flow = lsdflow_flow.log_flow_estimate(
            log_reward=record.log_reward,
            log_pb_move=record.log_pb_move,
            log_pf_move=record.log_pf_move,
            log_pf_stop=record.log_pf_stop,
        )
        self.obs_count = 1

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"ChildEstimate({self.key[:24]!r}, R={self.reward:.3g}, logF={self.log_flow:.3g})"


class Hub:
    """A pre-terminal state ``h`` and the distinct terminal children observed from it."""

    __slots__ = ("key", "stereo_key", "depth", "visit_count", "_children")

    def __init__(self, key: str, stereo_key: str = "", depth: int = 0):
        self.key = key
        self.stereo_key = stereo_key
        self.depth = depth
        self.visit_count = 0
        self._children: Dict[str, ChildEstimate] = {}

    # -- population ------------------------------------------------------------------
    def add(self, record: FlowRecord) -> None:
        existing = self._children.get(record.child_key)
        if existing is None:
            self._children[record.child_key] = ChildEstimate(record)
        else:
            existing.obs_count += 1
        self.depth = min(self.depth, record.hub_depth) if self._children else record.hub_depth

    # -- views ----------------------------------------------------------------------
    @property
    def children(self) -> List[ChildEstimate]:
        return list(self._children.values())

    @property
    def n_children(self) -> int:
        return len(self._children)

    def child_log_flows(self) -> List[float]:
        return [c.log_flow for c in self._children.values()]

    def child_rewards(self) -> List[float]:
        return [c.reward for c in self._children.values()]

    # -- aggregate flow / uncertainty (delegate to glue.metrics — the §2 math) -------
    def log_flow_consensus(self) -> float:
        return lsdflow_flow.consensus_log_flow(self.child_log_flows())

    def log_flow_terminating(self) -> float:
        return lsdflow_flow.total_terminating_log_flow(self.child_log_flows())

    def uncertainty(self) -> float:
        return uncertainty.flow_matching_uncertainty(self.child_log_flows())

    def effective_n(self) -> int:
        return uncertainty.effective_sample_count(self.child_log_flows())

    def best_reward(self, higher_is_better: bool) -> float:
        rewards = self.child_rewards()
        if not rewards:
            return float("nan")
        return max(rewards) if higher_is_better else min(rewards)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"Hub({self.key[:24]!r}, depth={self.depth}, n_children={self.n_children})"


class LiteHubDAG:
    """In-memory aggregation of :class:`FlowRecord`s into hubs, for the AL path (§4a)."""

    def __init__(
        self,
        total_trajectories: int = 0,
        log_z: float = 0.0,
        higher_is_better: bool = True,
    ):
        self.total_trajectories = total_trajectories
        self.log_z = log_z
        self.higher_is_better = higher_is_better
        self._hubs: Dict[str, Hub] = {}

    # -- population ------------------------------------------------------------------
    def add_record(self, record: FlowRecord) -> None:
        hub = self._hubs.get(record.hub_key)
        if hub is None:
            hub = Hub(record.hub_key, record.hub_stereo_key, record.hub_depth)
            self._hubs[record.hub_key] = hub
        hub.add(record)

    def set_visit_counts(self, visit_counts: Dict[str, int]) -> None:
        """Attach per-node trajectory visit counts (for the reward-free estimators)."""
        for key, hub in self._hubs.items():
            hub.visit_count = visit_counts.get(key, sum(c.obs_count for c in hub.children))

    # -- views ----------------------------------------------------------------------
    def hubs_iter(self) -> Iterable[Hub]:
        return self._hubs.values()

    def hub(self, key: str) -> Optional[Hub]:
        return self._hubs.get(key)

    def __len__(self) -> int:
        return len(self._hubs)

    def log_visitation(self, hub: Hub) -> float:
        return lsdflow_flow.log_visitation_estimate(
            hub.visit_count, self.total_trajectories, self.log_z
        )

    # -- construction ----------------------------------------------------------------
    @classmethod
    def from_records(
        cls,
        records: Sequence[FlowRecord],
        *,
        total_trajectories: int = 0,
        log_z: float = 0.0,
        higher_is_better: bool = True,
        visit_counts: Optional[Dict[str, int]] = None,
    ) -> "LiteHubDAG":
        dag = cls(
            total_trajectories=total_trajectories or 0,
            log_z=log_z,
            higher_is_better=higher_is_better,
        )
        for record in records:
            dag.add_record(record)
        if visit_counts is not None:
            dag.set_visit_counts(visit_counts)
        else:
            dag.set_visit_counts({})
        return dag
