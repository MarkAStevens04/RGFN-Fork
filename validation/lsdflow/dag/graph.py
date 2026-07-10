"""``HubDAG`` — the rich, persisted, cross-model DAG (``docs/LSD_FLOW_PROPOSAL.md`` §6).

Built from a model's :class:`~validation.lsdflow.adapters.base.FlowSample`, this wraps the
shared aggregation (:class:`glue.samplers.lsdflow.dag.LiteHubDAG` — so the ``glue`` selection
strategies run on it unchanged, via the §6 duck type) and adds the validation-only extras: a
``networkx`` view for analysis, per-node stats, a run summary, and on-disk persistence keyed by
``(model, reward, run_id)`` (§6). The pipeline never imports this; the AL loop uses the
lightweight DAG directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, Optional

from glue.samplers.lsdflow.dag import Hub, LiteHubDAG
from validation.lsdflow.adapters.base import FlowSample
from validation.lsdflow.dag.node import NodeStats

try:
    import networkx as nx
except Exception:  # pragma: no cover - networkx optional; persistence still works without it
    nx = None


class HubDAG:
    """Rich DAG over one model+reward run. Conforms to the §6 HubDAG duck type."""

    def __init__(self, sample: FlowSample, run_id: Optional[str] = None):
        self.sample = sample
        self.model = sample.model
        self.reward_name = sample.reward_name
        self.run_id = run_id
        self.lite = LiteHubDAG.from_records(
            sample.records,
            total_trajectories=sample.n_trajectories,
            log_z=sample.log_z,
            higher_is_better=sample.higher_is_better,
            visit_counts=sample.visit_counts,
        )
        self._node_stats: Optional[Dict[str, NodeStats]] = None

    # -- §6 HubDAG duck type (delegate to the shared aggregation) --------------------
    @property
    def total_trajectories(self) -> int:
        return self.lite.total_trajectories

    @property
    def log_z(self) -> float:
        return self.lite.log_z

    @property
    def higher_is_better(self) -> bool:
        return self.lite.higher_is_better

    def hubs_iter(self) -> Iterable[Hub]:
        return self.lite.hubs_iter()

    def hub(self, key: str) -> Optional[Hub]:
        return self.lite.hub(key)

    def log_visitation(self, hub: Hub) -> float:
        return self.lite.log_visitation(hub)

    def __len__(self) -> int:
        return len(self.lite)

    # -- node stats + networkx -------------------------------------------------------
    def node_stats(self) -> Dict[str, NodeStats]:
        """Per-molecule aggregates over the run (built once, cached)."""
        if self._node_stats is not None:
            return self._node_stats
        stats: Dict[str, NodeStats] = {}

        def _get(key: str, stereo: str = "") -> NodeStats:
            node = stats.get(key)
            if node is None:
                node = NodeStats(key=key, stereo_key=stereo)
                stats[key] = node
            return node

        higher = self.higher_is_better
        for rec in self.sample.records:
            h = _get(rec.hub_key, rec.hub_stereo_key)
            h.n_as_hub += 1
            h.depth = min(h.depth, rec.hub_depth) if h.n_as_hub > 1 else rec.hub_depth
            c = _get(rec.child_key, rec.child_stereo_key)
            c.n_as_terminal += 1
            c.depth = rec.hub_depth + 1
            if c.best_reward != c.best_reward:
                c.best_reward = rec.reward
            else:
                c.best_reward = (
                    max(c.best_reward, rec.reward) if higher else min(c.best_reward, rec.reward)
                )
        for key, count in self.sample.visit_counts.items():
            _get(key).visit_count = count
        self._node_stats = stats
        return stats

    def to_networkx(self):
        """A ``networkx.DiGraph``: molecule nodes (visit/depth/reward) + hub->child edges
        carrying the recovered ``log_flow`` (§6). Returns ``None`` if networkx is absent."""
        if nx is None:
            return None
        g = nx.DiGraph(model=self.model, reward=self.reward_name, run_id=self.run_id or "")
        for key, s in self.node_stats().items():
            g.add_node(
                key,
                visit_count=s.visit_count,
                depth=s.depth,
                n_as_hub=s.n_as_hub,
                n_as_terminal=s.n_as_terminal,
                best_reward=s.best_reward,
            )
        for hub in self.hubs_iter():
            for child in hub.children:
                g.add_edge(
                    hub.key,
                    child.key,
                    log_flow=child.log_flow,
                    log_pf_move=child.log_pf_move,
                    log_pb_move=child.log_pb_move,
                    log_pf_stop=child.log_pf_stop,
                    reward=child.reward,
                )
        return g

    # -- summary + persistence -------------------------------------------------------
    def summary(self) -> dict:
        hubs = list(self.hubs_iter())
        multichild = [h for h in hubs if h.n_children >= 2]
        depths = [h.depth for h in multichild] or [0]
        return {
            "model": self.model,
            "reward": self.reward_name,
            "run_id": self.run_id,
            "n_trajectories": self.total_trajectories,
            "n_valid_terminals": self.sample.n_valid_terminals,
            "log_z": self.log_z,
            "higher_is_better": self.higher_is_better,
            "n_hubs": len(hubs),
            "n_multichild_hubs": len(multichild),
            "max_children_per_hub": max((h.n_children for h in hubs), default=0),
            "mean_multichild_hub_depth": sum(depths) / len(depths),
            "n_distinct_nodes": len(self.node_stats()),
        }

    def save(self, out_dir) -> Path:
        """Persist records + per-hub summary + meta (+ networkx pickle) keyed by the run."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        with open(out / "records.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "hub_key",
                    "child_key",
                    "reward",
                    "log_reward",
                    "log_pf_move",
                    "log_pb_move",
                    "log_pf_stop",
                    "hub_depth",
                    "hub_stereo_key",
                    "child_stereo_key",
                ]
            )
            for r in self.sample.records:
                w.writerow(
                    [
                        r.hub_key,
                        r.child_key,
                        r.reward,
                        r.log_reward,
                        r.log_pf_move,
                        r.log_pb_move,
                        r.log_pf_stop,
                        r.hub_depth,
                        r.hub_stereo_key,
                        r.child_stereo_key,
                    ]
                )

        with open(out / "hub_summary.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "hub_key",
                    "depth",
                    "visit_count",
                    "n_children",
                    "log_flow_consensus",
                    "log_flow_terminating",
                    "uncertainty",
                    "effective_n",
                    "best_reward",
                    "log_visitation",
                ]
            )
            for h in self.hubs_iter():
                w.writerow(
                    [
                        h.key,
                        h.depth,
                        h.visit_count,
                        h.n_children,
                        h.log_flow_consensus(),
                        h.log_flow_terminating(),
                        h.uncertainty(),
                        h.effective_n(),
                        h.best_reward(self.higher_is_better),
                        self.log_visitation(h),
                    ]
                )

        with open(out / "meta.json", "w") as fh:
            json.dump(self.summary(), fh, indent=2)

        g = self.to_networkx()
        if g is not None:
            try:
                import pickle

                with open(out / "graph.gpickle", "wb") as fh:
                    pickle.dump(g, fh)
            except Exception as exc:  # noqa: BLE001 - persistence of the graph is best-effort
                print(f"[HubDAG] graph pickle skipped: {exc}", flush=True)
        return out
