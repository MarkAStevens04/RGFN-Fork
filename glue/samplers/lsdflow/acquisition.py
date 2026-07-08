"""``LSDFlowAcquisition`` — the AL-facing batch-selection entry point (proposal §4a).

**Decision (§4a): the AL loop consumes molecules, not hubs.** Hubs are an internal detail of
*how* the batch is chosen. This keeps the production-side change to a single sampler plugin:
the loop hands over the trajectories it already sampled (plus the objective, so we can recover
per-step log-probs) and gets back a flat list of molecule SMILES. Hub structure, ``U(h)``, and
amortized-cost accounting all stay inside.

    build lightweight HubDAG from the loop's trajectories   (glue.samplers.lsdflow.dag)
        -> hub_strategy ranks hubs                          (may use U(h))
        -> molecule_strategy picks products per hub
        -> flatten (dedup) to a molecule batch of batch_size

**Deferred to phase 2 (§4a):** a hub-aware loop that acquires a parent + its children as a
unit and credits the parent's synthesis once against the oracle budget. That changes the
loop's batch abstraction and touches ``glue/active_learning/loop.py``; v1 is a drop-in that
returns molecules.

Signature note: unlike the proposal's sketch ``select_batch(trajectories, proxy)``, this takes
the ``objective`` explicitly — recovering ``log P_F`` / ``log P_B`` requires
``objective.assign_log_probs`` (the trained forward+backward policies), which the trajectories
alone don't carry. In the AL loop the objective is ``trainer.objective``. See
``docs/REFACTOR_LOG.md`` for the loop-wiring note (the current loop has no pluggable-sampler
hook yet, so phase-2 integration is a small, tracked ``loop.py`` change).
"""

from __future__ import annotations

import random
from typing import List, Optional, Union

import gin

from glue.samplers.lsdflow.dag import LiteHubDAG
from glue.samplers.lsdflow.hub.base import HubSelectionStrategy
from glue.samplers.lsdflow.hub.registry import get_hub_strategy
from glue.samplers.lsdflow.molecule.base import MoleculeSelectionStrategy
from glue.samplers.lsdflow.molecule.registry import get_molecule_strategy
from glue.samplers.lsdflow.rgfn_extract import extract_flow_records


@gin.configurable()
class LSDFlowAcquisition:
    """Batch-selection strategy: choose hubs internally, return their diversified products."""

    def __init__(
        self,
        hub_strategy: Union[str, HubSelectionStrategy] = "highest_terminating_flow",
        molecule_strategy: Union[str, MoleculeSelectionStrategy] = "topk_reward",
        batch_size: int = 96,
        per_hub: int = 8,
        seed: int = 0,
        higher_is_better: Optional[bool] = None,
        strip_stereo: bool = True,
    ):
        """
        Args:
            hub_strategy / molecule_strategy: a strategy instance, or a registry name
                (resolved via ``get_hub_strategy`` / ``get_molecule_strategy``).
            batch_size: number of molecules to return.
            per_hub: max products drawn from a single hub before moving to the next-ranked
                hub — the concurrency knob (§11: batch size from one shared parent).
            seed: seeds the molecule strategy's RNG for reproducible batches.
            higher_is_better: reward orientation; if ``None`` it is read from the proxy
                passed to :meth:`select_batch` (falling back to ``True``).
            strip_stereo: aggregate hubs on the stereo-stripped cross-model key (§6).
        """
        self.hub_strategy = (
            get_hub_strategy(hub_strategy) if isinstance(hub_strategy, str) else hub_strategy
        )
        self.molecule_strategy = (
            get_molecule_strategy(molecule_strategy)
            if isinstance(molecule_strategy, str)
            else molecule_strategy
        )
        self.batch_size = batch_size
        self.per_hub = max(1, per_hub)
        self.seed = seed
        self.higher_is_better = higher_is_better
        self.strip_stereo = strip_stereo

    # ------------------------------------------------------------------ DAG building
    def build_dag(self, trajectories, objective, log_z: float = 0.0) -> LiteHubDAG:
        """Recover a lightweight HubDAG from sampled trajectories (proposal §4a step 1)."""
        higher = self.higher_is_better if self.higher_is_better is not None else True
        records, visit_counts, n_traj = extract_flow_records(
            objective, trajectories, strip_stereo=self.strip_stereo
        )
        return LiteHubDAG.from_records(
            records,
            total_trajectories=n_traj,
            log_z=log_z,
            higher_is_better=higher,
            visit_counts=visit_counts,
        )

    # ------------------------------------------------------------------ selection
    def select_grouped(self, dag: LiteHubDAG) -> List:
        """Rank hubs and pick products per hub, keeping the hub->children grouping (§4a
        steps 2-3). Returns ``[(hub, [child, ...]), ...]`` in hub-rank order, deduplicated
        across hubs and capped at ``batch_size`` molecules total. The grouping is what the
        amortized-cost accounting needs (one parent built per group)."""
        rng = random.Random(self.seed)
        ranked = self.hub_strategy.rank(dag)
        groups: List = []
        seen = set()
        total = 0
        for hub in ranked:
            if total >= self.batch_size:
                break
            picks = self.molecule_strategy.select(
                hub, self.per_hub, higher_is_better=dag.higher_is_better, rng=rng
            )
            chosen = []
            for child in picks:
                key = getattr(child, "stereo_key", "") or child.key
                if key in seen:
                    continue
                seen.add(key)
                chosen.append(child)
                total += 1
                if total >= self.batch_size:
                    break
            if chosen:
                groups.append((hub, chosen))
        return groups

    def select_from_dag(self, dag: LiteHubDAG) -> List[str]:
        """Rank hubs, pick products per hub, flatten+dedup to ``batch_size`` (§4a steps 2-4)."""
        out: List[str] = []
        for _hub, children in self.select_grouped(dag):
            for child in children:
                out.append(getattr(child, "stereo_key", "") or child.key)
        return out[: self.batch_size]

    def select_batch(
        self,
        sampled_trajectories,
        proxy=None,
        *,
        objective,
        log_z: float = 0.0,
    ) -> List[str]:
        """The loop-facing call: trajectories (+ objective) -> a flat molecule batch."""
        if self.higher_is_better is None and proxy is not None:
            self.higher_is_better = bool(getattr(proxy, "higher_is_better", True))
        dag = self.build_dag(sampled_trajectories, objective, log_z=log_z)
        return self.select_from_dag(dag)
