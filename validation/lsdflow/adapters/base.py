"""``GFNAdapter`` — the per-model analysis contract (``docs/LSD_FLOW_PROPOSAL.md`` §4b).

**Non-negotiable constraint (§4b):** the four target models are two API families in four
mutually-incompatible conda envs; no two co-import in one process. So the adapter is not one
imported class — it is a subprocess/RPC contract. Each model runs in a per-env worker; the
main process exchanges SMILES + action-ids + logprobs over the boundary (reusing the
``scripts/score_batch.py`` bridge shape). The in-process ``GFNAdapter`` below is the client
mirror; the RGFN anchor happens to *be* the ``rgfn`` env, so its adapter runs in-process.

Two method groups:

  * **sampled-trajectory flow recovery (phase 1, the vertical slice).** ``sample_flow_records``
    is all the DAG needs: it returns the §2 log-terms already composed per terminal
    transition (:class:`FlowSample`). Rewards for sampled children come free from training —
    reuse cached training rewards, do not re-dock (§6).
  * **post-selection full enumeration (phase 2+).** The other five methods of the §4b
    six-method contract (``reward`` / ``enumerate_children`` / ``forward_logprob`` /
    ``backward_logprob`` / ``stop_logprob``) are only needed to enumerate a *selected* hub's
    children beyond what was sampled, and pay fresh oracle cost (budget-capped for docking,
    free for surrogates — §6). A model's per-method support (§4b table) is expressed by which
    of these it overrides; the base raises ``NotImplementedError`` so an un-wired capability
    fails loudly rather than silently returning wrong numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from glue.samplers.lsdflow.records import FlowRecord


@dataclass
class FlowSample:
    """Everything a HubDAG needs from one model on one reward target (§2, §6).

    ``records`` are the observed ``h -> x`` terminal transitions with the four §2 log-terms;
    ``visit_counts`` is the reward-free visitation estimator's raw data; ``terminal_depths``
    is each valid terminal's ``num_reactions`` (the depth-parity severe test, §7 #2).
    """

    records: List[FlowRecord]
    visit_counts: Dict[str, int]
    n_trajectories: int
    terminal_depths: List[int] = field(default_factory=list)
    log_z: float = 0.0
    higher_is_better: bool = True
    model: str = "unknown"
    reward_name: str = "unknown"
    # Per-molecule dynamic-fragment composition (SCENT nested cost, Logs/028):
    # ``{node_key: {"promoted": [fragment_smiles...], "num_reactions": k}}``. Empty for models
    # with no promoted fragments (RGFN — all base blocks free), so the cost model is a no-op there.
    compositions: Dict[str, dict] = field(default_factory=dict)
    # Full synthesis route per molecule (for reconstruction / "how to make it" — Logs/032):
    # ``{stripped_smiles: {"seed": <smiles>, "num_reactions": k, "steps": [{reaction, reactants,
    # input, product}...]}}``, canonical schema = ``glue.samplers.lsdflow.route_steps``.
    #
    # THIS DEFAULT IS A LANDMINE, AND IT WENT OFF. It used to be documented as "empty for adapters
    # that don't emit routes yet (e.g. RGFN)", which made silence look sanctioned: three of four
    # adapters took the default, every run wrote a well-formed routes.json containing ``{}``, exited
    # 0, and was promoted -- and the loss only surfaced when the competitor arm needed routes months
    # later. It is UNRECOVERABLE by then: the trajectory is gone once sampling ends and
    # ``compositions`` keeps only num_reactions, so 36 of 40 sampled cell-seeds need a full re-sample.
    #
    # So an empty dict is no longer an acceptable outcome for a route-bearing generator. Every worker
    # now calls ``_routes.validate_sample_routes`` before exiting, which fails the run on empty and
    # writes route_status.json beside the artifacts. If a generator genuinely has no synthesis routes
    # (FragGFN: its move is an attachment, not a reaction), DECLARE it in ``_routes.ROUTE_CONTRACT``
    # rather than leaving this dict empty and hoping the reader notices.
    #
    # Keys must come from the SAME key function as the flow records (``rgfn_extract._stripped_key``).
    # A route keyed differently from its hub is worse than no route: the lookup returns nothing and
    # the caller prices the hub instead of the child, with no error.
    routes: Dict[str, dict] = field(default_factory=dict)

    @property
    def n_valid_terminals(self) -> int:
        return len(self.terminal_depths)


class GFNAdapter(ABC):
    """In-process client mirror of a per-env GFN worker (§4b)."""

    model_name: str = "base"

    # -- phase 1: sampled-trajectory flow recovery -----------------------------------
    @abstractmethod
    def sample_flow_records(self, n_trajectories: int) -> FlowSample:
        """Sample ``n_trajectories`` and return their §2 terminal-transition flow records."""

    # -- phase 2+: full-enumeration surface (§4b six-method contract) ----------------
    def reward(self, smiles: List[str]) -> List[float]:
        """Active oracle/proxy value for a batch of SMILES (§4b). Override to enable
        full-enumeration analysis; sampled flow recovery does not use it (rewards are
        attached during sampling)."""
        raise NotImplementedError(
            f"{type(self).__name__}.reward is a phase-2 full-enumeration capability "
            "(§6) and is not wired for the sampled-trajectory vertical slice."
        )

    def enumerate_children(self, smiles: str) -> List[Tuple[object, str]]:
        """Applicable one-move products of a state: ``[(move, product_smiles), ...]`` (§4b)."""
        raise NotImplementedError(
            f"{type(self).__name__}.enumerate_children is a phase-2 capability (§6)."
        )

    def forward_logprob(self, state: str, move: object) -> float:
        raise NotImplementedError(
            f"{type(self).__name__}.forward_logprob is a phase-2 capability (§4b)."
        )

    def backward_logprob(self, child: str, parent: str) -> float:
        raise NotImplementedError(
            f"{type(self).__name__}.backward_logprob is a phase-2 capability (§4b, §5)."
        )

    def stop_logprob(self, state: str) -> float:
        raise NotImplementedError(
            f"{type(self).__name__}.stop_logprob is a phase-2 capability (§4b)."
        )

    def close(self) -> None:
        """Release any worker/subprocess/GPU resources. No-op for in-process adapters."""
