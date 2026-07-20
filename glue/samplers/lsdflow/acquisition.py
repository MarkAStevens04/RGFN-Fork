"""``LSDFlowAcquisition`` — the AL-facing hub-batching acquisition (proposal §4a).

The active-learning loop's scarce budget is **oracle (docking) calls**. This is the batch-selection
strategy that spends them well: instead of docking whatever the policy happened to sample, it reads
the trained flow field, ranks pre-terminal **hubs** by an explore/exploit score
(``score(h) = z(reward(h)) + λ·U(h)`` — :class:`~glue.samplers.lsdflow.hub.ucb.UcbHubStrategy`),
then diversifies the best hubs into a batch of distinct high-reward **modes** to dock. It is a drop-in
sampler: the loop hands it the trained sampler + objective and gets a flat molecule batch back — the
loop's fit → train → sample → dock → grow structure is unchanged (proposal §4a v1).

Three arms (the loop's ``acquisition`` knob), all returning a flat batch so the loop is arm-agnostic:

- **``hub_batching``** — the LSD-Flow arm. Sample → ``LiteHubDAG`` → UCB hub rank → walk hubs
  best-first, **enumerate** each hub's one-reaction children (scored by the reward-generator ``M`` —
  the "reward-gen calls" axis, unlimited), pre-select-K high-value dynamic-library fragments (inert
  until the generator has a dynamic library, e.g. SCENT — RGFN has none), keep ``free_frag`` children,
  accept reward-gated + Tanimoto-diverse **modes** up to the per-round budget.
- **``best_candidate``** — the control (the researcher's "policy"): rank the *sampled* terminals by
  ``M`` and apply the **same** hit-bar + diversity filter to the same budget. Reuses scores from
  sampling → 0 new reward-gen calls. Isolates what the flow-based hub selection buys.
- (``random`` — the forgiving floor — stays in the loop: uniform policy, no filters, N raw samples.)

**Two cost axes, both tracked** (the researcher's ask): **oracle calls** = molecules docked this round
(the batch, ``budget_modes`` cap) — the Fig.7 x-axis; **reward-gen calls** = ``M`` evaluations spent
enumerating children (unlimited; the compute axis pre-select-K attacks). Plus **avg molecules/hub**.

**Modularity (explicit requirement).** The hub-selection strategy, its ``U(h)`` and ``reward(h)``
terms (via :mod:`glue.samplers.lsdflow.hub.ucb`), the within-hub child policy
(:mod:`glue.samplers.lsdflow.child_select`), and the mode selector
(:mod:`glue.samplers.lsdflow.mode_select`) are all swappable behind stable interfaces. Adding a new
molecule-selection rule (e.g. "pick the most flow-divergent children") is a new ``ChildSelectionPolicy``;
a new acquisition score is a new hub strategy — nothing else changes.

This module is rgfn-bound (it samples + enumerates through the live model); the *selection* logic it
calls (DAG, hub strategies, campaign primitives, mode selector) is pure, so the same pieces drive the
cross-env SCENT path where selection runs in the ``rgfn`` env on the worker's enumeration files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gin
import torch

from glue.active_learning.route import extract_route
from glue.samplers.lsdflow.child_select import ChildSelectionPolicy, make_child_policy
from glue.samplers.lsdflow.dag import LiteHubDAG
from glue.samplers.lsdflow.hub.ucb import UcbHubStrategy
from glue.samplers.lsdflow.mode_select import DiverseThresholdModeSelector
from glue.samplers.lsdflow.rgfn_enumerate import (
    enumerate_terminal_children,
    hub_state_from_smiles,
)
from glue.samplers.lsdflow.rgfn_extract import extract_flow_records
from rgfn.gfns.reaction_gfn.api.reaction_api import ReactionStateTerminal

try:  # RDKit present in-env; degrade to the raw SMILES key if a round-trip fails.
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

_ARMS = ("hub_batching", "best_candidate")


@dataclass
class AcquisitionResult:
    """A round's query batch + the provenance/accounting the loop logs and the curve reads."""

    smiles: List[str]
    routes: List[dict]
    arm: str
    oracle_calls: int = 0  # molecules docked this round (== len(smiles)); Fig.7 x-axis
    reward_gen_calls: int = 0  # M evaluations spent (enumerated children scored); compute axis
    n_hubs_used: int = 0  # distinct hubs that contributed >=1 accepted mode
    avg_mols_per_hub: float = float("nan")
    per_hub: List[dict] = field(
        default_factory=list
    )  # hub_key, n_children, n_accepted, U(h), score
    n_trajectories: int = 0
    n_hubs_ranked: int = 0


@gin.configurable()
class LSDFlowAcquisition:
    """Batch-selection strategy: ranked hubs → diversified modes → flat docking batch (§4a)."""

    def __init__(
        self,
        arm: str = "hub_batching",
        *,
        n_sample_trajectories: int = 2000,
        sample_batch_size: int = 100,
        higher_is_better: bool = False,
        budget_modes: int = 100,
        reward_threshold: Optional[float] = -1.5,
        similarity: float = 0.5,
        # hub ranking (UCB) --------------------------------------------------------------
        lam: float = 1.0,
        hub_min_children: int = 2,
        hub_reward_fn="best_child",
        hub_uncertainty_fn="flow_variance",
        # within-hub molecule selection + pre-select-K -----------------------------------
        child_policy: str = "free_frag",
        prebuild_k: int = 20,
        max_children_per_hub: int = 2000,
        max_hubs_walked: int = 1000,
        enumerate_chunk_size: int = 64,
        strip_stereo: bool = True,
    ):
        """
        Args:
            arm: ``"hub_batching"`` (LSD-Flow) or ``"best_candidate"`` (control).
            n_sample_trajectories/sample_batch_size: how many trajectories to sample to build the
                hub DAG + candidate pool each round.
            higher_is_better: oracle/reward orientation. MUST match the proxy ``M``. 6TD3 ``dvina``
                is ``False`` (more negative = better glue).
            budget_modes: docking budget — accept at most this many diverse modes per round (the
                researcher's 100/turn). This is the round's oracle-call count.
            reward_threshold: the "mode" hit bar in ``M``'s units. For 6TD3 this is the glue/decoy
                discrimination cut **−1.5** (Logs/002; Youden −1.58, Logs/006) — a child is a mode
                iff ``M(x) <= −1.5``. ``None`` disables the hit gate (diversity + budget only).
            similarity: Tanimoto (ECFP r=3/2048) diversity cutoff for modes (campaign default 0.5).
            lam: λ, the exploration weight on the z-scored ``U(h)`` term of the hub score.
            hub_min_children/hub_reward_fn/hub_uncertainty_fn: :class:`UcbHubStrategy` knobs
                (swappable exploitation / exploration terms).
            child_policy: within-hub child ordering — ``"free_frag"`` (pre-select-K compatible) or
                ``"reward"`` (naive). For RGFN (no dynamic library) both are reward-order.
            prebuild_k: pre-select-K — stock the top-K high-value dynamic-library fragments up front
                (Logs/037). Inert without a cost table (RGFN); the seam for the SCENT run.
            max_children_per_hub: per-hub enumeration cap (docking-budget guard, §6). For an
                in-loop *proxy* reward-gen this bounds compute, not oracle calls.
            max_hubs_walked: safety cap on hubs enumerated in a round (stop once budget met anyway).
            strip_stereo: use the stereo-stripped cross-model node key (§6).
        """
        if arm not in _ARMS:
            raise ValueError(f"arm must be one of {_ARMS}, got {arm!r}.")
        self.arm = arm
        self.n_sample_trajectories = n_sample_trajectories
        self.sample_batch_size = sample_batch_size
        self.higher_is_better = higher_is_better
        self.budget_modes = budget_modes
        self.reward_threshold = reward_threshold
        self.similarity = similarity
        self.lam = lam
        self.hub_min_children = hub_min_children
        self.hub_reward_fn = hub_reward_fn
        self.hub_uncertainty_fn = hub_uncertainty_fn
        self.child_policy_name = child_policy
        self.prebuild_k = prebuild_k
        self.max_children_per_hub = max_children_per_hub
        self.max_hubs_walked = max_hubs_walked
        self.enumerate_chunk_size = enumerate_chunk_size
        self.strip_stereo = strip_stereo

    # --------------------------------------------------------------------- public
    @torch.no_grad()
    def select_batch(self, sampler, objective) -> AcquisitionResult:
        """Propose the round's docking batch from the trained ``sampler`` + ``objective``.

        ``sampler`` must expose ``get_trajectories_iterator``, ``.env`` and ``.reward`` (the shaped
        reward wrapping the proxy ``M``) — the RGFN ``Sampler`` contract. ``objective`` provides
        ``assign_log_probs`` (the §2 P_F/P_B). Nothing is docked here; the loop docks the returned
        SMILES with the expensive oracle ``O``.
        """
        records, terminal_routes, n_traj = self._sample(sampler, objective)
        if self.arm == "best_candidate":
            return self._best_candidate(records, terminal_routes, n_traj)
        return self._hub_batching(records, sampler, objective, n_traj)

    # --------------------------------------------------------------------- sampling
    def _sample(self, sampler, objective):
        """One sampling pass → (flow records, terminal→route map, n_trajectories).

        Records feed the DAG (hub ranking) and the best-candidate pool; the route map gives every
        sampled terminal its full synthesis route (for best-candidate provenance)."""
        records = []
        terminal_routes: Dict[str, dict] = {}
        n_traj = 0
        for traj in sampler.get_trajectories_iterator(
            self.n_sample_trajectories, self.sample_batch_size
        ):
            recs, _visits, n = extract_flow_records(objective, traj, strip_stereo=self.strip_stereo)
            records.extend(recs)
            n_traj += n
            states_list = traj._states_list
            actions_list = traj._actions_list
            for i, states in enumerate(states_list):
                if not states or not isinstance(states[-1], ReactionStateTerminal):
                    continue
                key = self._key(states[-2].molecule) if len(states) >= 2 else None
                if key is None or key in terminal_routes:
                    continue
                terminal_routes[key] = extract_route(states, actions_list[i])
        return records, terminal_routes, n_traj

    # --------------------------------------------------------------------- best-candidate arm
    def _best_candidate(self, records, terminal_routes, n_traj) -> AcquisitionResult:
        """Top-``M`` sampled terminals under the same hit-bar + diversity filter (the control)."""
        # Dedup candidates by canonical key, keeping the best-reward observation of each.
        best: Dict[str, float] = {}
        for r in records:
            v = r.reward
            if v != v:
                continue
            cur = best.get(r.child_key)
            if cur is None or ((v > cur) == self.higher_is_better):
                best[r.child_key] = v
        order = sorted(
            best.items(),
            key=lambda kv: kv[1] if kv[1] == kv[1] else float("-inf"),
            reverse=self.higher_is_better,
        )
        selector = self._mode_selector()
        smiles, routes = [], []
        for key, reward in order:
            if len(smiles) >= self.budget_modes:
                break
            if selector.accept(key, reward):
                smiles.append(key)
                routes.append(terminal_routes.get(key, {}))
        return AcquisitionResult(
            smiles=smiles,
            routes=routes,
            arm=self.arm,
            oracle_calls=len(smiles),
            reward_gen_calls=len(best),  # candidates scored during sampling (reused, not fresh)
            n_hubs_used=0,
            avg_mols_per_hub=float("nan"),
            n_trajectories=n_traj,
        )

    # --------------------------------------------------------------------- hub-batching arm
    def _hub_batching(self, records, sampler, objective, n_traj) -> AcquisitionResult:
        """UCB-rank hubs, walk best-first, enumerate + diversify into modes up to the budget."""
        env = getattr(sampler, "env", None)
        reward = getattr(sampler, "reward", None)
        if env is None or reward is None:
            raise RuntimeError("hub_batching needs sampler.env + sampler.reward (the shaped M).")

        dag = LiteHubDAG.from_records(records, higher_is_better=self.higher_is_better)
        strategy = UcbHubStrategy(
            min_children=self.hub_min_children,
            min_effective_n=self.hub_min_children,
            lam=self.lam,
            reward_fn=self.hub_reward_fn,
            uncertainty_fn=self.hub_uncertainty_fn,
        )
        ranked = strategy.rank(dag)[: self.max_hubs_walked]
        breakdown = {row["hub_key"]: row for row in strategy.score_breakdown(dag)}

        child_policy: ChildSelectionPolicy = make_child_policy(self.child_policy_name)
        selector = self._mode_selector()
        # pre-select-K stock: inert without a cost table (RGFN). The seam is here; the SCENT path
        # supplies a FragmentCostTable + promoted fragments and stocks the top-K by build-score.
        prebuilt: set = set()

        smiles, routes, per_hub = [], [], []
        reward_gen_calls = 0
        hubs_used = 0
        for hub in ranked:
            if len(smiles) >= self.budget_modes:
                break
            hub_state = hub_state_from_smiles(hub.stereo_key or hub.key, hub.depth)
            if hub_state is None:
                continue
            recs, _n_paths = enumerate_terminal_children(
                env,
                objective,
                reward,
                hub_state,
                max_children=self.max_children_per_hub,
                chunk_size=self.enumerate_chunk_size,
                strip_stereo=self.strip_stereo,
            )
            reward_gen_calls += len(recs)  # every enumerated child was scored by M
            children = [_EnumChild(r.child_key, r.reward) for r in recs]
            ordered = child_policy.order(
                children,
                higher_is_better=self.higher_is_better,
                cost_table=None,
                available=set(prebuilt),
            )
            n_accepted = 0
            for child in ordered:
                if len(smiles) >= self.budget_modes:
                    break
                if selector.accept(child.smiles, child.reward):
                    smiles.append(child.smiles)
                    routes.append({"source_hub": hub.key, "product_smiles": child.smiles})
                    n_accepted += 1
            if n_accepted:
                hubs_used += 1
            row = breakdown.get(hub.key, {})
            per_hub.append(
                {
                    "hub_key": hub.key,
                    "depth": hub.depth,
                    "n_enumerated": len(recs),
                    "n_accepted": n_accepted,
                    "uncertainty": row.get("uncertainty"),
                    "reward": row.get("reward"),
                    "score": row.get("score"),
                }
            )
        return AcquisitionResult(
            smiles=smiles,
            routes=routes,
            arm=self.arm,
            oracle_calls=len(smiles),
            reward_gen_calls=reward_gen_calls,
            n_hubs_used=hubs_used,
            avg_mols_per_hub=(len(smiles) / hubs_used) if hubs_used else float("nan"),
            per_hub=per_hub,
            n_trajectories=n_traj,
            n_hubs_ranked=len(ranked),
        )

    # --------------------------------------------------------------------- helpers
    def _mode_selector(self) -> DiverseThresholdModeSelector:
        return DiverseThresholdModeSelector(
            self.reward_threshold, self.similarity, self.higher_is_better
        )

    def _key(self, molecule) -> Optional[str]:
        """Stereo-stripped canonical key (§6), matching ``rgfn_extract``'s record keys."""
        smi = getattr(molecule, "smiles", None)
        if smi is None:
            return None
        if not self.strip_stereo or Chem is None:
            return smi
        try:
            mol = molecule.rdkit_mol
            return Chem.MolToSmiles(mol, isomericSmiles=False) if mol is not None else smi
        except Exception:
            return smi


class _EnumChild:
    """Minimal duck-typed child for the child policy / mode selector (``.smiles``, ``.reward``,
    ``.added_promoted``). RGFN attaches no promoted fragments, so ``added_promoted`` is empty."""

    __slots__ = ("smiles", "reward", "added_promoted")

    def __init__(self, smiles: str, reward: float):
        self.smiles = smiles
        self.reward = reward
        self.added_promoted = ()
