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

import time
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
    # Measured per-component wall-clock (Logs/039), seconds — the "where does the acquisition's
    # time go" breakdown behind the hub-batching-vs-best-candidate compute comparison:
    #   sampling_s     — trajectory generation (the policy forward pass)
    #   flow_extract_s — assign_log_probs -> P_F/P_B (sampled DAG + enumerated children)
    #   enumeration_s  — RDKit one-reaction child construction (hub_batching only)
    #   reward_gen_s   — proxy M scoring the enumerated children (hub_batching only)
    #   hub_rank_s     — UCB z-scoring + sort (hub_batching only)
    #   mode_select_s  — reward-gate + Tanimoto diversity accept loop
    # (Docking time is the loop's separate `oracle_score` phase in phase_timings.csv.)
    timing: Dict[str, float] = field(default_factory=dict)


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
        proxy_label_mean: float = 0.0,
        proxy_label_std: float = 1.0,
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
        n_candidate_hubs: int = 150,
        min_hub_visits: int = 1,
        min_hub_depth: int = 1,
        max_hub_depth: int = 3,
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
            max_hubs_walked: safety cap on ranked hubs walked for mode selection.
            n_candidate_hubs: how many hubs to enumerate for ranking. Candidates are the most-visited
                hubs from sampling (visit-count is robust to sparse per-hub sampled children — unlike
                a sampled-``U(h)`` gate, which the boundary artifact starves). ``U(h)`` + ``reward(h)``
                are then computed from each candidate's **enumerated** neighborhood (always many
                children → well-defined), so the UCB ranking is trustworthy. This is the reward-gen
                (enumeration) cost knob.
            min_hub_visits: drop candidate hubs visited fewer times than this (noise floor).
            min_hub_depth/max_hub_depth: keep candidate hubs in this reaction-depth band. Default
                ``[1, 3]`` — skip depth-0 building blocks (huge fan-out, zero amortization — not the
                "build once, diversify" signal, Logs/025) and depth-4 children at the reaction cap
                (``P_B`` unrecoverable).
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
        self.proxy_label_mean = proxy_label_mean
        self.proxy_label_std = proxy_label_std
        self.lam = lam
        self.hub_min_children = hub_min_children
        self.hub_reward_fn = hub_reward_fn
        self.hub_uncertainty_fn = hub_uncertainty_fn
        self.child_policy_name = child_policy
        self.prebuild_k = prebuild_k
        self.max_children_per_hub = max_children_per_hub
        self.max_hubs_walked = max_hubs_walked
        self.n_candidate_hubs = n_candidate_hubs
        self.min_hub_visits = min_hub_visits
        self.min_hub_depth = min_hub_depth
        self.max_hub_depth = max_hub_depth
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
        timing: Dict[str, float] = {}
        records, terminal_routes, visit_counts, n_traj = self._sample(sampler, objective, timing)
        if self.arm == "best_candidate":
            res = self._best_candidate(records, terminal_routes, n_traj, timing)
        else:
            res = self._hub_batching(records, visit_counts, sampler, objective, n_traj, timing)
        res.timing = {k: round(v, 4) for k, v in timing.items()}
        return res

    def _sync(self) -> None:
        """Fence async CUDA work so each timed stage is charged its real wall-clock (Logs/039)."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # --------------------------------------------------------------------- sampling
    def _sample(self, sampler, objective, timing: Dict[str, float]):
        """One sampling pass → (flow records, terminal→route map, n_trajectories).

        Records feed the DAG (hub ranking) and the best-candidate pool; the route map gives every
        sampled terminal its full synthesis route (for best-candidate provenance). Times trajectory
        generation (``sampling_s``) apart from flow recovery + route extraction (``flow_extract_s``).
        """
        records = []
        terminal_routes: Dict[str, dict] = {}
        visit_counts: Dict[str, int] = {}
        n_traj = 0
        it = sampler.get_trajectories_iterator(self.n_sample_trajectories, self.sample_batch_size)
        while True:
            self._sync()
            t0 = time.perf_counter()
            try:
                traj = next(it)
            except StopIteration:
                break
            self._sync()
            timing["sampling_s"] = timing.get("sampling_s", 0.0) + (time.perf_counter() - t0)
            t1 = time.perf_counter()
            recs, visits, n = extract_flow_records(objective, traj, strip_stereo=self.strip_stereo)
            records.extend(recs)
            n_traj += n
            for k, c in visits.items():  # accumulate visit counts (candidate-hub discovery signal)
                visit_counts[k] = visit_counts.get(k, 0) + c
            states_list = traj._states_list
            actions_list = traj._actions_list
            for i, states in enumerate(states_list):
                if not states or not isinstance(states[-1], ReactionStateTerminal):
                    continue
                key = self._key(states[-2].molecule) if len(states) >= 2 else None
                if key is None or key in terminal_routes:
                    continue
                terminal_routes[key] = extract_route(states, actions_list[i])
            self._sync()
            timing["flow_extract_s"] = timing.get("flow_extract_s", 0.0) + (
                time.perf_counter() - t1
            )
        return records, terminal_routes, visit_counts, n_traj

    # --------------------------------------------------------------------- best-candidate arm
    def _best_candidate(self, records, terminal_routes, n_traj, timing) -> AcquisitionResult:
        """Top-``M`` sampled terminals under the same hit-bar + diversity filter (the control)."""
        t_ms = time.perf_counter()
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
        cum_reactions = (
            0  # independent molecules: each charged its own route's num_reactions (no amortization)
        )
        for key, reward in order:
            if len(smiles) >= self.budget_modes:
                break
            if selector.accept(key, reward):
                route = dict(terminal_routes.get(key, {}))
                rx = int(route.get("num_reactions", 0) or 0)
                cum_reactions += rx
                route.update({"reactions_added": rx, "cum_reactions": cum_reactions})
                smiles.append(key)
                routes.append(route)
        timing["mode_select_s"] = timing.get("mode_select_s", 0.0) + (time.perf_counter() - t_ms)
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
    def _hub_batching(
        self, records, visit_counts, sampler, objective, n_traj, timing
    ) -> AcquisitionResult:
        """Enumerate the most-visited hubs, rank them by ``z(reward)+λ·z(U(h))`` computed from their
        **enumerated** neighborhoods, then diversify the best into modes up to the budget.

        ``U(h)`` is taken from *enumerated* children (always many → well-defined), not sampled ones:
        sampling under-counts a hub's children (Logs/025), so a sampled-``U(h)`` gate is starved (the
        smoke's 0-mode failure). Candidate hubs are chosen by **visit count**, which sampling does
        estimate robustly — the genuine high-traffic pre-terminal "hubs"."""
        env = getattr(sampler, "env", None)
        reward = getattr(sampler, "reward", None)
        if env is None or reward is None:
            raise RuntimeError("hub_batching needs sampler.env + sampler.reward (the shaped M).")

        # Stage 1 — candidate hubs = most-visited pre-terminal states in the depth band.
        t_rank = time.perf_counter()
        sampled = LiteHubDAG.from_records(
            records, higher_is_better=self.higher_is_better, visit_counts=visit_counts
        )
        candidates = [
            h
            for h in sampled.hubs_iter()
            if h.visit_count >= self.min_hub_visits
            and self.min_hub_depth <= h.depth <= self.max_hub_depth
        ]
        candidates.sort(key=lambda h: h.visit_count, reverse=True)
        candidates = candidates[: self.n_candidate_hubs]
        timing["hub_rank_s"] = timing.get("hub_rank_s", 0.0) + (time.perf_counter() - t_rank)

        # Stage 2 — enumerate each candidate's full one-reaction neighborhood (scored by M) → the
        # enumerated FlowRecords give well-defined per-hub U(h) + reward(h).
        enum_records: List = []
        for hub in candidates:
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
                timing=timing,  # accumulates enumeration_s / reward_gen_s / flow_extract_s
                sync=self._sync,
            )
            enum_records.extend(recs)
        reward_gen_calls = len(enum_records)  # every enumerated child was scored by M
        enum_dag = LiteHubDAG.from_records(enum_records, higher_is_better=self.higher_is_better)

        # Stage 3 — UCB-rank the enumerated hubs by z(reward)+λ·z(U(h)).
        t_rank2 = time.perf_counter()
        strategy = UcbHubStrategy(
            min_children=self.hub_min_children,
            min_effective_n=self.hub_min_children,
            lam=self.lam,
            reward_fn=self.hub_reward_fn,
            uncertainty_fn=self.hub_uncertainty_fn,
        )
        ranked = strategy.rank(enum_dag)[: self.max_hubs_walked]
        breakdown = {row["hub_key"]: row for row in strategy.score_breakdown(enum_dag)}
        timing["hub_rank_s"] = timing.get("hub_rank_s", 0.0) + (time.perf_counter() - t_rank2)

        # Stage 4 — walk ranked hubs, diversify their enumerated children into modes up to budget.
        child_policy: ChildSelectionPolicy = make_child_policy(self.child_policy_name)
        selector = self._mode_selector()
        # pre-select-K stock: inert without a cost table (RGFN). The seam is here; the SCENT path
        # supplies a FragmentCostTable + promoted fragments and stocks the top-K by build-score.
        prebuilt: set = set()

        smiles, routes, per_hub = [], [], []
        hubs_used = 0
        # Count-once reaction accounting (per round, matching the SCENT selector / campaign model):
        # build each distinct hub scaffold once (its depth reactions), then +1 marginal reaction per
        # accepted mode. RGFN has no promoted fragments, so there are no extra fragment builds.
        built_hubs: Dict[str, int] = {}
        cum_reactions = 0
        for hub in ranked:
            if len(smiles) >= self.budget_modes:
                break
            t_ms = time.perf_counter()
            children = [_EnumChild(c.key, c.reward) for c in hub.children]
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
                    rx = 1  # the final diversifying reaction (one coupling)
                    if hub.key not in built_hubs:  # build the shared scaffold once
                        built_hubs[hub.key] = hub.depth
                        rx += hub.depth
                    cum_reactions += rx
                    smiles.append(child.smiles)
                    routes.append(
                        {
                            "source_hub": hub.key,
                            "product_smiles": child.smiles,
                            "reactions_added": rx,
                            "cum_reactions": cum_reactions,
                        }
                    )
                    n_accepted += 1
            timing["mode_select_s"] = timing.get("mode_select_s", 0.0) + (
                time.perf_counter() - t_ms
            )
            if n_accepted:
                hubs_used += 1
            row = breakdown.get(hub.key, {})
            per_hub.append(
                {
                    "hub_key": hub.key,
                    "depth": hub.depth,
                    "n_enumerated": hub.n_children,
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
        """Build the mode selector, transforming the hit bar into the proxy's output units.

        ``reward_threshold`` (e.g. 6TD3's −1.5) is in **real ΔVina kcal/mol**, but the reward
        generator ``M`` outputs **standardized** predictions (it standardizes labels at ``fit`` time).
        A child's ``record.reward`` is that standardized value, so the bar must be mapped into the
        same space: ``thr_std = (thr_real − label_mean) / label_std``. Ranking is unaffected (z-scored
        ⇒ invariant to this affine map); only the gate needs it. When ``label_std == 1`` /
        ``label_mean == 0`` (a proxy that already emits real units) this is a no-op."""
        thr = self.reward_threshold
        if thr is not None and self.proxy_label_std and self.proxy_label_std > 0:
            thr = (thr - self.proxy_label_mean) / self.proxy_label_std
        return DiverseThresholdModeSelector(thr, self.similarity, self.higher_is_better)

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
