"""Exhaustive one-reaction child enumeration for a hub (proposal §4b `enumerate_children`, §6).

Sampling only credits a hub with a terminal child when a trajectory happens to **stop exactly
one reaction downstream** — so a hub looks sparse even when many good products branch from it in
one step (the boundary-artifact finding, Logs/025). This module enumerates **all** terminal
products reachable from a hub by exactly one reaction then stop, so a *selected* hub's true
diversification neighborhood is measured, not just the part sampling happened to hit. Free for
surrogate rewards (sEH/DRD2); budget-capped for docking (§6 — hence ``max_children``).

It reuses the sampled-path machinery exactly: per enumerated child it rebuilds the micro-step
trajectory ``h -[A:template -> B:reactant* -> C:commit]-> A(x,k+1) -[stop]-> Terminal(x)`` using
the env's **own** forward/backward action spaces (the same objects the sampler records), then
hands the batch to :func:`glue.samplers.lsdflow.rgfn_extract.extract_flow_records`. So the
enumerated flow terms (``P_F``, ``P_F(stop)``, ``P_B``, ``R``) are computed by identical code to
the sampled ones and the enumerated ``FlowRecord``s merge with / are directly comparable to the
sampled DAG.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

from glue.samplers.lsdflow.records import FlowRecord
from glue.samplers.lsdflow.rgfn_extract import extract_flow_records
from rgfn.api.trajectories import Trajectories
from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionActionC,
    ReactionStateA,
    ReactionStateB,
    ReactionStateC,
    ReactionStateTerminal,
)


def _is_stop(action) -> bool:
    """A phase-A action with no reaction is the terminate/stop action."""
    return getattr(action, "anchored_reaction", None) is None


def _enumerate_product_paths(env, hub_state: ReactionStateA, max_children: int) -> List[list]:
    """DFS every one-reaction micro-step path ``hub -> A(x, k+1)``.

    Returns a list of paths; each path is a list of ``(state, forward_action_space, action,
    next_state)`` micro-steps ending at the product ``ReactionStateA``. Stops once ``max_children``
    complete paths are collected (the truncation is reported by the caller — never silent).
    """
    out: List[list] = []

    def dfs(state, steps):
        if len(out) >= max_children:
            return
        fas = env.get_forward_action_spaces([state])[0]
        if not hasattr(fas, "get_possible_actions_indices"):
            return  # EarlyTerminate action space — dead end
        if isinstance(state, ReactionStateA):
            # Root of the reaction: expand template actions only (exclude the stop action).
            for idx in fas.get_possible_actions_indices():
                act = fas.get_action_at_idx(idx)
                if _is_stop(act):
                    continue
                nxt = env.apply_forward_actions([state], [act])[0]
                dfs(nxt, steps + [(state, fas, act, nxt)])
                if len(out) >= max_children:
                    return
        elif isinstance(state, ReactionStateB):
            for idx in fas.get_possible_actions_indices():
                act = fas.get_action_at_idx(idx)
                nxt = env.apply_forward_actions([state], [act])[0]
                dfs(nxt, steps + [(state, fas, act, nxt)])
                if len(out) >= max_children:
                    return
        elif isinstance(state, ReactionStateC):
            for idx in fas.get_possible_actions_indices():
                act = fas.get_action_at_idx(idx)
                nxt = env.apply_forward_actions([state], [act])[0]  # -> A(x, k+1)
                out.append(steps + [(state, fas, act, nxt)])
                if len(out) >= max_children:
                    return

    dfs(hub_state, [])
    return out


def _build_child_trajectory(env, hub_state, path) -> Optional[Trajectories]:
    """Extend a product path with the stop micro-step and build a single-trajectory
    ``Trajectories`` carrying the env's forward+backward action spaces (as the sampler records)."""
    x_state = path[-1][3]  # A(x, k+1)
    stop_fas = env.get_forward_action_spaces([x_state])[0]
    if not hasattr(stop_fas, "get_possible_actions_indices"):
        return None
    stop_act = None
    for idx in stop_fas.get_possible_actions_indices():
        a = stop_fas.get_action_at_idx(idx)
        if _is_stop(a):
            stop_act = a
            break
    if stop_act is None:
        return None
    terminal = env.apply_forward_actions([x_state], [stop_act])[0]
    if not isinstance(terminal, ReactionStateTerminal):
        return None
    full = path + [(x_state, stop_fas, stop_act, terminal)]
    traj: Trajectories = Trajectories()
    traj.add_source_states([hub_state])
    for _state, fas, act, nxt in full:
        bas = env.get_backward_action_spaces([nxt])[0]
        # P_B is only recoverable if the child's backward action space actually contains the
        # disconnection back to this parent. At the max-reaction boundary the env sometimes
        # can't invert the reaction (a decomposability-depth limit) and returns a degenerate
        # space — scoring the commit against it would silently yield P_B=1 (wrong). Drop the
        # child instead, so an enumerated FlowRecord never carries a fabricated P_B.
        if isinstance(act, ReactionActionC):
            possible = getattr(bas, "possible_actions", None)
            if possible is None or act not in possible:
                return None
        traj.add_actions_states([act], [nxt], [fas], [bas], not_terminated_mask=None)
    return traj


def _add(timing, key, dt):
    if timing is not None:
        timing[key] = timing.get(key, 0.0) + dt


def _extract(
    objective,
    reward,
    trajs: List[Trajectories],
    strip_stereo: bool,
    timing=None,
    sync=None,
    gate_component: Optional[str] = None,
) -> List[FlowRecord]:
    """Score + flow-extract a chunk of enumerated child trajectories.

    When ``timing`` (a dict accumulator) is passed, the three §Logs/039 components are timed
    separately: ``enumeration_s`` (the ``Trajectories`` assembly book-keeping), ``reward_gen_s``
    (the reward-generator ``M`` scoring the children — the axis that dominates for docking), and
    ``flow_extract_s`` (``assign_log_probs`` → P_F/P_B, the hub-ranking / ``U(h)`` signal). ``sync``
    (e.g. ``torch.cuda.synchronize``) is called at GPU-timing boundaries so async CUDA work is
    charged to the right component. Both default off → behaviour unchanged."""
    _sync = sync or (lambda: None)
    _t0 = time.perf_counter()
    big = Trajectories.from_trajectories(trajs) if len(trajs) > 1 else trajs[0]
    terminals = big.get_last_states_flat()
    _add(timing, "enumeration_s", time.perf_counter() - _t0)
    _sync()
    _r0 = time.perf_counter()
    reward_output = reward.compute_reward_output(terminals)
    _sync()
    _add(timing, "reward_gen_s", time.perf_counter() - _r0)
    big.set_reward_outputs(reward_output)
    _f0 = time.perf_counter()
    records, _visits, _n = extract_flow_records(
        objective, big, strip_stereo=strip_stereo, gate_component=gate_component
    )
    _sync()
    _add(timing, "flow_extract_s", time.perf_counter() - _f0)
    return records


@torch.no_grad()
def enumerate_terminal_children(
    env,
    objective,
    reward,
    hub_state: ReactionStateA,
    *,
    max_children: int = 2000,
    chunk_size: int = 64,
    strip_stereo: bool = True,
    timing=None,
    sync=None,
    gate_component: Optional[str] = None,
) -> Tuple[List[FlowRecord], int]:
    """Enumerate a hub's one-reaction terminal children as ``FlowRecord``s.

    Args:
        env: the reaction ``ReactionEnv`` (forward, not reversed).
        objective: the trained ``ObjectiveBase`` (for ``assign_log_probs`` via extraction).
        reward: the ``Reward`` whose proxy scores the enumerated terminals (free for sEH).
        hub_state: the hub ``ReactionStateA(mol, k)`` to enumerate from.
        max_children: cap on enumerated children (docking budget guard, §6).
        chunk_size: children per policy batch; a chunk that fails ``assign_log_probs`` (e.g. a
            backward disconnection the policy can't score) is retried per-child, skipping the bad
            ones — so one pathological product never voids the whole hub.
        timing: optional dict accumulator for the §Logs/039 per-component wall-clock
            (``enumeration_s`` / ``reward_gen_s`` / ``flow_extract_s``); ``None`` → no timing.
        sync: optional no-arg callable (e.g. ``torch.cuda.synchronize``) fired at GPU-timing
            boundaries so async CUDA work is attributed to the right component.

    Returns:
        ``(records, n_enumerated_paths)``. ``len(records) <= n_enumerated_paths`` when some
        products fail to build a scorable trajectory.
    """
    _t0 = time.perf_counter()
    paths = _enumerate_product_paths(env, hub_state, max_children)
    trajs = [
        t for t in (_build_child_trajectory(env, hub_state, p) for p in paths) if t is not None
    ]
    _add(timing, "enumeration_s", time.perf_counter() - _t0)  # RDKit child construction
    if not trajs:
        return [], len(paths)
    records: List[FlowRecord] = []
    for i in range(0, len(trajs), chunk_size):
        chunk = trajs[i : i + chunk_size]
        try:
            records.extend(
                _extract(
                    objective,
                    reward,
                    chunk,
                    strip_stereo,
                    timing=timing,
                    sync=sync,
                    gate_component=gate_component,
                )
            )
        except Exception:  # noqa: BLE001 - isolate a bad product, keep the rest of the hub
            for t in chunk:
                try:
                    records.extend(
                        _extract(
                            objective,
                            reward,
                            [t],
                            strip_stereo,
                            timing=timing,
                            sync=sync,
                            gate_component=gate_component,
                        )
                    )
                except Exception:  # noqa: BLE001
                    continue
    return records, len(paths)


def hub_state_from_smiles(smiles: str, depth: int) -> Optional[ReactionStateA]:
    """Rebuild a hub ``ReactionStateA`` from a SMILES + observed depth (the §4b string contract).

    Uses the stereo-aware SMILES so the reconstructed molecule matches what the model saw.
    Returns ``None`` if the SMILES doesn't parse to a valid molecule.
    """
    mol = Molecule(smiles)
    if not mol.valid:
        return None
    return ReactionStateA(molecule=mol, num_reactions=int(depth))
