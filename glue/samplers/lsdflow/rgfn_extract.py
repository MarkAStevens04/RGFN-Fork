"""RGFN-native flow extraction: a sampled ``Trajectories`` -> ``FlowRecord``s.

This is the one place the ``docs/LSD_FLOW_PROPOSAL.md`` §2 log-terms are read out of the
real RGFN objects, and it is shared by both project axes: the production AL path calls it on the
trajectories the loop already sampled, and the validation RGFN adapter
(``validation/lsdflow/adapters/rgfn_adapter.py``) calls it to emit canonical records for the
cross-model DAG. It lives in ``glue/`` because it imports ``rgfn`` (allowed) and must be
callable from the single-env loop without any ``validation/`` dependency.

The mapping (from the RGFN micro-step state machine — see
``rgfn/gfns/reaction_gfn/api/reaction_api.py`` and ``.../policies/reaction_*_policy.py``):

    ... StateA(mol=h, k=n-1) --A:template--> [StateB.. --B:reactant-->]* StateC
        --C:commit--> StateA(mol=x, k=n) --A:stop--> Terminal(x)
                      ^^^^^^^^^^^^^^^^^^                ^^^^^^^^^
                      one molecule->molecule move       the stop micro-step

Per §2, for the terminal product ``x`` reached by one reaction from hub ``h``:
  * ``log P_F(x | h)`` = sum of that last reaction's micro-step **forward** log-probs
    (A-template + B-reactant(s) + C-commit) — the composed move (§4b);
  * ``log P_F(stop | x)`` = the forward log-prob of the final ``StateA(x) -> Terminal`` stop
    micro-step (the very last action of the trajectory) — the terminating factor, **never
    dropped** (§2);
  * ``log P_B(h | x)`` = sum of that last reaction's micro-step **backward** log-probs. In
    RGFN only the reverse-C step is learned; reversing the template / reactant micro-steps is
    deterministic (log-prob 0), so this equals the learned C-disconnection log-prob (§5);
  * ``R(x)`` / ``log R(x)`` = the proxy value / shaped log reward attached to the trajectory.

Only trajectories that end in a genuine ``ReactionStateTerminal`` contribute (early-terminal
/ invalid endings are skipped). A terminal always has ``num_reactions >= 1`` (RGFN forbids
stopping at ``k=0``), so the hub ``StateA(k=n-1)`` always exists — it may be the initial
building block (``k=0``, the cheapest possible hub).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from glue.samplers.lsdflow.records import FlowRecord
from rgfn.gfns.reaction_gfn.api.reaction_api import (
    ReactionStateA,
    ReactionStateTerminal,
)

try:  # RDKit is present in the rgfn env; degrade gracefully if a caller lacks it.
    from rdkit import Chem
except Exception:  # pragma: no cover - RDKit always available in-env
    Chem = None


def _stripped_key(molecule) -> Tuple[str, str]:
    """Return ``(cross_model_key, stereo_key)`` for a molecule (§6).

    Cross-model key = RDKit canonical SMILES with stereo stripped (so a FragGFN skeleton
    matches the same reaction-model skeleton); stereo key = the model's own canonical
    (stereo-aware) SMILES. Falls back to the stored SMILES if RDKit round-trip fails.
    """
    stereo_key = molecule.smiles
    if Chem is None:
        return stereo_key, stereo_key
    try:
        mol = molecule.rdkit_mol
        if mol is None:
            return stereo_key, stereo_key
        return Chem.MolToSmiles(mol, isomericSmiles=False), stereo_key
    except Exception:
        return stereo_key, stereo_key


@torch.no_grad()
def extract_flow_records(
    objective,
    trajectories,
    *,
    strip_stereo: bool = True,
) -> Tuple[List[FlowRecord], Dict[str, int], int]:
    """Extract §2 terminal-transition flow records from a batch of trajectories.

    Args:
        objective: a built RGFN ``ObjectiveBase`` whose forward/backward policies carry the
            **trained** weights (P_B must be the trained policy — §5). Used only for
            ``assign_log_probs`` (read-only).
        trajectories: a sampled ``Trajectories`` with rewards attached (as the sampler
            produces them). Modified in place by ``assign_log_probs``.
        strip_stereo: use the stereo-stripped cross-model key as the primary node key
            (§6). Keep ``True`` for cross-model aggregation.

    Returns:
        ``(records, visit_counts, n_trajectories)`` where ``visit_counts`` maps each
        molecule (``StateA``) node key to the number of trajectories passing through it —
        the reward-free visitation estimator's raw data (§2).
    """
    objective.assign_log_probs(trajectories)
    fwd = trajectories.get_forward_log_probs_flat().detach().cpu().tolist()
    bwd = trajectories.get_backward_log_probs_flat().detach().cpu().tolist()

    reward_outputs = trajectories.get_reward_outputs()
    log_rewards = reward_outputs.log_reward.detach().cpu().tolist()
    proxies = reward_outputs.proxy.detach().cpu().tolist()

    states_list = trajectories._states_list
    actions_list = trajectories._actions_list

    records: List[FlowRecord] = []
    visit_counts: Dict[str, int] = {}
    n_trajectories = len(states_list)

    offset = 0
    for t in range(n_trajectories):
        states = states_list[t]
        actions = actions_list[t]
        n_actions = len(actions)
        traj_fwd = fwd[offset : offset + n_actions]
        traj_bwd = bwd[offset : offset + n_actions]
        offset += n_actions

        # Count trajectory visits to each molecule node (dedup within-trajectory).
        seen_nodes = set()
        for s in states:
            if isinstance(s, ReactionStateA):
                key, _ = _stripped_key(s.molecule) if strip_stereo else (s.molecule.smiles, "")
                seen_nodes.add(key)
        for key in seen_nodes:
            visit_counts[key] = visit_counts.get(key, 0) + 1

        # Only genuine terminals contribute a flow record.
        if not states or not isinstance(states[-1], ReactionStateTerminal):
            continue
        # s[-1] = Terminal ; s[-2] = StateA(mol=x, k=n) ; the stop micro-step is action -1.
        idx_stop = n_actions - 1
        state_x = states[-2]
        if not isinstance(state_x, ReactionStateA):
            continue  # defensive: a well-formed terminal is always reached from a StateA

        # Walk back to the hub: the StateA immediately preceding StateA(x).
        p = None
        for j in range(len(states) - 3, -1, -1):
            if isinstance(states[j], ReactionStateA):
                p = j
                break
        if p is None:
            continue  # no preceding molecule state (should not happen for k>=1 terminals)
        state_h = states[p]

        # Compose the last reaction (actions p .. idx_stop-1) and the stop factor (idx_stop).
        log_pf_move = float(sum(traj_fwd[p:idx_stop]))
        log_pb_move = float(sum(traj_bwd[p:idx_stop]))
        log_pf_stop = float(traj_fwd[idx_stop])

        if strip_stereo:
            hub_key, hub_stereo = _stripped_key(state_h.molecule)
            child_key, child_stereo = _stripped_key(state_x.molecule)
        else:
            hub_key = hub_stereo = state_h.molecule.smiles
            child_key = child_stereo = state_x.molecule.smiles

        records.append(
            FlowRecord(
                hub_key=hub_key,
                child_key=child_key,
                reward=float(proxies[t]),
                log_reward=float(log_rewards[t]),
                log_pf_move=log_pf_move,
                log_pb_move=log_pb_move,
                log_pf_stop=log_pf_stop,
                hub_depth=int(state_h.num_reactions),
                hub_stereo_key=hub_stereo,
                child_stereo_key=child_stereo,
            )
        )

    return records, visit_counts, n_trajectories
