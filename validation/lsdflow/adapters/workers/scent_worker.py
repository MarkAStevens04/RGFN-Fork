#!/usr/bin/env python
"""SCENT per-env worker — sampled flow extraction + hub-child enumeration (``LSD_FLOW_PROPOSAL`` §4b).

SCENT lives in its own ``scent`` conda env and its package is **also** named ``rgfn`` (it is an
RGFN fork), so it can never co-import with our ``rgfn``/``glue`` in one process (§4b). This worker
runs standalone *inside the scent env* (the in-process :class:`SCENTAdapter` client shells to it,
the ``scripts/score_batch.py`` bridge shape) and exchanges results over files.

Two modes:
  * ``--mode sample`` — sample trajectories, extract the §2 terminal-transition flow records, and
    write ``records.csv`` + ``visit_counts.json`` + ``routes.json`` + ``meta.json`` (the canonical
    :class:`glue.samplers.lsdflow.records.FlowRecord` schema the harness reads back). ``routes.json``
    maps every product SMILES -> its full min-reaction synthesis route (ordered steps: reaction id +
    reactants + input/product), so any hub or terminal is reconstructable step-by-step.
  * ``--mode enumerate`` — for each hub in ``--hubs-file`` (``smiles,depth`` rows), exhaustively
    enumerate its one-reaction terminal children and write ``enumerated_records.csv`` +
    ``enum_per_hub.json`` + ``enum_children.json`` (mirrors :class:`RGFNAdapter.enumerate_hub_children`).
    Each child carries ``reaction`` = the full hub->child final step(s), so the diversifying reaction
    (not just the fragment added) is recorded. Full route of a hub-batching hit = the hub's route
    (from ``routes.json``) + this final reaction; expand any promoted reactant via its recipe in
    ``fragments_<N>.json`` (recipe_logging).

**Dynamic-library freeze (§4b, §9).** SCENT promotes high-reward intermediates into its fragment
vocabulary during training; building from the checkpoint alone leaves the model restricted to the
418 base fragments (``current_fragments=418``). To analyze the *full trained* SCENT — and for the
coincidence study (§8) — we freeze the library to its final promoted-fragment snapshot
(``additional_fragments/fragments_<N>.json``: ordered ``chosen_smiles`` + costs + min-reactions) by
firing ``trainer.on_update_fragments_library`` exactly as training does (grows the env reactant
set, the policy's fragment embedding, and the cost-guided backward policy's cost table together).
Applied to **both** modes (our "faithful full SCENT" substrate choice). ``--no-freeze`` reverts to
the 418-base substrate.

Build recipe = ``validation/generators/scent/verify_pb_recovery.py`` (proven login-safe): chdir into
the SCENT clone, parse the gin config, build the ``trainer`` (the freeze root, wires objective +
valid_sampler + cost proxy), load ``last_gfn.pt`` (forward policy + logZ) **and** the
``guidance_models.pt`` sidecar so the backward policy P_B is the model's *trained* one (§5, entry
024). Sampling uses the pure-policy ``valid_sampler`` (not the exploratory training sampler) so the
sampled distribution matches the P_F ``assign_log_probs`` scores against — a clean ``U(h)`` (§5).

Flow extraction + enumeration mirror ``glue.samplers.lsdflow.rgfn_extract`` /
``glue.samplers.lsdflow.rgfn_enumerate`` (can't import them here — their ``import rgfn`` would
resolve to *our* rgfn; SCENT's fork exposes the same API so the algorithms transfer verbatim).
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# This worker predates _artifacts and inlines its own writers, so it imports only what it needs:
# _artifacts for the shared PartialFlusher, _docking for the socket guard. Both are stdlib-only and
# safe to import from the scent env.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A  # noqa: E402
import _routes  # noqa: E402
import _docking  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
SCENT_ROOT = REPO_ROOT / "external" / "scent"
GEN_DIR = REPO_ROOT / "validation" / "generators" / "scent"

# Reward orientation fallback by target when the proxy doesn't advertise it (surrogates are
# higher-is-better; docking ΔVina is lower-is-better). The proxy's own flag wins when present.
_HIGHER_IS_BETTER_BY_REWARD = {"seh": True, "drd2": True, "clpp": False, "6td3": False}

_REC_COLS = [
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


def _parse_args():
    p = argparse.ArgumentParser(description="SCENT LSD-Flow worker (sample | enumerate).")
    p.add_argument("--mode", choices=["sample", "enumerate", "probe_hubs"], default="sample")
    p.add_argument(
        "--config",
        required=True,
        help="SCENT gin config (e.g. validation/configs/scent_seh_fixed.gin)",
    )
    p.add_argument("--checkpoint", required=True, help="trained SCENT last_gfn.pt")
    p.add_argument(
        "--guidance",
        default="",
        help="guidance_models.pt sidecar (default: sibling of --checkpoint)",
    )
    p.add_argument(
        "--freeze-snapshot",
        default="",
        help="dynamic-library snapshot fragments_<N>.json to freeze to (default: the highest-N "
        "snapshot in <checkpoint-run>/additional_fragments). --no-freeze disables.",
    )
    p.add_argument(
        "--no-freeze", dest="freeze", action="store_false", help="stay on the 418 base library"
    )
    p.set_defaults(freeze=True)
    p.add_argument("--n-trajectories", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--reward-name", default="seh")
    p.add_argument("--model-name", default="scent")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--run-dir", default="/tmp/lsdflow_scent", help="scratch user_root_dir for gin")
    p.add_argument("--out-dir", required=True, help="where to write outputs")
    p.add_argument("--no-strip-stereo", dest="strip_stereo", action="store_false")
    p.set_defaults(strip_stereo=True)
    # enumerate mode
    p.add_argument("--hubs-file", default="", help="enumerate mode: CSV with 'smiles,depth' rows")
    p.add_argument("--enum-max-children", type=int, default=4000, help="per-hub enumeration cap")
    p.add_argument(
        "--count-only",
        action="store_true",
        help="enumerate mode: COUNT each hub's children and stop -- no reward/docking, no flow "
        "extraction. Sizes a re-enumeration before paying for it: docking is 97%% of a SCENT "
        "docking cell's cost, so the true child count of a capped hub is otherwise unknowable "
        "without buying the whole thing. Writes enum_counts.json.",
    )
    return p.parse_args()


# ----------------------------------------------------------------- flow extraction (vendored)
def _reaction_id(reaction) -> str:
    """Readable, stable id for an AnchoredReaction (name/smarts if available, else repr).
    Mirror of ``validation/generators/scent/recipe_logging.py:_reaction_id`` (kept self-contained —
    the worker can't import our-rgfn-bound modules)."""
    if reaction is None:
        return ""
    for attr in ("name", "reaction_name", "smarts"):
        v = getattr(reaction, attr, None)
        if v:
            return str(v)
    inner = getattr(reaction, "reaction", None)
    if inner is not None:
        for attr in ("name", "smarts"):
            v = getattr(inner, attr, None)
            if v:
                return str(v)
    return str(reaction)


def _reaction_step(act):
    """The full synthesis step for a ``ReactionActionC`` (schema shared with recipe_logging +
    fragments_<N>.json routes): reaction id + reactant fragments + input & product molecule."""
    return {
        "reaction": _reaction_id(getattr(act, "input_reaction", None)),
        "reactants": [f.smiles for f in getattr(act, "input_fragments", ()) or ()],
        "input": getattr(getattr(act, "input_molecule", None), "smiles", None),
        "product": getattr(getattr(act, "output_molecule", None), "smiles", None),
    }


def _stripped_key(Chem, molecule):
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


def extract_flow_records(
    objective,
    trajectories,
    RSA,
    RST,
    RSC,
    Chem,
    strip_stereo,
    chosen_set=None,
    RAC=None,
    routes_out=None,
    gate_component=None,
):
    """§2 terminal-transition flow records from a batch of trajectories -> list of dict rows.

    Also builds per-molecule dynamic-fragment ``compositions`` (Logs/028): for each molecule node,
    the promoted fragments (in ``chosen_set``) used to build it — its seed (first fragment) plus the
    reactant fragments attached at each reaction (``ReactionStateC.fragments``). Intermediate
    molecules that merely *pass through* a promoted-fragment structure are NOT counted (only attached
    building blocks are). ``num_reactions`` is the base-assembly step count (a promoted attach = 1
    step). Empty ``chosen_set`` (no freeze / RGFN) -> all compositions empty."""
    import torch

    chosen_set = chosen_set or set()

    with torch.no_grad():
        objective.assign_log_probs(trajectories)
        fwd = trajectories.get_forward_log_probs_flat().detach().cpu().tolist()
        bwd = trajectories.get_backward_log_probs_flat().detach().cpu().tolist()
        ro = trajectories.get_reward_outputs()
        log_rewards = ro.log_reward.detach().cpu().tolist()
        proxies = ro.proxy.detach().cpu().tolist()
        if gate_component:
            # DOCKING only. The proxy value a docking model trains on is ReLU(-raw/norm), never
            # negative, while the calibrated bars are RAW energies (-8.0 ClpP). Recording the proxy
            # value in `reward` would gate a never-negative column against a negative bar and
            # qualify NOTHING -- an empty result, not a crash. log_reward is left alone so the flow
            # terms still match training. Mirrors glue/samplers/lsdflow/rgfn_extract.
            comps = getattr(ro, "proxy_components", None) or {}
            if gate_component not in comps:
                raise KeyError(
                    f"gate_component={gate_component!r} not in proxy_components {sorted(comps)}; "
                    "DockingBridgeProxy must emit 'raw_score'. Refusing to fall back to the proxy "
                    "value, which would silently qualify nothing."
                )
            proxies = comps[gate_component].detach().cpu().tolist()

    states_list = trajectories._states_list
    actions_list = trajectories._actions_list
    records, visit_counts, compositions = [], {}, {}
    offset = 0
    for t in range(len(states_list)):
        states, actions = states_list[t], actions_list[t]
        n_actions = len(actions)
        traj_fwd = fwd[offset : offset + n_actions]
        traj_bwd = bwd[offset : offset + n_actions]
        offset += n_actions

        # Full synthesis route per product molecule (sample mode only; keyed by product SMILES,
        # min-reaction — matches the promoted-fragment recipes in fragments_<N>.json). This makes
        # every hub AND every terminal reconstructable step-by-step for a chemist.
        if routes_out is not None and RAC is not None:
            steps = []
            for act in actions:
                if not isinstance(act, RAC):
                    continue
                steps.append(_reaction_step(act))
                product = steps[-1]["product"]
                if not product:
                    continue
                # Key by the SAME convention as records.csv/enum (stripped when strip_stereo) so the
                # hub_key/child_key -> route join is exact; the steps keep raw (stereo) SMILES.
                key = product
                if strip_stereo and Chem is not None:
                    m = Chem.MolFromSmiles(product)
                    if m is not None:
                        key = Chem.MolToSmiles(m, isomericSmiles=False)
                stored = routes_out.get(key)
                if stored is None or len(steps) < stored["num_reactions"]:
                    routes_out[key] = {
                        "seed": steps[0]["input"],
                        "num_reactions": len(steps),
                        "steps": list(steps),
                    }

        seen = set()
        for s in states:
            if isinstance(s, RSA):
                key, _ = (
                    _stripped_key(Chem, s.molecule) if strip_stereo else (s.molecule.smiles, "")
                )
                seen.add(key)
        for key in seen:
            visit_counts[key] = visit_counts.get(key, 0) + 1

        # Composition: accumulate attached promoted fragments (seed + each reaction's reactants).
        running: set = set()
        seed_done = False
        for s in states:
            if isinstance(s, RSA):
                if not seed_done:
                    if s.molecule.smiles in chosen_set:
                        running.add(s.molecule.smiles)
                    seed_done = True
                ckey, _ = (
                    _stripped_key(Chem, s.molecule) if strip_stereo else (s.molecule.smiles, "")
                )
                prev = compositions.get(ckey)
                if prev is None or int(s.num_reactions) < prev["num_reactions"]:
                    compositions[ckey] = {
                        "promoted": sorted(running),
                        "num_reactions": int(s.num_reactions),
                    }
            elif isinstance(s, RSC):
                for f in getattr(s, "fragments", ()) or ():
                    if f.smiles in chosen_set:
                        running.add(f.smiles)

        if not states or not isinstance(states[-1], RST):
            continue
        idx_stop = n_actions - 1
        state_x = states[-2]
        if not isinstance(state_x, RSA):
            continue
        p = None
        for j in range(len(states) - 3, -1, -1):
            if isinstance(states[j], RSA):
                p = j
                break
        if p is None:
            continue
        state_h = states[p]
        log_pf_move = float(sum(traj_fwd[p:idx_stop]))
        log_pb_move = float(sum(traj_bwd[p:idx_stop]))
        log_pf_stop = float(traj_fwd[idx_stop])
        # Z-anchored half of the SAME trajectory: source -> hub. Chaining detailed balance forward
        # from F(s_0)=Z gives log F_prefix(h) = logZ + sum_{t<=p}[logP_F - logP_B], an exact second
        # estimate of F(h) sharing no terms with the R-anchored one above. Their product is exactly
        # trajectory balance, so the log-difference IS the per-trajectory TB residual (logZ is in
        # meta.json). Written to the prefix_terms.csv sidecar, not into _REC_COLS.
        log_pf_prefix = float(sum(traj_fwd[:p]))
        log_pb_prefix = float(sum(traj_bwd[:p]))
        # Z-anchored half: source -> hub (steps before the hub). See
        # _artifacts.write_prefix_terms — log F_prefix(h) = logZ + log_pf_prefix - log_pb_prefix,
        # an exact second estimate of F(h) sharing no terms with the R-anchored one above.
        log_pf_prefix = float(sum(traj_fwd[:p]))
        log_pb_prefix = float(sum(traj_bwd[:p]))
        if strip_stereo:
            hub_key, hub_stereo = _stripped_key(Chem, state_h.molecule)
            child_key, child_stereo = _stripped_key(Chem, state_x.molecule)
        else:
            hub_key = hub_stereo = state_h.molecule.smiles
            child_key = child_stereo = state_x.molecule.smiles
        records.append(
            {
                "hub_key": hub_key,
                "child_key": child_key,
                "reward": float(proxies[t]),
                "log_reward": float(log_rewards[t]),
                "log_pf_move": log_pf_move,
                "log_pb_move": log_pb_move,
                "log_pf_prefix": log_pf_prefix,
                "log_pb_prefix": log_pb_prefix,
                "log_pf_prefix": log_pf_prefix,
                "log_pb_prefix": log_pb_prefix,
                "log_pf_stop": log_pf_stop,
                "hub_depth": int(state_h.num_reactions),
                "hub_stereo_key": hub_stereo,
                "child_stereo_key": child_stereo,
            }
        )
    return records, visit_counts, compositions, len(states_list)


# ----------------------------------------------------------------- enumeration (vendored from rgfn_enumerate)
def _make_hub_prober(Trajectories, RST, torch):
    """Measure R(h) and P_F(stop|h) AT each hub — the two quantities the dimensionless
    flow-conservation test needs (``experiments/lsd_hubs/matrix16/hub_flow_estimators.py``).

    Eliminating F(h) from ``F(h) = R(h) + N(h)`` and ``R(h) = F(h) P_F(stop|h)`` gives
    ``R(h)/(R(h)+N(h)) == P_F(stop|h)`` — no near-1 cancellation, and independent of the
    enumeration normalizer. SCENT is the informative case for it: unlike RxnFlow (heuristic P_B,
    S(h)~=0.999 so the test is near-tautological) SCENT has a TRAINED P_B and real stop-mass
    (S(h) 5th percentile ~0.40), so the identity has power here.

    One single-step trajectory per hub, ``StateA(h,k) -[stop]-> Terminal(h)``, scored by the same
    ``objective.assign_log_probs`` the flow extraction uses. Returns a list aligned with
    ``hub_states``; ``None`` where the policy cannot terminate at that hub."""

    def _is_stop(action):
        return getattr(action, "anchored_reaction", None) is None

    def build_hub_stop_trajectory(env, hub_state):
        fas = env.get_forward_action_spaces([hub_state])[0]
        if not hasattr(fas, "get_possible_actions_indices"):
            return None
        stop_act = None
        for idx in fas.get_possible_actions_indices():
            a = fas.get_action_at_idx(idx)
            if _is_stop(a):
                stop_act = a
                break
        if stop_act is None:
            return None  # the env forbids terminating here (e.g. below min reactions)
        terminal = env.apply_forward_actions([hub_state], [stop_act])[0]
        if not isinstance(terminal, RST):
            return None
        traj = Trajectories()
        traj.add_source_states([hub_state])
        bas = env.get_backward_action_spaces([terminal])[0]
        traj.add_actions_states([stop_act], [terminal], [fas], [bas], not_terminated_mask=None)
        return traj

    def probe(env, objective, reward, hub_states, chunk=32):
        out = [None] * len(hub_states)
        idx_ok, trajs = [], []
        for i, hs in enumerate(hub_states):
            t = build_hub_stop_trajectory(env, hs) if hs is not None else None
            if t is not None:
                idx_ok.append(i)
                trajs.append(t)
        with torch.no_grad():
            for st in range(0, len(trajs), chunk):
                blk, blk_idx = trajs[st : st + chunk], idx_ok[st : st + chunk]
                big = Trajectories.from_trajectories(blk) if len(blk) > 1 else blk[0]
                terminals = big.get_last_states_flat()
                big.set_reward_outputs(reward.compute_reward_output(terminals))
                objective.assign_log_probs(big)
                # exactly ONE action per trajectory -> fwd is 1:1 with the block
                fwd = big.get_forward_log_probs_flat().detach().cpu().tolist()
                ro = big.get_reward_outputs()
                proxy = ro.proxy.detach().cpu().tolist()
                log_r = ro.log_reward.detach().cpu().tolist()
                for j, i in enumerate(blk_idx):
                    out[i] = {
                        "allow_stop": True,
                        "log_pf_stop_h": float(fwd[j]),
                        "reward_h": float(proxy[j]),
                        "log_reward_h": float(log_r[j]),
                    }
        return out

    return probe


def _make_enumerator(rgfn_api, Trajectories, RSA, RSB, RSC, RST, RAC, Molecule):
    """Build the enumeration closures bound to SCENT's fork classes (mirror of rgfn_enumerate)."""

    def _is_stop(action):
        return getattr(action, "anchored_reaction", None) is None

    def enumerate_product_paths(env, hub_state, max_children):
        out = []

        def dfs(state, steps):
            if len(out) >= max_children:
                return
            # Only mid-trajectory states (A/B/C) expand. A STOP or EARLY-TERMINATE action can land on
            # a terminal / early-terminal state (RST / ReactionStateEarlyTerminal) that has NO forward
            # action space -> get_forward_action_spaces raises KeyError (crashed the full 200-hub
            # enum: some hub's DFS reached an early-terminal before any hub was written). Those are
            # dead-ends for enumeration anyway: valid one-reaction children are collected at RSC below,
            # and build_child_trajectory only keeps paths that STOP to a proper RST. So skip them.
            # (SCENT's env raises; RGFN returns a degenerate space instead -- hence the guard is needed
            # here specifically. Both branches hit this independently; see matrix16 MERGE_NOTES §1.)
            if not isinstance(state, (RSA, RSB, RSC)):
                return
            fas = env.get_forward_action_spaces([state])[0]
            if not hasattr(fas, "get_possible_actions_indices"):
                return
            if isinstance(state, RSA):
                for idx in fas.get_possible_actions_indices():
                    act = fas.get_action_at_idx(idx)
                    if _is_stop(act):
                        continue
                    nxt = env.apply_forward_actions([state], [act])[0]
                    dfs(nxt, steps + [(state, fas, act, nxt)])
                    if len(out) >= max_children:
                        return
            elif isinstance(state, RSB):
                for idx in fas.get_possible_actions_indices():
                    act = fas.get_action_at_idx(idx)
                    nxt = env.apply_forward_actions([state], [act])[0]
                    dfs(nxt, steps + [(state, fas, act, nxt)])
                    if len(out) >= max_children:
                        return
            elif isinstance(state, RSC):
                for idx in fas.get_possible_actions_indices():
                    act = fas.get_action_at_idx(idx)
                    nxt = env.apply_forward_actions([state], [act])[0]
                    out.append(steps + [(state, fas, act, nxt)])
                    if len(out) >= max_children:
                        return

        dfs(hub_state, [])
        return out

    def build_child_trajectory(env, hub_state, path):
        x_state = path[-1][3]
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
        if not isinstance(terminal, RST):
            return None
        full = path + [(x_state, stop_fas, stop_act, terminal)]
        traj = Trajectories()
        traj.add_source_states([hub_state])
        added_fragments = []  # the reactant(s) attached in this one diversifying reaction
        reaction_steps = []  # the full hub->child reaction step(s) (schema shared w/ recipe routes)
        for _s, fas, act, nxt in full:
            bas = env.get_backward_action_spaces([nxt])[0]
            if isinstance(act, RAC):
                possible = getattr(bas, "possible_actions", None)
                if possible is None or act not in possible:
                    return None  # P_B not invertible -> never fabricate P_B=1
                added_fragments = [f.smiles for f in getattr(act, "input_fragments", ())]
                # Build the step from the STATES (hub in, child out) — in the enumerate path the
                # action's output_molecule may not be populated pre-apply (unlike the post-hoc
                # trajectory recipe_logging reads); the states always are. Product == child_stereo_key.
                reaction_steps.append(
                    {
                        "reaction": _reaction_id(getattr(act, "input_reaction", None)),
                        "reactants": [f.smiles for f in getattr(act, "input_fragments", ()) or ()],
                        "input": getattr(getattr(hub_state, "molecule", None), "smiles", None),
                        "product": getattr(getattr(x_state, "molecule", None), "smiles", None),
                    }
                )
            traj.add_actions_states([act], [nxt], [fas], [bas], not_terminated_mask=None)
        return traj, x_state.molecule.smiles, added_fragments, reaction_steps

    def enumerate_terminal_children(
        env, objective, reward, hub_state, extract, max_children, chunk_size=64, sync=None
    ):
        """Returns ``(records, n_paths, added_by_stereo, reaction_by_stereo, timing)``.

        ``added_by_stereo`` maps each child's stereo SMILES -> the fragment(s) attached in its final
        reaction (cost / composition), ``reaction_by_stereo`` -> the full final reaction step(s)
        (reconstruction). ``timing`` is the measured per-hub compute-time breakdown (Logs/039):
        ``{enumeration_s, reward_gen_s, flow_extract_s}`` in seconds — the RDKit child construction,
        the reward-generator call, and the P_F/P_B flow-extraction, timed separately so the campaign
        can attribute each strategy's compute to where it actually went. ``sync`` (a no-arg callable,
        e.g. ``torch.cuda.synchronize``) is called at each GPU-timing boundary so async CUDA work is
        charged to the right component; pass ``None`` on CPU."""
        _sync = sync or (lambda: None)
        timing = {"enumeration_s": 0.0, "reward_gen_s": 0.0, "flow_extract_s": 0.0}
        _sync()
        _t0 = time.perf_counter()
        paths = enumerate_product_paths(env, hub_state, max_children)
        built = [
            b for b in (build_child_trajectory(env, hub_state, p) for p in paths) if b is not None
        ]
        timing["enumeration_s"] += time.perf_counter() - _t0
        if not built:
            return [], len(paths), {}, {}, timing
        trajs = [b[0] for b in built]
        added_by_stereo = {b[1]: b[2] for b in built}
        reaction_by_stereo = {b[1]: b[3] for b in built}

        def _extract_batch(chunk):
            # batch prep (Trajectories assembly) is CPU book-keeping -> charge to enumeration
            _p0 = time.perf_counter()
            big = Trajectories.from_trajectories(chunk) if len(chunk) > 1 else chunk[0]
            terminals = big.get_last_states_flat()
            timing["enumeration_s"] += time.perf_counter() - _p0
            # reward generation (proxy / docking oracle) — the axis that gets big for docking
            _sync()
            _r0 = time.perf_counter()
            reward_output = reward.compute_reward_output(terminals)
            _sync()
            timing["reward_gen_s"] += time.perf_counter() - _r0
            big.set_reward_outputs(reward_output)
            # flow extraction: assign_log_probs -> P_F / P_B (the hub-ranking / U(h) signal)
            _r1 = time.perf_counter()
            recs, _v, _c, _n = extract(objective, big)
            _sync()
            timing["flow_extract_s"] += time.perf_counter() - _r1
            return recs

        records = []
        for i in range(0, len(trajs), chunk_size):
            chunk = trajs[i : i + chunk_size]
            try:
                records.extend(_extract_batch(chunk))
            except Exception:
                for t in chunk:
                    try:
                        records.extend(_extract_batch([t]))
                    except Exception:
                        continue
        return records, len(paths), added_by_stereo, reaction_by_stereo, timing

    def hub_state_from_smiles(smiles, depth):
        mol = Molecule(smiles)
        if getattr(mol, "rdkit_mol", None) is None:
            return None
        return RSA(molecule=mol, num_reactions=int(depth))

    def count_children(env, hub_state, max_children):
        """``(n_paths, n_children, seconds)`` for one hub — DFS only, NO reward and NO flow extraction.

        Exists to size a re-enumeration honestly. A hub capped at ``ENUM_MAX`` has an unknown TRUE child
        count, and on a docking cell reward generation is ~97% of the bill, so discovering that count by
        running the real enumeration means buying the whole thing first. This pays only the DFS, which is
        ~0.01 s/child against docking's ~0.95. ``n_children`` applies the same
        ``build_child_trajectory`` filter the real path uses, so it is the count that would actually be
        docked, not the raw path count.
        """
        t0 = time.perf_counter()
        paths = enumerate_product_paths(env, hub_state, max_children)
        n_built = sum(
            1 for p in paths if build_child_trajectory(env, hub_state, p) is not None
        )
        return len(paths), n_built, time.perf_counter() - t0

    return enumerate_terminal_children, hub_state_from_smiles, count_children


# ----------------------------------------------------------------- freeze
def _derive_snapshot(checkpoint: str) -> str:
    """Highest-N fragments_<N>.json in <checkpoint-run>/additional_fragments (the final library)."""
    run_dir = Path(checkpoint).resolve().parents[2]
    frag_dir = run_dir / "additional_fragments"
    snaps = sorted(frag_dir.glob("fragments_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    return str(snaps[-1]) if snaps else ""


def _freeze_library(trainer, env, snapshot_path: str, Molecule):
    """Grow env + policy embedding + cost proxy to the full trained vocabulary (§4b freeze)."""
    snap = json.load(open(snapshot_path))
    chosen = snap["chosen_smiles"]
    mnr = snap["smiles_to_min_num_reactions"]
    costs = snap["chosen_smiles_costs"]
    n_init = len(env.data_factory.get_fragments())
    frags = [Molecule(s, idx=n_init + i, num_reactions=int(mnr[s])) for i, s in enumerate(chosen)]
    # Fire the exact hook training uses (trainer root: env via _setup_fragments, the forward
    # policy's FragmentOneHotEmbedding counter, and the cost-guided backward policy's cost table).
    trainer.on_update_fragments_library(len(chosen), frags, costs)
    print(
        f"[scent_worker] froze library: +{len(frags)} promoted fragments (base {n_init} -> {n_init + len(frags)})",
        flush=True,
    )
    return len(frags)


# --------------------------------------------------------------------------------- build + run
def main():
    _worker_start = time.perf_counter()  # for setup_s (model load + library freeze) in enumerate
    args = _parse_args()
    os.environ.setdefault("WANDB_MODE", "offline")

    config_path = str(
        (REPO_ROOT / args.config).resolve() if not os.path.isabs(args.config) else args.config
    )
    checkpoint = str(Path(args.checkpoint).resolve())
    guidance = args.guidance or str(Path(checkpoint).parent / "guidance_models.pt")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(GEN_DIR))
    sys.path.insert(1, str(SCENT_ROOT))
    os.chdir(SCENT_ROOT)

    import gin
    import torch

    gin.add_config_file_search_path(str(SCENT_ROOT))
    from rgfn.utils.helpers import seed_everything

    seed_everything(args.seed)
    import fixed_reward  # noqa: F401  (registers @ScentFixedRewardRun)

    # Docking configs reference @DockingBridgeProxy, which registers only when its module is
    # imported -- exactly the failure mode CLAUDE.md warns about ("No configurable matching @X" ==
    # the defining module was not on the startup path). run_scent_fixed.py imports it for training;
    # the worker must too, or a docking cell dies in gin parsing before any GPU work.
    try:
        import docking_bridge_proxy  # noqa: F401  (registers @DockingBridgeProxy)
    except ImportError as exc:
        print(
            f"[scent_worker] NOTE docking_bridge_proxy unavailable ({exc}); "
            "surrogate cells are unaffected",
            flush=True,
        )
    from guidance_io import load_guidance_models

    import rgfn  # noqa: F401
    from rgfn.api.trajectories import Trajectories
    from rgfn.gfns.reaction_gfn.api.data_structures import Molecule
    from rgfn.gfns.reaction_gfn.api.reaction_api import (
        ReactionActionC,
        ReactionStateA,
        ReactionStateB,
        ReactionStateC,
        ReactionStateTerminal,
    )
    from rgfn.trainer.trainer import Trainer  # noqa: F401  (registers @Trainer)

    try:
        from rdkit import Chem
    except Exception:
        Chem = None

    _bindings = [
        f'user_root_dir="{run_dir}"',
        'run_name="lsdflow_scent"',
        "Trainer.n_iterations=1",
        f'ScentFixedRewardRun.run_dir="{run_dir}/run"',
        f'ScentFixedRewardRun.repo_root="{REPO_ROOT}"',
        f"ScentFixedRewardRun.seed={args.seed}",
    ]
    if args.reward_name in ("6td3", "clpp"):
        # DockingBridgeProxy defaults its workdir to <repo>/reward_bridge_scent and mkdir()s it in
        # __init__ -- and $HOME is READ-ONLY on compute nodes (Logs/012), so construction dies with
        # PermissionError before we could mutate the attribute. It must therefore be a gin BINDING,
        # applied before the trainer builds. (RxnFlow's twin takes workdir as a constructor arg, so
        # it is fixed differently there -- same bug, two injection points.)
        _bindings.append(f'DockingBridgeProxy.repo_root="{REPO_ROOT}"')
        _bindings.append(f'DockingBridgeProxy.workdir="{run_dir}/reward_bridge_scent"')
        # --count-only never scores anything, so demanding the docking server would defeat its purpose
        # (it exists to size a re-enumeration WITHOUT paying for docking) and would tie a pure-DFS job
        # to a GPU node running a server. The gin bindings above stay: the trainer is still constructed,
        # so the proxy is still instantiated -- it just never gets called.
        if not getattr(args, "count_only", False):
            _docking.require_socket()  # fail now, not after a 40 s oracle construction

    gin.parse_config_files_and_bindings(
        [config_path],
        bindings=_bindings,
        finalize_config=False,
    )

    # Build the trainer (the freeze root: wires objective + valid_sampler + cost proxy together).
    trainer = gin.get_configurable("trainer/gin.singleton")()
    objective = trainer.objective
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    res = objective.load_state_dict(state, strict=False)
    real_missing = [k for k in res.missing_keys if "_cache" not in k]
    print(
        f"[scent_worker] loaded forward policy + logZ (real-missing={len(real_missing)})",
        flush=True,
    )
    if not Path(guidance).exists():
        raise SystemExit(
            f"[scent_worker] guidance sidecar not found: {guidance}\nSCENT's trained P_B is "
            "unrecoverable without it (entry 024). Pass --guidance or use a patched re-run."
        )
    loaded, unmatched = load_guidance_models(objective, guidance, map_location="cpu", strict=True)
    print(
        f"[scent_worker] loaded guidance sidecar: {loaded} keys (unmatched={unmatched})", flush=True
    )

    sampler = trainer.valid_sampler
    env = sampler.env

    # Freeze the dynamic library to the full trained vocabulary (both modes; §4b).
    n_promoted = 0
    chosen_set: set = set()
    if args.freeze:
        snapshot = args.freeze_snapshot or _derive_snapshot(checkpoint)
        if not snapshot or not Path(snapshot).exists():
            raise SystemExit(
                f"[scent_worker] --freeze on but no dynamic-library snapshot found "
                f"({snapshot!r}). Pass --freeze-snapshot or use --no-freeze."
            )
        n_promoted = _freeze_library(trainer, env, snapshot, Molecule)
        chosen_set = set(json.load(open(snapshot)).get("chosen_smiles", []))

    device = (
        ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    )
    for pol in (
        getattr(objective, "forward_policy", None),
        getattr(objective, "backward_policy", None),
    ):
        if pol is not None and hasattr(pol, "set_device"):
            pol.set_device(device)
    if hasattr(sampler, "policy") and hasattr(sampler.policy, "set_device"):
        sampler.policy.set_device(device)
    reward = getattr(sampler, "reward", None)
    proxy = getattr(reward, "proxy", None)
    if proxy is not None and hasattr(proxy, "set_device"):
        try:
            proxy.set_device(device)
        except Exception as exc:
            print(f"[scent_worker] proxy.set_device({device}) skipped: {exc}", flush=True)
    try:
        objective.device = device
    except Exception:
        pass

    higher_is_better = bool(
        getattr(proxy, "higher_is_better", _HIGHER_IS_BETTER_BY_REWARD.get(args.reward_name, True))
    )
    log_z = 0.0
    lz = getattr(objective, "logZ", None)
    if lz is not None:
        try:
            log_z = float(lz.detach().sum().item())
        except Exception:
            log_z = 0.0
    print(
        f"[scent_worker] built on {device}; higher_is_better={higher_is_better}, logZ={log_z:.4f}, frozen={args.freeze}",
        flush=True,
    )

    # Docking targets gate on the RAW energy; surrogates gate on the proxy value itself.
    _gate_component = "raw_score" if args.reward_name in ("6td3", "clpp") else None

    def _extract(obj, traj, routes_out=None):
        return extract_flow_records(
            obj,
            traj,
            ReactionStateA,
            ReactionStateTerminal,
            ReactionStateC,
            Chem,
            args.strip_stereo,
            chosen_set,
            RAC=ReactionActionC,
            routes_out=routes_out,  # sample mode only; enumerate passes None (final rxn captured separately)
            gate_component=_gate_component,
        )

    meta = {
        "model": args.model_name,
        "reward_name": args.reward_name,
        "log_z": log_z,
        "higher_is_better": higher_is_better,
        "checkpoint": checkpoint,
        "guidance": guidance,
        "config": config_path,
        "strip_stereo": bool(args.strip_stereo),
        "frozen": bool(args.freeze),
        "n_promoted_fragments": n_promoted,
    }

    # Compute-time accounting (Logs/039), shared by both modes: charge the one-time setup (import +
    # gin build + ckpt load + library freeze) once, and synchronize CUDA at timing boundaries so the
    # async GPU work (reward-gen vs flow-extract) is charged to the right component.
    _use_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

    def _sync():
        if _use_cuda:
            torch.cuda.synchronize()

    _sync()
    setup_s = time.perf_counter() - _worker_start

    if args.mode == "sample":
        all_records, visit_counts, compositions, total = [], {}, {}, 0
        routes: dict = (
            {}
        )  # product SMILES -> full min-reaction synthesis route (for reconstruction)
        # Stage-1 compute-time (Logs/039): sampling (trajectory generation incl. the reward the
        # sampler computes) vs flow-extraction (assign_log_probs -> P_F/P_B). This is the shared pool
        # best-candidate reuses; it cancels in the hub-vs-best head-to-head but is tracked for the
        # whole-pipeline "where did time go" view.
        sample_timing = {"sampling_s": 0.0, "flow_extract_s": 0.0}
        _it = sampler.get_trajectories_iterator(args.n_trajectories, args.batch_size)
        while True:
            _sync()
            _s0 = time.perf_counter()
            try:
                traj = next(_it)
            except StopIteration:
                break
            _sync()
            sample_timing["sampling_s"] += time.perf_counter() - _s0
            _e0 = time.perf_counter()
            recs, visits, comps, n = _extract(objective, traj, routes_out=routes)
            _sync()
            sample_timing["flow_extract_s"] += time.perf_counter() - _e0
            all_records.extend(recs)
            for k, c in visits.items():
                visit_counts[k] = visit_counts.get(k, 0) + c
            for k, comp in comps.items():  # keep the cheapest (fewest-reaction) synthesis seen
                prev = compositions.get(k)
                if prev is None or comp["num_reactions"] < prev["num_reactions"]:
                    compositions[k] = comp
            total += n
        print(
            f"[scent_worker] sampled {total} trajectories -> {len(all_records)} records, {len(visit_counts)} nodes",
            flush=True,
        )
        _write_records(out_dir / "records.csv", all_records)
        _write_prefix_terms(out_dir / "prefix_terms.csv", all_records)
        json.dump(visit_counts, open(out_dir / "visit_counts.json", "w"))
        json.dump(compositions, open(out_dir / "compositions.json", "w"))
        json.dump(routes, open(out_dir / "routes.json", "w"))
        _routes.validate_sample_routes("scent", out_dir, routes=routes)
        json.dump(
            {
                "meta": {
                    "setup_s": round(setup_s, 3),
                    "device": str(device),
                    "cuda_synchronized": _use_cuda,
                    "reward_name": args.reward_name,
                    "model": args.model_name,
                    "n_trajectories": total,
                    "totals_s": {k: round(v, 3) for k, v in sample_timing.items()},
                },
            },
            open(out_dir / "sample_timings.json", "w"),
            indent=2,
        )
        print(
            f"[scent_worker] compute-time: setup {setup_s:.1f}s | "
            f"sampling {sample_timing['sampling_s']:.1f}s flow {sample_timing['flow_extract_s']:.1f}s "
            f"over {total} trajectories -> sample_timings.json",
            flush=True,
        )
        meta.update(
            {
                "n_trajectories": total,
                "n_records": len(all_records),
                "n_compositions": len(compositions),
                "n_routes": len(routes),
            }
        )
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
        print(
            f"[scent_worker] wrote records.csv + visit_counts.json + compositions.json + routes.json + meta.json -> {out_dir}",
            flush=True,
        )

    elif args.mode == "probe_hubs":
        _, hub_state_from_smiles, _ = _make_enumerator(
            rgfn,
            Trajectories,
            ReactionStateA,
            ReactionStateB,
            ReactionStateC,
            ReactionStateTerminal,
            ReactionActionC,
            Molecule,
        )
        probe = _make_hub_prober(Trajectories, ReactionStateTerminal, torch)
        hubs = _read_hubs(args.hubs_file)
        states = [hub_state_from_smiles(smi, depth) for smi, depth in hubs]
        results = probe(env, objective, reward, states)
        probe_out = {}
        for (smi, depth), r in zip(hubs, results):
            probe_out[smi] = {"depth": int(depth), **(r or {"allow_stop": False})}
        json.dump(probe_out, open(out_dir / "hub_terminal.json", "w"), indent=2)
        n_ok = sum(1 for v in probe_out.values() if v.get("allow_stop"))
        print(
            f"[scent_worker] probe_hubs: {n_ok}/{len(hubs)} hubs measured (R(h) + P_F(stop|h)) -> "
            f"{out_dir / 'hub_terminal.json'}",
            flush=True,
        )

    else:  # enumerate
        if not args.freeze:
            print(
                "[scent_worker] WARNING enumerate --no-freeze: promoted-fragment children will be missed",
                flush=True,
            )
        enumerate_terminal_children, hub_state_from_smiles, count_children = _make_enumerator(
            rgfn,
            Trajectories,
            ReactionStateA,
            ReactionStateB,
            ReactionStateC,
            ReactionStateTerminal,
            ReactionActionC,
            Molecule,
        )
        hubs = _read_hubs(args.hubs_file)

        # COUNT-ONLY: size the job, then stop. Deliberately before the flusher/oracle setup so this
        # mode never touches the docking server and can run on any node.
        if args.count_only:
            counts = []

            # Write after EVERY hub, not just at the end. The DFS is ~260 s on a real capped hub, so a
            # 25-hub sizing pass is ~2 h and a walltime kill would otherwise discard all of it -- the
            # same mistake PartialFlusher exists to prevent for the docking path. Cost is one small
            # json.dump per hub against a 4-minute DFS.
            def _flush_counts():
                done_ = [c for c in counts if "n_children" in c]
                json.dump(
                    {
                        "max_children": args.enum_max_children,
                        "reward_name": args.reward_name,
                        "n_hubs_requested": len(hubs),
                        "n_hubs_done": len(done_),
                        "partial": len(done_) < len(hubs),
                        "total_children": sum(c["n_children"] for c in done_),
                        "total_dfs_s": round(sum(c["dfs_s"] for c in done_), 2),
                        "n_hit_cap": sum(1 for c in done_ if c["hit_cap"]),
                        "per_hub": counts,
                    },
                    open(out_dir / "enum_counts.json", "w"),
                    indent=2,
                )

            for _i, (smiles, depth) in enumerate(hubs):
                hs = hub_state_from_smiles(smiles, depth)
                if hs is None:
                    counts.append({"hub": smiles, "depth": depth, "error": "invalid_smiles"})
                    _flush_counts()
                    continue
                n_paths, n_children, secs = count_children(env, hs, args.enum_max_children)
                counts.append(
                    {
                        "hub": smiles,
                        "depth": int(depth),
                        "n_paths": n_paths,
                        "n_children": n_children,
                        "dfs_s": round(secs, 3),
                        "hit_cap": n_paths >= args.enum_max_children,
                    }
                )
                _flush_counts()
                print(
                    f"[scent_worker] count hub {len(counts)}/{len(hubs)} depth={depth} -> {n_children} "
                    f"children ({n_paths} paths, {secs:.1f}s)"
                    f"{' HIT CAP' if n_paths >= args.enum_max_children else ''}  {smiles[:40]}",
                    flush=True,
                )
            done = [c for c in counts if "n_children" in c]
            print(
                f"[scent_worker] COUNT-ONLY done: {sum(c['n_children'] for c in done):,} children over "
                f"{len(done)} hubs, {sum(1 for c in done if c['hit_cap'])} still at the cap "
                f"-> {out_dir}/enum_counts.json",
                flush=True,
            )
            return

        all_records, per_hub, enum_hubs = [], [], []
        # Per-hub enumeration / reward-gen / flow-extract wall-clock (Logs/039); setup_s + _sync are
        # defined once above (shared with sample mode).
        hub_timings = []
        # Persist every 10 hubs. A docking enumeration is 6-20 GPU-hours per slice, so an unflushed
        # walltime kill discards most of a day of A100 time (it did: rxnflow_clpp 72248-72253).
        # Downstream rejects a <90%-coverage enumeration, so a partial is usable evidence and cannot
        # be mistaken for a complete cell.
        flusher = A.PartialFlusher(
            out_dir,
            every=10,
            n_hubs=len(hubs),
            timing_meta=dict(
                setup_s=setup_s,
                device=str(device),
                reward_name=args.reward_name,
                model=args.model_name,
                cuda_synchronized=_use_cuda,
            ),
        )
        for _i, (smiles, depth) in enumerate(hubs):
            hub_state = hub_state_from_smiles(smiles, depth)
            if hub_state is None:
                per_hub.append(
                    {
                        "hub": smiles,
                        "depth": depth,
                        "n_enumerated_paths": 0,
                        "n_records": 0,
                        "error": "invalid_smiles",
                    }
                )
                continue
            (
                recs,
                n_paths,
                added_by_stereo,
                reaction_by_stereo,
                hub_timing,
            ) = enumerate_terminal_children(
                env, objective, reward, hub_state, _extract, args.enum_max_children, sync=_sync
            )
            all_records.extend(recs)
            hub_timings.append(
                {
                    "hub_key": recs[0]["hub_key"] if recs else smiles,  # join key vs enum_children
                    "hub_input": smiles,
                    "depth": int(depth),
                    "n_children": len(recs),
                    **{k: round(v, 6) for k, v in hub_timing.items()},
                }
            )
            # Per-hub children with the promoted fragment(s) added in the final reaction — the
            # campaign's EnumeratedHub.children (its EnumChild.added_promoted). The hub's own
            # promoted composition is joined from the sampled compositions.json downstream.
            # ``reaction`` = the full hub->child final reaction step(s) (reaction id + reactants +
            # input/product SMILES) so the diversifying step is reconstructable, not just the fragment.
            children = [
                {
                    "smiles": r["child_key"],
                    "reward": r["reward"],
                    "added_promoted": [
                        f for f in added_by_stereo.get(r["child_stereo_key"], []) if f in chosen_set
                    ],
                    "reaction": reaction_by_stereo.get(r["child_stereo_key"], []),
                }
                for r in recs
            ]
            u_h, n_eff = _hub_uncertainty(recs)
            enum_hubs.append(
                {
                    "hub_input": smiles,
                    "hub_key": recs[0]["hub_key"] if recs else smiles,  # cross-model key (join key)
                    "depth": int(depth),
                    "uncertainty": None if u_h != u_h else u_h,  # U(h); NaN -> null in JSON
                    "n_effective": n_eff,
                    "children": children,
                }
            )
            per_hub.append(
                {
                    "hub": smiles,
                    "depth": depth,
                    "n_enumerated_paths": n_paths,
                    "n_records": len(recs),
                }
            )
            print(
                f"[scent_worker]   hub depth={depth} -> {n_paths} paths / {len(recs)} records  {smiles[:48]}",
                flush=True,
            )
            flusher.maybe(_i, enum_hubs, hub_timings)
        _write_records(out_dir / "enumerated_records.csv", all_records)
        json.dump({"per_hub": per_hub}, open(out_dir / "enum_per_hub.json", "w"), indent=2)
        json.dump({"hubs": enum_hubs}, open(out_dir / "enum_children.json", "w"))
        _routes.validate_enum_reactions("scent", out_dir)
        # Measured compute-time sidecar (Logs/039): per-hub enumeration / reward-gen / flow-extract
        # wall-clock + the one-time setup, joined to enum_children by hub_key. When the 200-hub run is
        # split into hub slices, merge these per-hub (union) and take setup_s once.
        _tt_keys = ("enumeration_s", "reward_gen_s", "flow_extract_s")
        totals = {k: sum(h.get(k, 0.0) for h in hub_timings) for k in _tt_keys}
        json.dump(
            {
                "meta": {
                    "setup_s": round(setup_s, 3),
                    "device": str(device),
                    "cuda_synchronized": _use_cuda,
                    "reward_name": args.reward_name,
                    "model": args.model_name,
                    "n_hubs": len(hub_timings),
                    "n_children": sum(h["n_children"] for h in hub_timings),
                    "totals_s": {k: round(v, 3) for k, v in totals.items()},
                },
                "per_hub": hub_timings,
            },
            open(out_dir / "enum_timings.json", "w"),
            indent=2,
        )
        print(
            f"[scent_worker] compute-time: setup {setup_s:.1f}s | enum {totals['enumeration_s']:.1f}s "
            f"reward {totals['reward_gen_s']:.1f}s flow {totals['flow_extract_s']:.1f}s "
            f"over {len(hub_timings)} hubs -> enum_timings.json",
            flush=True,
        )
        meta.update(
            {
                "n_hubs": len(hubs),
                "n_enumerated_records": len(all_records),
                "n_enum_children": sum(len(h["children"]) for h in enum_hubs),
            }
        )
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
        print(
            f"[scent_worker] enumerated {len(hubs)} hubs -> {len(all_records)} records "
            f"+ enum_children.json -> {out_dir}",
            flush=True,
        )


_PREFIX_COLS = ["child_key", "child_stereo_key", "hub_stereo_key", "log_pf_prefix", "log_pb_prefix"]


def _write_prefix_terms(path, rows):
    """prefix_terms.csv — the source->hub half for the Z-anchored reconstruction (sidecar, so
    records.csv's schema and all its readers stay untouched)."""
    rows = [r for r in rows if r.get("log_pf_prefix") is not None]
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_PREFIX_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _write_records(path, rows):
    with open(path, "w", newline="") as fh:
        # extrasaction=ignore: rows also carry the Z-anchored prefix terms, which go to
        # prefix_terms.csv instead -- records.csv's schema stays fixed for all readers.
        w = csv.DictWriter(fh, fieldnames=_REC_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _read_hubs(path):
    hubs = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            hubs.append((r["smiles"], int(r["depth"])))
    return hubs


def _hub_uncertainty(recs):
    """U(h) = population variance of the enumerated children's log F_hat (§2 flow-matching
    residual), computed from the flow log-terms the enumeration already produced. Not used as a
    signal yet — carried so it's trivially extractable later. Returns (U(h), n_effective); U is
    NaN for < 2 finite estimates (undefined variance)."""
    import math

    xs = []
    for r in recs:
        lf = r["log_reward"] + r["log_pb_move"] - r["log_pf_move"] - r["log_pf_stop"]
        if math.isfinite(lf):
            xs.append(lf)
    n = len(xs)
    if n < 2:
        return float("nan"), n
    mean = sum(xs) / n
    return sum((x - mean) ** 2 for x in xs) / n, n


if __name__ == "__main__":
    main()
