#!/usr/bin/env python
"""SCENT per-env worker — sampled flow extraction + hub-child enumeration (``LSD_FLOW_PROPOSAL`` §4b).

SCENT lives in its own ``scent`` conda env and its package is **also** named ``rgfn`` (it is an
RGFN fork), so it can never co-import with our ``rgfn``/``glue`` in one process (§4b). This worker
runs standalone *inside the scent env* (the in-process :class:`SCENTAdapter` client shells to it,
the ``scripts/score_batch.py`` bridge shape) and exchanges results over files.

Two modes:
  * ``--mode sample`` — sample trajectories, extract the §2 terminal-transition flow records, and
    write ``records.csv`` + ``visit_counts.json`` + ``meta.json`` (the canonical
    :class:`glue.samplers.lsdflow.records.FlowRecord` schema the harness reads back).
  * ``--mode enumerate`` — for each hub in ``--hubs-file`` (``smiles,depth`` rows), exhaustively
    enumerate its one-reaction terminal children and write ``enumerated_records.csv`` +
    ``enum_per_hub.json`` (mirrors :class:`RGFNAdapter.enumerate_hub_children`).

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
from pathlib import Path

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
    p.add_argument("--mode", choices=["sample", "enumerate"], default="sample")
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
    return p.parse_args()


# ----------------------------------------------------------------- flow extraction (vendored)
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
    objective, trajectories, RSA, RST, RSC, Chem, strip_stereo, chosen_set=None
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
                "log_pf_stop": log_pf_stop,
                "hub_depth": int(state_h.num_reactions),
                "hub_stereo_key": hub_stereo,
                "child_stereo_key": child_stereo,
            }
        )
    return records, visit_counts, compositions, len(states_list)


# ----------------------------------------------------------------- enumeration (vendored from rgfn_enumerate)
def _make_enumerator(rgfn_api, Trajectories, RSA, RSB, RSC, RST, RAC, Molecule):
    """Build the enumeration closures bound to SCENT's fork classes (mirror of rgfn_enumerate)."""

    def _is_stop(action):
        return getattr(action, "anchored_reaction", None) is None

    def enumerate_product_paths(env, hub_state, max_children):
        out = []

        def dfs(state, steps):
            if len(out) >= max_children:
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
        for _s, fas, act, nxt in full:
            bas = env.get_backward_action_spaces([nxt])[0]
            if isinstance(act, RAC):
                possible = getattr(bas, "possible_actions", None)
                if possible is None or act not in possible:
                    return None  # P_B not invertible -> never fabricate P_B=1
            traj.add_actions_states([act], [nxt], [fas], [bas], not_terminated_mask=None)
        return traj

    def enumerate_terminal_children(
        env, objective, reward, hub_state, extract, max_children, chunk_size=64
    ):
        paths = enumerate_product_paths(env, hub_state, max_children)
        trajs = [
            t for t in (build_child_trajectory(env, hub_state, p) for p in paths) if t is not None
        ]
        if not trajs:
            return [], len(paths)

        def _extract_batch(chunk):
            big = Trajectories.from_trajectories(chunk) if len(chunk) > 1 else chunk[0]
            terminals = big.get_last_states_flat()
            big.set_reward_outputs(reward.compute_reward_output(terminals))
            recs, _v, _c, _n = extract(objective, big)
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
        return records, len(paths)

    def hub_state_from_smiles(smiles, depth):
        mol = Molecule(smiles)
        if getattr(mol, "rdkit_mol", None) is None:
            return None
        return RSA(molecule=mol, num_reactions=int(depth))

    return enumerate_terminal_children, hub_state_from_smiles


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

    gin.parse_config_files_and_bindings(
        [config_path],
        bindings=[
            f'user_root_dir="{run_dir}"',
            'run_name="lsdflow_scent"',
            "Trainer.n_iterations=1",
            f'ScentFixedRewardRun.run_dir="{run_dir}/run"',
            f'ScentFixedRewardRun.repo_root="{REPO_ROOT}"',
            f"ScentFixedRewardRun.seed={args.seed}",
        ],
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

    def _extract(obj, traj):
        return extract_flow_records(
            obj,
            traj,
            ReactionStateA,
            ReactionStateTerminal,
            ReactionStateC,
            Chem,
            args.strip_stereo,
            chosen_set,
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

    if args.mode == "sample":
        all_records, visit_counts, compositions, total = [], {}, {}, 0
        for traj in sampler.get_trajectories_iterator(args.n_trajectories, args.batch_size):
            recs, visits, comps, n = _extract(objective, traj)
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
        json.dump(visit_counts, open(out_dir / "visit_counts.json", "w"))
        json.dump(compositions, open(out_dir / "compositions.json", "w"))
        meta.update(
            {
                "n_trajectories": total,
                "n_records": len(all_records),
                "n_compositions": len(compositions),
            }
        )
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
        print(
            f"[scent_worker] wrote records.csv + visit_counts.json + compositions.json + meta.json -> {out_dir}",
            flush=True,
        )

    else:  # enumerate
        if not args.freeze:
            print(
                "[scent_worker] WARNING enumerate --no-freeze: promoted-fragment children will be missed",
                flush=True,
            )
        enumerate_terminal_children, hub_state_from_smiles = _make_enumerator(
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
        all_records, per_hub = [], []
        for smiles, depth in hubs:
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
            recs, n_paths = enumerate_terminal_children(
                env, objective, reward, hub_state, _extract, args.enum_max_children
            )
            all_records.extend(recs)
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
        _write_records(out_dir / "enumerated_records.csv", all_records)
        json.dump({"per_hub": per_hub}, open(out_dir / "enum_per_hub.json", "w"), indent=2)
        meta.update({"n_hubs": len(hubs), "n_enumerated_records": len(all_records)})
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
        print(
            f"[scent_worker] enumerated {len(hubs)} hubs -> {len(all_records)} records -> {out_dir}",
            flush=True,
        )


def _write_records(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_REC_COLS)
        w.writeheader()
        w.writerows(rows)


def _read_hubs(path):
    hubs = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            hubs.append((r["smiles"], int(r["depth"])))
    return hubs


if __name__ == "__main__":
    main()
