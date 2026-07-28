#!/usr/bin/env python
"""FragGFN per-env worker — sampled flow extraction + hub-child enumeration (``LSD_FLOW_PROPOSAL`` §4b).

FragGFN (Recursion's ``gflownet`` fragment env) lives in its own ``fraggfn`` conda env and cannot
co-import with our ``rgfn``/``glue`` process, so this worker runs standalone inside that env and
exchanges results over files — the same shape as ``scent_worker.py``. Output is byte-identical in
schema to the other workers via the shared stdlib :mod:`_artifacts` writer.

**FragGFN is the non-reaction control (§4b).** Two structural facts drive the recipe:

  1. The trained policy builds the whole junction-tree SKELETON first (all ``AddNode``), then sets ALL
     edge attachment points, then ``Stop``. So intermediate trajectory states are NOT renderable
     molecules — the hub is *reconstructed* from a terminal as "x minus its last fragment"
     (``graph_without_node``). P_B is fixed-uniform.
  2. ``max_nodes=9`` saturates → sampled molecules are ~9 fragments, ``log_pf_stop≈0`` (forced stop).

**Enumerate (§6):** for a selected hub, enumerate ALL one-fragment-attachment children exhaustively
(``AddNode`` × fragment × stem → ``SetEdgeAttr`` src+dst on the single new edge → ``Stop``) and score
each with the trained model (``construct_batch`` + ``fwd_cat.log_prob``). The hub GRAPH is taken from
``hub_graphs.pkl`` (persisted in sample mode) — NOT reconstructed from SMILES: ``obj_to_graph``
mis-decomposes ~6% of hubs (picks a different valid fragmentation that renders to a different
molecule), so it is only a *round-trip-guarded* fallback.

**Two caveats (by design; FragGFN is the control):**
  * The enumerate "move" (AddNode + 2 SetEdgeAttr on an already-attributed hub, 3 actions) is a
    DIFFERENT conditional probability than the *sampled* move (which finalizes all deferred
    attachments, ~9-17 actions) for the same (h,x). So FragGFN enumerated F̂ is NOT comparable to
    FragGFN *sampled* F̂ (do not merge them). Within the enumerated set children ARE mutually
    comparable — which is what the campaign / U(h) need. (RGFN can merge; FragGFN can't.)
  * The complete-molecule hub is off the forward policy's training manifold (it never decorates a
    complete multi-fragment intermediate), so interpret hub values as a control, not a peer.

Modes:
  * ``--mode sample``    -> records.csv, visit_counts.json, compositions.json (empty), routes.json
                            (empty), hub_graphs.pkl, meta.json
  * ``--mode enumerate`` -> enum_children.json, enumerated_records.csv, enum_per_hub.json, meta.json
"""

import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

# Single-threaded torch/BLAS: multi-thread burns the login node's cumulative `ulimit -t` ~N_cores×
# faster (it kills long enumerations). Compute nodes are unaffected; harmless there.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

torch.set_num_threads(1)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gflownet.config import Config, init_empty  # noqa: E402
from gflownet.envs.graph_building_env import (  # noqa: E402
    ActionIndex,
    GraphAction,
    GraphActionType,
    GraphBuildingEnv,
    graph_without_node,
)
from omegaconf import OmegaConf  # noqa: E402
from rdkit import Chem  # noqa: E402

from validation.generators.fraggfn.fixed_reward import (  # noqa: E402
    DRD2FrozenReward,
    SEHFrozenReward,
)
from validation.generators.fraggfn.task import (  # noqa: E402
    FragGFNTrainer,
    build_constant_temperature,
)

STOP, ADDNODE, SETEDGE = (
    GraphActionType.Stop,
    GraphActionType.AddNode,
    GraphActionType.SetEdgeAttr,
)


def _stripped_key(ctx, g):
    """(cross-model key = stereo-stripped canonical, stereo key) from a fragment graph. None on fail."""
    try:
        mol = ctx.graph_to_obj(g)
    except Exception:
        return None, None
    if mol is None:
        return None, None
    try:
        return Chem.MolToSmiles(mol, isomericSmiles=False), Chem.MolToSmiles(
            mol, isomericSmiles=True
        )
    except Exception:
        return None, None


def _build_reward(reward_name, reward_c, device):
    if reward_name == "seh":
        return SEHFrozenReward(
            device=device,
            clip=float(reward_c.get("clip", 10.0)),
            batch_size=int(reward_c.get("batch_size", 128)),
        )
    if reward_name == "drd2":
        return DRD2FrozenReward(
            model_path=reward_c["model_path"], clip=float(reward_c.get("clip", 10.0))
        )
    raise SystemExit(
        f"[fraggfn_worker] reward '{reward_name}' not wired. Docking targets (6td3/clpp) are deferred."
    )


def _build_trainer(config_path, reward_name, device, seed, out_dir):
    cfg = OmegaConf.load(config_path)
    fr_c = cfg.get("fixed_reward", {})
    gfn_c = cfg.get("gflownet", {})
    reward_c = cfg.get("reward", {})
    beta = float(fr_c.get("beta", 8))
    clip = float(reward_c.get("clip", 10.0))

    reward = _build_reward(reward_name, reward_c, device)

    gcfg = init_empty(Config())
    gcfg.log_dir = str(out_dir / "_fgfn_scratch_logdir")
    gcfg.device = device
    gcfg.seed = seed
    gcfg.overwrite_existing_exp = True
    gcfg.print_every = 1000
    gcfg.num_training_steps = int(fr_c.get("n_train_steps", 5000))
    gcfg.algo.max_nodes = int(gfn_c.get("max_nodes", 9))
    gcfg.algo.sampling_tau = float(gfn_c.get("sampling_tau", 0.9))
    gcfg.model.num_emb = int(gfn_c.get("num_emb", 128))
    gcfg.model.num_layers = int(gfn_c.get("num_layers", 4))
    gcfg.opt.learning_rate = float(gfn_c.get("learning_rate", 1e-4))
    build_constant_temperature(gcfg, beta)

    trainer = FragGFNTrainer(gcfg, proxy=reward)
    return trainer, reward, beta, clip


# ============================================================ sample mode
def _extract_batch(trainer, reward, beta, clip, enc, trajs):
    """§2 flow records + hub graphs for one sampled batch (reconstruct-hub recipe).
    Returns (records, visit_counts, hub_graphs) — hub_graphs maps hub_stereo_key -> the hub Graph
    (persisted so enumerate can reload the exact scaffold rather than re-decompose from SMILES)."""
    model, algo, ctx, dev = trainer.model, trainer.algo, trainer.ctx, trainer.device

    term_smiles = []
    for t in trajs:
        try:
            term_smiles.append(Chem.MolToSmiles(ctx.graph_to_obj(t["result"])))
        except Exception:
            term_smiles.append(None)
    valid = [s for s in term_smiles if s is not None]
    vmap = dict(zip(valid, reward.predict(valid))) if valid else {}
    values = [vmap.get(s, float("nan")) for s in term_smiles]

    def shaped(v):
        return beta * (-clip if v != v else float(np.clip(v, -clip, clip)))

    log_rewards = torch.tensor([shaped(v) for v in values], dtype=torch.float, device=dev)
    batch = algo.construct_batch(trajs, enc, log_rewards).to(dev)
    n = len(trajs)
    bidx = torch.arange(n, device=dev).repeat_interleave(batch.traj_lens.to(dev))
    with torch.no_grad():
        fwd_cat, _ = model(batch, batch.cond_info[bidx])
        log_p_F = fwd_cat.log_prob(batch.actions).detach().cpu().tolist()
    log_p_B = batch.log_p_B.detach().cpu().tolist()
    offs = np.cumsum([0] + batch.traj_lens.detach().cpu().tolist())

    records, visit_counts, hub_graphs = [], {}, {}
    for t in range(n):
        traj = trajs[t]["traj"]
        L = len(traj)
        if not trajs[t].get("is_valid", True) or L == 0 or traj[-1][1].action is not STOP:
            continue
        idx_stop = L - 1
        x = trajs[t]["result"]
        n_x = len(x.nodes)
        if n_x < 2:
            continue
        child_key, child_stereo = _stripped_key(ctx, x)
        if child_key is None:
            continue
        add_idx = next(
            (i for i in range(idx_stop - 1, -1, -1) if traj[i][1].action is ADDNODE), None
        )
        if add_idx is None:
            continue
        last_id = max(x.nodes)
        if x.degree[last_id] > 1:
            continue
        hub_graph = graph_without_node(x, last_id)
        hub_key, hub_stereo = _stripped_key(ctx, hub_graph)
        if hub_key is None:
            continue
        pf = log_p_F[offs[t] : offs[t + 1]]
        pb = log_p_B[offs[t] : offs[t + 1]]
        v = values[t]
        records.append(
            {
                "hub_key": hub_key,
                "child_key": child_key,
                "reward": float(v) if v == v else float("nan"),
                "log_reward": float(log_rewards[t].item()),
                "log_pf_move": float(sum(pf[add_idx:idx_stop])),
                "log_pb_move": float(sum(pb[add_idx:idx_stop])),
                "log_pf_stop": float(pf[idx_stop]),
                "hub_depth": int((n_x - 1) - 1),
                "hub_stereo_key": hub_stereo,
                "child_stereo_key": child_stereo,
                # Z-anchored half: source -> anchor state (see _artifacts.write_prefix_terms).
                "log_pf_prefix": float(sum(pf[:add_idx])),
                "log_pb_prefix": float(sum(pb[:add_idx])),
            }
        )
        for k in {hub_key, child_key}:
            visit_counts[k] = visit_counts.get(k, 0) + 1
        if hub_stereo not in hub_graphs:
            try:
                hub_graph.clear_cache()  # drop torch _Data cache -> tiny pickle
            except Exception:
                pass
            hub_graphs[hub_stereo] = hub_graph
    return records, visit_counts, hub_graphs


def _run_sample(args, trainer, reward, beta, clip, out_dir, device):
    model, task = trainer.model, trainer.task
    all_records, visit_counts, hub_graphs, total = [], {}, {}, 0
    remaining = args.n_trajectories
    while remaining > 0:
        b = min(args.batch_size, remaining)
        cond = task.sample_conditional_information(b, 0)
        enc = cond["encoding"].to(device)
        with torch.no_grad():
            trajs = trainer.algo.create_training_data_from_own_samples(
                model, b, enc, random_action_prob=0.0
            )
        recs, visits, hgs = _extract_batch(trainer, reward, beta, clip, enc, trajs)
        all_records.extend(recs)
        for k, c in visits.items():
            visit_counts[k] = visit_counts.get(k, 0) + c
        for k, g in hgs.items():
            hub_graphs.setdefault(k, g)
        total += b
        remaining -= b
    log_z = (
        float(model.logZ(torch.zeros(1, task.num_cond_dim, device=device)).item())
        if hasattr(task, "num_cond_dim")
        else 0.0
    )
    A.write_records(out_dir / "records.csv", all_records)
    A.write_prefix_terms(out_dir / "prefix_terms.csv", all_records)
    json.dump(visit_counts, open(out_dir / "visit_counts.json", "w"))
    # No promoted fragments, but charge each molecule its flat fragment-attachment count so the
    # count-once cost is internally consistent (best-candidate not undercounted to 1). NOTE: for
    # FragGFN "reactions" = fragment attachments, an approximation; the meaningful library cost
    # comes from post-hoc retrosynthesis (the researcher's downstream step), not this number.
    json.dump(A.compositions_from_records(all_records), open(out_dir / "compositions.json", "w"))
    json.dump({}, open(out_dir / "routes.json", "w"))
    pickle.dump(hub_graphs, open(out_dir / "hub_graphs.pkl", "wb"))  # for enumerate (§ hub-state)
    A.write_json(
        out_dir / "meta.json",
        {
            "model": args.model_name,
            "reward_name": args.reward_name,
            "higher_is_better": True,
            "log_z": log_z,
            "strip_stereo": True,
            "frozen": False,
            "n_promoted_fragments": 0,
            "n_trajectories": total,
            "n_records": len(all_records),
            "n_hub_graphs": len(hub_graphs),
            "checkpoint_it": args._it,
        },
    )
    print(
        f"[fraggfn_worker] sample: {total} trajectories -> {len(all_records)} records, "
        f"{len(visit_counts)} nodes, {len(hub_graphs)} hub graphs -> {out_dir}",
        flush=True,
    )


# ============================================================ enumerate mode
def _legal_actions(ctx, g, atype):
    """All legal GraphActions of one type from graph g, read from the ctx masks (as the model sees)."""
    data = ctx.graph_to_Data(g)
    tidx = ctx.action_type_order.index(atype)
    mask = getattr(data, atype.mask_name)
    acts = []
    for row in mask.nonzero().tolist():
        r, c = int(row[0]), int(row[1])
        acts.append(
            ctx.ActionIndex_to_GraphAction(
                data, ActionIndex(action_type=tidx, row_idx=r, col_idx=c)
            )
        )
    return acts


def _enumerate_children(ctx, env, hub_g, max_children):
    """All one-fragment-attachment terminal children of hub_g, deduped by child SMILES.

    Path: AddNode(source, frag) -> SetEdgeAttr(src_attach) -> SetEdgeAttr(dst_attach) -> x -> Stop.
    The hub (from a terminal) already has all EXISTING edges attributed, so only the ONE new edge
    needs its two attach points. src-first canonical order dedups the two attach orderings; we also
    dedup by child_key (symmetric fragments give idempotent paths)."""
    out, seen = [], set()
    if len(hub_g.nodes) >= ctx.max_frags:
        return out  # boundary: no AddNode legal -> only Stop -> 0 children
    for a1 in _legal_actions(ctx, hub_g, ADDNODE):
        g1 = env.step(hub_g, a1)
        new_node = max(g1.nodes)
        for a2 in _legal_actions(ctx, g1, SETEDGE):
            if a2.attr != "src_attach" or new_node not in (a2.source, a2.target):
                continue
            g2 = env.step(g1, a2)
            for a3 in _legal_actions(ctx, g2, SETEDGE):
                if a3.attr != "dst_attach" or new_node not in (a3.source, a3.target):
                    continue
                g3 = env.step(g2, a3)
                if not ctx.is_sane(g3):
                    continue
                ck, _ = _stripped_key(ctx, g3)
                if ck is None or ck in seen:
                    continue
                seen.add(ck)
                out.append((hub_g, a1, g1, a2, g2, a3, g3))
                if len(out) >= max_children:
                    return out
    return out


def _score_children(trainer, reward, beta, clip, paths, chunk=256):
    """§2 flow terms per enumerated child (construct_batch + model, chunked for memory).
    Returns (records, added_by_child, reaction_by_child)."""
    model, algo, ctx, task, dev = (
        trainer.model,
        trainer.algo,
        trainer.ctx,
        trainer.task,
        trainer.device,
    )
    env = algo.env

    def shaped(v):
        return beta * (-clip if v != v else float(np.clip(v, -clip, clip)))

    records, added_by_child, reaction_by_child = [], {}, {}
    for start in range(0, len(paths), chunk):
        block = paths[start : start + chunk]
        traj_dicts, xg = [], []
        for hub_g, a1, g1, a2, g2, a3, g3 in block:
            traj = [(hub_g, a1), (g1, a2), (g2, a3), (g3, GraphAction(STOP))]
            bck = torch.tensor(
                [
                    math.log(1.0 / env.count_backward_transitions(g1)),
                    math.log(1.0 / env.count_backward_transitions(g2)),
                    math.log(1.0 / env.count_backward_transitions(g3)),
                    0.0,  # Stop reverse deterministic
                ],
                dtype=torch.float,
            )
            traj_dicts.append({"traj": traj, "bck_logprobs": bck, "result": g3, "is_valid": True})
            xg.append(g3)
        x_smis = [Chem.MolToSmiles(ctx.graph_to_obj(g)) for g in xg]
        vals = reward.predict(x_smis)
        log_rewards = torch.tensor([shaped(v) for v in vals], dtype=torch.float, device=dev)
        enc = torch.zeros(len(traj_dicts), task.num_cond_dim, device=dev)
        batch = algo.construct_batch(traj_dicts, enc, log_rewards).to(dev)
        n = len(traj_dicts)
        bidx = torch.arange(n, device=dev).repeat_interleave(batch.traj_lens.to(dev))
        with torch.no_grad():
            fwd_cat, _ = model(batch, batch.cond_info[bidx])
            pf = fwd_cat.log_prob(batch.actions).detach().cpu().tolist()
        pb = batch.log_p_B.detach().cpu().tolist()
        offs = np.cumsum([0] + batch.traj_lens.detach().cpu().tolist())
        for t, (hub_g, a1, g1, a2, g2, a3, g3) in enumerate(block):
            s, e = offs[t], offs[t + 1]
            idx_stop = (e - s) - 1
            hub_key, hub_stereo = _stripped_key(ctx, hub_g)
            child_key, child_stereo = _stripped_key(ctx, g3)
            v = vals[t]
            n_x = len(g3.nodes)
            records.append(
                {
                    "hub_key": hub_key,
                    "child_key": child_key,
                    "reward": float(v) if v == v else float("nan"),
                    "log_reward": float(log_rewards[t].item()),
                    "log_pf_move": float(sum(pf[s : s + idx_stop])),
                    "log_pb_move": float(sum(pb[s : s + idx_stop])),
                    "log_pf_stop": float(pf[s + idx_stop]),
                    "hub_depth": int((n_x - 1) - 1),
                    "hub_stereo_key": hub_stereo,
                    "child_stereo_key": child_stereo,
                }
            )
            frag_smi = ctx.frags_smi[a1.value]
            added_by_child[child_stereo] = [frag_smi]
            reaction_by_child[child_stereo] = [
                {
                    "op": "add_fragment",
                    "source_node": int(a1.source),
                    "fragment": frag_smi,
                    "src_attach": int(a2.value),
                    "dst_attach": int(a3.value),
                    "input": hub_stereo,
                    "product": child_stereo,
                }
            ]
    return records, added_by_child, reaction_by_child


def _hub_graph_for(hub_stereo, hub_key, hub_graphs, ctx):
    """The drivable hub Graph for a selected hub. Prefer the persisted graph (exact); fall back to
    obj_to_graph with a round-trip guard (drop on mismatch — ~6% of hubs mis-decompose)."""
    g = hub_graphs.get(hub_stereo)
    if g is not None:
        return g
    mol = Chem.MolFromSmiles(hub_stereo)
    if mol is None:
        return None
    try:
        g = ctx.obj_to_graph(mol)
    except Exception:
        return None
    if g is None:
        return None
    rt_key, _ = _stripped_key(ctx, g)
    return g if rt_key == hub_key else None  # never enumerate a mis-decomposed scaffold


def _read_hubs(path):
    hubs = []
    with open(path) as fh:
        import csv

        for r in csv.DictReader(fh):
            hubs.append((r["smiles"], int(r["depth"])))
    return hubs


def _run_enumerate(args, trainer, reward, beta, clip, out_dir):
    ctx = trainer.ctx
    env = trainer.algo.env if getattr(trainer.algo, "env", None) is not None else GraphBuildingEnv()
    if not args.hubs_file:
        raise SystemExit("[fraggfn_worker] --mode enumerate requires --hubs-file")
    hub_pkl = Path(args.sample_dir or out_dir) / "hub_graphs.pkl"
    hub_graphs = pickle.load(open(hub_pkl, "rb")) if hub_pkl.exists() else {}
    if not hub_graphs:
        print(
            f"[fraggfn_worker] WARNING no hub_graphs.pkl at {hub_pkl} -> falling back to guarded "
            "obj_to_graph for every hub (drops ~6% mis-decomposed).",
            flush=True,
        )
    hubs = _read_hubs(args.hubs_file)
    all_records, enum_hubs, per_hub = [], [], []
    for hub_stereo, depth in hubs:
        hub_key = (
            Chem.MolToSmiles(Chem.MolFromSmiles(hub_stereo), isomericSmiles=False)
            if Chem.MolFromSmiles(hub_stereo)
            else hub_stereo
        )
        hub_g = _hub_graph_for(hub_stereo, hub_key, hub_graphs, ctx)
        if hub_g is None:
            per_hub.append(
                {"hub": hub_stereo, "depth": depth, "n_records": 0, "error": "no_hub_graph"}
            )
            continue
        paths = _enumerate_children(ctx, env, hub_g, args.enum_max_children)
        recs, added, reactions = _score_children(trainer, reward, beta, clip, paths)
        all_records.extend(recs)
        enum_hubs.append(
            A.build_enum_hub(
                hub_input=hub_stereo,
                hub_key=recs[0]["hub_key"] if recs else hub_key,
                depth=depth,
                recs=recs,
                added_by_child=added,
                reaction_by_child=reactions,
            )
        )
        per_hub.append({"hub": hub_stereo, "depth": depth, "n_records": len(recs)})
        print(
            f"[fraggfn_worker]   hub depth={depth} -> {len(recs)} children  {hub_stereo[:48]}",
            flush=True,
        )

    A.write_records(out_dir / "enumerated_records.csv", all_records)
    A.write_enum_children(out_dir / "enum_children.json", enum_hubs)
    A.write_json(out_dir / "enum_per_hub.json", {"per_hub": per_hub})
    A.write_json(
        out_dir / "meta.json",
        {
            "model": args.model_name,
            "reward_name": args.reward_name,
            "n_hubs": len(hubs),
            "n_enumerated_records": len(all_records),
            "n_enum_children": sum(len(h["children"]) for h in enum_hubs),
            "checkpoint_it": args._it,
        },
    )
    print(
        f"[fraggfn_worker] enumerate: {len(hubs)} hubs -> {len(all_records)} records + "
        f"enum_children.json -> {out_dir}",
        flush=True,
    )


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sample", "enumerate"], default="sample")
    p.add_argument("--config", required=True, help="fraggfn *_fixed_5k.yaml")
    p.add_argument("--checkpoint", required=True, help="trained last_gfn.pt")
    p.add_argument("--reward-name", default="seh")
    p.add_argument("--model-name", default="fraggfn")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-trajectories", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default="/tmp/lsdflow_fraggfn")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hubs-file", default="")
    p.add_argument(
        "--sample-dir",
        default="",
        help="enumerate mode: dir holding hub_graphs.pkl (default: --out-dir). submit_cell.sh "
        "points this at the sample stage's output.",
    )
    p.add_argument("--enum-max-children", type=int, default=4000)
    return p.parse_args()


def main():
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer, reward, beta, clip = _build_trainer(
        args.config, args.reward_name, device, args.seed, out_dir
    )
    model = trainer.model
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    args._it = state.get("it", "?")
    print(
        f"[fraggfn_worker] loaded checkpoint it={args._it} reward={args.reward_name} "
        f"mode={args.mode} device={device}",
        flush=True,
    )

    if args.mode == "sample":
        _run_sample(args, trainer, reward, beta, clip, out_dir, device)
    else:
        _run_enumerate(args, trainer, reward, beta, clip, out_dir)


if __name__ == "__main__":
    main()
