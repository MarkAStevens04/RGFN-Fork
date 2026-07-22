#!/usr/bin/env python
"""RxnFlow per-env worker — sampled flow extraction + hub-child enumeration (``LSD_FLOW_PROPOSAL`` §4b).

RxnFlow (synthesizable reaction-template + building-block GFlowNet) lives in its own ``rxnflow`` conda
env and cannot co-import with our ``rgfn``/``glue`` process, so this worker runs standalone inside that
env and exchanges results over files — the same shape as ``scent_worker.py``. Output is byte-identical
in schema to the other workers via the shared stdlib :mod:`_artifacts` writer.

**Sample (§2).** Two terminal kinds: RxnFlow mostly terminates by hitting the reaction cap
(``algo.max_len``), not an explicit ``Stop`` (~93% cap-truncated) — Case A (explicit Stop) scores the
Stop factor; Case B (cap) forces ``log_pf_stop=0``. Forward normalizer forced exact
(``sampling_ratio=1.0``; trained at 0.01). ``log P_B`` is RxnFlow's retro HEURISTIC
(``do_parameterize_p_b=False``) → muddy ``U(h)`` (§5), recorded flagged.

**Enumerate (§6).** RxnFlow is Markovian (``RxnFlow.forward`` embeds only the current molecule graph +
cond, no trajectory history), so a hub STATE is rebuilt directly from its SMILES — ``MolGraph(hub_smi)``
— with NO persistence needed (unlike FragGFN); reconstructed forward terms are bit-identical to
sampling. For each hub we enumerate ALL applicable one-reaction children (``ctx.create_masks`` →
protocol; ``env.birxn_block_indices`` → 2nd reactant blocks; ``env.step`` fires the RDKit reaction),
score them exactly, and take ``log_pf_stop=0`` for cap-depth children / the explicit Stop factor
otherwise. **P_B requires injecting the reverse action as a known retro branch AND a FRESH
``RetroSyntheticAnalyzer`` per child** (a shared analyzer's cache pollutes P_B → collapses to 0).

Modes:
  * ``--mode sample``    -> records.csv, visit_counts.json, compositions.json, routes.json, meta.json
  * ``--mode enumerate`` -> enum_children.json, enumerated_records.csv, enum_per_hub.json, meta.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _artifacts as A  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf  # noqa: E402
from rdkit import Chem  # noqa: E402
from rxnflow.config import Config, init_empty  # noqa: E402
from rxnflow.envs.action import RxnAction, RxnActionType  # noqa: E402
from rxnflow.envs.env import MolGraph  # noqa: E402
from rxnflow.envs.retrosynthesis import (  # noqa: E402
    RetroSynthesisTree,
    RetroSyntheticAnalyzer,
)

from validation.generators.rxnflow.fixed_reward import (  # noqa: E402
    DRD2FrozenReward,
    SEHFrozenReward,
)
from validation.generators.rxnflow.task import (  # noqa: E402
    RxnFlowGlueTrainer,
    build_constant_temperature,
)

_RXN = (RxnActionType.UniRxn, RxnActionType.BiRxn)
ILLEGAL = -75.0


def _set_if(cfg, dotted, value):
    if value is None:
        return
    obj = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not hasattr(obj, p):
            return
        obj = getattr(obj, p)
    if hasattr(obj, parts[-1]):
        setattr(obj, parts[-1], value)


def _stripped_key(smi):
    """(cross-model key = stereo-stripped canonical, stereo key). RxnFlow keeps stereo (§6)."""
    if not smi:
        return "", ""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi, smi
    return Chem.MolToSmiles(m, isomericSmiles=False), Chem.MolToSmiles(m)


def _build_reward(reward_name, reward_c, device):
    if reward_name == "seh":
        return SEHFrozenReward(device=device, clip=float(reward_c.get("clip", 10.0)))
    if reward_name == "drd2":
        return DRD2FrozenReward(
            model_path=reward_c["model_path"], clip=float(reward_c.get("clip", 10.0))
        )
    raise SystemExit(
        f"[rxnflow_worker] reward '{reward_name}' not wired. Docking targets (6td3/clpp) are deferred."
    )


def _build_trainer(config_path, reward_name, device, seed, out_dir):
    cfg = OmegaConf.load(config_path)
    fr_c = cfg.get("fixed_reward", {})
    reward_c = cfg.get("reward", {})
    rxn_c = cfg.get("rxnflow", {})
    beta = float(fr_c.get("beta", 8))
    clip = float(reward_c.get("clip", 10.0))

    reward = _build_reward(reward_name, reward_c, device)

    gcfg = init_empty(Config())
    _set_if(gcfg, "log_dir", str(out_dir / "_rxn_scratch_logdir"))
    _set_if(gcfg, "device", device)
    _set_if(gcfg, "seed", seed)
    _set_if(gcfg, "overwrite_existing_exp", True)
    _set_if(gcfg, "print_every", 1000)
    _set_if(gcfg, "num_training_steps", int(fr_c.get("n_train_steps", 5000)))
    _set_if(gcfg, "opt.learning_rate", float(rxn_c.get("learning_rate", 1e-4)))
    _set_if(gcfg, "model.num_emb", int(rxn_c.get("num_emb", 128)))
    _set_if(gcfg, "model.graph_transformer.num_layers", int(rxn_c.get("num_layers", 4)))
    _set_if(gcfg, "algo.sampling_tau", float(rxn_c.get("sampling_tau", 0.9)))
    _set_if(gcfg, "algo.max_len", int(rxn_c.get("max_reactions", 4)))
    env_dir = rxn_c.get("env_dir")
    if not env_dir:
        raise SystemExit("[rxnflow_worker] rxnflow.env_dir missing from config")
    _set_if(
        gcfg, "env_dir", str(REPO_ROOT / env_dir if not str(env_dir).startswith("/") else env_dir)
    )
    _set_if(gcfg, "num_workers_retrosynthesis", 4)
    _set_if(gcfg, "algo.action_subsampling.sampling_ratio", 1.0)  # exact normalizer for extraction
    build_constant_temperature(gcfg, beta)

    trainer = RxnFlowGlueTrainer(gcfg, proxy=reward, print_config=False)
    return trainer, beta, clip


# ============================================================ sample mode
def _extract_batch(trainer, beta, clip, enc, data):
    """§2 flow records for one sampled batch (both terminal kinds). Returns (records, visit_counts)."""
    algo, task, dev = trainer.algo, trainer.task, trainer.device
    keep, keep_idx = [], []
    for i, d in enumerate(data):
        tr = d["traj"]
        if (
            d.get("is_valid", True)
            and len(tr)
            and (tr[-1][1].action is RxnActionType.Stop or tr[-1][1].action in _RXN)
        ):
            keep.append(d)
            keep_idx.append(i)
    if not keep:
        return [], {}
    enc_keep = enc[keep_idx]
    x_smis = [d["result"].smi for d in keep]
    raw = task.proxy.predict(x_smis)
    log_rewards = torch.tensor(
        [max(beta * min(max(v, -clip), clip), ILLEGAL) if v == v else ILLEGAL for v in raw],
        dtype=torch.float32,
    )
    batch = algo.construct_batch(keep, enc_keep, log_rewards).to(dev)
    nt = int(batch.traj_lens.shape[0])
    bidx = torch.arange(nt, device=dev).repeat_interleave(batch.traj_lens)
    with torch.no_grad():
        fwd_cat, _ = trainer.model(batch, batch.cond_info[bidx])
        fwd = fwd_cat.log_prob(batch.actions).detach().cpu().tolist()
    bwd = batch.log_p_B.detach().cpu().tolist()
    lens = batch.traj_lens.detach().cpu().tolist()
    assert len(fwd) == len(bwd) == sum(lens), (len(fwd), len(bwd), sum(lens))

    records, visit_counts = [], {}
    off = 0
    for t, d in enumerate(keep):
        tr = d["traj"]
        L = lens[t]
        tfwd, tbwd = fwd[off : off + L], bwd[off : off + L]
        off += L
        seen = {_stripped_key(g.smi)[0] for g, _a in tr if g.smi}
        if d["result"].smi:
            seen.add(_stripped_key(d["result"].smi)[0])
        for k in seen:
            visit_counts[k] = visit_counts.get(k, 0) + 1
        if tr[-1][1].action is RxnActionType.Stop:  # Case A
            idx_stop, move_idx = L - 1, L - 2
            if move_idx < 0 or tr[move_idx][1].action not in _RXN:
                continue
            hub_g, x_g = tr[move_idx][0], tr[idx_stop][0]
            log_pf_stop = float(tfwd[idx_stop])
        else:  # Case B: cap-truncated
            move_idx = L - 1
            if tr[move_idx][1].action not in _RXN:
                continue
            hub_g, x_g = tr[move_idx][0], d["result"]
            log_pf_stop = 0.0
        total_rxn = sum(1 for _g, a in tr if a.action in _RXN)
        hub_key, hub_stereo = _stripped_key(hub_g.smi)
        child_key, child_stereo = _stripped_key(x_g.smi)
        records.append(
            {
                "hub_key": hub_key,
                "child_key": child_key,
                "reward": float(raw[t]) if raw[t] == raw[t] else float("nan"),
                "log_reward": float(log_rewards[t]),
                "log_pf_move": float(tfwd[move_idx]),
                "log_pb_move": float(tbwd[move_idx]),
                "log_pf_stop": log_pf_stop,
                "hub_depth": int(total_rxn - 1),
                "hub_stereo_key": hub_stereo,
                "child_stereo_key": child_stereo,
            }
        )
    return records, visit_counts


def _run_sample(args, trainer, beta, clip, out_dir, device):
    all_records, visit_counts, total = [], {}, 0
    remaining = args.n_trajectories
    enc = None
    while remaining > 0:
        b = min(args.batch_size, remaining)
        cond = trainer.task.sample_conditional_information(b, 0)
        enc = cond["encoding"].to(device)
        with torch.no_grad():
            data = trainer.algo.graph_sampler.sample_from_model(
                trainer.model, b, enc, random_action_prob=0.0
            )
        recs, visits = _extract_batch(trainer, beta, clip, enc, data)
        all_records.extend(recs)
        for k, c in visits.items():
            visit_counts[k] = visit_counts.get(k, 0) + c
        total += b
        remaining -= b
    try:
        log_z = float(trainer.model.logZ(enc[:1])[:, 0].item())
    except Exception:
        log_z = 0.0

    A.write_records(out_dir / "records.csv", all_records)
    json.dump(visit_counts, open(out_dir / "visit_counts.json", "w"))
    # No promoted fragments; charge each molecule its flat reaction depth so the count-once cost
    # charges best-candidate correctly (not the =1 fallback). RxnFlow reactions are real steps.
    json.dump(A.compositions_from_records(all_records), open(out_dir / "compositions.json", "w"))
    json.dump({}, open(out_dir / "routes.json", "w"))  # routes extractable later (ctx.read_traj)
    A.write_json(
        out_dir / "meta.json",
        {
            "model": args.model_name,
            "reward_name": args.reward_name,
            "higher_is_better": bool(getattr(trainer.task.proxy, "higher_is_better", True)),
            "log_z": log_z,
            "strip_stereo": True,
            "frozen": False,
            "n_promoted_fragments": 0,
            "n_trajectories": total,
            "n_records": len(all_records),
            "checkpoint_it": args._it,
            "extract_sampling_ratio": 1.0,
        },
    )
    print(
        f"[rxnflow_worker] sample: {total} trajectories -> {len(all_records)} records, "
        f"{len(visit_counts)} nodes -> {out_dir}",
        flush=True,
    )


# ============================================================ enumerate mode
def _pb_retro(trainer, hub_smi, hub_depth, child_smi, child_depth, reverse_action):
    """P_B(hub|child) via the retro heuristic. reverse_action MUST be injected as a known branch
    (RxnAction eq compares TYPE: BckBiRxn != BiRxn, else log P_B=0). FRESH analyzer per child
    (a shared one's (smiles,depth) cache pollutes P_B across children -> collapses to 0)."""
    env, algo = trainer.env, trainer.algo
    ana = RetroSyntheticAnalyzer(env.protocols, env.blocks, approx=True, max_decomposes=2)
    htree = ana.run(hub_smi, hub_depth, None) or RetroSynthesisTree("")
    ctree = ana.run(child_smi, child_depth, [(reverse_action, htree)])
    return algo.graph_sampler.calc_bck_logprob(reverse_action, ctree)


def _score_child_trajs(trainer, child_trajs):
    """Per-traj forward log-prob lists via construct_batch + model (exact normalizer)."""
    algo, task, dev = trainer.algo, trainer.task, trainer.device
    enc = task.sample_conditional_information(len(child_trajs), 0)["encoding"].to(dev)
    batch = algo.construct_batch(child_trajs, enc, torch.zeros(len(child_trajs))).to(dev)
    nt = int(batch.traj_lens.shape[0])
    bidx = torch.arange(nt, device=dev).repeat_interleave(batch.traj_lens)
    with torch.no_grad():
        fwd_cat, _ = trainer.model(batch, batch.cond_info[bidx])
        fwd = fwd_cat.log_prob(batch.actions).detach().cpu().tolist()
    lens = batch.traj_lens.detach().cpu().tolist()
    out, off = [], 0
    for L in lens:
        out.append(fwd[off : off + L])
        off += L
    return out


def _enumerate_hub(
    trainer, beta, clip, hub_smi, hub_depth, max_children, max_rxn, min_len, stop_proto
):
    """All one-reaction terminal children of a hub (approach a: MolGraph from SMILES).
    Returns (records, n_paths, added_by_child, reaction_by_child)."""
    ctx, env, task = trainer.ctx, trainer.env, trainer.task
    child_depth = hub_depth + 1
    if child_depth > max_rxn:
        return [], 0, {}, {}  # a hub at/after the cap has no one-reaction terminal children
    hub_g = MolGraph(hub_smi)
    if hub_g.mol is None:
        return [], 0, {}, {}
    hub_g.graph["sample_idx"] = 0
    hub_g.graph["allow_stop"] = hub_depth + 1 >= min_len
    is_cap = child_depth == max_rxn

    mask = ctx.create_masks(hub_g).tolist()
    enum = []  # (reaction_action, child_g)
    for pidx, m in enumerate(mask):
        if not m:
            continue
        proto = ctx.protocols[pidx]
        if proto.action is RxnActionType.UniRxn:
            act = RxnAction(RxnActionType.UniRxn, proto.name)
            try:
                child_g = env.step(hub_g, act)
            except Exception:
                continue
            if child_g.mol is not None:
                enum.append((act, child_g))
        elif proto.action is RxnActionType.BiRxn:
            for bidx in env.birxn_block_indices[proto.name].tolist():
                act = RxnAction(RxnActionType.BiRxn, proto.name, env.blocks[bidx], int(bidx))
                try:
                    child_g = env.step(hub_g, act)
                except Exception:
                    continue
                if child_g.mol is not None:
                    enum.append((act, child_g))
                if len(enum) >= max_children:
                    break
        if len(enum) >= max_children:
            break
    n_paths = len(enum)
    if not enum:
        return [], 0, {}, {}

    child_trajs, meta = [], []
    for act, child_g in enum:
        child_g.graph["sample_idx"] = 0
        child_g.graph["allow_stop"] = True
        rev = env.reverse(hub_g, act)
        logpb = _pb_retro(trainer, hub_smi, hub_depth, child_g.smi, child_depth, rev)
        if is_cap:
            traj, bck = [(hub_g, act)], [logpb]  # Case B: forced termination
        else:
            stop = RxnAction(RxnActionType.Stop, stop_proto)
            traj, bck = [(hub_g, act), (child_g, stop)], [logpb, 0.0]  # Case A: explicit Stop
        child_trajs.append(
            {"traj": traj, "bck_logprobs": torch.tensor(bck), "is_valid": True, "result": child_g}
        )
        meta.append((act, child_g, logpb))

    fwd_lists = _score_child_trajs(trainer, child_trajs)
    x_smis = [cg.smi for _a, cg, _lpb in meta]
    rewards = task.proxy.predict(x_smis)

    def shaped(v):
        return max(beta * min(max(v, -clip), clip), ILLEGAL) if v == v else ILLEGAL

    hub_key, hub_stereo = _stripped_key(hub_smi)
    best, added_by_child, reaction_by_child = {}, {}, {}
    for (act, child_g, logpb), fwdL, val in zip(meta, fwd_lists, rewards):
        child_key, child_stereo = _stripped_key(child_g.smi)
        rec = {
            "hub_key": hub_key,
            "child_key": child_key,
            "reward": float(val) if val == val else float("nan"),
            "log_reward": float(shaped(val)),
            "log_pf_move": float(fwdL[0]),
            "log_pb_move": float(logpb),
            "log_pf_stop": 0.0 if is_cap else float(fwdL[1]),
            "hub_depth": int(hub_depth),
            "hub_stereo_key": hub_stereo,
            "child_stereo_key": child_stereo,
        }
        lf = rec["log_reward"] + rec["log_pb_move"] - rec["log_pf_move"] - rec["log_pf_stop"]
        if child_stereo not in best or lf > best[child_stereo][0]:
            best[child_stereo] = (lf, rec)
            reaction_by_child[child_stereo] = [
                {
                    "op": "reaction",
                    "protocol": act.protocol,
                    "block": getattr(act, "block", None),
                    "input": hub_stereo,
                    "product": child_stereo,
                }
            ]
    recs = [v[1] for v in best.values()]
    return recs, n_paths, added_by_child, reaction_by_child


def _read_hubs(path):
    import csv

    hubs = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            hubs.append((r["smiles"], int(r["depth"])))
    return hubs


def _run_enumerate(args, trainer, beta, clip, out_dir):
    if not args.hubs_file:
        raise SystemExit("[rxnflow_worker] --mode enumerate requires --hubs-file")
    env = trainer.env
    max_len = int(trainer.cfg.algo.max_len)
    min_len = int(trainer.cfg.algo.min_len)
    max_rxn = max_len - 1  # FirstBlock is action 0 -> max reactions = max_len - 1
    stop_proto = env.stop_list[0].name
    hubs = _read_hubs(args.hubs_file)
    all_records, enum_hubs, per_hub = [], [], []
    for hub_stereo, depth in hubs:
        recs, n_paths, added, reactions = _enumerate_hub(
            trainer,
            beta,
            clip,
            hub_stereo,
            depth,
            args.enum_max_children,
            max_rxn,
            min_len,
            stop_proto,
        )
        all_records.extend(recs)
        enum_hubs.append(
            A.build_enum_hub(
                hub_input=hub_stereo,
                hub_key=recs[0]["hub_key"] if recs else _stripped_key(hub_stereo)[0],
                depth=depth,
                recs=recs,
                added_by_child=added,
                reaction_by_child=reactions,
            )
        )
        per_hub.append(
            {"hub": hub_stereo, "depth": depth, "n_paths": n_paths, "n_records": len(recs)}
        )
        print(
            f"[rxnflow_worker]   hub depth={depth} -> {n_paths} paths / {len(recs)} children  {hub_stereo[:44]}",
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
        f"[rxnflow_worker] enumerate: {len(hubs)} hubs -> {len(all_records)} records + "
        f"enum_children.json -> {out_dir}",
        flush=True,
    )


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["sample", "enumerate"], default="sample")
    p.add_argument("--config", required=True, help="rxnflow *_fixed_stdlib_5k.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--reward-name", default="seh")
    p.add_argument("--model-name", default="rxnflow")
    p.add_argument("--device", default="auto")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--n-trajectories", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-dir", default="/tmp/lsdflow_rxnflow")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--hubs-file", default="")
    p.add_argument("--enum-max-children", type=int, default=4000)
    return p.parse_args()


def main():
    import numpy as np

    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    trainer, beta, clip = _build_trainer(args.config, args.reward_name, device, args.seed, out_dir)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(state["model"])
    if trainer.sampling_model is not trainer.model:
        trainer.sampling_model.load_state_dict(state.get("sampling_model", state["model"]))
    trainer.model.to(device).eval()
    trainer.sampling_model.to(device).eval()
    args._it = state.get("it", "?")
    print(
        f"[rxnflow_worker] loaded checkpoint it={args._it} reward={args.reward_name} "
        f"mode={args.mode} device={device}",
        flush=True,
    )

    try:
        if args.mode == "sample":
            _run_sample(args, trainer, beta, clip, out_dir, device)
        else:
            _run_enumerate(args, trainer, beta, clip, out_dir)
    finally:
        try:
            trainer.env.retro_analyzer.terminate()  # kill the retro ProcessPoolExecutor
        except Exception:
            pass


if __name__ == "__main__":
    main()
