#!/usr/bin/env python
"""Entry point for the FragGFN **fixed-reward** (single-shot) run on sEH.

The fixed-reward counterpart of ``run_fraggfn_al.py``. Instead of the active-learning
loop (fit proxy / train / oracle-label / repeat), this trains Recursion's fragment-GFN
**once** against the frozen pretrained sEH proxy (:class:`SEHFrozenReward`) — no oracle,
no proxy refit — then samples a batch and emits it as a standard candidate dataset. This
is FragGFN's entry in the matched four-way sEH comparison (the non-synthesizable foil).

    conda run -n fraggfn python validation/generators/fraggfn/run_fraggfn_fixed.py \
        --cfg validation/configs/fraggfn_seh_fixed.yaml \
        --root-dir $SCRATCH/rgfn_runs/experiments

Because this env cannot import ``glue``, candidates are emitted by shelling out to
``scripts/ingest_candidates.py`` under the ``rgfn`` env (mirroring how the AL entrant
shells to the oracle bridge). Launch from the repo root.
"""

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path

import torch
from gflownet.config import Config, init_empty
from omegaconf import OmegaConf
from rdkit import Chem

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from validation.generators._trace import TraceWriter, write_timing
from validation.generators.fraggfn.al_loop import FragGFNActiveLearningLoop, LabelStore
from validation.generators.fraggfn.fixed_reward import (
    DockingBridgeReward,
    DRD2FrozenReward,
    SEHFrozenReward,
)
from validation.generators.fraggfn.task import (
    FragGFNTrainer,
    build_constant_temperature,
)


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _canonical(smiles):
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(mol) if mol is not None else None


def _sample_chunked(trainer, it, n_samples, oversample, chunk=128):
    """Sample ``n_samples`` unique valid SMILES from FragGFN in bounded-memory CHUNKS.

    FragGFN's ``_sample_query_batch`` samples all ``n*oversample`` trajectories in ONE
    ``create_training_data_from_own_samples`` call — fine for the AL batch (~800) but it
    OOMs at fixed-reward scale (n_samples=1000 -> 4000 graphs through the transformer at
    once, ~37 GB; job 69564). We sample in small chunks instead, keeping only SMILES and
    dropping each chunk's graphs, so peak memory stays at one chunk. Same per-trajectory
    logic as ``_sample_query_batch`` (graph_to_obj -> canonical -> dedup)."""
    tr = trainer
    tr.model.to(tr.device)
    tr.model.eval()
    budget = int(n_samples * oversample)
    seen, batch = set(), []
    drawn = 0
    with torch.no_grad():
        while len(batch) < n_samples and drawn < budget:
            this = min(chunk, budget - drawn)
            cond_info = tr.task.sample_conditional_information(this, it)
            trajs = tr.algo.create_training_data_from_own_samples(
                tr.model, this, cond_info["encoding"].to(tr.device), random_action_prob=0.0
            )
            drawn += this
            for t in trajs:
                if not t.get("is_valid", True):
                    continue
                try:
                    smi = Chem.MolToSmiles(tr.ctx.graph_to_obj(t["result"]))
                except Exception:
                    continue
                canon = _canonical(smi)
                if canon is None or canon in seen:
                    continue
                seen.add(canon)
                batch.append(canon)
                if len(batch) >= n_samples:
                    break
            del trajs, cond_info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return batch


class _TracedReward:
    """Wraps a frozen reward generator so every evaluation lands in trace.csv.

    WHY WRAP RATHER THAN EDIT EACH PROVIDER. FragGFN has three (sEH proxy, DRD2, docking bridge) and
    the task calls ``reward()`` while the candidate emitter calls ``predict()``. One wrapper catches
    both for all three, and keeps ``fixed_reward.py`` free of trace plumbing -- the same separation
    S3-GFN uses, and the reason its docking traces were correct while Saturn's and REINVENT's were
    not.

    RECORDS THE RAW ORACLE VALUE, never the shaped one. For docking, ``reward()`` is
    ``exp(clip(-vina))`` and ``predict()`` is ``clip(-vina)``; the mode gates are defined on raw Vina
    (ClpP -8.0, Logs/045), so the trace takes ``raw_scores()`` where the provider exposes it. Getting
    this wrong is not hypothetical: nine ClpP traces recorded the shaped value and NOT ONE row cleared
    the gate, so those cells' entire training histories read as empty (fixed 2026-08-28, commit
    736c8e0). Cached per SMILES inside the bridge, so tracing costs no extra docks.

    ``phase`` stays "train" here: unlike S3-GFN's ``evaluate()``, FragGFN's fixed-reward runner has no
    separate evaluation pass, so every call IS training signal.
    """

    def __init__(self, inner, trace):
        self._inner = inner
        self._trace = trace

    def _record(self, smiles):
        if self._trace is None or not smiles:
            return
        try:
            if hasattr(self._inner, "raw_scores"):
                vals = list(self._inner.raw_scores(smiles))
            else:
                vals = list(self._inner.predict(smiles))
            self._trace.add_many(list(smiles), vals)
        except Exception as exc:  # noqa: BLE001 - a trace failure must never kill a run
            print(f"[FGFN-FR] WARNING: trace write failed ({exc})", flush=True)

    def reward(self, smiles):
        self._record(smiles)
        return self._inner.reward(smiles)

    def predict(self, smiles):
        return self._inner.predict(smiles)

    def __getattr__(self, name):  # set_device, fit, raw_scores, ...
        return getattr(self._inner, name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/*_fixed.yaml)"
    )
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed (else cfg.run.seed)")
    ap.add_argument("--root-dir", default=None, help="base run dir (else cfg.run.root_dir)")
    ap.add_argument(
        "--run-dir",
        default=None,
        help="EXACT run dir (stable, no timestamp). Set this to reuse a run dir across 3-day "
        "auto-requeue chain links (campaign Logs/030): the run resumes from its last checkpoint "
        "and appends. Overrides --root-dir + cfg.run.name.",
    )
    ap.add_argument("--device", default=None, help="cpu | cuda (else auto)")
    ap.add_argument(
        "--n-train-steps", type=int, default=None, help="override n_train_steps (smoke)"
    )
    ap.add_argument("--n-samples", type=int, default=None, help="override n_samples (smoke)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    run_c = cfg.get("run", {})
    fr_c = cfg.get("fixed_reward", {})
    gfn_c = cfg.get("gflownet", {})
    reward_c = cfg.get("reward", {})

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    device = args.device or gfn_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    beta = float(fr_c.get("beta", 8))
    n_train_steps = (
        args.n_train_steps
        if args.n_train_steps is not None
        else int(fr_c.get("n_train_steps", 5000))
    )
    n_samples = args.n_samples if args.n_samples is not None else int(fr_c.get("n_samples", 1000))
    # Which fixed reward generator: sEH proxy (default) or DRD2 (RGFN paper's proxies).
    reward_type = reward_c.get("type", "seh_proxy")
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")

    if args.run_dir:  # stable dir for auto-requeue chain links (resume into the same place)
        run_dir = Path(args.run_dir)
    else:
        root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
        run_name = run_c.get("name", "fixed_reward/fraggfn_seh")
        run_dir = root / run_name / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    print(
        f"[FGFN-FR] run_dir={run_dir} device={device} seed={seed} steps={n_train_steps}", flush=True
    )

    # --- fixed reward generator (frozen; no refit): sEH proxy or DRD2 oracle. ------
    if reward_type == "drd2":
        reward = DRD2FrozenReward(
            model_path=reward_c.get("model_path", "oracle/drd2_current.pkl"),
            clip=float(reward_c.get("clip", 10.0)),
        )
    elif reward_type == "docking":
        # Per-step GPU docking across the env boundary (score_batch.py under rgfn).
        reward = DockingBridgeReward(
            oracle=reward_c.get("oracle", "docking_seh"),
            repo_root=str(_REPO_ROOT),
            norm=float(reward_c.get("norm", 1.0)),
            failed_score=float(reward_c.get("failed_score", 0.0)),
            clip=float(reward_c.get("clip", 10.0)),
            oracle_args=dict(reward_c.get("oracle_args", {})),
            workdir=str(run_dir / "reward_bridge"),
        )
    else:
        reward = SEHFrozenReward(
            device=device,
            clip=float(reward_c.get("clip", 10.0)),
            batch_size=int(reward_c.get("batch_size", 128)),
        )
    # Opened BEFORE training so a walltime kill keeps the history; the writer flushes per batch.
    run_t0 = time.time()
    trace = TraceWriter(run_dir / "trace.csv")
    reward = _TracedReward(reward, trace)
    print(
        f"[FGFN-FR] reward={reward_type} system={system} | trace -> {run_dir / 'trace.csv'}",
        flush=True,
    )

    # --- gflownet Config (mirrors run_fraggfn_al.py). -----------------------------
    gcfg = init_empty(Config())
    gcfg.log_dir = str(run_dir / "train")
    gcfg.device = device
    gcfg.seed = seed
    gcfg.overwrite_existing_exp = True
    gcfg.print_every = int(gfn_c.get("print_every", 100))
    gcfg.num_training_steps = n_train_steps
    # STEPS x num_from_policy IS THE ORACLE BUDGET, so pin it from the config rather than inheriting
    # a library default that could change under us. 64 is both the gflownet default and what the
    # authors set in seh_frag.py; 157 x 64 = 10,048 matches REINVENT and S3-GFN exactly.
    gcfg.algo.num_from_policy = int(gfn_c.get("num_from_policy", 64))
    gcfg.algo.max_nodes = int(gfn_c.get("max_nodes", 9))
    gcfg.algo.sampling_tau = float(gfn_c.get("sampling_tau", 0.9))
    gcfg.model.num_emb = int(gfn_c.get("num_emb", 128))
    gcfg.model.num_layers = int(gfn_c.get("num_layers", 4))
    gcfg.opt.learning_rate = float(gfn_c.get("learning_rate", 1e-4))
    # Temperature schedule. Default 'constant' β (matches RGFN's single fixed β — used by
    # the matched four-way runs). 'uniform' enables gflownet's native temperature-annealed
    # exploration (β sampled per-trajectory from [low, high]) — the DRD2 exploration
    # diagnostic (Logs/019): a constant high β can't discover a sparse-reward mode.
    temp_c = cfg.get("temperature", {})
    temp_mode = temp_c.get("mode", "constant")
    temp_high = float(temp_c.get("high", 64.0))
    if temp_mode == "uniform":
        temp_low = float(temp_c.get("low", 0.0))
        gcfg.cond.temperature.sample_dist = "uniform"
        gcfg.cond.temperature.dist_params = [temp_low, temp_high]
        print(
            f"[FGFN-FR] TRAIN temperature=uniform[{temp_low},{temp_high}] (annealed exploration)",
            flush=True,
        )
    else:
        build_constant_temperature(gcfg, beta)  # fixed β, matches RGFN
        print(f"[FGFN-FR] TRAIN temperature=constant beta={beta}", flush=True)

    trainer = FragGFNTrainer(gcfg, proxy=reward)

    # Reuse the AL loop's train/sample internals (no duplication) without running the
    # AL algorithm: we never call loop.run(), only _train_steps() and
    # _sample_query_batch(). dataset/bridge_cmd are unused here.
    loop = FragGFNActiveLearningLoop(
        trainer=trainer,
        proxy=reward,
        dataset=LabelStore(lower_is_better=False),
        bridge_cmd=[],
        run_dir=str(run_dir),
        n_rounds=1,
        n_train_steps=n_train_steps,
        query_batch_size=n_samples,
        sample_oversample=float(fr_c.get("sample_oversample", 4.0)),
        top_k=int(fr_c.get("top_k", 100)),
        system=system,
        seed=seed,
        oracle_higher_is_better=True,  # sEH/DRD2 rewards are both higher-is-better
    )

    out_dir = run_dir / "fixed_reward"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. train the fragment-GFN ONCE. If a prior 3-day auto-requeue chain link left a
    #    checkpoint (campaign Logs/030), resume from it and train only the REMAINING steps to
    #    reach n_train_steps total (loop._train_steps adds steps from loop._it+1).
    loop.load_checkpoint()
    remaining = n_train_steps - loop._it
    if remaining > 0:
        print(
            f"[FGFN-FR] training {remaining} steps (of {n_train_steps}; resumed at {loop._it}) "
            f"against frozen {reward_type} reward (beta={beta})",
            flush=True,
        )
        _t0 = time.time()
        loop._train_steps(remaining)
        train_s = time.time() - _t0
    else:
        train_s = 0.0
        print(
            f"[FGFN-FR] already trained {loop._it} >= {n_train_steps} steps; skipping to sampling.",
            flush=True,
        )

    # For a uniform-temperature (annealed) run, sample the final batch from the
    # EXPLOITATION policy: condition at a fixed high β via the *uniform* branch
    # (dist_params=[sb, sb]) so the thermometer encoding matches what the model saw for
    # high-β trajectories. (The 'constant' branch encodes β as zeros, which a
    # uniform-trained model reads as β≈0 = explore — the opposite of what we want.)
    if temp_mode == "uniform":
        from gflownet.utils.conditioning import TemperatureConditional

        sb = float(temp_c.get("sample_beta", temp_high))
        trainer.cfg.cond.temperature.sample_dist = "uniform"
        trainer.cfg.cond.temperature.dist_params = [sb, sb]
        trainer.task.temperature_conditional = TemperatureConditional(trainer.cfg)
        print(f"[FGFN-FR] final sampling at fixed beta={sb} (exploitation policy)", flush=True)

    # 2. sample a batch of unique valid molecules (chunked to bound GPU memory; a single
    #    n_samples*oversample sampling call OOMs at this scale — job 69564).
    _t0 = time.time()
    batch = _sample_chunked(trainer, loop._it, n_samples, float(fr_c.get("sample_oversample", 4.0)))
    sample_s = time.time() - _t0
    print(f"[FGFN-FR] sampled {len(batch)} unique valid candidates", flush=True)

    # 3. score them with the reward generator itself (its VALUE = the score column,
    #    higher-is-better). Docking additionally exposes the raw Vina/dvina, kept as a
    #    provenance column (matches RGFN's OracleRewardProxy 'raw_score' component).
    scores = reward.predict(batch)
    raws = reward.raw_scores(batch) if hasattr(reward, "raw_scores") else None

    # 4. write pairs.csv, then emit the standard candidate dataset via the rgfn env.
    pairs_path = out_dir / "pairs.csv"
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"] + (["raw_score"] if raws is not None else []))
        for i, (smi, sc) in enumerate(zip(batch, scores)):
            w.writerow([smi, sc] + ([raws[i]] if raws is not None else []))

    ingest_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "rgfn",
        "python",
        "scripts/ingest_candidates.py",
        "--pairs",
        str(pairs_path),
        "--out-dir",
        str(out_dir / "candidates"),
        "--generator",
        "fraggfn",
        "--reward-name",
        reward_name,
        "--system",
        system,
        "--seed",
        str(seed),
        "--score-higher-is-better",
        "--score-units",
        score_units,
        "--source",
        str(run_dir),
    ]
    print(f"[FGFN-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True)

    trainer.terminate()
    # Close the trace and record where the wall-clock went, so FragGFN can appear in the end-to-end
    # compute comparison (training + pool + retrosynthesis + selection) alongside the other five.
    trace.close()
    n_scored, n_distinct = trace.n_scored, trace.n_distinct
    write_timing(
        run_dir / "timing.json",
        {"train": round(train_s, 1), "sample": round(sample_s, 1)},
        total_s=round(time.time() - run_t0, 1),
    )
    print(
        f"[FGFN-FR] trace closed: {n_scored} scored / {n_distinct} distinct "
        f"-> {run_dir / 'trace.csv'}; timing -> timing.json",
        flush=True,
    )
    print(f"[FGFN-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
