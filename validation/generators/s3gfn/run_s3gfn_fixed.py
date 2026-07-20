#!/usr/bin/env python
"""Entry point for the S3-GFN **fixed-reward** (single-shot) run — the MARQUEE non-reaction
baseline for the LSD-Flow library-efficiency benchmark (`[kim2026s3gfn]`, T3.1).

The fixed-reward counterpart of the RxnFlow/FragGFN/SCENT entrants: train S3-GFN's SMILES
GFlowNet (GP-MolFormer + soft-synthesizability contrastive buffers) **once** against a FROZEN
reward provider, then sample a pool and emit a standard candidate dataset. Unlike the
reaction-GFN entrants, S3-GFN carries **no shared-route structure** -> emitted with
``has_route=0`` (routes are recovered post-hoc via AiZynth->SPARROW). It is the foil for the
"reaction MDP is necessary for library economics" headline.

CUSTOM MODULAR WIRING (the point of this adapter): S3-GFN must optimize the SAME reward the
reaction-GFN entrants do. We inject a swappable reward provider
(:mod:`validation.generators.s3gfn.fixed_reward`) in place of S3-GFN's native module-global
``get_scores`` (train.py:377). For ``seh_proxy`` the provider IS the Bengio-2021 MPNN our
benchmark + S3-GFN both use, and the injected training reward matches native ``mol2seh`` exactly
(``clip(raw/8, 1e-4, 100)``) -> this run is the VERIFICATION case (should reproduce S3-GFN's
published sEH result, proving the injection is faithful). ``drd2`` / ``docking`` slot in through
the same seam (same provider interface, injected identically) with no S3-GFN-side changes.

Two env-isolation fixes keep the s3gfn env minimal (docking is bridged, never rxnflow's Vina):
  * ``rxnflow.tasks.unidock_vina`` is STUBBED in ``sys.modules`` before S3-GFN loads its runtime
    deps (that module needs openbabel, which we don't install; it is docking-only and unused here).
  * ``external/s3gfn/src`` is prepended to ``sys.path`` so both ``s3gfn`` and its vendored
    ``rxnflow`` (retro-synthesizability analyzer) import from source.

Run (from repo root, s3gfn env; candidate emission shells to the rgfn env):
    conda run -n s3gfn python validation/generators/s3gfn/run_s3gfn_fixed.py \
        --cfg validation/configs/s3gfn_seh_fixed.yaml \
        --root-dir $SCRATCH/rgfn_runs/experiments
"""

import argparse
import csv
import subprocess
import sys
import time
import types
from argparse import Namespace
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
_S3GFN_SRC = _REPO_ROOT / "external" / "s3gfn" / "src"
for _p in (str(_REPO_ROOT), str(_S3GFN_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validation.generators.s3gfn.fixed_reward import (  # noqa: E402
    DockingBridgeReward,
    DRD2FrozenReward,
    SEHFrozenReward,
)


def _timestamp() -> str:
    import datetime

    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _stub_unidock_vina() -> None:
    """Register a stub ``rxnflow.tasks.unidock_vina`` so S3-GFN's ``_load_runtime_dependencies``
    (``from rxnflow.tasks.unidock_vina import VinaReward``) succeeds without openbabel. VinaReward
    is docking-only and never instantiated here — our docking goes through DockingBridgeReward."""
    if "rxnflow.tasks.unidock_vina" in sys.modules:
        return
    stub = types.ModuleType("rxnflow.tasks.unidock_vina")

    class _StubVinaReward:  # pragma: no cover - never constructed
        def __init__(self, *a, **k):
            raise RuntimeError(
                "rxnflow VinaReward is stubbed in the S3-GFN adapter; docking uses "
                "DockingBridgeReward (score_batch.py bridge), not rxnflow's Vina."
            )

    stub.VinaReward = _StubVinaReward
    sys.modules["rxnflow.tasks.unidock_vina"] = stub


class _OracleAdapter:
    """A ``get_scores(smiles, mode, vina, hist)``-compatible callable backed by a swappable reward
    provider. Returns S3-GFN's TRAINING reward as a flat list (shaped ``(-1, num_metric=1)`` by the
    trainer). ``provider.predict`` yields the RAW oracle value (higher-is-better; ``nan`` invalid);
    this maps it to the training-reward scale.

    For ``seh_proxy`` the mapping is ``clip(raw/8, lo, hi)`` — bit-identical to native ``mol2seh``
    (the verification guarantee). Other oracles use ``reward_scale=1.0`` (the provider already
    returns a training-suitable value); tune per-oracle here as they are wired in."""

    def __init__(self, provider, reward_scale: float, lo: float = 1e-4, hi: float = 100.0):
        self.provider = provider
        self.reward_scale = float(reward_scale)
        self.lo = float(lo)
        self.hi = float(hi)

    def __call__(self, smiles, mode=None, vina=None, hist=None, **_kw):
        out = []
        for v in self.provider.predict(list(smiles)):
            if v != v:  # NaN -> invalid molecule
                out.append(self.lo)
            else:
                out.append(float(min(max(v * self.reward_scale, self.lo), self.hi)))
        return out


def _build_provider(reward_c, device: str, run_dir: Path):
    """Frozen reward provider (no refit). seh_proxy (implemented) | drd2 | docking (seams)."""
    rtype = reward_c.get("type", "seh_proxy")
    if rtype == "drd2":
        provider = DRD2FrozenReward(
            model_path=reward_c.get("model_path", "oracle/drd2_current.pkl"),
            clip=float(reward_c.get("clip", 10.0)),
        )
        scale = 1.0  # DRD2 prob in [0,1]; use as-is
    elif rtype == "docking":
        provider = DockingBridgeReward(
            oracle=reward_c.get("oracle", "docking_6td3_gpu"),
            repo_root=str(_REPO_ROOT),
            norm=float(reward_c.get("norm", 1.0)),
            failed_score=float(reward_c.get("failed_score", 0.0)),
            clip=float(reward_c.get("clip", 10.0)),
            oracle_args=dict(reward_c.get("oracle_args", {})),
            workdir=str(run_dir / "reward_bridge"),
        )
        scale = 1.0  # provider.predict already returns the GFN VALUE (clip(-raw/norm,0,inf))
    else:  # seh_proxy — the verification oracle
        provider = SEHFrozenReward(
            device=device,
            clip=float(reward_c.get("clip", 10.0)),
            batch_size=int(reward_c.get("batch_size", 128)),
        )
        scale = 1.0 / 8.0  # match native mol2seh: reward = clip(raw/8, 1e-4, 100)
    return provider, scale, rtype


def _make_configs(s3_c, fr_c, seed: int, run_dir: Path, TaskSpec) -> Namespace:
    """Build the ``configs`` Namespace the S3-GFN trainer reads. The task is ALWAYS a scalar,
    non-docking spec (num_metrics=1, receptor=None) so the trainer's native Vina path is never
    taken — the real oracle is the injected provider (works uniformly for sEH/DRD2/docking)."""
    task = TaskSpec("SEH", None, "s3gfn_seh", 1, "lsdflow-s3gfn", "seh")
    return Namespace(
        task=task,
        num_training_steps=int(fr_c.get("n_train_steps", 5000)),
        num_warmup_steps=int(s3_c.get("num_warmup_steps", 100)),
        batch_size=int(s3_c.get("batch_size", 64)),
        replay_batch_size=int(s3_c.get("replay_batch_size", 64)),
        buffer_size=int(s3_c.get("buffer_size", 6400)),
        learning_rate=float(s3_c.get("learning_rate", 1e-4)),
        log_z_learning_rate=float(s3_c.get("log_z_learning_rate", 1e-3)),
        beta=float(s3_c.get("beta", 50.0)),
        sampling_temperature=float(s3_c.get("sampling_temperature", 1.0)),
        eval_sampling_temperature=float(s3_c.get("eval_sampling_temperature", 1.0)),
        eval_every=int(s3_c.get("eval_every", 100000)),
        eval_samples=int(s3_c.get("eval_samples", 1000)),
        output_dir=str(run_dir),
        save_periodic_every=int(s3_c.get("save_periodic_every", 0)),
        training_mode=s3_c.get("training_mode", "s3gfn"),
        use_retrosynthesis=bool(s3_c.get("use_retrosynthesis", True)),
        retro_env=s3_c.get("retro_env", "zincfrag"),
        retro_steps=int(s3_c.get("retro_steps", 3)),
        sa_threshold=float(s3_c.get("sa_threshold", 4.0)),
        aux_coefficient=float(s3_c.get("aux_coefficient", 0.0001)),
        catalog=s3_c.get("catalog", "") or "",
        property_rule=s3_c.get("property_rule", "none"),
        wandb_mode="disabled",
        run_name=s3_c.get("run_name", "s3gfn_seh"),
        seed=int(seed),
    )


def _sample_pool(
    trainer, n_target: int, temperature: float, max_batches: int, per_batch: int = 128
):
    """Sample the trained model until ``n_target`` unique valid canonical SMILES (or the batch cap).
    Mirrors S3-GFN's own eval sampling (train.py:664-678)."""
    import torch
    from rdkit import Chem

    model, tok = trainer.model, trainer.tokenizer
    model.eval()
    seen: dict = {}
    nb = 0
    with torch.no_grad():
        while len(seen) < n_target and nb < max_batches:
            nb += 1
            seqs = model.generate(
                do_sample=True,
                max_length=trainer.max_length,
                num_return_sequences=per_batch,
                temperature=temperature,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                use_cache=True,
            )
            for s in tok.batch_decode(seqs, skip_special_tokens=True):
                m = Chem.MolFromSmiles(s) if s else None
                if m is None:
                    continue
                canon = Chem.MolToSmiles(m)
                if canon not in seen:
                    seen[canon] = None
                    if len(seen) >= n_target:
                        break
            if nb % 20 == 0:
                print(
                    f"[S3-FR] sampling: {len(seen)}/{n_target} unique valid (batch {nb})",
                    flush=True,
                )
    return list(seen.keys())[:n_target]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/s3gfn_*_fixed.yaml)"
    )
    ap.add_argument("--seed", type=int, default=None, help="override RNG seed (else cfg.run.seed)")
    ap.add_argument("--root-dir", default=None, help="base run dir (else cfg.run.root_dir)")
    ap.add_argument("--run-dir", default=None, help="EXACT run dir (stable, no timestamp)")
    ap.add_argument("--device", default=None, help="cpu | cuda (else auto)")
    ap.add_argument(
        "--n-train-steps", type=int, default=None, help="override n_train_steps (smoke)"
    )
    ap.add_argument("--n-samples", type=int, default=None, help="override n_samples (smoke)")
    ap.add_argument(
        "--retro-env", default=None, help="override retro_env (smoke uses zincfrag_10k)"
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    run_c = cfg.get("run", {})
    fr_c = cfg.get("fixed_reward", {})
    reward_c = cfg.get("reward", {})
    s3_c = cfg.get("s3gfn", {})

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    if args.n_train_steps is not None:
        fr_c["n_train_steps"] = args.n_train_steps
    if args.n_samples is not None:
        fr_c["n_samples"] = args.n_samples
    if args.retro_env is not None:
        s3_c["retro_env"] = args.retro_env

    n_samples = int(fr_c.get("n_samples", 2000))
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
        run_dir = root / run_c.get("name", "fixed_reward/s3gfn_seh") / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")

    # --- load S3-GFN runtime deps (with the unidock_vina stub), then inject our reward. --------
    _stub_unidock_vina()
    import s3gfn.train as s3train

    s3train._load_runtime_dependencies()

    import torch

    device = args.device or s3_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[S3-FR] run_dir={run_dir} device={device} seed={seed} "
        f"steps={fr_c.get('n_train_steps')} retro_env={s3_c.get('retro_env')}",
        flush=True,
    )

    provider, scale, rtype = _build_provider(reward_c, device, run_dir)
    s3train.get_scores = _OracleAdapter(provider, reward_scale=scale)  # <-- the injection
    print(f"[S3-FR] injected get_scores <- provider={rtype} (reward_scale={scale:g})", flush=True)

    configs = _make_configs(s3_c, fr_c, seed, run_dir, s3train.TaskSpec)

    # --- train the SMILES GFlowNet ONCE against the frozen reward. -----------------------------
    trainer = s3train.SynthSmilesTrainer(logger=None, configs=configs)
    t0 = time.time()
    trainer.train()
    print(f"[S3-FR] training done in {time.time()-t0:.1f}s", flush=True)

    # --- sample the candidate pool + re-score on the RAW provider value (the mode-gate scale). --
    t0 = time.time()
    pool = _sample_pool(
        trainer,
        n_target=n_samples,
        temperature=float(s3_c.get("eval_sampling_temperature", 1.0)),
        max_batches=int(fr_c.get("max_sample_batches", 4000)),
    )
    print(
        f"[S3-FR] sampled {len(pool)} unique valid candidates in {time.time()-t0:.1f}s", flush=True
    )

    scores = provider.predict(pool)  # RAW value = the 'score' column (higher is better)
    raws = provider.raw_scores(pool) if hasattr(provider, "raw_scores") else None

    # sanity: S3-GFN's own retro synthesizability rate of the emitted pool (T3.1 acceptance signal;
    # the authoritative synthesizable gate is post-hoc AiZynth-routability, but this is a fast check).
    try:
        synth = trainer.synthesizability_evaluator.score_batch(pool)
        synth_rate = sum(synth) / len(synth) if synth else 0.0
        print(
            f"[S3-FR] pool internal retro-synth rate: {synth_rate:.3f} ({sum(synth):.0f}/{len(pool)})",
            flush=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[S3-FR] (retro synth-rate check skipped: {type(exc).__name__}: {exc})", flush=True)

    # --- write pairs.csv, then emit the standard candidate dataset (has_route=0). --------------
    out_dir = run_dir / "fixed_reward"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.csv"
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"] + (["raw_score"] if raws is not None else []))
        for i, (smi, sc) in enumerate(zip(pool, scores)):
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
        "s3gfn",
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
    ]  # NOTE: no --routes -> has_route=0 (S3-GFN carries no shared-route structure)
    # ingest_candidates.py imports `glue` (-> rgfn -> dgl/graphbolt), which needs the rgfn env's
    # CUDA libs on LD_LIBRARY_PATH. We must NOT set those in THIS (s3gfn, torch cu121) process, so
    # the caller passes them via RGFN_INGEST_LD_LIBRARY_PATH for the ingest child only (the submit
    # wrapper sets it from `module load cuda/11.8.0` on Balam compute, or the rgfn torch nvlibs on a
    # login node — same value rgfn-smoke-env.sh computes).
    import os

    ingest_env = os.environ.copy()
    _ingest_ld = os.environ.get("RGFN_INGEST_LD_LIBRARY_PATH")
    if _ingest_ld:
        ingest_env["LD_LIBRARY_PATH"] = _ingest_ld
    print(f"[S3-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True, cwd=str(_REPO_ROOT), env=ingest_env)

    print(f"[S3-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
