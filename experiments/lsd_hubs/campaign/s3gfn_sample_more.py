#!/usr/bin/env python
"""Sample MORE candidates from an already-trained S3-GFN checkpoint, to give it a full 500-pool.

WHY THIS EXISTS. The workshop claim is that S3-GFN -> retrosynthesis -> SPARROW cannot match
hub-batching's mode count under a 100-reaction budget EVEN WITH every advantage. That claim is
unarguable only if SPARROW is given a pool it could plausibly win from. On sEH it is not: S3-GFN's
2,000-candidate sample clears the 7.0 gate with 154/65/168 distinct molecules for seeds 42/43/44,
against Saturn's ~1,900. Hub-batching reports 82 modes. A 65-molecule pool caps SPARROW at 65 modes
before the comparison starts, so beating it would demonstrate nothing about selection at all.

WHAT THIS IS AND IS NOT. It is NOT extra training and NOT a bigger oracle budget: the policy is
loaded frozen from the checkpoint the budget-faithful run produced, and not a gradient is taken.
It draws more samples from that same trained policy, which is what anyone with a trained model would
do before committing to a synthesis campaign. The training budget (10,048 scored molecules, their
own PMO default) is untouched and must still be reported as such.

WHAT IT DOES COST, and this must be reported beside the result: scoring the extra samples is extra
ORACLE CALLS, in the eval phase rather than the training phase. Reaching 500 above-gate molecules at
seed 43's 3.3% rate needs roughly 15,000 scored candidates -- more than the training budget itself.
That is legitimate (the budget constrains learning, not deployment) but it is not free, and a table
that shows the enlarged pool without the sample count is hiding the price.

WRITES TO A SEPARATE DIRECTORY. The budget-faithful candidates.csv is the artifact every existing
number was computed from and is never touched; the enlarged pool lands under a `_bigsample` run dir
so the two can never be confused in an inventory scan.

Run (s3gfn env, one GPU):
    python experiments/lsd_hubs/campaign/s3gfn_sample_more.py \
        --run-dir /scratch/.../experiments/fixed_reward/s3gfn_seh/seed43 \
        --target-above-gate 500 --gate 7.0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_S3_SRC = _REPO_ROOT / "external" / "s3gfn" / "src"
for _p in (str(_REPO_ROOT), str(_S3_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from omegaconf import OmegaConf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--run-dir", required=True, help="the finished S3-GFN run (holds run_config.yaml + ckpt)"
    )
    ap.add_argument("--target-above-gate", type=int, default=500)
    ap.add_argument("--gate", type=float, default=7.0)
    ap.add_argument("--lower-is-better", action="store_true", help="docking targets")
    ap.add_argument("--max-samples", type=int, default=60000, help="hard cap on unique valid drawn")
    ap.add_argument(
        "--round-size", type=int, default=4000, help="unique valid molecules per scoring round"
    )
    ap.add_argument("--out-dir", default="", help="default: <run-dir>_bigsample")
    a = ap.parse_args()

    run_dir = Path(a.run_dir)
    out_dir = Path(a.out_dir) if a.out_dir else run_dir.parent / f"{run_dir.name}_bigsample"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(run_dir / "run_config.yaml")
    s3_c = OmegaConf.to_container(cfg.get("s3gfn", {}), resolve=True) or {}
    fr_c = OmegaConf.to_container(cfg.get("fixed_reward", {}), resolve=True) or {}
    reward_c = OmegaConf.to_container(cfg.get("reward", {}), resolve=True) or {}
    run_c = OmegaConf.to_container(cfg.get("run", {}), resolve=True) or {}
    seed = int(run_c.get("seed", 42))

    import torch

    from validation.generators.s3gfn.run_s3gfn_fixed import (
        _build_provider,
        _make_configs,
        _sample_pool,
        _stub_unidock_vina,
    )

    # BOTH of these are required before the trainer can be constructed, and the adapter does them in
    # this order for reasons that are not optional. _stub_unidock_vina lets `rxnflow.tasks.
    # unidock_vina` import without openbabel; _load_runtime_dependencies is what actually binds
    # `np` (and friends) into s3gfn.train's module namespace -- upstream imports numpy INSIDE that
    # function (train.py:112), so skipping it fails later and obscurely, at
    # `np.random.seed(configs.seed)` in the trainer's __init__.
    _stub_unidock_vina()
    import s3gfn.train as s3train

    s3train._load_runtime_dependencies()

    # The trainer's constructor reads get_scores; nothing here trains, so a stub is enough and it
    # keeps the real provider out of the trainer's path entirely.
    s3train.get_scores = lambda *args, **kwargs: [0.0]

    device = reward_c.get("rescore_device") or reward_c.get("device", "cpu")
    provider, _scale, rtype = _build_provider(reward_c, device, out_dir)
    print(f"[BIG] provider={rtype} device={device}", flush=True)

    configs: Namespace = _make_configs(s3_c, fr_c, seed, out_dir, s3train.TaskSpec)
    configs.num_training_steps = 0  # belt and braces: never train
    trainer = s3train.SynthSmilesTrainer(logger=None, configs=configs)

    ckpts = sorted(run_dir.glob("*/*_model.pt"))
    final = [c for c in ckpts if "step" not in c.name]
    ckpt_path = final[0] if final else (ckpts[-1] if ckpts else None)
    if ckpt_path is None:
        raise SystemExit(f"[BIG] no checkpoint under {run_dir}")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    trainer.model.load_state_dict(ck["model_state_dict"])
    print(
        f"[BIG] loaded FROZEN policy from {ckpt_path.name} (trained to step {ck.get('step')})",
        flush=True,
    )

    hib = not a.lower_is_better
    passes = (lambda v: v > a.gate) if hib else (lambda v: v < a.gate)

    all_scored: dict = {}  # smiles -> raw score
    above: set = set()
    t0 = time.time()
    rounds = 0
    while len(above) < a.target_above_gate and len(all_scored) < a.max_samples:
        rounds += 1
        want = min(len(all_scored) + a.round_size, a.max_samples)
        pool = _sample_pool(
            trainer,
            n_target=want,
            temperature=float(s3_c.get("eval_sampling_temperature", 1.0)),
            max_batches=int(fr_c.get("max_sample_batches", 4000)),
        )
        fresh = [s for s in pool if s not in all_scored]
        if not fresh:
            print(
                f"[BIG] round {rounds}: sampling saturated — no new molecules. Stopping.",
                flush=True,
            )
            break
        scores = provider.predict(fresh)
        for s, v in zip(fresh, scores):
            all_scored[s] = v
            if v is not None and v == v and passes(float(v)):
                above.add(s)
        print(
            f"[BIG] round {rounds}: {len(all_scored)} scored, {len(above)} above gate "
            f"({100*len(above)/max(1,len(all_scored)):.1f}%)  [{time.time()-t0:.0f}s]",
            flush=True,
        )

    elapsed = time.time() - t0
    ok = len(above) >= a.target_above_gate
    print(
        f"\n[BIG] {'REACHED' if ok else 'SHORT OF'} the target: {len(above)}/{a.target_above_gate} "
        f"above gate from {len(all_scored)} scored in {elapsed/60:.1f} min",
        flush=True,
    )

    pairs = out_dir / "pairs.csv"
    with open(pairs, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        for s, v in sorted(all_scored.items(), key=lambda kv: -kv[1] if hib else kv[1]):
            w.writerow([s, v])
    json.dump(
        {
            "source_run": str(run_dir),
            "checkpoint": ckpt_path.name,
            "trained_to_step": ck.get("step"),
            "gate": a.gate,
            "higher_is_better": hib,
            "n_scored_eval_phase": len(all_scored),
            "n_distinct_above_gate": len(above),
            "target_above_gate": a.target_above_gate,
            "reached_target": ok,
            "sampling_rounds": rounds,
            "elapsed_s": round(elapsed, 1),
            "NOTE": (
                "Extra samples from a FROZEN policy. Training budget unchanged; the scored count "
                "above is EVAL-phase oracle calls and must be reported beside any pool built here."
            ),
        },
        open(out_dir / "bigsample_meta.json", "w"),
        indent=2,
    )
    print(f"[BIG] wrote {pairs} and bigsample_meta.json", flush=True)

    # Emit a standard candidate dataset so the route pipeline can consume this exactly like any
    # other cell. Written under the _bigsample run dir, never over the budget-faithful one.
    cand_dir = out_dir / "fixed_reward" / "candidates"
    ingest = [
        "conda", "run", "--no-capture-output", "-n", "rgfn",
        "python", "scripts/ingest_candidates.py",
        "--pairs", str(pairs), "--out-dir", str(cand_dir),
        "--generator", "s3gfn",
        "--system", str(fr_c.get("system", "seh")),
        "--seed", str(seed),
        "--score-units", str(fr_c.get("score_units", "seh_proxy (higher is better)")),
        "--source", str(out_dir),
    ]  # fmt: skip
    if hib:
        ingest.append("--score-higher-is-better")
    import subprocess

    rc = subprocess.run(ingest, cwd=str(_REPO_ROOT)).returncode
    if rc != 0:
        raise SystemExit(f"[BIG] ingest failed rc={rc}")
    print(f"[BIG] candidates -> {cand_dir}", flush=True)


if __name__ == "__main__":
    main()
