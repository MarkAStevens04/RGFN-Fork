#!/usr/bin/env python
"""STAGE 2 — keep sampling a trained generator until its pool holds N diverse modes.

WHY THIS STAGE EXISTS (decided 2026-08-28). Until now every competitor cell took a FIXED 2,000-molecule
post-training sample, and a cell that could not reach 500 modes from those 2,000 was recorded as
"pool-limited". That conflated two completely different things:

  * the generator genuinely cannot produce 500 mutually-dissimilar molecules above the gate, and
  * we simply did not ask it for enough molecules.

Measured on the 2,000-sample pools: REINVENT reaches ~500 modes from 587 eligible molecules, while
Saturn puts 1,890-1,996 of 2,000 over the gate and still yields only 164-390 modes, and S3-GFN gets
28-233. Those are different failures wearing the same label. Sampling to a FIXED MODE TARGET makes
the distinction measurable: a cell either reaches the target (and we report what that cost) or it
stalls (and "cannot" is then a finding, not an artefact of our sample size).

WHY 500 AND NOT ~150. The primary readout is modes at a fixed 100-REACTION budget, and across 54
priced cells that budget buys a median of 56 modes and never more than 100 -- so ~150 modes would
suffice for the headline. 500 is deliberate anyway: SPARROW's advantage IS having alternatives to
choose between, since cheap routes come from shared intermediates. Handing it a bare 150-molecule
diverse pool would quietly handicap the competitor's strongest arm. The extra cost buys the
competitor its best case, which is the point.

THE ASYMMETRY THIS CREATES, AND WHY WE REPORT RATHER THAN MATCH IT. Only the SPARROW pipeline needs a
500-mode pool; hub-batching does not. So Stage 2's oracle calls are a cost one arm pays and the other
does not, and matching them would mean replacing the fixed-REACTION headline with a fixed-MODE one --
a readout this project explicitly demoted to secondary. Report instead, and note the bias is NOT
directional: the calls make the competitor look expensive, while the resulting pool makes it stronger.

WHAT IS COUNTED. Only DISTINCT molecules count against the cap, because a chemist who sees a
duplicate does not re-run the assay. Repeats are recorded separately -- they are both an interesting
property of the generator and the sharpest stall signal available (S3-GFN re-emitted 2,979 of 12,048,
25%, during training; Saturn refuses to score repeats at all and logged 4,753).

STOP REASONS, all three reported explicitly because they support different claims:
  target-reached  -- the pool holds ``--target-modes``. Quote freely.
  stalled         -- a round added fewer than ``--stall-modes`` new modes. This is a claim ABOUT THE
                     GENERATOR: it has run out of distinct chemistry above the gate.
  cap             -- hit ``--max-scored``. A claim about OUR budget, not the generator. Never present
                     a capped cell as evidence the generator cannot do better.

SYNFORMER IS EXEMPT AND THAT IS A FINDING. Its candidates are a slice of an accumulated genetic-
algorithm population (``pool = have[:n_samples]``), not draws from a sampler, so there is nothing to
ask for more of -- getting further molecules means running more GA generations, which is more
TRAINING. Generate-then-sample methods can be upsampled and a GA cannot; that is an architectural
difference, so SynFormer cells stay at whatever their run produced and are reported pool-limited.

TAU IS PASSED EXPLICITLY, NEVER DEFAULTED. ``mode_representatives`` defaults to 0.7 (the RGFN paper's
threshold) while this campaign uses 0.5 everywhere. Taking the default here would silently change
every mode count in the benchmark, so ``--cutoff`` has no default that could be wrong by omission.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Per-target oracle-call ceilings. Deliberately NOT derived from hub-batching's own call counts:
# those span 0 to 315,539 across our arms (entry 044), so "2x hub-batching" would make the
# competitor's allowance depend on a knob of OUR method -- and 2x free-frag on ClpP is ~894 GPU-hours
# per cell at 5.1 s/molecule. These are absolute, generous, and affordable.
DEFAULT_CAP = {"seh": 50_000, "drd2": 50_000, "clpp": 20_000}


def gate_for(target: str):
    """(gate, score_column, higher_is_better), read from the ONE source of truth.

    NEVER HARDCODE A GATE HERE. Every target's bar is the score at which 5% of that target's
    property-matched decoys pass, settled 2026-08-21 and living in
    ``experiments/lsd_hubs/matrix16/targets.py``. This file previously carried its own copy --
    seh 7.0, drd2 0.5, clpp -8.0 -- which are the PRE-STANDARD values, so every mode count it
    produced was on a bar nothing else in the campaign uses any more. The current bars are
    5.68 / 0.345 / -9.1, and the ClpP move matters most: -8.0 admitted 23% of decoys against the 5%
    the standard fixes, so a ClpP "mode" was about 6x more contaminated than a DRD2 one.
    sparrow_select_frontier, mode_saturation and build_s3gfn_pools all made --gate required=True for
    exactly this reason; importing is the equivalent guarantee for a script that resolves its own.

    The bars are empirical grid points -- DO NOT ROUND them, that breaks the exact-FPR property.

    score_column follows the direction, matching candidates.csv: docking gates are defined on raw
    Vina (lower is better) and surrogates on the training value.
    """
    import sys as _sys

    mtx = _REPO_ROOT / "experiments" / "lsd_hubs" / "matrix16"
    if str(mtx) not in _sys.path:
        _sys.path.insert(0, str(mtx))
    from targets import get_target

    t = get_target(target)
    hib = bool(t.higher_is_better)
    return float(t.mode_reward_threshold), ("score" if hib else "raw_score"), hib


def load_scored(path: Path, score_column: str) -> Dict[str, float]:
    """{canonical smiles: gated value} from a candidates.csv, keeping the BEST value per molecule."""
    best: Dict[str, float] = {}
    if not path.exists():
        return best
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            smi = (row.get("smiles") or "").strip()
            raw = row.get(score_column)
            if raw is None:
                raw = row.get("score", row.get("reward"))
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not smi:
                continue
            if smi not in best:
                best[smi] = val
            else:
                best[smi] = max(best[smi], val) if val == val else best[smi]
    return best


def load_trace(path: Path, hib: bool) -> Dict[str, float]:
    """{smiles: best raw oracle value} from trace.csv -- the FREE starting pool.

    HARVEST THIS BEFORE SAMPLING ANYTHING. trace.csv records every molecule the oracle ever scored
    during training, with its value, so those molecules are already paid for. The previous pipeline
    threw them away and took a fresh 2,000-molecule sample, which was an arbitrary choice rather than
    a principled one -- a chemist would obviously use a molecule the model produced and the assay
    already measured. Measured 2026-08-28 at tau=0.5, modes from the pool vs from the trace:

        reinvent/seh/s42    494  ->   560       saturn/seh/s42     338  ->  1251
        s3gfn/seh/s42        89  ->   133       s3gfn/seh/s43       28  ->    50
        reinvent_clpp/s42   642  ->  2589       saturn_clpp/s42    386  ->  3177

    So several cells clear a 500-mode target from the training history alone, at zero additional
    oracle calls. On ClpP that is ~28 GPU-hours per cell saved.

    ONE THING TO REPORT, NOT HIDE: these molecules come from earlier, weaker policies, so the reward
    profile of the selected modes shifts relative to a final-policy-only pool. Modes are chosen
    best-reward-first, so weak early molecules are simply not picked and the risk is bounded -- but
    the shift is real and belongs in the violin plot, and the final-policy-only count should stay
    available for anyone who wants it.

    NaN is skipped: for docking that is a clip-censored or failed dock, not a measurement.
    """
    best: Dict[str, float] = {}
    # READ trace.csv AND EVERY ROTATED SIBLING. A runner re-invoked by this very script rotates the
    # live trace to trace.csv.N (see TraceWriter), so the full history across rounds is the union.
    # Reading only trace.csv would silently shrink the free pool with each round -- which is exactly
    # how s3gfn_seh/seed43's history was lost before the rotation existed.
    paths = [path] + sorted(path.parent.glob(path.name + ".*"))
    for pth in paths:
        if not pth.exists():
            continue
        with open(pth, newline="") as fh:
            for row in csv.DictReader(fh):
                smi = (row.get("smiles") or "").strip()
                try:
                    val = float(row.get("raw_score"))
                except (TypeError, ValueError):
                    continue
                if not smi or val != val:
                    continue
                if smi not in best or ((val > best[smi]) if hib else (val < best[smi])):
                    best[smi] = val
    return best


def count_modes(
    scored: Dict[str, float], gate: float, hib: bool, cutoff: float, cap: Optional[int] = None
) -> List[str]:
    """Mode representatives, using the ONE canonical implementation in the repo.

    Imported rather than reimplemented on purpose: nine campaign scripts touch mode logic and a
    second copy is how the metric drifts (see the shared-tree note in Logs/062).
    """
    from validation.lsdflow.metrics.diversity import ecfp, mode_representatives

    smis = list(scored)
    rews = [scored[s] for s in smis]
    fps = [ecfp(s) for s in smis]
    idx = mode_representatives(
        smis,
        rews,
        higher_is_better=hib,
        reward_threshold=gate,
        similarity_threshold=cutoff,  # NEVER the 0.7 default; see the module docstring
        max_modes=cap,
        fps=fps,
    )
    return [smis[i] for i in idx]


def sample_round(
    runner: str, env: str, cfg: str, seed: int, run_dir: Path, n: int, repo_root: Path
) -> Tuple[int, float]:
    """Ask the generator for a pool of ``n`` distinct molecules. Returns (rc, seconds).

    Runs the generator's OWN runner in its OWN conda env, which is how every other stage invokes
    these adapters. The runner resumes from the trained checkpoint and skips training, so this costs
    sampling + scoring only.
    """
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env,
        "python",
        runner,
        "--cfg",
        cfg,
        "--seed",
        str(seed),
        "--run-dir",
        str(run_dir),
        "--n-samples",
        str(n),
    ]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(repo_root))
    return proc.returncode, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--generator",
        required=True,
        choices=["reinvent", "saturn", "s3gfn", "tango", "fraggfn"],
        help="fraggfn added 2026-08-28 when it moved into the competitor block: it is a "
        "fragment-based GFlowNet with a real sampler, so `--n-samples N` on a trained checkpoint "
        "draws more molecules exactly as reinvent/saturn/s3gfn do. synformer stays absent because "
        "its candidates are a slice of an accumulated GA population (`pool = have[:n_samples]`), "
        "not draws from a sampler -- more molecules there means more GA generations, which is more "
        "TRAINING. That is an architectural difference between generate-then-sample methods and a "
        "GA, so synformer cells are reported pool-limited rather than upsampled.",
    )
    ap.add_argument("--target", required=True, choices=sorted(DEFAULT_CAP))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument(
        "--run-dir", required=True, help="the generator's run dir (holds the checkpoint)"
    )
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--env", required=True, help="conda env for this generator's runner")
    ap.add_argument("--runner", required=True, help="path to run_<gen>_fixed.py")
    ap.add_argument("--target-modes", type=int, default=500)
    ap.add_argument(
        "--cutoff",
        type=float,
        required=True,
        help="tau for sphere exclusion. REQUIRED -- the canonical helper defaults to "
        "0.7 and this campaign uses 0.5; a default here could be silently wrong.",
    )
    ap.add_argument(
        "--round-size",
        type=int,
        default=2000,
        help="distinct molecules to request per round. Rounds exist so the stall "
        "detector can stop paying: on ClpP a wasted round is GPU-hours.",
    )
    ap.add_argument(
        "--stall-modes",
        type=int,
        default=10,
        help="stop if a round adds fewer than this many NEW modes",
    )
    ap.add_argument(
        "--max-scored", type=int, default=0, help="0 = per-target default (%s)" % DEFAULT_CAP
    )
    ap.add_argument("--out", default="", help="where to write upsample_log.json (default: run-dir)")
    a = ap.parse_args()

    gate, col, hib = gate_for(a.target)
    cap = a.max_scored or DEFAULT_CAP[a.target]
    run_dir = Path(a.run_dir)
    cand = run_dir / "fixed_reward" / "candidates" / "candidates.csv"
    out = Path(a.out) if a.out else run_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"[upsample] {a.generator}/{a.target}/seed{a.seed}", flush=True)
    print(
        f"[upsample] target {a.target_modes} modes | gate {col}{'>' if hib else '<'}{gate} "
        f"| tau {a.cutoff} | cap {cap} distinct | round {a.round_size}",
        flush=True,
    )

    # --- SNAPSHOT THE BUDGET-FAITHFUL RUN BEFORE TOUCHING IT -------------------------------------
    # Re-invoking the runner OVERWRITES the cell's outputs: candidates.csv, pairs.csv, timing.json,
    # run_config.yaml. Those are the artifacts of the authors'-default-budget run -- the fairness
    # anchor this whole benchmark rests on -- and post-training sampling is stochastic, so once
    # overwritten the original 2,000-molecule pool cannot be re-derived by re-running.
    #
    # This is the SECOND time the same root cause bit: the trace fix (TraceWriter rotation) covered
    # trace.csv only, because I looked at the file I happened to be reading rather than enumerating
    # everything a runner writes. s3gfn_seh/seed42's candidates.csv went 2,000 -> 20,000 rows before
    # this snapshot existed. Its derived pools (_N154 / _N89) and their routes survive, so published
    # numbers are intact, but the cell can no longer be reproduced from its own candidates file.
    #
    # One snapshot, taken once, never overwritten: if fixed_reward.budget_faithful/ already exists a
    # previous Stage-2 run made it and it is the ORIGINAL -- re-copying would capture upsampled data
    # and destroy the very thing being preserved.
    _fr = run_dir / "fixed_reward"
    _snap = run_dir / "fixed_reward.budget_faithful"
    if _fr.is_dir() and not _snap.exists():
        import shutil

        shutil.copytree(_fr, _snap)
        _n = (
            sum(1 for _ in open(_snap / "candidates" / "candidates.csv")) - 1
            if (_snap / "candidates" / "candidates.csv").exists()
            else 0
        )
        print(
            f"[upsample] snapshot: budget-faithful run preserved at {_snap.name} "
            f"({_n} candidates)",
            flush=True,
        )
    elif _snap.exists():
        print(f"[upsample] snapshot: {_snap.name} already present — left untouched", flush=True)

    # --- free pool first: everything the oracle already scored during training -------------------
    free = load_trace(run_dir / "trace.csv", hib)
    free_modes = count_modes(free, gate, hib, a.cutoff, cap=a.target_modes) if free else []
    print(
        f"[upsample] training history: {len(free)} scored molecules -> {len(free_modes)} modes "
        f"(FREE — already paid for)",
        flush=True,
    )

    rounds: List[dict] = []
    asked = a.round_size
    reason = "target-reached" if len(free_modes) >= a.target_modes else "cap"
    prev_modes = len(free_modes)
    # Sampling is skipped entirely when the training history already meets the target -- the whole
    # point of harvesting it. On ClpP that is the difference between 0 and 28 GPU-hours per cell.
    while len(free_modes) < a.target_modes:
        scored_before = load_scored(cand, col)
        rc, secs = sample_round(a.runner, a.env, a.cfg, a.seed, run_dir, asked, _REPO_ROOT)
        free = load_trace(run_dir / "trace.csv", hib)  # now includes this round's rotated history
        scored = dict(free)
        scored.update(load_scored(cand, col))
        if rc != 0:
            # A FAILED ROUND IS NEVER A STALL. Previously this only tripped when the disk was ALSO
            # empty, so a runner that crashed on startup left the stale candidates.csv in place, the
            # round added 0 modes, and the next check declared `stalled` -- a claim about the
            # GENERATOR -- when the truth was that our compute nodes cannot reach huggingface.co.
            # Measured 2026-08-28 on s3gfn_seh/seed42: two rounds rc=1, candidates.csv untouched
            # since Aug 21, and the log confidently reported "STOP=stalled modes=169/500".
            # Stop reasons are read as findings, so one that can be produced by an environment
            # failure is worse than no stop reason at all.
            reason = "sampling-failed"
            print(
                f"[upsample] round {len(rounds) + 1} FAILED (runner rc={rc}). Not a stall — "
                "refusing to report a generator property from a failed round.",
                flush=True,
            )
            rounds.append(
                dict(
                    asked=asked,
                    distinct=len(scored),
                    eligible=0,
                    modes=prev_modes,
                    modes_added=0,
                    seconds=round(secs, 1),
                    rc=rc,
                )
            )
            break

        n_distinct = len(scored)
        eligible = sum(1 for v in scored.values() if ((v > gate) if hib else (v < gate)))
        modes = count_modes(scored, gate, hib, a.cutoff, cap=a.target_modes)
        added = len(modes) - prev_modes
        rounds.append(
            dict(
                asked=asked,
                distinct=n_distinct,
                eligible=eligible,
                modes=len(modes),
                modes_added=added,
                seconds=round(secs, 1),
                rc=rc,
            )
        )
        print(
            f"[upsample] round {len(rounds)}: asked {asked} -> {n_distinct} distinct, "
            f"{eligible} eligible, {len(modes)} modes (+{added}) in {secs/60:.1f} min",
            flush=True,
        )

        if len(modes) >= a.target_modes:
            reason = "target-reached"
            break
        if len(rounds) > 1 and added < a.stall_modes:
            # A CLAIM ABOUT THE GENERATOR, not about our budget -- it has run out of distinct
            # chemistry above the gate. Distinguish this from `cap` in every downstream table.
            reason = "stalled"
            break
        if n_distinct >= cap:
            reason = "cap"
            break
        prev_modes = len(modes)
        asked = min(asked + a.round_size, cap)

    free = load_trace(run_dir / "trace.csv", hib)
    scored = dict(free)
    scored.update(load_scored(cand, col))
    modes = count_modes(scored, gate, hib, a.cutoff, cap=a.target_modes)
    rews = sorted((scored[m] for m in modes), reverse=hib)
    summary = dict(
        generator=a.generator,
        target=a.target,
        seed=a.seed,
        target_modes=a.target_modes,
        cutoff=a.cutoff,
        gate=gate,
        score_column=col,
        higher_is_better=hib,
        cap=cap,
        round_size=a.round_size,
        stall_modes=a.stall_modes,
        stop_reason=reason,
        modes_reached=len(modes),
        pool_limited=len(modes) < a.target_modes,
        distinct_scored=len(scored),
        free_from_training=len(free),
        modes_from_training_alone=len(free_modes),
        newly_sampled=max(len(scored) - len(free), 0),
        eligible=sum(1 for v in scored.values() if ((v > gate) if hib else (v < gate))),
        # The reward profile of the SELECTED modes. Needed because reaching 500 modes by digging far
        # down the ranking is not the same result as reaching it from the top: the 500th mode's
        # reward differs a lot between generators, and a count alone hides that.
        mode_reward_best=rews[0] if rews else None,
        mode_reward_median=rews[len(rews) // 2] if rews else None,
        mode_reward_worst=rews[-1] if rews else None,
        rounds=rounds,
        total_sampling_seconds=round(sum(r["seconds"] for r in rounds), 1),
    )
    (out / "upsample_log.json").write_text(json.dumps(summary, indent=2))
    (out / "modes.smi").write_text("\n".join(modes) + ("\n" if modes else ""))

    # EMIT A CANDIDATES CSV, not just the mode list, so the existing pool builder consumes this
    # unchanged. Handing build_s3gfn_pools.py the raw scored set lets IT do the mode selection with
    # the same canonical helper -- one implementation, and this script's count becomes a cross-check
    # rather than a second source of truth. Columns mirror candidates.csv exactly: `raw_score` is the
    # oracle's own value (raw Vina for docking, LOWER is better) and `score` is the higher-is-better
    # training value, because the builder defaults to raw_score under --lower-is-better and to score
    # otherwise. Writing only one of them would silently gate a docking pool on the wrong axis --
    # the bug that made all sixteen ClpP cells read as empty on 2026-08-26.
    cand_out = out / "stage2_candidates.csv"
    with open(cand_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score", "raw_score"])
        for smi, val in scored.items():
            if hib:
                w.writerow([smi, val, val])
            else:
                w.writerow([smi, max(-float(val), 0.0), val])
    print(
        f"[upsample] wrote {cand_out} ({len(scored)} molecules) — feed this to "
        f"build_s3gfn_pools.py with TAG_SUFFIX=_stage2",
        flush=True,
    )
    print(
        f"[upsample] STOP={reason}  modes={len(modes)}/{a.target_modes}  "
        f"distinct_scored={len(scored)}  pool_limited={summary['pool_limited']}",
        flush=True,
    )
    print(
        f"[upsample] mode rewards: best {summary['mode_reward_best']} "
        f"median {summary['mode_reward_median']} worst {summary['mode_reward_worst']}",
        flush=True,
    )
    print(f"[upsample] wrote {out}/upsample_log.json + modes.smi", flush=True)
    if reason == "sampling-failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
