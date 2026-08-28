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
GATES = {
    "seh": (7.0, "score", True),
    "drd2": (0.5, "score", True),
    "clpp": (-8.0, "raw_score", False),
}


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
        choices=["reinvent", "saturn", "s3gfn", "tango"],
        help="synformer is deliberately absent: a GA population cannot be upsampled",
    )
    ap.add_argument("--target", required=True, choices=sorted(GATES))
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

    gate, col, hib = GATES[a.target]
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

    rounds: List[dict] = []
    asked = a.round_size
    reason = "cap"
    prev_modes = 0
    while True:
        scored_before = load_scored(cand, col)
        rc, secs = sample_round(a.runner, a.env, a.cfg, a.seed, run_dir, asked, _REPO_ROOT)
        scored = load_scored(cand, col)
        if rc != 0 and not scored:
            reason = "sampling-failed"
            print(f"[upsample] runner exited {rc} and no candidates on disk — stopping", flush=True)
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

    scored = load_scored(cand, col)
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
