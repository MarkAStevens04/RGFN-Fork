#!/usr/bin/env python
"""PRE-FLIGHT: how many distinct modes does a candidate pool actually contain, as a function of size?

WHY THIS IS A GATE AND NOT A CURIOSITY. The competitor pipeline's expensive step is MultiAiZ route
discovery: ~16 s per molecule, linear, so ~2.25 h for an N=500 pool (Logs/056), and its cache key is
the POOL rather than the molecule — change the pool and the whole thing is re-paid. If a generator's
pool does not contain the deliverable (100 distinct modes at the target's gate and tau), every hour
of that is wasted and the cell is unreportable. This script answers "is 100 modes reachable, and from
how many candidates?" in seconds, before any of it is spent.

It also produces a real result, not just a go/no-go: the size at which a pool saturates IS the
"candidates you must score" axis — the one the benchmark honestly LOSES on (Logs/056: S3-GFN needs
~230 candidates to contain 100 distinct molecules where our enumerated pool needs ~1,010). Run it on
every arm and the trade is measured rather than asserted.

Entry [056] used a scratch copy of this that was never committed; this is that check, made
reproducible. Mode counting uses the project's canonical metric
(``validation/lsdflow/metrics/diversity``: Morgan r=3/2048, greedy sphere exclusion, best-reward-
first), so its numbers are directly comparable to every reactions-per-mode figure in the benchmark.

REGRESSION PIN (checked 2026-08-14). On the seed-42 S3-GFN sEH pool
(``$SCRATCH/rgfn_runs/experiments/fixed_reward/s3gfn_seh/71007/.../candidates.csv``) at gate 7.0 /
tau 0.5 this reports **206 modes at n=500, mode rate 0.412** — bit-identical to the 41.2% Logs/056
publishes for that pool, and 100 modes first reached at 250 candidates against the ~230 that entry
interpolates. If a change here moves those numbers, the mode metric moved, and every
reactions-per-mode figure in the benchmark moved with it.

Usage (rgfn env):
  python experiments/lsd_hubs/campaign/mode_saturation.py \
      --candidates $SCRATCH/rgfn_runs/experiments/fixed_reward/reinvent_seh/seed42/fixed_reward/candidates/candidates.csv \
      --gate 7.0 --cutoff 0.5 --target-modes 100 --out-dir <dir>

Exit status is the gate: 0 if the pool reaches --target-modes, 1 if it does not.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from validation.lsdflow.metrics.diversity import (  # noqa: E402
    ecfp,
    mean_pairwise_similarity,
    mode_representatives,
)


def load_ranked(path: Path, gate: float, higher_is_better: bool):
    """[(smiles, score)] above the gate, DEDUPLICATED, best-score-first.

    Dedup is load-bearing for the same reason it is in ``build_s3gfn_pools.py``: a row in a generator
    dump is a sampling EVENT, not a candidate. Counting modes over raw rows would report the pool as
    both larger and less diverse than it is."""
    best: dict = {}
    n_rows = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            smi = row.get("smiles") or row.get("SMILES") or row.get("child_key")
            try:
                val = float(row.get("score", row.get("reward")))
            except (TypeError, ValueError):
                continue
            if not smi:
                continue
            passes = val > gate if higher_is_better else val < gate
            if passes and (smi not in best or (val > best[smi]) == higher_is_better):
                best[smi] = val
    rows = sorted(best.items(), key=lambda t: -t[1] if higher_is_better else t[1])
    return rows, n_rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="candidates.csv (or any smiles+score CSV)")
    ap.add_argument("--out-dir", default="", help="write saturation.csv + summary.json here")
    ap.add_argument(
        "--gate", type=float, default=7.0, help="per-target reward gate (sEH 7.0, DRD2 0.5)"
    )
    ap.add_argument("--cutoff", type=float, default=0.5, help="tau for mode counting")
    ap.add_argument(
        "--target-modes", type=int, default=100, help="the deliverable; sets exit status"
    )
    ap.add_argument(
        "--lower-is-better",
        action="store_true",
        help="docking targets: the gate is an upper bound on a raw energy",
    )
    ap.add_argument(
        "--sizes",
        default="",
        help="comma-separated pool sizes to evaluate (default: a geometric ladder to the pool size)",
    )
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    hib = not a.lower_is_better
    ranked, n_rows = load_ranked(Path(a.candidates), a.gate, hib)
    if not ranked:
        raise SystemExit(
            f"[saturation] NO molecules pass the gate ({'>' if hib else '<'} {a.gate}) in "
            f"{a.candidates} ({n_rows} rows). That is a broken or mis-gated run, not a pool ceiling."
        )
    smis = [s for s, _ in ranked]
    rewards = [r for _, r in ranked]
    print(
        f"[saturation] {len(smis)} distinct above gate {'>' if hib else '<'} {a.gate} "
        f"(from {n_rows} rows) | tau={a.cutoff}"
    )

    # Fingerprint once and reuse. Every size below is a PREFIX of the same ranking, so this is one
    # pass over the pool rather than one per size.
    fps = [ecfp(s) for s in smis]

    if a.sizes:
        sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    else:
        sizes, n = [], 25
        while n < len(smis):
            sizes.append(n)
            n = int(n * 1.5)
        sizes.append(len(smis))
    sizes = sorted({min(s, len(smis)) for s in sizes})

    rows = []
    for n in sizes:
        reps = mode_representatives(
            smis[:n], rewards[:n], higher_is_better=hib,
            reward_threshold=a.gate, similarity_threshold=a.cutoff, fps=fps[:n],
        )  # fmt: skip
        rows.append({"n_candidates": n, "n_modes": len(reps), "mode_rate": round(len(reps) / n, 4)})
        print(f"  n={n:<6} modes={len(reps):<5} rate={rows[-1]['mode_rate']:.3f}")

    total_modes = rows[-1]["n_modes"]
    # The headline number: the smallest pool that already contains the deliverable. This is the
    # "candidates you must score" axis, directly comparable across generators.
    reached_at = next((r["n_candidates"] for r in rows if r["n_modes"] >= a.target_modes), None)
    mps = mean_pairwise_similarity(smis[: min(500, len(smis))], fps=fps[: min(500, len(smis))])

    summary = {
        "tag": a.tag or Path(a.candidates).parent.name,
        "candidates": str(a.candidates),
        "n_rows": n_rows,
        "n_distinct_above_gate": len(smis),
        "gate": a.gate,
        "higher_is_better": hib,
        "cutoff": a.cutoff,
        "target_modes": a.target_modes,
        "total_modes": total_modes,
        "candidates_to_reach_target": reached_at,
        "mean_pairwise_similarity_top500": round(mps, 4) if mps is not None else None,
        "rows": rows,
    }

    if a.out_dir:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "saturation.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"[saturation] wrote {out}/saturation.csv + summary.json")

    print("")
    if reached_at is not None:
        print(
            f"[saturation] PASS: {a.target_modes} modes reached at {reached_at} candidates "
            f"({total_modes} modes in the full pool of {len(smis)})."
        )
        return
    # Report the shortfall as what it is. A pool that cannot reach the deliverable is a real
    # property of the generator at this budget -- but it must be SAID, not discovered later as a
    # confusing frontier that stops early.
    print(
        f"[saturation] FAIL: pool tops out at {total_modes} modes, short of {a.target_modes}. "
        "Do NOT spend route discovery on it. Either raise the generator's diversity setting / "
        "sample more, or report the cell as pool-limited -- but not as a cost result."
    )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
