#!/usr/bin/env python
"""PRE-FLIGHT: how many distinct modes does a candidate pool actually contain, as a function of size?

WHY THIS IS A GATE AND NOT A CURIOSITY. The competitor pipeline's expensive step is MultiAiZ route
discovery: ~16 s per molecule, linear, so ~2.25 h for an N=500 pool (Logs/056), and its cache key is
the POOL rather than the molecule — change the pool and the whole thing is re-paid. This script
answers, in seconds, what that pool can possibly deliver, before any of it is spent.

WHAT THE GATE ASKS CHANGED WITH THE READOUT (2026-08-19). The primary readout is now MODES AT A FIXED
REACTION BUDGET, not reactions for a fixed mode target (CLAUDE.md, "THE BENCHMARK'S PRIMARY READOUT").
Under a mode target, a pool short of the target was unreportable, so this script REFUSED it. Under a
reaction budget every non-empty pool yields a number, so refusing would delete a legitimate datapoint
— the exact failure CLAUDE.md warns against ("turns an EXCLUSION into a FLAGGED DATAPOINT").

So the gate now PREDICTS THE STOP REASON instead of excluding:
  * ``budget-binding possible``  — the pool holds at least as many modes as the budget could buy, so
    the budget can be what stops the selection. The only like-for-like case.
  * ``pool-exhausted expected``  — the pool holds fewer, so the selection will run out of qualifying
    candidates with budget UNSPENT. Still run, still reported, but never scored as a cost win or loss,
    and ``used_rxns`` must be quoted beside the mode count.
It aborts only on ``--min-modes``: a pool too small to measure anything at all, where 2.25 h of route
discovery really would buy nothing.

Upper bound used for the prediction: a mode costs at least ``--min-rxn-per-mode`` reactions (default
1.0 — a route with no reactions is a purchasable molecule, which these pools do not contain), so a
budget of R reactions can buy at most R / min_rxn_per_mode modes. Measured rates are 1.2-2.7
reactions per mode, so this bound is deliberately loose in the pool's favour.

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

Exit status is the gate: 0 if the pool is worth routing, 1 if it holds fewer than
--min-modes modes. The predicted stop reason is printed and stored in summary.json either way.
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


def load_ranked(path: Path, gate: float, higher_is_better: bool, score_column: str = "score"):
    """[(smiles, score)] above the gate, DEDUPLICATED, best-score-first.

    Dedup is load-bearing for the same reason it is in ``build_s3gfn_pools.py``: a row in a generator
    dump is a sampling EVENT, not a candidate. Counting modes over raw rows would report the pool as
    both larger and less diverse than it is.

    ``score_column`` MUST match ``higher_is_better``, and this gate read the wrong one for every
    docking target until 2026-08-26. candidates.csv carries both ``score`` -- the generator's training
    reward, which for docking is clip(-vina), a POSITIVE 0..11 number -- and ``raw_score``, raw Vina
    kcal/mol. The mode gates are defined on the RAW value (ClpP -8.0, Logs/045). Reading ``score``
    against -8.0 finds NOTHING below the bar, so this pre-flight aborted all sixteen ClpP chain-cells
    as "fewer than 10 modes -- nothing to measure" while 645 of 2,000 molecules were in fact passing
    on ``raw_score``. That would have entered the paper as a pool-size finding about the baselines.
    ``build_s3gfn_pools.py`` already resolved this correctly; the two must agree or the pre-flight
    gates a pool the builder would happily have built."""
    best: dict = {}
    n_rows = 0
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            n_rows += 1
            smi = row.get("smiles") or row.get("SMILES") or row.get("child_key")
            try:
                raw = row.get(score_column)
                if raw is None:
                    raw = row.get("score", row.get("reward"))
                val = float(raw)
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
        "--gate",
        type=float,
        required=True,
        help="per-target reward gate — see experiments/lsd_hubs/matrix16/targets.py (5%-FPR standard, 2026-08-21); passing a stale hand-picked bar is the bug this replaced",
    )
    ap.add_argument("--cutoff", type=float, default=0.5, help="tau for mode counting")
    ap.add_argument(
        "--target-modes",
        type=int,
        default=100,
        help="SECONDARY readout's mode target; reported, no longer the exit status",
    )
    ap.add_argument(
        "--rxn-budget",
        type=int,
        default=100,
        help="PRIMARY readout's reaction budget; sets the predicted stop reason",
    )
    ap.add_argument(
        "--min-rxn-per-mode",
        type=float,
        default=1.0,
        help="loose lower bound on a mode's route cost, used only to bound modes-per-budget",
    )
    ap.add_argument(
        "--min-modes",
        type=int,
        default=10,
        help="ABORT floor: fewer modes than this and route discovery measures nothing",
    )
    ap.add_argument(
        "--pool",
        choices=("naive", "pruned"),
        default="naive",
        help="which pool construction will be built; decides what the gate evaluates "
        "(naive = the top-N prefix, pruned = N mutually distinct drawn from the whole set)",
    )
    ap.add_argument(
        "--score-column",
        default=None,
        help="which candidates.csv column carries the gated value. Defaults to `raw_score` when "
        "--lower-is-better is set (the docking convention) and `score` otherwise -- the SAME rule "
        "build_s3gfn_pools.py uses; they must not diverge.",
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
    score_col = a.score_column or ("score" if hib else "raw_score")
    ranked, n_rows = load_ranked(Path(a.candidates), a.gate, hib, score_column=score_col)
    print(
        f"[saturation] gating column: {score_col} (higher_is_better={hib}, gate={a.gate})",
        flush=True,
    )
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

    modes_at_largest = rows[-1]["n_modes"]
    # The smallest pool that already contains the SECONDARY readout's deliverable. This is also the
    # "candidates you must score" axis, directly comparable across generators.
    reached_at = next((r["n_candidates"] for r in rows if r["n_modes"] >= a.target_modes), None)
    mps = mean_pairwise_similarity(smis[: min(500, len(smis))], fps=fps[: min(500, len(smis))])

    # Modes in the WHOLE above-gate set, which is what the pruned pool draws from. Distinct from
    # modes_at_largest (a prefix of the ranking) and the gap between them is the entire point of the
    # two-pool design: Saturn sEH s42 has 18 modes in its top 500 and 338 in its full 1,911.
    all_reps = mode_representatives(
        smis, rewards, higher_is_better=hib,
        reward_threshold=a.gate, similarity_threshold=a.cutoff, fps=fps,
    )  # fmt: skip
    modes_available = len(all_reps)

    # What the pool that will ACTUALLY be built can deliver. The naive pool is the top-N prefix, so
    # it is capped by the modes inside that prefix; the pruned pool is N mutually distinct molecules
    # drawn from the whole set, so it is capped by the set's total and by N itself.
    largest_size = rows[-1]["n_candidates"]
    if a.pool == "pruned":
        modes_in_pool = min(largest_size, modes_available)
    else:
        modes_in_pool = modes_at_largest

    # Predict the stop reason. A mode costs >= min_rxn_per_mode reactions, so R reactions buy at most
    # this many modes; if the pool holds fewer, the pool runs out before the budget does.
    max_modes_budget_can_buy = int(a.rxn_budget / max(a.min_rxn_per_mode, 1e-9))
    budget_can_bind = modes_in_pool >= max_modes_budget_can_buy
    predicted_stop = "budget-binding possible" if budget_can_bind else "pool-exhausted expected"

    summary = {
        "tag": a.tag or Path(a.candidates).parent.name,
        "candidates": str(a.candidates),
        "n_rows": n_rows,
        "n_distinct_above_gate": len(smis),
        "gate": a.gate,
        "higher_is_better": hib,
        "cutoff": a.cutoff,
        "target_modes": a.target_modes,
        "pool_variant": a.pool,
        "total_modes": modes_at_largest,  # name kept: modes in the largest evaluated PREFIX
        "modes_in_largest_prefix": modes_at_largest,
        "modes_available_whole_set": modes_available,
        "modes_in_pool_to_be_built": modes_in_pool,
        "rxn_budget": a.rxn_budget,
        "max_modes_budget_can_buy": max_modes_budget_can_buy,
        "predicted_stop_reason": predicted_stop,
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
    print(f"[saturation] pool variant   : {a.pool}")
    print(
        f"[saturation] modes available: {modes_available} in the whole above-gate set "
        f"({modes_at_largest} inside the top-{largest_size} prefix)"
    )
    print(f"[saturation] pool to be built: {modes_in_pool} modes")
    if reached_at is not None:
        print(
            f"[saturation] secondary readout: {a.target_modes} modes reached at "
            f"{reached_at} candidates"
        )
    else:
        print(
            f"[saturation] secondary readout: {a.target_modes} modes NOT reachable from this "
            f"pool — the fixed-MODE number is undefined here, the fixed-REACTION one is not"
        )

    # ---- the gate ------------------------------------------------------------------------------
    # Abort ONLY when there is nothing to measure. A pool that merely cannot saturate the budget is
    # a legitimate, reportable datapoint under the reaction-budget readout — see the module
    # docstring and CLAUDE.md.
    if modes_in_pool < a.min_modes:
        print("")
        print(
            f"[saturation] FAIL: pool holds {modes_in_pool} modes, under the --min-modes "
            f"floor of {a.min_modes}. Route discovery would measure nothing. Raise the "
            "generator's diversity setting or sample more; do not report this as a cost result."
        )
        raise SystemExit(1)

    print("")
    if budget_can_bind:
        print(
            f"[saturation] PASS ({predicted_stop}): {modes_in_pool} modes >= the "
            f"{max_modes_budget_can_buy} that {a.rxn_budget} reactions could buy at "
            f">={a.min_rxn_per_mode} rxn/mode. This cell CAN be like-for-like — but confirm it "
            "from used_rxns in the frontier output, do not assume it."
        )
    else:
        print(
            f"[saturation] PASS ({predicted_stop}): {modes_in_pool} modes < the "
            f"{max_modes_budget_can_buy} that {a.rxn_budget} reactions could buy. The selection "
            "will very likely run out of qualifying candidates with budget UNSPENT."
        )
        print(
            "[saturation]   -> Report modes AND used_rxns. Flag the cell pool-exhausted. Never "
            "score it as a cost win or loss; it is a POOL-SIZE result, which is itself a finding."
        )


if __name__ == "__main__":
    main()
