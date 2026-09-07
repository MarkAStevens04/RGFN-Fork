#!/usr/bin/env python
"""Which v1 training cells can be COPIED into benchmark_v2, and which must be regenerated?

WHY THIS EXISTS. `grid.csv`'s ``train_plan`` column was written on 2026-08-28 from what was on disk
*then*. Since then FragGFN was retrained at the normalized budget, SynFormer went from 1 of 9 cells to
9 of 9, and S3-GFN lost trace files to a runner re-invocation. Planning a 108-cell campaign against a
stale plan would copy the wrong things and regenerate things we already hold. So the plan is
re-derived from the filesystem, every time, and this script is the only thing allowed to set it.

WHAT MAKES A CELL COPYABLE. Four artifacts, and the trace is the one people forget:

  candidates.csv   the training run's own pool. NOTE the two locations below.
  trace.csv        one row per oracle call. REQUIRED, not optional: stage 2 harvests it as a free
                   pool of already-scored molecules, and saturn_clpp/s42 reaches its 500-mode target
                   from history alone -- 28 GPU-hours to zero. A copy without it silently throws
                   that away, and nothing downstream reports the loss.
  checkpoint       generator-specific; stage 2 samples it when the trace runs out.
  run_config       what the cell was actually run with, as opposed to what we meant to run.

THE TWO CANDIDATES DIRECTORIES, AND WHY THE DEFAULT IS THE SAFE ONE. Stage 2 (`dd8f1a9`) snapshots
``fixed_reward/`` to ``fixed_reward.budget_faithful/`` before it upsamples, because upsampling
OVERWRITES the training run's own 2,000-molecule sample -- that is how s3gfn_seh/seed42's went from
2,000 to 20,000 rows. So where the snapshot exists it is the arm-A artifact and ``fixed_reward/`` is
a post-Stage-2 pool; where it does not, ``fixed_reward/`` is still original. Copying the wrong one
ships an upsampled pool as if it were the training sample, which is invisible downstream. This script
reports both and names which it would copy.

BUDGET CHECK -- COUNT THE TRAINING ROWS, DO NOT READ ``n_scored``. Arm A is 10,000 oracle calls,
and the honest measure is the number of ``phase == "train"`` rows. ``n_scored`` is a single
cumulative counter SHARED across phases, and S3-GFN INTERLEAVES a 2,000-molecule evaluation sample
with training (s3gfn_drd2/42: three phase switches, eval spanning n_scored 65..12,048). So its last
``n_scored`` reads 12,048 and even ``max(n_scored)`` over training rows reads 11,048, while the true
training budget is 10,048 -- exactly on budget, like every other entrant. Reading either number makes
a healthy cell look 10-20% over and invites a re-train nobody needs. Steps x batch is wrong for a
third reason (replay handling differs per generator), which is why the trace exists at all.

Read-only. Touches nothing, decides nothing on its own -- prints a table and writes JSON.

    python experiments/benchmark_v2/tools/inventory_v1_cells.py
    python experiments/benchmark_v2/tools/inventory_v1_cells.py --json out.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

FR_ROOT = Path("/scratch/markymoo/rgfn_runs/experiments/fixed_reward")

# (generator, run-dir template). The reaction-GFNs use the `_5k` suffix from the v1 campaign; the
# competitors do not, because they were never run at a 5,000-step budget.
COMPETITORS = ["fraggfn", "s3gfn", "synformer", "reinvent", "saturn", "tango"]
REACTION_GFNS = ["rgfn", "rxnflow", "scent"]
TARGETS = ["seh", "drd2", "clpp"]  # phase 1. 6td3b has never been run for any generator.
SEEDS = [42, 43, 44]

ARM_A_CALLS = 10_000
# A trace may legitimately overshoot by one batch (the run stops at the first step PAST the budget).
# 12,048 is REINVENT's 157x64 plus S3-GFN's 1,000-molecule eval; 15,000 is comfortably outside that.
ARM_A_MAX = 15_000
ARM_A_MIN = 9_000

# Where each generator leaves its weights. Checked as a glob under the run dir.
CKPT_GLOBS = {
    "reinvent": ["agent.chkpt", "checkpoints/*.chkpt"],
    "saturn": ["checkpoints/*.ckpt"],
    "tango": ["checkpoints/*.ckpt"],
    "s3gfn": ["*/model_state*.pt", "*/*.pt", "checkpoints/*.pt"],
    "synformer": ["population_checkpoints/*", "checkpoints/*"],
    "fraggfn": ["checkpoints/*.pt"],
    "rgfn": ["train/checkpoints/*.pt"],
    "rxnflow": ["checkpoints/*.pt"],
    "scent": ["train/checkpoints/*.pt"],
}


def run_dir(gen: str, target: str, seed: int) -> Path:
    suffix = "_5k" if gen in REACTION_GFNS else ""
    return FR_ROOT / f"{gen}_{target}{suffix}" / f"seed{seed}"


def _rows(path: Path):
    """Row count of a CSV, excluding the header. None if absent."""
    if not path.is_file():
        return None
    with open(path, newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def best_trace(d: Path):
    """The cell's real trace, which is often NOT ``trace.csv``.

    RECOVERY, measured 2026-09-07. Re-invoking a runner truncates ``trace.csv`` to its header (59
    bytes) — that is the defect `3281bce` fixed by making TraceWriter ROTATE rather than overwrite.
    The consequence for us is the good news: on ten cells the full trace survives as ``trace.csv.1``
    while ``trace.csv`` is an empty stub, so reading only the canonical name would have condemned
    those cells to a needless re-train. Verified on s3gfn_drd2/42, s3gfn_clpp/44 and fraggfn_drd2/42:
    the rotation holds a complete 10,048–12,048-row trace ending at the run's real budget.

    Rule: take the file with the most rows among ``trace.csv`` and every ``trace.csv.N``. Ties go to
    the canonical name. Where every candidate is a header-only stub the trace is genuinely gone
    (s3gfn_seh/43 — all three are 59 bytes), which is a different verdict and must not be papered
    over by silently reporting zero.
    """
    cands = [d / "trace.csv"] + sorted(d.glob("trace.csv.*"))
    best, best_stats = None, None
    for p in cands:
        st = _trace_stats(p)
        if st is None or "error" in st:
            continue
        if best_stats is None or (st.get("n_rows") or 0) > (best_stats.get("n_rows") or 0):
            best, best_stats = p, st
    if best_stats is None:
        return None
    best_stats["source"] = best.name
    best_stats["n_rotations"] = len(cands) - 1
    return best_stats


def _trace_stats(path: Path):
    """(last n_scored, last n_distinct, max n_scored among phase=='train' rows, n_rows).

    The LAST row's counters are the authority on how much budget the run consumed. Reading the row
    count instead would be wrong for any generator that writes a header-only or partial file.
    """
    if not path.is_file():
        return None
    last = None
    train_max = 0
    train_rows = 0
    n = 0
    try:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                n += 1
                last = r
                if (r.get("phase") or "") == "train":
                    train_rows += 1
                    try:
                        train_max = max(train_max, int(r["n_scored"]))
                    except (ValueError, KeyError, TypeError):
                        pass
    except Exception as exc:  # a truncated trace is a finding, not a crash
        return {"error": str(exc), "n_rows": n}
    if last is None:
        return {"n_rows": 0, "n_scored": 0, "n_distinct": 0,
                "train_rows": 0, "train_max": 0}

    def _i(k):
        try:
            return int(last[k])
        except (ValueError, KeyError, TypeError):
            return None

    # THE BUDGET IS THE TRAIN ROW COUNT, NOT max(n_scored) OVER TRAIN ROWS. `n_scored` is a single
    # cumulative counter SHARED across phases, and S3-GFN INTERLEAVES its 2,000-molecule evaluation
    # sample with training -- measured on s3gfn_drd2/42: three phase switches, eval n_scored spanning
    # 65..12,048. So by the last training row the counter has already absorbed eval calls, and
    # train_max reads 11,048 where the true training budget is 10,048, making an ON-BUDGET cell look
    # 10% over. Counting rows is immune to the interleaving. train_max is kept only to expose the gap.
    return {
        "n_rows": n,
        "n_scored": _i("n_scored"),
        "n_distinct": _i("n_distinct"),
        "train_rows": train_rows,
        "train_max": train_max,
    }


def inspect(gen: str, target: str, seed: int) -> dict:
    d = run_dir(gen, target, seed)
    cand_live = d / "fixed_reward" / "candidates" / "candidates.csv"
    cand_snap = d / "fixed_reward.budget_faithful" / "candidates" / "candidates.csv"
    trace = best_trace(d)
    ckpts = []
    for g in CKPT_GLOBS.get(gen, []):
        ckpts += [p for p in d.glob(g) if p.is_file()]
    rec = {
        "generator": gen,
        "target": target,
        "seed": seed,
        "run_dir": str(d),
        "exists": d.is_dir(),
        "candidates_live": _rows(cand_live),
        "candidates_budget_faithful": _rows(cand_snap),
        # The snapshot wins where it exists: `fixed_reward/` is post-Stage-2 there.
        "arm_a_candidates": str(cand_snap if cand_snap.is_file() else cand_live),
        "trace": trace,
        "trace_source": (trace or {}).get("source"),
        "n_checkpoints": len(ckpts),
        "has_run_config": any(
            (d / n).is_file() for n in ("run_config.yaml", "run_config_effective.yaml")
        ),
        "trace_rotations": (trace or {}).get("n_rotations", len(list(d.glob("trace.csv.*")))),
    }
    rec["verdict"], rec["why"] = _verdict(rec)
    return rec


def _verdict(r: dict):
    """copy | generate | attention, with the reason stated in the same breath."""
    if not r["exists"]:
        return "generate", "no run directory"
    t = r["trace"]
    cand = r["candidates_budget_faithful"] or r["candidates_live"]
    if not cand:
        # A missing pool is NOT a missing model. Where the checkpoint survives, the pool is one
        # sampling pass from the frozen policy -- minutes, not the GPU-days a re-train costs.
        # s3gfn_seh/42 is exactly this: candidates.csv deleted, 10 checkpoints and a 12,048-row
        # trace intact. Calling that "generate" would have bought a re-train we do not need.
        if r["n_checkpoints"]:
            return "resample", (
                f"candidates.csv gone but {r['n_checkpoints']} checkpoint(s) + trace "
                f"survive — re-SAMPLE the frozen policy, do not re-train"
            )
        return "generate", "no candidates.csv and no checkpoint"
    if t is None:
        return "attention", "NO trace.csv — copyable but stage 2 loses its free-pool harvest"
    if "error" in t:
        return "attention", f"trace unreadable ({t['error']})"
    # Phase-filtered where a phase column exists; n_scored only as a fallback.
    scored = t.get("train_rows") or t.get("n_scored") or 0
    if scored == 0:
        return "attention", (
            f"trace GONE — trace.csv and all {r['trace_rotations']} rotation(s) are "
            f"header-only stubs; stage 2 loses its free-pool harvest"
        )
    if scored > ARM_A_MAX:
        return "generate", f"over budget: trace reached {scored:,} oracle calls (arm A is 10,000)"
    if scored < ARM_A_MIN:
        return "attention", f"short: trace stopped at {scored:,} of 10,000 oracle calls"
    if r["n_checkpoints"] == 0:
        return "attention", f"trace OK ({scored:,}) but NO checkpoint found — stage 2 cannot sample"
    return "copy", f"{scored:,} oracle calls, {r['n_checkpoints']} checkpoint(s), trace intact"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--json", type=Path, help="also write the full records here")
    ap.add_argument(
        "--include-reaction-gfns",
        action="store_true",
        help="also inspect rgfn/rxnflow/scent (all expected to regenerate)",
    )
    a = ap.parse_args()

    gens = COMPETITORS + (REACTION_GFNS if a.include_reaction_gfns else [])
    recs = [inspect(g, t, s) for g in gens for t in TARGETS for s in SEEDS]

    print(
        f"{'gen':<10} {'tgt':<5} {'seed':>4} {'cand(bf/live)':>15} {'n_scored':>9} "
        f"{'train':>8} {'ckpt':>5}  verdict"
    )
    print("-" * 100)
    for r in recs:
        t = r["trace"] or {}
        bf, live = r["candidates_budget_faithful"], r["candidates_live"]
        cand = f"{bf if bf is not None else '-'}/{live if live is not None else '-'}"
        print(
            f"{r['generator']:<10} {r['target']:<5} {r['seed']:>4} {cand:>15} "
            f"{(t.get('n_scored') if t.get('n_scored') is not None else '-')!s:>9} "
            f"{(t.get('train_max') or '-')!s:>8} {r['n_checkpoints']:>5}  "
            f"{r['verdict']}: {r['why']}"
        )

    print()
    tally = {}
    for r in recs:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    print("verdicts:", ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    n_snap = sum(1 for r in recs if r["candidates_budget_faithful"] is not None)
    print(
        f"cells whose fixed_reward/ was already overwritten by stage 2 "
        f"(budget_faithful snapshot present, and it is what we copy): {n_snap}/{len(recs)}"
    )

    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(recs, indent=2))
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
