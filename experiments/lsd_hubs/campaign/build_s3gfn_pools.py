#!/usr/bin/env python
"""Build the NESTED top-N candidate pools that MultiAiZ route discovery is run over.

WHAT A POOL IS, AND WHY ITS COMPOSITION MATTERS. `submit_multiaiz_discover.sh` caches one
`multiaiz_routes.json` per pool directory, and its cache key is the POOL, never the molecule:
MultiAiZ is set-based (it runs AiZynth over the whole target set for n_iters cycles, appending
discovered intermediates to stock so targets converge on shared chemistry), so a molecule's route
depends on what else was planned alongside it. This script is therefore the definition of the unit
that gets planned, and re-running it with different inputs invalidates the artifact downstream.

TWO PROPERTIES ARE LOAD-BEARING.

1. **Dedup by SMILES.** A row in a generator dump is a *sampling event*, not a candidate. Our own
   reaction-GFN re-samples constantly (the top-500 rows of the sEH run are only 115 distinct
   molecules), so taking rows verbatim would hand the planner the same target dozens of times and
   silently shrink "the 500 best candidates" to a fraction of that. S3-GFN's `candidates.csv`
   happens to already be distinct (it emits unique valid molecules), so here dedup is a no-op —
   which is exactly why it must be written down rather than relied upon: the same pattern is used
   for pools built from `records.csv`, where it is not.
2. **Nesting.** Every N is a prefix of the largest N, so the route-planning COST-SCALING curve
   (`discovery_timing.json` across pool sizes) varies only the pool SIZE and not its membership.
   Sorting once and slicing gives this for free.

Ties in reward are broken by first appearance, which makes the output a deterministic function of
the input file rather than of dict iteration order.

Usage
-----
Build (writes <out-root>/<tag>_N<N>/{pool.smi,pool_scores.csv} for each N):

    python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
        --candidates $SCRATCH/rgfn_runs/experiments/fixed_reward/s3gfn_seh/seed43/fixed_reward/candidates/candidates.csv \
        --out-root   $SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools \
        --tag        s3gfn_seh_seed43 --sizes 500

Verify against an existing reference pool (used to prove this script reproduces the seed-42
artifacts that the seed-43/44 replicates are compared against):

    python experiments/lsd_hubs/campaign/build_s3gfn_pools.py \
        --candidates .../s3gfn_seh/71007/fixed_reward/candidates/candidates.csv \
        --verify $SCRATCH/rgfn_runs/lsdflow_sparrow/multiaiz_pools/s3gfn_seh_N500
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_ranked(
    path: Path,
    gate: float = 0.0,
    higher_is_better: bool = True,
    score_column: str = "score",
):
    """[(smiles, best_score)] sorted best-first, one entry per distinct SMILES.

    ``gate`` is the benchmark's quality bar. It exists because the pool feeds a reward-maximizing
    selector downstream, so admitting molecules the bar excludes would let the planner spend its
    budget on material that cannot count as a delivered mode.

    ``score_column`` AND ``higher_is_better`` must BOTH be set for a docking target, and getting
    either wrong is silent. `candidates.csv` carries two columns: ``score`` is the generator's
    training reward -- for docking that is clip(-vina), a POSITIVE 0..11 number -- and ``raw_score``
    is the oracle's own value, raw Vina kcal/mol. The gates are defined on the RAW value, so a ClpP
    pool must read ``raw_score`` with ``higher_is_better=False``. Reading ``score`` with the -8.0
    gate keeps all 2,000 molecules (every clip(-vina) exceeds -8.0) and ranks them by training
    reward; reading ``raw_score`` with the surrogate default keeps only non-binders. Neither failure
    is visible downstream.

    ``higher_is_better`` FLIPS BOTH the gate test and the ranking, and it is not optional for the
    docking targets. ClpP's bar is raw Vina <= -8.0 (Logs/045), so with the surrogate default this
    function would keep everything ABOVE -8.0 and rank the WORST binders first -- a pool built
    backwards, which no downstream stage could detect. Read the target's convention from
    ``experiments/lsd_hubs/matrix16/targets.py``; never infer it from the column name.
    """
    best: dict[str, float] = {}
    order: dict[str, int] = {}
    n_rows = 0
    with open(path, newline="") as fh:
        for i, r in enumerate(csv.DictReader(fh)):
            smi = r.get("smiles") or r.get("SMILES")
            raw = r.get(score_column)
            if raw is None:
                raw = r.get("score", r.get("reward"))
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if not smi:
                continue
            # NaN IS NOT A PASSING SCORE, and this gate cannot see that on its own. It is phrased as
            # "skip if it FAILS", and every comparison against NaN is False -- so `nan < gate` does
            # not fire the skip and an UNSCORED molecule falls straight through as if it had passed.
            # `float("nan")` also parses cleanly, so the except above never catches it.
            # Docking targets are hit hardest because a failed dock writes score=nan while a
            # surrogate almost always returns a number: measured 2026-09-06, s3gfn_clpp_seed43's
            # _stage2 pool held 422 unscored molecules of 500 (84%), against 1 for the sEH pools.
            # Every OTHER gate in the pipeline phrases the test positively (mode_saturation:104,
            # sparrow_select_frontier.passes_gate) or guards NaN explicitly
            # (metrics/diversity._passes_gate), so they all reject it correctly; this was the one hole.
            if val != val:
                continue
            if (val < gate) if higher_is_better else (val > gate):
                continue
            n_rows += 1
            if smi not in best or ((val > best[smi]) if higher_is_better else (val < best[smi])):
                best[smi] = val
            order.setdefault(smi, i)
    sign = -1.0 if higher_is_better else 1.0
    ranked = sorted(best.items(), key=lambda t: (sign * t[1], order[t[0]]))
    return ranked, n_rows


def write_pool(out_dir: Path, rows) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "pool.smi", "w") as fh:
        for smi, _ in rows:
            fh.write(f"{smi}\n")
    with open(out_dir / "pool_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        w.writerows(rows)


def verify(ref: Path, rows) -> int:
    """Compare this script's output against an existing pool dir. Returns an exit code."""
    ref_smi = (ref / "pool.smi").read_text().split()
    mine = [s for s, _ in rows[: len(ref_smi)]]
    ok_order = mine == ref_smi
    ok_set = set(mine) == set(ref_smi)
    ref_scores = {r["smiles"]: r["score"] for r in csv.DictReader(open(ref / "pool_scores.csv"))}
    ok_scores = all(s in ref_scores and float(ref_scores[s]) == v for s, v in rows[: len(ref_smi)])
    print(f"[verify] {ref}  (N={len(ref_smi)})")
    print(f"  identical order   : {ok_order}")
    print(f"  identical set     : {ok_set}")
    print(f"  identical scores  : {ok_scores}")
    if ok_order and ok_set and ok_scores:
        print("  VERIFIED — this script reproduces the reference pool exactly")
        return 0
    print("  MISMATCH — do NOT build replicate pools with this script until reconciled")
    return 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True, help="candidates.csv (smiles + score columns)")
    ap.add_argument("--out-root", default="", help="parent dir for <tag>_N<N> pool dirs")
    ap.add_argument("--tag", default="", help="pool-dir prefix, e.g. s3gfn_seh_seed43")
    ap.add_argument(
        "--sizes",
        default="500",
        help="comma-separated pool sizes; each is a PREFIX of the largest (nested)",
    )
    ap.add_argument(
        "--gate",
        type=float,
        required=True,
        help="per-target reward gate — see experiments/lsd_hubs/matrix16/targets.py (5%-FPR standard, 2026-08-21)",
    )
    ap.add_argument(
        "--lower-is-better",
        action="store_true",
        help="docking targets: the gate is an UPPER bound on a raw energy (ClpP -8.0) and better "
        "means smaller. Mirrors the same flag on mode_saturation.py.",
    )
    ap.add_argument(
        "--score-column",
        default=None,
        help="which candidates.csv column carries the gated value. Defaults to `raw_score` when "
        "--lower-is-better is set (the docking convention) and `score` otherwise.",
    )
    ap.add_argument(
        "--strict-naive",
        action="store_true",
        help="restore the pre-2026-08-26 behaviour: SKIP a naive pool that cannot supply N distinct "
        "molecules instead of clamping to what exists. Skipping loses the cell entirely, which "
        "disproportionately deletes weak-baseline cells (7 of the 8 short cells are S3-GFN).",
    )
    ap.add_argument("--verify", default="", help="compare against this existing pool dir and exit")
    ap.add_argument(
        "--pruned",
        action="store_true",
        help="PRUNED POOL: take the top-N mutually DISTINCT molecules (greedy sphere exclusion in "
        "reward order, Morgan r=3/2048) instead of the top-N by reward. See docs/RESEARCH_CONTEXT.md "
        "'The two pools and the two numbers'. Scanning deeper is free (CPU-seconds) and we still "
        "route only N, so this costs no extra MultiAiZ -- it just reaches further down the reward "
        "ranking to find N genuinely different molecules.",
    )
    ap.add_argument(
        "--cutoff", type=float, default=0.5, help="--pruned: tau for the sphere-exclusion filter"
    )
    a = ap.parse_args()

    hib = not a.lower_is_better
    # Default the column to the target's own convention rather than making every caller remember it.
    col = a.score_column or ("score" if hib else "raw_score")
    rows, n_rows = load_ranked(Path(a.candidates), a.gate, higher_is_better=hib, score_column=col)
    print(
        f"[pools] {a.candidates}\n  {n_rows} rows -> {len(rows)} distinct molecules "
        f"(gate {col}{'>=' if hib else '<='}{a.gate}); best={rows[0][1]:.4f} "
        f"worst={rows[-1][1]:.4f}"
    )

    if a.pruned:
        # Sphere exclusion over the WHOLE above-gate set, reward-ordered. The survivors are mutually
        # dissimilar by construction, so any subset SPARROW later selects is automatically
        # all-distinct -- which is why reactions/candidate and reactions/mode must COINCIDE on this
        # pool, and their disagreement would be a bug rather than a finding.
        import sys as _sys

        _sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from validation.lsdflow.metrics.diversity import ecfp, mode_representatives

        biggest = max(int(x) for x in a.sizes.split(",") if x.strip())
        smis = [s_ for s_, _ in rows]
        rews = [v for _, v in rows]
        fps = [ecfp(s_) for s_ in smis]
        idx = mode_representatives(
            smis,
            rews,
            higher_is_better=hib,
            reward_threshold=a.gate,
            similarity_threshold=a.cutoff,
            fps=fps,
        )
        depth = (idx[biggest - 1] + 1) if len(idx) >= biggest else len(smis)
        rows = [rows[i] for i in idx]
        print(
            f"  PRUNED (tau={a.cutoff}): {len(idx)} distinct molecules exist; scanned {depth} "
            f"candidates to collect the top {min(biggest, len(idx))} "
            f"(reward {rews[0]:.3f} -> {rows[min(biggest, len(rows)) - 1][1]:.3f})"
        )
        if len(idx) < biggest:
            # Not fatal, but it MUST be visible: a short pruned pool means the generator cannot
            # supply N distinct molecules at all, which is a property of the generator and has to be
            # reported as one rather than surfacing later as a cheap-looking frontier.
            print(
                f"  WARNING: only {len(idx)} distinct molecules available, short of {biggest}. "
                "This cell is POOL-LIMITED on the pruned pool — report it as such."
            )
        pruned_meta = {
            "pool_variant": "pruned",
            "cutoff": a.cutoff,
            "gate": a.gate,
            "n_above_gate": len(smis),
            "n_distinct_available": len(idx),
            "scan_depth": depth,
            "reward_top": round(rews[0], 4),
            "reward_last_kept": round(rows[-1][1], 4),
        }

    if a.verify:
        raise SystemExit(verify(Path(a.verify), rows))

    if not a.out_root or not a.tag:
        raise SystemExit("[pools] --out-root and --tag are required unless --verify")

    sizes = [int(x) for x in a.sizes.split(",") if x.strip()]
    if not a.pruned and not a.strict_naive:
        # CLAMP THE NAIVE POOL TOO (changed 2026-08-26). The strict skip this replaces was a
        # deliberate choice -- its reasoning, preserved below, is that an _N500 directory holding 300
        # molecules misstates the pool. That is right, but it argues against MISNAMING, not against
        # building: the pruned path answered the same objection by naming the directory for the size
        # actually written, and an _N154 dir claims nothing it does not hold.
        #
        # The skip had to go because of WHICH cells it deletes. Eight of thirty-five fall short of
        # 500 on the naive arm and SEVEN are S3-GFN -- 154/65/168 above the sEH gate across its three
        # seeds, 263/243/302 on ClpP, 97 on DRD2 s44 -- against Saturn's ~1,900 every time. So the
        # skip silently removed exactly the cells where a baseline is weakest, and a published table
        # with S3-GFN's sEH column blank would be hiding its worst result rather than reporting it.
        # A pool of 65 is a finding; a missing directory is a pipeline artifact.
        #
        # Clamped naive pools are marked pool_limited in pool_meta.json so downstream can never quote
        # one as a 500-molecule pool, and their stop reason is pool-exhausted, not budget-binding.
        # --strict-naive restores the old behaviour.
        clamped = sorted({min(n, len(rows)) for n in sizes})
        if clamped != sorted(set(sizes)):
            print(
                f"  WARNING: only {len(rows)} distinct molecules above the gate. This cell is "
                f"POOL-LIMITED on the naive pool — report it as such."
            )
            print(f"  naive: sizes clamped to availability {sorted(set(sizes))} -> {clamped}")
        sizes = clamped
    if a.pruned:
        # A pruned pool SHORT of the request is the RESULT, not an error, so emit it at its true
        # size rather than skipping. Clamping (not skipping) is what makes a mode-collapsed entrant
        # priceable at all: Saturn sEH s42 supplies 338 of a requested 500, and 338 still clears the
        # ~100 modes a 100-reaction budget could buy. The directory is named for the size actually
        # written, so an _N338 dir never claims to be 500.
        # The NAIVE path clamps the same way as of 2026-08-26 -- see the block above for why the
        # strict skip it used to keep was removed, and --strict-naive to get it back.
        clamped = sorted({min(n, len(rows)) for n in sizes})
        if clamped != sorted(set(sizes)):
            print(f"  pruned: sizes clamped to availability {sorted(set(sizes))} -> {clamped}")
        sizes = clamped
    for n in sorted(sizes):
        if n > len(rows):
            print(f"  N={n:<6} SKIP — only {len(rows)} distinct candidates available")
            continue
        out_dir = Path(a.out_root) / f"{a.tag}_N{n}"
        if (out_dir / "multiaiz_routes.json").exists():
            # Never silently rewrite the pool a cached routes artifact was planned over: the routes
            # would then describe a different target set than pool.smi claims.
            print(f"  N={n:<6} SKIP — {out_dir} already has multiaiz_routes.json (planned pool)")
            continue
        write_pool(out_dir, rows[:n])
        n_requested = max(int(x) for x in a.sizes.split(",") if x.strip())
        if a.pruned:
            meta = dict(pruned_meta, n_requested=n_requested)
        else:
            # A clamped NAIVE pool needs the same marker as a clamped pruned one, or nothing
            # downstream can tell an _N154 pool from a 154-molecule slice of a larger one -- and the
            # difference is whether the cell is pool-exhausted or budget-binding.
            meta = {
                "pool": "naive",
                "gate": a.gate,
                "score_column": col,
                "higher_is_better": not a.lower_is_better,
                "n_distinct_above_gate": len(rows),
                "n_requested": n_requested,
            }
        meta["n_written"] = n
        meta["pool_limited"] = meta["n_written"] < meta["n_requested"]
        (out_dir / "pool_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  N={n:<6} -> {out_dir}/pool.smi")


if __name__ == "__main__":
    main()
