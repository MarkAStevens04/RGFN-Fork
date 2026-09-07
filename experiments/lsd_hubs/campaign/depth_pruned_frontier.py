#!/usr/bin/env python
"""OURS. Re-run the competitor's SPARROW selection over a pool with the TRIVIALLY-MAKEABLE
molecules removed, and record the full funnel that leads to the number.

WHY THIS EXISTS
---------------
On sEH and ClpP the competitor pipeline (generator -> MultiAiZ -> SPARROW) reaches ~100 distinct
molecules for 100 reactions. Auditing that result showed WHY: 98-100% of the molecules it selects
are ONE reaction away from purchasable ZINC material. One reaction per molecule is the floor of the
reactions-per-mode metric, so the competitor has saturated the readout rather than out-planned
anyone -- and it did so by buying its way there out of a 17.4M-compound catalogue.

This script asks the question the saturated metric can no longer answer: **when the deliverable must
be a molecule you cannot simply buy-and-couple, who wins?** It keeps the catalogue exactly as it is
and instead applies a DELIVERABLE SPECIFICATION -- "the library must consist of molecules that take
at least K reactions to make, even with everything purchasable" -- which is a constraint a chemist
states in advance (escaping two-component amide space, IP space, elaboration).

The symmetric rule on our side is that a hub must cost >= 1 reaction to build, i.e. we may not
deliver children of a purchasable depth-0 hub either. That side is read off
`curve_hub_batching.csv` and is NOT handled here (see WHAT THIS DOES NOT DO).

WHY IT IS A RE-SOLVE AND NOT A TRIM
-----------------------------------
CLAUDE.md: trimming a solved SPARROW selection is exact for greedy and for hub-batching (greedy mode
selection is prefix-stable) but NOT for SPARROW, whose selection is jointly optimised over shared
intermediates -- a trim yields a feasible but suboptimal set, i.e. a lower bound on the competitor,
which flatters us. So we filter the CANDIDATE POOL and re-solve from scratch. This is cheap: route
discovery (MultiAiZ, ~2.25 h/pool) is already cached; the MILP itself builds in ~0.03 s and solves
in 0.2-2.2 s.

WHY A WRAPPER AND NOT A FLAG ON sparrow_select_frontier.py
----------------------------------------------------------
`sparrow_select_frontier.py` derives its targets from the --pool CSV (the entries loop iterates
`pool`, not the routes artifact), so filtering the pool is sufficient and needs no change to the
solver path -- the numbers stay bit-for-bit comparable with every existing cell. Writing into a NEW
output directory also keeps us clear of the cached-artifact trap that bit this project twice: several
submit scripts key their expensive artifacts on the POOL directory alone, so a second run with
different parameters into the same directory silently returns the first run's answer.

WHAT THIS DOES NOT DO
---------------------
* It does NOT handle `--route-source native`. A native route is SHALLOW -- a promoted dynamic-library
  fragment is attached in one step rather than synthesized -- so its `num_reactions` understates the
  true depth until `expand_route_with_recipes` has run. Depth-filtering on the unexpanded number
  would silently classify built scaffolds as trivial. Our side is measured from the campaign curve
  instead.
* It does NOT change the catalogue. That is the separate stock-ladder experiment; this one holds the
  catalogue fixed and varies the deliverable.

GATE PROVENANCE (read before quoting anything this produces)
------------------------------------------------------------
The cached competitor pools were built at the SUPERSEDED gates (sEH 7.0, DRD2 0.5, ClpP -8.0), not
the 5%-FPR standard now in `matrix16/targets.py` (sEH 5.68, DRD2 0.345, ClpP -9.1). Both sides of
each head-to-head sit at the same old bar, so the comparison is internally consistent -- but these
runs are a DIAGNOSTIC and must not go into a figure. A publishable version needs the retrain at the
new bars first, then this experiment on top. The funnel record stamps the gate for exactly this
reason.

Usage (rgfn env; the frontier shells out to the sparrow env for the MILP itself):

    python experiments/lsd_hubs/campaign/depth_pruned_frontier.py \
        --pool-dir /scratch/.../multiaiz_pools/s3gfn_seh_seed43_big_pruned_N270 \
        --out-dir  /scratch/.../results/s3gfn_seh_seed43_big_pruned_depth2_select \
        --gate 7.0 --higher-is-better true --min-steps 2 --budgets 50,100,150

    # scan every cell's funnel without solving anything:
    ... --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from validation.lsdflow.eval.network import canonical  # noqa: E402

FRONTIER = REPO / "experiments/lsd_hubs/campaign/sparrow_select_frontier.py"


def passes_gate(val: float, gate: float, higher_is_better: bool) -> bool:
    """Mirrors sparrow_select_frontier.passes_gate. Duplicated (3 lines) rather than imported so
    this wrapper never drags the frontier's module-level imports into a --dry-run scan."""
    return val > gate if higher_is_better else val < gate


def read_pool_scores(path: Path, gate: float, higher_is_better: bool):
    """[(smiles, score)] above the gate, deduplicated best-first — the same semantics the frontier's
    load_pool applies, so the filtered CSV we emit is a strict subset of what it would have seen."""
    best: dict[str, float] = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            smi = r.get("smiles") or r.get("SMILES") or r.get("child_key")
            try:
                val = float(r.get("score", r.get("reward")))
            except (TypeError, ValueError):
                continue
            if not smi or not passes_gate(val, gate, higher_is_better):
                continue
            if smi not in best or (val > best[smi] if higher_is_better else val < best[smi]):
                best[smi] = val
    return sorted(best.items(), key=lambda t: (-t[1] if higher_is_better else t[1]))


def route_depths(pool, routes, strip_stereo: bool):
    """{smiles: cheapest route length in reactions} for pool molecules that have a route.

    Lookup mirrors the frontier's multiaiz path exactly (`canonical(smi, strip_stereo)` against the
    routes artifact's keys); a mismatch here would silently drop molecules and read as "unroutable".
    """
    depth = {}
    for smi, _rew in pool:
        c = canonical(smi, strip_stereo)
        rts = routes.get(c) if c else None
        if not rts:
            continue
        depth[smi] = min(len(r.get("steps") or []) for r in rts)
    return depth


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pool-dir", required=True, help="a multiaiz_pools/<cell>_N<k> directory")
    ap.add_argument("--out-dir", required=True, help="NEW directory; never write into --pool-dir")
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda v: v.lower() != "false", default=True)
    ap.add_argument(
        "--min-steps",
        type=int,
        default=2,
        help="keep only molecules whose CHEAPEST cached route needs at least this many reactions. "
        "2 = 'not buy-two-things-and-couple-them'.",
    )
    ap.add_argument("--cutoff", type=float, default=0.5, help="tau for mode counting")
    ap.add_argument("--budgets", default="50,100,150", help="--selection sparrow: reaction budgets")
    ap.add_argument("--mode-points", default="", help="--selection greedy: mode counts to price")
    ap.add_argument("--selection", default="sparrow", choices=["sparrow", "greedy"])
    ap.add_argument("--max-seconds", type=int, default=1800)
    ap.add_argument("--strip-stereo", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--size-match",
        type=int,
        default=0,
        help="CONTROL ARM. Instead of depth-filtering, keep N routed molecules chosen WITHOUT regard "
        "to route depth, so the pool is the same SIZE as a depth-filtered one but not the same "
        "COMPOSITION. This separates the two things a depth filter does at once: it removes shallow "
        "molecules AND it shrinks SPARROW's choice. Sharing is what makes routes cheap, so less "
        "choice alone costs the competitor modes; without this control a size effect reads as a "
        "depth effect. Mutually exclusive with the depth filter (--min-steps is ignored).",
    )
    ap.add_argument(
        "--size-match-mode",
        default="random",
        choices=["random", "topn"],
        help="random = an unbiased draw (the neutral size control); topn = the N highest-reward "
        "routed molecules (how pools are normally built, so the pessimistic-for-us variant).",
    )
    ap.add_argument("--size-match-seed", type=int, default=0, help="draw seed; report it")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="write funnel.json and stop — no MILP. Use to scan every cell's wiggle room cheaply.",
    )
    a = ap.parse_args()

    pool_dir, out_dir = Path(a.pool_dir), Path(a.out_dir)
    if out_dir.resolve() == pool_dir.resolve():
        raise SystemExit("[depth] --out-dir must differ from --pool-dir (cached-artifact hazard)")
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_csv, routes_json = pool_dir / "pool_scores.csv", pool_dir / "multiaiz_routes.json"
    for p in (scores_csv, routes_json):
        if not p.exists():
            raise SystemExit(f"[depth] missing {p}")
    meta = {}
    if (pool_dir / "pool_meta.json").exists():
        meta = json.loads((pool_dir / "pool_meta.json").read_text())

    # The gate the pool was BUILT at must match the gate we now select at, or the funnel's first
    # stages describe a different population from its last ones.
    if meta.get("gate") is not None and abs(float(meta["gate"]) - a.gate) > 1e-9:
        raise SystemExit(
            f"[depth] GATE MISMATCH: pool built at {meta['gate']}, --gate is {a.gate}. "
            "The funnel would mix two populations. Pass the pool's own gate, or rebuild the pool."
        )

    pool = read_pool_scores(scores_csv, a.gate, a.higher_is_better)
    routes = json.loads(routes_json.read_text())
    depth = route_depths(pool, routes, a.strip_stereo)
    routed_pool = [(s, v) for s, v in pool if s in depth]
    if a.size_match:
        if a.size_match > len(routed_pool):
            raise SystemExit(
                f"[depth] --size-match {a.size_match} exceeds the {len(routed_pool)} routed molecules"
            )
        if a.size_match_mode == "topn":
            kept = routed_pool[: a.size_match]  # already best-reward-first
        else:
            rng = random.Random(a.size_match_seed)
            kept = sorted(rng.sample(routed_pool, a.size_match), key=lambda t: routed_pool.index(t))
        arm = f"size-match:{a.size_match_mode}:n={a.size_match}:seed={a.size_match_seed}"
    else:
        kept = [(s, v) for s, v in pool if depth.get(s, 0) >= a.min_steps]
        arm = f"depth>={a.min_steps}"

    hist_routed = Counter(depth.values())
    funnel = {
        "cell": pool_dir.name,
        "gate": a.gate,
        "higher_is_better": a.higher_is_better,
        "gate_provenance": "SUPERSEDED bar (see module docstring) — diagnostic only, not for figures",
        "cutoff": a.cutoff,
        "min_steps": a.min_steps,
        # --- the funnel, widest to narrowest -------------------------------------------------
        "n_sampled_above_gate": meta.get("n_above_gate"),  # after any upsampling
        "n_distinct_available": meta.get("n_distinct_available"),  # tau-distinct among those
        "n_pool_written": meta.get("n_written"),  # handed to MultiAiZ
        "pool_limited": meta.get("pool_limited"),
        "n_pool_above_gate": len(pool),  # what the frontier would see
        "n_routed": len(depth),  # survived retrosynthesis
        "routed_fraction": round(len(depth) / len(pool), 4) if pool else None,
        "arm": arm,  # depth filter or size control
        "size_match_n": a.size_match or None,
        "size_match_mode": a.size_match_mode if a.size_match else None,
        "size_match_seed": a.size_match_seed if a.size_match else None,
        "n_depth_kept": len(kept),  # SPARROW's wiggle room  <-- key
        "kept_depth_histogram": dict(sorted(Counter(depth[s] for s, _ in kept).items())),
        "depth_kept_fraction_of_routed": round(len(kept) / len(depth), 4) if depth else None,
        "route_depth_histogram_routed": dict(sorted(hist_routed.items())),
        "n_one_step": hist_routed.get(1, 0),
        "one_step_share_of_routed": round(hist_routed.get(1, 0) / len(depth), 4) if depth else None,
    }
    (out_dir / "funnel.json").write_text(json.dumps(funnel, indent=2))

    print(f"[depth] {pool_dir.name}  gate={a.gate}  arm={arm}")
    print(
        f"[depth] FUNNEL  above-gate {funnel['n_sampled_above_gate']} -> tau-distinct "
        f"{funnel['n_distinct_available']} -> pool {len(pool)} -> routed {len(depth)} "
        f"({funnel['routed_fraction']:.1%}) -> {arm} {len(kept)}"
    )
    print(f"[depth] route-depth histogram (routed): {funnel['route_depth_histogram_routed']}")

    if not kept:
        print("[depth] NOTHING survives the depth filter — cell is depth-exhausted, no solve.")
        return
    if a.dry_run:
        print(f"[depth] --dry-run: wrote {out_dir}/funnel.json, no MILP.")
        return

    filtered = out_dir / "pool_scores.csv"
    with open(filtered, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        w.writerows(kept)

    cmd = [
        sys.executable,
        str(FRONTIER),
        "--routes",
        str(routes_json),
        "--route-source",
        "multiaiz",
        "--pool",
        str(filtered),
        "--gate",
        str(a.gate),
        "--higher-is-better",
        "true" if a.higher_is_better else "false",
        "--cutoff",
        str(a.cutoff),
        "--out-dir",
        str(out_dir),
        "--selection",
        a.selection,
        "--max-seconds",
        str(a.max_seconds),
        "--strip-stereo",
        "true" if a.strip_stereo else "false",
        "--tag",
        a.tag or f"{pool_dir.name}_{arm.replace(chr(62),'ge').replace(':','_')}_{a.selection}",
    ]
    cmd += ["--mode-points", a.mode_points] if a.selection == "greedy" and a.mode_points else []
    cmd += ["--budgets", a.budgets] if a.selection == "sparrow" else []
    print("[depth] ->", " ".join(cmd))
    rc = subprocess.call(cmd)

    # Fold the solved result back into the funnel so one file carries the whole chain.
    fcsv = out_dir / "select_frontier.csv"
    if fcsv.exists():
        funnel["selection"] = [
            {
                k: r.get(k)
                for k in (
                    "budget_rxns",
                    "used_rxns",
                    "n_selected",
                    "n_modes_kept",
                    "cost_kept_rxns",
                    "milp_status",
                    "time_capped",
                )
            }
            for r in csv.DictReader(open(fcsv))
        ]
        (out_dir / "funnel.json").write_text(json.dumps(funnel, indent=2))
    sys.exit(rc)


if __name__ == "__main__":
    main()
