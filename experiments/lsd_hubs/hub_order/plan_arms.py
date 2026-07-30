#!/usr/bin/env python
"""Plan the hub-ordering ablation (Logs/053): build every arm's hub set, diff it against the
enumerations we already have, and slice the remainder into debug-sized GPU jobs.

Each arm is one ``pick_hubs.py`` invocation — same records, same flow estimate, different
``--pool`` / ``--order``. Since hub-batching walks ``hubs.csv`` top-to-bottom, the row order *is*
the strategy, so the arms differ only in which hubs are eligible and in what sequence they are
tried.

**Caching.** Enumeration is deterministic (same checkpoint, same frozen library, same
``--enum-max-children``), so a hub enumerated for one arm is reusable by any other. The planner
subtracts every hub already present in ``--cached`` dirs and only slices what is genuinely new;
``merge_enum.py`` reassembles each arm's full 200 in *its own* walk order afterwards.

**Slicing.** The worker writes its outputs only after the whole hub loop, so a slice that overruns
the 2 h ``debug`` wall clock loses everything. Slices are therefore packed to ``--slice-seconds``
of *estimated* GPU work (default 3600 s = half the limit) using per-depth p75 times measured from
the cached ``enum_timings.json``, with a hub-count cap as a second guard.

**Re-runnable by design.** ``--auto-cache`` (on by default) folds every finished slice under
``<out-root>/enum/`` back into the cache, so re-planning after each slice both drops the hubs
already done *and* refits the per-depth cost model on the timings just measured. That matters here:
the seed cost model is fitted on the incumbent's 200 hubs, which are shallow and high-flow, while
the reverse/random arms are dominated by depth-3 hubs we have barely measured. ``chain.sh`` re-plans
before every submission for exactly this reason.

    python experiments/lsd_hubs/hub_order/plan_arms.py --out-root $SCRATCH/rgfn_runs/lsdflow/hub_order

Pure stdlib; login-node safe. Writes ``manifest.json``, which ``chain.sh`` consumes.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PICK_HUBS = REPO / "experiments" / "lsd_hubs" / "campaign" / "pick_hubs.py"

# The ablation arms. ``args`` are appended to the shared pick_hubs call; ``needs_canonical`` marks
# the arm whose hub SET is pinned to the incumbent enumeration (ordering isolated from selection).
ARMS = [
    {
        "name": "flow_top",
        "label": "highest flow first (all hubs)",
        "args": ["--pool", "all", "--order", "flow_desc"],
    },
    {
        "name": "flow_bottom",
        "label": "lowest flow first (all hubs)",
        "args": ["--pool", "all", "--order", "flow_asc"],
    },
    {
        "name": "random",
        "label": "random order (all hubs, seed 0)",
        "args": ["--pool", "all", "--order", "random", "--seed", "0"],
    },
    {
        "name": "cand_order_fixedset",
        "label": "incumbent hub set, candidate-reward order",
        "args": ["--pool", "all", "--order", "candidate_reward"],
        "needs_canonical": True,
    },
    {
        "name": "cand_order",
        "label": "candidate-reward order (all hubs)",
        "args": ["--pool", "all", "--order", "candidate_reward"],
    },
    # `flow_bottom` is pool-limited across cutoffs 0.30-0.50, so its cost there is only a lower
    # bound. This triples the ENUMERATION POOL to separate "this ordering is bad" from "this
    # ordering needed more hubs".
    #
    # This stays a fair comparison against the 200-hub arms, because the budget is 300 MODES, not
    # hubs: at cutoff 0.5 the completing arms stop having used only 17-52% of their pool, so giving
    # them 600 hubs too would leave their results unchanged. (Measured caveat: `prebuild_k` ranks
    # fragments over the whole pool including unwalked hubs, so pool size leaks in at ~1.4% —
    # flow_top is 357 reactions at 200 hubs vs 352 at 100. The `--prebuild-k 0` control removes
    # that channel entirely.) Ascending flow order also makes the first 200 hubs byte-identical to
    # `flow_bottom`, so only the tail is new GPU work.
    {
        "name": "flow_bottom_600",
        "label": "lowest flow first, 600-hub pool",
        "args": ["--pool", "all", "--order", "flow_asc"],
        "n_hubs": 600,
    },
]

DEFAULT_ANALYSIS = "/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189"
DEFAULT_CANONICAL = "/scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363"


def _read_hubs(path: Path):
    with open(path) as fh:
        return [(r["smiles"], int(r["depth"])) for r in csv.DictReader(fh)]


def _cached_index(dirs):
    """``{(hub_input, depth): source_dir}`` for every hub already enumerated, + measured seconds.

    Keyed on ``hub_input`` (the raw stereo-aware SMILES we hand the worker) because that is the
    unique per-hub identity — ``hub_key`` is stereo-stripped and collides across stereoisomers.
    """
    have: dict = {}
    secs: dict = {}
    for d in dirs:
        d = Path(d)
        hubs_csv = d / "hubs.csv"
        enum = d / "enum_children.json"
        if not (hubs_csv.exists() and enum.exists()):
            print(f"[plan] skip cache {d} (no hubs.csv + enum_children.json)")
            continue
        for key in _read_hubs(hubs_csv):
            have.setdefault(key, str(d))
        tpath = d / "enum_timings.json"
        if tpath.exists():
            for h in json.load(open(tpath)).get("per_hub", []):
                key = (h.get("hub_input") or h.get("hub_key"), int(h["depth"]))
                secs[key] = (
                    float(h.get("enumeration_s", 0))
                    + float(h.get("reward_gen_s", 0))
                    + float(h.get("flow_extract_s", 0))
                )
    return have, secs


def _depth_cost_model(secs, cached_keys):
    """p75 measured seconds per hub, by depth — the packing estimate. Depths never measured fall
    back to the global p75 (conservative: the unmeasured depths are the shallow, child-heavy ones).
    """
    by_depth: dict = {}
    for (smiles, depth), s in secs.items():
        by_depth.setdefault(depth, []).append(s)
    model = {}
    for d, vals in by_depth.items():
        vals = sorted(vals)
        model[d] = vals[min(len(vals) - 1, int(0.75 * len(vals)))]
    allv = sorted(secs.values())
    model["_default"] = allv[min(len(allv) - 1, int(0.75 * len(allv)))] if allv else 200.0
    return model


def _est(model, depth):
    return float(model.get(depth, model["_default"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--analysis-dir", default=DEFAULT_ANALYSIS)
    ap.add_argument("--out-root", required=True, help="$SCRATCH dir to hold arms/ + manifest.json")
    ap.add_argument(
        "--cached",
        default=DEFAULT_CANONICAL,
        help="comma-separated enumeration dirs to reuse (each needs hubs.csv + enum_children.json)",
    )
    ap.add_argument(
        "--canonical",
        default=DEFAULT_CANONICAL,
        help="the incumbent enumeration — its hubs.csv pins the fixed-set arm",
    )
    ap.add_argument("--n-hubs", type=int, default=200)
    ap.add_argument("--top-k-candidates", type=int, default=1000)
    ap.add_argument(
        "--slice-seconds",
        type=float,
        default=3600.0,
        help="estimated GPU seconds per slice (half the 2h debug limit — the worker only writes "
        "at the end, so an overrun loses the whole slice)",
    )
    ap.add_argument(
        "--max-hubs-per-slice",
        type=int,
        default=60,
        help="second guard on slice size — the depth-3-heavy arms rest on few measured hubs",
    )
    ap.add_argument(
        "--no-auto-cache",
        dest="auto_cache",
        action="store_false",
        help="do NOT fold finished <out-root>/enum/* slices back into the cache",
    )
    ap.set_defaults(auto_cache=True)
    a = ap.parse_args()

    out_root = Path(a.out_root)
    (out_root / "arms").mkdir(parents=True, exist_ok=True)
    cached_dirs = [d for d in a.cached.split(",") if d]
    if a.auto_cache:
        cached_dirs += sorted(
            str(p) for p in (out_root / "enum").glob("*") if (p / "enum_children.json").exists()
        )
    have, secs = _cached_index(cached_dirs)
    model = _depth_cost_model(secs, set(have))
    print(
        f"[plan] cache: {len(have)} hubs enumerated across {len(cached_dirs)} dir(s); "
        f"per-depth p75 seconds {({k: round(v, 1) for k, v in sorted(model.items(), key=str)})}"
    )

    manifest = {
        "analysis_dir": a.analysis_dir,
        "canonical": a.canonical,
        "cached_sources": cached_dirs,
        "n_hubs": a.n_hubs,
        "slice_seconds": a.slice_seconds,
        "arms": [],
    }
    for arm in ARMS:
        adir = out_root / "arms" / arm["name"]
        (adir / "slices").mkdir(parents=True, exist_ok=True)
        n_hubs = int(arm.get("n_hubs", a.n_hubs))  # per-arm override (the deeper diagnostic arm)
        cmd = [
            sys.executable,
            str(PICK_HUBS),
            "--records",
            str(Path(a.analysis_dir) / "records.csv"),
            "--out",
            str(adir / "hubs.csv"),
            "--n-hubs",
            str(n_hubs),
            "--top-k-candidates",
            str(a.top_k_candidates),
            *arm["args"],
        ]
        if arm.get("needs_canonical"):
            cmd += ["--restrict-to", str(Path(a.canonical) / "hubs.csv")]
        subprocess.run(cmd, check=True)

        hubs = _read_hubs(adir / "hubs.csv")
        new = [h for h in hubs if h not in have]
        # Slice the NEW hubs in the arm's own walk order, so a truncated arm still has a usable
        # top-of-ranking prefix rather than a hole in the middle.
        slices, cur, cur_s = [], [], 0.0
        for h in new:
            e = _est(model, h[1])
            if cur and (cur_s + e > a.slice_seconds or len(cur) >= a.max_hubs_per_slice):
                slices.append((cur, cur_s))
                cur, cur_s = [], 0.0
            cur.append(h)
            cur_s += e
        if cur:
            slices.append((cur, cur_s))

        slice_meta = []
        for i, (rows, est_s) in enumerate(slices):
            spath = adir / "slices" / f"slice{i:02d}.csv"
            with open(spath, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["smiles", "depth"])
                w.writerows(rows)
            slice_meta.append(
                {"index": i, "file": str(spath), "n_hubs": len(rows), "est_seconds": round(est_s)}
            )
        est_total = sum(s["est_seconds"] for s in slice_meta)
        manifest["arms"].append(
            {
                "name": arm["name"],
                "label": arm["label"],
                "pick_args": arm["args"]
                + (["--restrict-to"] if arm.get("needs_canonical") else []),
                "dir": str(adir),
                "hubs_csv": str(adir / "hubs.csv"),
                "n_hubs": len(hubs),
                "n_cached": len(hubs) - len(new),
                "n_new": len(new),
                "slices": slice_meta,
                "est_new_seconds": est_total,
            }
        )
        print(
            f"[plan] {arm['name']:<20} {len(hubs)} hubs | cached {len(hubs) - len(new):3d} | "
            f"new {len(new):3d} -> {len(slice_meta)} slice(s), est {est_total / 3600:.1f} h"
        )
        # Newly planned hubs become cache for later arms in this same plan (they get enumerated
        # once, in whichever arm claims them first).
        for h in new:
            have.setdefault(h, f"PLANNED:{arm['name']}")

    total_h = sum(x["est_new_seconds"] for x in manifest["arms"]) / 3600
    n_slices = sum(len(x["slices"]) for x in manifest["arms"])
    manifest["est_total_hours"] = round(total_h, 2)
    manifest["n_slices"] = n_slices
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        f"[plan] TOTAL {n_slices} slice job(s), est {total_h:.1f} h GPU -> "
        f"{out_root / 'manifest.json'}"
    )


if __name__ == "__main__":
    main()
