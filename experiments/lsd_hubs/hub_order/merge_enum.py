#!/usr/bin/env python
"""Assemble each hub-ordering arm's enumeration from the shared hub cache (Logs/053).

Enumeration is deterministic — same checkpoint, same frozen library, same ``--enum-max-children`` —
so a hub enumerated once is reusable by every arm that selects it. The arms overlap heavily (the
incumbent's 200 hubs cover 91 of ``flow_top``'s and all of ``cand_order_fixedset``'s), so the GPU
only ever enumerates a hub once, and this script reassembles each arm's full 200 **in that arm's own
walk order** — which is the thing the strategy is defined by.

Sources = the incumbent enumeration + every finished slice under ``<out-root>/enum/``. For each arm
it writes a self-contained enumeration dir the campaign drivers can be pointed at unmodified:

    <out-root>/merged/<arm>/
        hubs.csv                 the arm's walk order (copied)
        enum_children.json       {"hubs": [...]} in walk order  <- --enum-children
        enum_timings.json        merged per-hub measured times  <- Logs/039 compute-time
        enum_per_hub.json        per-hub child counts (parity with the worker's output)
        pick_hubs_timing.json    the arm's Stage-2 hub-pick wall-clock
        meta.json                provenance: which source supplied each hub

A hub in an arm's ``hubs.csv`` with no enumeration anywhere is a hard error — a silently short arm
would read as "this ordering found fewer modes" when it actually just has holes.

    python experiments/lsd_hubs/hub_order/merge_enum.py --out-root $SCRATCH/rgfn_runs/lsdflow/hub_order
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from validation.lsdflow.metrics.cost.compute_time import EnumTimings  # noqa: E402


def _hub_id(h: dict) -> str:
    """Unique per-hub identity — raw stereo-aware SMILES (``hub_key`` is stereo-stripped and
    collides across stereoisomers, which were enumerated separately)."""
    return h.get("hub_input") or h["hub_key"]


def _read_hubs(path: Path):
    with open(path) as fh:
        return [(r["smiles"], int(r["depth"])) for r in csv.DictReader(fh)]


def _index_sources(source_dirs):
    """-> ({hub_input: hub_dict}, {hub_input: source_dir}, [timing paths])."""
    children: dict = {}
    origin: dict = {}
    timing_paths = []
    for d in source_dirs:
        d = Path(d)
        ec = d / "enum_children.json"
        if not ec.exists():
            continue
        data = json.load(open(ec))
        n_new = 0
        for h in data.get("hubs", []):
            hid = _hub_id(h)
            if hid in children:  # first source wins; enumeration is deterministic
                continue
            children[hid] = h
            origin[hid] = str(d)
            n_new += 1
        if (d / "enum_timings.json").exists():
            timing_paths.append(d / "enum_timings.json")
        print(f"[merge] source {d.name}: {len(data.get('hubs', []))} hubs ({n_new} new)")
    return children, origin, timing_paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-root", required=True)
    ap.add_argument(
        "--extra-sources",
        default="",
        help="comma-separated extra enumeration dirs (the incumbent is taken from manifest.json)",
    )
    ap.add_argument("--arms", default="", help="comma-separated arm names (default: all)")
    a = ap.parse_args()

    out_root = Path(a.out_root)
    manifest = json.load(open(out_root / "manifest.json"))
    sources = [manifest["canonical"]]
    sources += [d for d in a.extra_sources.split(",") if d]
    sources += sorted(
        str(p) for p in (out_root / "enum").glob("*") if (p / "enum_children.json").exists()
    )
    children, origin, timing_paths = _index_sources(sources)
    timings = EnumTimings.merge(timing_paths) if timing_paths else None
    print(
        f"[merge] {len(children)} distinct hubs cached; timings for {len(timings.per_hub) if timings else 0}"
    )

    wanted = {x.strip() for x in a.arms.split(",") if x.strip()}
    n_ok = 0
    for arm in manifest["arms"]:
        if wanted and arm["name"] not in wanted:
            continue
        hubs = _read_hubs(Path(arm["hubs_csv"]))
        missing = [h for h in hubs if h[0] not in children]
        if missing:
            print(
                f"[merge] SKIP {arm['name']}: {len(missing)}/{len(hubs)} hubs not enumerated yet "
                f"(first: {missing[0][0][:60]})"
            )
            continue

        mdir = out_root / "merged" / arm["name"]
        mdir.mkdir(parents=True, exist_ok=True)
        ordered = []
        per_hub = []
        for smiles, depth in hubs:
            h = children[smiles]
            if int(h["depth"]) != depth:  # the pair (smiles, depth) IS the hub identity upstream
                print(
                    f"[merge] WARNING depth mismatch for {smiles[:50]}: "
                    f"hubs.csv={depth} enum={h['depth']}"
                )
            ordered.append(h)
            per_hub.append(
                {
                    "hub": smiles,
                    "depth": int(h["depth"]),
                    "n_enumerated_paths": len(h["children"]),
                    "n_records": len(h["children"]),
                }
            )
        json.dump({"hubs": ordered}, open(mdir / "enum_children.json", "w"))
        json.dump({"per_hub": per_hub}, open(mdir / "enum_per_hub.json", "w"), indent=2)
        shutil.copy(arm["hubs_csv"], mdir / "hubs.csv")
        pt = Path(arm["dir"]) / "pick_hubs_timing.json"
        if pt.exists():
            shutil.copy(pt, mdir / "pick_hubs_timing.json")

        # Timings restricted to this arm's hubs, so meta totals describe THIS arm's enumeration.
        if timings is not None:
            arm_per_hub = [timings.per_hub[s] for s, _ in hubs if s in timings.per_hub]
            comps = ("enumeration_s", "reward_gen_s", "flow_extract_s", "unattributed_s")
            json.dump(
                {
                    "meta": {
                        **{
                            k: v
                            for k, v in timings.meta.items()
                            if k not in ("totals_s", "n_hubs", "n_children")
                        },
                        "arm": arm["name"],
                        "n_hubs": len(arm_per_hub),
                        "n_children": sum(int(h.get("n_children", 0)) for h in arm_per_hub),
                        "totals_s": {
                            k: round(sum(float(h.get(k, 0.0)) for h in arm_per_hub), 3)
                            for k in comps
                        },
                    },
                    "per_hub": arm_per_hub,
                },
                open(mdir / "enum_timings.json", "w"),
                indent=2,
            )
            n_timed = len(arm_per_hub)
        else:
            n_timed = 0

        src_counts = Counter(Path(origin[s]).name for s, _ in hubs)
        json.dump(
            {
                "arm": arm["name"],
                "label": arm["label"],
                "n_hubs": len(hubs),
                "n_children": sum(len(h["children"]) for h in ordered),
                "n_timed_hubs": n_timed,
                "hub_sources": dict(src_counts),
                "analysis_dir": manifest["analysis_dir"],
            },
            open(mdir / "meta.json", "w"),
            indent=2,
        )
        n_ok += 1
        print(
            f"[merge] {arm['name']:<20} {len(hubs)} hubs / "
            f"{sum(len(h['children']) for h in ordered):,} children, timed {n_timed} "
            f"-> {mdir}"
        )
    print(f"[merge] {n_ok} arm(s) assembled under {out_root / 'merged'}")


if __name__ == "__main__":
    main()
