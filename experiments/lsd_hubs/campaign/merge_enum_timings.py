#!/usr/bin/env python
"""Merge the timed-enumeration slices into one ``enum_timings.json`` (Logs/039).

The 200-hub timed re-run is split across parallel debug jobs (``submit_scent_seh_enum_timed.sh``),
each writing ``slice<i>/enum_timings.json`` with the per-hub enumeration / reward-gen / flow-extract
wall-clock for its hub subset. This unions them into a single ``enum_timings.json`` that the four
campaign drivers read (by default) beside ``enum_children.json``, so every operating point can
attribute measured time over its actual hub walk.

Integrity: the merged per-hub child counts are checked against the canonical enumeration's
``enum_per_hub.json`` (the timed re-run must reproduce the same children, since it re-runs the exact
deterministic enumeration) — a mismatch means a slice enumerated a different hub set and is flagged.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/merge_enum_timings.py \
        --timed-dir /scratch/.../lsdflow/campaign_enum_seh_timed \
        --canonical /scratch/.../lsdflow/campaign_enum_seh_70363 \
        --out /scratch/.../lsdflow/campaign_enum_seh_70363/enum_timings.json
"""
import argparse
import json
from pathlib import Path

from validation.lsdflow.metrics.cost.compute_time import EnumTimings


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timed-dir", required=True, help="dir holding slice*/enum_timings.json")
    ap.add_argument(
        "--canonical",
        required=True,
        help="canonical enum dir (for the enum_per_hub.json child-count integrity check)",
    )
    ap.add_argument(
        "--out",
        required=True,
        help="merged enum_timings.json (write it beside the canonical enum_children.json)",
    )
    ap.add_argument(
        "--pick-hubs-timing",
        default=None,
        help="optional pick_hubs_timing.json to copy beside --out (Stage-2 hub-pick time)",
    )
    a = ap.parse_args()

    timed = Path(a.timed_dir)
    slice_files = sorted(
        timed.glob("slice*/enum_timings.json"),
        key=lambda p: p.parent.name,
    )
    if not slice_files:
        raise SystemExit(f"no slice*/enum_timings.json under {timed}")
    print(f"[merge] {len(slice_files)} slices:")
    for p in slice_files:
        print(f"  {p.parent.name}/{p.name}")

    merged = EnumTimings.merge(slice_files)

    # Integrity: per-hub child counts vs the canonical enumeration.
    canon = json.load(open(Path(a.canonical) / "enum_per_hub.json"))["per_hub"]
    # canonical enum_per_hub is keyed by hub INPUT smiles, not hub_key; join on children count only
    # for a global sanity check, and per-hub via the enum_children hub_key where possible.
    canon_children = {h.get("hub"): h.get("n_records", 0) for h in canon}
    canon_total = sum(canon_children.values())
    merged_total = sum(int(v.get("n_children", 0)) for v in merged.per_hub.values())
    print(
        f"[merge] hubs: {len(merged.per_hub)} timed vs {len(canon)} canonical | "
        f"children: {merged_total:,} timed vs {canon_total:,} canonical"
    )
    if merged_total != canon_total:
        print(
            "[merge] WARNING total child count differs from canonical — a slice may have enumerated "
            "a different hub set (check ENUM_MAX / hubs slice files)."
        )
    # per-hub check by hub_input where available
    mism = 0
    for v in merged.per_hub.values():
        exp = canon_children.get(v.get("hub_input"))
        if exp is not None and int(v.get("n_children", 0)) != int(exp):
            mism += 1
    print(f"[merge] per-hub child-count mismatches (by hub_input): {mism}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": merged.meta, "per_hub": list(merged.per_hub.values())}
    out.write_text(json.dumps(payload, indent=2))
    t = merged.meta["totals_s"]
    print(
        f"[merge] wrote {out}\n"
        f"[merge] setup {merged.setup_s:.1f}s (slices {merged.meta['slice_setups_s']}) | "
        f"enum {t['enumeration_s']:.1f}s reward {t['reward_gen_s']:.1f}s flow {t['flow_extract_s']:.1f}s "
        f"over {merged.meta['n_hubs']} hubs / {merged.meta['n_children']:,} children"
    )

    if a.pick_hubs_timing:
        src = Path(a.pick_hubs_timing)
        if src.exists():
            (out.parent / "pick_hubs_timing.json").write_text(src.read_text())
            print(f"[merge] copied {src.name} -> {out.parent / 'pick_hubs_timing.json'}")


if __name__ == "__main__":
    main()
