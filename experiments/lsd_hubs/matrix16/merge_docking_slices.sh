#!/bin/bash
# Union a docking cell's hub slices into the single enumeration the campaign reads.
#
# submit_docking_cell.sh writes each slice to `enum/slice<i>of<n>/`; the slices hold DISJOINT hubs
# (round-robin), so merging is a union of `enum_children.json`'s hub list, a concatenation of
# `enumerated_records.csv`, and a union of the per-hub timings. Kept separate from the submit script
# so a partial slice set is never silently promoted to "the cell's enumeration".
#
# Refuses to publish unless every hub in the cell's hubs.csv is present exactly once — a missing
# slice would otherwise look like a smaller-but-valid enumeration, and the campaign has no way to
# tell that apart from a genuinely small pool. --force accepts a partial (recorded in meta).
#
# Usage:  bash experiments/lsd_hubs/matrix16/merge_docking_slices.sh <generator> <target> [--force]
set -uo pipefail

GEN=${1:?usage: merge_docking_slices.sh <generator> <target> [--force]}
TGT=${2:?usage: merge_docking_slices.sh <generator> <target> [--force]}
FORCE=${3:-}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/home/markymoo/miniconda3/envs/rgfn/bin:$PATH"
eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || exit 1

python - "$ENUM_DIR" "$FORCE" <<'PY'
import csv, glob, json, os, sys
from pathlib import Path

enum_dir, force = Path(sys.argv[1]), sys.argv[2]
slices = sorted(glob.glob(str(enum_dir / "slice*of*" / "enum_children.json")))
if not slices:
    raise SystemExit(f"[merge] no slices under {enum_dir} — run submit_docking_cell.sh first")

want = [r["smiles"] for r in csv.DictReader(open(enum_dir / "hubs.csv"))]
hubs, seen, dup = [], set(), []
for p in slices:
    for h in json.load(open(p)).get("hubs", []):
        key = h.get("hub_input") or h["hub_key"]
        if key in seen:
            dup.append(key)
            continue
        seen.add(key)
        hubs.append(h)

missing = [h for h in want if h not in seen]
n_children = sum(len(h.get("children", [])) for h in hubs)
print(f"[merge] {len(slices)} slice(s) -> {len(hubs)}/{len(want)} hubs, {n_children:,} children")
if dup:
    print(f"[merge] WARNING {len(dup)} duplicate hub(s) across slices (kept first) — slices should be disjoint")
if missing and force != "--force":
    raise SystemExit(
        f"[merge] REFUSING to publish: {len(missing)} hub(s) have no slice "
        f"(e.g. {missing[0][:60]}). Re-run the missing slice, or --force to accept a partial."
    )

json.dump({"hubs": hubs}, open(enum_dir / "enum_children.json", "w"))
print(f"[merge] wrote {enum_dir/'enum_children.json'}")

# Records: concatenate, keeping one header.
recs = sorted(glob.glob(str(enum_dir / "slice*of*" / "enumerated_records.csv")))
if recs:
    rows, cols = [], None
    for p in recs:
        r = csv.DictReader(open(p))
        cols = cols or r.fieldnames
        rows.extend(list(r))
    with open(enum_dir / "enumerated_records.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print(f"[merge] wrote enumerated_records.csv ({len(rows):,} rows)")

# Timings: union per-hub, charge setup ONCE (median across slices — a single production run pays it
# once). Same convention as merge_timings.sh.
tim = sorted(glob.glob(str(enum_dir / "slice*of*" / "enum_timings.json")))
if tim:
    import statistics as st
    per, setups, metas = {}, [], []
    for p in tim:
        d = json.load(open(p)); m = d.get("meta", {})
        setups.append(float(m.get("setup_s", 0.0))); metas.append(m)
        for h in d.get("per_hub", []):
            per[h.get("hub_input") or h["hub_key"]] = h
    keys = ("enumeration_s", "reward_gen_s", "flow_extract_s", "unattributed_s")
    totals = {k: round(sum(float(v.get(k, 0.0)) for v in per.values()), 3) for k in keys}
    totals = {k: v for k, v in totals.items() if v}
    json.dump({
        "meta": {
            "setup_s": round(st.median(setups) if setups else 0.0, 3),
            "slice_setups_s": [round(s, 3) for s in setups],
            "merged_from": [str(p) for p in tim],
            "n_slices": len(tim),
            "n_hubs": len(per),
            "n_children": sum(int(v.get("n_children", 0)) for v in per.values()),
            "totals_s": totals,
            "device": metas[0].get("device") if metas else None,
            "reward_name": metas[0].get("reward_name") if metas else None,
            "component_split": metas[0].get("component_split") if metas else None,
            "partial": bool(missing),
        },
        "per_hub": list(per.values()),
    }, open(enum_dir / "enum_timings.json", "w"), indent=2)
    print(f"[merge] wrote enum_timings.json ({len(per)} hubs) totals={totals}")

print("[merge] now run: run_cell_campaign.sh " + enum_dir.parent.name.replace("_", " ", 1))
PY
