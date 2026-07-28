#!/bin/bash
# Publish a cell's measured compute timings: merge the parallel hub slices produced by
# submit_timing.sh and copy the result into the cell's real enum dir as enum_timings.json, where
# run_campaign.py auto-detects it (beside enum_children.json).
#
# Kept SEPARATE from submit_timing.sh on purpose: the re-run writes to an isolated scratch tree, and
# only this script touches the validated enum dir. It refuses to publish a partial merge (every hub
# in hubs.csv must be timed) so a half-finished slice set can never masquerade as a full measurement.
#
# Usage:  bash experiments/lsd_hubs/matrix16/merge_timings.sh <generator> <target> [--force]
set -uo pipefail

GEN=${1:?usage: merge_timings.sh <generator> <target> [--force]}
TGT=${2:?usage: merge_timings.sh <generator> <target> [--force]}
FORCE=${3:-}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1
export PATH="/home/markymoo/miniconda3/envs/rgfn/bin:$PATH"

eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || exit 1
TIMING_ROOT=${TIMING_ROOT:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16_timing}

python - "$TIMING_ROOT/$CELL_TAG" "$ENUM_DIR" "$FORCE" <<'PY'
import json, sys, glob
from pathlib import Path
sys.path.insert(0, ".")
from validation.lsdflow.metrics.cost.compute_time import EnumTimings

src_root, enum_dir, force = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
slices = sorted(glob.glob(str(src_root / "slice*" / "enum_timings.json")))
if not slices:
    raise SystemExit(f"[merge] no slices under {src_root} — run submit_timing.sh first")

merged = EnumTimings.merge(slices) if len(slices) > 1 else EnumTimings.load(slices[0])

# Completeness gate: every hub the cell actually enumerated must have a timing row, else the summed
# per-strategy compute silently under-reports (a walked-but-untimed hub contributes 0).
import csv
want = {r["smiles"] for r in csv.DictReader(open(enum_dir / "hubs.csv"))}
have = set(merged.per_hub)
missing = want - have
print(f"[merge] {len(slices)} slice(s) -> {len(have)}/{len(want)} hubs timed, setup_s={merged.setup_s:.1f}")
for k, v in (merged.meta.get("totals_s") or {}).items():
    print(f"[merge]   {k:<18} {v:>10.1f}s")
if missing and force != "--force":
    raise SystemExit(
        f"[merge] REFUSING to publish: {len(missing)} enumerated hub(s) have no timing row "
        f"(e.g. {sorted(missing)[0][:60]}). Re-run the missing slice, or pass --force to accept a "
        f"partial (the compute-time section will then UNDER-report)."
    )
out = enum_dir / "enum_timings.json"
payload = {"meta": {**merged.meta, "setup_s": merged.setup_s}, "per_hub": list(merged.per_hub.values())}
json.dump(payload, open(out, "w"), indent=2)
print(f"[merge] wrote {out}  ({len(have)} hubs) — re-run run_cell_campaign.sh to pick it up")
PY
