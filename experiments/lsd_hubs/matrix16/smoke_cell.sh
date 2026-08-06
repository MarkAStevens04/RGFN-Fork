#!/bin/bash
# Tiny end-to-end smoke for ONE matrix cell — prove the whole path before spending GPU-days on it.
#
# WHY THIS EXISTS AS A SCRIPT rather than ad-hoc commands. Every docking bug we have hit so far was
# cheap to catch and expensive to discover late:
#   * PermissionError on <repo>/reward_bridge  ($HOME is read-only on compute nodes) — 56 s to catch
#   * cleanup trap importing glue from the generator's env (no gin)                   — 56 s
#   * the docking gate reading the ReLU'd column instead of raw energy (qualifies NOTHING, silently)
#   * no partial flush -> a walltime kill discards the whole slice        — cost ~4 GPU-hours to learn
# The first two were found by an ad-hoc smoke; the last two were found by reading, AFTER a 6-slice
# launch had already been cancelled once. So the smoke is now a first-class step with a fixed recipe.
#
# ISOLATION IS THE POINT. My ad-hoc smokes wrote a 200-trajectory sample and a 20-hub hubs.csv into
# the REAL cell directory, where submit_docking_cell.sh's own RESUME guard would have seen
# records.csv and skipped the production sample — quietly building a headline cell on 200
# trajectories. This script therefore ALWAYS writes under a `_smoke` scratch tag via MATRIX16_SCRATCH
# and refuses to run if that would collide with the real tree.
#
# What it asserts (not just "did it exit 0"):
#   1. artifacts exist and are non-empty
#   2. for a DOCKING cell: `reward` is the RAW energy (negative, lower-is-better) and
#      `log_reward == beta*clip(ReLU(-raw))` on EVERY row — the two-column contract
#   3. the calibrated gate actually qualifies a non-zero, non-total fraction (a 0% or 100% gate means
#      the wrong column or the wrong bar)
#   4. oracle failures are a small fraction (a wedged GPU shows up here, not as bad chemistry)
#
# Usage:  bash experiments/lsd_hubs/matrix16/smoke_cell.sh <generator> <target> [N_TRAJ] [N_HUBS]
#         defaults: N_TRAJ=200 N_HUBS=6   (docking: ~200 mols ~= 2 min; surrogate: seconds)
# Then:   read the verdict. Only launch production if it says SMOKE PASSED.
set -uo pipefail

GEN=${1:?usage: smoke_cell.sh <generator> <target> [n_traj] [n_hubs]}
TGT=${2:?usage: smoke_cell.sh <generator> <target> [n_traj] [n_hubs]}
N_TRAJ=${3:-200}
N_HUBS=${4:-6}

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO" || exit 1

# Isolated scratch tree — never the production one.
export MATRIX16_SCRATCH="${MATRIX16_SCRATCH:-/scratch/markymoo/rgfn_runs/lsdflow/matrix16_smoke}"
case "$MATRIX16_SCRATCH" in
    */matrix16) echo "REFUSING: MATRIX16_SCRATCH points at the production tree ($MATRIX16_SCRATCH)."; exit 2 ;;
esac
echo "=== SMOKE $GEN/$TGT  n_traj=$N_TRAJ n_hubs=$N_HUBS  scratch=$MATRIX16_SCRATCH ==="

source /home/markymoo/miniconda3/etc/profile.d/conda.sh; conda activate base
eval "$(python experiments/lsd_hubs/matrix16/manifest.py --emit "$GEN" "$TGT")" || exit 1
[ "$STATUS" = "ready" ] || { echo "SMOKE ABORT: cell is '$STATUS'"; exit 1; }

if [ "$REWARD_TYPE" = "docking" ]; then
    SUB=experiments/lsd_hubs/matrix16/submit_docking_cell.sh
    PART=${PART:-debug}; TIME=${TIME:-00:45:00}
else
    SUB=experiments/lsd_hubs/matrix16/submit_cell.sh
    PART=${PART:-debug}; TIME=${TIME:-00:30:00}
fi

JOB=$(STAGE=all N_TRAJ="$N_TRAJ" N_HUBS="$N_HUBS" TOPK=$((N_HUBS * 5)) SAMPLE_BATCH=100 \
      ENUM_MAX=${ENUM_MAX:-40} RESUME=0 \
      sbatch --parsable -J "smoke_${CELL_TAG}" -p "$PART" --time="$TIME" \
      --export=ALL,MATRIX16_SCRATCH="$MATRIX16_SCRATCH" "$SUB" "$GEN" "$TGT") || {
    echo "SMOKE ABORT: sbatch failed"; exit 1; }
echo "=== smoke job $JOB submitted; waiting ==="

while squeue -j "$JOB" -h -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|COMPLETING"; do sleep 15; done
STATE=$(sacct -j "$JOB" -X -n -o State 2>/dev/null | head -1 | tr -d ' ')
echo "=== smoke job $JOB finished: $STATE ==="
LOG=$(ls -t /scratch/markymoo/rgfn_runs/smoke_${CELL_TAG}-${JOB}.out 2>/dev/null | head -1)
[ -n "$LOG" ] && tail -25 "$LOG"

python - "$SAMPLE_DIR" "$ENUM_DIR" "$REWARD_TYPE" "$MODE_REWARD_THRESHOLD" "$HIGHER_IS_BETTER" "$CONFIG" <<'PY'
import csv, glob, json, math, sys
from pathlib import Path

sample, enum, rtype, bar, hib, cfg_path = sys.argv[1:7]
bar = float(bar); hib = hib == "true"
fail = []

def chk(ok, msg):
    print(("  OK   " if ok else "  FAIL ") + msg)
    if not ok:
        fail.append(msg)

print("\n=== SMOKE ASSERTIONS ===")
recs_p = Path(sample) / "records.csv"
chk(recs_p.exists() and recs_p.stat().st_size > 0, f"records.csv exists and is non-empty")
if not recs_p.exists():
    raise SystemExit("SMOKE FAILED: no records.csv")
rows = list(csv.DictReader(open(recs_p)))
chk(len(rows) > 0, f"records.csv has rows ({len(rows)})")

# enum_children.json lives in the SLICE dir for a docking cell (enum/slice<i>of<n>/) and at the
# enum root for a surrogate cell. Accept either rather than hard-coding one.
cands = [Path(enum) / "enum_children.json", *sorted(Path(enum).glob("slice*of*/enum_children.json"))]
enum_p = next((c for c in cands if c.exists()), None)
if enum_p:
    hubs = json.load(open(enum_p)).get("hubs", [])
    nk = sum(len(h.get("children", [])) for h in hubs)
    chk(len(hubs) > 0 and nk > 0, f"enum_children.json: {len(hubs)} hubs / {nk} children ({enum_p.parent.name})")
else:
    chk(False, f"enum_children.json exists (looked in {enum} and its slice*/ dirs)")

if rtype == "docking":
    rew = [float(r["reward"]) for r in rows]
    lr = [float(r["log_reward"]) for r in rows]
    finite = [(a, b) for a, b in zip(rew, lr) if a == a]
    nan = len(rew) - len(finite)
    chk(nan / max(len(rew), 1) < 0.20,
        f"oracle failure rate {100*nan/max(len(rew),1):.1f}% < 20% (higher => wedged GPU)")

    # `reward` must be the RAW energy. Most raw Vina scores are negative, but a clashing pose can be
    # POSITIVE, so require a large majority rather than all -- and require the mean to be clearly
    # negative, which the ReLU'd column (>= 0 by construction) can never be.
    neg = sum(1 for a, _ in finite if a < 0)
    mean = sum(a for a, _ in finite) / max(len(finite), 1)
    chk(neg / max(len(finite), 1) > 0.80 and mean < 0,
        f"`reward` is RAW energy: {neg}/{len(finite)} negative, mean {mean:.2f} "
        f"(a ReLU'd column would be >=0 everywhere)")

    # Two-column contract WITHOUT hard-coding beta/clip: both live in the cell's own training
    # config, which is gin for scent/rgfn and unreadable by OmegaConf. FIT BOTH from the data.
    #
    # Deriving beta alone from log_reward/|raw| is not enough, and getting that wrong twice is why
    # this is written out: SCENT's DockingBridgeProxy does NOT clip (its log_reward reached 56.8,
    # so an assumed clip=10 flagged 168/200 good rows), while FragGFN's DockingBridgeReward DOES
    # clip at 10 (so 60/200 rows sit at exactly beta*clip=40 and their ratio is < beta). Each
    # matches ITS OWN training transform, which is what the flow terms require -- so the assertion
    # has to accommodate both shapes rather than pick one.
    #
    # Fit: beta from the UNCAPPED rows (log_reward below its max), then clip = max(log_reward)/beta,
    # then require log_reward == beta*min(ReLU(-raw), clip) everywhere.
    pairs = [(a, b) for a, b in finite if -a > 1e-6 and b > 0]
    if pairs:
        lr_max = max(b for _, b in pairs)
        unc = [(a, b) for a, b in pairs if b < lr_max - 1e-9]
        ratios = sorted(b / -a for a, b in (unc or pairs))
        beta = ratios[len(ratios) // 2]
        clip = lr_max / beta if beta else float("inf")
        bad = sum(1 for a, b in pairs if abs(b - beta * min(max(-a, 0.0), clip)) > 1e-3)
        capped = sum(1 for _, b in pairs if abs(b - lr_max) < 1e-9)
        chk(bad == 0,
            f"two-column contract: log_reward == {beta:g}*min(ReLU(-raw), {clip:g}) on "
            f"{len(pairs)-bad}/{len(pairs)} rows (fitted beta {beta:g}, clip {clip:g}; "
            f"{capped} rows at the cap)")
    else:
        chk(False, "two-column contract: no rows with a positive transform to check")

    q = sum(1 for a, _ in finite if (a > bar) == hib)
    frac = q / max(len(finite), 1)
    chk(0.0 < frac < 1.0,
        f"gate {'>' if hib else '<'}{bar:g} qualifies {q}/{len(finite)} = {100*frac:.1f}%"
        + ("  [NOTE >90%: gate is nearly non-binding for this cell — consider the stricter variant]"
           if frac > 0.90 else ""))

print("\n" + ("SMOKE PASSED — safe to launch production" if not fail
              else f"SMOKE FAILED ({len(fail)} assertion(s)) — DO NOT launch"))
raise SystemExit(0 if not fail else 1)
PY
