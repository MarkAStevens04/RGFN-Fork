#!/bin/bash
# Make a verified cell's train/ directory READ-ONLY. The structural fix for the overwrite hazard.
#
# WHY THIS EXISTS, AND WHY IT IS NOT TIDINESS. Re-invoking any generator runner OVERWRITES that
# cell's trace.csv, candidates.csv, pairs.csv, timing.json and run_config.yaml. That is not
# hypothetical: it destroyed s3gfn_seh/seed43's entire training history, unrecoverably, and
# truncated s3gfn_seh/seed42's budget-faithful candidates.csv from 2,000 rows to 20,000. The
# artifacts a re-run cannot recreate -- a trained checkpoint, the trace that evidences its budget,
# and (for SCENT) recipes observable only while training -- are exactly the ones a stray re-invocation
# lands on.
#
# "Remember not to re-invoke a runner against a copied cell" is a discipline, and disciplines fail
# silently at 3am on cell 71 of 108. `chmod -R a-w` makes it a property of the filesystem: the
# runner gets EACCES and dies instead of quietly rewriting six months of compute. This matters most
# right when the copy-forward step is staging ~40 cells that would otherwise be destroyable.
#
# WHAT IT REFUSES TO DO. Freeze an unverified cell. A read-only wrapper around a broken artifact is
# worse than no wrapper, because it looks finished. Verification runs first, every time, and a
# failure aborts before anything is chmod'ed.
#
# Usage:
#   experiments/benchmark_v2/tools/freeze_cell.sh scent/seh/42 [--arm a]
#   experiments/benchmark_v2/tools/freeze_cell.sh scent/seh/42 --unfreeze     # deliberate re-work
#   experiments/benchmark_v2/tools/freeze_cell.sh --all --arm a               # sweep the grid
#
# Exit 0 = frozen (or already frozen). Non-zero = refused, and the reason is printed.

set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python}"

ARM=a
UNFREEZE=0
ALL=0
CELL=""
STRICT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --arm) ARM="$2"; shift 2 ;;
        --unfreeze) UNFREEZE=1; shift ;;
        --all) ALL=1; shift ;;
        --strict) STRICT=1; shift ;;   # also require a provenance row before freezing
        -h|--help) sed -n '2,30p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) CELL="$1"; shift ;;
    esac
done

if [ "$ALL" = 0 ] && [ -z "$CELL" ]; then
    echo "usage: freeze_cell.sh GEN/TARGET/SEED [--arm a] [--unfreeze] [--strict]" >&2
    echo "       freeze_cell.sh --all [--arm a]" >&2
    exit 2
fi

# ---------------------------------------------------------------------------------------------
freeze_one() {
    local cell="$1"
    local gen tgt seed train_dir
    gen="${cell%%/*}"; local rest="${cell#*/}"; tgt="${rest%%/*}"; seed="${rest##*/}"

    train_dir=$("$PY" "$HERE/manifest.py" --emit "$gen" "$tgt" "$seed" --arm "$ARM" 2>/dev/null \
                | sed -n 's/^TRAIN_DIR=//p' | tr -d "'")
    if [ -z "$train_dir" ]; then
        echo "REFUSED $cell arm$ARM: no such cell in grid.csv (or it has no arm $ARM)" >&2
        return 1
    fi

    if [ ! -d "$train_dir" ]; then
        echo "REFUSED $cell arm$ARM: $train_dir does not exist" >&2
        return 1
    fi

    if [ "$UNFREEZE" = 1 ]; then
        chmod -R u+w "$train_dir" || return 1
        echo "UNFROZEN $cell arm$ARM  $train_dir"
        echo "  NOTE: this cell is now overwritable again. Re-freeze as soon as the re-work lands."
        return 0
    fi

    # Already frozen? Idempotent, so a driver can call this unconditionally.
    if [ ! -w "$train_dir" ]; then
        echo "ok       $cell arm$ARM  already frozen"
        return 0
    fi

    # -- the gate. A read-only wrapper around a broken artifact just looks finished.
    if ! "$PY" "$HERE/verify_cell.py" --cell "$cell" --arm "$ARM" --stage train >/tmp/freeze_$$.log 2>&1; then
        echo "REFUSED $cell arm$ARM: verification failed -- NOT frozen" >&2
        sed -n '/FAIL/p' /tmp/freeze_$$.log | sed 's/^/    /' >&2
        rm -f /tmp/freeze_$$.log
        return 1
    fi
    rm -f /tmp/freeze_$$.log

    # -- provenance cross-check. A frozen artifact whose origin was never recorded defeats the one
    # thing this tree guarantees: that it can say what it contains. Warn by default (the row is the
    # producer's job, not the freezer's); --strict makes it a hard requirement.
    local n_rows
    n_rows=$("$PY" "$HERE/ledger.py" show --cell "$cell" --stage train 2>/dev/null | grep -c "^train" || true)
    if [ "${n_rows:-0}" -eq 0 ]; then
        if [ "$STRICT" = 1 ]; then
            echo "REFUSED $cell arm$ARM: no provenance row (--strict)" >&2
            return 1
        fi
        echo "  WARNING $cell has no PROVENANCE.csv row -- freezing an artifact whose origin was" >&2
        echo "          never recorded. Whoever produced it should run: ledger.py record ..." >&2
    fi

    # -- stamp BEFORE the chmod: once the directory is read-only nothing can be added to it.
    "$PY" - "$train_dir" <<'PYEOF'
import json, sys, datetime, pathlib
p = pathlib.Path(sys.argv[1]) / ".verified.json"
d = {}
if p.is_file():
    try:
        d = json.loads(p.read_text())
    except Exception:
        d = {}
d["frozen_at"] = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
p.write_text(json.dumps(d, indent=2, sort_keys=True))
PYEOF

    if ! chmod -R a-w "$train_dir"; then
        echo "REFUSED $cell arm$ARM: chmod failed" >&2
        return 1
    fi
    echo "FROZEN   $cell arm$ARM  $train_dir"
    return 0
}

# ---------------------------------------------------------------------------------------------
rc=0
if [ "$ALL" = 1 ]; then
    while read -r tag; do
        [ -z "$tag" ] && continue
        # tag is <gen>_<target>_s<seed>; recover the triple without guessing where the target ends
        seed="${tag##*_s}"
        rest="${tag%_s$seed}"
        gen="${rest%%_*}"
        tgt="${rest#${gen}_}"
        freeze_one "$gen/$tgt/$seed" || rc=1
    done < <("$PY" "$HERE/manifest.py" --list --arm "$ARM")
else
    freeze_one "$CELL" || rc=1
fi
exit $rc
