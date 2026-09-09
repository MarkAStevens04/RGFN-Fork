#!/usr/bin/env python
"""Verify a benchmark_v2 BACKUP against the copy manifests. Reads only the backup.

THE GAP THIS CLOSES. `backup_scratch_critical.sh`'s tier-1b assertion compares destination bytes to
source bytes and errors below 95%. That catches a PAYLOAD bug -- which is exactly what bit on
2026-09-08, when an allowlist copied 401 MB of a 57.2 GB tree and exited 0 -- but it cannot catch
CONTENT corruption, in transit or on the destination. Those are different failures and the backup
had a gauge for only one.

WHY IT READS ONLY THE BACKUP, WHICH IS THE WHOLE POINT. Every copied cell carries
``.copy_manifest.json``, keyed by DESTINATION-relative path, recording each file's size and md5 as
verified at copy time. So this needs no v1 access, no 236 GB ``ledger.tree_digest``, and it works
identically on a restored copy months from now **when the v1 tree may not exist at all**. That last
property is the one that matters: the backup exists precisely for the world in which scratch is gone,
and an integrity check that needs the source is no use in that world.

    ledger.py verify   have the v1 SOURCES moved since we copied them?   (reads v1)
    this               does the BACKUP still hold what we copied?        (reads only the backup)

They do not overlap, and neither substitutes for the other.

⛔ WHAT THIS CANNOT TELL YOU. It verifies the backup against the manifest, so it cannot detect a file
that was already wrong at copy time in a way the manifest recorded too. That is what the copy-time
md5 was for: `copy_forward.py` re-hashed every file after landing it and refused to record a
mismatch. This is the second line, not the first.

COVERAGE IS ASSERTED, NOT ASSUMED. A verifier that only compares the files it happens to FIND is the
allowlist bug wearing a different hat -- it would pass cleanly on a backup missing 99% of its
payload, because every file it found would match. So a manifest entry with no corresponding file is
a FAILURE, and files-checked is reported against files-in-manifest every run.

THREE OUTCOMES, KEPT DISTINCT, because collapsing the last two loses the diagnosis:
    match     present, right size, right md5
    MISMATCH  present but the bytes differ            -> corruption
    MISSING   in the manifest, absent from the backup  -> payload loss

    python experiments/benchmark_v2/tools/verify_backup.py                    # whole tree
    python experiments/benchmark_v2/tools/verify_backup.py --cell saturn/seh/42
    python experiments/benchmark_v2/tools/verify_backup.py --sizes-only       # structure, NOT content
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Where tier 1b lands it: `rsync "$V2_SRC/" "$DEST/v2/"` in scripts/backup_scratch_critical.sh.
DEFAULT_BACKUP = Path(
    "/project/def-naeilum/naeilum/SDL3/RGFN_LSD_MarkStevens/backups/scratch_runs/rgfn_runs/v2"
)
MANIFEST = ".copy_manifest.json"


def _md5(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def find_cells(root: Path):
    """Every backed-up cell dir holding a copy manifest, as (tag, arm_dir)."""
    return sorted(
        (p.parent.parent.name + "/" + p.parent.name, p.parent)
        for p in root.rglob(MANIFEST)
    )


def verify_cell(arm_dir: Path, sizes_only: bool = False) -> dict:
    """Check one cell's files against its manifest. Returns a per-cell result dict."""
    mp = arm_dir / MANIFEST
    try:
        manifest = json.loads(mp.read_text())
    except Exception as exc:
        return {"dir": str(arm_dir), "error": f"unreadable {MANIFEST}: {exc}",
                "n_expected": 0, "matched": 0, "mismatched": [], "missing": [], "extra": []}

    files = manifest.get("files", {})
    matched, mismatched, missing = 0, [], []
    for rel, rec in sorted(files.items()):
        f = arm_dir / rel
        if not f.is_file():
            missing.append(rel)
            continue
        if f.stat().st_size != rec.get("bytes"):
            mismatched.append(f"{rel} (size {f.stat().st_size} != {rec.get('bytes')})")
            continue
        if sizes_only:
            matched += 1
            continue
        want = rec.get("md5")
        if want and _md5(f) != want:
            mismatched.append(f"{rel} (md5 differs)")
        else:
            matched += 1

    # Files present but unrecorded. NOT a failure -- the backup is wholesale and legitimately carries
    # things the manifest never claimed (arm_meta.json, .verified.json, README.md). Reported because
    # a large unexpected count means the manifest and the tree have drifted apart.
    on_disk = {
        str(p.relative_to(arm_dir))
        for p in arm_dir.rglob("*")
        if p.is_file() and p.name != MANIFEST
    }
    extra = sorted(on_disk - set(files))

    return {"dir": str(arm_dir), "n_expected": len(files), "matched": matched,
            "mismatched": mismatched, "missing": missing, "extra": extra}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--backup-root", type=Path, default=DEFAULT_BACKUP)
    ap.add_argument("--cell", help="one GEN/TARGET/SEED, e.g. saturn/seh/42 (default: whole tree)")
    ap.add_argument("--arm", default="a")
    ap.add_argument(
        "--sizes-only",
        action="store_true",
        help="check presence and size but NOT content. Fast structural pass; never reports "
        "'content verified' and is not a substitute for the default.",
    )
    a = ap.parse_args()

    if not a.backup_root.is_dir():
        sys.exit(
            f"no backup at {a.backup_root}\n"
            "Nothing has been backed up yet, or /project is not mounted (it is on the login nodes, "
            "not on compute)."
        )

    if a.cell:
        gen, tgt, seed = a.cell.split("/")
        cells = [(f"{gen}_{tgt}_s{seed}/arm{a.arm}",
                  a.backup_root / "train" / f"{gen}_{tgt}_s{seed}" / f"arm{a.arm}")]
        if not cells[0][1].is_dir():
            sys.exit(f"no such cell in the backup: {cells[0][1]}")
    else:
        cells = find_cells(a.backup_root)
        if not cells:
            sys.exit(f"no {MANIFEST} anywhere under {a.backup_root} -- nothing to verify against")

    mode = "SIZES ONLY (content NOT checked)" if a.sizes_only else "content md5"
    print(f"backup : {a.backup_root}")
    print(f"cells  : {len(cells)}")
    print(f"mode   : {mode}\n")

    tot_expected = tot_matched = 0
    bad_cells = []
    for tag, d in cells:
        r = verify_cell(d, sizes_only=a.sizes_only)
        tot_expected += r["n_expected"]
        tot_matched += r["matched"]
        broken = r.get("error") or r["mismatched"] or r["missing"]
        if broken:
            bad_cells.append(tag)
            print(f"FAIL  {tag}")
            if r.get("error"):
                print(f"        {r['error']}")
            for m in r["missing"]:
                print(f"        MISSING   {m}")
            for m in r["mismatched"]:
                print(f"        MISMATCH  {m}")
        elif r["extra"]:
            print(f"ok    {tag}  {r['matched']}/{r['n_expected']}  (+{len(r['extra'])} unrecorded)")
        else:
            print(f"ok    {tag}  {r['matched']}/{r['n_expected']}")

    # COVERAGE, asserted rather than assumed -- a verifier that only compares what it finds would
    # pass on a backup missing almost everything.
    print(f"\nfiles verified: {tot_matched:,} of {tot_expected:,} recorded "
          f"({100 * tot_matched / tot_expected if tot_expected else 0:.1f}%)")
    print(f"cells: {len(cells) - len(bad_cells)} ok, {len(bad_cells)} failing")

    if bad_cells:
        print(f"\nFAILING CELLS: {', '.join(bad_cells)}")
        print("MISSING means payload loss -- the file never landed, or landed and was removed.")
        print("MISMATCH means corruption -- it landed and its bytes are not what we copied.")
        return 1
    if a.sizes_only:
        print("\nStructure and sizes are intact. CONTENT WAS NOT CHECKED -- re-run without "
              "--sizes-only before treating this backup as verified.")
        return 0
    print("\nEvery recorded file is present and its content matches the copy-time md5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
