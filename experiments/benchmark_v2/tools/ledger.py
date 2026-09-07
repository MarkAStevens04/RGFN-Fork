#!/usr/bin/env python
"""``PROVENANCE.csv`` — the ONLY record of which artifacts were generated here and which were copied.

WHY A LEDGER AND NOT A DIRECTORY SPLIT. The obvious design is a ``copied/`` subtree beside the
generated one. It was rejected on purpose: it forces every future reader to ask "am I looking at the
copied one or the new one?" before they can use anything, which is precisely the confusion
``benchmark_v2`` exists to remove. The tree is uniform and trusted; provenance is a column.

    stage,generator,target,seed,arm,origin,source_path,source_md5,date,operator,note

``origin`` is ``generated`` or ``copied``. Nothing else -- an unknown value is rejected rather than
written, because a third value would immediately mean two readers disagree about what the file says.

APPEND-ONLY, AND LOCKED. Cells run concurrently and at different stages, so several processes will
append at once; without a lock their writes interleave mid-line and the file becomes unparseable at
exactly the moment it matters. Every append takes an exclusive ``flock`` on the ledger. Rows are
never edited or deleted -- a correction is a NEW row whose note says what it supersedes, so the
history of what we believed stays readable.

WHAT ``source_md5`` MEANS. A cell is a directory, not a file, so this is a **tree digest**: md5 over
the sorted ``(relative path, size, content-md5)`` of every file under the source. One value that
changes if anything in the copied artifact changes. ``--quick`` substitutes mtime for content, which
is ~100x faster and enough for a smoke, but the default is a real content digest because proving a
copy is faithful is the entire reason the column exists.

``verify`` re-computes those digests against the destinations and reports drift. That is what turns
the ledger from a note-to-self into evidence: a copied cell that has since been overwritten shows up
here rather than in a confusing result three weeks later.

CLI
    python ledger.py record --stage train --cell scent/seh/42 --arm a --origin generated
    python ledger.py record --stage train --cell saturn/seh/42 --arm a --origin copied \\
        --source /scratch/.../fixed_reward/saturn_seh/seed42 --note "authors' 10k budget"
    python ledger.py show --cell saturn/seh/42
    python ledger.py verify                 # re-hash copied rows, report drift

IMPORTABLE. ``record()``, ``tree_digest()`` and ``read_rows()`` are the API; the copy-forward step
should call them rather than formatting CSV rows of its own, so there is exactly one definition of
what a provenance row means.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import fcntl
import hashlib
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
LEDGER = Path(os.environ.get("BENCHMARK_V2_LEDGER", str(V2_ROOT / "PROVENANCE.csv")))

COLUMNS = ["stage", "generator", "target", "seed", "arm", "origin",
           "source_path", "source_md5", "date", "operator", "note"]

# The stage graph, both pipelines. Validated rather than free-text: a typo'd stage silently creates
# a category nothing queries, and the ledger's whole value is that a query over it is complete.
STAGES = (
    "train",                                   # both pipelines
    "sample", "pick_hubs", "enumerate", "campaign",   # reaction-GFN pipeline
    "pool", "routes", "selection",             # competitor pipeline
    "backup",                                  # a copy landed in /project
)
ORIGINS = ("generated", "copied")

# Files large enough that hashing them dominates; hashed in chunks rather than read whole.
_CHUNK = 1 << 20


def _file_md5(p: Path, quick: bool = False) -> str:
    if quick:
        st = p.stat()
        return hashlib.md5(f"{st.st_size}:{int(st.st_mtime)}".encode()).hexdigest()
    h = hashlib.md5()
    with open(p, "rb") as fh:
        while True:
            b = fh.read(_CHUNK)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def tree_digest(root: Path, quick: bool = False,
                skip: Tuple[str, ...] = (".verified.json",)) -> Tuple[str, int, int]:
    """``(digest, n_files, n_bytes)`` over everything under ``root``.

    Deterministic: paths are sorted, so two runs over identical content agree. ``skip`` excludes
    markers this tooling writes itself -- a cell's digest must describe the ARTIFACT, not the
    bookkeeping we later attach to it, or freezing a cell would appear to change it.
    """
    root = Path(root)
    if root.is_file():
        return _file_md5(root, quick), 1, root.stat().st_size
    if not root.is_dir():
        raise FileNotFoundError(root)
    entries: List[str] = []
    n_files = n_bytes = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        if p.name in skip:
            continue
        rel = p.relative_to(root).as_posix()
        size = p.stat().st_size
        entries.append(f"{rel}:{size}:{_file_md5(p, quick)}")
        n_files += 1
        n_bytes += size
    return hashlib.md5("\n".join(entries).encode()).hexdigest(), n_files, n_bytes


def _ensure_header(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as fh:
            csv.writer(fh, lineterminator="\n").writerow(COLUMNS)


def record(stage: str, generator: str, target: str, seed: int, arm: str, origin: str,
           source_path: str = "", source_md5: str = "", operator: str = "",
           note: str = "", ledger: Path = LEDGER) -> Dict[str, str]:
    """Append one provenance row under an exclusive lock. Returns the row written."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; known: {list(STAGES)}")
    if origin not in ORIGINS:
        raise ValueError(f"origin must be one of {list(ORIGINS)}, got {origin!r}")
    if origin == "copied" and not source_path:
        raise ValueError("a copied row must name its source_path -- that is the point of the row")
    row = {
        "stage": stage, "generator": generator, "target": target, "seed": str(seed),
        "arm": arm, "origin": origin, "source_path": source_path, "source_md5": source_md5,
        "date": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "operator": operator or os.environ.get("USER", ""),
        "note": note,
    }
    ledger = Path(ledger)
    _ensure_header(ledger)
    # Lock the file itself, so concurrent cells serialise instead of interleaving mid-line.
    with open(ledger, "a", newline="") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n").writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    return row


def read_rows(ledger: Path = LEDGER) -> List[Dict[str, str]]:
    ledger = Path(ledger)
    if not ledger.is_file():
        return []
    with open(ledger) as fh:
        return list(csv.DictReader(fh))


def _parse_cell(s: str) -> Tuple[str, str, int]:
    """``generator/target/seed``."""
    parts = s.split("/")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--cell wants generator/target/seed, got {s!r}")
    return parts[0], parts[1], int(parts[2])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="append one provenance row")
    r.add_argument("--stage", required=True, choices=STAGES)
    r.add_argument("--cell", required=True, type=_parse_cell, metavar="GEN/TARGET/SEED")
    r.add_argument("--arm", default="a", choices=("a", "b"))
    r.add_argument("--origin", required=True, choices=ORIGINS)
    r.add_argument("--source", default="", help="source path (required for --origin copied)")
    r.add_argument("--digest", action="store_true",
                   help="compute the source tree digest (default when --origin copied)")
    r.add_argument("--quick", action="store_true", help="mtime-based digest instead of content")
    r.add_argument("--operator", default="")
    r.add_argument("--note", default="")

    s = sub.add_parser("show", help="print rows, optionally filtered")
    s.add_argument("--cell", type=_parse_cell, default=None)
    s.add_argument("--stage", default=None, choices=STAGES)

    v = sub.add_parser("verify", help="re-hash copied rows and report drift")
    v.add_argument("--quick", action="store_true")

    a = ap.parse_args()

    if a.cmd == "record":
        gen, tgt, seed = a.cell
        digest = ""
        if a.source and (a.digest or a.origin == "copied"):
            d, n, nb = tree_digest(Path(a.source), quick=a.quick)
            digest = d
            print(f"[ledger] digest {d}  ({n} files, {nb/1e6:.1f} MB{', quick' if a.quick else ''})")
        row = record(a.stage, gen, tgt, seed, a.arm, a.origin,
                     source_path=a.source, source_md5=digest,
                     operator=a.operator, note=a.note)
        print(f"[ledger] {row['stage']} {gen}/{tgt}/{seed} arm{row['arm']} -> {row['origin']}")
        return 0

    if a.cmd == "show":
        rows = read_rows()
        if a.cell:
            g, t, sd = a.cell
            rows = [r for r in rows if (r["generator"], r["target"], r["seed"]) == (g, t, str(sd))]
        if a.stage:
            rows = [r for r in rows if r["stage"] == a.stage]
        if not rows:
            print("(no matching rows)")
            return 0
        hdr = f"{'stage':<11}{'cell':<26}{'arm':<5}{'origin':<11}{'date':<21}note"
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            cell = f"{r['generator']}_{r['target']}_s{r['seed']}"
            print(f"{r['stage']:<11}{cell:<26}{r['arm']:<5}{r['origin']:<11}{r['date']:<21}{r['note']}")
        return 0

    if a.cmd == "verify":
        rows = [r for r in read_rows() if r["origin"] == "copied" and r["source_md5"]]
        if not rows:
            print("no copied rows carrying a digest -- nothing to verify")
            return 0
        bad = 0
        for r in rows:
            src = Path(r["source_path"])
            cell = f"{r['generator']}_{r['target']}_s{r['seed']}"
            if not src.exists():
                print(f"  GONE     {r['stage']:<11}{cell:<26} {src}")
                bad += 1
                continue
            d, _, _ = tree_digest(src, quick=a.quick)
            if d != r["source_md5"]:
                print(f"  DRIFT    {r['stage']:<11}{cell:<26} {r['source_md5'][:8]} -> {d[:8]}")
                bad += 1
            else:
                print(f"  ok       {r['stage']:<11}{cell:<26} {d[:8]}")
        print(f"\n{len(rows) - bad}/{len(rows)} copied rows still match their recorded source")
        return 1 if bad else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
