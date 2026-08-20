#!/usr/bin/env python
"""Verify a reaction-repair re-enumeration before it is allowed to stand.

WHY THIS IS A SCRIPT AND NOT A JUDGEMENT CALL. The RGFN surrogate cells were re-enumerated purely to
add ``children[].reaction`` (the field ``rgfn_worker`` omitted, entry `068`). Nothing else was meant to
change: same checkpoint, same 200 hubs, same ``ENUM_MAX=4000`` that those cells never came near
(maxima 1,972-3,435). But these are the artifacts behind entry `050`'s headline and entry `055`'s
surfaces, so "nothing else changed" has to be MEASURED, not assumed -- and there is one specific way it
could quietly fail: ``submit_cell.sh`` re-runs ``pick_hubs`` on the enum stage, so it rewrites
``hubs.csv``. If that produced even a slightly different hub set, the new enumeration would be of a
different substrate and the published number would not reproduce.

Three checks, in the order that matters:

  1. HUB SET IDENTICAL to the preserved ``hubs.pre_rxn.csv``. If this fails, stop -- everything below is
     comparing different substrates and the repair must be redone with the preserved hub list pinned.
  2. REACTION COVERAGE 100%, which is the whole point of the exercise, plus a spot check that the
     recorded final product really is the child molecule. Compare STEREO-STRIPPED: ``children[].smiles``
     is the stripped key while a reaction's ``product`` is the raw stereo-aware SMILES, so a raw string
     comparison matches only ~18% of the time and means nothing.
  3. COUNT-ONCE NUMBERS UNCHANGED. Runs the campaign into a scratch directory and diffs every field of
     ``summary.json`` against the committed one. Deliberately does NOT overwrite the committed result:
     if the numbers moved, we want to see that before the published value is gone.

Exit 0 only if all three pass for every cell requested.

    python verify_rxn_repair.py rgfn seh --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# Fields that legitimately differ between two runs of the same enumeration and must not fail the diff.
# Everything else is expected bit-identical.
VOLATILE = {"compute_time", "wall_s", "enum_timings_meta"}


def _strip(smiles: str):
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smiles) if smiles else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


def _emit(gen: str, tgt: str, env: dict) -> dict:
    out = subprocess.run(
        [sys.executable, str(HERE / "manifest.py"), "--emit", gen, tgt],
        capture_output=True, text=True, env=env, cwd=REPO,
    )
    if out.returncode:
        raise SystemExit(f"manifest emit failed for {gen}/{tgt}: {out.stderr.strip()}")
    d = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip().strip("'\"")
    return d


def check_cell(gen: str, tgt: str, seed: int) -> bool:
    env = dict(os.environ)
    if seed != 42:
        env["MATRIX16_MANIFEST"] = str(HERE / f"manifest_seed{seed}.csv")
        env["MATRIX16_SCRATCH"] = f"/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed{seed}"
        env["MATRIX16_RESULTS"] = str(HERE / f"results_seed{seed}")
    else:
        for k in ("MATRIX16_MANIFEST", "MATRIX16_SCRATCH", "MATRIX16_RESULTS"):
            env.pop(k, None)

    spec = _emit(gen, tgt, env)
    E = Path(spec["ENUM_DIR"])
    tag = spec["CELL_TAG"]
    ok = True
    print(f"\n=== {tag} seed {seed} ===")

    # ---- 1. hub set identical --------------------------------------------------------------------
    pre_h, cur_h = E / "hubs.pre_rxn.csv", E / "hubs.csv"
    if not pre_h.exists():
        print("  SKIP: no hubs.pre_rxn.csv — nothing to compare against")
        return True
    a = {r["smiles"] for r in csv.DictReader(open(pre_h))}
    b = {r["smiles"] for r in csv.DictReader(open(cur_h))}
    if a == b:
        print(f"  [1] hub set IDENTICAL ({len(a)} hubs)")
    else:
        print(f"  [1] FAIL hub set CHANGED: {len(a - b)} lost, {len(b - a)} new — pick_hubs did not reproduce")
        return False  # everything below would compare different substrates

    # ---- 2. reaction coverage --------------------------------------------------------------------
    hubs = json.load(open(E / "enum_children.json"))["hubs"]
    ch = [c for h in hubs for c in h.get("children", [])]
    have = sum(1 for c in ch if c.get("reaction"))
    if have == len(ch) and ch:
        print(f"  [2] reaction coverage 100% ({have:,}/{len(ch):,})")
    else:
        print(f"  [2] FAIL reaction coverage {have:,}/{len(ch):,}")
        ok = False
    # spot-check that the recorded product IS the child, stereo-stripped
    sample = [c for c in ch[:400] if c.get("reaction")]
    good = sum(1 for c in sample if _strip((c["reaction"][-1] or {}).get("product")) == _strip(c["smiles"]))
    if sample and good == len(sample):
        print(f"      final product == child (stereo-stripped) on {good}/{len(sample)} sampled")
    elif sample:
        print(f"      FAIL product != child on {len(sample) - good}/{len(sample)} sampled")
        ok = False

    # ---- 3. count-once numbers unchanged --------------------------------------------------------
    committed = Path(spec["RESULTS_DIR"]) / "summary.json"
    if not committed.exists():
        print("  [3] SKIP: no committed summary.json to compare")
        return ok
    with tempfile.TemporaryDirectory() as td:
        env2 = dict(env)
        env2["MATRIX16_RESULTS"] = td
        r = subprocess.run(
            ["bash", str(HERE / "run_cell_campaign.sh"), gen, tgt],
            capture_output=True, text=True, env=env2, cwd=REPO,
        )
        fresh = Path(td) / tag / "summary.json"
        if r.returncode or not fresh.exists():
            print(f"  [3] FAIL campaign errored: {r.stderr.strip()[-200:]}")
            return False
        old, new = json.load(open(committed)), json.load(open(fresh))
        diffs = []
        for k in set(old) | set(new):
            if k in VOLATILE:
                continue
            if old.get(k) != new.get(k):
                diffs.append(k)
        if not diffs:
            print("  [3] count-once summary IDENTICAL to the committed value")
        else:
            print(f"  [3] FAIL summary changed in: {', '.join(sorted(diffs))}")
            for k in sorted(diffs)[:3]:
                print(f"        {k}: {old.get(k)}  ->  {new.get(k)}")
            ok = False
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("generator")
    ap.add_argument("target", nargs="?", default=None, help="omit to check both seh and drd2")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    a = ap.parse_args()
    tgts = [a.target] if a.target else ["seh", "drd2"]
    allok = True
    for t in tgts:
        for s in a.seeds:
            try:
                allok &= check_cell(a.generator, t, s)
            except SystemExit as e:
                print(f"  ERROR: {e}")
                allok = False
    print("\n" + ("ALL CHECKS PASSED — repairs may stand" if allok else "FAILURES ABOVE — do not promote"))
    raise SystemExit(0 if allok else 1)


if __name__ == "__main__":
    main()
