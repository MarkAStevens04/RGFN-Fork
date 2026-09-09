#!/usr/bin/env python
"""Score SMILES against the frozen sEH proxy IN A SEPARATE PROCESS.

WHY THIS EXISTS. SynFormer's genetic loop forks worker processes every generation. Loading the
Bengio-2021 sEH MPNN in the COORDINATING process makes every subsequent worker spawn fail with
``No CUDA GPUs are available`` -- measured in job 75080 (entry [074]), where the parent was clean on
both the descriptor count (fds=0) and the thread count (33) and the spawn failed anyway. Three
successive guards each fixed their own measured symptom and the spawn still failed; all three were
reverted. The conclusion recorded there is that the model must never be loaded in the parent at all,
which leaves scoring out-of-process as the only route -- and entry [074] closes by noting that
``scripts/score_batch.py`` registers only docking oracles, so that entry point did not exist.

This is that entry point.

IT DOES NOT CROSS AN ENV BOUNDARY, and that is the difference from the docking bridge. The docking
bridge shells out to the ``rgfn`` env because SynFormer's own env has no docking stack. Here the
synformer env CAN import ``gflownet.models.bengio2021flow`` (the rgfn env cannot), so the child runs
in the same env and only the PROCESS differs -- which is precisely the thing the fork hazard cares
about.

ONE DEFINITION OF THE REWARD. It imports ``SEHFrozenReward`` rather than reimplementing the
featurisation, so the in-process and out-of-process paths cannot drift on what a score means. An
invalid molecule stays ``nan`` here exactly as it does there: scoring it 0 would teach the policy
that exotic elements are merely bad rather than unscoreable.

Usage (invoked by SEHBridgeReward, not by hand):
    python -m validation.generators.synformer.score_seh_subprocess \
        --smiles in.smi --out scores.csv [--device cpu] [--clip 10.0] [--batch-size 128]
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def _read_smiles(path: Path) -> list[str]:
    """One SMILES per line, or a CSV with a ``smiles`` column. Blank lines are dropped, but the
    ORDER of what remains is the contract -- the caller pairs the output back up by position."""
    text = path.read_text().splitlines()
    if not text:
        return []
    head = text[0].lower()
    if "smiles" in head and "," in text[0]:
        with path.open(newline="") as fh:
            rd = csv.DictReader(fh)
            col = (
                "smiles" if "smiles" in (rd.fieldnames or []) else (rd.fieldnames or ["smiles"])[0]
            )
            return [r[col].strip() for r in rd if r.get(col, "").strip()]
    return [ln.split()[0].strip() for ln in text if ln.strip() and not ln.startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smiles", required=True, help="input .smi or .csv with a smiles column")
    ap.add_argument("--out", required=True, help="output CSV: smiles,raw_score")
    ap.add_argument("--device", default="cpu", help="cpu | cuda (default cpu)")
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    # Import INSIDE main, after argument parsing: this module is only ever run as a child, but
    # keeping the heavy import off the module scope means a stray `import` of this file by a parent
    # process cannot drag the MPNN in -- the exact hazard the file exists to avoid.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from validation.generators.synformer.fixed_reward import SEHFrozenReward

    smis = _read_smiles(Path(a.smiles))
    scorer = SEHFrozenReward(device=a.device, clip=a.clip, batch_size=a.batch_size)
    vals = scorer.predict(smis) if smis else []

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "raw_score"])
        for s, v in zip(smis, vals):
            # nan is written literally and read back as nan; it means UNSCOREABLE, not zero.
            w.writerow([s, v])
    print(f"[score-seh] {len(smis)} smiles -> {out}", flush=True)


if __name__ == "__main__":
    main()
