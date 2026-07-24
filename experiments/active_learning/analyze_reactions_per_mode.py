#!/usr/bin/env python
"""Reactions-per-mode + modes-vs-reaction-budget analysis for the LSD-Flow AL campaigns.

The oracle-call curve (validation/harness/acquisition_curve.py) measures modes per *docking call* —
where hub-batching does NOT lead, because that axis doesn't charge synthesis. This script measures the
axis LSD-Flow's thesis lives on: **bench reactions**. For each arm it computes the count-once synthesis
reactions to build its docked library and the diverse "modes" that library contains, giving

    reactions / mode         (lower = better; forward construction's shared scaffolds should win)
    modes achieved at a fixed cumulative reaction budget   (the Q3 headline)

Reaction sources, in priority order (uniform post-fix; robust to older runs):
  1. per-round selector ``chosen.csv`` (``cum_reactions`` — the strategy's own count-once model, correct
     for hub_batching + best_candidate incl. shared-hub + pre-select-K amortization);
  2. per-round ``routes.jsonl`` ``num_reactions`` summed (random / any full-route arm; independent mols);
  3. ``hub_acquisition_round_*.csv`` reconstruction (RGFN in-env hub_batching pre-logging-fix: build each
     distinct hub scaffold once (depth) + 1 marginal reaction per accepted mode).
Modes = docked molecules past the per-target bar that are Tanimoto-<cut (Morgan r=3) from every mode
already counted — computed on the accumulated docked dataset (post-dock, the deliverable).

    python experiments/active_learning/analyze_reactions_per_mode.py \
        --root /scratch/markymoo/rgfn_runs/experiments/active_learning \
        --target scent_seh_lsdflow --bar -8.0 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _rounds(run: Path) -> List[Path]:
    return sorted((run / "active_learning" / "suggestions").glob("round_*_acq"))


def _reactions_from_chosen(run: Path) -> Optional[int]:
    """Sum each round's count-once total (chosen.csv cum_reactions final row). None if absent/empty.

    Reads both the SCENT cross-env path (suggestions/round_*_acq/chosen.csv) and the RGFN in-env path
    (active_learning/chosen_round_*.csv) — same schema, so both generators feed this uniformly."""
    files = sorted(
        (run / "active_learning" / "suggestions").glob("round_*_acq/chosen.csv")
    ) + sorted((run / "active_learning").glob("chosen_round_*.csv"))
    total, seen = 0, False
    for f in files:
        rows = list(csv.DictReader(open(f)))
        if rows and "cum_reactions" in rows[0] and rows[-1].get("cum_reactions"):
            total += int(float(rows[-1]["cum_reactions"]))
            seen = True
    return total if seen else None


def _reactions_from_routes(run: Path) -> Optional[int]:
    """Sum num_reactions over all per-round routes.jsonl molecules (independent; no amortization)."""
    total, seen = 0, False
    for f in sorted((run / "active_learning" / "suggestions").glob("round_*_routes.jsonl")):
        for line in open(f):
            try:
                total += int(json.loads(line).get("num_reactions", 0) or 0)
                seen = True
            except Exception:
                continue
    return total if seen else None


def _reactions_from_candidates(run: Path) -> Optional[int]:
    """Sum num_reactions over the standard candidate dataset (independent mols, no amortization) —
    the glue RGFN loop's full-route arms (best_candidate/random) populate this; hub_batching leaves it
    empty (minimal route) so this returns None and we fall through to the count-once reconstruction.
    """
    f = run / "active_learning" / "suggestions" / "candidates.csv"
    if not f.exists():
        return None
    total, seen = 0, False
    for r in csv.DictReader(open(f)):
        try:
            total += int(float(r["num_reactions"]))
            seen = True
        except (KeyError, TypeError, ValueError):
            continue
    return total if seen else None


def _reactions_from_hubcsv(run: Path) -> Optional[int]:
    """RGFN fallback: count-once from per-hub (depth built once) + 1 per accepted mode."""
    built, marginal, seen = {}, 0, False
    for f in sorted((run / "active_learning").glob("hub_acquisition_round_*.csv")):
        for r in csv.DictReader(open(f)):
            seen = True
            na = int(r.get("n_accepted", 0) or 0)
            if na <= 0:
                continue
            built.setdefault(r["hub_key"], int(r["depth"]))
            marginal += na
    return (sum(built.values()) + marginal) if seen else None


def _total_reactions(run: Path) -> Tuple[Optional[int], str]:
    for fn, src in (
        (_reactions_from_chosen, "chosen.csv/cum_reactions"),
        (_reactions_from_candidates, "candidates.csv/num_reactions"),
        (_reactions_from_routes, "routes.jsonl/num_reactions"),
        (_reactions_from_hubcsv, "hub_acquisition/depth-reconstruction"),
    ):
        v = fn(run)
        if v is not None:
            return v, src
    return None, "none"


def _modes(smis: List[str], labs: List[float], bar: float, lower: bool, cut: float) -> int:
    order = sorted(range(len(smis)), key=lambda i: labs[i], reverse=not lower)
    fps: list = []
    n = 0
    for i in order:
        if not ((labs[i] <= bar) if lower else (labs[i] >= bar)):
            continue
        m = Chem.MolFromSmiles(smis[i])
        if m is None:
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 3, 2048)
        if any(DataStructs.TanimotoSimilarity(fp, a) > cut for a in fps):
            continue
        fps.append(fp)
        n += 1
    return n


def analyze_arm(run: Path, bar: float, lower: bool, cut: float) -> dict:
    ds = sorted((run / "active_learning").glob("dataset_round_*.csv"))
    rows = list(csv.DictReader(open(ds[-1]))) if ds else []
    labs, smis = [], []
    for r in rows:
        try:
            labs.append(float(r["label"]))
            smis.append(r["smiles"])
        except (KeyError, TypeError, ValueError):
            continue
    modes = _modes(smis, labs, bar, lower, cut)
    rx, src = _total_reactions(run)
    return {
        "n_docked": len(smis),
        "modes": modes,
        "reactions": rx,
        "reactions_per_mode": (rx / modes) if (rx and modes) else None,
        "reaction_source": src,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/scratch/markymoo/rgfn_runs/experiments/active_learning")
    ap.add_argument("--target", required=True, help="e.g. scent_seh_lsdflow or 6td3_lsdflow")
    ap.add_argument("--bar", type=float, required=True, help="mode hit bar (real oracle units)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cut", type=float, default=0.5, help="Tanimoto diversity cutoff for modes")
    ap.add_argument("--higher-is-better", dest="lower", action="store_false", default=True)
    ap.add_argument("--arms", nargs="+", default=["hub_batching", "best_candidate", "random"])
    a = ap.parse_args(argv)

    print(f"=== {a.target} (bar {a.bar}, Tanimoto<{a.cut}, seed {a.seed}) — REACTIONS PER MODE ===")
    print(f"{'arm':>16} {'docked':>7} {'modes':>6} {'reactions':>10} {'rxn/mode':>9}  source")
    for arm in a.arms:
        runs = sorted(glob.glob(f"{a.root}/{a.target}/{arm}_seed{a.seed}/*/"))
        if not runs:
            print(f"{arm:>16}  (no run found)")
            continue
        r = analyze_arm(Path(runs[-1]), a.bar, a.lower, a.cut)
        rpm = f"{r['reactions_per_mode']:.2f}" if r["reactions_per_mode"] else "n/a"
        print(
            f"{arm:>16} {r['n_docked']:>7} {r['modes']:>6} {str(r['reactions']):>10} {rpm:>9}  {r['reaction_source']}"
        )


if __name__ == "__main__":
    main()
