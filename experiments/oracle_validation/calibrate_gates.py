#!/usr/bin/env python
"""Re-derive every target's hit gate from its own known-actives-vs-decoys set. ONE rule, four targets.

THE STANDARD (decided 2026-08-21): a target's gate is the score at which **5% of its
property-matched decoys pass**. Nothing else — not a published convention, not a round number, not
another target's value.

WHY A COMMON CRITERION AND NOT A COMMON NUMBER. The four rewards are on incompatible scales (an
arbitrary proxy value, a probability, kcal/mol, a pK estimate), so a shared bar is meaningless. What
can be equalised is the bar's EFFECT. Before this rule the decoy pass-rate ran from 4% (DRD2,
6TD3-B) to 23% (ClpP) — a "mode" on ClpP admitted realistic non-binders six times more often than a
mode on DRD2, silently making mode counts non-comparable across targets, which is the one thing a
four-target matrix must get right.

WHY FPR, NOT TPR OR YOUDEN. The gate answers "does this molecule COUNT", and the failure that
corrupts the benchmark is decoys inflating a count. Fixing FPR bounds that contamination identically
everywhere. Fixing TPR equalises something never reported and lets contamination float (at TPR 75%
the sEH gate admits 49% of decoys). Youden weights a false positive and a false negative equally,
which is wrong when the headline number is a COUNT.

WHY 5% AND NOT 1%. At 1% the weaker oracles starve: sEH retains 3% of known actives, which makes
most cells pool-limited — a different failure, not a fix.

This script is the reason the odd-looking bars in `targets.py` (5.68, 0.345, −9.1, 7.97) are
reproducible rather than magic. They are empirical grid points; rounding them breaks the exact
property the standard is defined by.

Run:  conda run -n rgfn python experiments/oracle_validation/calibrate_gates.py
      conda run -n rgfn python experiments/oracle_validation/calibrate_gates.py --fpr 0.01 --compare
"""
from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
OV = REPO / "experiments" / "oracle_validation"
VM = REPO / "data" / "validation-molecules"
# Each target: (actives path, decoys path, column, higher_is_better, current bar in targets.py)
SOURCES = {
    "sEH": (
        OV / "seh_proxy/actives_results.csv",
        OV / "seh_proxy/matched_decoys_results.csv",
        "proxy_score",
        True,
        5.68,
    ),
    "ClpP": (
        OV / "docking_clpp/actives_docking_results.csv",
        OV / "docking_clpp/matched_decoys_docking_results.csv",
        "vina",
        False,
        -9.1,
    ),
    "6TD3-B": (
        Path("/scratch/markymoo/rgfn_runs/dock_6td3_matched_74500/known_results.csv"),
        Path("/scratch/markymoo/rgfn_runs/dock_6td3_matched_74500/decoy_cdk_results.csv"),
        "__cnn_vs__",
        True,
        6.718,
    ),
    "DRD2": (
        VM / "DRD2_actives_chembl_heldout.csv",
        VM / "DRD2_decoys_matched.csv",
        "__drd2_oracle__",
        True,
        0.345,
    ),
}


def _drd2(path: Path) -> np.ndarray:
    """DRD2 has no stored score column — the frozen oracle IS the score."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.DisableLog("rdApp.*")
    model = pickle.load(open(REPO / "external/scent/oracle/drd2_current.pkl", "rb"))  # nosec

    def fp(m):
        f = AllChem.GetMorganFingerprint(m, 3, useCounts=True, useFeatures=True)
        n = np.zeros((1, 2048), np.int32)
        for i, v in f.GetNonzeroElements().items():
            n[0, i % 2048] += int(v)
        return n

    sm = [r["smiles"] for r in csv.DictReader(open(path)) if r.get("smiles")]
    ms = [m for m in (Chem.MolFromSmiles(s) for s in sm) if m]
    return model.predict_proba(np.concatenate([fp(m) for m in ms], 0))[:, 1]


def load(path: Path, col: str) -> np.ndarray:
    if col == "__drd2_oracle__":
        return _drd2(path)
    d = pd.read_csv(path)
    if "status" in d:
        d = d[d["status"] == "ok"]
    if col == "__cnn_vs__":
        # gnina's CNN_VS is a DERIVED column: CNNaffinity x CNNscore. Computed here rather than
        # stored, so this calibrator works against result CSVs written before cnn_vs existed.
        return (d["cnnaff_t2"] * d["cnnsc_t2"]).dropna().to_numpy(float)
    return d[col].dropna().to_numpy(float)


def rates(act, dec, cut, higher):
    tp = (act >= cut).mean() if higher else (act <= cut).mean()
    fp = (dec >= cut).mean() if higher else (dec <= cut).mean()
    return float(tp), float(fp)


def bar_at_fpr(act, dec, higher, target_fpr):
    """Loosest bar whose decoy pass-rate does not exceed `target_fpr` (ties broken on TPR).

    'Loosest' rather than 'closest' so the gate keeps as many real actives as the FPR budget allows
    — the criterion caps contamination, it does not ask us to be gratuitously strict.
    """
    best = None
    for c in np.unique(np.r_[act, dec]):
        tp, fp = rates(act, dec, float(c), higher)
        if fp <= target_fpr and (best is None or fp > best[2] or (fp == best[2] and tp > best[1])):
            best = (float(c), tp, fp)
    return best


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fpr", type=float, default=0.05, help="the standard; 0.05 unless exploring")
    ap.add_argument(
        "--compare",
        action="store_true",
        help="also show what the CURRENT targets.py bar does, to see the delta",
    )
    a = ap.parse_args()

    print(f"gate standard: {a.fpr:.0%} FPR against property-matched decoys\n")
    hdr = f"{'target':<9}{'n_act':>6}{'n_dec':>6}{'AUROC':>7}{'gate':>9}{'TPR':>6}{'FPR':>6}{'enrich':>8}"
    print(hdr + ("   |  current bar -> TPR / FPR" if a.compare else ""))
    for name, (pa, pd_, col, higher, cur) in SOURCES.items():
        if not pa.exists() or not pd_.exists():
            print(f"{name:<9} MISSING inputs — {pa if not pa.exists() else pd_}")
            continue
        act, dec = load(pa, col), load(pd_, col)
        y = np.r_[np.ones(len(act)), np.zeros(len(dec))]
        v = np.r_[act, dec]
        au = roc_auc_score(y, v if higher else -v)
        r = bar_at_fpr(act, dec, higher, a.fpr)
        if r is None:
            print(f"{name:<9} no bar reaches FPR <= {a.fpr:.0%}")
            continue
        cut, tp, fp = r
        line = (
            f"{name:<9}{len(act):>6}{len(dec):>6}{au:>7.3f}{cut:>9.3f}"
            f"{tp:>6.0%}{fp:>6.0%}{(tp/fp if fp else float('inf')):>7.1f}x"
        )
        if a.compare:
            ctp, cfp = rates(act, dec, cur, higher)
            line += f"   |  {cur:>+7.2f} -> {ctp:>4.0%} / {cfp:>4.0%}"
        print(line)
    print(
        "\nCopy the `gate` column into experiments/lsd_hubs/matrix16/targets.py. Do NOT round:\n"
        "these are empirical grid points and rounding breaks the exact-FPR property."
    )


if __name__ == "__main__":
    main()
