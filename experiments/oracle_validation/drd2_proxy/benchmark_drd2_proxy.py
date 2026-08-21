#!/usr/bin/env python
"""Validate the DRD2 activity oracle — the ONE of our four scoring systems with no calibration entry.

WHAT WAS ALREADY KNOWN (Logs/066). The bar of 0.5 is not too low: 0 of 3,408 unrelated drug-like
molecules cross it, while 97.5% of our generated pool does. But that was a NEGATIVE-ONLY test — it
shows the model says "no" to the wrong chemistry, not that it says "yes" only to real D2 ligands. This
script supplies the missing half.

WHY IT IS NOT THE SAME TEST AS THE DOCKING ORACLES. sEH / ClpP / 6TD3 are scored by physics that has
never seen a label. DRD2 is scored by a TRAINED CLASSIFIER (TDC/Olivecrona SVM, fit on ChEMBL DRD2
data around 2017). Scoring ChEMBL DRD2 actives with it would largely measure MEMORISATION. So two
active sets are scored and reported separately:

  ALL-YEARS  in-domain; an UPPER BOUND, never quoted as validation
  HELD-OUT   actives whose earliest ChEMBL evidence post-dates the cutoff (see fetch_drd2_actives.py)

**The gap between them is the estimate of how much of the in-domain number is memorisation** — which
is the actual deliverable here, more than either AUROC on its own.

The negatives are property-matched (MW / logP / HBD / HBA / RotB / charge) and topologically distinct
(ECFP4 Tanimoto < 0.35 to every active), built by the shared `docking_clpp/make_matched_decoys.py`.
They are PRESUMED inactive, not measured inactive — standard DUD-E practice, and the reason a small
false-positive rate should not be over-read.

Featurization is copied from `DRD2FrozenReward._fp` (count Morgan, radius 3, useFeatures=True,
folded to 2048) — TDC's exact recipe, verified bit-for-bit — so these are the same numbers the
benchmark gates on.

Run:  conda run -n rgfn python experiments/oracle_validation/drd2_proxy/benchmark_drd2_proxy.py
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODEL = REPO / "external" / "scent" / "oracle" / "drd2_current.pkl"
VM = REPO / "data" / "validation-molecules"
# The bars targets.py declares for DRD2: headline 0.5, stricter variants 0.7 / 0.9.
DECLARED_BARS = (0.5, 0.7, 0.9)


def fp(mol):
    """TDC's exact DRD2 featurization (FCFP6 counts folded to 2048)."""
    f = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
    n = np.zeros((1, 2048), np.int32)
    for i, v in f.GetNonzeroElements().items():
        n[0, i % 2048] += int(v)
    return n


def score(model, smiles: list[str]) -> np.ndarray:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    mols = [m for m in mols if m is not None]
    if not mols:
        return np.array([])
    X = np.concatenate([fp(m) for m in mols], axis=0)
    return model.predict_proba(X)[:, 1]


def read_smiles(path: Path, col: str = "smiles") -> list[str]:
    return [r[col] for r in csv.DictReader(open(path)) if r.get(col)]


def at_bar(act: np.ndarray, dec: np.ndarray, bar: float) -> dict:
    tpr = float((act >= bar).mean())
    fpr = float((dec >= bar).mean())
    return {
        "bar": bar,
        "tpr": tpr,
        "fpr": fpr,
        "youden_j": tpr - fpr,
        "enrichment": (tpr / fpr) if fpr > 0 else float("inf"),
    }


def analyse(act: np.ndarray, dec: np.ndarray) -> dict:
    y = np.r_[np.ones(len(act)), np.zeros(len(dec))]
    v = np.r_[act, dec]
    fpr, tpr, thr = roc_curve(y, v)
    j = int(np.argmax(tpr - fpr))
    return {
        "n_actives": int(len(act)),
        "n_decoys": int(len(dec)),
        "auroc": float(roc_auc_score(y, v)),
        "ap": float(average_precision_score(y, v)),
        "act_median": float(np.median(act)),
        "dec_median": float(np.median(dec)),
        "youden": at_bar(act, dec, float(thr[j])),
        "declared_bars": [at_bar(act, dec, b) for b in DECLARED_BARS],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--heldout", type=Path, default=VM / "DRD2_actives_chembl_heldout.csv")
    ap.add_argument("--all-years", type=Path, default=VM / "DRD2_actives_chembl_all.csv")
    ap.add_argument("--decoys", type=Path, default=VM / "DRD2_decoys_matched.csv")
    ap.add_argument("--out", type=Path, default=HERE / "summary_drd2_proxy.json")
    a = ap.parse_args()

    model = pickle.load(open(MODEL, "rb"))  # nosec - trusted local TDC oracle
    dec = score(model, read_smiles(a.decoys))
    out = {"model": str(MODEL), "sets": {}}
    print(
        f"property-matched decoys: n={len(dec)}  median p={np.median(dec):.4f}  "
        f"frac>0.5={np.mean(dec >= 0.5):.1%}"
    )

    for name, path in (("held_out", a.heldout), ("all_years", a.all_years)):
        if not path.exists():
            print(f"  SKIP {name}: {path} missing")
            continue
        act = score(model, read_smiles(path))
        r = analyse(act, dec)
        out["sets"][name] = r
        print(f"\n=== {name}  (actives {r['n_actives']}, decoys {r['n_decoys']}) ===")
        print(
            f"  AUROC {r['auroc']:.3f}   AP {r['ap']:.3f}   "
            f"median p: actives {r['act_median']:.3f} vs decoys {r['dec_median']:.4f}"
        )
        y = r["youden"]
        print(
            f"  Youden bar {y['bar']:.3f}  TPR {y['tpr']:.0%}  FPR {y['fpr']:.0%}  "
            f"enrichment {y['enrichment']:.1f}x"
        )
        for b in r["declared_bars"]:
            print(
                f"  declared bar {b['bar']:.1f}  TPR {b['tpr']:.0%}  FPR {b['fpr']:.0%}  "
                f"enrichment {b['enrichment']:.1f}x"
            )

    if {"held_out", "all_years"} <= set(out["sets"]):
        h, al = out["sets"]["held_out"], out["sets"]["all_years"]
        gap = al["auroc"] - h["auroc"]
        out["memorisation_estimate"] = {
            "auroc_all_years": al["auroc"],
            "auroc_held_out": h["auroc"],
            "auroc_gap": gap,
            "note": "in-domain minus held-out; a large positive gap means the in-domain number is "
            "substantially memorisation and must not be quoted as validation",
        }
        print(f"\n=== MEMORISATION ESTIMATE ===")
        print(f"  AUROC in-domain {al['auroc']:.3f}  -  held-out {h['auroc']:.3f}  =  {gap:+.3f}")
        print(
            f"  median p on actives: in-domain {al['act_median']:.3f}, "
            f"held-out {h['act_median']:.3f}"
        )
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
