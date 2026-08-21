#!/usr/bin/env python
"""Re-calibrate the 6TD3 hit gate against BOTH negative sets, side by side.

THE QUESTION. The 6TD3 gate is `raw_score < -2.0` on the neosubstrate differential
(Vina Tier2 - Tier1), and `targets.py` marks it PROVISIONAL. Entry 005 published AUROC 0.890 for
ABSOLUTE Tier-2 score; the differential was never given its own curve. And the negative set behind
both numbers is WARHEAD-matched, not property-matched: every decoy keeps the CR8-like purine (a hard
and meaningful control -- it rules out "the oracle just reads ATP-pocket binding") but molecular
weight is only range-bounded 250-650 and NO other property is matched. Entry 007 showed the size
confound is large here, so the remaining property axes need closing rather than assuming.

WHAT THIS SCRIPT DOES. Reports AUROC / AP / operating points for the differential and its
components, against EACH negative set, so the two controls can be read against each other:

  warhead-matched   shares the ATP-pocket warhead; unmatched on properties
  property-matched  DUD-E style: MW / logP / HBD / HBA / RotB / charge matched, and ECFP4
                    Tanimoto < 0.35 to every active (topologically distinct)

A gate that survives BOTH is defensible. One that only survives the warhead-matched set is reading
physicochemistry; one that only survives the property-matched set is reading warhead binding.

TWO OPERATING POINTS, because they answer different questions and disagree here.
  * Youden's J (max TPR-FPR) -- the standard single cutoff, treats FP and FN as equally costly.
  * MAX ENRICHMENT (TPR/FPR) -- the right criterion for a MODE GATE. The gate defines "is this
    molecule worth counting", not a diagnosis; admitting decoys inflates every arm's mode count with
    junk, so specificity is worth more than sensitivity. This is why the incumbent -2.0 can be a
    good gate while not being Youden-optimal.

NOTE ON FILE NAMES: `dock_cluster.py` always writes its negatives to `decoy_cdk_results.csv`
regardless of which set was docked, so the property-matched run's file has that name too. It is
distinguished by living in its own $OUTDIR -- pass it with --property-decoys.

Run:  conda run -n rgfn python experiments/oracle_validation/docking_6td3/benchmark_6td3_gate.py \
          --property-decoys /scratch/.../dock_6td3_matched_<jobid>/decoy_cdk_results.csv \
          --property-actives /scratch/.../dock_6td3_matched_<jobid>/known_results.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
# Lower-is-better columns (binding energies / differentials) vs higher-is-better (CNN affinity).
COLUMNS = {
    "ddb1_dvina": ("differential Vina (T2-T1) — THE GATED QUANTITY", "lower"),
    "vina_t2": ("absolute Tier-2 Vina", "lower"),
    "ddb1_dcnnaff": ("differential CNN affinity", "higher"),
}
INCUMBENT_GATE = -2.0  # targets.py, PROVISIONAL


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df[df["status"] == "ok"].copy()


def _report(act: np.ndarray, dec: np.ndarray, cutoff: float, sense: str) -> dict:
    """TPR/FPR/enrichment at one cutoff, in the column's own direction."""
    if sense == "lower":
        tp, fp = float((act <= cutoff).mean()), float((dec <= cutoff).mean())
    else:
        tp, fp = float((act >= cutoff).mean()), float((dec >= cutoff).mean())
    return {
        "cutoff": float(cutoff),
        "tpr": tp,
        "fpr": fp,
        "youden_j": tp - fp,
        "enrichment": (tp / fp) if fp > 0 else float("inf"),
    }


def analyse(act: np.ndarray, dec: np.ndarray, sense: str) -> dict:
    y = np.r_[np.ones(len(act)), np.zeros(len(dec))]
    v = np.r_[act, dec]
    # sklearn wants higher = more positive
    score = -v if sense == "lower" else v
    fpr, tpr, thr = roc_curve(y, score)
    j = int(np.argmax(tpr - fpr))
    youden_cut = float(-thr[j] if sense == "lower" else thr[j])

    # Max-enrichment point, restricted to cutoffs that keep a usable share of actives: TPR/FPR is
    # unbounded as FPR -> 0 and would otherwise select a cutoff admitting one active and no decoys.
    grid = np.unique(np.r_[act, dec])
    best = None
    for c in grid:
        r = _report(act, dec, float(c), sense)
        if r["tpr"] < 0.25 or r["fpr"] == 0.0:
            continue
        if best is None or r["enrichment"] > best["enrichment"]:
            best = r
    return {
        "n_actives": int(len(act)),
        "n_decoys": int(len(dec)),
        "auroc": float(roc_auc_score(y, score)),
        "ap": float(average_precision_score(y, score)),
        "youden": _report(act, dec, youden_cut, sense),
        "max_enrichment": best,
        "at_incumbent_gate": _report(act, dec, INCUMBENT_GATE, sense) if sense == "lower" else None,
        "act_median": float(np.median(act)),
        "dec_median": float(np.median(dec)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--actives",
        type=Path,
        default=HERE / "known_results.csv",
        help="docked actives for the WARHEAD-matched comparison (committed, June run)",
    )
    ap.add_argument("--warhead-decoys", type=Path, default=HERE / "decoy_cdk_results.csv")
    ap.add_argument("--property-decoys", type=Path, required=True)
    ap.add_argument(
        "--property-actives",
        type=Path,
        default=None,
        help="actives re-docked ALONGSIDE the property-matched decoys; defaults to "
        "--actives, but passing the same-job file removes any cross-run drift",
    )
    ap.add_argument("--out", type=Path, default=HERE / "summary_gate_recalibration.json")
    a = ap.parse_args()

    sets = {
        "warhead_matched": (load(a.actives), load(a.warhead_decoys)),
        "property_matched": (load(a.property_actives or a.actives), load(a.property_decoys)),
    }

    out: dict = {"incumbent_gate": INCUMBENT_GATE, "sets": {}}
    for set_name, (act_df, dec_df) in sets.items():
        out["sets"][set_name] = {}
        print(f"\n=== {set_name}  (actives {len(act_df)}, decoys {len(dec_df)}) ===")
        # size gap is the confound entry 007 measured; report it rather than assume it away
        if "smiles" in act_df and "smiles" in dec_df:
            from rdkit import Chem, RDLogger
            from rdkit.Chem import Descriptors

            RDLogger.DisableLog("rdApp.*")
            mw = lambda d: np.median(
                [Descriptors.MolWt(m) for m in (Chem.MolFromSmiles(s) for s in d["smiles"]) if m]
            )
            print(f"  median MW: actives {mw(act_df):.0f}  decoys {mw(dec_df):.0f}")
            out["sets"][set_name]["median_mw"] = {
                "actives": float(mw(act_df)),
                "decoys": float(mw(dec_df)),
            }
        for col, (label, sense) in COLUMNS.items():
            if col not in act_df or col not in dec_df:
                continue
            act = act_df[col].dropna().to_numpy(float)
            dec = dec_df[col].dropna().to_numpy(float)
            r = analyse(act, dec, sense)
            out["sets"][set_name][col] = r
            print(f"  {label}")
            print(
                f"     AUROC {r['auroc']:.3f}  AP {r['ap']:.3f}   "
                f"median act {r['act_median']:+.2f} vs dec {r['dec_median']:+.2f}"
            )
            y = r["youden"]
            print(
                f"     Youden      cutoff {y['cutoff']:+.2f}  TPR {y['tpr']:.0%}  "
                f"FPR {y['fpr']:.0%}  enrichment {y['enrichment']:.1f}x"
            )
            m = r["max_enrichment"]
            if m:
                print(
                    f"     max-enrich  cutoff {m['cutoff']:+.2f}  TPR {m['tpr']:.0%}  "
                    f"FPR {m['fpr']:.0%}  enrichment {m['enrichment']:.1f}x"
                )
            g = r["at_incumbent_gate"]
            if g:
                print(
                    f"     INCUMBENT {INCUMBENT_GATE:+.1f}  TPR {g['tpr']:.0%}  "
                    f"FPR {g['fpr']:.0%}  enrichment {g['enrichment']:.1f}x"
                )
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
