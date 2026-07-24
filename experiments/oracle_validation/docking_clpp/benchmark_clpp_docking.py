#!/usr/bin/env python
"""ClpP docking cutoff calibration: which Vina score best separates binders from decoys?

Reads the ClpP Vina energies from `dock_sets.py` (actives = real human-ClpP binders,
matched = property-matched DUD-E decoys) and finds the docking cutoff that best
discriminates them. This cutoff replaces the 16-cell ClpP cell's provisional
`raw_score < -2.0` "mode" gate — which ~100 % of every generator clears (raw ClpP
docking is -5..-14 kcal/mol), so it does not discriminate at all.

Docking convention: Vina binding energy in kcal/mol, **lower (more negative) = better
binder**. ROC orients the score as `-vina`; a molecule is called a hit if `vina <= cutoff`.

"Best discriminating cutoff" = the one maximizing Youden's J (TPR - FPR), the standard
single operating point; we also report the max-F1 and high-decoy-specificity cutoffs.

Run in the `rgfn` env from the repo root (no GPU needed — reads the docked CSVs):
    python experiments/oracle_validation/docking_clpp/benchmark_clpp_docking.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
SWEEP = np.arange(-5.0, -11.5, -0.5)  # Vina cutoffs (kcal/mol), loose -> strict
ACT_COLOR, DEC_COLOR = "#2b7bba", "#e07b39"


def _stats(x: np.ndarray) -> dict:
    q = np.percentile(x, [25, 50, 75])
    return {
        "n": int(x.size),
        "min": float(x.min()),
        "p25": float(q[0]),
        "median": float(q[1]),
        "mean": float(x.mean()),
        "p75": float(q[2]),
        "max": float(x.max()),
    }


def mw_match(df_a: pd.DataFrame, df_d: pd.DataFrame, bin_w: float, rng):
    lo = min(df_a.mw.min(), df_d.mw.min())
    hi = max(df_a.mw.max(), df_d.mw.max())
    edges = np.arange(lo, hi + bin_w, bin_w)
    ab = np.digitize(df_a.mw.values, edges)
    db = np.digitize(df_d.mw.values, edges)
    ka, kd = [], []
    for b in np.unique(np.concatenate([ab, db])):
        ia = np.where(ab == b)[0]
        idd = np.where(db == b)[0]
        k = min(len(ia), len(idd))
        if k:
            ka.extend(rng.choice(ia, k, replace=False))
            kd.extend(rng.choice(idd, k, replace=False))
    return df_a.iloc[ka].reset_index(drop=True), df_d.iloc[kd].reset_index(drop=True)


def _cutoff_report(act: np.ndarray, dec: np.ndarray, cutoff: float) -> dict:
    """Vina cutoff: call a hit if vina <= cutoff (stronger binder)."""
    tpr = float((act <= cutoff).mean())
    fpr = float((dec <= cutoff).mean())
    tp, fp = int((act <= cutoff).sum()), int((dec <= cutoff).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * precision * tpr / (precision + tpr)) if (precision + tpr) else 0.0
    return {
        "cutoff": float(cutoff),
        "tpr_actives_retained": tpr,
        "fpr_decoys_passing": fpr,
        "youden_j": tpr - fpr,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": 0.5 * (tpr + (1 - fpr)),
        "enrichment_factor": (tpr / fpr) if fpr > 0 else float("inf"),
        "n_actives_below": tp,
        "n_decoys_below": fp,
    }


def _best_cutoffs(a: np.ndarray, d: np.ndarray) -> dict:
    """Sweep every distinct docked score as a candidate cutoff; return the argmax by metric."""
    cuts = np.unique(np.concatenate([a, d]))
    reports = [_cutoff_report(a, d, float(c)) for c in cuts]
    by = lambda key: max(reports, key=lambda r: (r[key] if np.isfinite(r[key]) else -1e9))
    return {
        "youden": by("youden_j"),  # max TPR - FPR  (the recommended single operating point)
        "f1": by("f1"),
        "balanced_accuracy": by("balanced_accuracy"),
    }


def analyze(act: pd.DataFrame, dec: pd.DataFrame, rng) -> dict:
    a, d = act.vina.values, dec.vina.values
    y = np.concatenate([np.ones_like(a), np.zeros_like(d)])
    neg = np.concatenate([-a, -d])  # higher = better binder
    fpr, tpr, _ = roc_curve(y, neg)
    am, dm = mw_match(act, dec, 25.0, rng)
    ym = np.concatenate([np.ones(len(am)), np.zeros(len(dm))])
    negm = np.concatenate([-am.vina.values, -dm.vina.values])
    return {
        "n_actives": int(len(act)),
        "n_decoys": int(len(dec)),
        "actives_vina": _stats(a),
        "decoys_vina": _stats(d),
        "actives_mw": _stats(act.mw.values),
        "decoys_mw": _stats(dec.mw.values),
        "auroc": float(roc_auc_score(y, neg)),
        "pr_auc_average_precision": float(average_precision_score(y, neg)),
        "mw_matched": {
            "n_per_group": int(len(am)),
            "auroc": float(roc_auc_score(ym, negm)) if len(np.unique(ym)) == 2 else float("nan"),
        },
        "best_cutoffs": _best_cutoffs(a, d),
        "fixed_specificity_cutoffs": {
            f"{int(sp*100)}pct_decoy_specificity": _cutoff_report(
                a, d, float(np.quantile(d, 1 - sp))
            )
            for sp in (0.90, 0.95, 0.99)
        },
        "threshold_sweep": {f"vina_{t:.1f}": _cutoff_report(a, d, float(t)) for t in SWEEP},
        "_roc": (fpr, tpr),
    }


def _load(name_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(name_csv)
    return df[df.vina.notna()].reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(0)
    act = _load(HERE / "actives_docking_results.csv")
    dec = _load(HERE / "matched_decoys_docking_results.csv")
    r = analyze(act, dec, rng)

    summary = {
        "target": "human ClpP / QuickVina2-GPU docking (data/targets/ClpP.pdbqt = 7UVU)",
        "note": "Vina kcal/mol, lower = better binder. actives = ChEMBL4523305 binders "
        "(mostly EC50 imipridone activators); decoys = property-matched DUD-E.",
        "recommended_threshold_youden": r["best_cutoffs"]["youden"]["cutoff"],
        **{k: v for k, v in r.items() if k != "_roc"},
    }
    json.dump(summary, open(HERE / "summary_docking.json", "w"), indent=2)

    _make_figure(act, dec, r)
    _print_console(r)


def _make_figure(act, dec, r):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    lo = min(act.vina.min(), dec.vina.min())
    hi = max(act.vina.max(), dec.vina.max())
    bins = np.linspace(lo, hi, 40)
    ax[0, 0].hist(
        dec.vina,
        bins=bins,
        alpha=0.55,
        density=True,
        color=DEC_COLOR,
        label=f"matched decoys (n={len(dec)})",
    )
    ax[0, 0].hist(
        act.vina,
        bins=bins,
        alpha=0.55,
        density=True,
        color=ACT_COLOR,
        label=f"actives (n={len(act)})",
    )
    yc = r["best_cutoffs"]["youden"]["cutoff"]
    ax[0, 0].axvline(yc, ls="--", color="k", label=f"Youden cutoff {yc:.2f}")
    ax[0, 0].set_xlabel("ClpP Vina energy (kcal/mol, lower = better)")
    ax[0, 0].set_ylabel("density")
    ax[0, 0].set_title("Docking score: actives vs matched decoys")
    ax[0, 0].legend(fontsize=8)

    fpr, tpr = r["_roc"]
    ax[0, 1].plot(fpr, tpr, lw=2, color=ACT_COLOR, label=f"AUROC {r['auroc']:.3f}")
    yj = r["best_cutoffs"]["youden"]
    ax[0, 1].plot(
        yj["fpr_decoys_passing"],
        yj["tpr_actives_retained"],
        "ko",
        ms=8,
        label=f"Youden (J={yj['youden_j']:.2f})",
    )
    ax[0, 1].plot([0, 1], [0, 1], ls="--", color="gray")
    ax[0, 1].set_xlabel("false positive rate (decoys)")
    ax[0, 1].set_ylabel("true positive rate (actives)")
    ax[0, 1].set_title("ROC — ClpP docking discrimination")
    ax[0, 1].legend(fontsize=9)

    ax[1, 0].scatter(dec.mw, dec.vina, s=8, alpha=0.3, color=DEC_COLOR, label="matched decoys")
    ax[1, 0].scatter(act.mw, act.vina, s=8, alpha=0.3, color=ACT_COLOR, label="actives")
    ax[1, 0].axhline(yc, ls="--", color="k")
    ax[1, 0].set_xlabel("molecular weight (Da)")
    ax[1, 0].set_ylabel("ClpP Vina (kcal/mol)")
    ax[1, 0].set_title(
        f"Vina vs MW (size-confound check; MW-matched AUROC {r['mw_matched']['auroc']:.3f})"
    )
    ax[1, 0].legend(fontsize=9)

    efs = [r["threshold_sweep"][f"vina_{t:.1f}"]["enrichment_factor"] for t in SWEEP]
    efs = [e if np.isfinite(e) else np.nan for e in efs]
    ax[1, 1].plot(SWEEP, efs, "o-", color=ACT_COLOR)
    ax[1, 1].axhline(1.0, ls=":", color="gray", label="no enrichment")
    ax[1, 1].axvline(yc, ls="--", color="k", label=f"Youden {yc:.2f}")
    ax[1, 1].set_xlabel("Vina cutoff (kcal/mol, keep <=)")
    ax[1, 1].set_ylabel("enrichment factor (actives% / decoys%)")
    ax[1, 1].set_title("Enrichment vs cutoff")
    ax[1, 1].legend(fontsize=9)

    fig.suptitle("ClpP DOCKING discrimination: known binders vs matched decoys", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(HERE / "discrimination_docking.png", dpi=140)


def _print_console(r):
    print("\n" + "=" * 78)
    print(f"ClpP docking discrimination (human ClpP / 7UVU)")
    print("=" * 78)
    av, dv = r["actives_vina"], r["decoys_vina"]
    print(
        f"actives (n={r['n_actives']:3d}) vina median {av['median']:6.2f} [{av['p25']:.2f},{av['p75']:.2f}] min {av['min']:.2f}"
    )
    print(
        f"decoys  (n={r['n_decoys']:3d}) vina median {dv['median']:6.2f} [{dv['p25']:.2f},{dv['p75']:.2f}] min {dv['min']:.2f}"
    )
    print(
        f"AUROC {r['auroc']:.3f}   PR-AUC {r['pr_auc_average_precision']:.3f}   MW-matched AUROC {r['mw_matched']['auroc']:.3f}"
    )
    print("-" * 78)
    print("BEST DISCRIMINATING CUTOFFS (call hit if vina <= cutoff):")
    for name, c in r["best_cutoffs"].items():
        print(
            f"  {name:18s} cutoff {c['cutoff']:6.2f}  TPR {c['tpr_actives_retained']*100:5.1f}%  "
            f"FPR {c['fpr_decoys_passing']*100:5.1f}%  prec {c['precision']*100:5.1f}%  "
            f"J {c['youden_j']:.2f}  EF {c['enrichment_factor']:.1f}x"
        )
    print("HIGH-SPECIFICITY OPERATING POINTS:")
    for k, c in r["fixed_specificity_cutoffs"].items():
        print(
            f"  {k:24s} cutoff {c['cutoff']:6.2f}  TPR {c['tpr_actives_retained']*100:5.1f}%  "
            f"FPR {c['fpr_decoys_passing']*100:5.1f}%  EF {c['enrichment_factor']:.1f}x"
        )
    print("-" * 78)
    print(f"{'vina<=':>7} {'actives%':>9} {'decoys%':>9} {'enrich':>8}")
    for t in SWEEP:
        c = r["threshold_sweep"][f"vina_{t:.1f}"]
        ef = c["enrichment_factor"]
        print(
            f"{t:7.1f} {c['tpr_actives_retained']*100:8.1f}% {c['fpr_decoys_passing']*100:8.2f}% "
            f"{('inf' if ef==float('inf') else f'{ef:5.1f}x'):>8}"
        )
    print("=" * 78)
    yc = r["best_cutoffs"]["youden"]["cutoff"]
    print(f">>> RECOMMENDED ClpP mode threshold:  raw_score <= {yc:.2f} kcal/mol  (Youden-optimal)")
    print("Wrote summary_docking.json, discrimination_docking.png")


if __name__ == "__main__":
    main()
