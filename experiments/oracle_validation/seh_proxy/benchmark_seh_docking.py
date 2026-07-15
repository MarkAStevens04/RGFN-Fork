#!/usr/bin/env python
"""Docking-based discrimination + head-to-head vs the MPNN proxy (Logs/034).

Reads the sEH Vina energies produced by `dock_sets.py` for the actives and the two
decoy sets, and runs the *same* discrimination analysis as `benchmark_seh_proxy.py`
— ROC-AUC, PR-AUC, MW-matched AUROC, and an enrichment sweep per decoy set — but on
real docking scores instead of the surrogate proxy. It then prints the proxy AUROCs
(read from `summary.json`) side by side, so we can see whether real docking
discriminates known sEH inhibitors from decoys better than the pretrained proxy does.

Docking convention: Vina binding energy in kcal/mol, **lower (more negative) = better
binder**. So for ROC we orient the score as `-vina`, and the cutoff sweep keeps
molecules with `vina <= cutoff`.

Run in the `rgfn` env, from the repo root (no GPU needed — reads the docked CSVs):
    python experiments/oracle_validation/seh_proxy/benchmark_seh_docking.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

HERE = Path(__file__).resolve().parent
SWEEP = np.arange(-6.0, -10.5, -0.5)  # Vina cutoffs (kcal/mol), loose -> strict
DECOY_SETS = {  # name -> (docked csv, plot colour)
    "rgfn": ("rgfn_decoys_docking_results.csv", "#8c8c8c"),
    "matched": ("matched_decoys_docking_results.csv", "#e07b39"),
}


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
    """Vina cutoff: keep molecules with vina <= cutoff (stronger binders)."""
    tpr = float((act <= cutoff).mean())
    fpr = float((dec <= cutoff).mean())
    return {
        "cutoff": float(cutoff),
        "tpr_actives_retained": tpr,
        "fpr_decoys_passing": fpr,
        "enrichment_factor": (tpr / fpr) if fpr > 0 else float("inf"),
        "n_actives_below": int((act <= cutoff).sum()),
        "n_decoys_below": int((dec <= cutoff).sum()),
    }


def analyze_pair(act: pd.DataFrame, dec: pd.DataFrame, rng) -> dict:
    a, d = act.vina.values, dec.vina.values
    y = np.concatenate([np.ones_like(a), np.zeros_like(d)])
    neg = np.concatenate([-a, -d])  # higher = better binder
    fpr, tpr, _ = roc_curve(y, neg)
    cuts = np.unique(np.concatenate([a, d]))
    youden = float(max(((a <= c).mean() - (d <= c).mean(), c) for c in cuts)[1])
    am, dm = mw_match(act, dec, 25.0, rng)
    ym = np.concatenate([np.ones(len(am)), np.zeros(len(dm))])
    negm = np.concatenate([-am.vina.values, -dm.vina.values])
    return {
        "n_decoys": int(len(dec)),
        "decoys_vina": _stats(d),
        "decoys_mw": _stats(dec.mw.values),
        "auroc": float(roc_auc_score(y, neg)),
        "pr_auc_average_precision": float(average_precision_score(y, neg)),
        "mw_matched": {
            "n_per_group": int(len(am)),
            "auroc": float(roc_auc_score(ym, negm)) if len(np.unique(ym)) == 2 else float("nan"),
        },
        "youden_cutoff": {
            "cutoff": youden,
            "tpr_actives_retained": float((a <= youden).mean()),
            "fpr_decoys_passing": float((d <= youden).mean()),
        },
        "fixed_specificity_cutoffs": {
            f"{int(sp*100)}pct_decoy_specificity": _cutoff_report(
                a, d, float(np.quantile(d, 1 - sp))
            )
            for sp in (0.90, 0.95, 0.99)
        },
        "threshold_sweep": {f"vina_{t}": _cutoff_report(a, d, float(t)) for t in SWEEP},
        "_roc": (fpr, tpr),
    }


def _load(name_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(name_csv)
    return df[df.vina.notna()].reset_index(drop=True)


def main() -> None:
    rng = np.random.default_rng(0)
    act = _load(HERE / "actives_docking_results.csv")
    decoys, results = {}, {}
    for name, (csv, _c) in DECOY_SETS.items():
        p = HERE / csv
        if not p.exists():
            print(f"[skip] {name} docked set missing: {p}")
            continue
        decoys[name] = _load(p)
        results[name] = analyze_pair(act, decoys[name], rng)

    # proxy AUROCs for the head-to-head (from the proxy benchmark)
    proxy_auroc = {}
    sj = HERE / "summary.json"
    if sj.exists():
        pj = json.load(open(sj))
        for name in results:
            if name in pj.get("decoy_sets", {}):
                proxy_auroc[name] = pj["decoy_sets"][name]["auroc"]

    # does docking rank actives the same way the proxy does?
    corr = {}
    apx = HERE / "actives_results.csv"
    if apx.exists():
        m = act.merge(pd.read_csv(apx)[["smiles", "proxy_score"]], on="smiles", how="inner")
        if len(m) > 2:
            corr = {
                "n": int(len(m)),
                "pearson_proxy_vs_negvina": float(pearsonr(m.proxy_score, -m.vina)[0]),
                "spearman_proxy_vs_negvina": float(spearmanr(m.proxy_score, -m.vina)[0]),
            }

    summary = {
        "target": "sEH / QuickVina2-GPU docking (data/targets/sEH.pdbqt)",
        "note": "Vina kcal/mol, lower = better binder; subsampled sets (see dock_sets.py)",
        "n_actives_docked": int(len(act)),
        "actives_vina": _stats(act.vina.values),
        "actives_mw": _stats(act.mw.values),
        "proxy_auroc_for_comparison": proxy_auroc,
        "docking_vs_proxy_on_actives": corr,
        "decoy_sets": {n: {k: v for k, v in r.items() if k != "_roc"} for n, r in results.items()},
    }
    json.dump(summary, open(HERE / "summary_docking.json", "w"), indent=2)

    _make_figure(act, decoys, results, proxy_auroc)
    _print_console(summary, results, act, proxy_auroc, corr)


def _make_figure(act, decoys, results, proxy_auroc):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    lo = min([act.vina.min()] + [d.vina.min() for d in decoys.values()])
    hi = max([act.vina.max()] + [d.vina.max() for d in decoys.values()])
    bins = np.linspace(lo, hi, 45)
    for name, d in decoys.items():
        ax[0, 0].hist(
            d.vina,
            bins=bins,
            alpha=0.5,
            density=True,
            color=DECOY_SETS[name][1],
            label=f"{name} decoys (n={len(d)})",
        )
    ax[0, 0].hist(
        act.vina,
        bins=bins,
        alpha=0.55,
        density=True,
        color="#2b7bba",
        label=f"actives (n={len(act)})",
    )
    ax[0, 0].set_xlabel("sEH Vina energy (kcal/mol, lower = better)")
    ax[0, 0].set_ylabel("density")
    ax[0, 0].set_title("Docking score: actives vs decoys")
    ax[0, 0].legend(fontsize=8)

    for name, r in results.items():
        fpr, tpr = r["_roc"]
        lbl = f"vs {name}: dock {r['auroc']:.3f}"
        if name in proxy_auroc:
            lbl += f" (proxy {proxy_auroc[name]:.3f})"
        ax[0, 1].plot(fpr, tpr, lw=2, color=DECOY_SETS[name][1], label=lbl)
    ax[0, 1].plot([0, 1], [0, 1], ls="--", color="gray")
    ax[0, 1].set_xlabel("false positive rate (decoys)")
    ax[0, 1].set_ylabel("true positive rate (actives)")
    ax[0, 1].set_title("ROC — docking (proxy in parens)")
    ax[0, 1].legend(fontsize=9)

    for name, d in decoys.items():
        ax[1, 0].scatter(
            d.mw, d.vina, s=6, alpha=0.25, color=DECOY_SETS[name][1], label=f"{name} decoys"
        )
    ax[1, 0].scatter(act.mw, act.vina, s=6, alpha=0.25, color="#2b7bba", label="actives")
    ax[1, 0].set_xlabel("molecular weight (Da)")
    ax[1, 0].set_ylabel("sEH Vina (kcal/mol)")
    ax[1, 0].set_title("Vina vs MW (size-confound check)")
    ax[1, 0].legend(fontsize=9)

    for name, r in results.items():
        efs = [r["threshold_sweep"][f"vina_{t}"]["enrichment_factor"] for t in SWEEP]
        efs = [e if np.isfinite(e) else np.nan for e in efs]
        ax[1, 1].plot(SWEEP, efs, "o-", color=DECOY_SETS[name][1], label=f"vs {name}")
    ax[1, 1].axhline(1.0, ls=":", color="gray", label="no enrichment")
    ax[1, 1].set_xlabel("Vina cutoff (kcal/mol, keep <=)")
    ax[1, 1].set_ylabel("enrichment factor")
    ax[1, 1].set_title("Enrichment vs cutoff")
    ax[1, 1].legend(fontsize=9)

    fig.suptitle("sEH DOCKING discrimination: known inhibitors vs decoys (Logs/034)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(HERE / "discrimination_docking.png", dpi=140)


def _print_console(summary, results, act, proxy_auroc, corr):
    names = list(results.keys())
    print("\n" + "=" * (30 + 22 * len(names)))
    for name in names:
        r = results[name]
        px = f"  [proxy {proxy_auroc[name]:.3f}]" if name in proxy_auroc else ""
        print(
            f"[{name:>8}]  DOCKING AUROC {r['auroc']:.3f}  PR-AUC {r['pr_auc_average_precision']:.3f}  "
            f"MW-matched {r['mw_matched']['auroc']:.3f}{px}  "
            f"(decoys vina median {r['decoys_vina']['median']:.2f})"
        )
    av = summary["actives_vina"]
    print(
        f"actives vina median {av['median']:.2f} [{av['p25']:.2f}, {av['p75']:.2f}], "
        f"min {av['min']:.2f}  (n={summary['n_actives_docked']})"
    )
    if corr:
        print(
            f"docking-vs-proxy on actives: Pearson {corr['pearson_proxy_vs_negvina']:.3f} / "
            f"Spearman {corr['spearman_proxy_vs_negvina']:.3f} (n={corr['n']})"
        )
    hdr = f"{'vina<=':>7} {'actives%':>9}" + "".join(
        f" | {n[:7]+' %':>9} {'enrich':>7}" for n in names
    )
    print(hdr)
    for t in SWEEP:
        row = f"{t:7.1f} {(act.vina.values <= t).mean()*100:8.1f}%"
        for name in names:
            d = results[name]["threshold_sweep"][f"vina_{t}"]
            ef = d["enrichment_factor"]
            row += f" | {d['fpr_decoys_passing']*100:8.2f}% {('inf' if ef==float('inf') else f'{ef:5.1f}x'):>7}"
        print(row)
    print("=" * (30 + 22 * len(names)))
    print("Wrote *_docking_results.csv, summary_docking.json, discrimination_docking.png")


if __name__ == "__main__":
    main()
