#!/usr/bin/env python
"""Benchmark the pretrained sEH proxy: known inhibitors vs decoys (Logs/034).

Scores the positives and one or more decoy sets with the *exact* reward used to train
RGFN on the sEH benchmark (`SehMoleculeProxy`, the pretrained MPNN from
`[bengio2021gflownet]`), then measures how well the proxy score separates them and
derives an empirical "potential hit" cutoff.

  * POSITIVES     — known sEH inhibitors from ChEMBL (`fetch_seh_actives.py`).
  * DECOYS "rgfn" — randomly generated molecules from the untrained RGFN policy
                    (`make_decoys.py`); the operationally relevant negative (what the
                    generator proposes before training).
  * DECOYS "matched" — DUD-E-style property-matched decoys (`make_matched_decoys.py`):
                    drug-like ChEMBL molecules matched to the actives on
                    MW/logP/HBD/HBA/RotB/charge but topologically distinct
                    (ECFP4 Tanimoto < 0.35). The hard control: can the proxy separate
                    binders from look-alike non-binders it can't beat on properties?

Reported per decoy set: score distributions, ROC-AUC + PR-AUC, MW-matched AUROC (the
entry-`007` size control), Youden + fixed-specificity cutoffs, an enrichment sweep, and
how the project's existing sEH thresholds (7.0 in `rgfn_seh_proxy.gin`;
8.0 = `TanimotoSimilarityModes.proxy_term_threshold`) behave. Plus the proxy-vs-measured
-potency correlation among actives.

Scoring path note: `SehMoleculeProxy._compute_proxy_output` is literally
`self.model.compute_scores([...smiles...])`, so calling `proxy.model.compute_scores`
on raw SMILES is byte-identical to the in-training reward (same [1e-4, 100] clip).

Run in the `rgfn` env (`source ~/bin/rgfn-smoke-env.sh`), from the repo root:
    python experiments/oracle_validation/seh_proxy/benchmark_seh_proxy.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from rgfn.gfns.reaction_gfn.policies.graph_transformer import mol2graph
from rgfn.gfns.reaction_gfn.proxies.seh_proxy import SehMoleculeProxy

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXISTING_THRESHOLDS = [7.0, 8.0]  # 7.0 = entry 028 sEH hit bar; 8.0 = proxy_term_threshold
SWEEP = np.arange(3.0, 8.5, 0.5)  # cutoffs shown in the enrichment table
# display name -> (csv path, per-molecule results filename, plot colour)
DECOY_SETS = {
    "rgfn": ("seed_decoys_rgfn.csv", "decoys_results.csv", "#8c8c8c"),
    "matched": (
        REPO_ROOT / "data" / "validation-molecules" / "sEH_decoys_matched.csv",
        "matched_decoys_results.csv",
        "#e07b39",
    ),
}


def _scorable(smiles: list[str]) -> tuple[list[str], list[str]]:
    """(scorable SMILES, unscorable). Scorable iff RDKit parses AND `mol2graph` (the
    proxy's own featurizer) succeeds — so one bad molecule never crashes a batch."""
    good, bad = [], []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            bad.append(s)
            continue
        try:
            mol2graph(mol)
        except Exception:
            bad.append(s)
            continue
        good.append(s)
    return good, bad


def score_set(proxy: SehMoleculeProxy, smiles: list[str]) -> pd.DataFrame:
    good, bad = _scorable(smiles)
    if bad:
        print(f"  dropped {len(bad)} unscorable SMILES", flush=True)
    scores = proxy.model.compute_scores(good)  # == the in-training reward path
    mws = [Descriptors.MolWt(Chem.MolFromSmiles(s)) for s in good]
    return pd.DataFrame({"smiles": good, "proxy_score": scores, "mw": mws})


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


def _cutoff_report(act: np.ndarray, dec: np.ndarray, cutoff: float) -> dict:
    """TPR (actives kept), FPR (decoys passing) and enrichment factor (TPR/FPR)."""
    tpr = float((act >= cutoff).mean())
    fpr = float((dec >= cutoff).mean())
    return {
        "cutoff": float(cutoff),
        "tpr_actives_retained": tpr,
        "fpr_decoys_passing": fpr,
        "enrichment_factor": (tpr / fpr) if fpr > 0 else float("inf"),
        "n_actives_above": int((act >= cutoff).sum()),
        "n_decoys_above": int((dec >= cutoff).sum()),
    }


def mw_match(df_a: pd.DataFrame, df_d: pd.DataFrame, bin_w: float, rng):
    """Down-sample both groups so their MW histograms are identical (entry 007 control)."""
    lo = min(df_a.mw.min(), df_d.mw.min())
    hi = max(df_a.mw.max(), df_d.mw.max())
    edges = np.arange(lo, hi + bin_w, bin_w)
    a_bin = np.digitize(df_a.mw.values, edges)
    d_bin = np.digitize(df_d.mw.values, edges)
    keep_a, keep_d = [], []
    for b in np.unique(np.concatenate([a_bin, d_bin])):
        ia = np.where(a_bin == b)[0]
        idd = np.where(d_bin == b)[0]
        k = min(len(ia), len(idd))
        if k == 0:
            continue
        keep_a.extend(rng.choice(ia, k, replace=False))
        keep_d.extend(rng.choice(idd, k, replace=False))
    return df_a.iloc[keep_a].reset_index(drop=True), df_d.iloc[keep_d].reset_index(drop=True)


def analyze_pair(act: pd.DataFrame, dec: pd.DataFrame, mw_bin: float, rng) -> dict:
    """All metrics for actives vs one decoy set."""
    sa, sd = act.proxy_score.values, dec.proxy_score.values
    y = np.concatenate([np.ones_like(sa), np.zeros_like(sd)])
    s = np.concatenate([sa, sd])
    fpr, tpr, _ = roc_curve(y, s)
    # Youden-optimal cutoff = the score maximizing (TPR - FPR).
    cuts = np.unique(s)
    youden = float(max(((sa >= c).mean() - (sd >= c).mean(), c) for c in cuts)[1])
    am, dm = mw_match(act, dec, mw_bin, rng)
    ym = np.concatenate([np.ones(len(am)), np.zeros(len(dm))])
    sm = np.concatenate([am.proxy_score.values, dm.proxy_score.values])
    return {
        "n_decoys": int(len(dec)),
        "decoys_score": _stats(sd),
        "decoys_mw": _stats(dec.mw.values),
        "auroc": float(roc_auc_score(y, s)),
        "pr_auc_average_precision": float(average_precision_score(y, s)),
        "mw_matched": {
            "n_per_group": int(len(am)),
            "auroc": float(roc_auc_score(ym, sm)) if len(np.unique(ym)) == 2 else float("nan"),
            "decoys_mw_median": float(np.median(dm.mw)),
        },
        "youden_cutoff": {
            "cutoff": youden,
            "tpr_actives_retained": float((sa >= youden).mean()),
            "fpr_decoys_passing": float((sd >= youden).mean()),
        },
        "fixed_specificity_cutoffs": {
            f"{int(sp*100)}pct_decoy_specificity": _cutoff_report(
                sa, sd, float(np.quantile(sd, sp))
            )
            for sp in (0.90, 0.95, 0.99)
        },
        "existing_project_thresholds": {
            f"threshold_{t}": _cutoff_report(sa, sd, t) for t in EXISTING_THRESHOLDS
        },
        "threshold_sweep": {f"cutoff_{t}": _cutoff_report(sa, sd, float(t)) for t in SWEEP},
        "_roc": (fpr, tpr),  # for plotting; stripped before JSON
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--actives",
        type=Path,
        default=REPO_ROOT / "data" / "validation-molecules" / "sEH_actives_chembl.csv",
    )
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument("--mw-bin", type=float, default=25.0, help="MW-match bin width (Da)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    act_in = pd.read_csv(args.actives)
    print(f"Loaded {len(act_in)} actives. Building proxy...", flush=True)
    proxy = SehMoleculeProxy()

    print("Scoring actives...", flush=True)
    act = score_set(proxy, act_in["smiles"].tolist())
    act = act.merge(act_in[["smiles", "pchembl"]], on="smiles", how="left")
    act.to_csv(args.outdir / "actives_results.csv", index=False)
    sa = act.proxy_score.values

    # Score every available decoy set.
    decoys: dict[str, pd.DataFrame] = {}
    for name, (path, results_name, _c) in DECOY_SETS.items():
        p = path if Path(path).is_absolute() else (args.outdir / path)
        p = Path(p)
        if not p.exists():
            print(f"[skip] decoy set '{name}' not found at {p}", flush=True)
            continue
        print(f"Scoring '{name}' decoys ({p.name})...", flush=True)
        df = score_set(proxy, pd.read_csv(p)["smiles"].tolist())
        df.to_csv(args.outdir / results_name, index=False)
        decoys[name] = df

    results = {name: analyze_pair(act, df, args.mw_bin, rng) for name, df in decoys.items()}

    # proxy vs measured potency among actives
    apc = act.dropna(subset=["pchembl"])
    pear = float(pearsonr(apc.proxy_score, apc.pchembl)[0]) if len(apc) > 2 else float("nan")
    spear = float(spearmanr(apc.proxy_score, apc.pchembl)[0]) if len(apc) > 2 else float("nan")

    summary = {
        "target": "sEH / CHEMBL2409 (human EPHX2)",
        "proxy": "SehMoleculeProxy (bengio2021flow_proxy.pkl.gz)",
        "n_actives_scored": int(len(act)),
        "actives_score": _stats(sa),
        "actives_mw": _stats(act.mw.values),
        "proxy_vs_potency_among_actives": {
            "n": int(len(apc)),
            "pearson_r": pear,
            "spearman_r": spear,
        },
        "decoy_sets": {
            name: {k: v for k, v in r.items() if k != "_roc"} for name, r in results.items()
        },
    }
    with open(args.outdir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    _make_figure(args.outdir, act, decoys, results, sa)
    _print_console(summary, results, sa, pear, spear, len(apc))
    print(
        f"Wrote actives_results.csv, {', '.join(DECOY_SETS[n][1] for n in decoys)}, "
        f"summary.json, discrimination.png to {args.outdir}"
    )


def _make_figure(outdir, act, decoys, results, sa):
    fig, ax = plt.subplots(2, 2, figsize=(13, 10))
    hi = max([sa.max()] + [d.proxy_score.max() for d in decoys.values()])
    bins = np.linspace(0, hi, 50)
    # A: score distributions
    for name, df in decoys.items():
        ax[0, 0].hist(
            df.proxy_score,
            bins=bins,
            alpha=0.5,
            density=True,
            color=DECOY_SETS[name][2],
            label=f"{name} decoys (n={len(df)})",
        )
    ax[0, 0].hist(
        sa, bins=bins, alpha=0.55, density=True, color="#2b7bba", label=f"actives (n={len(sa)})"
    )
    for t, c in zip(EXISTING_THRESHOLDS, ["#d1495b", "#8b0000"]):
        ax[0, 0].axvline(t, ls="--", color=c, label=f"threshold {t}")
    ax[0, 0].set_xlabel("sEH proxy score (reward)")
    ax[0, 0].set_ylabel("density")
    ax[0, 0].set_title("Proxy score: actives vs decoys")
    ax[0, 0].legend(fontsize=8)
    # B: ROC per decoy set
    for name, r in results.items():
        fpr, tpr = r["_roc"]
        ax[0, 1].plot(
            fpr, tpr, lw=2, color=DECOY_SETS[name][2], label=f"vs {name} (AUROC={r['auroc']:.3f})"
        )
    ax[0, 1].plot([0, 1], [0, 1], ls="--", color="gray")
    ax[0, 1].set_xlabel("false positive rate (decoys)")
    ax[0, 1].set_ylabel("true positive rate (actives)")
    ax[0, 1].set_title("ROC")
    ax[0, 1].legend(fontsize=9)
    # C: score vs MW
    for name, df in decoys.items():
        ax[1, 0].scatter(
            df.mw,
            df.proxy_score,
            s=6,
            alpha=0.25,
            color=DECOY_SETS[name][2],
            label=f"{name} decoys",
        )
    ax[1, 0].scatter(act.mw, sa, s=6, alpha=0.25, color="#2b7bba", label="actives")
    ax[1, 0].set_xlabel("molecular weight (Da)")
    ax[1, 0].set_ylabel("sEH proxy score")
    ax[1, 0].set_title("Score vs MW (size-confound check)")
    ax[1, 0].legend(fontsize=9)
    # D: enrichment factor vs cutoff
    for name, r in results.items():
        efs = [r["threshold_sweep"][f"cutoff_{t}"]["enrichment_factor"] for t in SWEEP]
        efs = [e if np.isfinite(e) else np.nan for e in efs]
        ax[1, 1].plot(SWEEP, efs, "o-", color=DECOY_SETS[name][2], label=f"vs {name}")
    ax[1, 1].axhline(1.0, ls=":", color="gray", label="no enrichment")
    ax[1, 1].set_xlabel("proxy-score cutoff")
    ax[1, 1].set_ylabel("enrichment factor (actives/decoys)")
    ax[1, 1].set_title("Enrichment vs cutoff")
    ax[1, 1].legend(fontsize=9)

    fig.suptitle("sEH proxy discrimination: known inhibitors vs decoys (Logs/034)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(outdir / "discrimination.png", dpi=140)


def _print_console(summary, results, sa, pear, spear, n_apc):
    names = list(results.keys())
    print("\n" + "=" * (30 + 22 * len(names)))
    for name in names:
        r = results[name]
        print(
            f"[{name:>8}]  AUROC {r['auroc']:.3f}  PR-AUC {r['pr_auc_average_precision']:.3f}  "
            f"MW-matched AUROC {r['mw_matched']['auroc']:.3f}  "
            f"(decoys median {r['decoys_score']['median']:.2f}, MW {r['decoys_mw']['median']:.0f})"
        )
    print(
        f"actives score median {summary['actives_score']['median']:.2f} "
        f"[{summary['actives_score']['p25']:.2f}-{summary['actives_score']['p75']:.2f}], "
        f"MW {summary['actives_mw']['median']:.0f}"
    )
    print(f"proxy-vs-potency among actives: Pearson {pear:.3f} / Spearman {spear:.3f} (n={n_apc})")
    # combined enrichment table: one column-pair (decoys% + enrich) per decoy set
    hdr = f"{'cutoff':>7} {'actives%':>9}" + "".join(
        f" | {n[:7]+' %':>9} {'enrich':>7}" for n in names
    )
    print(hdr)
    for t in SWEEP:
        row = f"{t:7.1f} {(sa >= t).mean()*100:8.1f}%"
        for name in names:
            d = results[name]["threshold_sweep"][f"cutoff_{t}"]
            ef = d["enrichment_factor"]
            row += (
                f" | {d['fpr_decoys_passing']*100:8.2f}% "
                f"{('inf' if ef == float('inf') else f'{ef:5.1f}x'):>7}"
            )
        print(row)
    print("=" * (30 + 22 * len(names)))


if __name__ == "__main__":
    main()
