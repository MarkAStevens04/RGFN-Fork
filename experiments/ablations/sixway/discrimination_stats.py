"""
Scale-free discrimination stats for the 6TD3 (CDK12-DDB1) oracle:
known glues vs. decoys, per score/tier. Reuses Experiment 002 docking data
(no new docking).

Raw median gaps are NOT comparable across panels because the panels are in
different units (Vina kcal/mol vs. CNN pK vs. their differentials). This script
reports the unit-free measures that *are* comparable:

  - Cohen's d : standardized mean difference (pooled SD). Oriented so a positive
                d means binders are "better" (lower Vina / higher CNN).
                Caveat: assumes ~normal, equal-variance populations; docking-score
                tails violate this, so read d as a rough effect size and prefer
                AUROC for ranking.
  - AUROC     : P(a random binder scores better than a random decoy). Threshold-
                free, unit-free, and the fair cross-panel ranker. Equivalent to
                the Mann-Whitney U statistic / (n_known * n_decoy).
  - p (MWU)   : Mann-Whitney U one-sided p-value (binders better than decoys).

Note the six metrics are NOT independent: Tier 2 = Tier 1 + differential by
construction, so the absolute and differential scores share information.
Only status='ok' rows are used.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

ROOT = Path(__file__).resolve().parents[3]  # RGFN-Fork root
T2D3 = ROOT / "experiments" / "oracle_validation" / "docking_6td3"
# INPUTS ARE OVERRIDABLE (added 2026-08-20, Logs/069). The entry-006 ranking that selected
# Vina dT2-T1 as the oracle signal was measured against WARHEAD-matched decoys -- every decoy carries
# the CR8-like purine, but MW is only range-bounded and no other property is matched. Against
# PROPERTY-matched decoys the differential's AUROC falls 0.946 -> 0.688 and absolute Tier-2 overtakes
# it, so the six-way ranking has to be re-read on the harder negatives. Same six panels, same stats,
# different negative set:
#   KNOWN_CSV=... DECOY_CSV=... OUT_SUFFIX=_matched SET_LABEL="property-matched" python <script>
KNOWN_CSV = Path(os.environ.get("KNOWN_CSV", T2D3 / "known_results.csv"))
DECOY_CSV = Path(os.environ.get("DECOY_CSV", T2D3 / "decoy_cdk_results.csv"))
# Suffix keeps the two negative sets side by side instead of one overwriting the other.
OUT_SUFFIX = os.environ.get("OUT_SUFFIX", "")
SET_LABEL = os.environ.get("SET_LABEL", "warhead-matched")

# (label, column, better-direction) — must match plot_violins.py PANELS
PANELS = [
    ("Vina Tier 2", "vina_t2", "lower"),
    ("Vina Tier 1", "vina_t1", "lower"),
    ("CNN Tier 2", "cnnaff_t2", "higher"),
    ("CNN Tier 1", "cnnaff_t1", "higher"),
    ("Vina dT2-T1", "dvina", "lower"),
    ("CNN dT2-T1", "dcnnaff", "higher"),
]


def load(path):
    df = pd.read_csv(path)
    df = df[df["status"] == "ok"].copy()
    df["dvina"] = df["vina_t2"] - df["vina_t1"]
    df["dcnnaff"] = df["cnnaff_t2"] - df["cnnaff_t1"]
    return df


def cohens_d(a, b):
    """Pooled-SD standardized mean difference, a vs b."""
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return (a.mean() - b.mean()) / sp


def main():
    known, decoy = load(KNOWN_CSV), load(DECOY_CSV)
    n_known, n_decoy = len(known), len(decoy)
    rows = []
    for label, col, better in PANELS:
        binders, decoys = known[col].values, decoy[col].values
        sign = 1.0 if better == "higher" else -1.0  # orient so higher = better-for-binder
        b_or, d_or = sign * binders, sign * decoys
        d = cohens_d(b_or, d_or)
        U, p = mannwhitneyu(b_or, d_or, alternative="greater")
        auroc = U / (n_known * n_decoy)
        gap = float(np.median(binders) - np.median(decoys))  # in native units
        rows.append(
            {
                "panel": label,
                "column": col,
                "better": better,
                "decoy_median": float(np.median(decoys)),
                "known_median": float(np.median(binders)),
                "gap_known_minus_decoy": gap,
                "cohens_d": float(d),
                "auroc": float(auroc),
                "mwu_p": float(p),
            }
        )

    df = pd.DataFrame(rows).sort_values("auroc", ascending=False).reset_index(drop=True)
    out = Path(__file__).parent / f"discrimination_stats{OUT_SUFFIX}.csv"
    df.to_csv(out, index=False)

    print(f"6TD3 / CDK12-DDB1 — known vs. decoy (n={n_known} known / {n_decoy} decoy)")
    print("ranked by AUROC (the fair cross-panel discriminator):\n")
    print(f"{'panel':<13}{'gap':>8}{'cohen_d':>9}{'AUROC':>8}{'p(MWU)':>10}")
    for _, r in df.iterrows():
        print(
            f"{r['panel']:<13}{r['gap_known_minus_decoy']:>8.2f}"
            f"{r['cohens_d']:>9.2f}{r['auroc']:>8.3f}{r['mwu_p']:>10.1e}"
        )
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
