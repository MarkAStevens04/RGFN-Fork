#!/usr/bin/env python
"""Full comparative analysis of the 48-run publication-scale fixed-reward matrix (Logs/030).

4 generators (rgfn, fraggfn, rxnflow, scent) x 4 systems (6td3, clpp, seh, drd2) x 3 seeds (42/43/44).
Reads each cell's emitted candidates.csv and computes reward, modes, diversity, scaffolds, a
synthesizability proxy, and drug-likeness; aggregates over the 3 seeds (mean +/- std); writes tidy
per-cell + aggregate CSVs and prints a per-system generator-comparison summary.

Conventions (matched to the repo so numbers line up with prior logs):
  * ECFP = Morgan r=3, 2048 bits (validation/lsdflow/metrics/diversity.py -- the canonical recipe).
  * modes = greedy sphere-exclusion, best-reward-first, sim<=0.7, gated above the per-system bar.
  * internal diversity = 1 - mean pairwise Tanimoto (same ECFP recipe).
  * per-system mode bar on the clip reward `score` (>=bar), == the calibrated RAW bar below clip=10:
      6td3 >=2.0 (Tier2-Tier1 differential <=-2.0), clpp >=8.0 (ClpP Vina <=-8.0, entry 045),
      seh  >=5.0 (bar-5; PROXY scale -- NOT an activity claim, see seh-proxy note),
      drd2 >=0.5 (DRD2 TDC activity, 0-1).
CAVEATS respected: one fingerprint recipe for all 4 (fairness); RxnFlow uneven N (dedup) makes COUNTS
(n_modes / n_scaffolds) confounded -- per-molecule medians are the clean cross-gen comparison; FragGFN
= non-synthesizable foil (has_route=0); seh proxy scale != activity.

Run (rgfn env): source ~/bin/rgfn-smoke-env.sh && python experiments/fixed_reward/scale5k/analyze_matrix.py
"""
import os
import statistics as st
import sys

import pandas as pd

REPO = os.path.expanduser("~/projects/RGFN_Fork/RGFN-Fork")
sys.path.insert(0, REPO)
from validation.lsdflow.metrics import diversity as D  # noqa: E402

FR = "/scratch/markymoo/rgfn_runs/experiments/fixed_reward"
OUT = f"{REPO}/experiments/fixed_reward/scale5k/analysis"
os.makedirs(OUT, exist_ok=True)
GENS = ["rgfn", "fraggfn", "rxnflow", "scent"]
SYSS = ["6td3", "clpp", "seh", "drd2"]
SEEDS = [42, 43, 44]
BAR = {"6td3": 2.0, "clpp": 8.0, "seh": 5.0, "drd2": 0.5}  # on `score` (higher = better)
KIND = {"6td3": "docking", "clpp": "docking", "seh": "surrogate", "drd2": "surrogate"}


def _med(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series(dtype=float)
    return float(v.median()) if len(v) else float("nan")


def _frac(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna() if col in df else pd.Series(dtype=float)
    return float(v.mean()) if len(v) else float("nan")


def cell_metrics(gen, sys_, seed):
    cand = f"{FR}/{gen}_{sys_}_5k/seed{seed}/fixed_reward/candidates/candidates.csv"
    if not os.path.exists(cand):
        return None
    df = pd.read_csv(cand)
    smi = df["smiles"].astype(str).tolist()
    score = pd.to_numeric(df["score"], errors="coerce").fillna(float("-inf")).tolist()
    fps = [D.ecfp(s) for s in smi]  # Morgan r3/2048, cached once
    top = sorted((s for s in score if s == s), reverse=True)
    mps = D.mean_pairwise_similarity(smi, fps=fps)
    return dict(
        gen=gen,
        sys=sys_,
        kind=KIND[sys_],
        seed=seed,
        N=len(df),
        N_unique=len(set(smi)),
        validity=_frac(df, "valid"),
        reward_mean=float(st.fmean(top)) if top else float("nan"),
        reward_med=float(st.median(top)) if top else float("nan"),
        reward_top10=float(st.fmean(top[:10])) if top else float("nan"),
        reward_max=float(top[0]) if top else float("nan"),
        raw_med=_med(df, "raw_score"),
        n_modes=D.count_modes(
            smi,
            rewards=score,
            higher_is_better=True,
            reward_threshold=BAR[sys_],
            similarity_threshold=0.7,
            fps=fps,
        ),
        n_scaffolds=D.unique_scaffolds(smi),
        int_diversity=(1.0 - mps) if mps is not None else float("nan"),
        has_route=_frac(df, "has_route"),
        num_reactions_med=_med(df, "num_reactions"),
        qed_med=_med(df, "qed"),
        mw_med=_med(df, "mol_weight"),
        clogp_med=_med(df, "clogp"),
        tpsa_med=_med(df, "tpsa"),
        rings_med=_med(df, "num_rings"),
        rotb_med=_med(df, "rotatable_bonds"),
        heavy_med=_med(df, "heavy_atoms"),
        lipinski=_frac(df, "lipinski_pass"),
    )


rows = []
for gen in GENS:
    for sys_ in SYSS:
        for seed in SEEDS:
            r = cell_metrics(gen, sys_, seed)
            if r is None:
                print(f"  MISSING {gen}_{sys_}_s{seed}", flush=True)
                continue
            rows.append(r)
            print(
                f"  {gen:8s} {sys_:4s} s{seed}: N={r['N']:4d} rew_med={r['reward_med']:6.2f} "
                f"top10={r['reward_top10']:6.2f} modes={r['n_modes']:3d} div={r['int_diversity']:.2f} "
                f"QED={r['qed_med']:.2f} MW={r['mw_med']:.0f} route={r['has_route']:.2f}",
                flush=True,
            )

per = pd.DataFrame(rows)
per.to_csv(f"{OUT}/per_cell.csv", index=False)

# aggregate over the 3 seeds: mean +/- std per (system, generator)
MET = [
    "N",
    "reward_med",
    "reward_top10",
    "reward_max",
    "raw_med",
    "n_modes",
    "n_scaffolds",
    "int_diversity",
    "has_route",
    "num_reactions_med",
    "qed_med",
    "mw_med",
    "clogp_med",
    "lipinski",
]
agg = per.groupby(["sys", "gen"])[MET].agg(["mean", "std"])
agg.to_csv(f"{OUT}/aggregate.csv")


# ---- printed per-system comparison (mean +/- std over 3 seeds) ----
def cell(sys_, gen, m):
    sub = per[(per.sys == sys_) & (per.gen == gen)][m]
    if not len(sub):
        return "  --  "
    return f"{sub.mean():.2f}±{sub.std():.2f}" if sub.std() == sub.std() else f"{sub.mean():.2f}"


print("\n" + "=" * 90)
print("PER-SYSTEM GENERATOR COMPARISON  (mean ± std over seeds 42/43/44)")
for sys_ in SYSS:
    print(f"\n### {sys_}  ({KIND[sys_]}, mode bar score>={BAR[sys_]}) ###")
    print(
        f"  {'gen':8s} {'N':>13s} {'reward_med':>13s} {'reward_top10':>13s} {'n_modes':>13s} "
        f"{'int_div':>13s} {'QED':>11s} {'MW':>13s} {'has_route':>11s}"
    )
    for gen in GENS:
        print(
            f"  {gen:8s} {cell(sys_,gen,'N'):>13s} {cell(sys_,gen,'reward_med'):>13s} "
            f"{cell(sys_,gen,'reward_top10'):>13s} {cell(sys_,gen,'n_modes'):>13s} "
            f"{cell(sys_,gen,'int_diversity'):>13s} {cell(sys_,gen,'qed_med'):>11s} "
            f"{cell(sys_,gen,'mw_med'):>13s} {cell(sys_,gen,'has_route'):>11s}"
        )
print("\n" + "=" * 90)
print(f"wrote {OUT}/per_cell.csv  and  {OUT}/aggregate.csv  ({len(per)} cells)")
