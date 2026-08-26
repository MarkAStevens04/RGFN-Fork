#!/usr/bin/env python
"""What do OUR generated candidates score under 6TD3 vs 6TD3-B, and how far are they from real hits?

6TD3   = ddb1_dvina   (Vina Tier2 - Tier1 differential), lower is better  -- the incumbent reward
6TD3-B = cnnsc_t2     (gnina CNNscore on the selected Tier-2 pose), higher is better

THE QUESTION THIS ANSWERS, and why it is not just a third violin. A reward the generator was TRAINED
on is not a neutral measurement of that generator: the model has had thousands of steps to find
whatever the reward rewards. So the interesting comparison is three-way --

    known glues    what a real hit scores
    our candidates what the generator produces
    matched decoys what a realistic non-hit scores

-- because a generator that has exploited its reward will sit ABOVE the known hits on the trained
signal while sitting with the decoys on an untrained one. That pattern is invisible if you only look
at the trained signal, which is the whole reason for scoring the same molecules both ways.

Run:  conda run -n rgfn python experiments/oracle_validation/docking_6td3/compare_candidates_6td3b.py \
          --candidates <dock_6td3_scentsample_*/known_results.csv>
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
VAL = "/scratch/markymoo/rgfn_runs/dock_6td3_matched_74500"
SIGNALS = {"ddb1_dvina": ("6TD3   (differential)", "lower"),
           "cnnsc_t2":   ("6TD3-B (CNNscore T2)", "higher"),
           "cnnaff_t2":  ("       (CNN affinity T2)", "higher"),
           "vina_t2":    ("       (Vina T2 absolute)", "lower")}


def ld(p):
    d = pd.read_csv(p)
    return d[d.status == "ok"].copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", type=Path, required=True)
    ap.add_argument("--sample-csv", type=Path, default=HERE / "scent_6td3_sample.csv")
    a = ap.parse_args()

    act, dec, cand = ld(f"{VAL}/known_results.csv"), ld(f"{VAL}/decoy_cdk_results.csv"), ld(a.candidates)
    # join the subset label (top200 / random200) back on by id
    lab = pd.read_csv(a.sample_csv).set_index("DATAID")["subset"].to_dict()
    cand["subset"] = cand["id"].map(lab)

    print(f"known glues {len(act)} | matched decoys {len(dec)} | our candidates {len(cand)} "
          f"({cand.subset.value_counts().to_dict()})\n")
    for col, (name, sense) in SIGNALS.items():
        if col not in cand:
            continue
        A, D = act[col].dropna(), dec[col].dropna()
        print(f"{name}   ({'lower' if sense=='lower' else 'higher'} = better)")
        print(f"{'   known glues':<28}median {A.median():+7.3f}   p10 {A.quantile(.1):+7.3f}  p90 {A.quantile(.9):+7.3f}")
        print(f"{'   matched decoys':<28}median {D.median():+7.3f}   p10 {D.quantile(.1):+7.3f}  p90 {D.quantile(.9):+7.3f}")
        for sub in ("top200", "random200"):
            C = cand[cand.subset == sub][col].dropna()
            if C.empty:
                continue
            # where does our median sit relative to the two reference populations?
            pct_vs_act = (A < C.median()).mean() if sense == "higher" else (A > C.median()).mean()
            print(f"   our {sub:<23}median {C.median():+7.3f}   p10 {C.quantile(.1):+7.3f}  "
                  f"p90 {C.quantile(.9):+7.3f}   [beats {pct_vs_act:.0%} of real hits]")
        print()

    # the headline framing: does the generator look hit-like on the signal it was NOT trained on?
    print("=" * 78)
    for sub in ("top200", "random200"):
        C = cand[cand.subset == sub]
        if C.empty:
            continue
        d_pass = (C.ddb1_dvina <= -2.0).mean()
        b_pass = (C.cnnsc_t2 >= 0.96).mean()
        a_d = (act.ddb1_dvina <= -2.0).mean()
        a_b = (act.cnnsc_t2 >= 0.96).mean()
        dec_b = (dec.cnnsc_t2 >= 0.96).mean()
        print(f"{sub}:  clears 6TD3 gate (dvina<=-2.0) {d_pass:.0%}   |   "
              f"clears 6TD3-B (cnnsc>=0.96) {b_pass:.0%}")
        print(f"{'':<11}real hits            {a_d:.0%}   |   {'':<21}{a_b:.0%}")
        print(f"{'':<11}matched decoys        --    |   {'':<21}{dec_b:.0%}")
    print("=" * 78)


if __name__ == "__main__":
    main()
