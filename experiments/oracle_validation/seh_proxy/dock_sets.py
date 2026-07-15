#!/usr/bin/env python
"""Dock one molecule set against sEH with the real GPU-docking oracle (Logs/034).

The companion to `benchmark_seh_proxy.py`: instead of the pretrained MPNN surrogate,
score the same actives / decoys with **actual QuickVina2-GPU docking against sEH**
(`glue.oracles.DockingSEHOracle` — the exact engine/box upstream RGFN uses), to see
whether real docking discriminates known inhibitors from decoys better than the proxy.

This script docks a **single** set per invocation (subsampled to --n) and writes its
Vina energies. It is deliberately one-set-per-process: sEH GPU docking costs ~1 s CPU
/ molecule (Meeko conformer prep), and the Balam login node kills any process that
exceeds 3600 s of CPU time (SIGXCPU); a fresh process per set keeps each run's CPU
budget well clear of that cap. Run the three sets sequentially (they share one GPU).

    source ~/bin/rgfn-smoke-env.sh
    python dock_sets.py --set actives  --n 1000
    python dock_sets.py --set rgfn     --n 1000
    python dock_sets.py --set matched  --n 1000

Writes `<set>_docking_results.csv` (smiles, vina, mw[, pchembl]); nan = failed to dock.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
VM = REPO_ROOT / "data" / "validation-molecules"

SETS = {  # name -> (input csv, output filename, carries pchembl?)
    "actives": (VM / "sEH_actives_chembl.csv", "actives_docking_results.csv", True),
    "rgfn": (HERE / "seed_decoys_rgfn.csv", "rgfn_decoys_docking_results.csv", False),
    "matched": (VM / "sEH_decoys_matched.csv", "matched_decoys_docking_results.csv", False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--n", type=int, default=1000, help="subsample this many molecules")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    in_csv, out_name, has_pchembl = SETS[args.set]
    df = pd.read_csv(in_csv)
    if len(df) > args.n:
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    smiles = df["smiles"].tolist()
    print(f"[dock:{args.set}] docking {len(smiles)} molecules against sEH ...", flush=True)

    # Imported here so a missing GPU stack fails loudly only when actually docking.
    from glue.oracles.docking_seh_oracle import DockingSEHOracle

    oracle = DockingSEHOracle()
    t0 = time.time()
    vina = oracle.score(smiles)
    dt = time.time() - t0

    out = pd.DataFrame({"smiles": smiles, "vina": vina, "mw": df["mw"].values})
    if has_pchembl and "pchembl" in df.columns:
        out["pchembl"] = df["pchembl"].values
    out_path = args.outdir / out_name
    out.to_csv(out_path, index=False)

    ok = out["vina"].notna().sum()
    good = out.loc[out["vina"].notna(), "vina"]
    print(
        f"[dock:{args.set}] docked {ok}/{len(smiles)} ok "
        f"({len(smiles)-ok} nan) in {dt:.0f}s ({dt/max(1,len(smiles)):.2f}s/mol wall)",
        flush=True,
    )
    if len(good):
        print(
            f"[dock:{args.set}] vina median {good.median():.2f}  "
            f"[{good.quantile(.25):.2f}, {good.quantile(.75):.2f}]  "
            f"min {good.min():.2f}",
            flush=True,
        )
    print(f"[dock:{args.set}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
