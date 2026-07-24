#!/usr/bin/env python
"""Dock one molecule set against human ClpP with the real training oracle (ClpP calibration).

Scores actives / matched-decoys with the **exact** engine + box the 16-cell ClpP cell
trains on: ``glue.oracles.DockingClpPOracle`` (QuickVina2-GPU vs ``data/targets/ClpP.pdbqt``
= human ClpP / 7UVU). The returned Vina energies are on the same scale as the ClpP cell's
``raw_score`` column, so the cutoff we pick here is directly usable as that cell's mode gate.

One set per process (Balam login node kills > 3600 s CPU / SIGXCPU; a fresh process per set
keeps each run's CPU budget clear even though the sets are small). ``docking_batch_size=200``
docks each whole set in one QuickVina2-GPU process — the free 3.3x speed-up from Logs/036
(this is a fresh validation run, NOT the live campaign, so we apply it).

    source ~/bin/rgfn-smoke-env.sh          # rgfn env + QuickVina2-GPU + gnina + CUDA libs
    python dock_sets.py --set actives
    python dock_sets.py --set matched

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
    "actives": (VM / "ClpP_actives_chembl.csv", "actives_docking_results.csv", True),
    "matched": (VM / "ClpP_decoys_matched.csv", "matched_decoys_docking_results.csv", False),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", required=True, choices=list(SETS))
    ap.add_argument("--n", type=int, default=0, help="subsample this many (0 = dock ALL)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=200, help="SMILES per QuickVina2-GPU process")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    in_csv, out_name, has_pchembl = SETS[args.set]
    df = pd.read_csv(in_csv)
    if args.n and len(df) > args.n:
        df = df.sample(n=args.n, random_state=args.seed).reset_index(drop=True)
    smiles = df["smiles"].tolist()
    print(f"[dock:{args.set}] docking {len(smiles)} molecules against human ClpP ...", flush=True)

    # Imported here so a missing GPU stack fails loudly only when actually docking.
    from glue.oracles.docking_seh_oracle import DockingClpPOracle

    oracle = DockingClpPOracle(docking_batch_size=args.batch_size)
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
            f"[{good.quantile(.25):.2f}, {good.quantile(.75):.2f}]  min {good.min():.2f}",
            flush=True,
        )
    print(f"[dock:{args.set}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
