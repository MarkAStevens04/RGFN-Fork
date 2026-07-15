#!/usr/bin/env python
"""Property-matched (DUD-E-style) decoys for the sEH proxy benchmark (Logs/034).

The RGFN-policy decoys (`make_decoys.py`) are the operationally relevant negative,
but they differ from the actives in gross physicochemistry (they're larger, odd
poly-triazoles). A stronger control asks: does the proxy still separate real sEH
inhibitors from *drug-like molecules matched to the actives' properties*? That is
exactly a DUD-E decoy set: for each active, pick background molecules with the same
MW / logP / H-bond donor / H-bond acceptor / rotatable-bond / formal-charge profile
but a **different scaffold** (ECFP4 Tanimoto < 0.35 to every active), so the scorer
can't win on properties alone.

Pipeline:
  1. Fetch a large drug-like background pool from ChEMBL (molecule endpoint, binned
     across MW; cached to `background_pool_chembl.csv` so re-runs are offline).
  2. Clean/canonicalize the same way as the actives; drop anything that is (or is a
     near-duplicate of) a known sEH active.
  3. Per active, greedily pick the nearest-in-property unused background molecule
     (hard gates: |ΔMW| ≤ 40 Da, equal formal charge, Tanimoto < 0.35 to all actives)
     until we have ~one matched decoy per active.

Writes `data/validation-molecules/sEH_decoys_matched.csv` (curated, committed).

Run in the `rgfn` env (needs internet for the first fetch):
    python experiments/oracle_validation/seh_proxy/make_matched_decoys.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import requests
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
MOL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
TANIMOTO_MAX = 0.35  # a decoy must be < this to EVERY active (topologically distinct)
DMW_MAX = 40.0  # hard MW window for a candidate match (Da)


def clean_smiles(smiles: str) -> str | None:
    """Largest organic fragment, uncharged, canonical (matches fetch_seh_actives)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def fetch_background(cache: Path, mw_lo: int, mw_hi: int, bin_w: int, per_bin: int) -> list[str]:
    """Drug-like background SMILES from ChEMBL, binned across MW so the pool spans the
    actives' property range. Cached; only canonical SMILES are kept (payload projected)."""
    if cache.exists():
        smi = [r["smiles"] for r in csv.DictReader(open(cache))]
        print(f"[bg] loaded {len(smi)} cached background SMILES from {cache.name}", file=sys.stderr)
        return smi
    seen: set[str] = set()
    for lo in range(mw_lo, mw_hi, bin_w):
        hi = lo + bin_w
        offset, got = 0, 0
        while got < per_bin:
            params = {
                "molecule_properties__mw_freebase__gte": lo,
                "molecule_properties__mw_freebase__lte": hi,
                "only": "molecule_chembl_id,molecule_structures",
                "limit": 1000,
                "offset": offset,
            }
            for attempt in range(4):
                try:
                    r = requests.get(MOL_API, params=params, timeout=60)
                    r.raise_for_status()
                    break
                except requests.RequestException as exc:
                    if attempt == 3:
                        raise
                    time.sleep(2 * (attempt + 1))
            mols = r.json().get("molecules", [])
            for m in mols:
                ms = m.get("molecule_structures") or {}
                s = ms.get("canonical_smiles")
                if s:
                    seen.add(s)
            got += len(mols)
            offset += 1000
            if len(mols) < 1000:
                break
        print(f"[bg] MW [{lo},{hi}]: pool now {len(seen)}", file=sys.stderr)
    smi = sorted(seen)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles"])
        for s in smi:
            w.writerow([s])
    print(f"[bg] wrote {len(smi)} background SMILES to {cache}", file=sys.stderr)
    return smi


def featurize(smiles_list: list[str]):
    """(kept canonical SMILES, prop matrix [MW,logP,HBD,HBA,RotB,charge], fp list)."""
    canon, props, fps = [], [], []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)  # ECFP4
        except Exception:
            continue
        canon.append(Chem.MolToSmiles(mol))
        props.append(
            [
                Descriptors.MolWt(mol),
                Crippen.MolLogP(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                Chem.GetFormalCharge(mol),
            ]
        )
        fps.append(fp)
    return canon, np.array(props, dtype=float), fps


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--actives",
        type=Path,
        default=REPO_ROOT / "data" / "validation-molecules" / "sEH_actives_chembl.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "data" / "validation-molecules" / "sEH_decoys_matched.csv",
    )
    ap.add_argument("--cache", type=Path, default=HERE / "background_pool_chembl.csv")
    ap.add_argument("--mw-lo", type=int, default=180)
    ap.add_argument("--mw-hi", type=int, default=700)
    ap.add_argument("--bin-w", type=int, default=52)
    ap.add_argument("--per-bin", type=int, default=3000)
    ap.add_argument("--per-active", type=int, default=2, help="max passes / decoys per active")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    act_rows = list(csv.DictReader(open(args.actives)))
    act_smiles = [r["smiles"] for r in act_rows]
    act_set = set(act_smiles)
    print(f"[match] {len(act_smiles)} actives", file=sys.stderr)

    bg_raw = fetch_background(args.cache, args.mw_lo, args.mw_hi, args.bin_w, args.per_bin)
    # Clean + drop exact actives.
    bg_clean = []
    for s in bg_raw:
        c = clean_smiles(s)
        if c is not None and c not in act_set:
            bg_clean.append(c)
    bg_clean = sorted(set(bg_clean))
    print(
        f"[match] {len(bg_clean)} cleaned background candidates (deduped, minus actives)",
        file=sys.stderr,
    )

    print("[match] featurizing actives + background...", file=sys.stderr)
    _, act_props, act_fps = featurize(act_smiles)
    bg_canon, bg_props, bg_fps = featurize(bg_clean)

    # Topological-distinctness gate: keep bg with max ECFP4 Tanimoto to ANY active < TANIMOTO_MAX.
    print(
        f"[match] computing max Tanimoto-to-actives for {len(bg_fps)} candidates...",
        file=sys.stderr,
    )
    keep = np.zeros(len(bg_fps), dtype=bool)
    for i, fp in enumerate(bg_fps):
        if max(DataStructs.BulkTanimotoSimilarity(fp, act_fps)) < TANIMOTO_MAX:
            keep[i] = True
    elig = np.where(keep)[0]
    print(
        f"[match] {len(elig)} eligible (Tanimoto < {TANIMOTO_MAX} to all actives)", file=sys.stderr
    )

    ep = bg_props[elig]  # eligible property matrix
    charge_e = ep[:, 5]
    mw_e = ep[:, 0]
    # normalization scales for the property distance
    scale = np.array([20.0, 1.0, 1.0, 2.0, 2.0, 1.0])
    available = np.ones(len(elig), dtype=bool)
    target = len(act_smiles)
    chosen: list[int] = []  # indices into elig

    order = rng.permutation(len(act_smiles))
    for _pass in range(args.per_active):
        for ai in order:
            if len(chosen) >= target:
                break
            ap_ = act_props[ai]
            gate = available & (np.abs(mw_e - ap_[0]) <= DMW_MAX) & (charge_e == ap_[5])
            cand = np.where(gate)[0]
            if cand.size == 0:
                continue
            dist = np.sqrt((((ep[cand] - ap_) / scale) ** 2).sum(axis=1))
            best = cand[int(np.argmin(dist))]
            available[best] = False
            chosen.append(best)
        if len(chosen) >= target:
            break

    sel = elig[np.array(chosen)]
    out_rows = [(bg_canon[j], round(bg_props[j, 0], 2)) for j in sel]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "mw"])
        for smi, mw in out_rows:
            w.writerow([smi, mw])

    # diagnostics: how well do the matched decoys' properties track the actives'?
    sp = bg_props[sel]
    print(f"\n[match] selected {len(sel)} matched decoys (target {target})", file=sys.stderr)
    print(
        f"  actives  MW median {np.median(act_props[:,0]):.0f}  logP {np.median(act_props[:,1]):.2f}  "
        f"HBA {np.median(act_props[:,3]):.0f}  RotB {np.median(act_props[:,4]):.0f}",
        file=sys.stderr,
    )
    print(
        f"  decoys   MW median {np.median(sp[:,0]):.0f}  logP {np.median(sp[:,1]):.2f}  "
        f"HBA {np.median(sp[:,3]):.0f}  RotB {np.median(sp[:,4]):.0f}",
        file=sys.stderr,
    )
    print(f"[match] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
