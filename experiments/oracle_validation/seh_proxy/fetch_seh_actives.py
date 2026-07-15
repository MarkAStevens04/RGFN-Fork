#!/usr/bin/env python
"""Fetch known soluble-epoxide-hydrolase (sEH) inhibitors from ChEMBL.

The sEH proxy (`SehMoleculeProxy`, from `[bengio2021gflownet]`) is the field-standard
RGFN benchmark reward. To check whether its score is a meaningful "is this an sEH
binder?" signal (Logs/034) we need a set of real, experimentally-confirmed sEH
inhibitors as positives. This script pulls them straight from the ChEMBL REST API
(no exotic dependency: it uses `requests`, which `rgfn` already ships).

Target: **CHEMBL2409** — human "Bifunctional epoxide hydrolase 2" (gene EPHX2,
UniProt P34913), the single-protein sEH target.

We keep only potent, quantitative binding/functional potencies (IC50 / Ki / Kd with a
pChEMBL value, i.e. -log10(molar potency)) at or above a cutoff (default pChEMBL >= 6,
= 1 uM or better), deduplicate to one row per compound (best potency kept), strip
salts / take the largest organic fragment, canonicalize with RDKit, and write a small
CSV of unique actives. Output is the curated positives set for the proxy benchmark.

Usage (any machine with internet + rdkit):
    python fetch_seh_actives.py --min-pchembl 6 --out ../../../data/validation-molecules/sEH_actives_chembl.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")  # silence RDKit parse warnings on messy ChEMBL entries

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
SEH_TARGET = "CHEMBL2409"  # human bifunctional epoxide hydrolase 2 (EPHX2 / sEH)
KEEP_TYPES = {"IC50", "Ki", "Kd"}  # quantitative potency/affinity endpoints
PAGE_SIZE = 1000  # ChEMBL max


def fetch_activities(target: str, min_pchembl: float, page_size: int = PAGE_SIZE) -> list[dict]:
    """Page through all ChEMBL activities for `target` with pChEMBL >= min_pchembl."""
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "target_chembl_id": target,
            "pchembl_value__gte": min_pchembl,
            "limit": page_size,
            "offset": offset,
        }
        for attempt in range(4):
            try:
                r = requests.get(CHEMBL_API, params=params, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                print(f"  retry {attempt + 1} after error: {exc}", file=sys.stderr)
                time.sleep(2 * (attempt + 1))
        payload = r.json()
        batch = payload.get("activities", [])
        rows.extend(batch)
        total = payload.get("page_meta", {}).get("total_count")
        print(f"  fetched {len(rows)}/{total} activity rows (offset {offset})", file=sys.stderr)
        if len(batch) < page_size:
            break
        offset += page_size
    return rows


def clean_smiles(smiles: str) -> str | None:
    """Largest organic fragment, uncharged, canonical. None if unparseable/empty."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        mol = rdMolStandardize.FragmentParent(mol)  # drop salts/solvents -> parent
        mol = rdMolStandardize.Uncharger().uncharge(mol)
    except Exception:
        return None
    if mol is None or mol.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(mol)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-pchembl",
        type=float,
        default=6.0,
        help="Minimum pChEMBL value (6 = 1 uM, 7 = 100 nM, 8 = 10 nM).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "data"
        / "validation-molecules"
        / "sEH_actives_chembl.csv",
    )
    args = ap.parse_args()

    print(
        f"Fetching sEH ({SEH_TARGET}) actives with pChEMBL >= {args.min_pchembl} ...",
        file=sys.stderr,
    )
    acts = fetch_activities(SEH_TARGET, args.min_pchembl)

    # Dedup to one row per compound: keep the highest pChEMBL (most potent) measurement.
    best: dict[str, dict] = {}
    for a in acts:
        if a.get("standard_type") not in KEEP_TYPES:
            continue
        pchembl = a.get("pchembl_value")
        smi = a.get("canonical_smiles")
        chembl_id = a.get("molecule_chembl_id")
        if pchembl is None or not smi or not chembl_id:
            continue
        pchembl = float(pchembl)
        prev = best.get(chembl_id)
        if prev is None or pchembl > prev["pchembl"]:
            best[chembl_id] = {
                "chembl_id": chembl_id,
                "raw_smiles": smi,
                "pchembl": pchembl,
                "standard_type": a.get("standard_type"),
                "standard_value": a.get("standard_value"),
                "standard_units": a.get("standard_units"),
            }

    # Clean + canonicalize; collapse compounds that map to the same parent structure.
    by_canon: dict[str, dict] = {}
    n_unparse = 0
    for rec in best.values():
        canon = clean_smiles(rec["raw_smiles"])
        if canon is None:
            n_unparse += 1
            continue
        prev = by_canon.get(canon)
        if prev is None or rec["pchembl"] > prev["pchembl"]:
            rec = dict(rec)
            rec["smiles"] = canon
            rec["mw"] = round(Descriptors.MolWt(Chem.MolFromSmiles(canon)), 2)
            by_canon[canon] = rec

    out_rows = sorted(by_canon.values(), key=lambda r: -r["pchembl"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            [
                "smiles",
                "chembl_id",
                "pchembl",
                "standard_type",
                "standard_value",
                "standard_units",
                "mw",
            ]
        )
        for r in out_rows:
            w.writerow(
                [
                    r["smiles"],
                    r["chembl_id"],
                    r["pchembl"],
                    r["standard_type"],
                    r["standard_value"],
                    r["standard_units"],
                    r["mw"],
                ]
            )

    print(f"\nActivity rows pulled:      {len(acts)}", file=sys.stderr)
    print(f"Unique compounds (raw):    {len(best)}", file=sys.stderr)
    print(f"Unparseable/dropped:       {n_unparse}", file=sys.stderr)
    print(f"Unique canonical actives:  {len(out_rows)}", file=sys.stderr)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
