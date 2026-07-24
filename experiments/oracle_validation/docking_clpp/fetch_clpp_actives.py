#!/usr/bin/env python
"""Fetch known human ClpP binders from ChEMBL (ClpP docking-threshold calibration).

The 16-cell matrix's ClpP cell docks against **human mitochondrial ClpP** — the RGFN
paper's benchmark ``data/targets/ClpP.pdbqt`` receptor. We identified it as human ClpP
(PDB **7UVU**, ClpP + the imipridone activator TR-107): the pdbqt spans chains B/C
residues **58-249**, a range only human ClpP (277 aa, mitochondrial transit peptide
1-56 cleaved) can occupy — S. aureus (193 aa) / E. coli (207 aa) ClpP cannot reach
residue 249. So the matching ChEMBL target is **CHEMBL4523305** (Homo sapiens,
"ATP-dependent Clp protease proteolytic subunit, mitochondrial").

Why this matters for the reward threshold (Logs entry): the 16-cell ClpP gate was a
provisional ``raw_score < -2.0`` that ~100 % of every generator clears (raw docking is
-5..-14 kcal/mol), so it does not discriminate. This script pulls real ClpP binders as
positives; ``make_matched_decoys.py`` builds property-matched negatives; ``dock_sets.py``
docks both with the *same* ``DockingClpPOracle`` used in training; ``benchmark_clpp_docking.py``
then reads off the Vina cutoff that best separates binders from decoys.

NOTE on endpoints: unlike sEH (IC50/Ki inhibitors), human ClpP's ChEMBL actives are
**~93 % EC50** — imipridone/ONC201-class *activators* (agonists) that bind the same
apical hydrophobic pocket as 7UVU's co-crystal ligand TR-107, i.e. exactly the site the
docking box is centred on. They are bona-fide ClpP binders, so we keep EC50/AC50/Potency
alongside IC50/Ki/Kd (keeping only IC50/Ki/Kd would drop ~93 % of the set).

Usage (any machine with internet + rdkit; `source ~/bin/rgfn-smoke-env.sh` first):
    python fetch_clpp_actives.py --min-pchembl 5 \
        --out ../../../data/validation-molecules/ClpP_actives_chembl.csv
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
CLPP_TARGET = "CHEMBL4523305"  # human mitochondrial ClpP (matches 7UVU receptor)
# ClpP binders are dominated by EC50 activators; keep binding + functional affinity types.
KEEP_TYPES = {"EC50", "AC50", "Potency", "IC50", "Ki", "Kd"}
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
        default=5.0,
        help="Minimum pChEMBL value (5 = 10 uM, 6 = 1 uM). ClpP's pool is smaller than "
        "sEH's, so 5 is the default to retain enough actives.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "data"
        / "validation-molecules"
        / "ClpP_actives_chembl.csv",
    )
    args = ap.parse_args()

    print(
        f"Fetching human ClpP ({CLPP_TARGET}) actives with pChEMBL >= {args.min_pchembl} ...",
        file=sys.stderr,
    )
    acts = fetch_activities(CLPP_TARGET, args.min_pchembl)

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

    # endpoint breakdown so the log records what kind of "binders" these are
    from collections import Counter

    types = Counter(r["standard_type"] for r in out_rows)
    print(f"\nActivity rows pulled:      {len(acts)}", file=sys.stderr)
    print(f"Unique compounds (raw):    {len(best)}", file=sys.stderr)
    print(f"Unparseable/dropped:       {n_unparse}", file=sys.stderr)
    print(f"Unique canonical actives:  {len(out_rows)}", file=sys.stderr)
    print(f"Endpoint mix:              {dict(types)}", file=sys.stderr)
    print(f"Wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
