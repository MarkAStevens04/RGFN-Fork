#!/usr/bin/env python
"""Normalise the 6TD3 actives into the `smiles`-column form the shared decoy matcher expects.

WHY NOT read `data/validation-molecules/DDB1_CDK12_Glues.csv` directly. Two reasons.
(1) Its column is `SMILES`, while `make_matched_decoys.py` (shared with ClpP + sEH) reads `smiles`.
(2) More importantly, the positives of the ROC curve must be the molecules that ACTUALLY HAVE
    docking scores. `known_results.csv` is the docked set: 175 curated glues went in, 160 came out
    with `status == ok`. Matching decoys to the 175 would build a negative set balanced against
    molecules that never scored, quietly skewing the property match.

So the actives here are taken FROM `known_results.csv`, cleaned the same way the ClpP/sEH actives
were (largest organic fragment, uncharged, canonical) so the property comparison is apples to apples.

Run:  conda run -n rgfn python experiments/oracle_validation/docking_6td3/prep_actives_6td3.py
"""
import csv
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SRC = HERE / "known_results.csv"
OUT = REPO / "data" / "validation-molecules" / "DDB1_CDK12_actives_docked.csv"


def clean(smiles: str):
    """Largest organic fragment, uncharged, canonical — identical to make_matched_decoys.clean_smiles
    so actives and decoys are standardised the same way."""
    if not smiles:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = rdMolStandardize.FragmentParent(m)
        m = rdMolStandardize.Uncharger().uncharge(m)
    except Exception:
        return None
    if m is None or m.GetNumHeavyAtoms() == 0:
        return None
    return Chem.MolToSmiles(m)


def main():
    rows = [r for r in csv.DictReader(open(SRC)) if r.get("status") == "ok"]
    seen, out = set(), []
    for r in rows:
        c = clean(r["smiles"])
        if c and c not in seen:
            seen.add(c)
            out.append({"smiles": c, "id": r.get("id", ""), "raw_smiles": r["smiles"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["smiles", "id", "raw_smiles"])
        w.writeheader()
        w.writerows(out)
    print(f"docked knowns (status ok): {len(rows)}")
    print(f"distinct after cleaning  : {len(out)}  -> {OUT}")


if __name__ == "__main__":
    main()
