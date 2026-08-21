#!/usr/bin/env python
"""Fetch known dopamine-D2 binders from ChEMBL — with a TEMPORAL HOLDOUT, which is the whole point.

WHY THIS IS NOT THE SAME JOB AS THE OTHER THREE ORACLES. sEH, ClpP and 6TD3 are scored by *docking*:
a physics-based function that has never seen a label, so ChEMBL actives are a fair test of it. DRD2 is
scored by a **trained classifier** — the TDC/Olivecrona DRD2 SVM (`external/scent/oracle/drd2_current.pkl`),
fit on ChEMBL DRD2 actives-vs-inactives around 2017. Handing that model ChEMBL DRD2 actives and
reporting the AUROC would largely measure MEMORISATION, not discrimination: most of those molecules
are its training data. It would look excellent and mean almost nothing.

So this script builds two sets and keeps them apart:

  ALL-YEARS  every qualifying active. Scoring these gives IN-DOMAIN performance -- an upper bound,
             reported as such, never as validation.
  HELD-OUT   actives whose ChEMBL evidence is entirely from documents published in/after --since
             (default 2017). A molecule that appears in a 2020 paper but ALSO has a pre-2017 record
             is EXCLUDED, because it was available to the model regardless of which paper we found it
             in. That exclusion is the difference between a real holdout and a date filter.

Neither set is a perfect holdout -- ChEMBL deposition year is not the model's true training cutoff,
and the SVM's exact training set is not published -- so HELD-OUT is best read as "enriched for
molecules the model has not seen", and the gap between the two numbers is the honest measure of how
much of the in-domain score is memorisation.

Writes both to data/validation-molecules/ (curated, committed).

Run on a machine with internet (login node is fine):
    conda run -n rgfn python experiments/oracle_validation/drd2_proxy/fetch_drd2_actives.py
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import requests
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
DRD2_TARGET = "CHEMBL217"  # Dopamine D2 receptor (Homo sapiens)
# Binding endpoints only. DRD2 is a GPCR with a large functional-assay literature (EC50 agonism), but
# the oracle's notion of "active" is affinity, so functional potency is deliberately excluded here.
KEEP_TYPES = {"Ki", "Kd", "IC50"}
PAGE_SIZE = 1000
HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[3]


def _get(params: dict, tries: int = 5) -> dict:
    """One page, with backoff. The EBI endpoint times out intermittently on large pages and a bare
    request loses the whole fetch minutes in (observed: ConnectionError read-timeout at ~8 min)."""
    for k in range(tries):
        try:
            r = requests.get(CHEMBL_API, params=params, timeout=180)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - any transport failure is worth one more try
            if k == tries - 1:
                raise
            wait = 3 * (k + 1)
            print(
                f"    retry {k+1}/{tries - 1} after {type(e).__name__}: sleeping {wait}s",
                file=sys.stderr,
            )
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch(min_pchembl: float, cache: Path | None = None) -> list[dict]:
    """EVERY qualifying activity row, paged, cached as JSONL.

    Fetched ONCE, all years. The held-out subset is derived locally from `document_year` on these
    same rows -- a second server-side `document_year__gte` fetch would double both the network
    exposure and the (slow) RDKit standardisation for no new information.
    """
    if cache and cache.exists():
        import json as _json

        rows = [_json.loads(l) for l in open(cache)]
        print(f"    loaded {len(rows)} cached activity rows from {cache}", file=sys.stderr)
        return rows
    out, offset = [], 0
    params = {
        "target_chembl_id": DRD2_TARGET,
        "pchembl_value__gte": min_pchembl,
        "limit": PAGE_SIZE,
    }
    while True:
        params["offset"] = offset
        js = _get(dict(params))
        acts = js.get("activities", [])
        out.extend(acts)
        total = js["page_meta"]["total_count"]
        print(f"    fetched {len(out)}/{total}", file=sys.stderr)
        offset += PAGE_SIZE
        if offset >= total or not acts:
            break
        time.sleep(0.2)
    if cache:
        import json as _json

        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as fh:
            for r in out:
                fh.write(_json.dumps(r) + "\n")
        print(f"    cached {len(out)} rows -> {cache}", file=sys.stderr)
    return out


def clean_smiles(smiles: str) -> str | None:
    """Largest organic fragment, uncharged, canonical — identical to the ClpP/sEH fetchers and to
    make_matched_decoys, so actives and decoys are standardised the same way."""
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


def to_molecules(acts: list[dict]) -> dict[str, dict]:
    """{canonical_smiles: row}, keeping the highest pChEMBL AND the EARLIEST document year.

    The two are tracked separately on purpose. Keeping only the best-pChEMBL row would carry that
    row's year, which is not the year the structure first became available -- so a molecule measured
    in 2010 and re-measured more potently in 2020 would look like a 2020 molecule and pass a holdout
    filter it should fail. `min_year` is what the holdout test must key on.
    """
    best: dict[str, dict] = {}
    for a in acts:
        if a.get("standard_type") not in KEEP_TYPES:
            continue
        smi = clean_smiles(a.get("canonical_smiles"))
        pch = a.get("pchembl_value")
        if not smi or pch is None:
            continue
        pch = float(pch)
        yr = a.get("document_year")
        yr = int(yr) if yr else None
        prev = best.get(smi)
        if prev is None:
            best[smi] = {
                "smiles": smi,
                "chembl_id": a.get("molecule_chembl_id"),
                "pchembl": pch,
                "standard_type": a.get("standard_type"),
                "document_year": yr,
                "min_year": yr,
            }
            continue
        if pch > prev["pchembl"]:
            prev.update(
                pchembl=pch,
                standard_type=a.get("standard_type"),
                chembl_id=a.get("molecule_chembl_id"),
                document_year=yr,
            )
        if yr is not None and (prev["min_year"] is None or yr < prev["min_year"]):
            prev["min_year"] = yr
    return best


def write(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["smiles", "chembl_id", "pchembl", "standard_type", "document_year", "min_year"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows)} -> {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-pchembl", type=float, default=6.0)
    ap.add_argument(
        "--since",
        type=int,
        default=2017,
        help="held-out set = evidence only from documents in/after this year",
    )
    ap.add_argument("--out-dir", type=Path, default=REPO / "data" / "validation-molecules")
    a = ap.parse_args()

    print(f"[drd2] all-years actives (pChEMBL >= {a.min_pchembl})", file=sys.stderr)
    all_mols = to_molecules(fetch(a.min_pchembl, cache=HERE / "drd2_activities_raw.jsonl"))
    # The held-out set is a LOCAL partition of the same molecules: those whose earliest evidence is
    # at or after the cutoff. No second fetch, and no second standardisation pass.
    recent_mols = {
        s_: r
        for s_, r in all_mols.items()
        if r["min_year"] is not None and r["min_year"] >= a.since
    }

    # `recent_mols` is ALREADY the holdout: it is defined by `min_year >= since` over every activity
    # row for that molecule, so a structure first measured pre-cutoff and re-measured later is
    # excluded by construction. A molecule with an unknown earliest year is excluded too
    # (conservative -- it may predate the cutoff).
    held = recent_mols

    print(f"\n[drd2] all-years distinct molecules : {len(all_mols)}")
    print(f"[drd2] recent-document molecules    : {len(recent_mols)}")
    print(f"[drd2] HELD OUT (no pre-{a.since} record): {len(held)}")
    yr = Counter(r["min_year"] for r in held.values())
    print(f"[drd2] held-out by year: {dict(sorted(yr.items(), key=lambda kv: str(kv[0])))}")

    write(
        sorted(all_mols.values(), key=lambda r: -r["pchembl"]),
        a.out_dir / "DRD2_actives_chembl_all.csv",
    )
    write(
        sorted(held.values(), key=lambda r: -r["pchembl"]),
        a.out_dir / "DRD2_actives_chembl_heldout.csv",
    )


if __name__ == "__main__":
    main()
