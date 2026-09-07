#!/usr/bin/env python
"""OURS. For every gate-passing candidate, its maximum Tanimoto similarity to ANY purchasable
building block. One number per molecule, computed once, so a whole tau sweep is free.

WHY ONE NUMBER. A "catalogue-distinct mode" must be Tanimoto-< tau from every block as well as from
every mode already accepted. Testing that inside the mode walk would compare each candidate against
~179,000 block fingerprints on every accept() call. But the quantity being thresholded does not
depend on tau -- it is the candidate's max similarity to the block set -- so computing it ONCE turns
every rung of the sweep into a float comparison.

WHY ZINCFrag AND NOT ZINC. `zinc_stock.hdf5` stores InChIKeys only, with no structures, so it cannot
be fingerprinted at all. ZINCFrag (178,622 blocks, SMILES) is the ZINC-derived building-block set we
can actually compare against, and it is the same set S3-GFN's synthesizability signal was trained on.
Label figures ZINCFrag, never "ZINC" -- the routing work uses the 17.4M stock and the two are not
interchangeable.

Output: JSONL, one {smiles, max_block_sim} per line, appended and flushed, skipping molecules already
present -- so a killed run is finished by re-running it.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
_REF = None


def _fp(smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    global _GEN
    try:
        _GEN
    except NameError:
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    m = Chem.MolFromSmiles(smiles)
    return _GEN.GetFingerprint(m) if m is not None else None


def _load_ref(zincfrag: str, fragments_csv: str):
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    ref = []
    with open(zincfrag) as fh:
        for line in fh:
            p = line.split()
            if p:
                f = _fp(p[0])
                if f is not None:
                    ref.append(f)
    n_zf = len(ref)
    with open(fragments_csv) as fh:
        for r in csv.DictReader(fh):
            f = _fp(r["smiles"])
            if f is not None:
                ref.append(f)
    return ref, n_zf


def _init(zincfrag, fragments_csv):
    global _REF
    _REF, _ = _load_ref(zincfrag, fragments_csv)


def _work(smi):
    from rdkit import DataStructs

    f = _fp(smi)
    if f is None:
        # Unparseable -> 1.0 so it is rejected at every tau rather than silently admitted.
        return {"smiles": smi, "max_block_sim": 1.0, "error": "unparseable"}
    return {
        "smiles": smi,
        "max_block_sim": round(max(DataStructs.BulkTanimotoSimilarity(f, _REF)), 4),
    }


def block_sims_for(smiles, zincfrag, fragments_csv, cache=None, nproc=8):
    """{smiles: max Tanimoto to any block}, reading and extending an optional JSONL cache.

    Importable so the pool builder and the sweep share ONE definition of the reference set --
    two copies would drift and the two sides of the comparison would stop meaning the same thing.
    """
    import json as _json
    import multiprocessing as _mp
    from pathlib import Path as _Path

    out = {}
    cache_p = _Path(cache) if cache else None
    if cache_p and cache_p.exists():
        for line in cache_p.open():
            line = line.strip()
            if line:
                r = _json.loads(line)
                out[r["smiles"]] = r["max_block_sim"]
    todo = [s for s in dict.fromkeys(smiles) if s not in out]
    if todo:
        fh = cache_p.open("a") if cache_p else None
        with _mp.Pool(nproc, initializer=_init, initargs=(zincfrag, fragments_csv)) as pool:
            for rec in pool.imap_unordered(_work, todo, chunksize=64):
                out[rec["smiles"]] = rec["max_block_sim"]
                if fh:
                    fh.write(_json.dumps(rec) + "\n")
        if fh:
            fh.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--enum-children", default="", help="our side: enum_children.json")
    ap.add_argument(
        "--candidates", default="", help="competitor side: candidates CSV (smiles,score)"
    )
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--higher-is-better", type=lambda v: v.lower() != "false", default=True)
    ap.add_argument("--out", required=True, help="JSONL cache; re-running resumes")
    ap.add_argument(
        "--zincfrag",
        default=str(REPO / "external/s3gfn/data/envs/zincfrag_hb105/building_block.smi"),
    )
    ap.add_argument(
        "--fragments", default=str(REPO / "data/libraries/glue_standard_v1/fragments.csv")
    )
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()

    if not a.enum_children and not a.candidates:
        raise SystemExit("[blocksim] pass --enum-children or --candidates")
    want = set()
    if a.enum_children:
        data = json.loads(Path(a.enum_children).read_text())
        for h in data.get("hubs", []):
            for c in h["children"]:
                r = float(c["reward"])
                if (r >= a.gate) if a.higher_is_better else (r <= a.gate):
                    want.add(c["smiles"])
    else:
        with open(a.candidates) as fh:
            for row in csv.DictReader(fh):
                smi = row.get("smiles") or row.get("SMILES")
                try:
                    r = float(row.get("score") or row.get("reward"))
                except (TypeError, ValueError):
                    continue
                if smi and ((r >= a.gate) if a.higher_is_better else (r <= a.gate)):
                    want.add(smi)
    out_p = Path(a.out)
    done = set()
    if out_p.exists():
        for line in out_p.open():
            line = line.strip()
            if line:
                done.add(json.loads(line)["smiles"])
    todo = sorted(want - done)
    print(
        f"[blocksim] {len(want)} distinct candidates clear gate {a.gate}; "
        f"{len(done)} cached, {len(todo)} to do, {a.nproc} workers",
        flush=True,
    )
    if not todo:
        return

    t0 = time.time()
    with out_p.open("a") as fh, mp.Pool(
        a.nproc, initializer=_init, initargs=(a.zincfrag, a.fragments)
    ) as pool:
        for i, rec in enumerate(pool.imap_unordered(_work, todo, chunksize=64), 1):
            fh.write(json.dumps(rec) + "\n")
            if i % 5000 == 0:
                fh.flush()
                el = time.time() - t0
                print(
                    f"[blocksim] {i}/{len(todo)}  {el:.0f}s  "
                    f"eta {el / i * (len(todo) - i) / 60:.0f} min",
                    flush=True,
                )
    print(f"[blocksim] done in {time.time() - t0:.0f}s -> {a.out}")


if __name__ == "__main__":
    main()
