#!/usr/bin/env python
"""Build a NEW AiZynthFinder stock from the reaction-GFN's 418 "small" building blocks, and a
config that unions it with ZINC — WITHOUT touching the pristine ZINC stock.

The reaction-GFN (RGFN/SCENT) assembles molecules from a fixed 418-fragment library
(`data/libraries/glue_standard_v1/fragments.csv` == `external/scent/data/small/fragments.txt`).
Those blocks are its assumed starting materials but most are NOT in ZINC (reactive handles, e.g.
`BrCc1ccccn1`). To ask "are these molecules synthesizable FROM the reaction-GFN's own blocks?", we
add the blocks to AiZynth's stock as a separate library keyed `rgfnlib` and select `[zinc, rgfnlib]`
(union) at runtime; the ZINC hdf5 is never modified.

STEREO: the blocks carry defined stereocenters but AiZynth's retro-fragments come out flat, so we
write BOTH each block's stereo InChIKey and its stereo-stripped InChIKey — matching the benchmark's
`strip_stereo=True` convention. (A stereo-retained-only stock badly undercounts; see Logs/047.)

Usage (in the `aizynth` env):
    python build_block_stock.py <out_stock.hdf5> <out_config.yml> [base_config.yml]
Base config defaults to data/models/aizynthfinder/config.yml (absolute paths, safe to copy).
"""
import csv
import sys

import pandas as pd
import yaml
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

FRAGS = "data/libraries/glue_standard_v1/fragments.csv"
BASE_CONFIG = "data/models/aizynthfinder/config.yml"


def main():
    out_stock = sys.argv[1]
    out_config = sys.argv[2]
    base_config = sys.argv[3] if len(sys.argv) > 3 else BASE_CONFIG

    keys = set()
    n = 0
    for row in csv.DictReader(open(FRAGS)):
        smi = (row.get("smiles") or "").strip()
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        n += 1
        keys.add(Chem.MolToInchiKey(m))  # stereo-retained
        m2 = Chem.MolFromSmiles(smi)
        Chem.RemoveStereochemistry(m2)
        keys.add(Chem.MolToInchiKey(m2))  # flat (matches AiZynth's stereo-less leaves)
    pd.DataFrame({"inchi_key": sorted(keys)}).to_hdf(out_stock, "table")
    print(f"blocks parsed: {n} -> {len(keys)} inchikeys (stereo + flat) -> {out_stock}")

    cfg = yaml.safe_load(open(base_config))
    cfg["stock"]["rgfnlib"] = out_stock  # ZINC entry untouched; add a NEW library key
    yaml.safe_dump(cfg, open(out_config, "w"), default_flow_style=False)
    print(
        f"wrote {out_config} (stock keys: {list(cfg['stock'].keys())}) "
        f"-> select ['zinc','rgfnlib'] for the union"
    )


if __name__ == "__main__":
    main()
