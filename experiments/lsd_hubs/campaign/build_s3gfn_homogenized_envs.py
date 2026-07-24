#!/usr/bin/env python
"""Build the two chemistry-homogenized S3-GFN retro envs (Exp A / Exp B, docs/LSD_FLOW_BENCHMARK_PLAN
chemistry-fairness). FRAGMENTS ONLY — the reaction TEMPLATES are never transferred (user directive
2026-07-24, mirroring the AiZynth-referee test in Logs/047): we add the reaction-GFN's 418 "small"
building blocks to the stock, but each method keeps its own reactions.

S3-GFN's retro env = `external/s3gfn/data/envs/<env>/{building_block.smi, template.txt}`, where
building_block.smi is tab-separated `<stereo-stripped-canonical-smiles>\t<id>` (S3-GFN indexes by
the exact stereo-stripped string; template.txt is the hb 105-template set).

  Exp A  small_hb105          = 389 SMALL frags ONLY          + hb105 templates  ("small library alone")
  Exp B  zincfrag_small_hb105 = ZINCFrag (178,622) U SMALL    + hb105 templates  ("best chance")

Both keep template.txt = the existing 105-template hb set verbatim. Run from repo root (s3gfn env):
  conda run -n s3gfn python experiments/lsd_hubs/campaign/build_s3gfn_homogenized_envs.py
"""
import csv
import shutil
from pathlib import Path

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

REPO = Path(__file__).resolve().parents[3]
ENVS = REPO / "external/s3gfn/data/envs"
ZINCFRAG_BB = ENVS / "zincfrag/building_block.smi"  # real file behind the hb105 symlink
HB105 = ENVS / "zincfrag_hb105/template.txt"  # the 105-template set (unchanged)
FRAGS = REPO / "data/libraries/glue_standard_v1/fragments.csv"


def small_lines():
    """389 unique stereo-stripped SMALL blocks as `smiles\\tid` lines (matches S3-GFN's format)."""
    seen, out = set(), []
    for row in csv.DictReader(open(FRAGS)):
        smi = (row.get("smiles") or "").strip()
        m = Chem.MolFromSmiles(smi) if smi else None
        if m is None:
            continue
        canon = Chem.MolToSmiles(m, isomericSmiles=False)  # stereo-stripped, S3-GFN's query form
        if canon in seen:
            continue
        seen.add(canon)
        out.append(f"{canon}\tSMALL{len(out):04d}")
    return out


def write_env(name, bb_lines):
    d = ENVS / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "building_block.smi").write_text("\n".join(bb_lines) + "\n")
    shutil.copy(HB105, d / "template.txt")
    print(
        f"  {name}: {len(bb_lines)} blocks + {sum(1 for _ in open(d/'template.txt'))} templates -> {d}"
    )


def main():
    small = small_lines()
    zincfrag = [ln.rstrip("\n") for ln in open(ZINCFRAG_BB) if ln.strip()]
    print(f"SMALL (stereo-stripped unique): {len(small)}  |  ZINCFrag: {len(zincfrag)}")
    print("building envs (fragments only; templates = hb105 verbatim):")
    write_env("small_hb105", small)  # Exp A
    write_env("zincfrag_small_hb105", zincfrag + small)  # Exp B (union; overlap=0, verified)


if __name__ == "__main__":
    main()
