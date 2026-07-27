"""Falsification test for Exp A's negative result (Logs/048).

Exp A gave S3-GFN the reaction-GFN's 418 SMALL blocks but S3-GFN's OWN hb105 templates
(fragments-only, by design). It scored synth_ratio 0.0. Two rival explanations:
  (H1) GENUINE: GP-MolFormer's drug-like SMILES simply aren't reachable from 389 small blocks.
  (H2) ARTIFACT: hb105 templates cannot reverse SMALL-library chemistry at all (template mismatch),
       so NOTHING would score, not even molecules provably BUILT from those blocks.
Decisive probe: score SCENT's own molecules (built FROM the SMALL blocks via SMALL's 112 templates)
under small_hb105. H2 predicts ~0; H1 predicts a healthy rate.
"""
import csv
import os
import random
import sys

sys.path.insert(0, "external/s3gfn/src")
from s3gfn.synthesizability import SynthesizabilityEvaluator

random.seed(0)
N = int(os.environ.get("N", "30"))


def take(path, col, n=N):
    out, seen = [], set()
    with open(path) as fh:
        for row in csv.DictReader(fh):
            s = (row.get(col) or "").strip()
            if s and s not in seen:
                seen.add(s)
                out.append(s)
            if len(out) >= n * 8:
                break
    random.shuffle(out)
    return out[:n]


SETS = {
    "SCENT-sEH (built from SMALL blocks)": take(
        "/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv", "child_key"
    ),
    "SCENT-DRD2 (built from SMALL blocks)": take(
        "/scratch/markymoo/rgfn_runs/lsdflow/scent_drd2_70190/records.csv", "child_key"
    ),
    "S3-GFN output (ExpB pool)": take(
        "/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh_zincfrag_small/fixed_reward/candidates/candidates.csv",
        "smiles",
    ),
}
for k, v in SETS.items():
    print(f"  set '{k}': {len(v)} molecules", flush=True)

for env in ("small_hb105", "zincfrag_hb105"):
    print(f"\n=== retro env: {env} (max_steps=3, same as training) ===", flush=True)
    ev = SynthesizabilityEvaluator(use_retrosynthesis=True, env=env, max_steps=3, num_workers=4)
    for name, mols in SETS.items():
        if not mols:
            print(f"  {name:42s} (no molecules)")
            continue
        sc = ev.score_batch(mols)
        n_ok = sum(1 for x in sc if x > 0)
        print(f"  {name:42s} synthesizable: {n_ok}/{len(sc)} = {100*n_ok/len(sc):.0f}%", flush=True)
