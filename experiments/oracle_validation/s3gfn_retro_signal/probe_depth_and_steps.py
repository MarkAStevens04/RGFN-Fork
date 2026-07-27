"""Separate the rival explanations for the 0% scores (follow-up to expA_falsify.py).

  H2 TEMPLATE MISMATCH: hb105 templates can't reverse SMALL/SCENT chemistry.
  H3 STEP BUDGET:       retro max_steps=3 < SCENT's up-to-4 reactions, so full decomposition
                        is impossible regardless of chemistry.
Probes: (a) score the raw SMALL blocks themselves (in-stock => must be ~100% if the env works at
all); (b) score SCENT molecules binned by hub_depth (shallow = fewer reactions); (c) sweep max_steps.
"""
import csv
import random
import sys

sys.path.insert(0, "external/s3gfn/src")
from s3gfn.synthesizability import SynthesizabilityEvaluator

random.seed(0)

# (a) the raw SMALL blocks
blocks = []
for r in csv.DictReader(open("data/libraries/glue_standard_v1/fragments.csv")):
    s = (r.get("smiles") or "").strip()
    if s:
        blocks.append(s)
random.shuffle(blocks)
blocks = blocks[:20]

# (b) SCENT sEH molecules binned by hub_depth (proxy for #reactions)
by_depth = {}
for r in csv.DictReader(open("/scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv")):
    try:
        d = int(r["hub_depth"])
    except Exception:
        continue
    by_depth.setdefault(d, [])
    if len(by_depth[d]) < 15:
        by_depth[d].append(r["child_key"])
print(
    "SCENT sEH molecules by hub_depth:",
    {k: len(v) for k, v in sorted(by_depth.items())},
    flush=True,
)

for env in ("small_hb105", "zincfrag_hb105"):
    for steps in (3, 6):
        print(f"\n=== env={env}  max_steps={steps} ===", flush=True)
        ev = SynthesizabilityEvaluator(
            use_retrosynthesis=True, env=env, max_steps=steps, num_workers=4
        )
        sc = ev.score_batch(blocks)
        ok = sum(1 for x in sc if x > 0)
        print(
            f"  raw SMALL blocks (in-stock sanity)   {ok}/{len(sc)} = {100*ok/len(sc):.0f}%",
            flush=True,
        )
        for d in sorted(by_depth):
            mols = by_depth[d]
            sc = ev.score_batch(mols)
            ok = sum(1 for x in sc if x > 0)
            print(
                f"  SCENT sEH  hub_depth={d}  ({len(mols):2d} mols)  {ok}/{len(sc)} = {100*ok/len(sc):.0f}%",
                flush=True,
            )
