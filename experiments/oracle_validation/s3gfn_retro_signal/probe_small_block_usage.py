"""Corrected probe: do Exp B (ZINCFrag u SMALL) routes ever USE a SMALL block?
Blocks live on RxnAction._block (not leaf .smi), so collect every action's block across the tree."""
import csv
import sys

sys.path.insert(0, "external/s3gfn/src")
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
from s3gfn.synthesizability import SynthesizabilityEvaluator


def flat(s):
    m = Chem.MolFromSmiles(s) if s else None
    return Chem.MolToSmiles(m, isomericSmiles=False) if m else None


small = {
    c
    for c in (
        flat((r.get("smiles") or "").strip())
        for r in csv.DictReader(open("data/libraries/glue_standard_v1/fragments.csv"))
    )
    if c
}
print(f"SMALL blocks: {len(small)}")

mols = []
with open(
    "/scratch/markymoo/rgfn_runs/experiments/fixed_reward/s3gfn_seh_zincfrag_small/fixed_reward/candidates/candidates.csv"
) as fh:
    for row in csv.DictReader(fh):
        s = (row.get("smiles") or "").strip()
        if s:
            mols.append(s)
        if len(mols) >= 60:
            break

ev = SynthesizabilityEvaluator(
    use_retrosynthesis=True, env="zincfrag_small_hb105", max_steps=3, num_workers=4
)


def blocks_of(t, acc):
    for action, sub in getattr(t, "branches", None) or []:
        b = getattr(action, "_block", None)
        if b:
            acc.append(b)
        blocks_of(sub, acc)


n_solved = n_with_small = tot = hits_tot = 0
for s in mols:
    t = ev.get_synthesis(s)
    if t is None:
        continue
    n_solved += 1
    acc = []
    blocks_of(t, acc)
    fl = [flat(b) for b in acc]
    fl = [x for x in fl if x]
    tot += len(fl)
    h = sum(1 for x in fl if x in small)
    hits_tot += h
    if h:
        n_with_small += 1
print(f"solved: {n_solved}/{len(mols)}   total blocks used across routes: {tot}")
print(f"molecules using >=1 SMALL block: {n_with_small}/{max(n_solved,1)}")
print(f"block-uses that are SMALL blocks: {hits_tot}/{max(tot,1)} = {100*hits_tot/max(tot,1):.1f}%")
