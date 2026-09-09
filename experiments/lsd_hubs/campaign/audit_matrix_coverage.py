#!/usr/bin/env python
"""Full matrix coverage: which generator x target x seed x pool cells are MISSING which stage?

Stages checked per cell:
  pool    -- multiaiz_pools/<tag>_N*/pool.smi exists
  routes  -- cached routes (multiaiz_routes.json, or the generator's own routes.jsonl for SynFormer)
  greedy  -- results/<tag>_greedy_N*/greedy_frontier.csv with a readable R=100 row
  SB      -- results/<tag>_select_N500{,_longsolve}/select_frontier.csv with an R=100 row

A cell missing SB is HALF A COMPARISON: SB is the competitor's own selector and the arm it usually
wins on, so omitting it flatters us.
"""
import csv
import os
from pathlib import Path

SC = Path(os.environ["SCRATCH"]) / "rgfn_runs" / "lsdflow_sparrow"
POOLS, RES = SC / "multiaiz_pools", SC / "results"
FR = Path(os.environ["SCRATCH"]) / "rgfn_runs" / "experiments" / "fixed_reward"

GENS = ["s3gfn", "reinvent", "saturn", "tango", "fraggfn", "synformer"]
TGTS = ["seh", "drd2", "clpp"]
SEEDS = [42, 43, 44]


def tag_for(gen, tgt, seed, pool):
    # SynFormer never had Stage 2; the s3gfn DRD2 re-runs carry _stage2fix on seeds 43/44.
    if gen == "synformer":
        sfx = ""
    elif gen == "s3gfn" and tgt == "drd2" and seed in (43, 44):
        sfx = "_stage2fix"
    else:
        sfx = "_stage2"
    return f"{gen}_{tgt}_seed{seed}{sfx}" + ("_pruned" if pool == "pruned" else "")


def has_pool(tag):
    return any((d / "pool.smi").exists() for d in POOLS.glob(f"{tag}_N*"))


def has_routes(gen, tgt, seed, tag):
    if gen == "synformer":
        return (FR / f"{gen}_{tgt}" / f"seed{seed}" / "fixed_reward" / "candidates"
                / "routes.jsonl").exists()
    return any((d / "multiaiz_routes.json").exists() for d in POOLS.glob(f"{tag}_N*"))


def has_greedy(tag):
    for d in POOLS.parent.joinpath("results").glob(f"{tag}_greedy_N*"):
        f = d / "greedy_frontier.csv"
        if f.exists():
            for r in csv.DictReader(f.open()):
                try:
                    if float(r["used_rxns"]) <= 100:
                        return True
                except (KeyError, ValueError, TypeError):
                    pass
    return False


def has_sb(tag):
    # Glob the size, never assume _N500: a pool-limited cell writes _N<actual> (synformer drd2 44
    # pruned is _N495) and a hard-coded N500 reports a COMPLETE cell as missing.
    for d in sorted(RES.glob(f"{tag}_select_N*")):
        f = d / "select_frontier.csv"
        if not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            if r.get("budget_rxns") == "100":
                return True
    return False


missing = {"pool": [], "routes": [], "greedy": [], "SB": []}
complete = 0
total = 0
for gen in GENS:
    for tgt in TGTS:
        for seed in SEEDS:
            for pool in ("naive", "pruned"):
                total += 1
                tag = tag_for(gen, tgt, seed, pool)
                cell = f"{gen}:{tgt}:{seed}/{pool}"
                if not has_pool(tag):
                    missing["pool"].append(cell)
                    continue
                if not has_routes(gen, tgt, seed, tag):
                    missing["routes"].append(cell)
                    continue
                g, s = has_greedy(tag), has_sb(tag)
                if not g:
                    missing["greedy"].append(cell)
                if not s:
                    missing["SB"].append(cell)
                if g and s:
                    complete += 1

print(f"cells COMPLETE (pool+routes+greedy+SB): {complete} of {total}\n")
for stage in ("pool", "routes", "greedy", "SB"):
    m = missing[stage]
    print(f"missing {stage:<7} {len(m):>3}")
    if m:
        by_gen = {}
        for c in m:
            by_gen.setdefault(c.split(":")[0], []).append(c)
        for g, cs in sorted(by_gen.items()):
            print(f"    {g:<10} {len(cs):>2}  {' '.join(sorted(cs))[:150]}")
