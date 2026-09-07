#!/usr/bin/env python
"""OURS. Route one molecule list against SEVERAL purchasable catalogues at an identical search
budget, and report the solve rate + route length per catalogue.

THE QUESTION. The competitor pipeline (generator -> MultiAiZ -> SPARROW) reaches the floor of the
reactions-per-mode metric because 84-100% of the molecules it selects are ONE reaction from ZINC's
17.4M purchasable compounds. Our own cost model gives hub-batching 418 free purchasable blocks. That
is a ~40,000x asymmetry in what counts as free, and it is invisible in the headline number. This
script measures how much of the competitor's advantage is catalogue access, by holding the molecules
and the search budget fixed and varying only the catalogue.

SOLVABILITY, NOT COST. Deliberately plain AiZynthFinder per molecule, NOT MultiAiZ. MultiAiZ's whole
purpose is finding intermediates SHARED across a set, which is a pricing question; "can this molecule
be made from this catalogue at all" is a per-molecule question, and answering it per molecule is both
the right instrument and ~an order of magnitude cheaper -- which is what lets a rung fit in a 2 h
debug slot instead of needing MultiAiZ's ~2.25 h per pool on the backlogged compute queue.

THE RUNGS, AND WHY TWO OF THEM ARE CONTROLS. A small catalogue can hurt for two different reasons:
it is SMALL, or it is the WRONG CHEMISTRY for these molecules. Curated rungs (zincfrag, rgfnlib) are
paired with random draws from ZINC of the same size (zinc_rand178k, zinc_rand418), so the gap between
a curated rung and its random twin separates the two. Without the twins a size effect reads as a
chemistry effect -- the same trap the depth experiment's size-matched control was built to catch.

BUDGET IS FIXED ACROSS RUNGS AND THAT MATTERS. A smaller catalogue needs a DEEPER search, so a tight
budget penalises the small rungs for a reason that is not the catalogue. Defaults are Logs/047's
production budget (iter=100 / time=60 / depth=6), the same one the 48.7% -> 73.8% ladder was measured
at, so these numbers sit beside those. Do not tighten it to make a rung fit -- shorten the molecule
list instead.

STOCK-MAJOR ORDER + INCREMENTAL WRITES. Rungs run in the order given and each line is flushed, so a
walltime kill leaves COMPLETE rates for the early rungs rather than partial rates for all of them.
Put the rungs you most need first. Re-running resumes: (stock, smiles) pairs already in the JSONL are
skipped, so a killed job is finished by resubmitting it unchanged.

Usage (aizynth env):
    python aiz_stock_ladder.py --smi pool100.smi --out ladder.jsonl \
        --config /scratch/.../ladder/config_ladder.yml \
        --stocks zinc,rgfnlib,zincfrag,zinc_rand418
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

IT = int(os.environ.get("AIZ_IT", "100"))
TL = int(os.environ.get("AIZ_TL", "60"))
MT = int(os.environ.get("AIZ_MT", "6"))


_W = {}


def _winit(config, it, tl, mt):
    """One AiZynthFinder per worker. Built in the initializer, NOT per task -- model load dominates
    a single search, so re-loading per molecule would cost more than the search itself."""
    from aizynthfinder.aizynthfinder import AiZynthFinder

    f = AiZynthFinder(configfile=config)
    f.expansion_policy.select("uspto")
    f.filter_policy.select("uspto")
    f.config.search.iteration_limit = it
    f.config.search.time_limit = tl
    f.config.search.max_transforms = mt
    _W["f"] = f


def _wroute(task):
    import time as _t

    smi, stock = task
    f = _W["f"]
    t0 = _t.time()
    try:
        f.stock.select([stock])
        f.target_smiles = smi
        f.tree_search()
        f.build_routes()
        trees = [r for r in f.routes.reaction_trees if r is not None]
        solved_lens = [
            sum(1 for _ in r.reactions()) for r in trees if getattr(r, "is_solved", False)
        ]
        return {
            "stock": stock,
            "smiles": smi,
            "solved": bool(solved_lens),
            "min_steps": min(solved_lens) if solved_lens else None,
            "n_solved_routes": len(solved_lens),
            "n_routes": len(trees),
            "s": round(_t.time() - t0, 2),
            "iterations": f.search_stats.get("iterations"),
        }
    except Exception as exc:  # noqa: BLE001 - one bad molecule must not kill the worker
        return {
            "stock": stock,
            "smiles": smi,
            "solved": False,
            "error": str(exc)[:200],
            "s": round(_t.time() - t0, 2),
        }


def _run_parallel(a, smiles, stocks, done, out_p):
    """Same records, same resume semantics, same stock-major order -- only the search is spread."""
    import multiprocessing as mp
    import time

    tasks = [(smi, st) for st in stocks for smi in smiles if (st, smi) not in done]
    if not tasks:
        print("[ladder] nothing to do")
        return
    print(f"[ladder] {len(tasks)} searches over {a.nproc} workers")
    t0 = time.time()
    n_solved = 0
    with out_p.open("a") as fh, mp.Pool(
        a.nproc, initializer=_winit, initargs=(a.config, IT, TL, MT)
    ) as pool:
        # imap_unordered + flush per record keeps the walltime-kill guarantee: whatever finished is
        # on disk, and a resubmit resumes from it.
        for i, rec in enumerate(pool.imap_unordered(_wroute, tasks, chunksize=1), 1):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_solved += int(bool(rec.get("solved")))
            if i % 50 == 0:
                el = time.time() - t0
                print(
                    f"[ladder] {i}/{len(tasks)}  solved={n_solved}  "
                    f"{el:.0f}s elapsed, {el / i:.2f}s/search effective",
                    flush=True,
                )
    print(f"[ladder] done: {n_solved}/{len(tasks)} solved in {time.time() - t0:.0f}s")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--smi", required=True, help="one SMILES per line")
    ap.add_argument("--out", required=True, help="JSONL, appended + flushed per molecule")
    ap.add_argument(
        "--config", required=True, help="AiZynth config carrying every rung as a stock key"
    )
    ap.add_argument(
        "--stocks", required=True, help="comma-separated stock keys, MOST IMPORTANT FIRST"
    )
    ap.add_argument(
        "--nproc",
        type=int,
        default=1,
        help="worker processes. Each loads its OWN AiZynthFinder (models + stock), so this trades "
        "memory for wall-clock: measured ~11 s/molecule on one core, and a Balam node has 128 cores "
        "and ~1 TB, so 32 is comfortable. Default 1 keeps single-process behaviour for reproducing "
        "earlier runs.",
    )
    a = ap.parse_args()

    smiles = [l.split()[0] for l in open(a.smi) if l.strip()]
    stocks = [s.strip() for s in a.stocks.split(",") if s.strip()]

    done = set()
    out_p = Path(a.out)
    if out_p.exists():  # resume: a killed job is finished by resubmitting unchanged
        for line in out_p.open():
            try:
                r = json.loads(line)
                done.add((r["stock"], r["smiles"]))
            except Exception:  # noqa: BLE001 - a truncated final line must not abort the resume
                pass
        print(f"[ladder] resuming — {len(done)} (stock, molecule) pairs already done")

    if a.nproc > 1:
        _run_parallel(a, smiles, stocks, done, out_p)
        return

    from aizynthfinder.aizynthfinder import AiZynthFinder

    finder = AiZynthFinder(configfile=a.config)
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")
    finder.config.search.iteration_limit = IT
    finder.config.search.time_limit = TL
    finder.config.search.max_transforms = MT
    print(
        f"[ladder] budget iter={IT} time={TL} depth={MT} | {len(smiles)} molecules | rungs {stocks}"
    )

    fh = out_p.open("a")
    for stock in stocks:
        finder.stock.select([stock])
        t_rung, n_solved, n_run = time.time(), 0, 0
        for smi in smiles:
            if (stock, smi) in done:
                continue
            t0 = time.time()
            try:
                finder.target_smiles = smi
                finder.tree_search()
                finder.build_routes()
                # ``ReactionTree.is_solved`` == "all leaf nodes are in stock" — the ONLY per-route
                # flag that means "makeable from THIS catalogue", which is the whole measurement.
                # TWO TRAPS, both caught by a 3-molecule smoke before this reached the queue:
                #   * ``search_stats`` has NO is_solved key (it carries time / iterations /
                #     first_solution_* / returned_first), so reading one there returns None and
                #     marks EVERY molecule unsolved — a silent all-zero ladder.
                #   * ``routes.reaction_trees`` includes UNSOLVED partial trees, whose reaction
                #     count is not a route length. Counting those reports a depth for a molecule
                #     the catalogue cannot actually make.
                trees = [r for r in finder.routes.reaction_trees if r is not None]
                solved_lens = [
                    sum(1 for _ in r.reactions()) for r in trees if getattr(r, "is_solved", False)
                ]
                rec = {
                    "stock": stock,
                    "smiles": smi,
                    "solved": bool(solved_lens),
                    "min_steps": min(solved_lens) if solved_lens else None,
                    "n_solved_routes": len(solved_lens),
                    "n_routes": len(trees),
                    "s": round(time.time() - t0, 2),
                    "iterations": finder.search_stats.get("iterations"),
                }
            except Exception as exc:  # noqa: BLE001 - one bad molecule must not lose the rung
                rec = {
                    "stock": stock,
                    "smiles": smi,
                    "solved": False,
                    "error": str(exc)[:200],
                    "s": round(time.time() - t0, 2),
                }
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            n_run += 1
            n_solved += int(bool(rec.get("solved")))
        print(
            f"[ladder] {stock}: {n_solved}/{n_run} solved this run " f"({n_solved / n_run:.1%})"
            if n_run
            else f"[ladder] {stock}: nothing to do",
            f"| {time.time() - t_rung:.0f}s",
            flush=True,
        )
    fh.close()
    print(f"[ladder] wrote {a.out}")


if __name__ == "__main__":
    main()
