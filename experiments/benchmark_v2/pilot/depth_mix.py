#!/usr/bin/env python
"""How much of a hub-batching library sits on DEPTH-0 (bought) hubs?

WHY THIS IS NEEDED. A depth-0 hub is a catalogue building block: a chemist BUYS it, so the cost model
prices it at zero reactions and every one of its children costs exactly one. That is legitimate and
honestly priced -- but it is also the metric's named degenerate optimum, and switching the hub pool
from v1's reward pre-filter to `--pool all` raises depth-0 exposure ~4x in the walked prefix. The
share of a delivered library that rests on bought scaffolds therefore has to be REPORTED, not
assumed harmless. It is unmeasured today and unrecoverable for v1 (the hub-ordering arms kept only
plots, not per-step curves).

THE JOIN. `curve_hub_batching.csv` records `source_hub` per accepted molecule; the enumeration's
`hubs.csv` maps hub SMILES -> depth. Both are keyed on the raw hub SMILES the enumerator was handed,
so the join is exact -- 0 unjoined is asserted rather than hoped for.

Usage:
  python experiments/benchmark_v2/pilot/depth_mix.py \
      --run-dir /scratch/.../depthmix/base --hubs /scratch/.../enum/hubs.csv --budget 100
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def _strip_stereo(s: str) -> str:
    """Drop stereo marks. `hubs.csv` holds the RAW stereo-bearing SMILES handed to the enumerator,
    while some downstream keys are stereo-stripped (`meta.json` strip_stereo: true) — the same
    mismatch that made promoted-fragment recipes unlookup-able in cc25046. An exact join is tried
    first and this is only the fallback, counted and reported rather than applied silently."""
    return s.replace("@", "").replace("/", "").replace("\\", "")


def load_depth(hubs_csv: Path) -> tuple:
    with open(hubs_csv) as fh:
        exact = {r["smiles"]: int(r["depth"]) for r in csv.DictReader(fh)}
    stripped: dict = {}
    for k, v in exact.items():
        stripped.setdefault(_strip_stereo(k), v)
    return exact, stripped


def depth_of(hub: str, exact: dict, stripped: dict):
    """(depth, how) where how is 'exact' | 'stereo' | None."""
    if hub in exact:
        return exact[hub], "exact"
    d = stripped.get(_strip_stereo(hub))
    return (d, "stereo") if d is not None else (None, None)


def walk(curve_csv: Path, budget: int):
    """(rows within budget, all rows). Each row is one accepted molecule, in acceptance order."""
    with open(curve_csv) as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if int(r["cum_reactions"]) <= budget], rows


def report(run_dir: Path, depth: tuple, budget: int) -> dict:
    exact, stripped = depth
    out = {"run_dir": str(run_dir), "budget_rxns": budget}
    summ = json.loads((run_dir / "summary.json").read_text())
    out["min_synth_depth"] = summ.get("min_synth_depth")
    out["reward_threshold"] = summ.get("reward_threshold")

    for arm in ("hub_batching", "best_candidate"):
        p = run_dir / f"curve_{arm}.csv"
        if not p.exists():
            continue
        kept, allrows = walk(p, budget)
        hubs = [r.get("source_hub") or "" for r in kept]
        named = [h for h in hubs if h]
        resolved = [(h,) + depth_of(h, exact, stripped) for h in named]
        unjoined = [h for h, d, how in resolved if how is None]
        via_stereo = sum(1 for _, _, how in resolved if how == "stereo")
        d = Counter(dep for _, dep, how in resolved if how is not None)
        n = len(named)
        a = out.setdefault(arm, {})
        a["modes_at_budget"] = len(kept)
        a["total_modes"] = len(allrows)
        a["modes_with_named_hub"] = n
        a["unjoined_hubs"] = len(unjoined)
        a["joined_via_stereo_strip"] = via_stereo
        a["distinct_hubs_walked"] = len(set(named))
        a["hub_depth_of_walked_hubs"] = dict(
            sorted(Counter(dep for dep, how in (depth_of(h, exact, stripped) for h in set(named)) if how).items())
        )
        a["modes_by_hub_depth"] = dict(sorted(d.items()))
        a["share_modes_on_depth0"] = round(d.get(0, 0) / n, 4) if n else None

        # THE BUDGET SPLIT, which is the sharper reading. `reactions_added` on each accepted row is
        # what that molecule cost: the hub's own build the first time the walk enters it, plus one
        # marginal coupling. A depth-0 hub is BOUGHT, so it contributes 0 to the build and its
        # children cost exactly 1 each -- the cheapest mode a library can contain, and the reason
        # flow ranks such hubs highly. Splitting the budget by hub depth shows how much of the 100
        # reactions catalogue picking actually bought.
        rxn = Counter()
        for r, (_h, dep, how) in zip(kept, resolved):
            if how is None:
                continue
            try:
                rxn[dep] += int(r.get("reactions_added") or 0)
            except (TypeError, ValueError):
                pass
        tot = sum(rxn.values())
        a["reactions_by_hub_depth"] = dict(sorted(rxn.items()))
        a["reactions_accounted"] = tot
        a["share_reactions_on_depth0"] = round(rxn.get(0, 0) / tot, 4) if tot else None
        a["rxn_per_mode_by_hub_depth"] = {
            k: round(rxn[k] / d[k], 3) for k in sorted(d) if d.get(k)
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--hubs", type=Path, required=True)
    ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--json-out", type=Path, default=None)
    a = ap.parse_args()

    depth = load_depth(a.hubs)
    exact, _stripped = depth
    rep = report(a.run_dir, depth, a.budget)
    rep["n_hubs_in_hubs_csv"] = len(exact)
    rep["depth_of_full_hub_set"] = dict(sorted(Counter(exact.values()).items()))

    print(json.dumps(rep, indent=2))
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(json.dumps(rep, indent=2))

    for arm in ("hub_batching", "best_candidate"):
        if arm in rep and rep[arm]["unjoined_hubs"]:
            raise SystemExit(
                f"FATAL: {rep[arm]['unjoined_hubs']} accepted molecules name a hub absent from "
                f"{a.hubs}. The depth join is incomplete, so every share below is wrong."
            )


if __name__ == "__main__":
    main()
