#!/usr/bin/env python
"""LSD-hub filter-dropoff funnel: raw children -> binding gate -> Tanimoto-dissimilar modes.

For each hub in an ``enumerated_records.csv`` (produced by the LSD-Flow harness enumeration,
``validation/lsdflow/harness/run.py --enumerate-top-hubs``), report how many one-reaction
terminal children survive each filter stage:

    raw children  --(reward/binding gate)-->  hits  --(greedy Tanimoto dedup)-->  hit-modes

so we can see the dropoff at every stage and *which* filter does the work per hub. Uses the
canonical, paper-comparable mode definition from ``validation/lsdflow/metrics/diversity``
(reward-gated greedy sphere-exclusion, ECFP Morgan r=3, 2048, similarity 0.7 — matching upstream
``TanimotoSimilarityModes``), so nothing about "mode" is reimplemented here.

Reusable across reward targets and DAGs: sEH (higher-is-better) by default; for docking
differentials pass ``--lower-is-better`` and docking-scale thresholds.

Run (login node, no GPU needed — pure CSV + RDKit):
    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/dropoff/funnel.py \
        --records /scratch/.../lsdflow/seh_stdlib_70140/enumerated_records.csv \
        --thresholds 6,7,7.5,8 --headline 7.0 \
        --out-dir experiments/lsd_hubs/dropoff --tag seh_70140
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from validation.lsdflow.metrics.diversity import count_modes


def load_hub_children(records_csv: str) -> Dict[str, dict]:
    """Group an ``enumerated_records.csv`` by hub; dedup children by canonical key (keep the
    best reward seen). Returns ``{hub_key: {depth, keys, rewards}}``."""
    hubs: Dict[str, dict] = defaultdict(lambda: {"depth": 0, "child": {}})
    with open(records_csv) as fh:
        for r in csv.DictReader(fh):
            h = hubs[r["hub_key"]]
            h["depth"] = int(r["hub_depth"])
            reward = float(r["reward"])
            prev = h["child"].get(r["child_key"])
            if prev is None or reward > prev:
                h["child"][r["child_key"]] = reward
    out = {}
    for k, h in hubs.items():
        out[k] = {
            "depth": h["depth"],
            "keys": list(h["child"].keys()),
            "rewards": list(h["child"].values()),
        }
    return out


def _n_pass(rewards: List[float], t: float, higher_is_better: bool) -> int:
    if higher_is_better:
        return sum(1 for r in rewards if r == r and r >= t)
    return sum(1 for r in rewards if r == r and r <= t)


def build_funnel(
    hubs: Dict[str, dict],
    thresholds: List[float],
    headline: float,
    similarity: float,
    higher_is_better: bool,
) -> dict:
    rows = []
    agg = {"raw": 0, "struct_modes": 0, "hit_modes": 0, **{f"pass_{t}": 0 for t in thresholds}}
    for key, h in sorted(hubs.items(), key=lambda kv: (kv[1]["depth"], -len(kv[1]["keys"]))):
        keys, rew, depth = h["keys"], h["rewards"], h["depth"]
        raw = len(keys)
        passes = {t: _n_pass(rew, t, higher_is_better) for t in thresholds}
        struct_modes = count_modes(keys, similarity_threshold=similarity)  # no gate
        hit_modes = count_modes(
            keys,
            rew,
            higher_is_better=higher_is_better,
            reward_threshold=headline,
            similarity_threshold=similarity,
        )
        n_hit_headline = _n_pass(rew, headline, higher_is_better)
        rows.append(
            {
                "hub_key": key,
                "depth": depth,
                "raw_children": raw,
                **{f"reward_pass_{t}": passes[t] for t in thresholds},
                "structure_modes": struct_modes,
                "children_per_structure_mode": round(raw / struct_modes, 2)
                if struct_modes
                else None,
                "hit_modes": hit_modes,
                "hits_per_hit_mode": round(n_hit_headline / hit_modes, 2) if hit_modes else None,
            }
        )
        agg["raw"] += raw
        agg["struct_modes"] += struct_modes
        agg["hit_modes"] += hit_modes
        for t in thresholds:
            agg[f"pass_{t}"] += passes[t]
    summary = {
        "similarity_threshold": similarity,
        "headline_reward_threshold": headline,
        "higher_is_better": higher_is_better,
        "n_hubs": len(hubs),
        "total_raw_children": agg["raw"],
        "survival": {
            str(t): {
                "count": agg[f"pass_{t}"],
                "pct": round(100 * agg[f"pass_{t}"] / agg["raw"], 1) if agg["raw"] else 0,
            }
            for t in thresholds
        },
        "total_structure_modes": agg["struct_modes"],
        "raw_to_structure_redundancy": round(agg["raw"] / agg["struct_modes"], 2)
        if agg["struct_modes"]
        else None,
        "total_hit_modes_at_headline": agg["hit_modes"],
    }
    return {"summary": summary, "rows": rows}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--records", required=True, help="enumerated_records.csv from the harness")
    p.add_argument("--thresholds", default="6,7,7.5,8", help="comma-separated reward gate sweep")
    p.add_argument("--headline", type=float, default=7.0, help="reward threshold for hit-modes")
    p.add_argument("--similarity", type=float, default=0.7)
    p.add_argument("--lower-is-better", action="store_true", help="docking-style (else sEH)")
    p.add_argument("--out-dir", default="experiments/lsd_hubs/dropoff")
    p.add_argument("--tag", default="run", help="artifact filename tag, e.g. seh_70140")
    a = p.parse_args(argv)

    thresholds = [float(x) for x in a.thresholds.split(",") if x.strip()]
    hubs = load_hub_children(a.records)
    result = build_funnel(hubs, thresholds, a.headline, a.similarity, not a.lower_is_better)

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # `_results.csv` suffix so the small committed table survives the repo's global *.csv ignore.
    with open(out / f"funnel_{a.tag}_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(result["rows"][0].keys()))
        w.writeheader()
        w.writerows(result["rows"])
    with open(out / f"funnel_{a.tag}_summary.json", "w") as fh:
        json.dump(result["summary"], fh, indent=2)

    s = result["summary"]
    print(f"hubs={s['n_hubs']}  raw_children={s['total_raw_children']}")
    for t, d in s["survival"].items():
        print(f"  reward gate {t:>4}: {d['count']:>5} ({d['pct']}%)")
    print(
        f"  structure-modes(all children): {s['total_structure_modes']} "
        f"(raw->struct {s['raw_to_structure_redundancy']}x)"
    )
    print(f"  hit-modes(>= {a.headline}): {s['total_hit_modes_at_headline']}")
    print(f"wrote funnel_{a.tag}_results.csv + funnel_{a.tag}_summary.json -> {out}")


if __name__ == "__main__":
    main()
