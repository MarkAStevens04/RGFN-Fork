#!/usr/bin/env python
"""Propagate MEASURED docking noise into reactions/mode: does oracle scatter move the headline?

The replicate experiment (`submit_replicates.sh`, 400 molecules x 5 independent processes) found
QuickVina2-GPU is far noisier than an opportunistic estimate suggested: per-molecule SD median
**0.502** kcal/mol, and at the -8.0 ClpP bar **43%** of molecules (78% of near-bar ones) change
qualification between draws. That is alarming for mode MEMBERSHIP, and the obvious worry is that the
2.4-2.9x edges are an artifact of which molecules happened to clear the bar.

They should not be, for a reason worth testing rather than asserting: reactions/mode is a COUNT over a
deep pool (170k-450k enumerated children per cell), not a property of specific molecules. When noise
knocks one child out, another takes its place, and both strategies draw from the same oracle. This
script measures how much slack that argument actually has.

Method: re-run the campaign N times, each time adding independent N(0, sigma) to EVERY child's and
candidate's docking score, with sigma taken from the replicates. Everything else -- gate, similarity,
child policy, budget, cost model -- is held at the cell's production values, and the machinery is
`run_campaign`'s own (`_load_candidates`/`_load_enumerated_hubs`/`build_strategy`/`run_timed`), so a
draw with sigma=0 reproduces the committed number exactly.

Reports the spread of reactions/mode and of the edge across draws: an error bar attributable to oracle
noise alone, separate from seed variance.

    python noise_bootstrap.py scent clpp --sigma 0.502 --draws 20
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments" / "lsd_hubs" / "matrix16"),
          str(REPO / "experiments" / "lsd_hubs" / "campaign")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_campaign as RC  # noqa: E402
from manifest import get_cell  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("generator")
    ap.add_argument("target")
    ap.add_argument("--sigma", type=float, required=True, help="per-molecule docking SD, kcal/mol")
    ap.add_argument("--draws", type=int, default=20)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--budget-modes", type=int, default=300)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    cell = get_cell(a.generator, a.target)
    hib = cell.target.higher_is_better
    gate = cell.target.mode_reward_threshold

    # Per-generator policy, mirroring run_cell_campaign.sh (SCENT's hero policy is free_frag + K=20).
    policy = "free_frag" if a.generator == "scent" else "reward"
    kk = 20 if a.generator == "scent" else 0
    snap = {}
    if a.generator == "scent":
        rr = Path(cell.checkpoint).resolve().parents[2]
        s = sorted(rr.glob("additional_fragments/fragments_*.json"),
                   key=lambda p: int(p.stem.split("_")[-1]))
        if s:
            snap = json.load(open(s[-1]))

    cands, comps = RC._load_candidates(cell.sample_dir, hib)
    enum_hubs = RC._load_enumerated_hubs(cell.enum_dir / "enum_children.json", comps)
    cost_table = RC.load_cost_table_from_snapshot(snap)
    child_policy = RC.make_child_policy(policy)
    print(f"[boot] {cell.tag}: {len(cands)} candidates, {len(enum_hubs)} hubs, "
          f"{sum(len(h.children) for h in enum_hubs)} children; gate {gate} hib={hib} "
          f"policy={policy} K={kk} sigma={a.sigma} draws={a.draws}", flush=True)

    # Snapshot the clean scores once; each draw perturbs from these, never from a perturbed value.
    clean_children = [[c.reward for c in h.children] for h in enum_hubs]
    clean_cands = [c.reward for c in cands]

    rng = random.Random(a.seed)
    rows = []
    for d in range(a.draws + 1):          # draw 0 is sigma=0: must reproduce the committed number
        sd = 0.0 if d == 0 else a.sigma
        for h, base in zip(enum_hubs, clean_children):
            for c, b in zip(h.children, base):
                # EnumChild is frozen; mutate via object.__setattr__ to avoid rebuilding the pool.
                object.__setattr__(c, "reward", b + (rng.gauss(0, sd) if sd else 0.0))
        for c, b in zip(cands, clean_cands):
            object.__setattr__(c, "reward", b + (rng.gauss(0, sd) if sd else 0.0))

        prebuilt = None
        if kk > 0:
            ranked = RC.rank_fragments(enum_hubs, cost_table, gate, method="build_score",
                                       higher_is_better=hib)
            prebuilt = {f for f, _ in ranked[:kk]}
        common = dict(target=cell.tag, reward_threshold=gate, similarity=a.similarity,
                      higher_is_better=hib)
        budget = ("modes", a.budget_modes)
        bc, _ = RC.run_timed(RC.build_strategy("best_candidate", cands, cost_table, comps, **common), budget)
        hb, _ = RC.run_timed(RC.build_strategy("hub_batching", enum_hubs, cost_table, comps,
                                               child_policy=child_policy,
                                               prebuilt_fragments=prebuilt, **common), budget)
        rb = RC._readouts(bc, a.budget_reactions, a.budget_modes)
        rh = RC._readouts(hb, a.budget_reactions, a.budget_modes)
        edge = (rb["reactions_per_mode"] / rh["reactions_per_mode"]
                if rb["reactions_per_mode"] and rh["reactions_per_mode"] else None)
        rows.append(dict(draw=d, sigma=sd, hub_rpm=rh["reactions_per_mode"],
                         best_rpm=rb["reactions_per_mode"], edge=edge,
                         hub_modes=rh["total_modes"], best_modes=rb["total_modes"],
                         hub_hubs=rh["distinct_hubs_used"]))
        tag = "CLEAN" if d == 0 else f"draw {d}"
        print(f"[boot] {tag:>8}: hub {rh['reactions_per_mode']:.4f} ({rh['distinct_hubs_used']} hubs) "
              f"| best {rb['reactions_per_mode']:.4f} | edge {edge:.3f}x "
              f"| modes {rh['total_modes']}/{rb['total_modes']}", flush=True)

    pert = [r for r in rows if r["sigma"] > 0]
    clean = rows[0]
    hs = [r["hub_rpm"] for r in pert]
    es = [r["edge"] for r in pert if r["edge"]]
    print("\n[boot] ==== oracle-noise error bar ====")
    print(f"[boot] clean (sigma=0):  hub {clean['hub_rpm']:.4f}  edge {clean['edge']:.3f}x")
    print(f"[boot] hub reactions/mode: mean {st.mean(hs):.4f} sd {st.stdev(hs):.4f} "
          f"min {min(hs):.4f} max {max(hs):.4f}")
    print(f"[boot] edge:               mean {st.mean(es):.3f}x sd {st.stdev(es):.3f} "
          f"min {min(es):.3f}x max {max(es):.3f}x")
    print(f"[boot] every draw favours hub-batching: {all(e > 1.0 for e in es)}")
    out = Path(a.out) if a.out else HERE / f"bootstrap_{cell.tag}.json"
    json.dump(dict(cell=cell.tag, sigma=a.sigma, draws=a.draws, gate=gate, rows=rows),
              open(out, "w"), indent=2)
    print(f"[boot] wrote {out}")


if __name__ == "__main__":
    main()
