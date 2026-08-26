#!/usr/bin/env python
"""Does HUB-BATCHING inherit -- or amplify -- the generator's low-yield reaction preference?

Companion to ``yield_preference.py``, which establishes the generator-level bias (the policy picks
below the mean yield of the templates applicable at its state, in 77-87% of steps). This script asks
the selection-level question: at the SAME fixed 100-reaction budget, do hub-batching and
best-candidate deliver libraries with the same expected yield?

There is a specific reason to expect they do NOT, and it is structural rather than incidental:

  1. A hub earns its rank by FANOUT -- many enumerable children. Fanout comes from a reactive handle
     that a large slice of the fragment library can attach to. A carboxylic acid against a big amine
     library is exactly such a handle, so hub ranking is biased toward acid hubs, and therefore
     toward the acid+amine templates -- the lowest-yield rung the library offers (0.70 direct
     reduction, 0.75 amide). Best-candidate ranks by reward alone and inherits no such pressure.
  2. Hub-batching CONCENTRATES risk. Every molecule in a batch shares the hub's route prefix, so one
     failed hub step loses the whole batch. Best-candidate's routes fail independently. Two libraries
     with the same mean route yield are therefore NOT equally safe, and a reaction count cannot see
     the difference.

So this reports three things per arm: the mean route yield (does the chemistry differ), the expected
number of molecules actually delivered (sum of route survival probabilities), and the batch-level
risk concentration (P(a batch yields nothing) and the share of the library exposed to a single
point of failure).

Yields are per-template from SCENT's own library, joined by exact SMARTS -- see ``yield_preference``
for why row indices cannot be used. Route = hub prefix (``routes.json``) + final step
(``enum_children.json`` ``reaction``) for hub-batching; the whole ``routes.json`` entry for
best-candidate. Pure CPU.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/yield_by_strategy.py \
        --adir <scratch>/adir_seed43 --target seh --gate 7.0 \
        --snapshot .../scent_seh_5k/seed43/additional_fragments/fragments_4000.json
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from reaction_names import named_reaction  # noqa: E402
from run_campaign import _load_candidates, _load_enumerated_hubs, build_strategy  # noqa: E402
from yield_preference import DIRECT_LINES, ids_for, load_library  # noqa: E402

from glue.samplers.lsdflow.child_select import make_child_policy  # noqa: E402
from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: E402
    load_cost_table_from_snapshot,
)

OUT = Path("experiments/lsd_hubs/campaign/results/yield_preference")
TID = re.compile(r",\s*\d+,\s*(\d+)\)\s*$")


def step_ids(steps):
    out = []
    for s in steps or []:
        m = TID.search(s.get("reaction") or "")
        if m:
            out.append(int(m.group(1)))
    return out


def prod(xs):
    p = 1.0
    for x in xs:
        p *= x
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adir", required=True, help="dir with records.csv, compositions.json, "
                                                 "enum_children.json, routes.json")
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--gate", type=float, required=True)
    ap.add_argument("--budget-reactions", type=int, default=100)
    ap.add_argument("--similarity", type=float, default=0.5)
    ap.add_argument("--child-policy", default="free_frag", choices=["reward", "free_frag"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    adir = Path(a.adir)
    idmap, line_ids, _tx = load_library()
    Y = {i: idmap[i][2] for i in idmap}
    DIRECT = ids_for(line_ids, DIRECT_LINES)

    routes = json.loads((adir / "routes.json").read_text())
    cands, comps = _load_candidates(adir, True)
    enum = _load_enumerated_hubs(adir / "enum_children.json", comps)
    cost_table = load_cost_table_from_snapshot(json.loads(Path(a.snapshot).read_text()))

    # (hub_key, child_smiles) -> final-step template ids, straight from the enumeration
    raw = json.loads((adir / "enum_children.json").read_text())
    final = {}
    for h in raw["hubs"]:
        for c in h["children"]:
            final[(h["hub_key"], c["smiles"])] = step_ids(c.get("reaction"))

    common = dict(target=a.target, reward_threshold=a.gate, higher_is_better=True,
                  similarity=a.similarity)
    arms = [
        ("best_candidate", cands, None),
        ("hub_batching", enum, make_child_policy(a.child_policy)),
    ]

    rows, detail = [], []
    for label, pool, cp in arms:
        strat = build_strategy(label, pool, cost_table, comps, child_policy=cp, **common)
        res = strat.run(budget=("reactions", a.budget_reactions))
        acc = res.accepted
        # per-molecule full route -> per-step yields
        per, unpriced = [], 0
        for p in acc:
            if label == "best_candidate":
                ids = step_ids((routes.get(p.smiles) or {}).get("steps"))
            else:
                hub = p.source_hub or ""
                ids = step_ids((routes.get(hub) or {}).get("steps")) + \
                    final.get((hub, p.smiles), [])
            if not ids:
                unpriced += 1
                continue
            ys = [Y[i] for i in ids]
            per.append(dict(smiles=p.smiles, hub=p.source_hub or "", n_steps=len(ids),
                            route_yield=prod(ys), mean_step_yield=sum(ys) / len(ys),
                            n_direct=sum(1 for i in ids if i in DIRECT), ids=ids))
        n = len(per)
        if not n:
            print(f"  {label}: nothing priced (accepted {len(acc)})")
            continue
        allsteps = [i for r in per for i in r["ids"]]
        ry = [r["route_yield"] for r in per]

        # Batch-level risk. Molecules off one hub share its route PREFIX, so their failures are
        # CORRELATED: one bad hub step loses the whole batch. Note the expectation is identical
        # either way -- the prefix probability factors straight out of the sum over children -- so
        # comparing means here would show nothing. The difference is entirely in the SPREAD, plus
        # the mass exposed to a single point of failure. Both arms are scored with the same batch
        # model (a molecule with no shared hub is simply a batch of one), so the contrast is the
        # structure, not the accounting.
        batches = {}
        for r in per:
            batches.setdefault(r["hub"] or f"__solo__{r['smiles']}", []).append(r)
        exp_indep = sum(ry)
        var_indep = sum(v * (1 - v) for v in ry)
        var_batch, at_risk, worst_loss, biggest = 0.0, 0.0, 0.0, 0
        for hub, members in batches.items():
            if hub.startswith("__solo__"):
                q, fs = 1.0, [m["route_yield"] for m in members]
            else:
                q = prod([Y[i] for i in step_ids((routes.get(hub) or {}).get("steps"))])
                fs = [prod([Y[i] for i in final.get((hub, m["smiles"]), [])]) for m in members]
            S = sum(fs)
            # N_b = X_b * sum_j Y_j  with X_b ~ Bern(q) independent of the Y_j
            var_batch += q * (sum(f * (1 - f) for f in fs) + S * S) - (q * S) ** 2
            if len(members) > 1:
                at_risk += (1 - q) * S          # molecules lost specifically to a shared-prefix failure
                worst_loss = max(worst_loss, 1 - q)
                biggest = max(biggest, len(members))
        rows.append(dict(
            tag=a.tag, target=a.target, arm=label,
            child_policy=a.child_policy if label == "hub_batching" else "",
            modes=len(acc), priced=n, unpriced=unpriced,
            used_rxns=res.total_reactions, stop_reason=res.stop_reason,
            mean_route_yield=round(sum(ry) / n, 4),
            median_route_yield=round(sorted(ry)[n // 2], 4),
            mean_step_yield=round(sum(Y[i] for i in allsteps) / len(allsteps), 4),
            mean_route_len=round(len(allsteps) / n, 2),
            pct_steps_direct=round(100 * sum(1 for i in allsteps if i in DIRECT) / len(allsteps), 2),
            expected_molecules=round(exp_indep, 2),
            sd_if_independent=round(var_indep ** 0.5, 2),
            sd_with_shared_prefixes=round(var_batch ** 0.5, 2),
            n_shared_batches=len([h for h, v in batches.items()
                                  if not h.startswith("__solo__") and len(v) > 1]),
            pct_in_multi_batches=round(100 * sum(len(v) for h, v in batches.items()
                                                 if not h.startswith("__solo__")
                                                 and len(v) > 1) / n, 1),
            largest_batch=biggest,
            expected_lost_to_shared_prefix=round(at_risk, 2),
            worst_batch_loss_prob=round(worst_loss, 3),
        ))
        for r in per:
            detail.append(dict(tag=a.tag, arm=label, smiles=r["smiles"], hub=r["hub"],
                               n_steps=r["n_steps"], route_yield=round(r["route_yield"], 5),
                               n_direct_acid_steps=r["n_direct"],
                               reactions="|".join(named_reaction(f"R'(x, 0, {i})") for i in r["ids"])))

    OUT.mkdir(parents=True, exist_ok=True)
    sf = OUT / f"strategy_yield_{a.tag}.csv"
    with sf.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    df = OUT / f"strategy_yield_detail_{a.tag}.csv"
    with df.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(detail[0]))
        w.writeheader()
        w.writerows(detail)

    print(f"\n=== {a.tag} (target {a.target}, gate {a.gate}, R={a.budget_reactions}) ===")
    for r in rows:
        print(f"  {r['arm']:<16} modes={r['modes']:>4} used_rxns={r['used_rxns']:>4} "
              f"({r['stop_reason']})  route_yield={r['mean_route_yield']:.3f}  "
              f"step_yield={r['mean_step_yield']:.3f}  len={r['mean_route_len']:.2f}  "
              f"direct={r['pct_steps_direct']:.1f}%")
        print(f"  {'':<16} E[molecules]={r['expected_molecules']:.1f}  "
              f"sd: indep={r['sd_if_independent']:.1f} vs shared-prefix="
              f"{r['sd_with_shared_prefixes']:.1f}  batches={r['n_shared_batches']} "
              f"(largest {r['largest_batch']}, {r['pct_in_multi_batches']:.0f}% of library)")
        print(f"  {'':<16} E[lost to a shared-prefix failure]="
              f"{r['expected_lost_to_shared_prefix']:.1f} molecules; worst batch fails w.p. "
              f"{r['worst_batch_loss_prob']:.2f}")
    print(f"wrote {sf}\nwrote {df}")


if __name__ == "__main__":
    main()
