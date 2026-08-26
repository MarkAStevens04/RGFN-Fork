#!/usr/bin/env python
"""Does the generator prefer LOW-YIELD reactions? -- and does hub-batching inherit that preference?

WHY THIS EXISTS. Our headline cost metric counts REACTIONS (docs/LSD_FLOW_BENCHMARK_PLAN.md; the
fixed 100-reaction budget). A reaction count is yield-blind: it prices a 0.95-yield amide coupling
and a 0.70-yield carboxylic-acid reduction identically. SCENT's own library disagrees -- it ships a
per-template ``yield`` column -- so the question is empirical: does the trained policy spend its
reaction budget on cheap-to-count-but-expensive-to-run chemistry, and by how much does that inflate
the true bench cost behind our numbers?

THE NATURAL EXPERIMENT. The library encodes the same reactant pair (carboxylic acid + amine) under
two separately-curated families with different products and different yields:

    templates.txt line 39-40  ->  anchored ids 76-79    "Amide synthesis"       yield 0.75
    templates.txt line 69-70  ->  anchored ids 136-139  "direct acid -> amine"  yield 0.70

ONE TRAP IN THAT TABLE. ``templates_yields.csv`` lists the amide reaction TWICE: once at yield 0.95
with an isotope-labelled acid carbon (``[12C:3]``) and once at yield 0.75 with a plain ``[C:3]``.
``templates.txt`` -- the file the runs actually load -- contains no isotope labels at all, so the
operative amide yield is 0.75 and the 0.95 row is dead. Reading the 0.95 turns a 0.05 yield gap into
a fictitious 0.25 one, so ``check()`` asserts the joined value. It is also why the ``Family`` column
cannot be trusted here: the plain rows are appended at the end of the file with a BLANK family cell,
and forward-filling therefore attributes them to whatever block precedes them.

Their reactant SMARTS match IDENTICAL molecule sets (verified by substructure probe, and by
``--check`` here), and each family contributes exactly 4 anchored ids, so the two are available to
the policy as a matched 1:1 pair of actions from the same state. A yield-indifferent policy would
split them 50/50. Anything else is a revealed preference -- and because the products differ (amide
vs secondary amine), that preference is presumably driven by the REWARD, not by the yield.

That last step is what ``counterfactual`` tests, rather than assumes: take the acid+amine steps the
policy actually took with the 0.70 reduction, re-run the 0.95 amide template on the SAME two
reactants, and score both products with the same frozen oracle the run was trained against. If the
amine scores far above its amide counterfactual, the generator is not being careless -- it is paying
a yield premium to reach a pharmacophore the reward demands, which is a very different limitation
(and a different fix) from mere yield-blindness.

Yields are joined to templates by exact SMARTS, never by row index: ``templates.txt`` (112 rows) and
``templates_yields.csv`` (122 rows) are NOT aligned, and the anchored-id expansion in
``rgfn/gfns/reaction_gfn/api/reaction_data_factory.py`` emits one id per reactant slot, so neither
file's row numbers are the ids that appear in ``routes.json``.

Pure CPU. Run:
    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/yield_preference.py --check
    python experiments/lsd_hubs/campaign/yield_preference.py --counterfactual --n-pairs 4000
"""
import argparse
import collections
import csv
import json
import random
import re
from pathlib import Path

TEMPLATES = Path("external/scent/data/small/templates.txt")
YIELDS = Path("external/scent/data/small/templates_yields.csv")
SNAP = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820")
OUT = Path("experiments/lsd_hubs/campaign/results/yield_preference")

# The matched pair. Kept as line numbers (1-based, into templates.txt) so the ids are DERIVED by the
# same anchored expansion the runs used, rather than hard-coded and silently wrong if the library
# changes. Ids are asserted against these lines in ``load_library``.
AMIDE_LINES = (39, 40)  # "Amide synthesis"       yield 0.95   primary, secondary
DIRECT_LINES = (69, 70)  # "direct acid -> amine"  yield 0.70   primary, secondary

CELLS = [
    ("scent_seh", "seh", "seed43"),
    ("scent_seh", "seh", "seed44"),
    ("scent_drd2", "drd2", "drd2_seed43"),
    ("scent_drd2", "drd2", "drd2_seed44"),
]
TID = re.compile(r",\s*\d+,\s*(\d+)\)\s*$")


def norm(x):
    return re.sub(r"\s+", "", str(x))


def load_library():
    """(id -> (line, family, yield), templates list). Yields joined by exact SMARTS, not row index."""
    import pandas as pd

    from rgfn.gfns.reaction_gfn.api.data_structures import Reaction

    tx = [l.strip() for l in TEMPLATES.open() if l.strip()]
    cy = pd.read_csv(YIELDS)
    # Block-style column: blank means "same family as the row above". Correct WITHIN a block, but the
    # de-isotoped duplicates appended at the end of the file also have blank cells, so their filled
    # label is meaningless. Family is display-only here; ``yield`` is what the analysis uses.
    cy["Family"] = cy["Family"].ffill()
    ann = {norm(r): (str(f), float(y)) for r, f, y in zip(cy["Reaction"], cy["Family"], cy["yield"])}
    missing = [i + 1 for i, r in enumerate(tx) if norm(r) not in ann]
    if missing:
        raise SystemExit(f"templates.txt lines with no yield annotation: {missing}")

    idmap, idx, line_ids = {}, 0, collections.defaultdict(list)
    for i, r in enumerate(tx):
        fam, y = ann[norm(r)]
        for _ in range(len(Reaction(r, i).left_side_patterns)):
            idmap[idx] = (i + 1, fam, y)
            line_ids[i + 1].append(idx)
            idx += 1
    return idmap, line_ids, tx


def ids_for(line_ids, lines):
    return {i for ln in lines for i in line_ids[ln]}


def check(idmap, line_ids, tx):
    """Assert the two families are a matched, identically-applicable pair before relying on it."""
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    am, dr = ids_for(line_ids, AMIDE_LINES), ids_for(line_ids, DIRECT_LINES)
    print(f"amide  lines {AMIDE_LINES} -> ids {sorted(am)}  "
          f"family={idmap[min(am)][1]!r} yield={idmap[min(am)][2]}")
    print(f"direct lines {DIRECT_LINES} -> ids {sorted(dr)}  "
          f"family={idmap[min(dr)][1]!r} yield={idmap[min(dr)][2]}")
    assert len(am) == len(dr), "families are not a 1:1 action pair"
    # the isotope trap: templates.txt uses the plain [C:3] amide (0.75), not the [12C:3] row (0.95)
    ay, dy = idmap[min(am)][2], idmap[min(dr)][2]
    assert (ay, dy) == (0.75, 0.70), f"unexpected operative yields: amide={ay} direct={dy}"
    print(f"operative yields: amide={ay} direct={dy} (gap {ay-dy:.2f}); the 0.95 [12C:3] row is "
          f"NOT in templates.txt")

    # identical reactant applicability is what makes the 50/50 null legitimate
    probes = ["CC(=O)O", "OC(=O)c1ccccc1", "O=C(O)[C@H]1CCCN1", "CCN", "Nc1ccccc1", "CNC",
              "CC(=O)N", "CCO", "c1ccccc1", "O=C(O)CCN", "OC(=O)CN1CCOCC1", "NCc1ccncc1"]
    pairs = [(tx[AMIDE_LINES[k] - 1], tx[DIRECT_LINES[k] - 1]) for k in (0, 1)]
    dis = 0
    for a, d in pairs:
        for side in (0, 1):
            pa = Chem.MolFromSmarts(norm(a).split(">>")[0].split(".")[side])
            pd_ = Chem.MolFromSmarts(norm(d).split(">>")[0].split(".")[side])
            for s in probes:
                m = Chem.MolFromSmiles(s)
                if m.HasSubstructMatch(pa) != m.HasSubstructMatch(pd_):
                    dis += 1
    print(f"reactant-applicability disagreements over {len(probes)} probes x 4 patterns: {dis}")
    assert dis == 0, "families are NOT identically applicable -- the 50/50 null is invalid"
    print("OK: matched 1:1, identically applicable -> uniform baseline is 50.0%")


def usage(idmap, line_ids):
    """Generator-level: how the policy actually spent its steps, per cell."""
    am, dr = ids_for(line_ids, AMIDE_LINES), ids_for(line_ids, DIRECT_LINES)
    rows, pooled = [], collections.Counter()
    for gen, _target, cell in CELLS:
        d = json.loads((SNAP / cell / "routes.json").read_text())
        c, nsteps, ysum, rprod = collections.Counter(), 0, 0.0, []
        for rt in d.values():
            ys = []
            for s in rt.get("steps") or []:
                m = TID.search(s.get("reaction") or "")
                if not m:
                    continue
                t = int(m.group(1))
                c[t] += 1
                nsteps += 1
                ysum += idmap[t][2]
                ys.append(idmap[t][2])
            if ys:
                p = 1.0
                for y in ys:
                    p *= y
                rprod.append(p)
        pooled.update(c)
        a, r = sum(c[t] for t in am), sum(c[t] for t in dr)
        rows.append(dict(
            cell=f"{gen}/{cell}", n_routes=len(d), n_steps=nsteps,
            mean_step_yield=round(ysum / nsteps, 4),
            mean_route_yield=round(sum(rprod) / len(rprod), 4),
            median_route_yield=round(sorted(rprod)[len(rprod) // 2], 4),
            direct_steps=r, amide_steps=a,
            pct_steps_direct=round(100 * r / nsteps, 2),
            pct_of_acid_amine_direct=round(100 * r / (r + a), 2) if (r + a) else None,
        ))
    return rows, pooled


def revealed_preference(idmap, tx, n_steps=6000, seed=0):
    """The general test: is the CHOSEN template's yield below what was available at that state?

    The matched acid+amine pair only covers one transformation. This asks the question over every
    step: for the molecule actually in hand, collect every template with a reactant pattern that
    matches it -- the reaction types the policy could have fired -- and compare the yield it chose
    against the mean of that applicable set. This conditions on the state, so it is not confounded by
    some states simply offering better chemistry than others.

    Applicability here ignores whether a PARTNER fragment exists, which makes the applicable set a
    slight over-estimate; with a 400+ fragment library the binding constraint is the molecule in
    hand, so the direction of the bias is toward under-stating any preference, not inventing one.
    """
    import collections
    import random

    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    Y = [None] * len(tx)
    for _i, (line, _f, y) in idmap.items():
        Y[line - 1] = y
    pats = [[Chem.MolFromSmarts(q) for q in norm(t).split(">>")[0].split(".")] for t in tx]
    lib = sum(Y) / len(Y)
    print(f"library baseline: mean yield over {len(tx)} templates = {lib:.4f}   "
          f"distribution {sorted(collections.Counter(Y).items())}")

    out = []
    print(f"\n{'cell':<14}{'n':>7}{'chosen':>9}{'available':>11}{'delta':>9}{'below':>8}"
          f"{'route y':>9}{'route y @avail':>15}")
    for _gen, _target, cell in CELLS:
        d = json.loads((SNAP / cell / "routes.json").read_text())
        steps = []
        for rt in d.values():
            for st in rt.get("steps") or []:
                m = TID.search(st.get("reaction") or "")
                if m:
                    steps.append((int(m.group(1)), st.get("input")))
        random.Random(seed).shuffle(steps)
        ch, av, below = [], [], 0
        for t, inp in steps[:n_steps]:
            m = Chem.MolFromSmiles(inp) if inp else None
            if m is None:
                continue
            appl = [Y[i] for i, ps in enumerate(pats)
                    if any(q is not None and m.HasSubstructMatch(q) for q in ps)]
            if not appl:
                continue
            c, v = idmap[t][2], sum(appl) / len(appl)
            ch.append(c)
            av.append(v)
            below += c < v
        n = len(ch)
        mc, mv = sum(ch) / n, sum(av) / n
        # compound the per-step gap over the cell's mean route length
        nsteps = sum(len(rt.get("steps") or []) for rt in d.values())
        depth = nsteps / max(1, sum(1 for rt in d.values() if rt.get("steps")))
        out.append(dict(cell=cell, n_steps_sampled=n, mean_chosen_yield=round(mc, 4),
                        mean_available_yield=round(mv, 4), delta=round(mc - mv, 4),
                        pct_chose_below_available=round(100 * below / n, 2),
                        mean_route_len=round(depth, 2),
                        route_yield_chosen=round(mc ** depth, 4),
                        route_yield_at_available=round(mv ** depth, 4),
                        pct_route_yield_forgone=round(100 * (1 - (mc / mv) ** depth), 2)))
        r = out[-1]
        print(f"{cell:<14}{n:>7}{mc:>9.4f}{mv:>11.4f}{mc-mv:>9.4f}"
              f"{100*below/n:>7.1f}%{r['route_yield_chosen']:>9.3f}"
              f"{r['route_yield_at_available']:>15.3f}")
    print("  'route y' compounds the per-step yield over the cell's mean route length; the last "
          "column is the same molecule count\n  had every step been chosen at the mean of what was "
          "applicable. The gap is what a reaction count cannot see.")
    return out


def counterfactual(idmap, line_ids, tx, cell, target, n_pairs, seed=0):
    """Re-run the 0.95 amide template on the SAME reactants and score both products.

    Answers WHY the policy chose the low-yield option: reward, or indifference. Unique
    (input, added-reactant, template) triples so the statistic is about the chemistry rather than
    about how often the policy happened to repeat a pair.
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    def rxn(line):
        return AllChem.ReactionFromSmarts(norm(tx[line - 1]))

    # direct id -> the amide template of MATCHING amine substitution (primary vs secondary)
    amide_for = {}
    for k in (0, 1):
        for i in line_ids[DIRECT_LINES[k]]:
            amide_for[i] = rxn(AMIDE_LINES[k])
    direct_rxn = {i: rxn(ln) for ln in DIRECT_LINES for i in line_ids[ln]}

    def apply2(r, a, b):
        out = set()
        for pair in ((a, b), (b, a)):
            try:
                for ps in r.RunReactants(pair):
                    for p in ps:
                        try:
                            Chem.SanitizeMol(p)
                            out.add(Chem.MolToSmiles(p))
                        except Exception:
                            pass
            except Exception:
                pass
        return out

    d = json.loads((SNAP / cell / "routes.json").read_text())
    triples = {}
    for rt in d.values():
        for s in rt.get("steps") or []:
            m = TID.search(s.get("reaction") or "")
            if not m:
                continue
            t = int(m.group(1))
            if t not in direct_rxn:
                continue
            rs = s.get("reactants") or []
            if len(rs) != 1:
                continue
            triples.setdefault((s["input"], rs[0], t), s["product"])
    print(f"  {cell}: {len(triples):,} unique (input, reactant, template) direct-acid triples")
    keys = sorted(triples)
    random.Random(seed).shuffle(keys)
    keys = keys[:n_pairs]

    recorded, cf, reproduced, no_amide = [], [], 0, 0
    for inp, rct, t in keys:
        mi, mr = Chem.MolFromSmiles(inp), Chem.MolFromSmiles(rct)
        if mi is None or mr is None:
            continue
        got = apply2(direct_rxn[t], mi, mr)
        prod = Chem.MolToSmiles(Chem.MolFromSmiles(triples[(inp, rct, t)]))
        if prod not in got:
            continue  # cannot reproduce -> excluded rather than guessed
        reproduced += 1
        alt = apply2(amide_for[t], mi, mr)
        if not alt:
            no_amide += 1
            continue
        recorded.append(prod)
        cf.append(sorted(alt)[0])
    print(f"  reproduced recorded product: {reproduced}/{len(keys)}   "
          f"amide counterfactual unavailable: {no_amide}")

    score = make_scorer(target)
    sr, sc = score(recorded), score(cf)
    return recorded, cf, sr, sc


def make_scorer(target):
    if target == "drd2":
        import pickle
        import numpy as np
        from rdkit import Chem
        from rdkit.Chem import AllChem

        with open("external/scent/oracle/drd2_current.pkl", "rb") as fh:
            model = pickle.load(fh)  # nosec - local TDC oracle

        def fp(m):
            f = AllChem.GetMorganFingerprint(m, 3, useCounts=True, useFeatures=True)
            v = np.zeros((1, 2048), np.int32)
            for i, c in f.GetNonzeroElements().items():
                v[0, i % 2048] += int(c)
            return v

        def score(smis):
            ms = [Chem.MolFromSmiles(s) for s in smis]
            X = np.concatenate([fp(m) for m in ms], axis=0)
            return model.predict_proba(X)[:, 1].tolist()

        return score

    if target == "seh":
        from rgfn.gfns.reaction_gfn.proxies.seh_proxy import SEHProxyWrapper

        w = SEHProxyWrapper(batch_size=128)
        return lambda smis: w.compute_scores(smis)

    raise SystemExit(f"no scorer for target {target!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="validate the matched-pair assumption")
    ap.add_argument("--counterfactual", action="store_true")
    ap.add_argument("--n-pairs", type=int, default=4000)
    ap.add_argument("--cells", default="all", help="comma list of cell dirs, or 'all'")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    idmap, line_ids, tx = load_library()
    print(f"library: {len(tx)} templates -> {len(idmap)} anchored ids\n")
    if a.check:
        check(idmap, line_ids, tx)
        print()

    rp = revealed_preference(idmap, tx)
    with (OUT / "revealed_preference.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rp[0]))
        w.writeheader()
        w.writerows(rp)
    print(f"wrote {OUT}/revealed_preference.csv\n")

    rows, pooled = usage(idmap, line_ids)
    hdr = ["cell", "n_routes", "n_steps", "mean_step_yield", "mean_route_yield",
           "median_route_yield", "direct_steps", "amide_steps", "pct_steps_direct",
           "pct_of_acid_amine_direct"]
    print(f"{'cell':<24}{'steps':>10}{'mean y':>8}{'route y':>9}"
          f"{'%direct':>9}{'%amide':>8}{'direct|acid+amine':>19}")
    for r in rows:
        print(f"{r['cell']:<24}{r['n_steps']:>10,}{r['mean_step_yield']:>8.3f}"
              f"{r['mean_route_yield']:>9.3f}{r['pct_steps_direct']:>8.1f}%"
              f"{100*r['amide_steps']/r['n_steps']:>7.1f}%{r['pct_of_acid_amine_direct']:>18.1f}%")
    with (OUT / "usage_by_cell.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}/usage_by_cell.csv")

    if a.counterfactual:
        cells = CELLS if a.cells == "all" else [c for c in CELLS if c[2] in a.cells.split(",")]
        crows = []
        for _gen, target, cell in cells:
            print(f"\n=== counterfactual: {cell} (target {target}) ===")
            rec, cf, sr, sc = counterfactual(idmap, line_ids, tx, cell, target, a.n_pairs)
            n = len(sr)
            mr, mc = sum(sr) / n, sum(sc) / n
            win = sum(1 for x, y in zip(sr, sc) if x > y) / n
            print(f"  n={n}  recorded AMINE (y=0.70) mean={mr:.4f}   "
                  f"counterfactual AMIDE (y=0.75) mean={mc:.4f}")
            print(f"  amine scores higher in {100*win:.1f}% of matched pairs")
            crows.append(dict(cell=cell, target=target, n_pairs=n,
                              mean_recorded_amine=round(mr, 5),
                              mean_counterfactual_amide=round(mc, 5),
                              delta=round(mr - mc, 5),
                              pct_amine_higher=round(100 * win, 2)))
            with (OUT / f"counterfactual_{cell}.csv").open("w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["recorded_amine", "score_amine", "counterfactual_amide", "score_amide"])
                for i in range(n):
                    w.writerow([rec[i], round(sr[i], 6), cf[i], round(sc[i], 6)])
        with (OUT / "counterfactual_summary.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(crows[0]))
            w.writeheader()
            w.writerows(crows)
        print(f"\nwrote {OUT}/counterfactual_summary.csv")


if __name__ == "__main__":
    main()
