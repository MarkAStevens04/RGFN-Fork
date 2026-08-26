#!/usr/bin/env python
"""Template id -> named reaction, for the route dataset's schemes and protocols.

WHY A TABLE AND NOT A HEURISTIC ON THE FLY. The reaction templates are the ground truth: the
transformation is whatever the template encodes. A name is a READING AID and nothing else, so it must
never become the identifier -- ``template_id`` is what joins, and a wrong name should be visibly
wrong rather than quietly authoritative.

Two facts make this tractable. The 168 distinct templates across the four published cells are heavily
REDUNDANT -- most transformations appear 2-8 times with the reactant SMARTS in a different order
(136 and 137 are the same acid + primary amine -> amine template, anchored on either reactant) -- and they are heavily CONCENTRATED: the top
20 account for ~90% of all steps. So classifying by the transformation's signature, rather than
enumerating ids by hand, covers the corpus with a few dozen rules.

``emit_review_table()`` prints every id with its count, coverage and assigned name so a chemist can
audit the whole mapping in one pass. Anything the rules do not recognise is named ``?`` rather than
guessed -- an unnamed step is honest; a mislabelled one is not.
"""
import re
from typing import Optional, Tuple

# Ordered (signature, name) rules. FIRST match wins, so specific patterns precede general ones --
# e.g. the benzoxazole/benzimidazole cyclisations must be tested before plain amide coupling, since
# they also contain an acid SMARTS.
RULES = [
    # --- ring-forming condensations (test before the acyl/amine rules they contain) --------------
    ("[o:5]1[c:4][c:3][n:2][c:1]",              "Benzoxazole formation"),
    ("[n:5]1[c:4][c:3][nH:2][c:1]",             "Benzimidazole formation"),
    ("[n:5]1(-[#6:7])[c:4][c:3][n:2][c:1]",     "Benzimidazole formation (N-substituted)"),
    ("[n:5](-[#6:7])1[c:4][c:3][n:2][c:1]",     "Benzimidazole formation (N-substituted)"),
    ("[s:5]1[c:4][c:3][n:2][c:1]",              "Benzothiazole formation"),
    ("-[c:2]1[n:3][n](-[C:5]",                  "Tetrazole formation (nitrile + alkyl halide)"),
    ("-[c:2]1[n:3][n+0][n-0][n](-[C:5]",        "Tetrazole formation (nitrile + alkyl halide)"),
    ("[c:7]1nnn[n:6]",                          "Tetrazole formation (isothiocyanate)"),
    ("-[c:2]1[c:3][n](-[C:5]",                  "Triazole formation (alkyne + alkyl halide)"),
    ("[c:3]1[c:2](-[#6:1])[n](-[C:5]",          "Triazole formation (alkyne + alkyl halide)"),
    ("[c:6]([#6:7])1[n:3][c:4]",                "van Leusen oxazole synthesis"),
    ("[C:11][C:12][Nh",                         "Pictet-Spengler cyclisation"),
    ("[Nh&!R:3]-[C:4]-[C:5]-[c:6]1",            "Pictet-Spengler cyclisation"),
    ("[Nh2:3]-[C:4]-[C:5]-[c:6]1",              "Pictet-Spengler cyclisation"),
    # --- cross-couplings -------------------------------------------------------------------------
    ("[c:1]-[B]([Oh])([Oh])",                   "Suzuki coupling"),
    ("[c:1](B1OC(C)(C)C(C)(C)O1)",              "Suzuki coupling (pinacol boronate)"),
    ("[C:5]#[Ch:4]",                            "Sonogashira coupling"),
    ("[C:5]#[C:4]-[*:3]",                       "Sonogashira coupling"),
    # --- C-N bond formation ----------------------------------------------------------------------
    ("[F]-[c:4]1[n:5]",                         "SNAr (halo-heteroarene + amine)"),
    ("[Cl]-[c:4]1[n:5]",                        "SNAr (halo-heteroarene + amine)"),
    ("[F]-[c:2]1[c:3]",                         "SNAr (activated aryl fluoride + amine)"),
    ("[c:2]-[Br]",                              "Buchwald-Hartwig amination"),
    ("[c:2]-[I]",                               "Buchwald-Hartwig amination"),
    ("[c:1][n:2]([c:3])-[Ch2:4]",               "N-alkylation (azole N-H + alkyl halide)"),
    # --- carbonyl chemistry ----------------------------------------------------------------------
    ("[Ch:2]([#6:3])=[O]",                      "Reductive amination (aldehyde)"),
    ("[#6:3][C:2]([#6:4])=[O]",                 "Reductive amination (ketone)"),
    ("[C:3](=[O:4])-[O:2]-",                    "Esterification"),
    # The library encodes TWO distinct acid + amine reactions, and they are not interchangeable.
    # Family "Amide synthesis" maps the carbonyl through ([C:3](=[O:4])-...) and yields the amide.
    # Family "direct acid -> amine" leaves the carbonyl oxygen UNMAPPED on purpose -- the oxygen is
    # meant to leave, the product is a secondary amine, and the library prices it at yield 0.70
    # against 0.95 for the amide. Test the unmapped-carbonyl signature FIRST: it is unique to those
    # two templates (verified against external/scent/data/small/templates.txt lines 69-70), and
    # naming them "Amide coupling" tells a chemist to run an EDC/HATU coupling for a transformation
    # that actually needs a reduction.
    ("[C:1](=[O])[Oh]",                         "Direct acid -> amine (reductive)"),
    ("[C:3](=[O:4])-[Oh]",                      "Amide coupling"),
    ("[#6:6]-[C&!R:1](=[O])-[Oh]",              "Amide coupling"),
    # --- ureas, sulfonamides, conjugate additions -------------------------------------------------
    ("[N:2]=[C:3]=[O:4]",                       "Urea formation (isocyanate + amine)"),
    ("-[C](=O)-",                               "Urea formation (amine + amine)"),
    ("F-[S:4]",                                 "Sulfonamide formation (sulfonyl fluoride)"),
    ("[#6:2]=[#6:3]-[#6:4]=O",                  "aza-Michael addition (enone)"),
    ("[#6:2]=[#6:3]-[S:4]",                     "aza-Michael addition (vinyl sulfone)"),
    ("[Ch2:1]-[O:3]-[C:2]",                     "Williamson ether synthesis"),
]

# Single-reactant halogen/functional-group swaps: `[c:1]-[Br] >> [c:1]-[X]`. Separated because they
# render differently -- one input, nothing above the arrow.
FGI = re.compile(r"^\[c:1\]-\[(Br|Cl|I)\]\s*>>\s*\[c:1\]-\[?([A-Za-z#]+)")


def parse_template(t: str) -> Tuple[str, str]:
    """(template_id, bare SMARTS transform) from an ``R'(A.B >> P, flag, id)`` string."""
    if not t:
        return "", ""
    m = re.search(r",\s*\d+,\s*(\d+)\)\s*$", t)
    tid = m.group(1) if m else ""
    body = re.sub(r"^R'\(", "", t)
    body = re.sub(r",\s*\d+,\s*\d+\)\s*$", "", body)
    return tid, body


def named_reaction(t: str) -> str:
    """A readable name, or ``"?"`` when no rule matches. Never guesses."""
    _, body = parse_template(t)
    if not body:
        return "?"
    fgi = FGI.match(body.strip())
    if fgi:
        return f"Functional-group interconversion ({fgi.group(1)} -> {fgi.group(2)})"
    product = body.split(">>")[-1] if ">>" in body else body
    for sig, name in RULES:
        if sig in product or sig in body:
            return name
    return "?"


def is_single_reactant(t: str) -> bool:
    """True when the template consumes ONE molecule -- drawn ``Y -> Z``, nothing over the arrow."""
    _, body = parse_template(t)
    lhs = body.split(">>")[0] if ">>" in body else body
    # reactant SMARTS are joined by '.' at the top level; a lone reactant has none outside brackets
    depth = 0
    for ch in lhs:
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "." and depth == 0:
            return False
    return True


def emit_review_table(routes_paths, out_csv=None) -> None:
    """Print every template with count, coverage and assigned name, for a one-pass chemist audit."""
    import collections
    import csv as _csv
    import json

    cnt, ex = collections.Counter(), {}
    for p in routes_paths:
        for rt in json.loads(open(p).read()).values():
            for s in rt.get("steps") or []:
                t = s.get("reaction") or ""
                tid, _ = parse_template(t)
                cnt[tid] += 1
                ex.setdefault(tid, t)
    tot = sum(cnt.values()) or 1
    rows, run = [], 0
    for tid, n in cnt.most_common():
        run += n
        nm = named_reaction(ex[tid])
        rows.append(dict(template_id=tid, steps=n, cum_coverage=round(run / tot, 4),
                         named_reaction=nm, single_reactant=is_single_reactant(ex[tid]),
                         smarts=parse_template(ex[tid])[1]))
    unnamed = sum(r["steps"] for r in rows if r["named_reaction"] == "?")
    print(f"templates: {len(rows)}   steps: {tot:,}   UNNAMED: {unnamed:,} ({unnamed/tot:.2%})")
    print(f"{'id':>5}{'steps':>9}{'cum':>8}  {'1-react':>7}  named reaction")
    for r in rows[:34]:
        print(f"{r['template_id']:>5}{r['steps']:>9,}{r['cum_coverage']:>8.1%}"
              f"  {str(r['single_reactant']):>7}  {r['named_reaction']}")
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"\nfull table -> {out_csv}")


if __name__ == "__main__":
    import sys
    SNAP = "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820"
    paths = sys.argv[1:] or [f"{SNAP}/{c}/routes.json"
                             for c in ("seh_seed43", "seh_seed44", "drd2_seed43", "drd2_seed44")]
    emit_review_table(paths, "experiments/lsd_hubs/campaign/results/template_map.csv")
