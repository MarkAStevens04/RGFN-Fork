#!/usr/bin/env python
"""The two reactions the library offers for the SAME acid + amine, drawn side by side.

SCENT's library encodes carboxylic acid + amine TWICE, as two separately-curated families that
consume identical reactants and diverge only in the product:

  ids 76-79   "Amide synthesis"        [C:3](=[O:4])-[Oh].[Nh2:5]-R >> [C:3](=[O:4])-[Nh:5]-R
              -> the AMIDE, yield 0.75
  ids 136-139 "direct acid -> amine"   [C:1](=[O])[Oh].[Nh2:2]-R    >> [C:1]-[Nh:2]-R
              -> a secondary AMINE, yield 0.70 (a direct reductive amination of the acid)

The second reads like a broken copy of the first -- the carbonyl oxygen carries no atom map and
disappears -- and was initially reported as exactly that. It is not a defect. The unmapped ``=[O]`` is
how you write "this oxygen leaves"; the family has its own name, its own comments and its own yield,
and the ``O-2, H balanced`` mass balance is the CORRECT signature of a reduction whose reductant, like
every reagent, is simply not written into the template. Routes using it are valid synthesis
instructions.

The distinction still matters to whoever runs the chemistry -- one is a routine EDC/HATU coupling, the
other needs a reduction -- which is why ``reaction_names.py`` names them apart and why this figure
exists. Entry Logs/071 measures which one the model picks and why: DRD2 takes the amine ~100% of the
time because its reward strongly prefers it (0.559 vs 0.348), while sEH prefers the amide and takes
the amine below chance.

Per example it draws: the two reactants, the product of the direct acid -> amine template (what was
recorded), and the product of the amide template on the SAME two reactants -- both computed by running
the templates, neither hand-drawn.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/plot_acid_amine_families.py
"""
import io
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reaction_names import parse_template  # noqa: E402

RDLogger.DisableLog("rdApp.*")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
# Two families, not a right and a wrong one -- so two distinct hues carrying no verdict.
DIRECT_C, AMIDE_C = "#b8762a", "#2a6f9e"   # direct acid -> amine (0.70) | amide (0.75)
PANEL = (330, 240)
SNAP = "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820"
AMIDE_RXN = AllChem.ReactionFromSmarts(
    "[C:3](=[O:4])-[Oh].[Nh2:5]-[#6&!$(C=[O,N]):10]>>[C:3](=[O:4])-[Nh:5]-[#6&!$(C=[O,N]):10]")


def draw(smi, highlight=None, colour=None):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return b""
    Chem.rdDepictor.Compute2DCoords(m)
    d = rdMolDraw2D.MolDraw2DCairo(*PANEL)
    d.drawOptions().clearBackground = False
    d.drawOptions().highlightBondWidthMultiplier = 18
    if highlight:
        rdMolDraw2D.PrepareAndDrawMolecule(
            d, m, highlightAtoms=list(highlight),
            highlightAtomColors={i: colour for i in highlight})
    else:
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
    d.FinishDrawing()
    return d.GetDrawingText()


def amide_or_amine_site(smi, want_amide: bool):
    """Atoms of the C(=O)N amide (or the C-N amine) formed at the join, for highlighting."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return []
    patt = Chem.MolFromSmarts("[CX3](=O)[NX3]" if want_amide else "[CH2][NX3;H1]c")
    hits = m.GetSubstructMatches(patt)
    return list(hits[0]) if hits else []


def main():
    R = json.loads(Path(f"{SNAP}/seh_seed43/routes.json").read_text())
    examples, seen, seen_tids = [], set(), set()
    for smi, rt in R.items():
        for s in rt.get("steps") or []:
            tid, _ = parse_template(s.get("reaction") or "")
            if tid not in ("136", "137", "138", "139"):
                continue
            acid = (s.get("reactants") or [None])[0]
            amine, rec = s.get("input"), s.get("product")
            if not (acid and amine and rec) or acid in seen or tid in seen_tids:
                continue
            a, b = Chem.MolFromSmiles(acid), Chem.MolFromSmiles(amine)
            if a is None or b is None:
                continue
            prods = AMIDE_RXN.RunReactants((a, b)) or AMIDE_RXN.RunReactants((b, a))
            if not prods:
                continue
            seen.add(acid)
            seen_tids.add(tid)
            examples.append((tid, acid, amine, rec, Chem.MolToSmiles(prods[0][0])))
            break
        if len(examples) >= 3:
            break

    n = len(examples)
    fig, axes = plt.subplots(n, 4, figsize=(16.4, 3.5 * n), facecolor=SURFACE)
    if n == 1:
        axes = [axes]
    heads = ["acid reactant", "amine (the growing molecule)",
             "direct acid → amine (0.70)  — recorded", "amide coupling (0.75)  — same reactants"]
    for r, (tid, acid, amine, rec, exp) in enumerate(examples):
        cells = [(acid, None, None), (amine, None, None),
                 (rec, amide_or_amine_site(rec, False), (0.95, 0.55, 0.5)),
                 (exp, amide_or_amine_site(exp, True), (0.55, 0.90, 0.75))]
        for c, (smi, hl, col) in enumerate(cells):
            ax = axes[r][c]
            ax.set_facecolor(SURFACE); ax.axis("off")
            png = draw(smi, hl, col)
            if png:
                ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
            if r == 0:
                ax.set_title(heads[c], fontsize=10.5, color=DIRECT_C if c == 2 else (AMIDE_C if c == 3 else INK2),
                             fontweight="semibold", pad=8)
            for sp in ax.spines.values():
                sp.set_visible(c >= 2)
                sp.set_color(DIRECT_C if c == 2 else AMIDE_C); sp.set_linewidth(1.8)
            ax.set_xticks([]); ax.set_yticks([])
        axes[r][0].text(-0.04, 0.5, f"template {tid}\n“direct acid → amine”", transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=10, color=INK, fontweight="semibold")
        # the arrow between the two products: same reactants, different family
        axes[r][2].annotate("", xy=(1.06, 0.5), xytext=(0.99, 0.5), xycoords="axes fraction",
                            arrowprops=dict(arrowstyle="-|>", color=INK3, linewidth=1.4))
    fig.text(0.5, 0.982, "One acid + one amine, two reactions the library will sell you",
             ha="center", va="top", fontsize=15, color=INK, fontweight="bold")
    fig.text(0.5, 0.947,
             "Templates 136–139 (“direct acid → amine”, yield 0.70) leave the carbonyl oxygen UNMAPPED because it "
             "leaves: the product is a secondary amine.\nTemplates 76–79 (“Amide synthesis”, yield 0.75) map it "
             "through and give the amide. Identical reactants, so the model chooses between them — and the O₂ loss "
             "with hydrogens balanced is the signature of the reduction, not of a bug.",
             ha="center", va="top", fontsize=9.6, color=INK2, linespacing=1.6)
    fig.text(0.008, 0.014,
             "Both routes are valid synthesis instructions; they are not equally easy to run, and the reaction count "
             "we budget against cannot tell them apart. The direct reduction is\n267,204 of 747,472 steps (35.7%) "
             "across the four publishable cells — 99.8% of DRD2’s acid+amine steps, but only 31.7–42.3% of sEH’s. "
             "Logs/071 measures why.",
             fontsize=8.4, color=INK3, ha="left", va="bottom", linespacing=1.6)
    fig.subplots_adjust(left=0.115, right=0.995, top=0.885, bottom=0.085, wspace=0.05, hspace=0.16)
    out = Path("experiments/lsd_hubs/campaign/results/acid_amine_families")
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"acid_amine_families.{ext}", dpi=190, facecolor=SURFACE)
    print(f"wrote {out}/acid_amine_families.png")
    for tid, acid, amine, rec, exp in examples:
        print(f"  t{tid}: recorded {rec[:52]}")
        print(f"         correct  {exp[:52]}")


if __name__ == "__main__":
    main()
