#!/usr/bin/env python
"""Render one molecule's logged synthesis route as a reaction scheme.

A PROTOTYPE for the route dataset's ``batch_NN/scheme.png`` (docs/ROUTE_DATASET_SCHEMA.md §4.5),
written to be looked at before the exporter is built -- the point is to see whether a logged assembly
route reads as chemistry, not to produce a final artifact.

Two things it makes visible that a SMILES table cannot:

  * BUY vs MAKE. A reactant is coloured as bought iff it is in the run's ``initial_smiles_set``;
    anything else was built by an earlier step. This is the distinction a chemist acts on, and it is
    the reason the dataset expands promoted fragments inline rather than emitting "buy X" for an X
    nobody sells.
  * STEREOCHEMISTRY, drawn wedge/dash rather than flattened. Our targets' stereocentres are
    INHERITED from chiral-pool blocks -- measured 0 of 10,248 route steps create one -- so the centre
    visible in the product is the one you bought, not one you have to resolve. Seeing L-proline enter
    as ``O=C(O)[C@H]1CCCN1`` and survive to the product is the argument for shipping stereo-aware
    structures instead of squiggles.

The stereo-aware form is NOT in routes.json (its keys and products are stereo-stripped, only the
reactants keep stereo), so it is recovered by joining the flat key against ``child_stereo_key`` in
the sample/enumeration records -- exactly what the exporter will have to do.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/render_route_scheme.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BUY, MAKE = "#1baf7a", "#2a78d6"          # green = order it, blue = you made it
PANEL = (300, 210)

# Readable names for the templates we see most; the template id remains the identifier.
NAMED = [
    ("[Oh:5]-[c;r6:4][c;r6:3]-[Nh2:2]", "Benzoxazole/benzimidazole formation"),
    ("[C:3](=[O:4])-[Oh]", "Amide coupling"),
    ("[c:2]-[Br]", "Buchwald–Hartwig amination"),
    ("[F]-[c:2]", "SNAr (aryl fluoride + amine)"),
    ("B(-[Oh])", "Suzuki coupling"),
    ("C#C", "Sonogashira coupling"),
    ("nnn", "Tetrazole formation"),
]


def named_reaction(tmpl: str) -> str:
    for sig, name in NAMED:
        if sig in (tmpl or ""):
            return name
    return "reaction"


def draw(smi: str, colour: str, label: str, size=PANEL) -> bytes:
    """One molecule, wedge/dash stereo preserved, with a coloured frame meaning buy-vs-make."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return b""
    Chem.rdDepictor.Compute2DCoords(m)
    Chem.rdDepictor.StraightenDepiction(m)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    o = d.drawOptions()
    o.clearBackground = False
    o.addStereoAnnotation = True          # show R/S -- the centre is bought, so name it
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, legend=label)
    d.FinishDrawing()
    return d.GetDrawingText()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="/scratch/markymoo/rgfn_runs/lsdflow_sparrow/"
                    "_enum_snapshot_20260820/seh_seed43/routes.json")
    ap.add_argument("--records", nargs="*", default=[
        "/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43/scent_seh/sample/records.csv",
        "/scratch/markymoo/rgfn_runs/lsdflow/matrix16_seed43/scent_seh/enum/enumerated_records.csv"])
    ap.add_argument("--snapshot", default="/scratch/markymoo/rgfn_runs/experiments/fixed_reward/"
                    "scent_seh_5k/seed43/additional_fragments/fragments_4000.json")
    ap.add_argument("--min-steps", type=int, default=3)
    ap.add_argument("--n", type=int, default=3, help="how many routes to render")
    ap.add_argument("--out-dir", default="experiments/lsd_hubs/campaign/results/route_scheme_proto")
    a = ap.parse_args()

    routes = json.loads(Path(a.routes).read_text())
    snap = json.loads(Path(a.snapshot).read_text())
    buyable = set(snap.get("initial_smiles_set") or [])
    stereo = {}
    for p in a.records:
        if not Path(p).exists():
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):
                stereo.setdefault(r["child_key"], r["child_stereo_key"])

    # prefer routes that are multi-step AND whose target recovers a stereocentre: the case that
    # actually exercises both things this prototype exists to show
    picks = []
    for smi, rt in routes.items():
        steps = rt.get("steps") or []
        if len(steps) < a.min_steps:
            continue
        st = stereo.get(smi, smi)
        if "@" not in st:
            continue
        picks.append((smi, st, rt))
        if len(picks) >= a.n:
            break
    if not picks:
        raise SystemExit("no multi-step stereo-bearing route found — relax --min-steps")

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx, (flat, st_smi, rt) in enumerate(picks, 1):
        steps = rt["steps"]
        fig, axes = plt.subplots(len(steps), 3, figsize=(13.2, 3.05 * len(steps)),
                                 facecolor=SURFACE)
        if len(steps) == 1:
            axes = [axes]
        for i, s in enumerate(steps):
            reactants = list(s.get("reactants") or [])
            inp = s.get("input")
            prod = s.get("product")
            left = reactants[0] if reactants else inp
            mid = inp if reactants else None
            cells = [(left, "reactant"), (mid, "input"), (prod, "product")]
            for j, (smi_, role) in enumerate(cells):
                ax = axes[i][j]
                ax.set_facecolor(SURFACE); ax.axis("off")
                if not smi_:
                    continue
                is_buy = smi_ in buyable
                col = BUY if is_buy else MAKE
                tag = ("buy" if is_buy else "make") if role != "product" else "product"
                png = draw(stereo.get(smi_, smi_), col, "")
                if png:
                    ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
                ax.set_title(tag, fontsize=9.5, color=col, fontweight="semibold", pad=2)
                for sp in ax.spines.values():
                    sp.set_visible(True); sp.set_color(col); sp.set_linewidth(1.6)
                ax.set_xticks([]); ax.set_yticks([])
            axes[i][0].text(-0.05, 0.5, f"step {i+1}\n{named_reaction(s.get('reaction'))}",
                            transform=axes[i][0].transAxes, ha="right", va="center",
                            fontsize=9.5, color=INK, fontweight="semibold")
        fig.text(0.5, 0.985, f"Logged assembly route — {len(steps)} steps",
                 ha="center", va="top", fontsize=13, color=INK, fontweight="bold")
        fig.text(0.5, 0.955, st_smi, ha="center", va="top", fontsize=8.4, color=INK2,
                 family="monospace")
        fig.text(0.008, 0.012,
                 "green = purchasable building block (order it)   ·   blue = made in an earlier step."
                 "  Stereocentres are drawn wedge/dash with R/S annotation: they are INHERITED from a "
                 "chiral-pool block,\nnot created — 0 of 10,248 route steps create a centre — so the "
                 "configuration shown is the one you buy, not one requiring resolution. This is the "
                 "logged assembly route, not a claimed optimal retrosynthesis.",
                 fontsize=8, color=INK3, ha="left", va="bottom", linespacing=1.5)
        fig.subplots_adjust(left=0.115, right=0.99, top=0.93, bottom=0.10, hspace=0.28, wspace=0.04)
        f = out / f"route_{idx}.png"
        fig.savefig(f, dpi=190, facecolor=SURFACE); plt.close(fig)
        print(f"  wrote {f}  ({len(steps)} steps)  {st_smi[:58]}")


if __name__ == "__main__":
    main()
