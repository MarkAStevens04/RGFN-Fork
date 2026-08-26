#!/usr/bin/env python
"""One batch, drawn the way a chemist reads a scheme: make the hub, then split it.

This is the route dataset's per-batch figure (``batch_NN/scheme.png``).

TWO SECTIONS, because a batch is two different jobs.

  TOP -- how you make the hub. One row per reaction, written the standard way: ``X + Y -> Z`` when
  both starting materials are bought, otherwise ``Y -> Z`` with the bought reagent over the arrow.
  The rule behind that placement is buy-vs-make, not which SMILES the log calls the "input":
  anything you MAKE stands on the left as a substrate, anything you ORDER goes over the arrow.

  BOTTOM -- what you do with the hub. Every candidate is one reaction off it, so each row is the hub
  on the left, the candidate on the right, the reagent over the arrow, reaction and yield beneath,
  and the candidate's predicted score under the structure. This is the batch: one shared prefix,
  N ways to finish it.

COLOUR CARRIES THE BILL OF MATERIALS:
  blue = purchasable building block (order it)        black = intermediate you make on the way
  maroon = hub or promoted fragment                   green = final candidate
"Purchasable" is exactly the run's own ``initial_smiles_set`` and "promoted" its ``chosen_smiles``
(disjoint, verified), so the figure cannot disagree with what the model was allowed to buy.

LAYOUT. Every structure is drawn at ONE fixed bond length, so a building block genuinely looks
smaller than the candidate it becomes. Drawing into fixed boxes at a fixed bond length centres small
molecules in a large empty panel, which is what made the first version look sparse; instead each
render is CROPPED to its own ink and then anchored to the arrow, so scale stays uniform while the
spacing closes up. Row heights follow their contents for the same reason.

The reaction label sits just under its arrow, where it belongs. It is often wider than the arrow, so
whether it actually collides with the neighbouring structures is tested against their INK rather than
their bounding boxes -- a molecule drawn on a diagonal leaves its lower corners empty, and a box test
would push the label away from perfectly usable space. (Measured on the 4-ethynylaniline above: ink
spans the full 284px width overall but only columns 14-175 in the band the label occupies.) The label
is wrapped onto more lines only when the ink says it must, and only dropped below the structures if
even the narrowest wrapping still collides. All of this is computed in INCHES before the figure
exists, because the page size itself depends on the outcome.

YIELDS are the per-template nominal values from SCENT's own library (0.65-0.95), read from
``results/template_yields.csv`` -- generated once by ``yield_preference.py`` so this script needs no
heavy imports. They are LIBRARY DEFAULTS PER REACTION CLASS, not predictions for these specific
substrates: a chemist still has to optimise each step. They are shown because the reaction budget we
report against is yield-blind, which is exactly the limitation Logs/071 measures.

Pure CPU. Run:
    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/render_batch_scheme.py \
        --cell seed44 --target seh --hub 'Nc1ccc(-c2cn(CC3CC3)nn2)cc1' --max-children 4
"""
import argparse
import csv
import io
import json
import re
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from PIL import Image
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDepictor, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

RDLogger.DisableLog("rdApp.*")

SNAP = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/_enum_snapshot_20260820")
FR = Path("/scratch/markymoo/rgfn_runs/experiments/fixed_reward")
MTX = Path("/scratch/markymoo/rgfn_runs/lsdflow")
OUT = Path("experiments/lsd_hubs/campaign/results/batch_scheme")
YIELDS = Path("experiments/lsd_hubs/campaign/results/template_yields.csv")

SURFACE, INK, INK2, INK3 = "#fcfcfb", "#12100e", "#55524d", "#8d8a83"
BUY, MAKE, HUBC, PROD = "#41739e", "#17150f", "#7d2b40", "#2c7a51"

PPI = 200            # on-page pixels per inch for the cropped structures
ARROW_IN = 1.85      # arrow length, inches
GAP_IN = 0.24        # between a structure and the arrow it feeds
PLUS_IN = 0.34       # room for the "+" between two substrates
MARGIN_IN = 0.30
FIG_W_MIN = 9.8      # enough for the legend line regardless of how narrow the chemistry is
BOND_MAIN = 34       # fixed bond length -> uniform scale everywhere
BOND_REAG = 26       # reagents over the arrow, deliberately a size down
GAP_ABOVE = 0.05     # inches between the arrow and the reagent sitting over it
LINE_H = 0.185       # inches per line of text under a row
GAP_TEXT = 0.09      # inches between the arrow and its label
CLASH_PAD = 0.07     # inches of clear space demanded either side of the label
WRAPS = (70, 34, 26, 21, 17)   # characters per line, tried in order
PAD_Y = 0.30
TOP_PAD, BOT_PAD, HEAD_H = 0.80, 0.92, 0.44
TID = re.compile(r",\s*\d+,\s*(\d+)\)\s*$")


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


_POSE = {}


def _align(m, ref):
    """Give ``m`` the same 2D pose as ``ref`` over the part they share. True if it took.

    Keeping a hub in one orientation wherever it appears is what lets a chemist skim a batch: the
    shared scaffold sits still and only the new arm moves. The fast path uses the reference itself as
    the query, which is exact when the hub survives the reaction; a reaction that REBUILDS the hub
    (Pictet-Spengler closes a ring through it) falls back to the maximum common substructure, and
    ring atoms are only matched to ring atoms so a newly formed ring is never forced into the
    template's acyclic geometry.
    """
    try:
        if m.HasSubstructMatch(ref) and not _breaks_rings(m, ref):
            rdDepictor.GenerateDepictionMatching2DStructure(m, ref, acceptFailure=True)
            return True
        res = rdFMCS.FindMCS([ref, m], timeout=5, ringMatchesRingOnly=True,
                             completeRingsOnly=True)
        patt = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
        if patt is None or patt.GetNumAtoms() < 5:
            return False
        rdDepictor.GenerateDepictionMatching2DStructure(m, ref, refPatt=patt, acceptFailure=True)
        return True
    except Exception:
        return False


def _breaks_rings(m, ref):
    """True when the match would place ring atoms of ``m`` on acyclic template atoms."""
    match = m.GetSubstructMatch(ref)
    if not match:
        return True
    return any(m.GetAtomWithIdx(mi).IsInRing() and not ref.GetAtomWithIdx(ri).IsInRing()
               for ri, mi in enumerate(match))


def attach_atom(prod, ref):
    """Index (in ``ref``) of the reference atom that gains a substituent in ``prod``."""
    match = prod.GetSubstructMatch(ref)
    if not match:
        return None
    mapped = set(match)
    for i, pi in enumerate(match):
        for nb in prod.GetAtomWithIdx(pi).GetNeighbors():
            if nb.GetIdx() not in mapped:
                return i
    return None


def orient_for_growth(ref, products):
    """Rotate ``ref`` in place so the bond it will grow points RIGHT.

    Without this the constrained products still share the hub's pose, but their new arms hang
    wherever RDKit puts them -- in practice downwards, which stretches every row and makes the page
    twice as long. Pointing the growth vector along +x keeps each row short and wide, which is the
    shape the product column already has.
    """
    import math

    votes = {}
    for pm in products:
        i = attach_atom(pm, ref)
        if i is not None:
            votes[i] = votes.get(i, 0) + 1
    if not votes:
        return
    idx = max(votes, key=votes.get)
    conf = ref.GetConformer()
    n = ref.GetNumAtoms()
    cx = sum(conf.GetAtomPosition(i).x for i in range(n)) / n
    cy = sum(conf.GetAtomPosition(i).y for i in range(n)) / n
    a = conf.GetAtomPosition(idx)
    th = -math.atan2(a.y - cy, a.x - cx)
    cos, sin = math.cos(th), math.sin(th)
    for i in range(n):
        pt = conf.GetAtomPosition(i)
        dx, dy = pt.x - cx, pt.y - cy
        conf.SetAtomPosition(i, (cx + dx * cos - dy * sin, cy + dx * sin + dy * cos, 0.0))


def posed(smi, ref=None):
    """A molecule with 2D coordinates, aligned to ``ref``'s pose where they overlap."""
    if smi in _POSE:
        return _POSE[smi]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    if m.GetNumHeavyAtoms() <= 3:      # acetylene etc. draw as bare lines with implicit carbons
        m = Chem.AddHs(m)
    if not (ref is not None and _align(m, ref)):
        rdDepictor.Compute2DCoords(m)
        rdDepictor.StraightenDepiction(m)
    _POSE[smi] = m
    return m


def draw_mol(m, colour, bond_px=BOND_MAIN):
    """Render an ALREADY-POSED molecule at a FIXED bond length, cropped to its ink.

    Cropping is what lets one scale serve every structure without small molecules floating in a
    large empty panel -- the caller anchors the cropped image instead of centring it in a box.
    """
    if m is None:
        return None, 0, 0, None
    for canvas in ((1800, 1200), (3000, 2100), (4600, 3200)):
        d = rdMolDraw2D.MolDraw2DCairo(*canvas)
        o = d.drawOptions()
        o.clearBackground = False
        o.addStereoAnnotation = True
        # updateAtomPalette MERGES and N/O/halogens carry their own entries, so cover the range
        o.updateAtomPalette({i: rgb(colour) for i in range(-1, 119)})
        o.bondLineWidth = 2
        o.fixedBondLength = bond_px
        rdMolDraw2D.PrepareAndDrawMolecule(d, m)
        d.FinishDrawing()
        img = Image.open(io.BytesIO(d.GetDrawingText())).convert("RGBA")
        bb = img.getbbox()
        if bb is None:
            return None, 0, 0, None
        # a bbox touching the edge means the structure was clipped -- retry on a bigger canvas
        if bb[0] > 1 and bb[1] > 1 and bb[2] < canvas[0] - 1 and bb[3] < canvas[1] - 1:
            pad = 6
            bb = (max(0, bb[0] - pad), max(0, bb[1] - pad),
                  min(canvas[0], bb[2] + pad), min(canvas[1], bb[3] + pad))
            crop = img.crop(bb)
            buf = io.BytesIO()
            crop.save(buf, format="PNG")
            return buf.getvalue(), crop.width, crop.height, np.array(crop)[:, :, 3]
    return None, 0, 0, None


def place(fig, png, w, h, _alpha, x, y_mid, ha="left"):
    """Place a cropped render at its natural size, anchored (not stretched) to x / y_mid."""
    if not png:
        return 0.0
    fw, fh = fig.get_figwidth(), fig.get_figheight()
    wf, hf = (w / PPI) / fw, (h / PPI) / fh
    x0 = x if ha == "left" else (x - wf if ha == "right" else x - wf / 2)
    ax = fig.add_axes([x0, y_mid - hf / 2, wf, hf])
    ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
    ax.axis("off")
    ax.set_facecolor("none")
    return wf


_TW = {}


def text_w(txt, size, italic=False):
    """Width of a string in inches, without needing a renderer."""
    k = (txt, size, italic)
    if k not in _TW:
        prop = FontProperties(style="italic" if italic else "normal")
        _TW[k] = TextPath((0, 0), txt, size=size, prop=prop).get_extents().width / 72
    return _TW[k]


def ink_span(alpha, w_in, h_in, x_left, y_lo, y_hi):
    """Horizontal ink extent (inches) of a render within a vertical band.

    The structure is centred on the arrow line (y=0) and spans [-h/2, +h/2]. Testing INK here rather
    than the image rectangle is the whole point: a diagonal molecule leaves its lower corners empty,
    and a rectangle test would refuse to put a label in space that is demonstrably free.
    """
    if alpha is None:
        return None
    H, W = alpha.shape
    r0 = int(np.clip((h_in / 2 - y_hi) / h_in * H, 0, H))
    r1 = int(np.clip((h_in / 2 - y_lo) / h_in * H, 0, H))
    if r1 <= r0:
        return None
    cols = np.where(alpha[r0:r1].any(axis=0))[0]
    if cols.size == 0:
        return None
    return x_left + cols.min() / W * w_in, x_left + (cols.max() + 1) / W * w_in


def role(smi, buyable, promoted, hub):
    if smi == hub or smi in promoted:
        return HUBC
    return BUY if smi in buyable else MAKE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="seed44")
    ap.add_argument("--target", default="seh")
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--hub", required=True)
    ap.add_argument("--children", nargs="*", default=None)
    ap.add_argument("--detail-csv", default=None)
    ap.add_argument("--max-children", type=int, default=4)
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    cell, hub = a.cell, a.hub
    seed = "seed43" if "43" in cell else "seed44"
    sub = "scent_drd2_5k" if a.target == "drd2" else "scent_seh_5k"
    snap_p = Path(a.snapshot) if a.snapshot else (
        FR / sub / seed / "additional_fragments" / "fragments_4000.json")

    routes = json.loads((SNAP / cell / "routes.json").read_text())
    snap = json.loads(snap_p.read_text())
    buyable, promoted = set(snap["initial_smiles_set"]), set(snap["chosen_smiles"])
    ytab = {int(r["template_id"]): (float(r["yield"]), r["named_reaction"])
            for r in csv.DictReader(YIELDS.open())}

    def step_info(st):
        """(reaction name, yield) for one logged step, from the template id."""
        m = TID.search(st.get("reaction") or "")
        if not m or int(m.group(1)) not in ytab:
            return "reaction", None
        y, nm = ytab[int(m.group(1))]
        return nm, y

    raw = json.loads((SNAP / cell / "enum_children.json").read_text())
    kids = {c["smiles"]: c for h in raw["hubs"] if h["hub_key"] == hub for c in h["children"]}
    if not kids:
        raise SystemExit(f"hub not found in enumeration: {hub}")

    if a.children:
        chosen = a.children
    else:
        dcsv = Path(a.detail_csv) if a.detail_csv else Path(
            f"experiments/lsd_hubs/campaign/results/yield_preference/"
            f"strategy_yield_detail_{a.target}_{cell}.csv")
        chosen = [r["smiles"] for r in csv.DictReader(dcsv.open())
                  if r["arm"] == "hub_batching" and r["hub"] == hub] if dcsv.exists() else []
    n_total = len(chosen)
    chosen = [c for c in chosen if c in kids][:a.max_children]
    if not chosen:
        raise SystemExit("no children to draw")

    stereo = {}
    for p in (MTX / f"matrix16_{seed}" / f"scent_{a.target}" / "sample" / "records.csv",
              MTX / f"matrix16_{seed}" / f"scent_{a.target}" / "enum" / "enumerated_records.csv"):
        if p.exists():
            with p.open() as fh:
                for r in csv.DictReader(fh):
                    stereo.setdefault(r["child_key"], r.get("child_stereo_key") or r["child_key"])

    make_steps = (routes.get(hub) or {}).get("steps") or []

    # The hub is the anchor of the whole figure, so its pose is settled before anything else: first
    # a normal depiction, then rotated so the bond the batch grows from points right. Section 1 is
    # then posed BACKWARD from the hub (each substrate aligned to what it becomes) so the top and
    # bottom halves agree, rather than forward from an arbitrary starting block.
    hub_mol = Chem.MolFromSmiles(hub)
    if hub_mol is not None:
        rdDepictor.Compute2DCoords(hub_mol)
        rdDepictor.StraightenDepiction(hub_mol)
        prods = []
        for c in chosen:
            pm = Chem.MolFromSmiles(stereo.get(c, c))
            if pm is not None:
                prods.append(pm)
        orient_for_growth(hub_mol, prods)
        _POSE[hub] = hub_mol
    for st in reversed(make_steps):
        pm = posed(st.get("product"))
        if st.get("input"):
            posed(st["input"], pm)
    hub_yield = 1.0
    for st in make_steps:
        hub_yield *= (step_info(st)[1] or 1.0)

    score_lab = "DRD2" if a.target == "drd2" else "sEH"
    fmt = (lambda v: f"{v:.3f}") if a.target == "drd2" else (lambda v: f"{v:.2f}")

    # ---- pass 1a: render every structure; sizes drive both the page and the label placement ----
    def row_for(lefts, aboves, prod, prod_c, name, yld, sub_lines, prod_ref=None):
        r = dict(yld=yld, sub=sub_lines, name=name)
        r["lefts"] = [draw_mol(posed(x), role(x, buyable, promoted, hub)) for x in lefts]
        r["aboves"] = [draw_mol(posed(x), role(x, buyable, promoted, hub), BOND_REAG)
                       for x in aboves]
        r["prod"] = draw_mol(posed(prod, prod_ref), prod_c)
        r["h_main"] = max([im[2] for im in r["lefts"] + [r["prod"]] if im[0]] or [1]) / PPI
        r["h_above"] = max([im[2] for im in r["aboves"] if im[0]] or [0]) / PPI
        return r

    rows_make = []
    for st in make_steps:
        inp, rcts = st.get("input"), list(st.get("reactants") or [])
        parts = ([inp] if inp else []) + rcts
        lefts = [x for x in parts if x not in buyable]
        aboves = [x for x in parts if x in buyable]
        if not lefts:
            lefts, aboves = parts, []
        prod = st.get("product")
        nm, y = step_info(st)
        # the cumulative yield belongs under the HUB, where the hub is actually made
        sub = ([(f"route ≈{100 * hub_yield:.0f}%", INK3, "normal")] if prod == hub else [])
        rows_make.append(row_for(lefts, aboves, prod, HUBC if prod == hub else MAKE, nm, y, sub))

    rows_use = []
    for c in chosen:
        st = (kids[c].get("reaction") or [{}])[0]
        nm, y = step_info(st)
        rew = kids[c].get("reward")
        sub = ([(f"{score_lab} {fmt(float(rew))}", PROD, "bold")] if rew is not None else [])
        rows_use.append(row_for([hub], list(st.get("reactants") or []),
                                stereo.get(c, c), PROD, nm, y, sub, prod_ref=posed(hub)))

    sections = [("1 · Make the hub", rows_make),
                (f"2 · Split the hub — {n_total} candidate"
                 + ("s" if n_total != 1 else "") + " from one intermediate"
                 + (f" ({len(chosen)} shown)" if len(chosen) < n_total else ""), rows_use)]
    allrows = [r for _l, rws in sections for r in rws]

    # ---- page width follows the widest row, so there is no dead margin beside the scheme --------
    left_in = max((sum(im[1] for im in r["lefts"] if im[0]) / PPI
                   + (PLUS_IN if len(r["lefts"]) >= 2 else 0.0)) for r in allrows)
    prod_in = max((r["prod"][1] / PPI if r["prod"][0] else 0.0) for r in allrows)
    fig_w = max(FIG_W_MIN, 2 * MARGIN_IN + left_in + 2 * GAP_IN + ARROW_IN + prod_in)
    xa0 = MARGIN_IN + left_in + GAP_IN          # arrow start, inches from the page's left edge
    xa1 = xa0 + ARROW_IN
    xc = (xa0 + xa1) / 2

    # ---- pass 1b: place each label as close under its arrow as its neighbours' INK allows -------
    for r in allrows:
        h_main = r["h_main"]
        # where each structure sits, in inches, with the arrow line at y = 0
        boxes = []
        x = xa0 - GAP_IN
        for im in reversed(r["lefts"]):
            if im[0]:
                w = im[1] / PPI
                boxes.append((im[3], w, im[2] / PPI, x - w))
                x -= w + (PLUS_IN if len(r["lefts"]) >= 2 else 0.0)
        prod_box = None
        if r["prod"][0]:
            prod_box = (r["prod"][3], r["prod"][1] / PPI, r["prod"][2] / PPI, xa1 + GAP_IN)

        ylab = f"≈{100 * r['yld']:.0f}% yield" if r["yld"] is not None else None
        for wrap in WRAPS:
            lines_ = textwrap.wrap(r["name"], wrap) or [r["name"]]
            tw = max([text_w(t, 9.6, True) for t in lines_]
                     + ([text_w(ylab, 8.8)] if ylab else []))
            bh = (len(lines_) + (1 if ylab else 0)) * LINE_H
            y_hi, y_lo = -GAP_TEXT, -GAP_TEXT - bh
            tx0, tx1 = xc - tw / 2, xc + tw / 2
            hit = False
            for alpha, w, h, xl in boxes:
                sp = ink_span(alpha, w, h, xl, y_lo, y_hi)
                if sp and sp[1] > tx0 - CLASH_PAD:
                    hit = True
            if prod_box:
                sp = ink_span(*prod_box[:1], prod_box[1], prod_box[2], prod_box[3], y_lo, y_hi)
                if sp and sp[0] < tx1 + CLASH_PAD:
                    hit = True
            if not hit:
                break
        r["lines"], r["ylab"] = lines_, ylab
        if hit:      # even the narrowest wrapping collides -> drop the label under the structures
            r["y_text"] = -(h_main / 2 + 0.10)
        else:
            r["y_text"] = -GAP_TEXT
        text_bot = -r["y_text"] + bh
        sub_bot = ((r["prod"][2] / PPI) / 2 + 0.07 + len(r["sub"]) * LINE_H) if r["sub"] else 0.0
        r["top"] = max(h_main / 2, r["h_above"] + GAP_ABOVE)
        r["below"] = max(h_main / 2, text_bot, sub_bot)
        r["h"] = r["top"] + r["below"] + PAD_Y

    fig_h = (TOP_PAD + BOT_PAD + len(sections) * HEAD_H
             + sum(r["h"] for _l, rws in sections for r in rws))
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=SURFACE)
    fig.text(0.5, 1 - 0.28 / fig_h, "How to make this batch", ha="center", va="top",
             fontsize=16, color=INK, fontweight="bold")

    cur = TOP_PAD
    for label, rws in sections:
        if not rws:
            continue
        fig.text(MARGIN_IN / fig_w, 1 - (cur + 0.04) / fig_h, label, ha="left", va="top",
                 fontsize=12.5, color=INK, fontweight="semibold")
        fig.add_artist(plt.Line2D([MARGIN_IN / fig_w, 1 - MARGIN_IN / fig_w],
                                  [1 - (cur + 0.31) / fig_h] * 2, color=INK3, lw=0.9, alpha=0.55))
        cur += HEAD_H

        for r in rws:
            mid = 1.0 - (cur + r["top"] + PAD_Y / 2) / fig_h

            def fx(v):        # inches from the page's left edge -> figure fraction
                return v / fig_w

            def fy(v):        # inches above the arrow line -> figure fraction
                return mid + v / fig_h

            if len(r["lefts"]) >= 2:
                x = xa0 - GAP_IN
                w1 = place(fig, *r["lefts"][1], fx(x), mid, ha="right")
                fig.text(fx(x) - w1 - PLUS_IN / 2 / fig_w, mid, "+", ha="center", va="center",
                         fontsize=17, color=INK2)
                place(fig, *r["lefts"][0], fx(x) - w1 - PLUS_IN / fig_w, mid, ha="right")
            else:
                place(fig, *r["lefts"][0], fx(xa0 - GAP_IN), mid, ha="right")

            fig.add_artist(plt.Line2D([fx(xa0), fx(xa1)], [mid, mid], color=INK, lw=1.5))
            for dy in (0.0040, -0.0040):
                fig.add_artist(plt.Line2D([fx(xa1) - 0.009, fx(xa1)], [mid + dy, mid],
                                          color=INK, lw=1.5))
            if r["aboves"]:
                tot = sum(im[1] for im in r["aboves"] if im[0]) / PPI
                x = xc - tot / 2
                for im in r["aboves"]:
                    if not im[0]:
                        continue
                    hin = im[2] / PPI
                    place(fig, *im, fx(x), fy(GAP_ABOVE + hin / 2), ha="left")
                    x += im[1] / PPI

            ty = fy(r["y_text"])
            for ln in r["lines"]:
                fig.text(fx(xc), ty, ln, ha="center", va="top", fontsize=9.6,
                         color=INK2, style="italic")
                ty -= LINE_H / fig_h
            if r["ylab"]:
                fig.text(fx(xc), ty, r["ylab"], ha="center", va="top",
                         fontsize=8.8, color=INK3)

            wp = place(fig, *r["prod"], fx(xa1 + GAP_IN), mid, ha="left")
            sy = fy(-(r["prod"][2] / PPI) / 2 - 0.07)
            for txt, col, weight in r["sub"]:
                fig.text(fx(xa1 + GAP_IN) + wp / 2, sy, txt, ha="center", va="top",
                         fontsize=9.4, color=col, fontweight=weight)
                sy -= LINE_H / fig_h
            cur += r["h"]


    # legend + footnote are placed in INCHES: the page width follows the chemistry, so fractions
    # that fit one batch overlap on a narrower one.
    for x_in, c, lab in ((MARGIN_IN, BUY, "■  purchasable — order it"),
                         (MARGIN_IN + 2.30, MAKE, "■  intermediate — make it"),
                         (MARGIN_IN + 4.60, HUBC, "■  hub / promoted fragment"),
                         (MARGIN_IN + 6.95, PROD, "■  final candidate")):
        fig.text(x_in / fig_w, 0.50 / fig_h, lab, fontsize=9.0, color=c, ha="left",
                 va="center", fontweight="semibold")
    note = ("Yields are the reaction library's nominal per-class values, not predictions for these "
            f"substrates — each step still needs optimising. Scores are the {score_lab} reward the "
            "model was trained against.")
    for i, ln in enumerate(textwrap.wrap(note, int((fig_w - 2 * MARGIN_IN) / 0.062))):
        fig.text(MARGIN_IN / fig_w, (0.28 - i * 0.145) / fig_h, ln,
                 fontsize=8.2, color=INK3, ha="left", va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.target}_{cell}"
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"batch_scheme_{tag}.{ext}", dpi=185, facecolor=SURFACE)
    plt.close(fig)
    print(f"hub: {hub}")
    print(f"  hub route: {len(make_steps)} step(s), nominal yield ≈{100*hub_yield:.0f}%; "
          f"batch: {n_total} candidates, {len(chosen)} drawn")
    print(f"wrote {OUT}/batch_scheme_{tag}.png + .pdf")


if __name__ == "__main__":
    main()
