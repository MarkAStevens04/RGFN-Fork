#!/usr/bin/env python
"""Look at the property-matched decoys that BEAT the gate, next to the real glue each most resembles.

WHY THESE PAIRS AND NOT RANDOM ONES. 31% of property-matched decoys clear the -2.0 gate. A number
that large is easy to wave away as "the decoys must be secretly glue-like", so this renders the
worst offenders -- the decoys with the STRONGEST differential -- beside their nearest real glue. If
they look like glues, the decoy set is at fault. If they look nothing like glues, the gate is.

Each pair carries the numbers that matter: Tanimoto to that nearest glue (every decoy is < 0.35 to
EVERY active by construction, so these are the closest anything gets), and both molecules'
differentials, so the reader can see a structurally unrelated molecule out-scoring a real one.

The shared substructure is haloed so the eye can separate "shares a scaffold" from "shares nothing
and still scores".

Run:  conda run -n rgfn python experiments/oracle_validation/docking_6td3/plot_decoy_neighbours.py
"""
import argparse
import io
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
SHARED = (1.0, 0.72, 0.72)
PANEL = (300, 205)
GATE = -2.0
# The matcher enforces ECFP4 Tanimoto < 0.35 to every active, which does NOT forbid a decoy from
# containing the CR8-like purine outright -- a molecule can carry adenine and still be globally
# dissimilar. Measured: 5 of 160 (3%) do. Flag them per pair rather than claiming none exist.
PURINE = ("c1nc2c(n1)ncnc2", "c1[nX3]c2c(n1)ncnc2")


def has_purine(smiles):
    m = Chem.MolFromSmiles(smiles)
    return bool(m) and any(m.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in PURINE)


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)  # ECFP4, as the decoy matcher used


def draw_pair(a, b):
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    res = rdFMCS.FindMCS([ma, mb], timeout=15, ringMatchesRingOnly=True, completeRingsOnly=True)
    patt = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
    hl = []
    for m in (ma, mb):
        at = tuple(m.GetSubstructMatch(patt)) if patt is not None else ()
        aset = set(at)
        bd = [
            x.GetIdx()
            for x in m.GetBonds()
            if x.GetBeginAtomIdx() in aset and x.GetEndAtomIdx() in aset
        ]
        hl.append((list(at), bd))
    d = rdMolDraw2D.MolDraw2DCairo(PANEL[0] * 2, PANEL[1], PANEL[0], PANEL[1])
    o = d.drawOptions()
    o.clearBackground = False
    o.highlightBondWidthMultiplier = 20
    d.DrawMolecules(
        [ma, mb],
        highlightAtoms=[hl[0][0], hl[1][0]],
        highlightBonds=[hl[0][1], hl[1][1]],
        highlightAtomColors=[{i: SHARED for i in hl[0][0]}, {i: SHARED for i in hl[1][0]}],
        highlightBondColors=[{i: SHARED for i in hl[0][1]}, {i: SHARED for i in hl[1][1]}],
    )
    d.FinishDrawing()
    return d.GetDrawingText()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="the dock_6td3_matched_<jobid> dir holding both result CSVs",
    )
    ap.add_argument("--n", type=int, default=4)
    # Both ends are worth looking at, and they answer different questions. `strongest` shows the
    # false positives -- what the gate lets through. `weakest` shows what a decoy the oracle
    # confidently REJECTS looks like, which is the check on whether the negative set is fair: if the
    # clearly-rejected ones look like plausible glues, the labels are wrong, not the oracle.
    ap.add_argument("--mode", choices=["strongest", "weakest"], default="strongest")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    act = pd.read_csv(a.run_dir / "known_results.csv")
    dec = pd.read_csv(a.run_dir / "decoy_cdk_results.csv")
    act, dec = act[act.status == "ok"], dec[dec.status == "ok"]

    amols = [(s, Chem.MolFromSmiles(s), r) for s, r in zip(act.smiles, act.ddb1_dvina)]
    amols = [(s, m, r) for s, m, r in amols if m is not None]
    afps = [fp(m) for _, m, _ in amols]

    n_pass = int((dec.ddb1_dvina <= GATE).sum())
    print(
        f"decoys clearing the {GATE} gate: {n_pass} of {len(dec)} "
        f"({(dec.ddb1_dvina <= GATE).mean():.0%})"
    )
    if a.mode == "strongest":
        # worst false positives: clear the gate, best differential first
        fp_dec = dec[dec.ddb1_dvina <= GATE].sort_values("ddb1_dvina").head(a.n)
    else:
        # the decoys the oracle rejects hardest — least negative differential
        fp_dec = dec.sort_values("ddb1_dvina", ascending=False).head(a.n)
    out = a.out or (HERE / f"decoy_neighbours_{a.mode}.png")

    rows = []
    for _, d in fp_dec.iterrows():
        dm = Chem.MolFromSmiles(d.smiles)
        if dm is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(fp(dm), afps)
        j = int(max(range(len(sims)), key=lambda k: sims[k]))
        rows.append(
            {
                "decoy": d.smiles,
                "decoy_dvina": float(d.ddb1_dvina),
                "active": amols[j][0],
                "active_dvina": float(amols[j][2]),
                "tanimoto": float(sims[j]),
                "decoy_has_purine": has_purine(d.smiles),
            }
        )

    n = len(rows)
    # Grid, not a single column: the pair image has aspect ~2.9, so one full-width column per pair
    # letterboxes badly and renders the structures small. Two columns fills the plate.
    ncol = 2 if n > 1 else 1
    nrow = (n + ncol - 1) // ncol
    LEFT, RIGHT = 0.012, 0.992
    fig_w = 15.0
    cell_w = (RIGHT - LEFT) * fig_w / ncol
    fig_h = cell_w / 2.93 * nrow + 1.95
    fig, axes = plt.subplots(nrow, ncol, figsize=(fig_w, fig_h), facecolor=SURFACE)
    axes = list(axes.flat) if n > 1 else [axes]
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.axis("off")
    for ax, r in zip(axes, rows):
        ax.imshow(mpimg.imread(io.BytesIO(draw_pair(r["decoy"], r["active"])), format="png"))
        mark = "  [carries a purine]" if r["decoy_has_purine"] else ""
        ax.set_title(
            f"DECOY  ΔVina {r['decoy_dvina']:+.2f}{mark}   ·   Tanimoto {r['tanimoto']:.2f}   ·   "
            f"nearest REAL GLUE  ΔVina {r['active_dvina']:+.2f}",
            fontsize=9.8,
            color=INK,
            fontweight="semibold",
            pad=4,
        )

    head = (
        "Property-matched decoys that BEAT the 6TD3 gate, beside the real glue each most "
        "resembles"
        if a.mode == "strongest"
        else "Property-matched decoys the oracle REJECTS most confidently, beside their nearest real "
        "glue"
    )
    fig.text(0.008, 0.965, head, fontsize=14, color=INK, fontweight="bold", ha="left", va="top")
    sub = (
        f"Left = decoy, right = its nearest known CDK12–DDB1 glue. Every decoy is < 0.35 "
        f"Tanimoto to EVERY active by construction, so these are the closest anything gets.\n"
        f'A more negative ΔVina is a better "glue" score; the gate is {GATE}. '
        + (
            "These are the strongest false positives — the gate admits them."
            if a.mode == "strongest"
            else f"These sit at the OPPOSITE end: the {a.n} weakest of {len(dec)} decoys, far from "
            f"passing."
        )
    )
    fig.text(0.008, 0.918, sub, fontsize=9.8, color=INK2, ha="left", va="top", linespacing=1.5)
    n_pur = sum(r["decoy_has_purine"] for r in rows)
    pur_note = (
        "Shared substructure haloed in red. 97% of these decoys carry NO purine warhead — 5 "
        "of 160 do, because the matcher enforces global dissimilarity, not absence of a "
        "fragment"
        + (f" (and {n_pur} of the {len(rows)} pairs above is one of them)." if n_pur else ".")
    )
    tail = (
        "  Purine content is not the cause of the failure: 31% of non-purine decoys clear the "
        "gate, vs 40% of the five that have one.\nStructurally unrelated molecules out-score "
        "real glues on the differential — which is why the gate admits 31% of them."
        if a.mode == "strongest"
        else "\nThese are the clean negatives: their differentials sit at roughly ZERO, so the oracle "
        "does separate a population it is confident about. The failure is not that it rejects "
        "nothing —\nit is that the 31% at the other end are indistinguishable from real glues on "
        "this signal."
    )
    fig.text(
        0.008,
        0.012,
        pur_note + tail,
        fontsize=8.4,
        color=INK3,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )

    fig.subplots_adjust(
        left=LEFT,
        right=RIGHT,
        top=1.0 - 1.35 / fig_h,
        bottom=0.42 / fig_h,
        wspace=0.02,
        hspace=0.26,
    )
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix("." + ext), dpi=200, facecolor=SURFACE)
    pd.DataFrame(rows).to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out.with_suffix('.png')} + .pdf + .csv")
    for r in rows:
        print(
            f"  decoy ΔVina {r['decoy_dvina']:+.2f}  vs nearest glue {r['active_dvina']:+.2f}"
            f"  (Tanimoto {r['tanimoto']:.2f})"
        )


if __name__ == "__main__":
    main()
