#!/usr/bin/env python
"""Visual sanity check on what "distinct" actually means: the closest and most distant PAIRS in
each method's R=100 library.

A Tanimoto number is hard to feel. 0.55 sounds low; two molecules differing only in a halogen also
sound "different" until you see them side by side. This renders, per method, the three most similar
pairs and the three most dissimilar pairs from the library each method delivers at the SAME fixed
100-reaction budget -- so the reader can judge for themselves whether "82 distinct" and "24 distinct"
are measuring something real.

Emits TWO figures, because the two blocks answer different questions and are read separately:
  pair_gallery_closest.*  -- the three most similar pairs per method (its WORST case: how redundant
                             does the library get? this is the one the mode count is about)
  pair_gallery_distant.*  -- the three least similar pairs per method (its BEST case: how far apart
                             can it reach at all?)
Rows = method, columns = the three pairs, in both.

Structures are drawn with RDKit; this is a molecular illustration rather than a data chart, so it
uses the project's ink/surface tokens for consistency but no categorical palette.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/make_pair_gallery.py
"""
import json
from pathlib import Path

import matplotlib
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Draw import rdMolDraw2D

matplotlib.use("Agg")
import io

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

SCRATCH = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb")
SP = Path(
    "/tmp/claude-3160042/-home-markymoo-projects-RGFN-Fork-RGFN-Fork/"
    "a36a13bf-3840-4727-9bb8-995235e664cb/scratchpad"
)
OUT = Path("experiments/lsd_hubs/campaign/results/paper_pair_gallery")
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
# Shared substructure is haloed rather than recoloured: the halo sits BEHIND the bonds, so the
# structure stays legible in black and the eye still reads "this part is common to both".
SHARED = (1.0, 0.72, 0.72)
PANEL = (300, 205)  # per-molecule draw size; a pair panel is twice as wide
ASPECT = (PANEL[0] * 2) / PANEL[1]
N_PAIRS = 3

METHODS = [
    ("hub-batching (ours)", "json", SP / "hb_r100.json"),
    ("BC-Enum-SB (λ=1)", "milp", SCRATCH / "enumR100_L1_N50000/milp_L1_R100.json"),
    ("BC-SB (λ=1)", "milp", SCRATCH / "bcsbR100_seed43_L1_N21000/milp_L1_R100.json"),
]


def load(kind, path):
    d = json.loads(Path(path).read_text())
    return d if kind == "json" else (d.get("selected_target_smiles") or [])


def extreme_pairs(smis, n, closest=True):
    """The n most similar (or most distant) DISTINCT pairs, on the project's ECFP recipe."""
    mols = [(s, Chem.MolFromSmiles(s)) for s in smis]
    mols = [(s, m) for s, m in mols if m is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048) for _, m in mols]
    pairs = []
    for i in range(len(fps)):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1 :])
        for j, sv in enumerate(sims, start=i + 1):
            pairs.append((sv, mols[i][0], mols[j][0]))
    pairs.sort(key=lambda t: t[0], reverse=closest)
    # keep pairs that do not reuse a molecule, so three pairs show six molecules
    seen, out = set(), []
    for sv, a, b in pairs:
        if a in seen or b in seen:
            continue
        out.append((sv, a, b))
        seen.update((a, b))
        if len(out) == n:
            break
    return out


_MCS_CACHE = {}


def shared_core(ma, mb):
    """Atom/bond indices of the maximum common substructure, per molecule.

    Rings must match rings and be matched whole, so the "shared" region is a chemically meaningful
    scaffold rather than an arbitrary maximal atom mapping that clips a ring in half. Returns
    ((atoms_a, bonds_a), (atoms_b, bonds_b), n_shared_atoms).
    """
    key = (Chem.MolToSmiles(ma), Chem.MolToSmiles(mb))
    if key in _MCS_CACHE:
        return _MCS_CACHE[key]
    res = rdFMCS.FindMCS([ma, mb], timeout=20, ringMatchesRingOnly=True, completeRingsOnly=True)
    patt = Chem.MolFromSmarts(res.smartsString) if res.smartsString else None
    out = []
    for m in (ma, mb):
        atoms = tuple(m.GetSubstructMatch(patt)) if patt is not None else ()
        aset = set(atoms)
        bonds = [
            bd.GetIdx()
            for bd in m.GetBonds()
            if bd.GetBeginAtomIdx() in aset and bd.GetEndAtomIdx() in aset
        ]
        out.append((list(atoms), bonds))
    val = (out[0], out[1], len(out[0][0]))
    _MCS_CACHE[key] = val
    return val


def draw_pair(a, b):
    """The two members of a pair side by side, shared substructure haloed in red.

    Returns (png_bytes, n_shared_atoms, frac_shared) -- the fractions go in the CSV so the picture
    and the numbers come from the same MCS.
    """
    ma, mb = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
    (aa, ab), (ba, bb), n_shared = shared_core(ma, mb)
    d = rdMolDraw2D.MolDraw2DCairo(PANEL[0] * 2, PANEL[1], PANEL[0], PANEL[1])
    o = d.drawOptions()
    o.clearBackground = False
    o.highlightBondWidthMultiplier = 20
    d.DrawMolecules(
        [ma, mb],
        highlightAtoms=[aa, ba],
        highlightBonds=[ab, bb],
        highlightAtomColors=[{i: SHARED for i in aa}, {i: SHARED for i in ba}],
        highlightBondColors=[{i: SHARED for i in ab}, {i: SHARED for i in bb}],
    )
    d.FinishDrawing()
    denom = min(ma.GetNumHeavyAtoms(), mb.GetNumHeavyAtoms()) or 1
    return d.GetDrawingText(), n_shared, n_shared / denom


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data, summary = {}, []
    for label, kind, path in METHODS:
        if not Path(path).exists():
            print(f"  SKIP {label} (missing {path})")
            continue
        smis = load(kind, path)
        data[label] = {
            "n": len(smis),
            "closest": extreme_pairs(smis, N_PAIRS, closest=True),
            "distant": extreme_pairs(smis, N_PAIRS, closest=False),
        }
        c = [p[0] for p in data[label]["closest"]]
        f = [p[0] for p in data[label]["distant"]]
        summary.append((label, len(smis), c, f))
        print(
            f"  {label:<22} n={len(smis):>3}  closest={[f'{x:.2f}' for x in c]}  "
            f"most distant={[f'{x:.2f}' for x in f]}"
        )

    nrow = len(data)
    shared = {}
    for block, heading, note in (
        (
            "closest",
            "THREE CLOSEST PAIRS  —  each method's WORST case",
            "The pairs the metric is really about: how redundant does each library get? A pair above 0.5 "
            'would be merged into ONE\n"distinct molecule" by the mode metric, so our worst case sits at '
            "the 0.50 ceiling the selection enforces, while the competitors tolerate 0.83-0.90.",
        ),
        (
            "distant",
            "THREE MOST DISTANT PAIRS  —  each method's BEST case",
            "How far apart can each method reach at all? All three arms get to comparably distant pairs, "
            "so the libraries do not differ in\nreach -- they differ in the redundancy they tolerate, "
            "which is what the companion figure shows.",
        ),
    ):
        # Panels are 2-up grid images of aspect ~3:1 and imshow preserves aspect, so the row height
        # has to track the panel width or the figure fills with letterbox whitespace.
        panel_w = (1.0 - 0.155 - 0.008) * 11.5 / 3
        fig, axes = plt.subplots(
            nrow, N_PAIRS, figsize=(11.5, panel_w / ASPECT * nrow + 1.65), facecolor=SURFACE
        )
        if nrow == 1:
            axes = [axes]
        for r, (label, d) in enumerate(data.items()):
            for c in range(N_PAIRS):
                ax = axes[r][c]
                ax.set_facecolor(SURFACE)
                ax.axis("off")
                if c >= len(d[block]):
                    continue
                sv, a, b = d[block][c]
                png, n_sh, fr_sh = draw_pair(a, b)
                shared[(label, block, c)] = (n_sh, fr_sh)
                ax.imshow(mpimg.imread(io.BytesIO(png), format="png"))
                ax.set_title(
                    f"T {sv:.2f}   ·   {fr_sh:.0%} shared",
                    fontsize=10,
                    color=INK,
                    fontweight="semibold",
                    pad=3,
                )
            axes[r][0].text(
                -0.06,
                0.5,
                f"{label}\n({d['n']} molecules)",
                transform=axes[r][0].transAxes,
                ha="right",
                va="center",
                fontsize=10.5,
                color=INK,
                fontweight="semibold",
            )

        fig.text(
            0.5, 0.995, heading, ha="center", va="top", fontsize=12, color=INK, fontweight="bold"
        )
        fig.text(
            0.5,
            0.952,
            "all three libraries built at the SAME fixed 100-reaction budget  ·  "
            "T = Tanimoto on the mode metric (Morgan r=3 / 2048)\n"
            "the shared core (MCS) is haloed in red — what stays black is what actually differs",
            ha="center",
            va="top",
            fontsize=8.6,
            color=INK3,
            linespacing=1.6,
        )
        fig.text(
            0.014, 0.012, note, fontsize=8.2, color=INK2, ha="left", va="bottom", linespacing=1.5
        )
        top = 1.0 - 1.02 / fig.get_figheight()
        bot = 0.50 / fig.get_figheight()
        fig.subplots_adjust(left=0.155, right=0.992, top=top, bottom=bot, wspace=0.03, hspace=0.30)
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"pair_gallery_{block}.{ext}", dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {OUT}/pair_gallery_{block}.png + .pdf")

    with open(OUT / "pair_gallery.csv", "w") as fh:
        fh.write(
            "method,n_molecules,block,rank,tanimoto,mcs_shared_atoms,"
            "mcs_shared_frac,smiles_a,smiles_b\n"
        )
        for label, d in data.items():
            for block in ("closest", "distant"):
                for i, (sv, a, b) in enumerate(d[block], 1):
                    n_sh, fr_sh = shared.get((label, block, i - 1), (0, 0.0))
                    fh.write(
                        f'"{label}",{d["n"]},{block},{i},{sv:.4f},{n_sh},'
                        f'{fr_sh:.4f},"{a}","{b}"\n'
                    )
    print(f"wrote {OUT}/pair_gallery.csv")


if __name__ == "__main__":
    main()
