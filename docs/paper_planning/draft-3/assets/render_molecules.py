#!/usr/bin/env python
"""Render the schematic molecules for the method diagram.

These are ILLUSTRATIVE, not run outputs: they exist to show the shape of hub-batching in a
slide/Canva diagram. What is real about them is the geometry of the similarity relations,
which is checked by ``verify()`` below against the paper's own metric (Morgan r=3, 2048
bits, tau = 0.50). If an edit breaks the story -- the A1/A2 pair must exceed tau and
everything else must fall under it -- verify() fails loudly rather than letting a diagram
ship that contradicts the text.

Outputs, per molecule:
  svg/           vector, the best thing to hand Canva if it will take SVG
  png/           high-res, transparent background, cropped to the ink
  overview.png   the whole tree in one sheet, for checking the layout before assembling
"""
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")

HERE = Path(__file__).resolve().parent
TAU = 0.50

# One origin -> two hubs -> five candidates. A1/A2 are the near-duplicate pair.
# Deliberately small: 7-18 heavy atoms, against 9-21 in the first version. The two hubs
# differ by acyl SHAPE (a stick against a triangle) so the families read apart at a glance,
# while both keep the whole anilide core so they still read as siblings off one origin.
ORIGIN = "Nc1ccccc1"  # aniline
HUB_A = "CC(=O)Nc1ccccc1"  # acetanilide          -- stick acyl
HUB_B = "O=C(Nc1ccccc1)C1CC1"  # cyclopropyl anilide  -- triangle acyl

MOLS = {
    "00_origin": (ORIGIN, "origin"),
    "01_hubA": (HUB_A, "hub A"),
    "02_hubB": (HUB_B, "hub B"),
    "A1": ("CC(=O)Nc1ccc(C)cc1", "A1"),  # methyl
    "A2": ("CC(=O)Nc1ccc(CC)cc1", "A2  (duplicate of A1)"),  # ethyl -- one bond longer
    "A3": ("CC(=O)Nc1ccc(C2CC2)cc1", "A3"),  # cyclopropyl
    "B1": ("O=C(Nc1ccc(Cl)cc1)C1CC1", "B1"),  # chloro
    "B2": ("O=C(Nc1ccc(N2CCOCC2)cc1)C1CC1", "B2"),  # morpholino
}


def fp(m):
    return AllChem.GetMorganFingerprintAsBitVect(m, 3, nBits=2048)


def verify():
    """Fail loudly if an edit breaks the relations the diagram is meant to show."""
    m = {k: Chem.MolFromSmiles(v[0]) for k, v in MOLS.items()}
    missing = [k for k, v in m.items() if v is None]
    assert not missing, f"unparseable SMILES: {missing}"
    f = {k: fp(v) for k, v in m.items()}

    def t(a, b):
        return DataStructs.TanimotoSimilarity(f[a], f[b])

    checks = [
        ("A1", "A2", "above", "A1/A2 must be the rejected duplicate"),
        ("A1", "A3", "below", "A3 must survive the filter"),
        ("A2", "A3", "below", "A3 must survive the filter"),
        ("B1", "B2", "below", "both of family B must survive"),
        ("A1", "B1", "below", "different hubs must not collide"),
        ("A3", "B2", "below", "different hubs must not collide"),
    ]
    print(f"  similarity checks (Morgan r=3/2048, tau = {TAU:.2f})")
    for a, b, want, why in checks:
        v = t(a, b)
        ok = (v > TAU) if want == "above" else (v < TAU)
        print(f"    {a}-{b:<3} {v:.3f}  {'PASS' if ok else 'FAIL'}   {why}")
        assert ok, f"{a}-{b} = {v:.3f} is not {want} {TAU}: {why}"
    # both hubs must contain the whole origin, or they will not read as siblings
    o = Chem.MolFromSmiles(ORIGIN)
    for h in ("01_hubA", "02_hubB"):
        assert m[h].HasSubstructMatch(o), f"{h} does not contain the origin"
    print(
        f"    hubA-hubB {t('01_hubA','02_hubB'):.3f}  (both contain the origin; acyl shape differs)"
    )
    return m


def draw(mol, path, w, h, svg=False):
    """One molecule, one file. Heavy strokes, no highlight, transparent ground."""
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DSVG(w, h) if svg else rdMolDraw2D.MolDraw2DCairo(w, h)
    o = d.drawOptions()
    o.clearBackground = False  # transparent, so it drops onto any Canva slide
    o.bondLineWidth = max(4, round(w / 135))  # heavy: reads when scaled down in a slide
    o.multipleBondOffset = 0.18
    o.padding = 0.10
    o.fixedFontSize = max(28, round(w / 26))
    o.additionalAtomLabelPadding = 0.10
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    out = d.GetDrawingText()
    path.write_text(out) if svg else path.write_bytes(out)


def trim(path, pad=28):
    """Crop the transparent margin so each file arrives in Canva already tight.

    RDKit centres a molecule in whatever canvas it is given, so a wide molecule in a 4:3
    frame ships with most of the image empty -- which then has to be cropped by hand, eight
    times, in the design tool.
    """
    from PIL import Image

    im = Image.open(path).convert("RGBA")
    bbox = im.split()[-1].getbbox()  # alpha channel -> ink bounding box
    if not bbox:
        return
    l, t, r, b = bbox
    l, t = max(0, l - pad), max(0, t - pad)
    r, b = min(im.width, r + pad), min(im.height, b + pad)
    im.crop((l, t, r, b)).save(path)


def main():
    mols = verify()
    for sub in ("svg", "png"):
        (HERE / sub).mkdir(exist_ok=True)

    print("\n  rendering")
    for key, (smi, _label) in MOLS.items():
        m = mols[key]
        draw(m, HERE / "png" / f"{key}.png", 2400, 1800)
        trim(HERE / "png" / f"{key}.png")
        # Match the SVG canvas to the trimmed PNG's aspect ratio. RDKit bakes the canvas
        # into the SVG viewBox, so a square canvas ships a wide molecule with dead margin
        # that Canva would import as padding.
        from PIL import Image

        w, h = Image.open(HERE / "png" / f"{key}.png").size
        sw = 1400
        draw(m, HERE / "svg" / f"{key}.svg", sw, max(300, round(sw * h / w)), svg=True)
        print(f"    {key:<10} {w}x{h}")

    # one sheet, so the layout can be sanity-checked before assembly
    order = ["00_origin", "01_hubA", "02_hubB", "A1", "A2", "A3", "B1", "B2"]
    grid = Draw.MolsToGridImage(
        [mols[k] for k in order],
        legends=[MOLS[k][1] for k in order],
        molsPerRow=4,
        subImgSize=(520, 400),
        useSVG=False,
    )
    png = grid.data if hasattr(grid, "data") else grid
    (HERE / "overview.png").write_bytes(png) if isinstance(png, bytes) else grid.save(
        HERE / "overview.png"
    )
    print("    overview.png")


if __name__ == "__main__":
    main()
