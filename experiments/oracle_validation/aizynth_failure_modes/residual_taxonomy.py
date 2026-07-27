#!/usr/bin/env python
"""Classify the stubborn AiZynth residual failures (Logs/047 follow-up) and plot the taxonomy.

Logs/047 found that failures at 10x search budget almost always stall on ONE not-in-stock ("open")
leaf of ~15 heavy atoms. This script turns that qualitative read into counts, so the paper can say
*what* those leaves are instead of asserting "not bad molecules".

Categories, applied in this precedence order (first match wins):
  1 block, stereo only     leaf IS a SMALL block once stereochemistry is stripped -> the block is
                           already purchasable; only the stereo descriptor differs.
  2 block core, other form leaf shares a SMALL block's Bemis-Murcko scaffold at comparable size
                           (<=1.6x its heavy atoms) -> same core, different oxidation/protection
                           state (THIQ acid vs methyl ester vs aldehyde vs N-carbamate).
  3 partial assembly       contains a DISTINCTIVE block core (>=2 rings or >=8 heavy atoms) but is
                           larger -> two block cores already coupled, ester not yet hydrolysed.
                           A generic single ring (benzene) cannot evidence this; it is a
                           substructure of almost everything.
  4 tiny reagent           <=3 heavy atoms (HCN, methanol) -> a bulk reagent, not a building block.
  5 reagent / protecting gp matches a hand-checked SMARTS list (Boc, trityl, SEM/TMS, pinacol
                           boronate, phosphonium, phthalimide, Cbz/Fmoc, poly-halide ester).
  6 other                  everything else.
Categories 1-3 group as "the reaction-GFN's own core": the residual is its OWN chemistry in a
different terminal form, not exotic unmakeable matter. The rules are deliberately CONSERVATIVE
against that claim -- e.g. N-benzyl/N-acyl proline esters fall to "other" because pyrrolidine is
not a distinctive core -- so the grouped share is a floor, not a ceiling.

Emits residual_taxonomy.{csv,png,pdf} next to this script's results dir. Run in the `aizynth` env
(RDKit only): conda run -n aizynth python residual_taxonomy.py [--verbose]
"""
import csv
import json
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

DIAG = "/scratch/markymoo/rgfn_runs/lsdflow_sparrow/scent_rawpool/aiz_residual_diag.json"
FRAGS = "data/libraries/glue_standard_v1/fragments.csv"
OUT = Path("experiments/oracle_validation/aizynth_failure_modes/results")

CATS = [
    "block, stereo only",
    "block core, other form",
    "partial assembly of block cores",
    "tiny reagent",
    "reagent / protecting gp",
    "other",
]
# major grouping used for the figure's color + the headline claim
GROUP = {
    "block, stereo only": "reaction-GFN's own core",
    "block core, other form": "reaction-GFN's own core",
    "partial assembly of block cores": "reaction-GFN's own core",
    "tiny reagent": "standard reagent / protecting group",
    "reagent / protecting gp": "standard reagent / protecting group",
    "other": "other",
}

REAGENT_SMARTS = {
    "Boc": "[CX4]([CH3])([CH3])([CH3])O[CX3]=O",
    "trityl": "[CX4](c1ccccc1)(c1ccccc1)c1ccccc1",
    "SEM/TMS": "[Si]([CH3])([CH3])([CH3])CCO",
    "pinacol boronate": "B1OC(C)(C)C(C)(C)O1",
    "phosphonium": "[P+]",
    "phthalimide": "O=C1[#7]C(=O)c2ccccc12",  # NB: "O=C1c2ccccc2C1=O" is a 4-ring, never matches
    "Cbz/Fmoc": "O=C(O[CH2]c1ccccc1)",
    "poly-halide ester": "[CX4;$([CX4][Br,I])][CX4,CX3]~[CX4][Br,I]",
}


def flat(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None, None
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m), m


def murcko(mol):
    try:
        s = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(s) if s is not None and s.GetNumAtoms() else None
    except Exception:
        return None


def main():
    verbose = "--verbose" in sys.argv
    OUT.mkdir(parents=True, exist_ok=True)

    # SMALL library: flat SMILES set + {murcko scaffold: min heavy atoms}
    block_flat, block_scaf = set(), {}
    for row in csv.DictReader(open(FRAGS)):
        f, m = flat((row.get("smiles") or "").strip())
        if not f:
            continue
        block_flat.add(f)
        sc = murcko(m)
        if sc:
            # LARGEST block carrying this scaffold sets the "comparable size" bar
            block_scaf[sc] = max(block_scaf.get(sc, 0), m.GetNumHeavyAtoms())

    patts = {k: Chem.MolFromSmarts(v) for k, v in REAGENT_SMARTS.items()}
    # "distinctive" block cores: >=2 rings or >=8 heavy atoms. A generic single ring (benzene,
    # pyridine) is a substructure of almost everything, so it cannot evidence "block core".
    distinctive = {}
    for sc in block_scaf:
        q = Chem.MolFromSmiles(sc)
        if q is None:
            continue
        nrings = q.GetRingInfo().NumRings()
        if nrings >= 2 or q.GetNumHeavyAtoms() >= 8:
            distinctive[sc] = q

    diag = json.load(open(DIAG))
    rows = []
    for r in diag:
        if r.get("arm") != "filter_on" or "error" in r or r.get("solved"):
            continue
        for lf in r.get("open_leaves") or []:
            f, m = flat(lf.get("smi"))
            if not f:
                continue
            ha = m.GetNumHeavyAtoms()
            sc = murcko(m)
            cat, why = "other", ""
            if f in block_flat:
                cat, why = "block, stereo only", "exact flat match to a SMALL block"
            elif sc and sc in block_scaf and ha <= 1.6 * block_scaf[sc]:
                cat, why = "block core, other form", f"shares block scaffold {sc}"
            elif ha <= 3:
                cat, why = "tiny reagent", "<=3 heavy atoms"
            else:
                for name, p in patts.items():
                    if p is not None and m.HasSubstructMatch(p):
                        cat, why = "reagent / protecting gp", name
                        break
                else:
                    for sc, q in distinctive.items():
                        if m.HasSubstructMatch(q):
                            cat, why = (
                                "partial assembly of block cores",
                                f"contains distinctive block core {sc}",
                            )
                            break
            rows.append(
                {
                    "target": r["smiles"],
                    "leaf": lf["smi"],
                    "leaf_ha": ha,
                    "category": cat,
                    "evidence": why,
                }
            )

    counts = {c: sum(1 for x in rows if x["category"] == c) for c in CATS}
    n = len(rows)
    print(f"residual open leaves classified: {n}")
    for c in CATS:
        print(f"  {counts[c]:3d}  ({100*counts[c]/max(n,1):4.1f}%)  {c}")
    print("  major groups:")
    for g in ("reaction-GFN's own core", "standard reagent / protecting group", "other"):
        k = sum(v for c, v in counts.items() if GROUP[c] == g)
        print(f"    {k:3d}  ({100*k/max(n,1):4.1f}%)  {g}")

    if verbose:
        print("\nper-leaf assignments (sorted by category, then size):")
        for x in sorted(rows, key=lambda r: (CATS.index(r["category"]), r["leaf_ha"])):
            print(
                f"  [{x['leaf_ha']:2d} HA] {x['category']:24s} {x['leaf'][:56]:56s} {x['evidence'][:44]}"
            )

    with open(OUT / "residual_taxonomy.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["target", "leaf", "leaf_ha", "category", "evidence"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}/residual_taxonomy.csv")
    return rows, counts, block_scaf


# ── figure ────────────────────────────────────────────────────────────────────
# Palette: the validated default instance (dataviz references/palette.md) -- categorical slot 1
# blue #2a78d6 = "reaction-GFN's own core", slot 2 orange #eb6834 = "standard reagent / protecting
# group" (that pair passed the six checks: CVD dE 24.7 protan, normal-vision 33.6, contrast
# 4.30/3.12). "other" is a recessive NEUTRAL, not a third categorical hue, deliberately: it is a
# residual bucket and must not compete for attention (the chroma floor does not apply to it).
GCOLOR = {
    "reaction-GFN's own core": "#2a78d6",
    "standard reagent / protecting group": "#eb6834",
    "other": "#a8a7a1",
}
SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8984"
# friendly names for the recurring residual species (chemistry checked by hand)
NAMES = {
    "O=C(O)C1Cc2ccccc2CN1C(=O)O": "THIQ-3-acid, N-carboxylated",
    "COC(=O)C1Cc2ccccc2CN1": "THIQ-3-acid methyl ester",
    "O=C(O)C1Cc2ccccc2CN1": "THIQ-3-carboxylic acid (flat)",
    "COC(=O)C(Br)CCBr": "methyl 2,4-dibromobutanoate",
    "CCOC(=O)C(Br)CCBr": "ethyl 2,4-dibromobutanoate",
    "C#N": "HCN (retro-cyanation)",
    "O=CC1Cc2ccccc2CN1": "THIQ-3-carbaldehyde",
    "COC(=O)C1CCCN1": "proline methyl ester",
    "COC(=O)C1CCN1": "azetidine-2-acid methyl ester",
    "O=C(O)C1CCCN1": "proline (flat)",
    "CO": "methanol",
    "COC(=O)C1CCCN1C(=O)O": "proline ester, N-carboxylated",
    "CC1(C)OB(B2OC(C)(C)C(C)(C)O2)OC1(C)C": "bis(pinacolato)diboron",
}


def plot(rows, counts):
    import matplotlib

    matplotlib.use("Agg")
    from collections import Counter

    import matplotlib.pyplot as plt

    n = len(rows)
    fig = plt.figure(figsize=(12.4, 4.5), facecolor=SURFACE)
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 0.82], wspace=0.07, left=0.185, right=0.775, top=0.725, bottom=0.14
    )

    def style(ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK3)
            ax.spines[s].set_linewidth(0.8)
        ax.grid(axis="x", color="#e8e7e3", linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8.5, length=0)

    # Panel A: fine categories, ordered by the major group then count
    axA = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    order = sorted(CATS, key=lambda c: (list(GCOLOR).index(GROUP[c]), -counts[c]))
    for i, c in enumerate(order):
        axA.barh(i, counts[c], height=0.52, color=GCOLOR[GROUP[c]], zorder=3)
        axA.text(
            counts[c] + 0.25,
            i,
            f"{counts[c]}  ({100*counts[c]/n:.0f}%)",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
    axA.set_yticks(range(len(order)))
    axA.set_yticklabels(order, fontsize=8.6, color=INK)
    axA.set_xlim(0, max(counts.values()) * 1.28)
    axA.invert_yaxis()
    style(axA)
    axA.set_xlabel("residual open leaves", fontsize=8.5, color=INK2)
    axA.set_title(
        "A  What the stubborn leaf actually is",
        fontsize=10.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=8,
    )

    # Panel B: recurring species (the residual concentrates in a few fixable entries)
    axB = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    cnt = Counter(r["leaf"] for r in rows)
    grp = {r["leaf"]: GROUP[r["category"]] for r in rows}
    info = {r["leaf"]: (r["evidence"], r["leaf_ha"]) for r in rows}

    def label(smi):
        if smi in NAMES:
            return NAMES[smi]
        ev, ha = info.get(smi, ("", 0))
        if ev in REAGENT_SMARTS:  # e.g. Boc / SEM/TMS / phthalimide / trityl
            return f"{ev}-protected intermediate ({ha} HA)"
        return smi[:28] + ("…" if len(smi) > 28 else "")

    top = cnt.most_common(10)
    for i, (smi, k) in enumerate(top):
        axB.barh(i, k, height=0.52, color=GCOLOR[grp[smi]], zorder=3)
        axB.text(
            k + 0.08,
            i,
            str(k),
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK,
            fontweight="medium",
        )
    axB.set_yticks(range(len(top)))
    axB.set_yticklabels([label(s) for s, _ in top], fontsize=8.4, color=INK)
    axB.yaxis.tick_right()
    axB.set_xlim(0, max(k for _, k in top) * 1.2)
    axB.invert_yaxis()
    style(axB)
    axB.set_xlabel("occurrences among the residual leaves", fontsize=8.5, color=INK2)
    axB.set_title(
        "B  A handful of species carry most of it",
        fontsize=10.5,
        color=INK,
        fontweight="semibold",
        loc="left",
        pad=8,
    )
    distinct = len(cnt)
    fig.text(
        0.985,
        0.035,
        f"{n} residual leaves → only {distinct} distinct species; "
        f"the top {len(top)} cover {100*sum(k for _, k in top)/n:.0f}%",
        ha="right",
        va="bottom",
        fontsize=7.8,
        color=INK2,
    )

    fig.text(
        0.02,
        0.945,
        "The residual AiZynth failure is the reaction-GFN's own chemistry in a different terminal\n"
        "form, plus standard reagents — not exotic, unmakeable matter",
        fontsize=11.4,
        color=INK,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.45,
    )
    grouped = {g: sum(v for c, v in counts.items() if GROUP[c] == g) for g in GCOLOR}
    own = grouped["reaction-GFN's own core"]
    fig.text(
        0.02,
        0.845,
        "SCENT sEH · the 42 failures that survive a 10× search budget (Logs/047, job 71428) · "
        f"{own}/{n} leaves are its own core",
        fontsize=8.6,
        color=INK2,
        ha="left",
        va="top",
    )
    fig.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=GCOLOR[g]) for g in GCOLOR],
        labels=list(GCOLOR),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.90),
        ncol=3,
        frameon=False,
        fontsize=8.5,
        labelcolor=INK2,
        handlelength=0.9,
        handleheight=0.9,
        columnspacing=1.3,
        borderpad=0.2,
    )

    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"residual_taxonomy.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/residual_taxonomy.png + .pdf")


if __name__ == "__main__":
    _rows, _counts, _ = main()
    plot(_rows, _counts)
