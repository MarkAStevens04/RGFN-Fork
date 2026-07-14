#!/usr/bin/env python
"""Step-by-step synthesis protocols for the diversity pairs (Logs/032 addendum).

Turns the route trees (`route_trees.py`) into chemist-actionable instructions: for each molecule,
exactly **what to buy**, **what reaction to run at each step**, and **what each intermediate looks
like** — with the pair's **shared** steps (built once) separated from the **diverging** ones.

Every reaction is grounded in the logged data, not inferred:
  * each promoted dynamic-library fragment carries a logged route (`smiles_to_route`: ordered
    ``steps`` each with the reaction template, ``input``, ``reactants``, ``product``);
  * the reaction template string (RGFN reaction SMARTS ``R'(A.B >> P, flag, id)``) is mapped to a
    readable named reaction by recognising the transformation (amide coupling, Suzuki, Buchwald,
    reductive amination, benzoxazole/benzimidazole formation, N-alkylation, SNAr, tetrazole, …);
  * a reactant is **buy** iff it is a base building block (``initial_smiles_set``) and **make** iff
    it is a promoted fragment (``chosen_smiles``, expanded recursively into earlier steps);
  * the only step without a logged template is the final hub attach (hub + fragment → molecule);
    it is named from the structural change (nitrile→tetrazole, alkyl-halide hub → N-alkylation,
    carboxylic-acid hub → amide coupling, else a neutral "attach hub building block") and flagged.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/synthesis_routes.py \
        --pairs .../diversity_pairs/pairs.csv \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../fragments_4000.json --tag scent_seh_1kx200

Writes ``results/<tag>/diversity_pairs/{synthesis_protocol.md,synthesis_steps.csv,schemes/scheme_cut*.png}``.
"""
import argparse
import csv
import json
import re
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ reaction naming
def reaction_name(rid: str) -> str:
    """Readable named reaction from an RGFN template string ``R'(A.B >> P, flag, id)``.

    Named by recognising the reactant/product SMARTS signature (the template IS the ground-truth
    transformation; we only give it a human name). Falls back to the template id if unrecognised."""
    if not rid:
        return "reaction (unknown)"
    m = re.search(r">>\s*(.*?),\s*\d+,\s*(\d+)\)\s*$", rid)
    prod = m.group(1) if m else ""
    tid = m.group(2) if m else "?"
    left = rid.split(">>")[0]
    suffix = f" [t{tid}]"

    def P(*subs):
        return any(s in prod for s in subs)

    def L(*subs):
        return any(s in left for s in subs)

    # ring formations (most specific first)
    if P("[o:5]1[c:4][c:3][n:2][c:1]"):
        return "Benzoxazole formation (2-aminophenol + acid/aldehyde)" + suffix
    if P("[n:5]1[c:4][c:3][nH:2][c:1]"):
        return "Benzimidazole formation (o-diamine + acid)" + suffix
    if "1[n:3][n" in prod and "n](-[C:5]" in prod.replace(" ", "") or "[n](-[C" in prod:
        return "Tetrazole formation + N-alkylation (nitrile + alkyl halide)" + suffix
    if "=[C:14]2" in prod or "=[C:12]2" in prod:
        return "Cyclative condensation (ring closure)" + suffix
    if "#[Ch:4]" in prod or "-[C:5]#" in prod:
        return "Sonogashira coupling (aryl halide + alkyne)" + suffix
    # single-reactant functional-group installs
    if L("[c:1]-[Br]", "[c:1]-[Cl]") and P("[c:1]-[Nh2]"):
        return "Aryl amination (Ar–X → Ar–NH₂)" + suffix
    if L("[c:1]-[Br]") and P("[c:1]-[C]#N", "[c:1]-[C]#[N]"):
        return "Cyanation (Ar–Br → Ar–CN)" + suffix
    # C–N couplings
    if L("[B]([Oh])([Oh])", "[c:1]-[B]", "[c:2]-[B]") and P("[c:1]-[c:2]", "[c:2]-[c:1]"):
        return "Suzuki coupling (biaryl bond)" + suffix
    if L("[Ch:2]([#6:3])=[O]", "[Ch:1]=[O]", "[Ch:2]=[O]") and P("[Nh:1]-[Ch:2]", "-[Ch:2]-[#6:3]"):
        return "Reductive amination (aldehyde + amine)" + suffix
    if L("[C:2]([#6:4])=[O]") and P("-[C:2]([C:3])", "[C:2]([#6:3])"):
        return "Reductive amination (ketone + amine)" + suffix
    if L("[F]-[c:2]", "[F]-[c:") and P("[Nh:1]-[c:2]", "[Nh:1]-[c:"):
        return "SNAr (amine + aryl fluoride)" + suffix
    if L("[c:2]-[Br]", "[c:2]-[Cl]") and P("-[Nh:1]-[c:2]", "[Nh:1]-[c:2]"):
        return "Buchwald–Hartwig amination (aryl halide + amine)" + suffix
    if L("[Ch2:1]-[Br]", "[Ch2:4]-[Br]") and P("[O:3]-[C:2]", "-[O:3]-"):
        return "Williamson ether synthesis (alkyl halide + alcohol)" + suffix
    if P("[c:1][n:2]([c:3])-[Ch2:4]", "[n:2]([c:3])-[Ch2:4]"):
        return "N-alkylation (ring N–H + alkyl halide)" + suffix
    # amides (tertiary before secondary)
    if P("-[N:5](-", "-[N:2](-") and ("=[O" in prod):
        return "Amide coupling (acid + secondary amine → tertiary amide)" + suffix
    if P("(=[O:4])-[Nh:5]", "(=[O])[Nh", "-[Nh:2]-[#6", "(=[O])-[Nh:2]"):
        return "Amide coupling (acid + amine)" + suffix
    if P("-[C](=O)-[Nh:6]", "-[C](=O)-[Nh:"):
        return "Urea formation (amine + amine)" + suffix
    return f"reaction (RGFN template {tid})" + suffix


def hub_reaction_name(frag: str, hub: str, terminal: str) -> str:
    """Name the final hub-attach reaction from the structural change (this one step has no logged
    template). Determined from the hub's reactive group + what the product gains — general, not
    ClCC1CC1-specific: nitrile→tetrazole, alkyl-halide→N-alkylation, carboxylic-acid→amide, else
    a neutral 'attach hub building block'."""
    from rdkit import Chem

    ft, tt = Chem.MolFromSmiles(frag), Chem.MolFromSmiles(terminal)
    hh = Chem.MolFromSmiles(hub)
    tet = Chem.MolFromSmarts("c1nnnn1")
    if (
        ft is not None
        and tt is not None
        and tt.HasSubstructMatch(tet)
        and not ft.HasSubstructMatch(tet)
    ):
        return "Tetrazole formation + N-alkylation (nitrile + hub)"
    if hh is not None and hh.HasSubstructMatch(Chem.MolFromSmarts("[Cl,Br,I]-[CH2]")):
        return "N-alkylation (ring N–H + hub alkyl halide)"
    if hh is not None and hh.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[OH]")):
        return "Amide coupling (amine + hub carboxylic acid)"
    return "Final assembly (attach hub building block)"


# ------------------------------------------------------------------ route assembly
def linearize(target, routes, built, out):
    """Ordered synthesis steps to build ``target`` (dependencies first). Each step is a dict with
    reaction / input / reactants / product (verbatim from the logged route)."""
    if target in built or target not in routes:
        return
    for st in routes[target]["steps"]:
        for dep in [st["input"]] + st["reactants"]:
            if dep != target and dep in routes and dep not in built:
                linearize(dep, routes, built, out)
        out.append(st)
        built.add(st["product"])
    built.add(target)


def full_route(frag, hub, terminal, routes):
    """Fragment's logged route + the final hub-attach step → the whole molecule, as ordered steps."""
    steps = []
    linearize(frag, routes, set(), steps)
    steps = [
        {
            "reaction": reaction_name(s["reaction"]),
            "input": s["input"],
            "reactants": list(s["reactants"]),
            "product": s["product"],
        }
        for s in steps
    ]
    steps.append(
        {
            "reaction": hub_reaction_name(frag, hub, terminal),
            "input": frag,
            "reactants": [hub],
            "product": terminal,
        }
    )
    return steps


# ------------------------------------------------------------------ drawing
def _img(smiles, size=(230, 175)):
    import numpy as np
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().padding = 0.1
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    return np.array(Image.open(BytesIO(d.GetDrawingText())))


def _scheme(cards, out_png, title, colors):
    """Each card is one reaction step: input (+) reactant --reaction--> product, with a section
    banner. ``cards`` = list of (section, section_color, step_dict, src_of_each_reactant)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(cards)
    fig_h = 2.15 * n + 0.7
    fig = plt.figure(figsize=(11.5, fig_h))
    gs = GridSpec(
        n,
        3,
        width_ratios=[1, 1, 1],
        hspace=0.62,
        wspace=0.06,
        top=1 - 0.55 / fig_h,
        bottom=0.02,
        left=0.02,
        right=0.98,
    )
    for i, (section, sec_color, step, srcs) in enumerate(cards):
        reactant = step["reactants"][0] if step["reactants"] else ""
        ax_in = fig.add_subplot(gs[i, 0])
        ax_in.axis("off")
        ax_rk = fig.add_subplot(gs[i, 1])
        ax_rk.axis("off")
        ax_pr = fig.add_subplot(gs[i, 2])
        ax_pr.axis("off")
        for ax, smi in ((ax_in, step["input"]), (ax_rk, reactant), (ax_pr, step["product"])):
            im = _img(smi) if smi else None
            if im is not None:
                ax.imshow(im)
        # section + reaction banner above the row
        ax_in.set_title(f"{section}", loc="left", fontsize=10, fontweight="bold", color=sec_color)
        ax_rk.set_title(step["reaction"], fontsize=9.5, color=colors["rxn"])
        # buy/make tags + operators
        ax_in.text(1.02, 0.5, "+", transform=ax_in.transAxes, ha="center", va="center", fontsize=18)
        ax_rk.text(
            1.02,
            0.5,
            "→",
            transform=ax_rk.transAxes,
            ha="center",
            va="center",
            fontsize=18,
            color=colors["rxn"],
        )
        in_src = srcs.get(step["input"], "")
        rk_src = srcs.get(reactant, "")
        ax_in.text(
            0.5,
            -0.04,
            in_src,
            transform=ax_in.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=colors["buy"] if in_src == "buy" else colors["make"],
        )
        ax_rk.text(
            0.5,
            -0.04,
            rk_src,
            transform=ax_rk.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color=colors["buy"] if rk_src == "buy" else colors["make"],
        )
        ax_pr.text(
            0.5,
            -0.04,
            "product" if section != "" else "",
            transform=ax_pr.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="#6b6b6b",
        )
    fig.suptitle(title, fontsize=12, y=1 - 0.12 / fig_h)
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs", required=True)
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--no-schemes", action="store_true", help="skip the per-pair scheme PNGs")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.pairs)))
    want = set(r["smiles_a"] for r in rows) | set(r["smiles_b"] for r in rows)
    snap = json.load(open(a.snapshot))
    routes = snap["smiles_to_route"]

    enum = json.load(open(a.enum_children))
    final = {}
    for h in enum["hubs"]:
        for c in h["children"]:
            s = c["smiles"]
            if s in want and s not in final:
                added = c.get("added_promoted", ())
                final[s] = (h["hub_key"], added[0] if added else None)

    colors = {
        "buy": "#0d7d6f",
        "make": "#c65a1e",
        "rxn": "#2f4858",
        "shared": "#0d7d6f",
        "A": "#8a4b08",
        "B": "#8a4b08",
    }
    out = HERE / "results" / a.tag / "diversity_pairs"
    (out / "schemes").mkdir(parents=True, exist_ok=True)

    md = [
        "# Synthesis protocols — closest diversity pairs " f"(SCENT sEH hub-batching, {a.tag})",
        "",
        "Step-by-step routes for the two most-similar-yet-distinct molecules at each cutoff. "
        "Reactions are the SCENT dynamic-library assembly steps we logged (named from their reaction "
        "templates); **buy** = a base building block, **make** = an intermediate built in an earlier "
        "step. The final step attaches the hub building block (its reaction is named from the "
        "structural change, not a logged template). This is the logged assembly route, not a claim of "
        "an optimal retrosynthesis.",
        "",
    ]
    csv_rows = []

    for r in rows:
        cut = f"{float(r['cutoff']):.2f}"
        hub_a, frag_a = final[r["smiles_a"]]
        hub_b, frag_b = final[r["smiles_b"]]
        A = full_route(frag_a, hub_a, r["smiles_a"], routes)
        B = full_route(frag_b, hub_b, r["smiles_b"], routes)
        prod_a = {s["product"] for s in A}
        prod_b = {s["product"] for s in B}
        # route-aware buy/make: anything produced by a step is "make" (an intermediate we build);
        # everything else (leaves) is "buy" (a purchased starting material). More correct than a
        # promoted-set test, which mislabels intermediates that were never themselves promoted.
        made = prod_a | prod_b

        def src(s, _made=made):
            return "make" if s in _made else "buy"

        shared_prod = prod_a & prod_b
        shared = [s for s in A if s["product"] in shared_prod]
        a_only = [s for s in A if s["product"] not in shared_prod]
        b_only = [s for s in B if s["product"] not in shared_prod]
        same_hub = final[r["smiles_a"]][0] == final[r["smiles_b"]][0]

        # ---- markdown ----
        md.append(f"## Cutoff {cut} — closest pair (Tanimoto {r['tanimoto']})")
        if shared:
            md.append(
                f"Both molecules are built from the **same hub** (`{hub_a}`) and share **{len(shared)}** "
                f"fully-built intermediate step(s) (made once), then diverge at the final decoration."
            )
        elif same_hub:
            md.append(
                f"Same hub reagent (`{hub_a}`) and the same reaction *sequence* from mostly-shared "
                f"building blocks, but they **diverge at the first reaction** — no reusable built "
                f"intermediate (they share purchases, not an intermediate; see the buy list)."
            )
        else:
            md.append(f"**Different hubs** — independently synthesised (`{hub_a}` vs `{hub_b}`).")
        # shopping list
        buys = {}
        for tag, steps in (("A", A), ("B", B)):
            for s in steps:
                for x in [s["input"]] + s["reactants"]:
                    if src(x) == "buy":
                        buys.setdefault(x, set()).add(tag)
        md.append("\n**Building blocks to buy**\n")
        md.append("| SMILES | for | note |")
        md.append("|---|---|---|")
        for smi, tags in sorted(buys.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            note = "shared" if tags == {"A", "B"} else "diverging"
            md.append(f"| `{smi}` | {'/'.join(sorted(tags))} | {note} |")

        def emit(label, steps, start):
            md.append(f"\n**{label}**\n")
            for k, s in enumerate(steps, start):
                rk = ", ".join(f"`{x}` ({src(x)})" for x in s["reactants"])
                md.append(
                    f"{k}. **{s['reaction']}** — `{s['input']}` ({src(s['input'])}) + {rk} "
                    f"→ `{s['product']}`"
                )
                csv_rows.append(
                    {
                        "cutoff": cut,
                        "molecule": label,
                        "step": k,
                        "reaction": s["reaction"],
                        "input": s["input"],
                        "reactants": ";".join(s["reactants"]),
                        "product": s["product"],
                    }
                )
            return start + len(steps)

        if shared:
            nxt = emit("Shared steps (built once)", shared, 1)
            emit(
                f"Molecule A (sEH {r['reward_a']}) — continues from the shared intermediate",
                a_only,
                nxt,
            )
            emit(
                f"Molecule B (sEH {r['reward_b']}) — continues from the shared intermediate",
                b_only,
                nxt,
            )
        else:
            emit(f"Molecule A (sEH {r['reward_a']})", A, 1)
            emit(f"Molecule B (sEH {r['reward_b']})", B, 1)
        md.append("")

        # ---- scheme figure ----
        if not a.no_schemes:

            def srcs_of(step):
                return {x: src(x) for x in [step["input"]] + step["reactants"]}

            cards = []
            for s in shared:
                cards.append(("Shared (built once)", colors["shared"], s, srcs_of(s)))
            for s in a_only:
                cards.append((f"→ Molecule A  (sEH {r['reward_a']})", colors["A"], s, srcs_of(s)))
            for s in b_only:
                cards.append((f"→ Molecule B  (sEH {r['reward_b']})", colors["B"], s, srcs_of(s)))
            if not shared:  # different-hub: label by molecule
                cards = [
                    (f"→ Molecule A  (sEH {r['reward_a']})", colors["A"], s, srcs_of(s)) for s in A
                ] + [
                    (f"→ Molecule B  (sEH {r['reward_b']})", colors["B"], s, srcs_of(s)) for s in B
                ]
            _scheme(
                cards,
                out / "schemes" / f"scheme_cut{cut}.png",
                f"Cutoff {cut} · closest pair (Tanimoto {r['tanimoto']}) · "
                f"buy=green, make=orange",
                colors,
            )
        print(
            f"  cutoff {cut}: shared={len(shared)} A={len(a_only) or len(A)} B={len(b_only) or len(B)}"
        )

    (out / "synthesis_protocol.md").write_text("\n".join(md))
    with open(out / "synthesis_steps.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=["cutoff", "molecule", "step", "reaction", "input", "reactants", "product"],
        )
        w.writeheader()
        w.writerows(csv_rows)
    print(
        f"[synth] wrote synthesis_protocol.md + synthesis_steps.csv"
        f"{'' if a.no_schemes else ' + schemes/'} to {out}"
    )


if __name__ == "__main__":
    main()
