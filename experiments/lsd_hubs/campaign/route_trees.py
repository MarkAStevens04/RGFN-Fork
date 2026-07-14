#!/usr/bin/env python
"""Synthesis-route trees for the diversity pairs (Logs/032 addendum).

Companion to ``diversity_pairs.py`` (schematic; the per-reaction detail lives in
``synthesis_routes.py`` → ``synthesis_protocol.md``). For each closest-pair, draw a little tree: the
structure the two molecules **share**, then where they **diverge**. This makes the hub-batching
overlap concrete at a glance — the shared piece and the branch point.

What's grounded in the logs (nothing inferred): every molecule the campaign kept is a one-reaction
**diversification of a hub** scaffold (from ``enum_children.json``: its ``hub`` + the promoted
dynamic-library fragment attached in the final reaction, ``added_promoted``). Each such promoted
fragment carries a **logged synthesis route** in the recipe snapshot (``smiles_to_route``: a ``seed``
+ reaction ``steps`` whose reactants may themselves be promoted fragments → recursive). We take the
two final fragments of a pair, expand their routes, and split the fragments/intermediates into
**shared** (in both routes) vs **divergent** (one only), and draw the most-elaborated shared piece.
Same-hub pairs share building blocks (and sometimes a built intermediate); different-hub pairs share
nothing. NOTE: the count here is shared route *pieces* (building blocks + intermediates), not shared
*reactions* — ``synthesis_protocol.md`` gives the exact per-reaction shared-vs-diverging split.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/route_trees.py \
        --pairs experiments/lsd_hubs/campaign/results/scent_seh_1kx200/diversity_pairs/pairs.csv \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
        --tag scent_seh_1kx200

Writes ``results/<tag>/diversity_pairs/{route_trees.png,route_trees.csv}``.
"""
import argparse
import csv
import json
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent


# ------------------------------------------------------------------ route model
def route_nodes(frag, routes, promoted_set):
    """Every fragment/reactant in ``frag``'s logged route DAG (itself + seeds + step reactants,
    recursively through promoted sub-fragments). These are the synthesis 'steps' of ``frag``."""
    nodes, stack = set(), [frag]
    while stack:
        f = stack.pop()
        if f in nodes:
            continue
        nodes.add(f)
        r = routes.get(f)
        if r is None:
            continue
        parts = ([r["seed"]] if r.get("seed") else []) + [
            x for step in r.get("steps", []) for x in step.get("reactants", [])
        ]
        for p in parts:
            if p not in nodes:
                stack.append(p)
    return nodes


def _heavy(smiles):
    from rdkit import Chem

    m = Chem.MolFromSmiles(smiles)
    return m.GetNumHeavyAtoms() if m else 0


def shared_intermediate(shared, routes):
    """The most-built shared node to display: prefer an assembled intermediate (has a route),
    deepest first; else the largest shared building block. None if nothing shared."""
    if not shared:
        return None, None
    assembled = [s for s in shared if s in routes]
    if assembled:
        best = max(assembled, key=lambda s: (routes[s].get("num_reactions", 0), _heavy(s)))
        return best, "shared intermediate"
    return max(shared, key=_heavy), "shared building block"


# ------------------------------------------------------------------ drawing
def _mol_image(smiles, size=(300, 250), highlight_smarts=None):
    import numpy as np
    from PIL import Image
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.Compute2DCoords(mol)
    hi = []
    if highlight_smarts:
        q = Chem.MolFromSmiles(highlight_smarts)
        if q is not None:
            hi = list(mol.GetSubstructMatch(q))  # empty tuple if the reaction changed the atoms
    d = rdMolDraw2D.MolDraw2DCairo(*size)
    d.drawOptions().padding = 0.1
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=hi)
    d.FinishDrawing()
    return np.array(Image.open(BytesIO(d.GetDrawingText())))


def _panel(fig, gs, row, pair, tag_colors):
    """One pair's route tree: shared node (left) --fork--> product A / product B (right)."""

    r0 = 2 * row
    ax_share = fig.add_subplot(gs[r0 : r0 + 2, 0])
    ax_share.axis("off")
    ax_fork = fig.add_subplot(gs[r0 : r0 + 2, 1])
    ax_fork.axis("off")
    ax_a = fig.add_subplot(gs[r0, 2])
    ax_a.axis("off")
    ax_b = fig.add_subplot(gs[r0 + 1, 2])
    ax_b.axis("off")

    # shared node (or a note, for different-hub pairs)
    if pair["shared_smiles"]:
        img = _mol_image(pair["shared_smiles"])
        if img is not None:
            ax_share.imshow(img)
        ax_share.set_title(
            f"{pair['shared_label']}\n(shared)", fontsize=9, color=tag_colors["share"]
        )
    else:
        ax_share.text(
            0.5,
            0.5,
            "different hubs\n— no shared\nintermediate —",
            ha="center",
            va="center",
            fontsize=10,
            color=tag_colors["diverge"],
            style="italic",
        )

    # products, shared substructure highlighted where the reaction preserved it
    for ax, key in ((ax_a, "a"), (ax_b, "b")):
        img = _mol_image(pair[f"smiles_{key}"], highlight_smarts=pair["shared_smiles"] or None)
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"Molecule {key.upper()} · sEH {pair[f'reward_{key}']}", fontsize=9)

    # fork: a green shared stem (built once) splitting into two orange diverging branches
    have_stem = bool(pair["shared_smiles"])
    split = 0.46
    if have_stem:
        ax_fork.plot(
            [0.04, split], [0.5, 0.5], color=tag_colors["share"], lw=3, solid_capstyle="round"
        )
        ax_fork.text(
            0.25,
            0.60,
            "shared\nstructure",
            ha="center",
            va="bottom",
            fontsize=8,
            color=tag_colors["share"],
        )
    for y in (0.82, 0.18):
        ax_fork.annotate(
            "",
            xy=(0.99, y),
            xytext=(split, 0.5),
            arrowprops=dict(arrowstyle="-|>", lw=2.4, color=tag_colors["diverge"]),
        )
    ax_fork.text(
        0.75,
        0.60,
        "diverging\nfinal step",
        ha="center",
        va="bottom",
        fontsize=8,
        color=tag_colors["diverge"],
    )
    ax_fork.text(
        0.5,
        0.985,
        f"cutoff {pair['cutoff']}  ·  Tanimoto {pair['tanimoto']}",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        transform=ax_fork.transAxes,
    )
    note = (
        f"same hub ({pair['hub_a']}); shares early structure, then diverges "
        f"· exact reactions in synthesis_protocol.md"
        if have_stem
        else f"different hubs ({pair['hub_a']} vs {pair['hub_b']}) — built independently"
    )
    ax_fork.text(
        0.5,
        0.02,
        note,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#6b6b6b",
        transform=ax_fork.transAxes,
        wrap=True,
    )
    ax_fork.set_xlim(0, 1)
    ax_fork.set_ylim(0, 1)


def render(pairs, out_png, tag):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(pairs)
    colors = {"share": "#0d7d6f", "diverge": "#c65a1e"}
    header_in = 1.3
    fig_h = 2.5 * n + header_in
    fig = plt.figure(figsize=(11.0, fig_h))
    gs = GridSpec(
        2 * n,
        3,
        width_ratios=[1.05, 0.95, 1.15],
        height_ratios=[1] * (2 * n),
        hspace=0.55,
        wspace=0.05,
        top=1 - header_in / fig_h,
        bottom=0.01,
        left=0.01,
        right=0.99,
    )
    for i, p in enumerate(pairs):
        _panel(fig, gs, i, p, colors)
    fig.text(
        0.5,
        1 - 0.25 / fig_h,
        f"SCENT sEH hub-batching ({tag}) — shared-structure schematic for the closest pairs\n"
        "shared building block / intermediate (green) → diverging branches (orange) → the two molecules\n"
        "green highlight marks the shared piece still intact in each product · exact reactions in synthesis_protocol.md",
        ha="center",
        va="top",
        fontsize=12,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[routes] wrote {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pairs", required=True, help="pairs.csv from diversity_pairs.py")
    ap.add_argument("--enum-children", required=True)
    ap.add_argument("--snapshot", required=True, help="fragments_<N>.json with smiles_to_route")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.pairs)))
    want = set(r["smiles_a"] for r in rows) | set(r["smiles_b"] for r in rows)

    snap = json.load(open(a.snapshot))
    routes = snap.get("smiles_to_route") or {}
    promoted_set = set(snap.get("chosen_smiles", []))
    print(f"[routes] {len(routes)} logged routes, {len(promoted_set)} promoted fragments")

    print("[routes] indexing enum_children (hub + final fragment per molecule) ...")
    enum = json.load(open(a.enum_children))
    final_frag = {}  # smiles -> (hub, depth, added_promoted[0] or None)
    for h in enum["hubs"]:
        for c in h["children"]:
            s = c["smiles"]
            if s in want and s not in final_frag:
                added = c.get("added_promoted", ())
                final_frag[s] = (h["hub_key"], int(h["depth"]), added[0] if added else None)

    pairs, csv_rows = [], []
    for r in rows:
        hub_a, _, fa = final_frag.get(r["smiles_a"], ("?", 0, None))
        hub_b, _, fb = final_frag.get(r["smiles_b"], ("?", 0, None))
        na = route_nodes(fa, routes, promoted_set) if fa else set()
        nb = route_nodes(fb, routes, promoted_set) if fb else set()
        shared = (na & nb) - {
            fa,
            fb,
        }  # shared building steps (exclude the two final fragments themselves)
        same_hub = hub_a == hub_b and hub_a != "?"
        shared_for_pair = shared if same_hub else set()
        smi, label = shared_intermediate(shared_for_pair, routes)
        pairs.append(
            {
                "cutoff": f"{float(r['cutoff']):.2f}",
                "tanimoto": r["tanimoto"],
                "smiles_a": r["smiles_a"],
                "smiles_b": r["smiles_b"],
                "reward_a": r["reward_a"],
                "reward_b": r["reward_b"],
                "hub_a": hub_a,
                "hub_b": hub_b,
                "shared_smiles": smi or "",
                "shared_label": label or "",
                "n_shared": len(shared_for_pair),
                "n_only_a": len(na - nb - {fa}),
                "n_only_b": len(nb - na - {fb}),
            }
        )
        csv_rows.append(
            {
                "cutoff": r["cutoff"],
                "tanimoto": r["tanimoto"],
                "same_hub": same_hub,
                "hub_a": hub_a,
                "hub_b": hub_b,
                "frag_a": fa or "",
                "frag_b": fb or "",
                "n_shared_steps": len(shared_for_pair),
                "shared_drawn": smi or "",
                "shared_all": ";".join(sorted(shared_for_pair)),
            }
        )
        print(
            f"  cutoff {r['cutoff']}: same_hub={same_hub} shared_steps={len(shared_for_pair)} "
            f"draw='{label}'"
        )

    out = HERE / "results" / a.tag / "diversity_pairs"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "route_trees.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"[routes] wrote {out / 'route_trees.csv'}")
    render(pairs, out / "route_trees.png", a.tag)
    print(f"[routes] {len(pairs)} route trees -> {out}")


if __name__ == "__main__":
    main()
