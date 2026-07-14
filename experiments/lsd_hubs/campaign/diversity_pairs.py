#!/usr/bin/env python
"""Diversity gallery: the closest still-distinct pair at each cutoff (Logs/032).

A visual, intuition-building companion to the hub-batching campaign (`029`/`031`). For every
Tanimoto diversity cutoff in the sweep, we rebuild the **hub-batching** library exactly as the
campaign does — walk the pre-ranked, pre-enumerated hubs, feed their children best-reward-first
through :class:`DiverseThresholdModeSelector` (reward gate + pairwise Morgan-r3/2048 Tanimoto),
keep the accepted modes to the mode budget — and then find, among those accepted modes, the
**pair with the highest pairwise Tanimoto similarity**. By construction every accepted pair sits
at or below the cutoff (the selector rejects only *strictly* above it), so this "tightest" pair is
the two most similar molecules the cutoff still counts as distinct hits. Drawn side by side (shared
maximum-common-substructure aligned + highlighted), it gives an at-a-glance feel for what a given
diversity cutoff actually *means* structurally: at 0.30 the closest allowed pair looks clearly
different; at 0.90 it looks nearly identical.

Only ``enum_children.json`` is needed — acceptance depends solely on reward threshold, reward
ordering and the Tanimoto cutoff, so the cost table / recipe snapshot (which only fill reaction
accounting, never *which* molecules are modes) are deliberately omitted.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/diversity_pairs.py \
        --enum-children /scratch/.../campaign_enum_seh_70363/enum_children.json \
        --reward-threshold 7.0 --tag scent_seh_1kx200

Writes ``results/<tag>/diversity_pairs/{gallery.png,pairs.csv}``.
"""
import argparse
import csv
from io import BytesIO
from pathlib import Path

from run_campaign import _load_enumerated_hubs  # same-dir helper (reused loader)

from glue.samplers.lsdflow.campaign import HubBatchingStrategy
from glue.samplers.lsdflow.mode_select import ecfp

HERE = Path(__file__).resolve().parent


def _cutoff_grid(lo: float, hi: float, step: float):
    n = int(round((hi - lo) / step)) + 1
    return [round(lo + i * step, 4) for i in range(n)]


def _build_library(hubs, cutoff, reward_threshold, higher_is_better, budget_modes):
    """The hub-batching accepted modes at one cutoff (cost_table=None: acceptance is reward+Tanimoto
    only, so cost never changes *which* molecules are kept)."""
    result = HubBatchingStrategy(
        hubs,
        None,
        target="seh",
        reward_threshold=reward_threshold,
        similarity=cutoff,
        higher_is_better=higher_is_better,
    ).run(budget=("modes", budget_modes))
    return result


def _tightest_pair(points):
    """Among accepted modes, the pair with the highest pairwise Tanimoto (Morgan r=3/2048, the
    campaign fingerprint). Returns (similarity, point_a, point_b) or (nan, None, None)."""
    from rdkit import DataStructs

    fps = [(p, ecfp(p.smiles)) for p in points]
    fps = [(p, f) for p, f in fps if f is not None]
    best = (float("-inf"), None, None)
    for i in range(len(fps)):
        pi, fi = fps[i]
        for j in range(i + 1, len(fps)):
            t = DataStructs.TanimotoSimilarity(fi, fps[j][1])
            if t > best[0]:
                best = (t, pi, fps[j][0])
    return best


# ------------------------------------------------------------------ drawing
def _aligned_pair(smiles_a, smiles_b):
    """Parse both molecules, align B onto A over their maximum common substructure, and return the
    MCS atom indices to highlight in each (empty lists if no usable MCS). Aligning + highlighting the
    shared core is what makes 'how similar are these two' readable at a glance."""
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor, rdFMCS

    mol_a, mol_b = Chem.MolFromSmiles(smiles_a), Chem.MolFromSmiles(smiles_b)
    if mol_a is None or mol_b is None:
        return mol_a, mol_b, [], []
    AllChem.Compute2DCoords(mol_a)
    AllChem.Compute2DCoords(mol_b)
    hi_a, hi_b = [], []
    try:
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            timeout=10,
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
        )
        if mcs.numAtoms >= 4 and not mcs.canceled:
            patt = Chem.MolFromSmarts(mcs.smartsString)
            match_a, match_b = mol_a.GetSubstructMatch(patt), mol_b.GetSubstructMatch(patt)
            if match_a and match_b:
                hi_a, hi_b = list(match_a), list(match_b)
                try:  # lay B out so its shared core overlays A's
                    rdDepictor.GenerateDepictionMatching2DStructure(mol_b, mol_a, refPatt=patt)
                except Exception:
                    pass
    except Exception:
        pass
    return mol_a, mol_b, hi_a, hi_b


def _mol_image(mol, highlight, size=(360, 300)):
    import numpy as np
    from PIL import Image
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    drawer.drawOptions().padding = 0.12
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol, highlightAtoms=highlight or [])
    drawer.FinishDrawing()
    return np.array(Image.open(BytesIO(drawer.GetDrawingText())))


def _render_gallery(rows, out_png, tag, reward_threshold, budget_modes):
    """One row per cutoff (ascending, so pairs get more similar downward): label | mol A | mol B."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(rows)
    header_in = 1.6  # inches reserved at the top for the multi-line title
    fig_h = 2.55 * n + header_in
    fig = plt.figure(figsize=(9.0, fig_h))
    gs = GridSpec(
        n,
        3,
        width_ratios=[0.9, 1, 1],
        hspace=0.30,
        wspace=0.02,
        top=1 - header_in / fig_h,
        bottom=0.01,
        left=0.02,
        right=0.99,
    )
    for i, row in enumerate(rows):
        mol_a, mol_b, hi_a, hi_b = _aligned_pair(row["smiles_a"], row["smiles_b"])
        ax_lab = fig.add_subplot(gs[i, 0])
        ax_lab.axis("off")
        ax_lab.text(
            0.5,
            0.5,
            f"cutoff {row['cutoff']:.2f}\n\nmax Tanimoto\n{row['tanimoto']:.3f}\n\n"
            f"{row['n_modes']} modes",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )
        for col, (mol, hi, rew) in enumerate(
            ((mol_a, hi_a, row["reward_a"]), (mol_b, hi_b, row["reward_b"])), start=1
        ):
            ax = fig.add_subplot(gs[i, col])
            ax.axis("off")
            if mol is not None:
                ax.imshow(_mol_image(mol, hi))
            ax.set_title(f"sEH {rew:.2f}", fontsize=9)
    fig.text(
        0.5,
        1 - 0.28 / fig_h,
        f"SCENT sEH hub-batching ({tag})\n"
        f"closest still-distinct pair at each diversity cutoff\n"
        f"Morgan r=3/2048 Tanimoto · both molecules count as distinct modes\n"
        f"(reward ≥ {reward_threshold:g}, {budget_modes}-mode budget) · shared MCS core highlighted",
        ha="center",
        va="top",
        fontsize=11,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[pairs] wrote {out_png}")


def _write_csv(rows, out_csv):
    cols = [
        "cutoff",
        "tanimoto",
        "n_modes",
        "smiles_a",
        "reward_a",
        "hub_a",
        "smiles_b",
        "reward_b",
        "hub_b",
    ]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in cols})
    print(f"[pairs] wrote {out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--enum-children", required=True, help="enum_children.json from the scent worker"
    )
    ap.add_argument("--reward-threshold", type=float, required=True, help="hit bar (sEH ~7.0)")
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--budget-modes", type=int, default=300, help="library size (campaign target)")
    ap.add_argument("--cutoff-min", type=float, default=0.30)
    ap.add_argument("--cutoff-max", type=float, default=0.90)
    ap.add_argument("--cutoff-step", type=float, default=0.05)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    hubs = _load_enumerated_hubs(
        Path(a.enum_children), comps={}
    )  # comps only feed cost (unused here)
    cutoffs = _cutoff_grid(a.cutoff_min, a.cutoff_max, a.cutoff_step)
    print(f"[pairs] {len(hubs)} hubs; cutoffs={cutoffs}")

    rows = []
    for cut in cutoffs:
        result = _build_library(hubs, cut, a.reward_threshold, a.higher_is_better, a.budget_modes)
        sim, pa, pb = _tightest_pair(result.accepted)
        if pa is None:
            print(f"  cutoff {cut:.2f}: {len(result.accepted)} modes, no drawable pair — skipped")
            continue
        rows.append(
            {
                "cutoff": cut,
                "tanimoto": round(sim, 4),
                "n_modes": len(result.accepted),
                "smiles_a": pa.smiles,
                "reward_a": round(pa.reward, 3),
                "hub_a": pa.source_hub or "",
                "smiles_b": pb.smiles,
                "reward_b": round(pb.reward, 3),
                "hub_b": pb.source_hub or "",
            }
        )
        print(
            f"  cutoff {cut:.2f}: {len(result.accepted)} modes, "
            f"closest pair Tanimoto={sim:.3f} (stop={result.stop_reason})"
        )

    out = HERE / "results" / a.tag / "diversity_pairs"
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, out / "pairs.csv")
    _render_gallery(rows, out / "gallery.png", a.tag, a.reward_threshold, a.budget_modes)
    print(f"\n[pairs] {len(rows)} cutoff pairs -> {out}")


if __name__ == "__main__":
    main()
