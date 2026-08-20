#!/usr/bin/env python
"""EXPLORATORY (not a paper figure): the per-molecule similarity-neighbourhood heatmap.

One column per molecule in a library, sorted left-to-right by reward. Each column is that molecule's
histogram of Tanimoto similarity to every OTHER molecule in the same library. So a column answers
"for this one compound, how many of its library-mates sit at 0.1-0.15? at 0.45-0.5?" and the panel
as a whole answers "is this library uniformly spread, or does it have a redundant sub-population?"

Why this is worth looking at: the mode count collapses each library to a single integer, and the pair
gallery shows only the two extremes. This shows the whole distribution at once, per molecule, so a
redundant CLUSTER (a bright band high up, confined to some columns) is distinguishable from uniform
mild similarity (an even band low down) -- two very different libraries that can score the same.

Columns are normalised to percent of the library, because the arms deliver different numbers of
molecules (82 vs 91-98) and raw counts would not be comparable across panels. Each column therefore
sums to 100%.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/explore_similarity_heatmap.py
"""
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.stats import gaussian_kde

SCRATCH = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb")
LSD = Path("/scratch/markymoo/rgfn_runs/lsdflow")
SP = Path(
    "/tmp/claude-3160042/-home-markymoo-projects-RGFN-Fork-RGFN-Fork/"
    "a36a13bf-3840-4727-9bb8-995235e664cb/scratchpad"
)
OUT = Path("experiments/lsd_hubs/campaign/results/similarity_heatmap")
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
TAU = 0.5  # the mode threshold: pairs at or below this count as distinct molecules
W_MARG, W_HEAT = 0.9, 4.0  # width units of the marginal density vs the square heatmap
# How the columns are ordered. "mean_sim" makes the bulk trend monotone so structure separates from
# noise; "nn_sim" orders by the statistic the mode count actually keys on; "reward" was the original
# and looks like static, because reward and similarity are near-uncorrelated in these libraries.
SORT_BY = "mean_sim"  # one of: mean_sim | nn_sim | reward
SORT_LABEL = {
    "mean_sim": "mean similarity to the library",
    "nn_sim": "nearest-neighbour similarity",
    "reward": "reward",
}
BIN_W = 0.05
BINS = np.arange(0.0, 1.0 + 1e-9, BIN_W)

# All five are read at the SAME fixed 100-reaction budget.
# One arm throughout, so the only thing changing across the competitor panels is the diversity
# weight. lambda=10 is the TOP of the sweep that exists for BC-Enum-SB at R=100 -- {0, 0.1, 1, 10}
# were solved and lambda=100 was not -- and since lambda=10 already lands on the same 24 distinct
# molecules as lambda=1, the sweep has visibly saturated well before the top of the range.
PANELS = [
    ("hub-batching (ours)", "json", SP / "hb_r100.json"),
    ("BC-Enum-SB  λ=0", "milp", SCRATCH / "enumR100_L0_N50000/milp_L0_R100.json"),
    ("BC-Enum-SB  λ=1", "milp", SCRATCH / "enumR100_L1_N50000/milp_L1_R100.json"),
    ("BC-Enum-SB  λ=10", "milp", SCRATCH / "enumR100_L10_N50000/milp_L10_R100.json"),
]


def load_selection(kind, path):
    d = json.loads(Path(path).read_text())
    return d if kind == "json" else (d.get("selected_target_smiles") or [])


def reward_map():
    """SMILES -> reward, unioned over every artifact that carries one.

    No single artifact covers all five libraries: the hub-batching picks come from the big campaign
    enumeration, the BC-Enum-SB picks from the 64-hub enumeration, the native picks from the sampler's
    records. Later sources win only where earlier ones are silent.
    """
    m = {}
    for p in (
        LSD / "campaign_enum_seh_70363/enum_children.json",
        LSD / "bc_enum_seh_seed42/enum/enum_children.json",
    ):
        if p.exists():
            d = json.loads(p.read_text())
            for h in d["hubs"]:
                for c in h["children"]:
                    m.setdefault(c["smiles"], c["reward"])
    rec = LSD / "scent_seh_native_70974/sample/records.csv"
    if rec.exists():
        for r in csv.DictReader(open(rec)):
            m.setdefault(r["child_key"], float(r["reward"]))
    return m


def profile(smis, rewards, sort_by=SORT_BY):
    """Per-molecule similarity histograms, columns ordered by the chosen key.

    The similarity matrix is built in the input order and only then permuted, because two of the
    three sort keys are derived from the matrix itself.
    """
    keep = [(s, Chem.MolFromSmiles(s)) for s in smis]
    keep = [(s, mm) for s, mm in keep if mm is not None]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mm, 3, nBits=2048) for _, mm in keep]
    n = len(fps)
    sim = np.ones((n, n))
    for i in range(n):
        sim[i] = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    np.fill_diagonal(sim, np.nan)  # a molecule is not its own neighbour

    mean_sim = np.nanmean(sim, axis=1)
    nn = np.nanmax(np.where(np.isnan(sim), -np.inf, sim), axis=1)  # closest single twin
    rw = np.array([rewards.get(s, np.nan) for s, _ in keep])

    key = {"mean_sim": mean_sim, "nn_sim": nn, "reward": rw}[sort_by]
    idx = np.argsort(np.where(np.isnan(key), np.inf, key), kind="stable")
    sim = sim[np.ix_(idx, idx)]
    keep = [keep[i] for i in idx]
    mean_sim, nn, rw = mean_sim[idx], nn[idx], rw[idx]

    H = np.zeros((len(BINS) - 1, n))
    for i in range(n):
        v = sim[i][~np.isnan(sim[i])]
        H[:, i] = np.histogram(v, bins=BINS)[0] / max(len(v), 1) * 100.0

    pw = sim[np.triu_indices(n, k=1)]  # every unordered pair once, for the marginal KDE
    # Crowding: for each molecule, what share of the REST of the library sits above tau, averaged
    # over molecules. NaN never compares true, so the diagonal drops out of the count on its own.
    crowd = float(np.mean((sim > TAU).sum(axis=1) / max(n - 1, 1)) * 100.0)
    return dict(
        H=H, rw=rw, nn=nn, mean_sim=mean_sim, order=[s for s, _ in keep], pw=pw, crowd=crowd
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rewards = reward_map()
    print(f"reward map: {len(rewards):,} molecules")

    data = []
    for label, kind, path in PANELS:
        if not Path(path).exists():
            print(f"  SKIP {label} (missing {path})")
            continue
        smis = load_selection(kind, path)
        d = profile(smis, rewards)
        d["label"] = label
        miss = int(np.isnan(d["rw"]).sum())
        data.append(d)
        H, rw, nn = d["H"], d["rw"], d["nn"]
        print(
            f"  {label:<28} n={H.shape[1]:>3}  reward {np.nanmin(rw):.2f}-{np.nanmax(rw):.2f}"
            f"  (missing reward: {miss})  median NN-sim {np.median(nn):.2f}"
            f"  crowding {d['crowd']:.1f}%"
        )

    vmax = max(np.percentile(d["H"], 99.5) for d in data)
    # magma runs black -> purple -> orange -> pale: brighter means MORE, and an empty bin lands on
    # black rather than on the page colour, so absence reads as absence instead of as a hole.
    cmap = plt.get_cmap("magma").copy()
    cmap.set_under(cmap(0.0))
    cmap.set_bad(cmap(0.0))

    ncol = 2
    nrow = int(np.ceil(len(data) / ncol))
    fig = plt.figure(figsize=(11.2, 4.15 * nrow + 2.05), facecolor=SURFACE)
    gs = fig.add_gridspec(
        nrow, ncol, left=0.052, right=0.858, top=0.902, bottom=0.150, wspace=0.30, hspace=0.30
    )

    for i, d in enumerate(data):
        label, H, nn, pw = d["label"], d["H"], d["nn"], d["pw"]
        cell = gs[i // ncol, i % ncol].subgridspec(
            1, 2, width_ratios=[W_MARG, W_HEAT], wspace=0.045
        )
        axm = fig.add_subplot(cell[0, 0])  # marginal density of ALL pairwise similarities
        axh = fig.add_subplot(cell[0, 1], sharey=axm)

        im = axh.imshow(
            H,
            aspect="auto",
            origin="lower",
            cmap=cmap,
            norm=LogNorm(vmin=0.6, vmax=vmax),
            extent=[0, H.shape[1], 0.0, 1.0],
            interpolation="nearest",
        )

        # Marginal: a smooth KDE over every unordered pair, so it is the same population the heatmap
        # bins, just collapsed across molecules. Drawn side-on and mirrored, so its baseline sits
        # against the heatmap the way a seaborn jointplot marginal does.
        grid = np.linspace(0.0, 1.0, 400)
        dens = gaussian_kde(pw)(grid) if len(pw) > 2 and np.ptp(pw) > 0 else np.zeros_like(grid)
        axm.fill_betweenx(grid, 0, dens, color="#6a3d9a", alpha=0.30, lw=0)
        axm.plot(dens, grid, color="#4a2d6e", lw=1.4)
        axm.set_xlim(0, max(dens.max() * 1.08, 1e-9))
        axm.invert_xaxis()
        axm.set_xticks([])
        axm.set_ylabel("Tanimoto similarity", fontsize=8.5, color=INK2)
        axm.set_yticks([0, 0.25, 0.5, 0.75, 1.0])

        for ax in (axm, axh):
            ax.set_facecolor(SURFACE)
            ax.axhline(TAU, color="#1b7f5f", lw=1.4, ls="--", zorder=3)
            ax.tick_params(labelsize=7.5, colors=INK2)
            for sp in ax.spines.values():
                sp.set_color(INK3)
                sp.set_linewidth(0.6)
        axh.tick_params(labelleft=False)
        axh.set_ylim(0.0, 1.0)
        axh.invert_yaxis()  # diverse at the TOP, similar at the bottom
        axm.text(
            0.04,
            TAU - 0.012,
            f"τ = {TAU}",
            transform=axm.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=7,
            color="#1b7f5f",
            fontweight="bold",
            zorder=4,
        )

        # Squares: force the heatmap box square and give the marginal the matching height, or the
        # two drift apart vertically once box_aspect overrides the gridspec cell.
        axh.set_box_aspect(1.0)
        axm.set_box_aspect(W_HEAT / W_MARG)

        # Two stats, both about crowding but at different scales: the NEAREST neighbour (what the
        # mode count actually keys on) and the AVERAGE share of the library above tau (how crowded
        # a molecule's whole neighbourhood is). They can move in opposite directions.
        axh.set_title(
            f"{label}\n{H.shape[1]} mols  ·  med NN {np.median(nn):.2f}  ·  "
            f"{d['crowd']:.1f}% above τ (avg)",
            fontsize=8.6,
            color=INK,
            fontweight="semibold",
            loc="left",
            pad=5,
        )
        axh.set_xlabel(
            f"molecules, sorted by {SORT_LABEL[SORT_BY]} (low → high)", fontsize=8, color=INK2
        )

    # Orientation cue, on the first panel only -- the flipped axis is the one thing a reader can
    # misread, and repeating it on all five would just be noise.
    a0 = fig.axes[0]
    a0.annotate(
        "more diverse",
        xy=(0.72, 0.955),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="top",
        fontsize=7.5,
        color="#1b7f5f",
        fontweight="bold",
    )
    a0.annotate(
        "more similar",
        xy=(0.72, 0.045),
        xycoords="axes fraction",
        rotation=90,
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#8a3d3d",
        fontweight="bold",
    )

    fig.suptitle(
        "How close is each molecule to the rest of its own library?",
        fontsize=13.5,
        color=INK,
        fontweight="bold",
        x=0.012,
        ha="left",
        y=0.986,
    )
    fig.text(
        0.012,
        0.962,
        "one column per molecule  ·  its Tanimoto histogram against every other molecule in the "
        "SAME library  ·  curve at left is the density over all pairs\n"
        "all four libraries built at the same fixed 100-reaction budget",
        fontsize=8.4,
        color=INK3,
        ha="left",
        va="top",
        linespacing=1.5,
    )
    fig.text(
        0.012,
        0.014,
        "Vertical axis is flipped so HIGHER means MORE DIVERSE. Mass below the dashed line is "
        "redundancy the mode metric collapses: a pair above τ=0.5 counts as ONE distinct molecule.\n"
        '"% above τ (avg)" = for each molecule, the share of the REST of the library above τ, '
        "averaged over molecules — how crowded a typical neighbourhood is.\n"
        '"med NN" = the median nearest-neighbour similarity: the closest single twin, which is what '
        "the mode count actually keys on.\n"
        "The two move apart — λ=0→1 takes crowding from 94% to 7% while med NN goes only "
        "0.83 to 0.69.\n"
        "Brighter means more; black is an empty bin; colour is log-scaled so faint bands are real. "
        "Columns are percentages because the arms deliver 82–98 molecules.\n"
        "Columns are sorted by a quantity derived from this same data, so a left-to-right gradient "
        "exists by construction — what is worth reading is the SHAPE of the transition.",
        fontsize=7.6,
        color=INK2,
        ha="left",
        va="bottom",
        linespacing=1.6,
    )

    band_top = max(a.get_position().y1 for a in fig.axes)
    band_bot = min(a.get_position().y0 for a in fig.axes)
    cax = fig.add_axes([0.884, band_bot, 0.013, (band_top - band_bot) * 0.55])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(
        "% of the library at that similarity  (column sums to 100%)", fontsize=7.8, color=INK2
    )
    cb.ax.tick_params(labelsize=7.5, colors=INK2)
    for sp in cb.ax.spines.values():
        sp.set_color(INK3)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"similarity_heatmap.{ext}", dpi=200, facecolor=SURFACE)

    with open(OUT / "nn_similarity.csv", "w") as fh:
        fh.write(
            "method,n_molecules,rank_by_reward,reward,nearest_neighbour_sim,"
            "mean_pct_library_above_tau,smiles\n"
        )
        for d in data:
            for i, sm in enumerate(d["order"]):
                fh.write(
                    f'"{d["label"]}",{d["H"].shape[1]},{i},{d["rw"][i]:.4f},'
                    f'{d["nn"][i]:.4f},{d["crowd"]:.4f},"{sm}"\n'
                )
    print(f"wrote {OUT}/similarity_heatmap.png + .pdf + nn_similarity.csv")


if __name__ == "__main__":
    main()
