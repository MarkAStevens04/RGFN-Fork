#!/usr/bin/env python
"""T4.2/T4.3 generality panel: hub-batching vs best-candidate across generators AND targets.

Reads the merged matrix16 per-cell `results/<cell>/summary.json` (the committed count-once campaign
outputs) — every value transcribed, nothing from memory. Emits generality_panel.{csv,png,pdf}.

THE HONESTY POINT THIS FIGURE EXISTS TO MAKE. SCENT's headline rows use `free_frag` + pre-select-K=20
while every baseline uses the naive `reward` policy, so the headline edges (2.60x sEH / 3.17x DRD2) are
NOT an apples-to-apples generator comparison. Under the COMMON naive policy SCENT's edge is 1.22x /
1.38x — and RxnFlow's is 1.23x / 1.60x, i.e. slightly better. Both rows are therefore plotted. The
asymmetry is legitimate rather than unfair (pre-select-K needs promoted dynamic-library fragments,
which only SCENT has — it prebuilds 0 for the others), but the paper must say that instead of implying
flow-selection alone buys 2.6x. Every generator's edge exceeds 1.0, which IS the generality claim.

FragGFN is a CONTROL, not a peer (hatched): its count-once "reactions" are fragment attachments and its
molecules carry no synthesis route (has_route=0), so its edge is a cost-model artifact.

FragGFN MODEL VERSIONS. Logs/046 found our FragGFN ran at max_nodes=9 vs the paper's 6, which collapsed
DRD2. The corrected cap-6 DRD2 cell (job 71856, Logs/050) is now in — and it moves the row a LOT: best
4.68 / hub 1.173 (3.99x) at 300/300 modes, versus cap-9's 8.0 / 3.984 (2.01x) where best-candidate could
only assemble 19 modes. BOTH rows are plotted rather than swapped, because the cap-6 sEH cell is not
trained yet: swapping DRD2 alone would silently mix model versions across the two target blocks. Read
cap-6 as the live control and cap-9 as superseded; drop the cap-9 row once cap-6 sEH lands.

RGFN (upstream reaction-GFN, naive) is now IN (jobs 71981/71982, campaign'd 2026-08-04): sEH 2.993/4.000
(1.34x), DRD2 1.197/3.843 (3.21x) — a third reaction-GFN backbone showing hub-batching wins under the
common naive policy, which is the generality claim's core.

Palette = the validated default instance (blue #2a78d6 hub-batching / orange #eb6834 best-candidate;
six-checks validator: CVD dE 24.7 protan, normal-vision 33.6, contrast 4.30/3.12 - all PASS).

Run:  conda run -n rgfn python experiments/lsd_hubs/matrix16/make_generality_panel.py
"""
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path("experiments/lsd_hubs/matrix16/results")
OUT = RES / "generality_panel"
HUB, BEST = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, INK3 = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"

# (cell dir, display label, target, is_control)
ROWS = [
    ("scent_seh", "SCENT  free-frag+K20", "sEH", False),
    ("scent_seh_naive", "SCENT  naive", "sEH", False),
    ("rxnflow_seh", "RxnFlow  naive", "sEH", False),
    ("rgfn_seh", "RGFN  naive", "sEH", False),
    ("fraggfn_cap9_seh", "FragGFN  naive  (control)", "sEH", True),
    ("scent_drd2", "SCENT  free-frag+K20", "DRD2", False),
    ("scent_drd2_naive", "SCENT  naive", "DRD2", False),
    ("rxnflow_drd2", "RxnFlow  naive", "DRD2", False),
    ("rgfn_drd2", "RGFN  naive", "DRD2", False),
    ("fraggfn_drd2", "FragGFN cap-6  naive  (control)", "DRD2", True),
    ("fraggfn_cap9_drd2", "FragGFN cap-9  naive  (superseded)", "DRD2", True),
]


def load(cell):
    s = json.load(open(RES / cell / "summary.json"))
    return (
        s["hub_batching"]["reactions_per_mode"],
        s["best_candidate"]["reactions_per_mode"],
        s.get("reward_threshold"),
        s.get("child_policy"),
        s.get("prebuild_k"),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = []
    for cell, label, target, ctrl in ROWS:
        if not (RES / cell / "summary.json").exists():
            print(f"  SKIP {cell} (no summary.json)")
            continue
        hub, best, thr, pol, k = load(cell)
        data.append(
            dict(
                cell=cell,
                label=label,
                target=target,
                ctrl=ctrl,
                hub=hub,
                best=best,
                edge=best / hub,
                thr=thr,
                policy=pol,
                k=k,
            )
        )

    with open(OUT / "generality_panel.csv", "w", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "cell",
                "label",
                "target",
                "ctrl",
                "hub",
                "best",
                "edge",
                "thr",
                "policy",
                "k",
            ],
        )
        w.writeheader()
        for d in data:
            w.writerow(d)
    print(f"wrote {OUT}/generality_panel.csv")
    for d in data:
        print(
            f"  {d['target']:<5} {d['label']:<28} hub={d['hub']:.3f} best={d['best']:.3f} "
            f"edge={d['edge']:.2f}x"
        )

    n = len(data)
    fig, ax = plt.subplots(figsize=(9.6, 0.62 * n + 2.5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    H, GAP = 0.30, 0.02
    ymax = max(d["best"] for d in data)
    for i, d in enumerate(data):
        hatch = "//" if d["ctrl"] else None
        ax.barh(
            i - H / 2 - GAP,
            d["hub"],
            height=H,
            color=HUB,
            zorder=3,
            hatch=hatch,
            edgecolor=SURFACE,
            linewidth=0.9,
        )
        ax.barh(
            i + H / 2 + GAP,
            d["best"],
            height=H,
            color=BEST,
            zorder=3,
            hatch=hatch,
            edgecolor=SURFACE,
            linewidth=0.9,
        )
        ax.text(
            d["hub"] + 0.07,
            i - H / 2 - GAP,
            f"{d['hub']:.2f}",
            va="center",
            ha="left",
            fontsize=8.4,
            color=INK,
            fontweight="medium",
        )
        ax.text(
            d["best"] + 0.07,
            i + H / 2 + GAP,
            f"{d['best']:.2f}",
            va="center",
            ha="left",
            fontsize=8.4,
            color=INK,
            fontweight="medium",
        )
        ax.text(
            ymax * 1.20,
            i,
            f"{d['edge']:.2f}×",
            va="center",
            ha="right",
            fontsize=9.2,
            color=INK,
            fontweight="semibold",
        )
    ax.text(ymax * 1.20, -0.85, "edge", va="center", ha="right", fontsize=8.2, color=INK2)
    ax.set_yticks(range(n))
    ax.set_yticklabels([d["label"] for d in data], fontsize=8.6, color=INK)
    ax.set_xlim(0, ymax * 1.22)
    ax.invert_yaxis()
    # target separator between the sEH block and the DRD2 block
    tgts = [d["target"] for d in data]
    for i in range(1, n):
        if tgts[i] != tgts[i - 1]:
            ax.axhline(i - 0.5, color="#dedcd6", linewidth=1.0, zorder=1)
    for t in sorted(set(tgts), key=tgts.index):
        idx = [i for i, d in enumerate(data) if d["target"] == t]
        # offset clear of the tick labels: with an odd-sized block the centroid lands ON a row
        ax.text(
            -0.145,
            sum(idx) / len(idx),
            t,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9.6,
            color=INK,
            fontweight="semibold",
            rotation=90,
        )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK3)
        ax.spines[s].set_linewidth(0.8)
    ax.grid(axis="x", color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=8.6, length=0)
    ax.set_xlabel(
        "reactions per mode  ← lower is better        (count-once, native routes)",
        fontsize=8.8,
        color=INK2,
    )
    ax.legend(
        handles=[plt.Rectangle((0, 0), 1, 1, color=HUB), plt.Rectangle((0, 0), 1, 1, color=BEST)],
        labels=["hub-batching", "best-candidate"],
        loc="upper right",
        frameon=False,
        fontsize=8.8,
        labelcolor=INK2,
        handlelength=0.9,
        handleheight=0.9,
        borderpad=0.3,
    )

    fig.text(
        0.012,
        0.975,
        "Hub-batching is cheaper per mode for every generator and both targets —\n"
        "but SCENT's headline edge comes from an acquisition only it can run",
        fontsize=11.2,
        color=INK,
        fontweight="semibold",
        ha="left",
        va="top",
        linespacing=1.4,
    )
    fig.text(
        0.012,
        0.877,
        "matrix16 count-once campaign · 30k trajectories / 200 hubs · gates: sEH reward>7.0, "
        "DRD2 reward>0.5 · seed 42",
        fontsize=8.4,
        color=INK2,
        ha="left",
        va="top",
    )
    fig.text(
        0.012,
        0.022,
        "Apples-to-apples (all naive): SCENT 1.22×/1.38× · RxnFlow 1.23×/1.60× · RGFN 1.34×/3.21×. "
        "SCENT's free-frag+K20 rows need promoted dynamic-library fragments only it has.\n"
        'Hatched = FragGFN is a CONTROL, not a peer: its "reactions" are fragment attachments and '
        "has_route=0. Both FragGFN model versions are shown (Logs/046/050): cap-6 is the\n"
        "corrected model, cap-9 is superseded and kept only because the cap-6 sEH cell is not "
        "trained yet, so swapping DRD2 alone would mix versions across blocks.",
        fontsize=7.4,
        color=INK2,
        ha="left",
        va="bottom",
        linespacing=1.5,
    )
    fig.subplots_adjust(left=0.275, right=0.985, top=0.845, bottom=0.155)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"generality_panel.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/generality_panel.png + .pdf")


if __name__ == "__main__":
    main()
