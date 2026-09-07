#!/usr/bin/env python
"""OURS. The three figures from the depth / catalogue / catalogue-distinct experiments.

Reads results off disk and draws whatever is present, so re-running as replicates land extends the
error bars rather than requiring a rewrite. Every panel carries the caveat that governs how it may be
read, IN the figure rather than only in the caption -- these plots have already been wrong once by
mixing two different x-axes, and a caption is easy to crop away.

FIG 1  catalogue ladder: route-solve rate vs purchasable catalogue. The random-draw rungs are the
       point -- a random slice of ZINC the same size as ZINCFrag routes nothing, so the effect is
       CURATION and not size.
FIG 2  depth-cost on ONE axis: both methods' depth counted from ZINC. This is the headline. Ours is
       drawn as a BAND because the two unroutable policies bracket the truth: `drop` removes the
       molecules furthest from the competitor's catalogue (understates us) and `deep` counts them as
       passing (overstates us).
FIG 3  catalogue-distinct sweep: modes at a 100-reaction budget vs the tau that governs BOTH mutual
       and block dissimilarity.

Usage:  python plot_depth_catalogue.py --out-dir <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results")
LAD = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/ladder")
OURS, THEIRS, BC = "#2a6f97", "#c1462f", "#8a8a8a"


def _read_ladder():
    out = {}
    for f in sorted(LAD.glob("*_ladder.jsonl")):
        cell = f.name.replace("_stage2_pruned_N500_ladder.jsonl", "")
        per = defaultdict(lambda: [0, 0])
        for line in f.open():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            per[r["stock"]][1] += 1
            per[r["stock"]][0] += int(bool(r["solved"]))
        if per:
            out[cell] = {k: (v[0], v[1]) for k, v in per.items()}
    return out


def _read_zinc_axis():
    """{(seed, policy): {depth: modes}} — the LAST round of each driver run is its result."""
    out = defaultdict(dict)
    conv = {}
    for d in sorted(RES.glob("zinc_axis_seed*_d*_*")):
        m = re.search(r"zinc_axis_seed(\d+)_d(\d)_(drop|deep)$", d.name)
        if not m:
            continue
        seed, depth, pol = int(m.group(1)), int(m.group(2)), m.group(3)
        dr = d / "driver_rounds.json"
        if not dr.exists():
            continue
        j = json.loads(dr.read_text())
        rounds = [d / f"round{r['round']}" for r in j["rounds"]]
        sm = [p for p in rounds if (p / "summary.json").exists()]
        if not sm:
            continue
        s = json.loads((sm[-1] / "summary.json").read_text())
        out[(seed, pol)][depth] = s["hub_batching"]["case1_modes_at_100rxn"]
        conv[(seed, pol, depth)] = j["rounds"][-1]["n_unjudged"] == 0
    return out, conv


def _read_competitor_depth():
    """{seed: {depth: (modes, capped)}} from the depth-pruned stage-2 cells."""
    out = defaultdict(dict)
    for d in sorted(RES.glob("s3gfn_seh_seed*_stage2_pruned_N500_depth*_select")):
        m = re.search(r"seed(\d+)_stage2_pruned_N500_depth(\d)_select$", d.name)
        f = d / "select_frontier.csv"
        if not m or not f.exists():
            continue
        for r in csv.DictReader(f.open()):
            if r["budget_rxns"] == "100":
                out[int(m.group(1))][int(m.group(2))] = (
                    int(r["n_modes_kept"]),
                    r["time_capped"] == "True",
                )
    return out


def _read_cmode():
    ours, theirs = defaultdict(dict), defaultdict(dict)
    for d in sorted(RES.glob("cmode_ours_seed*_t*")):
        m = re.search(r"cmode_ours_seed(\d+)_t([\d.]+)$", d.name)
        f = d / "summary.json"
        if m and f.exists():
            j = json.loads(f.read_text())
            ours[float(m.group(2))][int(m.group(1))] = (
                j["hub_batching"]["case1_modes_at_100rxn"],
                j["best_candidate"]["case1_modes_at_100rxn"],
                j["hub_batching"]["stop_reason"],
            )
    for d in sorted(RES.glob("s3gfn_seh_seed*_cmode*_select")):
        m = re.search(r"seed(\d+)_cmode(\d\d)_N(\d+)_select$", d.name)
        f = d / "select_frontier.csv"
        if not m or not f.exists():
            continue
        tau = float(f"0.{m.group(2)[1]}")
        for r in csv.DictReader(f.open()):
            if r["budget_rxns"] == "100":
                theirs[tau][int(m.group(1))] = (int(r["n_modes_kept"]), int(m.group(3)))
    return ours, theirs


def _band(ax, xs, lo, hi, color, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(xs, hi, "-o", color=color, ms=5, lw=2, label=label)
    ax.plot(xs, lo, "--", color=color, lw=1.2, alpha=0.8)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", default="experiments/lsd_hubs/campaign/results/depth_catalogue")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- FIG 1: catalogue ladder -------------------------------------------------------------
    lad = _read_ladder()
    if lad:
        order = [
            ("zinc", "ZINC\n17.4M"),
            ("zincfrag", "ZINCFrag\n178,598"),
            ("zinc_rand178k", "random ZINC\n178,598"),
            ("rgfnlib", "our blocks\n456"),
            ("zinc_rand418", "random ZINC\n456"),
        ]
        fig, ax = plt.subplots(figsize=(8.2, 4.6))
        cells = sorted(lad)
        w = 0.8 / max(len(cells), 1)
        for i, cell in enumerate(cells):
            xs, ys = [], []
            for k, (key, _) in enumerate(order):
                if key in lad[cell]:
                    s, n = lad[cell][key]
                    xs.append(k + i * w - 0.4 + w / 2)
                    ys.append(100.0 * s / n)
            ax.bar(xs, ys, width=w * 0.9, label=cell.replace("s3gfn_", ""))
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([lbl for _, lbl in order], fontsize=8)
        ax.set_ylabel("molecules with a route (%)")
        ax.set_title(
            "Catalogue ladder: it is CURATION, not size\n"
            "a random slice of ZINC the same size as ZINCFrag routes nothing",
            fontsize=10,
        )
        ax.legend(fontsize=7)
        ax.grid(axis="y", alpha=0.25)
        ax.text(
            0.99,
            0.97,
            "ZINC and ZINCFrag are NOT nested in this build\n"
            "0% from our blocks = USPTO templates cannot reverse our chemistry,\n"
            "NOT chemical unreachability (cf. Logs/048 Exp A)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig1_catalogue_ladder.png", dpi=160)
        fig.savefig(out / "fig1_catalogue_ladder.pdf")
        plt.close(fig)
        print(f"[plot] fig1: {len(cells)} cells -> {out}/fig1_catalogue_ladder.png")

    # ---- FIG 2: depth-cost on ONE axis (the headline) ----------------------------------------
    za, conv = _read_zinc_axis()
    comp = _read_competitor_depth()
    if za:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        seeds = sorted({s for s, _ in za})
        depths = sorted({d for v in za.values() for d in v})
        for s in seeds:
            lo = [za.get((s, "drop"), {}).get(d) for d in depths]
            hi = [za.get((s, "deep"), {}).get(d) for d in depths]
            if all(x is not None for x in lo + hi):
                _band(
                    ax,
                    depths,
                    lo,
                    hi,
                    OURS,
                    f"hub-batching, seed {s} (drop..deep)" if s == seeds[0] else None,
                )
        for s in sorted(comp):
            xs = sorted(comp[s])
            ys = [comp[s][d][0] for d in xs]
            ax.plot(
                xs,
                ys,
                "-s",
                color=THEIRS,
                ms=5,
                lw=2,
                label="S3-GFN + MultiAiZ + SPARROW" if s == sorted(comp)[0] else None,
            )
            for d in xs:
                if comp[s][d][1]:
                    ax.annotate(
                        "capped\n(lower bound)",
                        (d, comp[s][d][0]),
                        fontsize=6,
                        color=THEIRS,
                        xytext=(4, -14),
                        textcoords="offset points",
                    )
        ax.set_xlabel("minimum synthetic depth (reactions from ZINC-purchasable material)")
        ax.set_ylabel("distinct molecules delivered at 100 reactions")
        ax.set_title(
            "Depth vs delivered library, BOTH sides measured from ZINC\n"
            "the advantage grows with depth once the axis is shared",
            fontsize=10,
        )
        ax.set_xticks(depths)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
        ax.text(
            0.99,
            0.97,
            "BAND = the two unroutable policies, which bracket the truth:\n"
            "  upper (`deep`)  unroutable counts as deep — CONVERGED\n"
            "  lower (`drop`)  unroutable dropped — hit the 40-round cap, an UPPER bound\n"
            "Ours routed with plain AiZynth, theirs with MultiAiZ (stronger) — favours them\n"
            "Superseded hit gates: diagnostic only, not for publication",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig2_depth_zinc_axis.png", dpi=160)
        fig.savefig(out / "fig2_depth_zinc_axis.pdf")
        plt.close(fig)
        print(
            f"[plot] fig2: seeds {seeds}, competitor seeds {sorted(comp)} -> fig2_depth_zinc_axis.png"
        )

    # ---- FIG 3: catalogue-distinct sweep -----------------------------------------------------
    co, ct = _read_cmode()
    if co:
        fig, ax = plt.subplots(figsize=(7.4, 5.0))
        taus = sorted(co, reverse=True)
        for arm, idx, col, lbl in ((0, 0, OURS, "hub-batching"), (1, 1, BC, "best-candidate")):
            ys = [sum(v[idx] for v in co[t].values()) / len(co[t]) for t in taus]
            er = [
                (max(v[idx] for v in co[t].values()) - min(v[idx] for v in co[t].values())) / 2
                for t in taus
            ]
            ax.errorbar(taus, ys, yerr=er, fmt="-o", color=col, ms=5, lw=2, capsize=3, label=lbl)
        if ct:
            tt = sorted(ct, reverse=True)
            ax.plot(
                tt,
                [sum(v[0] for v in ct[t].values()) / len(ct[t]) for t in tt],
                "-s",
                color=THEIRS,
                ms=5,
                lw=2,
                label="S3-GFN + MultiAiZ + SPARROW",
            )
            for t in tt:
                for _, (mo, pool) in ct[t].items():
                    if pool < 500:
                        ax.annotate(
                            f"pool {pool}\n(pool-limited)",
                            (t, mo),
                            fontsize=6,
                            color=THEIRS,
                            xytext=(4, 6),
                            textcoords="offset points",
                        )
        ax.invert_xaxis()
        ax.set_xlabel(
            "tau — governs dissimilarity from other modes AND from every building block\n"
            "(stricter to the right)"
        )
        ax.set_ylabel("distinct molecules delivered at 100 reactions")
        ax.set_title(
            "Catalogue-distinct modes: what if the library must differ from what you can buy?\n"
            "no retrosynthesis involved — a structural constraint",
            fontsize=10,
        )
        ax.set_xticks(taus)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        ax.text(
            0.01,
            0.03,
            "tau=0.3 is BOTH pool-limited and draw-sensitive (yield swings ~2.7x between\n"
            "two draws from one frozen checkpoint) — it carries no conclusion alone.\n"
            "Block reference set is ZINCFrag + our 418, NOT the 17.4M ZINC stock\n"
            "(zinc_stock.hdf5 is InChIKeys only and cannot be fingerprinted).",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.5,
            color="#555",
        )
        fig.tight_layout()
        fig.savefig(out / "fig3_catalogue_distinct.png", dpi=160)
        fig.savefig(out / "fig3_catalogue_distinct.pdf")
        plt.close(fig)
        print(
            f"[plot] fig3: taus {taus}, ours seeds "
            f"{sorted({s for t in co for s in co[t]})} -> fig3_catalogue_distinct.png"
        )


if __name__ == "__main__":
    main()
