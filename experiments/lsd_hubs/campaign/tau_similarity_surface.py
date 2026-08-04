#!/usr/bin/env python
"""The two-knob cost surface: reactions/mode over (reward bar τ) × (diversity cutoff), fixed budget.

WHY. Both robustness sweeps we run are one-dimensional slices of the same surface: `sweep_campaign.py`
walks the **diversity cutoff** at a fixed reward bar, and `matrix16/gate_curve.py` walks the **reward
bar** at a fixed cutoff. `docs/paper_planning/lsd-flow-publication-strategy.md` §2.3/§2.5 wants them
read as ONE framework — the denominator defense (cutoff) and the numerator/degeneracy defense (τ) are
the same argument in two directions. This driver measures the full grid so the claim "our advantage is
not an artifact of where we set the two knobs" is a surface, not two lines.

It is the strategy-side companion to `experiments/lsd_hubs/reward_diversity/` (Logs/051), which measures
what τ does to the *pools* with no strategy or cost model involved. That analysis says the pools' own
diversity is flat in τ below ~7 and collapses above it; this one says what the *cost* does.

WHAT a cell is. Fix a reaction budget (`--budget-reactions`, default **300**) and run both selection
strategies to it, then read the last accepted point that fits inside the budget:

    reactions/mode = reactions_used / modes_reached          (lower is better)

Because the budget is fixed, this is monotone in "modes reached at 300 reactions" — i.e. the campaign's
Case-1 readout expressed as a ratio, so it is directly comparable to `summary.json::case1_modes_at_*`.

**Pool-limited cells are flagged, never silently plotted as wins.** If a strategy exhausts its library
before spending the budget (`stop_reason != "reactions"`), its ratio is measured over a shorter run and
is not comparable to a budget-limited cell; those cells carry `pool_limited=True` and are hatched in the
figure. At strict cutoffs and high τ this is the common case, and hiding it would flatter whichever
strategy runs dry first.

Cost: pure CPU, no GPU/model/oracle — it re-scores the cached enumeration (`enum_children.json`, rewards
already computed). The pool + cost table load ONCE and every cell runs in-process (same trick as
`gate_curve.py`), which is what makes a 100+ cell grid tractable.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/campaign/tau_similarity_surface.py \
        --analysis-dir  /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189 \
        --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
        --taus 4:8:0.5 --similarities 0.3:0.9:0.05 --budget-reactions 300 --tag scent_seh_surface

Output: ``results/<tag>/{surface.csv,surface.json,surface.png,surface.pdf}``. Re-plot without
recomputing via ``--plot-only``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from run_campaign import (  # noqa: E402  same-dir helpers, so a cell is bit-identical to run_campaign
    _load_candidates,
    _load_enumerated_hubs,
    build_strategy,
)

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments  # noqa: E402
from glue.samplers.lsdflow.child_select import make_child_policy  # noqa: E402
from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: E402
    load_cost_table_from_snapshot,
)
from validation.lsdflow.plot_style import ideal_marker  # noqa: E402

STRATS = ("hub_batching", "best_candidate")
# The two absolute panels encode MAGNITUDE (reactions/mode), not identity, so they share ONE
# single-hue ramp light(cheap) -> dark(expensive) and ONE colorbar: a reader must be able to compare
# them by eye, which two different hues would prevent. Panel identity lives in the titles.
# Deliberately a slate hue, not the campaign's blue/orange — those two mean hub-batching vs
# best-candidate, and they are used ONLY in the advantage panel, where that is exactly the meaning,
# as a diverging pair with a neutral grey midpoint at parity (never a hue at the midpoint).
SEQ_RAMP = ("#fbfbf9", "#6f8fae", "#14263c")
DIVERGING = (
    "#eb6834",
    "#f2f2ef",
    "#2a78d6",
)  # best-candidate better | parity | hub-batching better
STRAT_LABEL = {"hub_batching": "Hub batching", "best_candidate": "Best candidate (previous)"}
# The single (τ, cutoff) point every headline campaign number is quoted at (Logs/029/033, matrix16).
OPERATING_POINT = (7.0, 0.5)


def parse_grid(spec: str) -> list:
    """``"4:8:0.5"`` (inclusive range) or ``"5,6,7"`` -> list of floats, deduped and sorted."""
    out: list = []
    for term in spec.split(","):
        term = term.strip()
        if not term:
            continue
        if ":" in term:
            lo, hi, step = (float(x) for x in term.split(":"))
            v = lo
            while (step > 0 and v <= hi + 1e-9) or (step < 0 and v >= hi - 1e-9):
                out.append(round(v, 4))
                v += step
        else:
            out.append(round(float(term), 4))
    return sorted(dict.fromkeys(out))


def read_at_budget(result, budget_reactions: int) -> dict:
    """Last accepted point inside the budget -> the cell's readout.

    Reading the prefix (rather than the run's totals) is what makes the number exact: a run stopped by
    a reaction budget overshoots by the mode that crosses it, so its totals can exceed the budget.
    """
    inside = [p for p in result.accepted if p.cum_reactions <= budget_reactions]
    last = inside[-1] if inside else None
    pool_limited = result.stop_reason != "reactions"
    return {
        "modes": last.cum_modes if last else 0,
        "reactions_used": last.cum_reactions if last else 0,
        "reactions_per_mode": (
            round(last.cum_reactions / last.cum_modes, 4) if last and last.cum_modes else None
        ),
        "reward_gen_calls": last.cum_reward_gen_calls if last else 0,
        "stop_reason": result.stop_reason,
        "pool_limited": pool_limited,
        "total_modes_run": result.total_modes,
        "total_reactions_run": result.total_reactions,
    }


def run_grid(cands, hubs, comps, cost_table, taus, sims, args) -> list:
    """Every (strategy, τ, cutoff) cell. Loads nothing — the caller holds the pools."""
    child_policy = make_child_policy(args.child_policy)
    prebuilt_cache: dict = {}
    rows: list = []
    n_cells = len(taus) * len(sims)
    t_start = time.time()
    for i_tau, tau in enumerate(taus):
        if (
            args.prebuild_k > 0
        ):  # the pre-select-K ranking is τ-dependent -> one per τ, not per cell
            if tau not in prebuilt_cache:
                ranked = rank_fragments(
                    hubs,
                    cost_table,
                    tau,
                    method=args.rank_by,
                    higher_is_better=args.higher_is_better,
                )
                prebuilt_cache[tau] = {f for f, _ in ranked[: args.prebuild_k]}
            prebuilt = prebuilt_cache[tau]
        else:
            prebuilt = None
        for sim in sims:
            for name in STRATS:
                common = dict(
                    target=args.tag,
                    reward_threshold=tau,
                    similarity=sim,
                    higher_is_better=args.higher_is_better,
                )
                pool = hubs if name == "hub_batching" else cands
                t0 = time.time()
                result = build_strategy(
                    name,
                    pool,
                    cost_table,
                    comps,
                    child_policy=child_policy if name == "hub_batching" else None,
                    prebuilt_fragments=prebuilt if name == "hub_batching" else None,
                    **common,
                ).run(("reactions", args.budget_reactions))
                row = {
                    "strategy": name,
                    "tau": tau,
                    "similarity": sim,
                    "wall_s": round(time.time() - t0, 2),
                }
                row.update(read_at_budget(result, args.budget_reactions))
                rows.append(row)
        done = (i_tau + 1) * len(sims)
        el = time.time() - t_start
        print(
            f"[surface] τ={tau:<5} done ({done}/{n_cells} cells, {el:.0f}s elapsed, "
            f"~{el / done * (n_cells - done):.0f}s left)",
            flush=True,
        )
    return rows


CSV_COLUMNS = (
    "strategy",
    "tau",
    "similarity",
    "modes",
    "reactions_used",
    "reactions_per_mode",
    "reward_gen_calls",
    "stop_reason",
    "pool_limited",
    "total_modes_run",
    "total_reactions_run",
    "wall_s",
)


def write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_COLUMNS})


def read_csv_rows(path: Path) -> list:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append(
                {
                    "strategy": r["strategy"],
                    "tau": float(r["tau"]),
                    "similarity": float(r["similarity"]),
                    "modes": int(r["modes"]),
                    "reactions_used": int(r["reactions_used"]),
                    "reactions_per_mode": float(r["reactions_per_mode"])
                    if r["reactions_per_mode"]
                    else None,
                    "stop_reason": r["stop_reason"],
                    "pool_limited": r["pool_limited"] == "True",
                }
            )
    return rows


# One ink for names, a lighter one for the glosses under them. Keeping the gloss visibly secondary is
# what lets a reader take in "Reward bar τ" first and "low -> high predicted affinity" only if they want
# it; same weight for both would make every axis a two-line sentence to parse.
INK, MUTED = "#1a1a18", "#7a7a72"
# What the two knobs MEAN, in the reader's terms. The reward axis says "predicted": the sEH surrogate
# separates real inhibitors from decoys only weakly (Logs/034, AUROC 0.76 / 0.68 vs matched decoys), so
# calling it affinity outright would overclaim what the number is.
X_GLOSS = "similar modes → diverse modes"
Y_GLOSS = "low → high predicted affinity"


# Points of clearance from the axes for the gloss, and how far the axis NAME is pushed beyond it, so the
# reading order is the same on both axes: axis, then name, then gloss going outward. On the rotated
# y-axis that means the gloss sits to the RIGHT of the name (nearer the plot), which is the mirror of the
# x-axis where the name is above the gloss.
_GLOSS_PAD = {"x": -34, "y": -22}
_NAME_PAD = {"x": 4, "y": 26}


def _sublabel(ax, axis: str, text: str) -> None:
    """A muted gloss beside the axis name — never a second line of the label itself.

    Why not just ``"name\ngloss"``: matplotlib styles a label as ONE text object, so a two-line label
    cannot have a de-emphasized second line. Two objects can."""
    if axis == "x":
        ax.annotate(
            text,
            xy=(0.5, 0),
            xycoords="axes fraction",
            xytext=(0, _GLOSS_PAD["x"]),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
        )
    else:
        ax.annotate(
            text,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(_GLOSS_PAD["y"], 0),
            textcoords="offset points",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8,
            color=MUTED,
        )


def _nice_label_step(values) -> float:
    """Smallest "round" step that labels <= 7 of ``values`` — so every measured cell keeps a tick mark
    but only a legible subset carries text. Derived, not hardcoded, so it survives a re-gridded sweep.
    """
    span = max(values) - min(values)
    for step in (0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0):
        if step >= (values[1] - values[0] if len(values) > 1 else step) and span / step <= 6.5:
            return step
    return span or 1.0


def _ticks(ax, axis: str, values) -> None:
    """Tick marks on EVERY measured value; labels only on the round ones (the dense sweep is accurate
    but unreadable if every 0.05 step is spelled out)."""
    step = _nice_label_step(values)
    labelled = [v for v in values if abs(v / step - round(v / step)) < 1e-6]
    setter = ax.set_xticks if axis == "x" else ax.set_yticks
    setter(values, minor=True)
    setter(labelled)
    fmt = "{:.2f}" if step < 0.5 else "{:g}"
    (ax.set_xticklabels if axis == "x" else ax.set_yticklabels)([fmt.format(v) for v in labelled])
    ax.tick_params(axis=axis, which="minor", length=2.5, color=MUTED)
    ax.tick_params(axis=axis, which="major", length=4.5, labelsize=9)


def _grid(rows, strategy, taus, sims, field):
    """(len(taus) x len(sims)) nested list of ``field``, ``None`` where the cell is missing."""
    by = {(r["tau"], r["similarity"]): r for r in rows if r["strategy"] == strategy}
    return [[(by.get((t, s)) or {}).get(field) for s in sims] for t in taus]


def _edges(vals):
    """Cell edges from centres, so pcolormesh puts each measured value in its own cell."""
    out = [vals[0] - (vals[1] - vals[0]) / 2 if len(vals) > 1 else vals[0] - 0.5]
    for a, b in zip(vals, vals[1:]):
        out.append((a + b) / 2)
    out.append(vals[-1] + (vals[-1] - vals[-2]) / 2 if len(vals) > 1 else vals[-1] + 0.5)
    return out


_GEN_LABEL = {"scent": "SCENT", "rgfn": "RGFN", "rxnflow": "RxnFlow", "fraggfn": "FragGFN"}
_TGT_LABEL = {
    "seh": "sEH surrogate",
    "drd2": "DRD2 surrogate",
    "6td3": "6TD3 docking",
    "clpp": "ClpP docking",
}


def _pretty_cell(tag: str) -> str:
    """'fraggfn_seh' -> 'FragGFN sEH surrogate'. The subtitle used to be the literal string
    "SCENT sEH surrogate" because this driver was written for one cell; running the same grid on the
    other three generators silently mislabelled every one of them. Falls back to the raw tag for any
    name that does not parse, so an unknown tag is visibly unknown rather than wrongly attributed.
    """
    parts = tag.split("_")
    if len(parts) >= 2 and parts[0] in _GEN_LABEL and parts[1] in _TGT_LABEL:
        extra = " ".join(parts[2:])
        return f"{_GEN_LABEL[parts[0]]} {_TGT_LABEL[parts[1]]}" + (f" ({extra})" if extra else "")
    return tag


def plot(path: Path, rows, tag: str, budget: int) -> None:
    """Three panels on the same grid: each strategy's reactions/mode, then the advantage ratio."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    taus = sorted({r["tau"] for r in rows})
    sims = sorted({r["similarity"] for r in rows})
    xe, ye = _edges(sims), _edges(taus)

    rpm = {
        s: np.array(_grid(rows, s, taus, sims, "reactions_per_mode"), dtype=float) for s in STRATS
    }
    lim = {s: np.array(_grid(rows, s, taus, sims, "pool_limited"), dtype=object) for s in STRATS}
    # Shared scale across the two absolute panels: comparing them by eye is the whole point.
    finite = np.concatenate([v[np.isfinite(v)] for v in rpm.values()])
    vmin, vmax = float(finite.min()), float(finite.max())

    # Explicit gridspec slots for the two colorbars: auto-placed colorbars and subplots_adjust fight
    # each other (the bar lands on top of a panel), and the sequential bar must sit with the PAIR it
    # serves, not at the figure edge.
    fig = plt.figure(figsize=(15.4, 6.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.045], height_ratios=[1, 0.05])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    # The shared scale gets a HORIZONTAL bar spanning exactly the two panels it serves — placed under
    # them, it cannot be misread as belonging to the advantage panel (which keeps its own bar, right).
    cax_seq, cax_div = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[0, 3])
    fig.suptitle(
        "Dependence of reward cutoff with similarity cutoff.", fontsize=14, y=0.985, color=INK
    )
    fig.text(
        0.5,
        0.925,
        f"{_pretty_cell(tag)} — fixed budget {budget} reactions",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )

    seq = LinearSegmentedColormap.from_list("rpm", list(SEQ_RAMP))
    seq.set_bad("#e9e9e4")  # cells that reached no mode at all inside the budget
    last_mesh = None
    for ax, name in zip(axes[:2], STRATS):
        last_mesh = ax.pcolormesh(
            xe, ye, np.ma.masked_invalid(rpm[name]), cmap=seq, vmin=vmin, vmax=vmax
        )
        ax.set_title(STRAT_LABEL[name], fontsize=10.5, color=INK)
        _hatch_limited(ax, lim[name], taus, sims)
    # ONE colorbar for the pair — two would imply two scales.
    cb = fig.colorbar(last_mesh, cax=cax_seq, orientation="horizontal")
    cb.set_label(
        f"reactions / mode {ideal_marker('lower', axis='x')} — shared scale", fontsize=9
    )  # horizontal bar -> "(←)"

    # Advantage: how many times cheaper hub-batching is. Diverging around parity (1.0).
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = rpm["best_candidate"] / rpm["hub_batching"]
    both_ok = ~(lim["hub_batching"].astype(bool) | lim["best_candidate"].astype(bool))
    dv = LinearSegmentedColormap.from_list("advantage", list(DIVERGING))
    dv.set_bad("#e9e9e4")
    finite_r = ratio[np.isfinite(ratio)]
    vhi = float(max(finite_r.max(), 1.05)) if finite_r.size else 1.05
    vlo = float(min(finite_r.min(), 0.95)) if finite_r.size else 0.95
    mesh = axes[2].pcolormesh(
        xe,
        ye,
        np.ma.masked_invalid(ratio),
        cmap=dv,
        norm=TwoSlopeNorm(vcenter=1.0, vmin=vlo, vmax=vhi),
    )
    cb = fig.colorbar(mesh, cax=cax_div)
    cb.set_label(f"× cheaper with hub batching {ideal_marker('higher')} — 1.0 = parity", fontsize=9)
    axes[2].set_title("Hub-batching advantage", fontsize=10.5, color=INK)
    _hatch_limited(axes[2], ~both_ok, taus, sims)

    for i, ax in enumerate(axes):
        ax.set_xlabel(
            "Diversity cutoff (Tanimoto)", fontsize=10, color=INK, labelpad=_NAME_PAD["x"]
        )
        ax.invert_xaxis()  # 0.9 -> 0.3, matching every other LSD-Flow diversity panel
        ax.set_ylabel("Reward bar τ", fontsize=10, color=INK, labelpad=_NAME_PAD["y"])
        _sublabel(ax, "x", X_GLOSS)
        _sublabel(ax, "y", Y_GLOSS)
        _ticks(ax, "x", sims)
        _ticks(ax, "y", taus)
        ax.plot(
            *OPERATING_POINT[::-1], marker="*", ms=13, mfc="none", mec="#111111", mew=1.4, zorder=5
        )
        if i == 0:  # label the star once; repeating it three times only risks collisions
            ax.annotate(
                "headline operating point",
                xy=OPERATING_POINT[::-1],
                xytext=(-6, -16),
                textcoords="offset points",
                fontsize=7.5,
                color="#111111",
                ha="left",
            )
    fig.text(
        0.008,
        0.015,
        "★ = the (τ=7.0, cutoff=0.5) point every headline number is quoted at.  Hatched cells are "
        "POOL-LIMITED: a strategy ran out of library before spending the budget, so its ratio is "
        "measured over a shorter run and is not comparable to a budget-limited cell.",
        fontsize=7.5,
        color="#666660",
    )
    fig.subplots_adjust(left=0.055, right=0.975, top=0.85, bottom=0.13, wspace=0.30, hspace=0.62)
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _hatch_limited(ax, mask, taus, sims) -> None:
    """Cross-hatch every pool-limited cell — a visible caveat, not a footnote."""
    dx = (sims[1] - sims[0]) if len(sims) > 1 else 1.0
    dy = (taus[1] - taus[0]) if len(taus) > 1 else 1.0
    import matplotlib.patches as mpatches

    for i, t in enumerate(taus):
        for j, s in enumerate(sims):
            if mask[i][j]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (s - dx / 2, t - dy / 2),
                        dx,
                        dy,
                        fill=False,
                        hatch="xxx",
                        edgecolor="#555550",
                        lw=0.0,
                        alpha=0.55,
                        zorder=4,
                    )
                )


def summarize(rows, budget: int) -> dict:
    """The numbers a reader wants without reading the grid: where the advantage is largest/smallest
    among comparable (not pool-limited) cells, and the value at the headline operating point."""
    by = {(r["strategy"], r["tau"], r["similarity"]): r for r in rows}
    comparable = []
    for (strat, tau, sim), r in by.items():
        if strat != "hub_batching":
            continue
        other = by.get(("best_candidate", tau, sim))
        if not other or r["pool_limited"] or other["pool_limited"]:
            continue
        if not r["reactions_per_mode"] or not other["reactions_per_mode"]:
            continue
        comparable.append(
            {
                "tau": tau,
                "similarity": sim,
                "hub": r["reactions_per_mode"],
                "best": other["reactions_per_mode"],
                "ratio": round(other["reactions_per_mode"] / r["reactions_per_mode"], 4),
                "hub_modes": r["modes"],
                "best_modes": other["modes"],
            }
        )
    comparable.sort(key=lambda c: c["ratio"])
    op = next(
        (
            c
            for c in comparable
            if abs(c["tau"] - OPERATING_POINT[0]) < 1e-9
            and abs(c["similarity"] - OPERATING_POINT[1]) < 1e-9
        ),
        None,
    )
    n_cells = len({(r["tau"], r["similarity"]) for r in rows})
    return {
        "budget_reactions": budget,
        "n_grid_cells": n_cells,
        "n_comparable_cells": len(comparable),
        "n_pool_limited_cells": n_cells - len(comparable),
        "hub_wins_cells": sum(1 for c in comparable if c["ratio"] > 1.0),
        "ratio_min": comparable[0] if comparable else None,
        "ratio_max": comparable[-1] if comparable else None,
        "at_operating_point": op,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis-dir", help="sampling-stage dir (records.csv + compositions.json)")
    ap.add_argument("--enum-children", help="enum_children.json from the enumeration worker")
    ap.add_argument(
        "--snapshot", default="", help="SCENT fragments_<N>.json for the nested cost model"
    )
    ap.add_argument("--taus", default="4:8:0.5", help="reward bars, 'lo:hi:step' and/or comma list")
    ap.add_argument("--similarities", default="0.3:0.9:0.05", help="diversity cutoffs")
    ap.add_argument("--budget-reactions", type=int, default=300)
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--child-policy", default="free_frag", choices=["reward", "free_frag"])
    ap.add_argument(
        "--prebuild-k", type=int, default=20, help="pre-select-K (Logs/037); 0 disables"
    )
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--plot-only", action="store_true", help="redraw from results/<tag>/surface.csv"
    )
    args = ap.parse_args()

    out = Path(args.out_dir) if args.out_dir else HERE / "results" / args.tag
    if args.plot_only:
        rows = read_csv_rows(out / "surface.csv")
        plot(out / "surface", rows, args.tag, args.budget_reactions)
        print(f"redrew {out}/surface.{{png,pdf}} from {len(rows)} rows")
        return

    if not args.analysis_dir or not args.enum_children:
        ap.error("--analysis-dir and --enum-children are required unless --plot-only")
    taus, sims = parse_grid(args.taus), parse_grid(args.similarities)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cands, comps = _load_candidates(Path(args.analysis_dir), args.higher_is_better)
    hubs = _load_enumerated_hubs(Path(args.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(
        json.load(open(args.snapshot)) if args.snapshot else {}
    )
    print(
        f"[surface] {len(cands)} candidates, {len(hubs)} hubs, {len(cost_table.promoted_set)} promoted "
        f"(recipes={bool(cost_table.recipes)}) loaded in {time.time() - t0:.1f}s\n"
        f"[surface] grid {len(taus)} τ x {len(sims)} cutoffs = {len(taus) * len(sims)} cells x 2 "
        f"strategies, budget {args.budget_reactions} reactions, child_policy={args.child_policy}, "
        f"prebuild_k={args.prebuild_k}",
        flush=True,
    )

    rows = run_grid(cands, hubs, comps, cost_table, taus, sims, args)
    write_csv(out / "surface.csv", rows)
    payload = {
        "tag": args.tag,
        "analysis_dir": args.analysis_dir,
        "enum_children": args.enum_children,
        "snapshot": args.snapshot,
        "taus": taus,
        "similarities": sims,
        "budget_reactions": args.budget_reactions,
        "child_policy": args.child_policy,
        "prebuild_k": args.prebuild_k,
        "rank_by": args.rank_by if args.prebuild_k else None,
        "wall_s": round(time.time() - t0, 1),
        "summary": summarize(rows, args.budget_reactions),
    }
    (out / "surface.json").write_text(json.dumps(payload, indent=2))
    plot(out / "surface", rows, args.tag, args.budget_reactions)
    s = payload["summary"]
    print(
        f"\n[surface] {s['n_comparable_cells']}/{s['n_grid_cells']} cells comparable "
        f"({s['n_pool_limited_cells']} pool-limited); hub-batching cheaper in "
        f"{s['hub_wins_cells']}/{s['n_comparable_cells']}"
    )
    for k in ("ratio_min", "ratio_max", "at_operating_point"):
        print(f"  {k}: {json.dumps(s[k])}")
    print(f"[surface] wrote {out}/surface.{{csv,json,png,pdf}} in {payload['wall_s']}s")


if __name__ == "__main__":
    main()
