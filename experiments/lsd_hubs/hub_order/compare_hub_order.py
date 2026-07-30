#!/usr/bin/env python
"""Overlay the hub-ordering arms (Logs/053) — does sorting hubs by flow actually buy anything?

Every arm ran the same campaign over the same candidate pool, cost model and acquisition config;
the only difference is which 200 hubs hub-batching may walk and in what order. This script reads the
committed per-arm ``sweep_summary.json`` / ``summary.json`` (CSVs are gitignored) and produces the
head-to-head:

  * ``cost_vs_cutoff.png``   — reactions to reach M* modes vs the diversity cutoff, one line per arm.
    The headline. best-candidate is drawn once as a dashed reference: it never touches hub data, so
    it is identical across arms by construction — the script asserts that and flags any drift.
  * ``compute_vs_cutoff.png`` — measured hub-batching wall-clock vs cutoff (Logs/039). An arm that
    matches on reactions but walks far more hubs to get there is not actually equivalent.
  * ``summary.csv`` + a printed table — the operating point per arm: reactions/mode, modes reached,
    hubs walked, reward-gen calls, compute time, and whether the arm ran out of library.

Arms that cannot reach M* modes at a cutoff show a **gap** in the line and are listed as
pool-limited, never plotted as if they had won — running out of enumerated hubs is a real result,
but it is not the same measurement as completing the budget.

    python experiments/lsd_hubs/hub_order/compare_hub_order.py --results <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from validation.lsdflow.plot_style import title_with_ideal  # noqa: E402

# Display order + colours. `incumbent` is the published strategy (Logs/029/031); the rest are the
# Logs/053 controls. Grey = the two "ignore the flow signal" controls.
ARMS = [
    ("incumbent", "Incumbent: top-1k candidates → flow", "#1f77b4", "-"),
    ("flow_top", "Highest flow first (all hubs)", "#2a9d8f", "-"),
    ("cand_order", "Best-candidate order (all hubs)", "#e9a20c", "--"),
    ("cand_order_fixedset", "Best-candidate order (incumbent set)", "#b8860b", ":"),
    ("random", "Random order (all hubs)", "#9aa0a6", "--"),
    ("flow_bottom", "Lowest flow first (all hubs)", "#b23a48", "-"),
    # Same ordering, 3x the enumeration pool, testing whether flow_bottom's pool-limited cells are
    # the ordering or just too few hubs. Still a fair comparison: the budget is 300 MODES, and the
    # other arms stop having used 17-52% of their own pool, so 600 hubs would not move them (bar a
    # ~1.4% prebuild-K coupling — see the k=0 control).
    ("flow_bottom_600", "Lowest flow first, 600-hub pool", "#d98a96", "-."),
]
BEST_COLOR = "#444444"


def _load(results: Path, arm: str, suffix: str):
    d = results / f"hubord_{arm}{suffix}"
    sweep = d / "sweep_summary.json"
    summ = d / "summary.json"
    if not sweep.exists() and not summ.exists():
        return None
    return {
        "dir": d,
        "sweep": json.load(open(sweep)) if sweep.exists() else None,
        "summary": json.load(open(summ)) if summ.exists() else None,
    }


def _plot_lines(path: Path, series, best_ref, cutoffs, ylabel, title):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[hubord] plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    if best_ref is not None:
        ax.plot(
            cutoffs,
            best_ref,
            color=BEST_COLOR,
            ls=(0, (6, 3)),
            lw=1.6,
            label="best-candidate (no hubs)",
            zorder=1,
        )
    pool_limited = []
    for name, label, color, ls, ys in series:
        xs = [c for c, y in zip(cutoffs, ys) if y is not None]
        vs = [y for y in ys if y is not None]
        ax.plot(xs, vs, color=color, ls=ls, lw=1.9, marker="o", ms=3.4, label=label, zorder=3)
        gaps = [c for c, y in zip(cutoffs, ys) if y is None]
        if gaps:
            pool_limited.append((name, color, gaps))
    # Pool-limited cutoffs are marked only after every line is drawn, so the y-limit they sit on is
    # the final one. An arm that never reaches the mode budget has NO cost to plot there — the mark
    # says "ran out of library", which must not read as a cheap point.
    for name, color, gaps in pool_limited:
        top = ax.get_ylim()[1]
        ax.plot(gaps, [top] * len(gaps), marker="x", ls="none", color=color, ms=7, zorder=4)
        print(f"[hubord] {name}: pool-limited at cutoffs {gaps} (never reached the mode budget)")
    ax.set_xlabel("diversity cutoff (Tanimoto)\nstricter modes ← → looser modes")
    ax.set_ylabel(ylabel)
    ax.set_title(title_with_ideal(title, "lower"), fontsize=10)
    ax.legend(fontsize=7.5, loc="best", framealpha=0.92)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[hubord] wrote {path}")


def _plot_pareto(path: Path, pts, budget_modes, cutoff):
    """Reactions (bench cost) vs measured compute (enumeration cost) at the operating point.

    The two costs trade against each other, so "which ordering is best" has no single answer — but
    *dominance* does: a point beaten on both axes is strictly worse, no weighting required. That is
    the strongest form of the result, so it gets its own figure.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"[hubord] pareto plot skipped ({exc})")
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ok = [p for p in pts if p["reactions"] is not None]
    # Arms can sit almost on top of each other (flow_top and the incumbent differ by 8 reactions and
    # 37 s), so labels are placed by scanning left-to-right and pushing each one clear of the last.
    order = sorted(
        range(len(pts)), key=lambda i: (pts[i]["reactions"] or pts[i]["reactions_at_exhaustion"])
    )
    offsets = {}
    last_y, flip = None, 1
    for i in order:
        y = pts[i]["compute_s"]
        close = last_y is not None and abs(y - last_y) < 0.08 * (
            max(p["compute_s"] for p in pts) - min(p["compute_s"] for p in pts) or 1
        )
        offsets[i] = (10, 14 * flip) if close else (10, 0)
        if close:
            flip = -flip
        last_y = y
    for idx, p in enumerate(pts):
        dominated = any(
            q["reactions"] < p["reactions"] and q["compute_s"] < p["compute_s"]
            for q in ok
            if q is not p and p["reactions"] is not None
        )
        failed = p["reactions"] is None
        x = p["reactions"] if not failed else p["reactions_at_exhaustion"]
        ax.scatter(
            x,
            p["compute_s"],
            s=150 if not failed else 170,
            color=p["color"],
            marker="X" if failed else ("o" if not dominated else "s"),
            facecolors=p["color"] if not dominated or failed else "none",
            edgecolors=p["color"],
            linewidths=2.0,
            zorder=4,
        )
        note = (
            f"{p['label']}\n✗ only {p['modes']}/{budget_modes} modes"
            if failed
            else p["label"] + ("\n(dominated)" if dominated else "")
        )
        ax.annotate(
            note,
            (x, p["compute_s"]),
            textcoords="offset points",
            xytext=offsets[idx],
            fontsize=7.6,
            va="center",
            ha="right" if offsets[idx][0] < 0 else "left",
            color=p["color"],
        )
    front = sorted(
        (
            p
            for p in ok
            if not any(
                q["reactions"] < p["reactions"] and q["compute_s"] < p["compute_s"]
                for q in ok
                if q is not p
            )
        ),
        key=lambda p: p["reactions"],
    )
    if len(front) > 1:
        ax.plot(
            [p["reactions"] for p in front],
            [p["compute_s"] for p in front],
            color="#555",
            lw=1.2,
            ls="-",
            alpha=0.55,
            zorder=1,
            label="Pareto frontier",
        )
    ax.set_xlabel(f"reactions to reach {budget_modes} modes  (bench cost, ↓)")
    ax.set_ylabel("measured compute, seconds  (enumeration cost, ↓)")
    ax.set_title(
        f"Two cost axes at τ=7 / cutoff {cutoff}: every frontier point is a\n"
        "flow- or reward-informed ordering; both no-signal controls are dominated",
        fontsize=9.5,
    )
    ax.margins(x=0.22, y=0.12)
    ax.grid(alpha=0.25, lw=0.6)
    if len(front) > 1:
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"[hubord] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument("--suffix", default="", help="tag suffix (e.g. an alternative child policy)")
    a = ap.parse_args()
    results = Path(a.results)
    out = results / f"comparison{a.suffix}"
    out.mkdir(parents=True, exist_ok=True)

    loaded = [(n, lb, c, ls, _load(results, n, a.suffix)) for n, lb, c, ls in ARMS]
    have = [(n, lb, c, ls, d) for n, lb, c, ls, d in loaded if d]
    if not have:
        print(f"[hubord] no arm results under {results}; run run_arms.sh first")
        return
    missing = [n for n, _, _, _, d in loaded if not d]
    if missing:
        print(f"[hubord] note: no results yet for {', '.join(missing)}")

    # ---- headline: reactions to reach M* modes vs cutoff -------------------------------------
    ref = next((d for _, _, _, _, d in have if d["sweep"]), None)
    if ref is None:
        print("[hubord] no sweep_summary.json found; skipping the cutoff figures")
    else:
        cutoffs = ref["sweep"]["cutoffs"]
        budget_modes = ref["sweep"]["budget_modes"]
        key = "fixed_modes_reactions_at_M"
        cost_series, time_series = [], []
        best_ref, best_drift = None, []
        for n, lb, c, ls, d in have:
            sw = d["sweep"]
            if not sw:
                continue
            if sw["cutoffs"] != cutoffs:
                print(f"[hubord] WARNING {n} swept different cutoffs; excluded from the overlay")
                continue
            cost_series.append((n, lb, c, ls, sw[key]["hub_batching"]))
            bc = sw[key]["best_candidate"]
            if best_ref is None:
                best_ref = bc
            elif bc != best_ref:
                best_drift.append(n)
            ct = (sw.get("compute_time") or {}).get("hub_total_s_by_cutoff")
            if ct:
                time_series.append((n, lb, c, ls, ct))
        if best_drift:
            print(
                "[hubord] WARNING best-candidate differs across arms "
                f"({', '.join(best_drift)}) — it must not depend on hub selection; investigate"
            )
        else:
            print("[hubord] best-candidate identical across arms ✓ (it never reads hub data)")
        _plot_lines(
            out / "cost_vs_cutoff.png",
            cost_series,
            best_ref,
            cutoffs,
            f"reactions to reach {budget_modes} modes",
            f"Hub ordering: cost to build a {budget_modes}-mode library",
        )
        if time_series:
            _plot_lines(
                out / "compute_vs_cutoff.png",
                time_series,
                None,
                cutoffs,
                "measured hub-batching compute (s)",
                "Hub ordering: measured compute to build the library",
            )

    # ---- operating-point table ----------------------------------------------------------------
    cols = [
        "arm",
        "reactions_per_mode",
        "total_modes",
        "total_reactions",
        "distinct_hubs_used",
        "reward_gen_calls",
        "compute_total_s",
        "median_reward",
        "n_scaffolds",
        "stop_reason",
    ]
    rows = []
    for n, lb, _, _, d in have:
        s = d["summary"]
        if not s:
            continue
        hb = s["hub_batching"]
        ct = (s.get("compute_time") or {}).get("hub_batching") or {}
        rows.append(
            {
                "arm": n,
                "reactions_per_mode": hb["reactions_per_mode"],
                "total_modes": hb["total_modes"],
                "total_reactions": hb["total_reactions"],
                "distinct_hubs_used": hb["distinct_hubs_used"],
                "reward_gen_calls": hb["total_reward_gen_calls"],
                "compute_total_s": round(float(ct.get("total_s", 0.0)), 1),
                "median_reward": hb["median_reward"],
                "n_scaffolds": hb["n_scaffolds"],
                "stop_reason": hb["stop_reason"],
            }
        )
    # ---- the two-cost-axis Pareto view ---------------------------------------------------------
    pts = []
    for n, lb, c, _, d in have:
        s = d["summary"]
        if not s:
            continue
        hb = s["hub_batching"]
        ct = (s.get("compute_time") or {}).get("hub_batching") or {}
        reached = hb["stop_reason"] == "modes"
        pts.append(
            {
                "arm": n,
                "label": lb.split(":")[0],
                "color": c,
                "reactions": hb["total_reactions"] if reached else None,
                "reactions_at_exhaustion": hb["total_reactions"],
                "compute_s": float(ct.get("total_s", 0.0)),
                "modes": hb["total_modes"],
            }
        )
    if pts and any(p["compute_s"] for p in pts):
        s0 = next(d["summary"] for _, _, _, _, d in have if d["summary"])
        _plot_pareto(out / "cost_pareto.png", pts, s0["budget_modes"], s0.get("similarity", 0.5))

    ref_summary = next((d["summary"] for _, _, _, _, d in have if d["summary"]), None)
    if rows and ref_summary:
        bc = ref_summary["best_candidate"]
        rows.append(
            {
                "arm": "best_candidate (reference)",
                "reactions_per_mode": bc["reactions_per_mode"],
                "total_modes": bc["total_modes"],
                "total_reactions": bc["total_reactions"],
                "distinct_hubs_used": bc["distinct_hubs_used"],
                "reward_gen_calls": bc["total_reward_gen_calls"],
                "compute_total_s": "",
                "median_reward": bc["median_reward"],
                "n_scaffolds": bc["n_scaffolds"],
                "stop_reason": bc["stop_reason"],
            }
        )
        with open(out / "summary.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
        print()
        print("  ".join(c.ljust(widths[c]) for c in cols))
        for r in rows:
            print("  ".join(str(r[c]).ljust(widths[c]) for c in cols))
        print(f"\n[hubord] wrote {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
