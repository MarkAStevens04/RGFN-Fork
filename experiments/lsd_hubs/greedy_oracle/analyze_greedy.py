#!/usr/bin/env python
"""Cross-cell readout for the greedy-oracle arm (Logs/060).

Reads every `results/<cell>/summary.json` + `curve.csv` produced by `run_greedy_oracle.py` and answers
three questions in one table and one figure:

1. **How far below an adaptive optimiser does the flow ordering sit?** — on both budget readings
   (reactions at a mode target, and modes at a reaction budget), because they do not agree and the
   honest answer reports both.
2. **What fraction of the achievable improvement does flow recover?** — the gap from the standing
   baseline to the greedy is the improvement on offer; flow's share of it is the number that says
   whether a static ranking is good enough.
3. **Does the greedy independently rediscover the high-flow region?** — the hub pool is stored in
   flow-rank order, so the rank of each hub the greedy chose is read directly. If an optimiser that
   never reads flow concentrates on high-flow hubs anyway, that is mechanism evidence for Logs/053's
   "flow selects the neighbourhood" claim, obtained from a completely different direction.

Pure CPU, seconds. Reads only committed artifacts.

    python experiments/lsd_hubs/greedy_oracle/analyze_greedy.py
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

CELL_ORDER = [
    "rgfn_seh",
    "scent_seh",
    "rxnflow_seh",
    "fraggfn_seh",
    "rgfn_drd2",
    "scent_drd2",
    "rxnflow_drd2",
    "fraggfn_drd2",
    "scent_clpp",
    "rxnflow_clpp",
    "fraggfn_clpp",
    "scent_6td3",
    "rxnflow_6td3",
    "fraggfn_6td3",
]
DOCKING = {"clpp", "6td3"}


def _hub_rank_map(cell: str) -> dict | None:
    """hub_key -> its index in the flow ranking (the order enum_children.json stores hubs in)."""
    import subprocess

    try:
        emit = subprocess.run(
            [
                "python",
                str(REPO / "experiments/lsd_hubs/matrix16/manifest.py"),
                "--emit",
                cell.split("_")[0],
                cell.split("_", 1)[1],
            ],
            capture_output=True,
            text=True,
            cwd=REPO,
            timeout=120,
        )
        if emit.returncode != 0:
            return None
        enum_dir = next(
            (
                ln.split("=", 1)[1].strip().strip('"')
                for ln in emit.stdout.splitlines()
                if ln.startswith("ENUM_DIR=")
            ),
            None,
        )
        if not enum_dir:
            return None
        path = Path(enum_dir) / "enum_children.json"
        if not path.exists():
            return None
        data = json.load(open(path))
    except Exception:
        return None
    ranks: dict = {}
    for i, h in enumerate(data.get("hubs", [])):
        ranks.setdefault(h["hub_key"], i)  # first occurrence = best flow rank
    return ranks


def _walk_hubs(curve_path: Path) -> dict:
    """Ordered distinct source hubs per arm, from the accepted-mode curve."""
    out: dict = {}
    with open(curve_path) as fh:
        for r in csv.DictReader(fh):
            seq = out.setdefault(r["arm"], [])
            hk = r["source_hub"]
            if hk and hk not in seq:
                seq.append(hk)
    return out


def _pct(x):
    return None if x is None else round(100 * x, 1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--results", default=str(HERE / "results"))
    ap.add_argument(
        "--no-ranks", action="store_true", help="skip the hub-rank analysis (needs enums)"
    )
    a = ap.parse_args()
    root = Path(a.results)

    rows = []
    for cell in CELL_ORDER:
        sp = root / cell / "summary.json"
        if not sp.exists():
            continue
        s = json.load(open(sp))
        arms = s["arms"]
        bc, hb, gr = arms["best_candidate"], arms["hub_batching"], arms["greedy_oracle"]
        mkey = f"case1_modes_at_{s['budget_reactions']}rxn"

        # --- cost reading (reactions per mode at each arm's own stopping point)
        bc_rm, hb_rm, gr_rm = (
            bc["reactions_per_mode"],
            hb["reactions_per_mode"],
            gr["reactions_per_mode"],
        )
        # --- yield reading (modes delivered inside a fixed reaction budget) — the headline metric
        bc_m, hb_m, gr_m = bc[mkey], hb[mkey], gr[mkey]

        # Share of the achievable improvement the STATIC flow ranking recovers. The greedy defines
        # what was on offer; 1.0 = flow matches it, 0.0 = flow is no better than the baseline.
        recov_cost = (
            (bc_rm - hb_rm) / (bc_rm - gr_rm) if (bc_rm and gr_rm and bc_rm != gr_rm) else None
        )
        recov_mode = (hb_m - bc_m) / (gr_m - bc_m) if (gr_m is not None and gr_m != bc_m) else None

        rows.append(
            {
                "cell": cell,
                "docking": cell.split("_", 1)[1] in DOCKING,
                "bar": s["reward_threshold"],
                "higher_is_better": s["higher_is_better"],
                "bc_rxn_per_mode": bc_rm,
                "flow_rxn_per_mode": hb_rm,
                "greedy_rxn_per_mode": gr_rm,
                "flow_excess_pct_cost": s["flow_vs_greedy"]["flow_excess_pct"],
                "bc_modes_at_budget": bc_m,
                "flow_modes_at_budget": hb_m,
                "greedy_modes_at_budget": gr_m,
                "flow_modes_deficit_pct": _pct((gr_m - hb_m) / gr_m) if gr_m else None,
                "recovered_frac_cost": None if recov_cost is None else round(recov_cost, 4),
                "recovered_frac_modes": None if recov_mode is None else round(recov_mode, 4),
                "flow_total_modes": hb["total_modes"],
                "greedy_total_modes": gr["total_modes"],
                "pool_limited": hb["stop_reason"] != "modes" or gr["stop_reason"] != "modes",
                "flow_oracle_calls": hb["total_reward_gen_calls"],
                "greedy_pool_calls": gr.get("pool_reward_gen_calls"),
                "oracle_overhead_x": s["flow_vs_greedy"]["oracle_call_overhead_x"],
                "flow_select_s": hb["selection_wall_s"],
                "greedy_select_s": gr["selection_wall_s"],
                "select_overhead_x": round(gr["selection_wall_s"] / hb["selection_wall_s"], 1)
                if hb["selection_wall_s"]
                else None,
                "regression_pass": (s.get("regression") or {}).get("identical"),
            }
        )

        # --- where do the greedy's hubs sit in the flow ranking?
        if not a.no_ranks:
            ranks = _hub_rank_map(cell)
            cp = root / cell / "curve.csv"
            if ranks and cp.exists():
                walks = _walk_hubs(cp)
                n = len(ranks)
                for arm, key in (("greedy_oracle", "greedy"), ("hub_batching", "flow")):
                    hubs = [ranks[h] for h in walks.get(arm, []) if h in ranks]
                    if hubs:
                        rows[-1][f"{key}_hub_rank_median"] = statistics.median(hubs)
                        rows[-1][f"{key}_hub_rank_p90"] = sorted(hubs)[int(0.9 * (len(hubs) - 1))]
                        rows[-1][f"{key}_hubs_in_top_decile_pct"] = _pct(
                            sum(1 for r in hubs if r < 0.1 * n) / len(hubs)
                        )
                        rows[-1][f"{key}_n_hubs"] = len(hubs)
                rows[-1]["n_hubs_pool"] = n
                rows[-1]["uniform_null_median_rank"] = n // 2
                g = set(walks.get("greedy_oracle", [])) & set(ranks)
                f = set(walks.get("hub_batching", [])) & set(ranks)
                if g and f:
                    rows[-1]["hub_jaccard_flow_greedy"] = round(len(g & f) / len(g | f), 3)

    if not rows:
        raise SystemExit(f"no summaries under {root}")

    out_csv = root / "greedy_oracle_summary.csv"
    fields = sorted({k for r in rows for k in r}, key=lambda k: (k != "cell", k))
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    # ---------------------------------------------------------------- console table
    print(
        f"\n{'cell':<15} {'bar':>6} | {'best-c':>7} {'flow':>7} {'greedy':>7} {'flow+%':>7} | "
        f"{'bc':>4} {'flow':>5} {'grdy':>5} {'-%':>6} | {'recov':>6} {'oracle':>7} {'sel':>6}"
    )
    print(
        f"{'':<15} {'':>6} | {'--- reactions/mode ---':^31} | {'-- modes @ rxn budget --':^24} | "
        f"{'share':>6} {'x':>7} {'x':>6}"
    )
    print("-" * 122)
    for r in rows:
        flag = "*" if r["pool_limited"] else " "
        print(
            f"{r['cell']:<15} {r['bar']:>6} |"
            f" {r['bc_rxn_per_mode']:>7.3f} {r['flow_rxn_per_mode']:>7.3f} {r['greedy_rxn_per_mode']:>7.3f}"
            f" {r['flow_excess_pct_cost']:>6.1f}%{flag}|"
            f" {r['bc_modes_at_budget']:>4} {r['flow_modes_at_budget']:>5} {r['greedy_modes_at_budget']:>5}"
            f" {r['flow_modes_deficit_pct'] if r['flow_modes_deficit_pct'] is not None else 0:>5.1f}% |"
            f" {r['recovered_frac_cost'] if r['recovered_frac_cost'] is not None else float('nan'):>6.2f}"
            f" {r['oracle_overhead_x'] if r['oracle_overhead_x'] is not None else float('nan'):>7.2f}"
            f" {r['select_overhead_x'] if r['select_overhead_x'] is not None else float('nan'):>6.1f}"
        )
    print("-" * 122)
    print(
        "* = at least one arm could not reach the mode budget (pool-limited); its reactions/mode is\n"
        "    measured over a shorter run, so read the modes-at-budget columns for those cells."
    )

    ex = [r["flow_excess_pct_cost"] for r in rows if r["flow_excess_pct_cost"] is not None]
    df = [r["flow_modes_deficit_pct"] for r in rows if r["flow_modes_deficit_pct"] is not None]
    rc = [r["recovered_frac_cost"] for r in rows if r["recovered_frac_cost"] is not None]
    print(
        f"\nflow excess over greedy (reactions/mode): median {statistics.median(ex):.1f}%  "
        f"range {min(ex):.1f}-{max(ex):.1f}%   n={len(ex)}"
    )
    print(
        f"flow shortfall vs greedy (modes at budget): median {statistics.median(df):.1f}%  "
        f"range {min(df):.1f}-{max(df):.1f}%   n={len(df)}"
    )
    print(
        f"share of achievable cost gain recovered by the static flow ranking: "
        f"median {statistics.median(rc):.3f}  range {min(rc):.3f}-{max(rc):.3f}"
    )
    bad = [r["cell"] for r in rows if r["regression_pass"] is False]
    print(
        f"regression (static greedy == shipped hub_batching): "
        f"{sum(1 for r in rows if r['regression_pass'])}/{len(rows)} pass"
        + (f"  FAILURES: {bad}" if bad else "")
    )

    rk = [r for r in rows if "greedy_hub_rank_median" in r]
    if rk:
        print("\nWhere the greedy's chosen hubs sit in the FLOW ranking (it never reads flow):")
        print(
            f"{'cell':<15} {'pool':>6} {'greedy med':>11} {'flow med':>9} {'null':>6} "
            f"{'greedy top-10%':>15} {'jaccard':>8}"
        )
        for r in rk:
            print(
                f"{r['cell']:<15} {r['n_hubs_pool']:>6} {r['greedy_hub_rank_median']:>11.1f} "
                f"{r['flow_hub_rank_median']:>9.1f} {r['uniform_null_median_rank']:>6} "
                f"{r.get('greedy_hubs_in_top_decile_pct', float('nan')):>14.1f}% "
                f"{r.get('hub_jaccard_flow_greedy', float('nan')):>8}"
            )
        med = statistics.median([r["greedy_hub_rank_median"] for r in rk])
        null = statistics.median([r["uniform_null_median_rank"] for r in rk])
        print(
            f"\nmedian across cells: greedy picks hubs at flow-rank {med:.0f} vs a uniform null of {null:.0f}"
        )

    print(f"\n-> {out_csv}")


if __name__ == "__main__":
    main()
