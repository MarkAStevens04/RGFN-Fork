#!/usr/bin/env python
"""Headline figure (Logs/056): the competitor pipeline run as a chemist would run it.

PANEL A is a 2x2, and it is drawn that way ON PURPOSE. The earlier framing (entries 041/048) priced
BOTH pipelines by re-deriving every route from scratch, and under that pricing the non-reaction
baseline won. The reversal is not a different measurement of the same thing — it is the difference
between "re-derive the routes" and "each pipeline priced the way it is actually run". Showing only
the solid pair would be the flattering half of the story, so both regimes are plotted:

    hue     = pipeline (blue = LSD-Flow, orange = S3-GFN)     -- colour follows the ENTITY
    style   = pricing regime (dashed = from-scratch re-derivation, solid = as actually run)

Read three comparisons off the 100-mode guide line:
  * dashed vs dashed  -> from-scratch re-derivation: baseline AHEAD (269 vs 305). We lose.
  * solid  vs solid   -> each as actually run: 411 vs 131 = 3.13x in our favour (the headline).
  * our solid vs their STRONGEST configuration (235: MultiAiZ + diversity-aware greedy) -> 1.79x.
    The CONSERVATIVE number, and the one to quote if a reviewer objects that SPARROW-selects is not
    the baseline at its best. Plotted as a third orange series (dotted).

THE FORMERLY-MISSING CELL WAS RUN, and it did what we suspected: S3-GFN with MultiAiZ AND a
diversity-aware greedy selection reaches 100 modes in **235** reactions, beating both previously
tested baseline configurations (269 from-scratch+greedy, 411 MultiAiZ+SPARROW-selects). It is now
the strongest baseline and the one the conservative claim is measured against, which moves that
claim from 2.05x down to **1.79x**. Running it was the point: it could only ever hurt our number,
so leaving it untested would have left the weakest link unexamined.

PANEL B measures SPARROW's diversity blind spot rather than asserting it: cheap syntheses come from
shared intermediates, which come from structurally similar molecules, so its selections are LESS
diverse than the pool it draws from exactly when the budget binds.

PANEL C (added for Logs/059) closes the loophole panels A/B leave open. A and B change TWO things at
once -- a different generator AND a different way of choosing what to make -- so neither can say which
does the work. C hands the batch optimizer OUR OWN molecules and lets it choose:
  * BC-SB          -- the 21,000 distinct molecules our generator sampled, best-score-first (n=3 seeds)
  * BC-Enum-SB     -- the hardest control we can build without flow: the 131,474 molecules enumerated
                      one step from the 64 intermediates a naive best-score walk collects (n=1)
If a standard optimizer found a cheap DIVERSE library from our own molecules, our contribution would
be the generator and not the selection. It does not: it wins on cost per molecule and loses badly on
how many of those molecules differ, because it concentrates its picks on a handful of intermediates
(at a 100-reaction budget, 3 of them, 78% from one) whose products share their whole construction
history. Panel C is what makes "enumeration alone is not enough -- spreading across intermediates is"
a measurement instead of an argument.

RAW COUNTS, NOT PER-DISTINCT RATIOS, in panel C -- this is deliberate (Logs/059). At tight budgets
BC-Enum-SB yields as few as 2 distinct molecules, so any "reactions per distinct molecule" figure
derived from it moves ~2x on a single molecule. The counts themselves are stable; ratios of them are
not, so the panel plots counts and the ratio annotations are taken only at the 300-reaction budget
where every arm has tens of distinct molecules.

WE ALSO REPORT THE METRIC WE LOSE. Per molecule made, the optimizer is cheaper than us (~1.08 and
~1.01 reactions/molecule vs our 1.32): it is explicitly minimizing that and has no reason to care
about anything else. That sits in the caption rather than being left out -- the claim is specifically
about cost per DISTINCT molecule, and a paper that states where it loses is harder to dismiss.

Palette = the project's validated pair (blue #2a78d6 / orange #eb6834). `node` is unavailable on this
cluster, so the six-checks validator was not re-run here; this exact pair's recorded result is
CVD dE 24.7 protan / normal-vision 33.6 / contrast 4.30/3.12, all PASS. Only two hues are used --
the same-entity variants differ by line STYLE, never by a new hue. Panel C keeps that rule: ours is
blue, and both optimizer arms are the same orange separated by style (solid vs dotted), because they
are the same entity (SPARROW-Batching) given different material.

Run:  conda run -n rgfn python experiments/lsd_hubs/campaign/make_pipeline_headline.py
"""
import csv
import json
import statistics as st
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RES = Path("experiments/lsd_hubs/campaign/results")
OUT = RES / "paper_pipeline_headline"
OURS, THEIRS = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e8e7e3"
TARGET_MODES = 100
POOL_MODE_RATE = 0.412  # S3-GFN top-500's own mode rate (pre-flight saturation, tau=0.5)
PANEL_C_BUDGET = 300  # the deepest budget every arm in panel C reaches

# Panel C inputs. Staged copies of the sweep outputs on $SCRATCH; the loader falls back to the
# original path so the figure still builds on a machine where the staged copy is missing (campaign
# CSVs are git-ignored, so the staged copy is a convenience, never the only record).
SCRATCH_BC = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/bc_sb")
SCRATCH_RES = Path("/scratch/markymoo/rgfn_runs/lsdflow_sparrow/results")

# The competitor pipeline, one entry per REPLICATE (retrain -> MultiAiZ -> select). Seed 42 is the
# original Logs/056 run; 43/44 are produced by submit_s3gfn_replicate_routes.sh. Missing seeds are
# skipped, so this file needs no edit when they land -- the band appears on its own.
THEIRS_SELECT_SEEDS = {
    42: "s3gfn_seh_select_N500",
    43: "s3gfn_seh_seed43_select_N500",
    44: "s3gfn_seh_seed44_select_N500",
}
THEIRS_GREEDY_SEEDS = {
    42: "s3gfn_seh_greedy_N500",
    43: "s3gfn_seh_seed43_greedy_N500",
    44: "s3gfn_seh_seed44_greedy_N500",
}
BC_SB_SEEDS = {
    42: ("bc_sb_seh/bc_sb_seed42_N21000.csv", "bc_sb_seh_seed42_N21000/select_frontier.csv"),
    43: ("bc_sb_seh/bc_sb_seed43_N21000.csv", "bc_sb_seh_seed43_N21000/select_frontier.csv"),
    44: ("bc_sb_seh/bc_sb_seed44_N21000.csv", "bc_sb_seh_seed44_N21000/select_frontier.csv"),
}
BC_ENUM = ("bc_sb_seh/bc_enum_sb_seed42_N50000.csv", "bc_enum_sb_seh42_N50000/select_frontier.csv")

# Our own side, 3 seeds, at the 100-mode deliverable. These are the SPARROW-audited values (the
# like-for-like pricing the panel-A readout uses), one per reconcile summary.
OURS_SEED_RECONCILES = {
    42: "scent_seh_freefrag_k0_reconcile",
    43: "t45_seh_seed43_k0_reconcile",
    44: "t45_seh_seed44_k0_reconcile",
}

# BC-Enum-SB's mechanism, measured (Logs/059): how few intermediates its picks come from.
ENUM_CONCENTRATION = {100: (98, 3, 0.78), 300: (296, 6, 0.45)}  # budget -> (picked, hubs, top share)

# SYMMETRY FIX. Three of the four curves are already SPARROW-priced (both from-scratch arms and the
# baseline's as-run arm all end in the SPARROW MILP). Our native curve is the exception: it is our
# own count-once estimate, because that is what budget_efficiency.csv records. Reading the headline
# off it would compare count-once against SPARROW -- two different cost models -- so the readout is
# overridden with the INDEPENDENT SPARROW audit of the same library at the same 100 modes
# (Logs/056: free-frag K=0, 125 count-once vs 131 SPARROW, agreeing to 4.8%). The curve is still
# drawn from count-once for its shape; only the quoted readout is the audited value, which is what
# makes the headline ratio a like-for-like SPARROW-vs-SPARROW comparison.
OURS_NATIVE_SPARROW = 131


def _budget_curve(path, strategy, cutoff="0.5"):
    """[(reactions, modes)] from a budget_efficiency.csv."""
    pts = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["strategy"] == strategy and r["cutoff"] == cutoff:
                pts.append((int(r["cum_reactions"]), int(r["cum_modes"])))
    return sorted(pts)


def _lerp_x_at_y(pts, y):
    """Interpolate the x (reactions) at which the curve reaches y modes."""
    lo = hi = None
    for x, yy in pts:
        if yy <= y:
            lo = (x, yy)
        if yy >= y and hi is None:
            hi = (x, yy)
    if lo and hi and hi[1] != lo[1]:
        f = (y - lo[1]) / (hi[1] - lo[1])
        return lo[0] + f * (hi[0] - lo[0])
    return (hi or lo)[0] if (hi or lo) else None


def _first_existing(rel_repo, rel_scratch):
    """Prefer the staged copy under results/; fall back to the sweep's own output on $SCRATCH."""
    for p in (RES / rel_repo, SCRATCH_BC / rel_scratch):
        if p.exists():
            return p
    raise FileNotFoundError(f"neither {RES / rel_repo} nor {SCRATCH_BC / rel_scratch} exists")


def _select_frontier(path):
    """{budget: {'selected': n, 'modes': n, 'rxn_per_selected': x}} from a select_frontier.csv."""
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("milp_status") != "Optimal":
                # A truncated solve is not a datapoint about the arm's strength, so it is dropped
                # loudly rather than plotted as if it were a proven optimum.
                print(f"  [panelC] SKIP {path.name} budget={r['budget_rxns']} ({r['milp_status']})")
                continue
            out[int(r["budget_rxns"])] = {
                "selected": int(r["n_selected"]),
                "modes": int(r["n_modes"]),
                "rxn_per_selected": float(r["rxn_per_selected"]),
            }
    return out


def load_panel_c():
    """The three arms of panel C, all on OUR molecules, as raw counts per reaction budget."""
    per_seed = {s: _select_frontier(_first_existing(*p)) for s, p in BC_SB_SEEDS.items()}
    budgets = sorted(set.intersection(*(set(d) for d in per_seed.values())))
    bc_sb = []
    for b in budgets:
        modes = [per_seed[s][b]["modes"] for s in per_seed]
        sel = [per_seed[s][b]["selected"] for s in per_seed]
        rps = [per_seed[s][b]["rxn_per_selected"] for s in per_seed]
        bc_sb.append(
            {
                "budget": b,
                "modes_mean": st.mean(modes),
                "modes_sd": st.stdev(modes) if len(modes) > 1 else 0.0,
                "modes": modes,
                "selected_mean": st.mean(sel),
                "rxn_per_selected": st.mean(rps),
            }
        )
    enum = _select_frontier(_first_existing(*BC_ENUM))
    bc_enum = [{"budget": b, **v} for b, v in sorted(enum.items())]
    return bc_sb, bc_enum, len(per_seed)


def load_ours_seed_band():
    """Our 100-mode readout across 3 seeds, SPARROW-priced -- the band panel A was missing.

    The quoted panel-A readout stays the SEED-42 audit (see OURS_NATIVE_SPARROW): it is the WORST of
    the three seeds, so quoting it keeps the headline ratio conservative while the band shows the
    spread. Swapping the readout to the 3-seed mean would move the headline in our favour, and that
    is a claim change for the paper's author to make deliberately, not a side effect of drawing a
    band -- both numbers are printed and written to the CSV.
    """
    vals = {}
    for seed, d in OURS_SEED_RECONCILES.items():
        j = json.load(open(RES / d / "reconcile_summary.json"))
        for r in j["records"]:
            if r["strategy"] == "hub_batching" and r["n_modes"] == TARGET_MODES:
                vals[seed] = r["sparrow_reactions"]
    return vals


def _n_phrase(band):
    """"n=1" vs "n=3 (a ± b)" -- so the caption never claims replicates that are not on disk."""
    vals = sorted(band.values())
    if len(vals) < 2:
        return "n=1"
    return f"n={len(vals)} ({st.mean(vals):.0f} ± {st.stdev(vals):.0f})"


def _modes_at(pts, budget):
    """Modes delivered by the time `budget` reactions are spent (step function, no interpolation).

    Deliberately NOT interpolated, unlike the panel-A readout: panel C's competitor arms report the
    integer count SPARROW's MILP actually chose at that exact budget, so reading a fractional mode
    count off our curve would compare a real count against an imagined one.
    """
    reached = [m for x, m in pts if x <= budget]
    return max(reached) if reached else 0


def _assert_comparable_schema(path, fieldnames):
    """Refuse a frontier CSV that was written by a different VERSION of the frontier script.

    THIS HAPPENED (2026-08-13). This repo is one working tree shared by three agents. A co-author
    rewrote `sparrow_select_frontier.py`'s measurement while the seed-43/44 replicate jobs were
    QUEUED -- `n_modes` (modes among the selected set) became `n_modes_kept` (a pruned subset priced
    by a separate solve) and `mode_rate` was redefined -- so the jobs silently measured a different
    quantity than seed 42 and their numbers looked like a 2x seed effect. Nothing crashed.

    A band is only a band if every seed in it was measured the same way, so the schema is checked
    rather than trusted: an unexpected column set means a version difference, and averaging across it
    would manufacture a spread out of a definition change. Fail loudly and name the fix.
    """
    got = set(fieldnames or ())
    if "n_modes" in got:
        return
    drifted = sorted(got & {"n_modes_kept", "lambda_div", "cost_kept_rxns", "mean_pairwise_sim"})
    raise SystemExit(
        f"[headline] REFUSING {path}\n"
        f"  it has no `n_modes` column, so it was not written by the frontier version seed 42 was.\n"
        f"  version-specific columns present: {drifted or sorted(got)}\n"
        f"  Re-run that seed's frontier against the pinned script version before plotting a band —\n"
        f"  see Logs/061 Method 6. Mixing versions turns a definition change into a fake seed effect."
    )


def _competitor_seed_curves(subdirs, filename):
    """{seed: [(reactions, modes)]} for every seed whose frontier CSV exists.

    THE COMPETITOR'S MISSING ERROR BAR. Every other arm in the benchmark is replicated; theirs was
    one training run plus one MultiAiZ planning run (F1's open item). The replicates each repeat all
    three stages -- retrain, plan, select -- because MultiAiZ is set-based and its routes depend on
    the pool, so a "seed" here is a whole pipeline re-run, not a re-solve.

    Absent seeds are simply skipped, so this reduces to the original single-seed behaviour until the
    replicate jobs land, and picks them up with no further edit once they do.
    """
    curves = {}
    for seed, sub in subdirs.items():
        for root in (RES, SCRATCH_RES):
            p = root / sub / filename
            if p.exists():
                pts = []
                with open(p) as fh:
                    rdr = csv.DictReader(fh)
                    _assert_comparable_schema(p, rdr.fieldnames)
                    for r in rdr:
                        if r.get("milp_status") not in (None, "", "Optimal"):
                            continue  # a truncated solve is not a frontier point
                        pts.append((int(float(r["used_rxns"])), int(float(r["n_modes"]))))
                curves[seed] = sorted(pts)
                break
    return curves


def _mean_curve(curves, over="x"):
    """Average across seeds along whichever axis was actually HELD FIXED by the experiment.

    This is not a cosmetic choice. The two competitor arms are swept along different axes:

      * SPARROW-Batching sweeps the reaction BUDGET, and `used_rxns` equals that budget, so
        reactions are the fixed grid and the modes delivered vary -> average modes at shared x.
      * the diversity-aware greedy sweeps MODE POINTS and asks SPARROW what they cost, so the mode
        count is the fixed grid and reactions vary -> average reactions at shared y.

    Averaging the greedy arm on x would intersect on `used_rxns`, which no two seeds share, silently
    yielding an EMPTY curve and a missing readout. Both ladders are pinned per seed by
    submit_s3gfn_replicate_routes.sh, so intersecting on the right axis keeps every plotted point a
    true n-seed mean instead of mixing seed counts along one curve.
    """
    if not curves:
        return []
    if over == "x":
        xs = sorted(set.intersection(*(set(x for x, _ in c) for c in curves.values())))
        return [(x, st.mean(next(m for xx, m in c if xx == x) for c in curves.values())) for x in xs]
    ys = sorted(set.intersection(*(set(y for _, y in c) for c in curves.values())))
    return sorted(
        (st.mean(next(x for x, yy in c if yy == y) for c in curves.values()), y) for y in ys
    )


def load():
    ours_native = _budget_curve(RES / "scent_seh_freefrag/budget_efficiency.csv", "hub_batching")
    ours_scratch = _budget_curve(
        RES / "scent_seh_sparrow_headline/budget_efficiency.csv", "hub_batching"
    )
    # baseline, as actually run: MultiAiZ routes + SPARROW doing its own selection
    real_seeds = _competitor_seed_curves(THEIRS_SELECT_SEEDS, "select_frontier.csv")
    theirs_real = _mean_curve(real_seeds)
    # Panel B's mode-rate reads off seed 42, the one seed for which the pool's own 41% rate was
    # measured; averaging a rate against a single-seed reference line would mix populations.
    rates = []
    with open(RES / "s3gfn_seh_select_N500/select_frontier.csv") as fh:
        for r in csv.DictReader(fh):
            rates.append((int(float(r["used_rxns"])), float(r["mode_rate"])))
    rates.sort()
    # baseline at its STRONGEST: MultiAiZ routes + a diversity-aware greedy selection (SPARROW prices)
    greedy_seeds = _competitor_seed_curves(THEIRS_GREEDY_SEEDS, "greedy_frontier.csv")
    theirs_greedy = _mean_curve(greedy_seeds, over="y")  # mode points are the fixed grid
    # baseline, from-scratch re-derivation (entry 048's T3.2 arm); curve rows are [modes, reactions]
    summ = json.load(open(RES / "s3gfn_seh/s3gfn_frontier_summary.json"))
    theirs_scratch = sorted(
        (int(rx), int(md)) for md, rx in summ["curves"]["0.50"] if rx is not None and md is not None
    )
    return (ours_native, ours_scratch, theirs_real, theirs_scratch, theirs_greedy, rates,
            real_seeds, greedy_seeds)  # fmt: skip


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (ours_native, ours_scratch, theirs_real, theirs_scratch, theirs_greedy, rates,
     real_seeds, greedy_seeds) = load()  # fmt: skip
    bc_sb, bc_enum, n_bc_seeds = load_panel_c()
    ours_band = load_ours_seed_band()

    ours_native_countonce = _lerp_x_at_y(ours_native, TARGET_MODES)
    reads = {
        "ours_native": OURS_NATIVE_SPARROW,  # audited value, not the count-once curve (see above)
        "ours_scratch": _lerp_x_at_y(ours_scratch, TARGET_MODES),
        "theirs_real": _lerp_x_at_y(theirs_real, TARGET_MODES),
        "theirs_scratch": _lerp_x_at_y(theirs_scratch, TARGET_MODES),
        "theirs_greedy": _lerp_x_at_y(theirs_greedy, TARGET_MODES) if theirs_greedy else None,
    }
    print(
        f"  (our native count-once curve reads {ours_native_countonce:.0f}; "
        f"quoting the SPARROW audit {OURS_NATIVE_SPARROW} for like-for-like pricing)"
    )
    print(f"reactions to reach {TARGET_MODES} modes:")
    for k, v in reads.items():
        print(f"   {k:<15} " + ("(absent)" if v is None else f"{v:.0f}"))
    headline = reads["theirs_real"] / reads["ours_native"]
    # conservative = vs the baseline's BEST result over every configuration we have tested
    best_baseline = min(v for k, v in reads.items() if k.startswith("theirs") and v)
    conservative = best_baseline / reads["ours_native"]
    print(f"   headline (solid vs solid)      {headline:.2f}x")
    print(f"   conservative (vs their best)   {conservative:.2f}x")

    band = sorted(ours_band.values())
    band_mean, band_sd = st.mean(band), st.stdev(band)
    print(
        f"   our side, {len(band)} seeds (SPARROW-priced): {band} -> {band_mean:.1f} +/- {band_sd:.1f}"
        f"  [quoted readout stays the seed-42 value {reads['ours_native']:.0f}, the worst of the three;"
        f" using the mean would read {reads['theirs_real'] / band_mean:.2f}x]"
    )

    # The competitor's own seed spread, once its replicates exist. Reported per seed rather than only
    # as a mean, because with n=3 the spread is the whole point of running them.
    def _seed_readouts(curves):
        vals = {s: _lerp_x_at_y(c, TARGET_MODES) for s, c in curves.items()}
        return {s: v for s, v in vals.items() if v is not None}

    theirs_real_band = _seed_readouts(real_seeds)
    theirs_greedy_band = _seed_readouts(greedy_seeds)
    for name, b in (("MultiAiZ+SB", theirs_real_band), ("MultiAiZ+greedy", theirs_greedy_band)):
        vals = sorted(b.values())
        if len(vals) > 1:
            print(
                f"   competitor {name}, {len(vals)} seeds: "
                f"{[round(v) for v in vals]} -> {st.mean(vals):.1f} +/- {st.stdev(vals):.1f}"
            )
        else:
            print(f"   competitor {name}: n=1 (replicates not yet on disk)")

    ours_at_c = _modes_at(ours_native, PANEL_C_BUDGET)
    sb_at_c = next(r for r in bc_sb if r["budget"] == PANEL_C_BUDGET)
    enum_at_c = next(r for r in bc_enum if r["budget"] == PANEL_C_BUDGET)
    print(f"panel C, at a {PANEL_C_BUDGET}-reaction budget (raw counts, sEH):")
    print(f"   hub-batching (ours)  {ours_at_c} distinct  ({ours_at_c} molecules made)")
    print(
        f"   BC-SB (n={n_bc_seeds})          {sb_at_c['modes_mean']:.1f} +/- {sb_at_c['modes_sd']:.1f}"
        f" distinct  ({sb_at_c['selected_mean']:.0f} molecules made)"
    )
    print(
        f"   BC-Enum-SB (n=1)     {enum_at_c['modes']} distinct  "
        f"({enum_at_c['selected']} molecules made)"
    )
    sb_ratio = ours_at_c / sb_at_c["modes_mean"]
    enum_ratio = ours_at_c / enum_at_c["modes"]
    print(f"   -> {sb_ratio:.1f}x more distinct than BC-SB, {enum_ratio:.1f}x more than BC-Enum-SB")

    fig = plt.figure(figsize=(14.6, 7.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.52, 1.0], wspace=0.22, hspace=0.52)
    axA = fig.add_subplot(gs[:, 0])
    axB, axC = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])

    # ---------------- Panel A: the 2x2 frontier ----------------
    axA.set_facecolor(SURFACE)
    series = [
        (ours_scratch, OURS, "--", "LSD-Flow — routes re-derived from scratch", 1.9),
        (theirs_scratch, THEIRS, "--", "S3-GFN — routes re-derived from scratch", 1.9),
        (ours_native, OURS, "-", "LSD-Flow — its own routes (as run)", 2.4),
        (theirs_real, THEIRS, "-", "S3-GFN → MultiAiZ → SPARROW selects (as run)", 2.4),
    ]
    if theirs_greedy:
        series.append(
            (
                theirs_greedy,
                THEIRS,
                ":",
                "S3-GFN → MultiAiZ → diversity-aware greedy (their best)",
                2.4,
            )
        )
    for pts, c, ls, lab, lw in series:
        axA.plot(
            [p[0] for p in pts], [p[1] for p in pts],
            ls, color=c, linewidth=lw, label=lab, zorder=3,
            alpha=0.55 if ls == "--" else 1.0,
            marker="o" if len(pts) < 20 else None, markersize=4.5,
            markeredgecolor=SURFACE, markeredgewidth=1.2,
        )  # fmt: skip

    axA.axhline(TARGET_MODES, color=INK3, linewidth=1.0, linestyle=":", zorder=2)
    axA.text(
        548, TARGET_MODES + 4, f"the deliverable: {TARGET_MODES} modes",
        fontsize=8.2, color=INK2, va="bottom", ha="right",
    )  # fmt: skip
    # selective direct labels: ONLY the four readouts that the comparisons are made from
    for key, c, dy in (
        ("ours_native", OURS, -13), ("ours_scratch", OURS, -13),
        ("theirs_real", THEIRS, -13), ("theirs_scratch", THEIRS, -13),
        ("theirs_greedy", THEIRS, -13),
    ):  # fmt: skip
        x = reads.get(key)
        if x is None:
            continue
        axA.plot([x], [TARGET_MODES], marker="o", markersize=9, color=c,
                 markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)  # fmt: skip
        axA.annotate(f"{x:.0f}", (x, TARGET_MODES), textcoords="offset points",
                     xytext=(0, dy), ha="center", fontsize=9.4, color=INK,
                     fontweight="semibold", zorder=6)  # fmt: skip

    # Our side's SEED SPREAD at the deliverable. Drawn, not asserted: the quoted 131 is one seed of
    # three and happens to be the worst of them, so showing the band is strictly against our own
    # interest and belongs on the figure.
    axA.errorbar(
        [band_mean], [TARGET_MODES],
        xerr=[[band_mean - band[0]], [band[-1] - band_mean]],
        fmt="none", ecolor=OURS, elinewidth=1.5, capsize=3.5, capthick=1.5, alpha=0.9, zorder=4,
    )  # fmt: skip
    axA.annotate(
        f"{len(band)} seeds: {band_mean:.0f} ± {band_sd:.0f}",
        (band[0], TARGET_MODES), textcoords="offset points", xytext=(-9, -11),
        ha="right", fontsize=8.0, color=INK2, zorder=6,
    )  # fmt: skip
    # The competitor gets the SAME treatment as soon as its replicates exist -- an error bar drawn
    # from its own seeds. Until then nothing is drawn and the caption says n=1, rather than an
    # unmarked point implying a precision it does not have.
    for cb, dy in ((theirs_real_band, -27), (theirs_greedy_band, -27)):
        vals = sorted(cb.values())
        if len(vals) < 2:
            continue
        m = st.mean(vals)
        axA.errorbar(
            [m], [TARGET_MODES], xerr=[[m - vals[0]], [vals[-1] - m]], fmt="none", ecolor=THEIRS,
            elinewidth=1.5, capsize=3.5, capthick=1.5, alpha=0.9, zorder=4,
        )  # fmt: skip
        axA.annotate(
            f"{len(vals)} seeds: {m:.0f} ± {st.stdev(vals):.0f}", (m, TARGET_MODES),
            textcoords="offset points", xytext=(0, dy), ha="center", fontsize=8.0, color=INK2,
            zorder=6,
        )  # fmt: skip

    axA.annotate(
        "", xy=(reads["ours_native"], TARGET_MODES + 26),
        xytext=(reads["theirs_real"], TARGET_MODES + 26),
        arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.3),
    )  # fmt: skip
    axA.text(
        (reads["ours_native"] + reads["theirs_real"]) / 2, TARGET_MODES + 32,
        f"{headline:.2f}× fewer reactions", ha="center", fontsize=10.2,
        color=INK, fontweight="bold",
    )  # fmt: skip

    axA.set_xlim(0, 560)
    axA.set_ylim(0, 245)
    axA.set_xlabel("reactions (synthetic effort)  ← lower is better", fontsize=9, color=INK2)
    axA.set_ylabel("distinct molecule families delivered", fontsize=9, color=INK2)
    axA.legend(loc="lower right", frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=2.2)
    axA.set_title(
        "A · Same deliverable, two pricing regimes", fontsize=10, color=INK,
        fontweight="semibold", loc="left", pad=8,
    )  # fmt: skip

    # ---------------- Panel B: the measured diversity blind spot ----------------
    axB.set_facecolor(SURFACE)
    axB.plot(
        [r[0] for r in rates], [r[1] for r in rates], "-o", color=THEIRS, linewidth=2.2,
        markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3,
        label="SPARROW's selection",
    )  # fmt: skip
    axB.axhline(POOL_MODE_RATE, color=INK3, linewidth=1.6, linestyle="--", zorder=2,
                label="the pool it draws from")  # fmt: skip
    worst = min(rates, key=lambda r: r[1])
    axB.annotate(
        f"{worst[1]:.0%} where the\nbudget binds", (worst[0], worst[1]),
        textcoords="offset points", xytext=(14, -6), fontsize=8.6, color=INK,
        fontweight="medium",
    )  # fmt: skip
    axB.text(1010, POOL_MODE_RATE + 0.012, f"{POOL_MODE_RATE:.0%}", fontsize=8.6,
             color=INK2, ha="right")  # fmt: skip
    axB.set_ylim(0, 0.52)
    axB.set_xlabel("reaction budget", fontsize=9, color=INK2)
    axB.set_ylabel("fraction of picks that are distinct", fontsize=9, color=INK2)
    axB.legend(loc="lower right", frameon=False, fontsize=8.4, labelcolor=INK2, handlelength=2.2)
    axB.set_title(
        "B · Selections are less diverse than the pool",
        fontsize=10, color=INK, fontweight="semibold", loc="left", pad=8,
    )  # fmt: skip

    # ---------------- Panel C: the optimizer given OUR OWN molecules ----------------
    # Raw counts on both axes (Logs/059): at tight budgets BC-Enum-SB delivers 2-3 distinct
    # molecules, so a per-distinct ratio would swing ~2x on one molecule. Ratios appear only as the
    # two annotations at the 300-reaction budget, where every arm has tens of distinct molecules.
    axC.set_facecolor(SURFACE)
    c_budgets = [r["budget"] for r in bc_sb]
    axC.plot(
        c_budgets, [_modes_at(ours_native, b) for b in c_budgets], "-", color=OURS, linewidth=2.4,
        marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
        label="hub-batching (ours, n=1)",
    )  # fmt: skip
    sb_mean = [r["modes_mean"] for r in bc_sb]
    sb_lo = [r["modes_mean"] - r["modes_sd"] for r in bc_sb]
    sb_hi = [r["modes_mean"] + r["modes_sd"] for r in bc_sb]
    axC.fill_between(c_budgets, sb_lo, sb_hi, color=THEIRS, alpha=0.18, linewidth=0, zorder=2)
    axC.plot(
        c_budgets, sb_mean, "-", color=THEIRS, linewidth=2.4, marker="o", markersize=5,
        markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4,
        label=f"BC-SB (n={n_bc_seeds})",
    )  # fmt: skip
    axC.plot(
        [r["budget"] for r in bc_enum], [r["modes"] for r in bc_enum], ":", color=THEIRS,
        linewidth=2.4, marker="s", markersize=4.5, markeredgecolor=SURFACE, markeredgewidth=1.2,
        zorder=4, label="BC-Enum-SB (n=1)",
    )  # fmt: skip

    # the readout: raw distinct counts at the shared 300-reaction budget
    for y, txt, c in (
        (ours_at_c, f"{ours_at_c}", OURS),
        (sb_at_c["modes_mean"], f"{sb_at_c['modes_mean']:.0f}", THEIRS),
        (enum_at_c["modes"], f"{enum_at_c['modes']}", THEIRS),
    ):
        axC.annotate(
            txt, (PANEL_C_BUDGET, y), textcoords="offset points", xytext=(7, -3), ha="left",
            fontsize=9.2, color=INK, fontweight="semibold", zorder=6,
        )  # fmt: skip
    axC.annotate(
        "", xy=(PANEL_C_BUDGET - 8, ours_at_c), xytext=(PANEL_C_BUDGET - 8, sb_at_c["modes_mean"]),
        arrowprops=dict(arrowstyle="<->", color=INK2, linewidth=1.1),
    )  # fmt: skip
    axC.text(
        PANEL_C_BUDGET - 16, (ours_at_c + sb_at_c["modes_mean"]) / 2,
        f"{sb_ratio:.1f}× / {enum_ratio:.1f}×\nmore distinct",
        ha="right", va="center", fontsize=8.6, color=INK, fontweight="semibold", linespacing=1.3,
    )  # fmt: skip
    picked, n_hubs, top_share = ENUM_CONCENTRATION[PANEL_C_BUDGET]
    axC.text(
        0.03, 0.99,
        f"same {PANEL_C_BUDGET} reactions, MORE molecules made:\n"
        f"{sb_at_c['selected_mean']:.0f} and {enum_at_c['selected']} vs our {ours_at_c} — but its "
        f"{picked} picks come\nfrom {n_hubs} intermediates, {top_share:.0%} from one",
        transform=axC.transAxes, fontsize=7.8, color=INK2, va="top", ha="left", linespacing=1.5,
    )  # fmt: skip
    axC.set_xlim(30, 345)
    axC.set_ylim(0, 310)
    axC.set_xlabel("reaction budget", fontsize=9, color=INK2)
    axC.set_ylabel("distinct molecules delivered", fontsize=9, color=INK2)
    # Legend ABOVE the axes: the blue curve sweeps this panel's diagonal and the two orange arms hug
    # the floor, so every in-axes position collides with a series.
    axC.legend(loc="lower left", frameon=False, fontsize=7.8, labelcolor=INK2, handlelength=2.0,
               ncol=3, columnspacing=1.6, bbox_to_anchor=(0.0, 1.0))  # fmt: skip
    axC.set_title(
        "C · Handed our molecules, a cost-only optimizer concentrates",
        fontsize=10, color=INK, fontweight="semibold", loc="left", pad=24,
    )  # fmt: skip

    for ax in (axA, axB, axC):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK3)
            ax.spines[s].set_linewidth(0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(colors=INK2, labelsize=8.6, length=0)

    fig.text(
        0.008, 0.975,
        "Priced the way each pipeline is actually run, LSD-Flow delivers the same 100-family "
        "library for 3.1× fewer reactions —\nthe opposite of what the earlier from-scratch pricing "
        "showed, because that regime measured whose chemistry the planner recognised",
        fontsize=11.4, color=INK, fontweight="semibold", ha="left", va="top", linespacing=1.45,
    )  # fmt: skip
    # Caption paragraphs are wrapped to the figure width at draw time rather than hand-broken, so
    # editing the wording cannot silently push a line off the right edge.
    caption = [
        f"A/B — SCENT sEH vs S3-GFN (500 candidates, 478 routed) · reward > 7.0 · τ = 0.5 · seed 42. "
        f"Both arms priced by SPARROW; our readout is the independent SPARROW audit of the same "
        f"library (our own count-once estimate reads {ours_native_countonce:.0f}, agreeing to 4.8%). "
        f"Our side also has {len(band)} seeds ({band_mean:.0f} ± {band_sd:.0f}, error bar) while the "
        f"competitor is {_n_phrase(theirs_real_band)}; the quoted {reads['ours_native']:.0f} is the "
        f"WORST of our three seeds, "
        f"so the ratio drawn is the conservative one. Conservative reading — our "
        f"{reads['ours_native']:.0f} vs the baseline's STRONGEST tested configuration "
        f"({best_baseline:.0f}: MultiAiZ + a diversity-aware selection) = {conservative:.2f}×.",
        f"C — the same optimizer on OUR molecules: BC-SB over 21,001 distinct candidates "
        f"(n={n_bc_seeds} seeds); BC-Enum-SB over 131,474 molecules enumerated from the 64 "
        f"intermediates a naive best-score walk collects (n=1). Raw counts, not per-distinct ratios: "
        f"at tight budgets BC-Enum-SB delivers 2–3 distinct molecules, so a ratio there swings ~2× on "
        f"one molecule. Ours in C is the seed-42 curve, count-once priced (the SPARROW audit of the same "
        f"libraries agrees to 0.8–4.8% across our three seeds).",
        f"WHERE WE LOSE, plainly — per molecule MADE the optimizer is cheaper: "
        f"{sb_at_c['rxn_per_selected']:.2f} (BC-SB) and {enum_at_c['rxn_per_selected']:.2f} "
        f"(BC-Enum-SB) reactions/molecule versus our 1.32. It is explicitly minimizing that and has "
        f"no reason to care about anything else; the claim is cost per DISTINCT molecule. Other "
        f"asymmetries reported, not matched — pool 500 vs 26,069; surrogate calls ~500 vs 155,764; "
        f"route planning 2.25 h vs 0.",
    ]
    fig.text(
        0.008, 0.018, "\n".join(textwrap.fill(par, 252) for par in caption),
        fontsize=7.4, color=INK2, ha="left", va="bottom", linespacing=1.55,
    )  # fmt: skip
    fig.subplots_adjust(left=0.055, right=0.988, top=0.855, bottom=0.275)

    with open(OUT / "pipeline_headline.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["series", "pricing_regime", f"reactions_for_{TARGET_MODES}_modes", "source"])
        w.writerow(["LSD-Flow", "native (as run)", f"{reads['ours_native']:.0f}",
                    "results/scent_seh_freefrag/budget_efficiency.csv"])  # fmt: skip
        w.writerow(["LSD-Flow", "from-scratch", f"{reads['ours_scratch']:.0f}",
                    "results/scent_seh_sparrow_headline/budget_efficiency.csv (Logs/041)"])  # fmt: skip
        w.writerow(["S3-GFN", "MultiAiZ + SPARROW selects (as run)", f"{reads['theirs_real']:.0f}",
                    "results/s3gfn_seh_select_N500/select_frontier.csv (Logs/056)"])  # fmt: skip
        w.writerow(["S3-GFN", "from-scratch", f"{reads['theirs_scratch']:.0f}",
                    "results/s3gfn_seh/s3gfn_frontier_summary.json (Logs/048)"])  # fmt: skip
        w.writerow(["ratio", "headline (solid vs solid)", f"{headline:.3f}", ""])
        if reads.get("theirs_greedy"):
            w.writerow(
                [
                    "S3-GFN",
                    "MultiAiZ + diversity-aware greedy (their BEST)",
                    f"{reads['theirs_greedy']:.0f}",
                    "results/s3gfn_seh_greedy_N500/greedy_frontier.csv (Logs/056)",
                ]
            )
        w.writerow(
            ["ratio", "conservative (vs their strongest tested config)", f"{conservative:.3f}", ""]
        )
        w.writerow(["LSD-Flow", f"native (as run), {len(band)}-seed SPARROW band",
                    f"{band_mean:.1f} +/- {band_sd:.1f}  seeds=" + "/".join(str(v) for v in band),
                    "results/{scent_seh_freefrag,t45_seh_seed43,t45_seh_seed44}_k0_reconcile"])  # fmt: skip
        w.writerow(["ratio", "headline if the 3-seed mean were quoted instead of seed 42",
                    f"{reads['theirs_real'] / band_mean:.3f}", "NOT the quoted headline"])  # fmt: skip

    # Panel C is a different experiment from A/B (our molecules, three selectors), so it gets its
    # own record rather than being squeezed into the reactions-for-100-modes schema above.
    with open(OUT / "panel_c_our_molecules.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["arm", "n_seeds", "budget_rxns", "molecules_made", "distinct_molecules",
             "distinct_sd", "rxn_per_molecule_made", "source"]
        )  # fmt: skip
        for b in c_budgets:
            w.writerow(["hub_batching (ours)", 1, b, _modes_at(ours_native, b),
                        _modes_at(ours_native, b), "", "", "count-once curve (seed 42)"])  # fmt: skip
        for r in bc_sb:
            w.writerow(["BC-SB", n_bc_seeds, r["budget"], f"{r['selected_mean']:.1f}",
                        f"{r['modes_mean']:.2f}", f"{r['modes_sd']:.2f}",
                        f"{r['rxn_per_selected']:.3f}",
                        "seeds=" + "/".join(str(m) for m in r["modes"])])  # fmt: skip
        for r in bc_enum:
            w.writerow(["BC-Enum-SB", 1, r["budget"], r["selected"], r["modes"], "",
                        f"{r['rxn_per_selected']:.3f}", "bc_enum_sb_seh42_N50000"])  # fmt: skip
        w.writerow(["ratio", "", PANEL_C_BUDGET, "", f"ours/BC-SB={sb_ratio:.2f}", "",
                    f"ours/BC-Enum-SB={enum_ratio:.2f}", "raw distinct counts (Logs/059)"])  # fmt: skip
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"pipeline_headline.{ext}", dpi=300, facecolor=SURFACE)
    print(f"wrote {OUT}/pipeline_headline.png + .pdf + .csv")


if __name__ == "__main__":
    main()
