#!/usr/bin/env python
"""What do the reward filter and the diversity filter each buy? (Logs/054)

WHY this exists. A "mode" — the denominator of every cost number we publish — is defined by TWO
filters, both living in ``glue/samplers/lsdflow/mode_select.py::DiverseThresholdModeSelector``:

  1. a **reward gate** (``reward >= τ``): the molecule has to be worth making, and
  2. a **diversity filter** (Tanimoto ``<= cutoff`` vs every mode already accepted, ECFP r=3/2048):
     it has to be a molecule we do not already have.

``docs/paper_planning/lsd-flow-publication-strategy.md`` §2.4/§2.5 says why both must be *justified*
rather than asserted: reactions/mode is a ratio we also happen to win on, so a reviewer will ask what
stops a strategy from lowering it by delivering trivial molecules. Each filter blocks one of those two
exploits — the reward gate blocks "cheap because it's weak" (the depth-0 catalog degenerate optimum of
§2.5), the diversity filter blocks "cheap because it's the same molecule 300 times". Logs/026
(``dropoff/funnel.py``) measured how many children each filter *kills*; nothing measured what the
delivered **library** looks like when one is removed. That is this driver.

WHAT it measures. Removing a filter is the permissive endpoint of that filter's own knob, so the
ablation is scanned rather than asserted: two 1-D scans crossing at the headline operating point
(τ=7.0, cutoff=0.5), plus the both-off corner —

    reward-filter scan:     τ ∈ {off, 4.0 … 8.0}          at cutoff = 0.5
    diversity-filter scan:  cutoff ∈ {off, 0.9 … 0.3}     at τ = 7.0
    corner:                 both off

Each cell runs BOTH selection strategies to a fixed **300-mode** target, so every arm delivers a
library of the same size and the quality readouts are not confounded by set size (the design
constraint Logs/051 pins down). Four readouts per library:

  * **reactions / mode** — the headline cost, exactly as the campaign counts it. What relaxing a
    filter *appears* to buy.
  * **library reward** (median, p10, fraction over each calibrated bar) — the quality the reward
    filter protects (§2.4's numerator guard).
  * **library self-similarity** (mean pairwise Tanimoto; plus paper-comparable modes at 0.7/0.5,
    reward-blind Butina clusters, Murcko scaffolds) — the quality the diversity filter protects.
  * **reactions / QUALIFIED mode** — the honest cost: re-impose the canonical mode definition
    (τ=7.0 AND mutual Tanimoto ≤ 0.5) on the delivered library and divide by what survives. For the
    unablated arm this equals reactions/mode by construction (a self-check the driver asserts); for
    an ablated arm it is what the cheap-looking library really costs per molecule you would keep.

Nested synthetic depth is carried alongside (§2.4 asks for depth as well as reward): SCENT's fully
nested ``num_reactions`` for a sampled candidate, ``hub.depth + 1`` for an enumerated child — the same
scale, so the two strategies' libraries are comparable.

Ablation semantics (both deliberate, both reported):
  * *no diversity filter* = :class:`RewardOnlyModeSelector` — keeps everything clearing the gate but
    never the same canonical SMILES twice. The knob under test is the *similarity* criterion, not
    deduplication; ``duplicates_suppressed`` records how often that mattered.
  * *no reward filter* = ``reward_threshold=None``. The strategies still feed candidates
    best-reward-first — that ordering is the *strategy*, not the filter — so this isolates the hard
    gate. Expect best-candidate to barely move (Logs/052 found it τ-invariant at a fixed budget) and
    hub-batching to move, since it walks hubs in flow-rank order and a weak hub's children become
    acceptable.

Cost: pure CPU, no GPU/model/oracle — it re-scores the cached enumeration (rewards already in
``enum_children.json``). Pools load once; ~1 s per strategy-run, so the whole scan is a couple of
minutes on a login node.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/filter_ablation/filter_ablation_scan.py \
        --analysis-dir  /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189 \
        --enum-children /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enum_children.json \
        --snapshot /scratch/.../scent_seh/2026-07-10_17-28-06/additional_fragments/fragments_4000.json \
        --tag scent_seh

Output: ``results/<tag>/{filter_ablation_results.csv,filter_ablation.json,filter_ablation.png,.pdf}``.
``--plot-only`` redraws the figure from the committed CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for p in (str(REPO), str(REPO / "experiments" / "lsd_hubs" / "campaign")):
    if p not in sys.path:
        sys.path.insert(0, p)

import run_campaign as RC  # noqa: E402  (experiments/lsd_hubs/campaign/run_campaign.py)

from glue.samplers.lsdflow.campaign import RANK_METHODS, rank_fragments  # noqa: E402
from glue.samplers.lsdflow.child_select import make_child_policy  # noqa: E402
from glue.samplers.lsdflow.mode_select import (  # noqa: E402
    DiverseThresholdModeSelector,
    RewardOnlyModeSelector,
)
from validation.lsdflow.metrics.cost.dynamic_amortization import (  # noqa: E402
    load_cost_table_from_snapshot,
)
from validation.lsdflow.metrics.diversity import (  # noqa: E402
    count_butina_clusters,
    count_modes,
    ecfp,
    mean_pairwise_similarity,
    murcko_scaffold,
    unique_scaffolds,
)
from validation.lsdflow.plot_style import ideal_marker  # noqa: E402

STRATS = ("hub_batching", "best_candidate")
# The campaign's established identity colours — this figure IS a strategy comparison, so it uses them.
STRAT_STYLE = {
    "hub_batching": dict(color="#2a78d6", marker="o", ls="-", label="Hub batching"),
    "best_candidate": dict(color="#eb6834", marker="s", ls="--", label="Best candidate (previous)"),
}
# The both-filters-off corner belongs to neither scan line, so it is drawn as its own isolated marker.
CORNER_STYLE = dict(marker="D", ms=7, mfc="none", mew=1.6, ls="none")
OFF = None  # the permissive endpoint of either knob ("filter removed")
# The repo gitignores ``*.csv`` except ``experiments/**/*_results.csv`` (same convention as
# ``dropoff/funnel_<tag>_results.csv``), so the machine-readable artifact has to carry that suffix
# to be committable alongside the figures.
CSV_NAME = "filter_ablation_results.csv"
OFF_LABEL = "off"
INK, MUTED = "#1a1a18", "#7a7a72"
# Diversity readouts use the paper-comparable 0.7 and the campaign's own 0.5, both reported.
MODE_SIMILARITIES = (0.7, 0.5)
# Aligns Butina's "same family" predicate with sphere exclusion's, so the two counts answer the SAME
# question and can be read side by side. Greedy rejects a molecule when similarity **> c**; Butina
# treats two molecules as neighbours when distance **<= 1 − c**, i.e. similarity **>= c**. The rules
# therefore disagree on exactly the boundary — and a library *built* at cutoff c piles pairs up on that
# boundary (the unablated 300-molecule library here has 36 pairs at exactly 0.500 and none above), so
# an unaligned Butina reads 266 of 300 and looks like a diversity deficit that is not there. Nudging
# the threshold by an epsilon shrinks the neighbour radius just below the boundary, making Butina's
# predicate "similarity > c" too. The shared metric in ``validation.lsdflow.metrics.diversity`` keeps
# its own semantics untouched (Logs/051's numbers stand).
_BUTINA_EPS = 1e-9


# ----------------------------------------------------------------------------- knob parsing
def parse_knob(spec: str) -> list:
    """``"off,4:8:0.5"`` -> ``[None, 4.0, 4.5, ...]``. Comma-separated terms; ``off``/``none`` is the
    ablation (filter removed) and sorts first; a term with two colons is an inclusive
    ``lo:hi:step`` range. Numeric values are returned ascending, deduped."""
    off = False
    vals: list = []
    for term in spec.split(","):
        term = term.strip().lower()
        if not term:
            continue
        if term in ("off", "none", "-"):
            off = True
        elif ":" in term:
            lo, hi, step = (float(x) for x in term.split(":"))
            v = lo
            while (step > 0 and v <= hi + 1e-9) or (step < 0 and v >= hi - 1e-9):
                vals.append(round(v, 4))
                v += step
        else:
            vals.append(round(float(term), 4))
    out = sorted(dict.fromkeys(vals))
    return ([OFF] if off else []) + out


def knob_label(v) -> str:
    return OFF_LABEL if v is OFF else f"{v:g}"


# ----------------------------------------------------------------------------- the cells
def build_cells(taus, sims, op_tau: float, op_sim: float) -> list:
    """The cross design: one scan per filter through the operating point, plus the both-off corner.

    Each entry is ``{"tau", "similarity", "scans"}`` where ``scans`` names which scan line(s) the cell
    belongs to — the operating point sits on both, and is therefore measured once and drawn twice.
    """
    cells: dict = {}

    def add(tau, sim, scan):
        key = (tau, sim)
        cells.setdefault(key, {"tau": tau, "similarity": sim, "scans": []})
        cells[key]["scans"].append(scan)

    for tau in taus:
        add(tau, op_sim, "reward")
    for sim in sims:
        add(op_tau, sim, "diversity")
    add(OFF, OFF, "corner")  # both filters removed — the interaction check
    return list(cells.values())


def make_selector_factory(tau, sim, higher_is_better: bool, created: list):
    """Selector factory for one ablation cell + the list it records its instances in.

    ``sim is OFF`` swaps in :class:`RewardOnlyModeSelector` (no similarity test, exact duplicates
    still dropped); ``tau is OFF`` passes ``reward_threshold=None`` (no gate). With neither off this
    builds exactly the selector the strategies build for themselves, so the unablated cell is
    bit-identical to a plain ``run_campaign.py`` invocation. ``created`` lets the caller read state
    off the selector afterwards (duplicate counts), since the strategy builds it inside ``run()``.
    """

    def factory():
        sel = (
            RewardOnlyModeSelector(tau, higher_is_better)
            if sim is OFF
            else DiverseThresholdModeSelector(tau, sim, higher_is_better)
        )
        created.append(sel)
        return sel

    return factory


# ----------------------------------------------------------------------------- readouts
class MolCache:
    """ECFP + Murcko cache. Libraries overlap heavily across the ~19 cells, so each distinct molecule
    is fingerprinted once for the whole scan."""

    def __init__(self):
        self._fp: dict = {}
        self._scaf: dict = {}

    def fps(self, smiles):
        for s in smiles:
            if s not in self._fp:
                self._fp[s] = ecfp(s)
        return [self._fp[s] for s in smiles]

    def scaffolds(self, smiles):
        for s in smiles:
            if s not in self._scaf:
                self._scaf[s] = murcko_scaffold(s)
        return [self._scaf[s] for s in smiles]

    def __len__(self):
        return len(self._fp)


def _sum_or_none(values) -> Optional[int]:
    """Sum, or ``None`` if nothing reported a value (not-applicable, distinct from a measured zero)."""
    kept = [v for v in values if v is not None]
    return sum(kept) if kept else None


def _pct(sorted_vals, q: float) -> Optional[float]:
    """Simple nearest-rank percentile (``q`` in [0,1]) — no numpy, no interpolation games."""
    if not sorted_vals:
        return None
    i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[i]


def qualified_modes(smiles, rewards, *, tau, sim, higher_is_better, fps) -> int:
    """How many of the delivered molecules survive the CANONICAL mode definition (gate ``tau`` +
    greedy sphere exclusion at ``sim``). This is the ablation's honest yield: an arm that skipped a
    filter still has to face it here."""
    return count_modes(
        smiles,
        rewards,
        higher_is_better=higher_is_better,
        reward_threshold=tau,
        similarity_threshold=sim,
        fps=fps,
    )


def library_readouts(result, depth_of, cache: MolCache, args) -> dict:
    """Cost + quality of one delivered library. ``depth_of(point) -> int|None`` gives a member's
    fully-nested synthetic depth."""
    pts = result.accepted
    smiles = [p.smiles for p in pts]
    rewards = [p.reward for p in pts]
    fps = cache.fps(smiles)
    n = len(pts)
    out: dict = {
        "library_size": n,
        "reactions": result.total_reactions,
        "reactions_per_mode": round(result.total_reactions / n, 4) if n else None,
        "reward_gen_calls": result.total_reward_gen_calls,
        "distinct_hubs_used": result.distinct_hubs_used,
        "distinct_promoted_fragments": result.distinct_promoted_fragments,
        "stop_reason": result.stop_reason,
        "reached_mode_target": result.stop_reason == "modes",
    }

    # --- reward side (the numerator guard) --------------------------------------------------------
    vals = sorted(r for r in rewards if r == r)
    out["reward_mean"] = round(statistics.fmean(vals), 4) if vals else None
    out["reward_median"] = round(statistics.median(vals), 4) if vals else None
    out["reward_p10"] = round(_pct(vals, 0.10), 4) if vals else None
    out["reward_min"] = round(vals[0], 4) if vals else None
    out["reward_max"] = round(vals[-1], 4) if vals else None
    for bar in args.report_bars:
        frac = (
            sum(1 for r in vals if (r >= bar if args.higher_is_better else r <= bar)) / len(vals)
            if vals
            else None
        )
        out[f"frac_over_{bar:g}"] = round(frac, 4) if frac is not None else None

    # --- diversity side (what the Tanimoto filter protects) ---------------------------------------
    out["mean_pairwise_similarity"] = (
        round(mean_pairwise_similarity(smiles, fps=fps) or float("nan"), 4) if n > 1 else None
    )
    # Two counts of "how many distinct molecules is this library really?", at both cutoffs:
    #   modes  = greedy sphere exclusion fed BEST-REWARD-FIRST (the paper's definition), and
    #   Butina = the same question with reward taken OUT (centres by neighbourhood density).
    # Both are reported because the greedy count is order-dependent and its order is the reward —
    # circular for an ablation whose whole subject is the reward filter. Butina never reads the
    # reward, so when the two agree the collapse is a property of the molecules, not of how we count
    # them (the same cross-check Logs/051 runs on the pools).
    for sim in MODE_SIMILARITIES:
        m = count_modes(
            smiles,
            rewards,
            higher_is_better=args.higher_is_better,
            similarity_threshold=sim,
            fps=fps,
        )
        out[f"modes_at_{sim:g}"] = m
        out[f"modes_at_{sim:g}_per_mol"] = round(m / n, 4) if n else None
        butina = count_butina_clusters(smiles, fps=fps, similarity_threshold=sim + _BUTINA_EPS)
        out[f"butina_clusters_at_{sim:g}"] = butina
        out[f"butina_at_{sim:g}_per_mol"] = round(butina / n, 4) if (butina and n) else None
    scaf = unique_scaffolds(smiles, scaffolds=cache.scaffolds(smiles))
    out["unique_scaffolds"] = scaf
    out["scaffolds_per_mol"] = round(scaf / n, 4) if n else None

    # --- the honest cost: re-impose the canonical definition on whatever was delivered ------------
    q = qualified_modes(
        smiles,
        rewards,
        tau=args.qualify_tau,
        sim=args.qualify_similarity,
        higher_is_better=args.higher_is_better,
        fps=fps,
    )
    out["qualified_modes"] = q
    out["qualified_frac"] = round(q / n, 4) if n else None
    out["reactions_per_qualified_mode"] = round(result.total_reactions / q, 4) if q else None

    # --- the fixed-reaction-budget view of the same run (Case 1) ----------------------------------
    inside = [p for p in pts if p.cum_reactions <= args.budget_reactions]
    out["modes_at_rxn_budget"] = inside[-1].cum_modes if inside else 0
    out["reactions_at_rxn_budget"] = inside[-1].cum_reactions if inside else 0
    out["rxn_per_mode_at_rxn_budget"] = (
        round(inside[-1].cum_reactions / inside[-1].cum_modes, 4) if inside else None
    )
    if inside:
        pre_s = [p.smiles for p in inside]
        qb = qualified_modes(
            pre_s,
            [p.reward for p in inside],
            tau=args.qualify_tau,
            sim=args.qualify_similarity,
            higher_is_better=args.higher_is_better,
            fps=cache.fps(pre_s),
        )
        out["qualified_modes_at_rxn_budget"] = qb
        out["rxn_per_qualified_mode_at_rxn_budget"] = (
            round(inside[-1].cum_reactions / qb, 4) if qb else None
        )
    else:
        out["qualified_modes_at_rxn_budget"] = 0
        out["rxn_per_qualified_mode_at_rxn_budget"] = None

    # --- synthetic depth (§2.4's second numerator axis) -------------------------------------------
    depths = [d for d in (depth_of(p) for p in pts) if d is not None]
    out["depth_median"] = statistics.median(depths) if depths else None
    out["depth_mean"] = round(statistics.fmean(depths), 3) if depths else None
    return out


# ----------------------------------------------------------------------------- the scan
def run_scan(cands, hubs, comps, cost_table, cells, args) -> list:
    """Every (cell, strategy). Pools are already loaded; nothing here touches disk."""
    child_policy = make_child_policy(args.child_policy)
    cache = MolCache()
    hub_depth = {h.hub_key: h.depth for h in hubs}
    cand_depth = {c.smiles: c.num_reactions for c in cands}
    # A hub-batching member's own nested depth is its charged hub's depth + the diversifying coupling;
    # a best-candidate member carries its own. Same (fully nested) scale, so they are comparable.
    depth_of = {
        "hub_batching": lambda p: (
            hub_depth[p.source_hub] + 1 if p.source_hub in hub_depth else None
        ),
        "best_candidate": lambda p: cand_depth.get(p.smiles),
    }
    prebuilt_cache: dict = {}
    rows: list = []
    t_start = time.time()
    for i, cell in enumerate(cells, 1):
        tau, sim = cell["tau"], cell["similarity"]
        if args.prebuild_k > 0:  # the pre-select-K ranking is τ-dependent only -> cache per τ
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
        for name in STRATS:
            created: list = []
            t0 = time.time()
            result = RC.build_strategy(
                name,
                hubs if name == "hub_batching" else cands,
                cost_table,
                comps,
                target=args.tag,
                reward_threshold=tau,
                similarity=sim if sim is not OFF else 1.0,  # unused: the factory owns the selector
                higher_is_better=args.higher_is_better,
                child_policy=child_policy if name == "hub_batching" else None,
                prebuilt_fragments=prebuilt if name == "hub_batching" else None,
                mode_selector_factory=make_selector_factory(
                    tau, sim, args.higher_is_better, created
                ),
            ).run(("modes", args.budget_modes))
            row = {
                "strategy": name,
                "tau": tau,
                "similarity": sim,
                "reward_filter": "off" if tau is OFF else "on",
                "diversity_filter": "off" if sim is OFF else "on",
                "scans": "+".join(cell["scans"]),
                # Blank, not 0, when the selector does not track duplicates: the Tanimoto test
                # subsumes them (an identical molecule scores 1.0 against itself), so a zero here
                # would read as a measured "no duplicates occurred" rather than "not applicable".
                "duplicates_suppressed": _sum_or_none(
                    getattr(s, "n_duplicates_suppressed", None) for s in created
                ),
                "wall_s": round(time.time() - t0, 2),
            }
            row.update(library_readouts(result, depth_of[name], cache, args))
            rows.append(row)
        el = time.time() - t_start
        print(
            f"[filters] cell {i}/{len(cells)} τ={knob_label(tau):>4} cutoff={knob_label(sim):>4} "
            f"| {el:.0f}s elapsed, ~{el / i * (len(cells) - i):.0f}s left",
            flush=True,
        )
    print(f"[filters] {len(cache)} distinct molecules fingerprinted across the scan")
    return rows


# ----------------------------------------------------------------------------- outputs
CSV_COLUMNS = (
    "strategy",
    "tau",
    "similarity",
    "reward_filter",
    "diversity_filter",
    "scans",
    "library_size",
    "reactions",
    "reactions_per_mode",
    "qualified_modes",
    "qualified_frac",
    "reactions_per_qualified_mode",
    "reward_mean",
    "reward_median",
    "reward_p10",
    "reward_min",
    "reward_max",
    "mean_pairwise_similarity",
    "modes_at_0.7",
    "modes_at_0.7_per_mol",
    "butina_clusters_at_0.7",
    "butina_at_0.7_per_mol",
    "modes_at_0.5",
    "modes_at_0.5_per_mol",
    "butina_clusters_at_0.5",
    "butina_at_0.5_per_mol",
    "unique_scaffolds",
    "scaffolds_per_mol",
    "depth_median",
    "depth_mean",
    "duplicates_suppressed",
    "reward_gen_calls",
    "distinct_hubs_used",
    "distinct_promoted_fragments",
    "modes_at_rxn_budget",
    "reactions_at_rxn_budget",
    "rxn_per_mode_at_rxn_budget",
    "qualified_modes_at_rxn_budget",
    "rxn_per_qualified_mode_at_rxn_budget",
    "stop_reason",
    "reached_mode_target",
    "wall_s",
)
_FRAC_PREFIX = "frac_over_"


def write_csv(path: Path, rows) -> None:
    # The fraction-over-bar columns are named from --report-bars at runtime, so they are spliced in
    # beside the other reward readouts rather than hardcoded in CSV_COLUMNS.
    bars = sorted({k for r in rows for k in r if k.startswith(_FRAC_PREFIX)})
    split = CSV_COLUMNS.index("mean_pairwise_similarity")
    cols = list(CSV_COLUMNS[:split]) + bars + list(CSV_COLUMNS[split:])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def read_csv_rows(path: Path) -> list:
    """Committed CSV -> rows, for ``--plot-only``. ``off`` knobs come back as ``None``."""
    rows = []
    ints = {
        "library_size",
        "reactions",
        "qualified_modes",
        "modes_at_0.7",
        "butina_clusters_at_0.7",
        "modes_at_0.5",
        "butina_clusters_at_0.5",
        "unique_scaffolds",
        "duplicates_suppressed",
        "reward_gen_calls",
        "distinct_hubs_used",
        "distinct_promoted_fragments",
        "modes_at_rxn_budget",
        "reactions_at_rxn_budget",
        "qualified_modes_at_rxn_budget",
    }
    text = {"strategy", "reward_filter", "diversity_filter", "scans", "stop_reason"}
    for r in csv.DictReader(open(path, newline="")):
        row: dict = {}
        for k, v in r.items():
            if k in text:
                row[k] = v
            elif k in ("tau", "similarity"):
                row[k] = OFF if v in ("", "off") else float(v)
            elif k == "reached_mode_target":
                row[k] = v == "True"
            elif v == "":
                row[k] = None
            else:
                row[k] = int(float(v)) if k in ints else float(v)
        rows.append(row)
    return rows


# Panel spec: (csv field, y-label, ideal direction, gloss). Row = which filter is being scanned,
# column = readout, so a reader compares like-with-like DOWN a column: the same number under two
# different ablations.
PANELS = (
    ("reactions_per_mode", "Reactions per mode", "lower", "the headline cost, as counted", False),
    ("reward_median", "Median library reward", "higher", "what the reward gate protects", False),
    (
        "mean_pairwise_similarity",
        "Mean pairwise Tanimoto",
        "lower",
        "what the diversity filter protects",
        False,
    ),
    (
        "reactions_per_qualified_mode",
        "Reactions per QUALIFIED mode",
        "lower",
        "cost after re-imposing both filters",
        True,  # log y: the penalty spans ~1.2 to ~36, and a linear axis flattens the whole scan
    ),
)


def plot(path: Path, rows, args) -> None:
    """Two rows (one per ablated filter) × four readouts. ``off`` is the leftmost x position on both
    rows — the permissive endpoint of that filter's knob — separated by a rule so nobody reads it as a
    numeric value on the same scale."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Both rows read left-to-right as "no filter -> loosest -> strictest", so the ablation is visibly
    # the permissive end of the same axis: τ ascends (4 is loose), the cutoff DEscends (0.9 is loose,
    # matching every other LSD-Flow diversity panel), and "off" is leftmost on both.
    taus = _scan_values(rows, "reward")
    sims = _scan_values(rows, "diversity", descending=True)
    corner = {r["strategy"]: r for r in rows if r["tau"] is OFF and r["similarity"] is OFF}
    scans = (
        ("reward", "Reward filter ablated", "Reward bar τ", taus),
        ("diversity", "Diversity filter ablated", "Diversity cutoff (Tanimoto)", sims),
    )

    fig, axes = plt.subplots(2, len(PANELS), figsize=(17.2, 7.6))
    fig.suptitle(
        "What does each filter buy? Removing one is the permissive end of its own knob",
        fontsize=14,
        y=0.985,
        color=INK,
    )
    fig.text(
        0.5,
        0.930,
        f"SCENT sEH surrogate — every arm delivers a {args.budget_modes}-molecule library "
        f"(matched size); ★ = the headline operating point (τ={args.qualify_tau:g}, "
        f"cutoff={args.qualify_similarity:g})",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )

    handles: list = []
    for r_i, (scan, row_title, xlabel, knobs) in enumerate(scans):
        # x positions are categorical (uniform steps + a non-numeric "off"), so index == position.
        xs = list(range(len(knobs)))
        for c_i, (field, ylabel, ideal, gloss, log_y) in enumerate(PANELS):
            ax = axes[r_i][c_i]
            _knob_axis(ax, xs, knobs, scan, args)
            for name in STRATS:
                style = dict(STRAT_STYLE[name])
                label = style.pop("label")
                by = {
                    r["tau"] if scan == "reward" else r["similarity"]: r
                    for r in rows
                    if r["strategy"] == name and scan in r["scans"].split("+")
                }
                pts = [(x, by[k].get(field)) for x, k in zip(xs, knobs) if k in by]
                pts = [(x, y) for x, y in pts if y is not None]
                (line,) = ax.plot(
                    [p[0] for p in pts],
                    [p[1] for p in pts],
                    lw=1.9,
                    ms=4.6,
                    mew=0,
                    zorder=3,
                    label=label,
                    **style,
                )
                if r_i == 0 and c_i == 0:
                    handles.append(line)
                # POOL-LIMITED points are flagged, never plotted as ordinary measurements: the
                # strategy ran out of library before delivering the mode target, so that cell's
                # library is smaller than every other cell's and its cost is measured over a shorter
                # run. Same convention (and, on this anchor, the same strict corner) as Logs/052.
                short = [
                    (x, by[k].get(field))
                    for x, k in zip(xs, knobs)
                    if k in by and not by[k].get("reached_mode_target", True)
                ]
                short = [(x, y) for x, y in short if y is not None]
                if short:
                    ax.plot(
                        [p[0] for p in short],
                        [p[1] for p in short],
                        marker=style["marker"],
                        ls="none",
                        ms=8.5,
                        mfc="white",
                        mec=style["color"],
                        mew=1.5,
                        zorder=5,
                    )
                cval = (corner.get(name) or {}).get(field)
                if cval is not None:  # both filters off: its own marker, offset to stay legible
                    ax.plot(
                        [xs[0] + (0.16 if name == "best_candidate" else -0.16)],
                        [cval],
                        mec=style["color"],
                        zorder=4,
                        **CORNER_STYLE,
                    )
            if log_y:  # readable decade-spanning ticks: the default log formatter shows only "10^1"
                ax.set_yscale("log")
                lo, hi = ax.get_ylim()
                ticks = [t for t in (0.5, 1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50) if lo <= t <= hi]
                ax.set_yticks(ticks)
                ax.set_yticklabels([f"{t:g}" for t in ticks])
                ax.minorticks_off()
            ax.set_ylabel(f"{ylabel} {ideal_marker(ideal)}", fontsize=9.5, color=INK)
            # Left-aligned: the ★ sits at the operating point, which on the τ row is near the right
            # edge, and a right-aligned gloss lands underneath it.
            ax.set_title(gloss, fontsize=8.5, color=MUTED, loc="left", pad=4)
            ax.grid(axis="y", color="#e6e6e1", lw=0.7, zorder=0)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if c_i == 0:
                ax.text(
                    -0.30,
                    0.5,
                    row_title,
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                    color=INK,
                )
            ax.set_xlabel(xlabel, fontsize=8.8, color=INK)

    # ONE proxy entry for the corner: the marker is drawn in each strategy's own colour, so a per-line
    # legend entry would say "Both filters off" twice and read as two different things.
    handles.append(
        plt.Line2D(
            [],
            [],
            mec=MUTED,
            label="Both filters off",
            **CORNER_STYLE,
        )
    )
    if any(not r.get("reached_mode_target", True) for r in rows):
        handles.append(
            plt.Line2D(
                [],
                [],
                marker="o",
                ls="none",
                ms=8.5,
                mfc="white",
                mec=MUTED,
                mew=1.5,
                label="Pool-limited",
            )
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.912),
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
    )
    # Hard-wrapped on purpose: with ``bbox_inches="tight"`` a single long caption line is an artist
    # wider than the canvas, and the saved figure stretches to contain it (squashing the panels).
    fig.text(
        0.008,
        0.008,
        '"off" (left of the rule) = that filter removed; the other filter stays at the operating point.'
        " A QUALIFIED mode is a delivered molecule that survives the canonical definition\n"
        f"(reward ≥ {args.qualify_tau:g} AND Tanimoto ≤ {args.qualify_similarity:g} vs every mode kept),"
        " so the rightmost column is what each library really costs per molecule worth keeping."
        " Hollow points are pool-limited\n(that library is smaller than every other cell's, so its cost"
        " is measured over a shorter run and is not comparable).",
        fontsize=7.5,
        color="#666660",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0.022, 0.050, 1, 0.895))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def _scan_values(rows, scan: str, *, descending: bool = False) -> list:
    """The knob values measured on one scan line, ablation ("off") ALWAYS first — it is the permissive
    end of the knob, whichever way the numeric values run."""
    key = "tau" if scan == "reward" else "similarity"
    vals = {r[key] for r in rows if scan in r["scans"].split("+")}
    numeric = sorted((v for v in vals if v is not OFF), reverse=descending)
    return ([OFF] if OFF in vals else []) + numeric


def _knob_axis(ax, xs, knobs, scan: str, args) -> None:
    """Categorical knob axis: ticks for every measured value, ``off`` set apart by a rule, and a ★ at
    the operating point."""
    ax.set_xticks(xs)
    ax.set_xticklabels([knob_label(k) for k in knobs], fontsize=8.5)
    for lbl, k in zip(ax.get_xticklabels(), knobs):
        if k is OFF:
            lbl.set_color("#b23a48")  # the ablation reads as a different kind of value
            lbl.set_fontstyle("italic")
    if knobs and knobs[0] is OFF and len(knobs) > 1:
        ax.axvline(0.5, color="#b9b9b4", lw=0.9, ls=(0, (2, 2)), zorder=1)
    op = args.qualify_tau if scan == "reward" else args.qualify_similarity
    for x, k in zip(xs, knobs):
        if k is not OFF and abs(k - op) < 1e-9:
            ax.plot(
                [x],
                [1.0],
                transform=ax.get_xaxis_transform(),
                marker="*",
                ms=11,
                mfc="none",
                mec="#111111",
                mew=1.2,
                clip_on=False,
                zorder=6,
            )
    ax.set_xlim(xs[0] - 0.45, xs[-1] + 0.45)


# The four factorial arms, in the order they read as an argument: baseline, then one filter removed,
# then both.
FACTORIAL_ARMS = (
    ("both_on", "Both filters\n(as published)"),
    ("reward_off", "Reward gate\nOFF"),
    ("diversity_off", "Diversity filter\nOFF"),
    ("both_off", "Both\nOFF"),
)


def factorial_corners(args) -> dict:
    """arm name -> the ``(tau, similarity)`` cell it refers to. ONE definition, shared by the summary
    and the figure: keying an arm by anything looser (e.g. "is this knob off?") silently matches every
    numeric cell in the scans as well, and the last one wins."""
    op_t, op_s = args.qualify_tau, args.qualify_similarity
    return {
        "both_on": (op_t, op_s),
        "reward_off": (OFF, op_s),
        "diversity_off": (op_t, OFF),
        "both_off": (OFF, OFF),
    }


# Counting the SAME quantity two ways at each cutoff — greedy (reward-ordered) vs Butina
# (reward-blind) — so each pair shares a hue and differs in lightness: same question, two estimators.
# The final bar is a different quantity (both filters re-imposed), so it gets its own hue.
FACTORIAL_BARS = (
    ("modes_at_0.7", "greedy @0.7  (reward-ordered)", "#8fb8d8"),
    ("butina_clusters_at_0.7", "Butina @0.7  (reward-blind)", "#2a6f97"),
    ("modes_at_0.5", "greedy @0.5  (reward-ordered)", "#c9a8dd"),
    ("butina_clusters_at_0.5", "Butina @0.5  (reward-blind)", "#8a4fbf"),
    ("qualified_modes", "qualified (reward + diversity re-imposed)", "#b23a48"),
)


def plot_factorial(path: Path, rows, args) -> None:
    """The four arms as delivered libraries: of the N molecules each one hands over, how many are
    distinct molecules — counted with the reward in the procedure and with it taken out — and how many
    survive the canonical definition.

    This is the ablation's headline artifact. The scan figure shows the trends; this one answers "what
    did 300 molecules actually buy?" in a single glance, with the reward-blind count sitting beside the
    reward-ordered one so the collapse cannot be blamed on how we count."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by = {(r["strategy"], r["tau"], r["similarity"]): r for r in rows}
    corners = factorial_corners(args)
    fig, axes = plt.subplots(1, len(STRATS), figsize=(13.6, 5.4), sharey=True)
    delivered = max(
        by[(name, *corners[arm])]["library_size"]
        for name in STRATS
        for arm, _ in FACTORIAL_ARMS
        if (name, *corners[arm]) in by
    )
    fig.suptitle(
        f"Of the {delivered} molecules each library hands over, how many are distinct molecules?",
        fontsize=13.5,
        y=0.995,
        color=INK,
    )
    fig.text(
        0.5,
        0.938,
        "SCENT sEH surrogate — each arm run to the same "
        f"{delivered}-molecule target, so the bars are directly comparable",
        ha="center",
        fontsize=9.5,
        color=MUTED,
    )

    handles: list = []
    width = 0.16
    for ax, name in zip(axes, STRATS):
        xs = list(range(len(FACTORIAL_ARMS)))
        for b_i, (field, label, color) in enumerate(FACTORIAL_BARS):
            offs = (b_i - (len(FACTORIAL_BARS) - 1) / 2) * width
            vals, pos = [], []
            for x, (arm, _) in zip(xs, FACTORIAL_ARMS):
                r = by.get((name, *corners[arm]))
                if r and r.get(field) is not None:
                    vals.append(r[field])
                    pos.append(x + offs)
            bars = ax.bar(pos, vals, width=width * 0.92, color=color, zorder=3, label=label)
            if ax is axes[0]:
                handles.append(bars)
            for rect, v in zip(bars, vals):
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    v + delivered * 0.012,
                    f"{v:g}",
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                    color=INK,
                    rotation=90 if v >= delivered * 0.9 else 0,
                )
        ax.axhline(delivered, color="#9a9a94", lw=1.0, ls=(0, (4, 3)), zorder=2)
        ax.text(
            len(FACTORIAL_ARMS) - 0.45,
            delivered * 1.015,
            f"{delivered} delivered",
            ha="right",
            va="bottom",
            fontsize=8,
            color=MUTED,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels([lbl for _, lbl in FACTORIAL_ARMS], fontsize=9)
        for lbl, (arm, _) in zip(ax.get_xticklabels(), FACTORIAL_ARMS):
            if arm != "both_on":  # the ablated arms read as a different kind of arm
                lbl.set_color("#b23a48")
        ax.set_title(STRAT_STYLE[name]["label"], fontsize=11, color=INK)
        ax.grid(axis="y", color="#e6e6e1", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_ylim(0, delivered * 1.16)
        if ax is axes[0]:
            ax.set_ylabel(
                f"distinct molecules in the delivered library {ideal_marker('higher')}", fontsize=10
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.915),
        ncol=3,
        frameon=False,
        fontsize=8.6,
    )
    fig.text(
        0.008,
        0.008,
        "Greedy = the paper's sphere-exclusion count, fed best-reward-first. Butina = the same "
        '"how many families?" question with reward taken OUT of the procedure (centres by\nneighbourhood '
        "density), so a collapse cannot be an artifact of counting in reward order. Qualified = members "
        f"surviving the canonical definition (reward ≥ {args.qualify_tau:g} and Tanimoto\n"
        f"≤ {args.qualify_similarity:g}) — for the unablated arm that is the whole library, by "
        "construction.",
        fontsize=7.5,
        color="#666660",
        linespacing=1.5,
    )
    fig.tight_layout(rect=(0, 0.062, 1, 0.905))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize(rows, args) -> dict:
    """The 2×2 factorial at the operating point + the per-scan extremes, so the verdict is readable
    without the grid. ``cost_illusion`` is the ratio the ablation *appears* to buy on reactions/mode;
    ``true_cost_ratio`` is what it does to reactions per qualified mode."""
    by = {(r["strategy"], r["tau"], r["similarity"]): r for r in rows}
    op_t, op_s = args.qualify_tau, args.qualify_similarity
    out: dict = {
        "factorial": {},
        "per_strategy": {},
        # Cells whose library is smaller than the target: matched-N does not hold there, so their
        # quality readouts are not comparable to the rest of the scan (flagged, never dropped).
        "pool_limited_cells": [
            {
                "strategy": r["strategy"],
                "tau": knob_label(r["tau"]),
                "similarity": knob_label(r["similarity"]),
                "library_size": r["library_size"],
                "reactions": r["reactions"],
            }
            for r in rows
            if not r["reached_mode_target"]
        ],
    }
    corners = factorial_corners(args)
    keep = (
        "library_size",
        "reactions",
        "reactions_per_mode",
        "qualified_modes",
        "reactions_per_qualified_mode",
        "reward_median",
        "reward_p10",
        "mean_pairwise_similarity",
        # Both counts at both cutoffs: the reward-ORDERED greedy count next to the reward-BLIND
        # Butina count. The pair is the point — an ablation of the reward filter cannot rest on a
        # diversity number whose own procedure is ordered by reward.
        "modes_at_0.7",
        "butina_clusters_at_0.7",
        "modes_at_0.5",
        "butina_clusters_at_0.5",
        "unique_scaffolds",
        "depth_median",
        "duplicates_suppressed",
        "stop_reason",
    )
    for name in STRATS:
        base = by.get((name, op_t, op_s))
        out["factorial"][name] = {}
        for cname, (t, s) in corners.items():
            r = by.get((name, t, s))
            if not r:
                continue
            entry = {k: r.get(k) for k in keep}
            if base and base.get("reactions_per_mode") and r.get("reactions_per_mode"):
                entry["cost_illusion_vs_both_on"] = round(
                    base["reactions_per_mode"] / r["reactions_per_mode"], 4
                )
            if (
                base
                and base.get("reactions_per_qualified_mode")
                and r.get("reactions_per_qualified_mode")
            ):
                entry["true_cost_ratio_vs_both_on"] = round(
                    r["reactions_per_qualified_mode"] / base["reactions_per_qualified_mode"], 4
                )
            out["factorial"][name][cname] = entry
        # Self-check: at the operating point the delivered library IS the canonical definition, so
        # every member must qualify. A mismatch means the ablation seam changed the unablated path.
        if base:
            out["per_strategy"][name] = {
                "unablated_all_members_qualify": base.get("qualified_modes")
                == base.get("library_size"),
                "unablated_reactions_per_mode": base.get("reactions_per_mode"),
            }
    return out


def print_factorial(summary: dict) -> None:
    """The 2×2 table, printed so a login-node run tells the story without opening the JSON.

    ``greedy@x`` / ``butina@x`` are the reward-ORDERED and reward-BLIND counts of how many distinct
    molecules the delivered library actually holds — printed side by side so the collapse can be read
    without having to trust a reward-ordered procedure."""
    for name, corners in summary["factorial"].items():
        print(f"\n  {name}:")
        print(
            f"    {'arm':<15} {'lib':>4} {'rxn':>5} {'rxn/mode':>9} {'qual':>5} "
            f"{'rxn/qual':>9} {'medR':>6} {'pairwise':>9} "
            f"{'greedy@.7':>9} {'butina@.7':>9} {'greedy@.5':>9} {'butina@.5':>9}"
        )
        for arm, e in corners.items():
            print(
                f"    {arm:<15} {e['library_size']:>4} {e['reactions']:>5} "
                f"{e['reactions_per_mode'] or float('nan'):>9.4f} {e['qualified_modes']:>5} "
                f"{e['reactions_per_qualified_mode'] or float('nan'):>9.4f} "
                f"{e['reward_median'] or float('nan'):>6.3f} "
                f"{e['mean_pairwise_similarity'] or float('nan'):>9.4f} "
                f"{e['modes_at_0.7']:>9} {e['butina_clusters_at_0.7']:>9} "
                f"{e['modes_at_0.5']:>9} {e['butina_clusters_at_0.5']:>9}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--analysis-dir", help="sampling-stage dir (records.csv + compositions.json)")
    ap.add_argument("--enum-children", help="enum_children.json from the enumeration worker")
    ap.add_argument(
        "--snapshot", default="", help="SCENT fragments_<N>.json for the nested cost model"
    )
    ap.add_argument(
        "--taus",
        default="off,4:8:0.5",
        help="reward-filter scan; 'off' = gate removed (the ablation)",
    )
    ap.add_argument(
        "--similarities",
        default="off,0.3:0.9:0.1",
        help="diversity-filter scan; 'off' = Tanimoto test removed (exact duplicates still dropped)",
    )
    ap.add_argument(
        "--qualify-tau",
        type=float,
        default=7.0,
        help="the canonical hit bar: the operating point of the reward knob AND the gate a delivered "
        "molecule must clear to count as a QUALIFIED mode",
    )
    ap.add_argument(
        "--qualify-similarity",
        type=float,
        default=0.5,
        help="the canonical diversity cutoff: operating point of the diversity knob AND the cutoff "
        "used to re-count qualified modes",
    )
    ap.add_argument(
        "--budget-modes",
        type=int,
        default=300,
        help="library size every arm delivers (matched N, so quality readouts are comparable)",
    )
    ap.add_argument(
        "--budget-reactions",
        type=int,
        default=300,
        help="the fixed-reaction-budget (Case 1) view, read off the same curve prefix",
    )
    ap.add_argument(
        "--report-bars",
        default="5,6,7",
        help="reward bars to report the library's fraction-over (Logs/034's calibrated range)",
    )
    ap.add_argument("--higher-is-better", type=lambda s: s.lower() != "false", default=True)
    ap.add_argument("--child-policy", default="free_frag", choices=["reward", "free_frag"])
    ap.add_argument(
        "--prebuild-k", type=int, default=20, help="pre-select-K (Logs/037); 0 disables"
    )
    ap.add_argument("--rank-by", default="build_score", choices=list(RANK_METHODS))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="")
    ap.add_argument(
        "--plot-only", action="store_true", help="redraw from the committed results CSV"
    )
    args = ap.parse_args()
    args.report_bars = [float(x) for x in args.report_bars.split(",") if x.strip()]

    out = Path(args.out_dir) if args.out_dir else HERE / "results" / args.tag
    if args.plot_only:
        rows = read_csv_rows(out / CSV_NAME)
        plot(out / "filter_ablation", rows, args)
        plot_factorial(out / "filter_ablation_factorial", rows, args)
        print(f"redrew {out}/filter_ablation{{,_factorial}}.{{png,pdf}} from {len(rows)} rows")
        return

    if not args.analysis_dir or not args.enum_children:
        ap.error("--analysis-dir and --enum-children are required unless --plot-only")
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    cands, comps = RC._load_candidates(Path(args.analysis_dir), args.higher_is_better)
    hubs = RC._load_enumerated_hubs(Path(args.enum_children), comps)
    cost_table = load_cost_table_from_snapshot(
        json.load(open(args.snapshot)) if args.snapshot else {}
    )
    cells = build_cells(
        parse_knob(args.taus),
        parse_knob(args.similarities),
        args.qualify_tau,
        args.qualify_similarity,
    )
    print(
        f"[filters] {len(cands)} candidates, {len(hubs)} hubs, {len(cost_table.promoted_set)} promoted "
        f"(recipes={bool(cost_table.recipes)}) loaded in {time.time() - t0:.1f}s\n"
        f"[filters] {len(cells)} cells x {len(STRATS)} strategies to a {args.budget_modes}-mode "
        f"library; child_policy={args.child_policy}, prebuild_k={args.prebuild_k}; qualified mode = "
        f"reward ≥ {args.qualify_tau:g} & Tanimoto ≤ {args.qualify_similarity:g}",
        flush=True,
    )

    rows = run_scan(cands, hubs, comps, cost_table, cells, args)
    write_csv(out / CSV_NAME, rows)
    summary = summarize(rows, args)
    payload = {
        "tag": args.tag,
        "analysis_dir": args.analysis_dir,
        "enum_children": args.enum_children,
        "snapshot": args.snapshot,
        "taus": [knob_label(v) for v in parse_knob(args.taus)],
        "similarities": [knob_label(v) for v in parse_knob(args.similarities)],
        "qualify_tau": args.qualify_tau,
        "qualify_similarity": args.qualify_similarity,
        "budget_modes": args.budget_modes,
        "budget_reactions": args.budget_reactions,
        "report_bars": args.report_bars,
        "child_policy": args.child_policy,
        "prebuild_k": args.prebuild_k,
        "rank_by": args.rank_by if args.prebuild_k else None,
        "mode_similarities_reported": list(MODE_SIMILARITIES),
        "diversity_off_semantics": "RewardOnlyModeSelector: no Tanimoto test, exact canonical-SMILES "
        "duplicates still suppressed (count reported per cell)",
        "reward_off_semantics": "reward_threshold=None; best-reward-first ordering retained (that is "
        "the strategy, not the filter)",
        "wall_s": round(time.time() - t0, 1),
        "summary": summary,
    }
    (out / "filter_ablation.json").write_text(json.dumps(payload, indent=2))
    plot(out / "filter_ablation", rows, args)
    plot_factorial(out / "filter_ablation_factorial", rows, args)
    print_factorial(summary)
    for c in summary["pool_limited_cells"]:
        print(
            f"\n[filters] POOL-LIMITED {c['strategy']} τ={c['tau']} cutoff={c['similarity']}: "
            f"library {c['library_size']} < {args.budget_modes} target — not matched-N, "
            f"cost measured over a shorter run"
        )
    for name, chk in summary["per_strategy"].items():
        flag = "OK" if chk["unablated_all_members_qualify"] else "MISMATCH"
        print(f"\n[filters] self-check {name}: unablated library all-qualify = {flag}")
    print(
        f"[filters] wrote {out}/{CSV_NAME} + filter_ablation.{{json,png,pdf}} "
        f"+ filter_ablation_factorial.{{png,pdf}} in {payload['wall_s']}s"
    )


if __name__ == "__main__":
    main()
