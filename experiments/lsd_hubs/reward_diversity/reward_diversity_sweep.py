#!/usr/bin/env python
"""Is a molecule set's intrinsic diversity coupled to the reward cutoff we condition it on? (Logs/051)

WHY this exists. Every headline cost number we report is conditioned on a reward gate: reactions/mode
counts a molecule only if it clears a "hit" bar (7.0 in Logs/029/033, 5/6 after the sEH-proxy
calibration in Logs/034, swept in Logs/035 and ``matrix16/gate_curve.py``). The publication notes
(``docs/paper_planning/lsd-flow-publication-strategy.md`` §2.4-2.5) name the exposure precisely: the
mode *denominator* is only a fair yardstick if raising the bar does not by itself change how similar
the surviving molecules are. If high-reward molecules are intrinsically more alike, then a stricter
gate shrinks the mode count for reasons that have nothing to do with which selection strategy ran —
and the τ-sweeps inherit that confound. Nothing in the repo measures it, so this does.

WHAT it measures. For each reward cutoff τ, keep the molecules scoring ``>= τ`` and report four
readouts on the survivors:

  * **mean pairwise Tanimoto** — the set-level "how alike is an average pair?" (Morgan r=3/2048, the
    same fingerprint the mode definition uses). Unbiased under uniform subsampling; insensitive by
    construction, so never read alone.
  * **modes per molecule kept** — the paper's greedy sphere-exclusion count (Tanimoto 0.7) divided by
    the number of molecules scored. This is literally the reactions/mode denominator, restricted to
    the τ-gated pool, so it answers the confound in the metric's own units.
  * **Butina clusters per molecule kept** — the same "how many families?" question with reward taken
    OUT of the procedure. Greedy exclusion is order-dependent and its order is the reward, which is
    circular for a question about reward vs diversity; Butina picks centres by neighbourhood density.
  * **unique Bemis-Murcko scaffolds per molecule kept** — the fingerprint-independent cross-check
    (§2.3 asks for a second descriptor so the conclusion is not an ECFP artifact).

THE ONE DESIGN CONSTRAINT THAT MATTERS. Mode counts and scaffold counts **saturate** with set size, so
comparing a 600k-molecule set at τ=4 against a 5k-molecule set at τ=8 would measure set size, not
reward. Every cutoff is therefore scored on the **same number of molecules** (``--subsample``, default
2500 = the largest N that fits inside the smallest gated set on both pools), drawn uniformly at random.
The draw uses ``--replicates`` independent random *permutations* of the pool, taking the first N
qualifying molecules in each: that is a uniform random size-N subset of the gated set (a random
permutation restricted to a subset is a random permutation of it), and because the gated sets are
nested in τ, consecutive cutoffs reuse most of the same molecules — the curve is paired across τ rather
than independently jittered, and the fingerprint cache is hit instead of recomputed.

Costs nothing but CPU: it re-reads two already-persisted record files, no model, no GPU, no oracle.

    source ~/bin/rgfn-smoke-env.sh
    python experiments/lsd_hubs/reward_diversity/reward_diversity_sweep.py \
        --sample-records  /scratch/markymoo/rgfn_runs/lsdflow/scent_seh_70189/records.csv \
        --enum-records    /scratch/markymoo/rgfn_runs/lsdflow/campaign_enum_seh_70363/enumerated_records.csv \
        --tag scent_seh

Output: ``results/<tag>/{reward_diversity.csv,reward_diversity.json,reward_diversity.png,.pdf}``
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from validation.lsdflow.metrics.diversity import (  # noqa: E402
    count_butina_clusters,
    ecfp,
    mean_pairwise_similarity,
    mode_representatives,
    murcko_scaffold,
    unique_scaffolds,
)
from validation.lsdflow.plot_style import pareto_marker  # noqa: E402

# Two SETS, not two strategies: teal = what the model generated, purple = what we enumerated from its
# hubs. Deliberately NOT the campaign's blue #2a78d6 / orange #eb6834 — those mean hub-batching vs
# best-candidate in every other figure and would misread here. Six-checks validator (light, surface
# #fcfcfb): chroma PASS, CVD ΔE 15.5 deutan / 18.1 tritan PASS, normal-vision 25.6 PASS, contrast PASS.
# Linestyle + direct labels carry identity too, so nothing rests on hue alone.
SET_STYLE = {
    "sample": dict(
        color="#2a9d8f", ls="-", marker="o", label="Generated sample (30k trajectories)"
    ),
    "enumerated": dict(color="#8a4fbf", ls="--", marker="s", label="Enumerated library (200 hubs)"),
}
# The reward bars the campaign has actually reported at, drawn for orientation (Logs/029/033/034/035).
CAMPAIGN_BARS = (5.0, 6.0, 7.0)
MODE_SIMILARITY = 0.7


# ----------------------------------------------------------------------------- inputs
def parse_cutoffs(spec: str) -> list:
    """``"1,2,3,4.0:8.0:0.1"`` -> [1, 2, 3, 4.0, 4.1, ...]. Comma-separated terms; a term with two
    colons is an inclusive ``lo:hi:step`` range. Rounded to 3 dp so 0.1 steps stay clean keys."""
    out: list = []
    for term in spec.split(","):
        term = term.strip()
        if not term:
            continue
        if ":" in term:
            lo, hi, step = (float(x) for x in term.split(":"))
            v = lo
            while (step > 0 and v <= hi + 1e-9) or (step < 0 and v >= hi - 1e-9):
                out.append(round(v, 3))
                v += step
        else:
            out.append(round(float(term), 3))
    return sorted(dict.fromkeys(out))


def load_pool(path: Path, higher_is_better: bool = True):
    """A flow-record CSV (``records.csv`` or ``enumerated_records.csv``) -> ``(smiles, rewards)``,
    **deduplicated by molecule** keeping its best reward — the same pool construction
    ``run_campaign.py::_load_candidates`` uses, so the sweep scores the library the campaign costs.
    Rows are (hub_key, child_key, reward, ...); ``child_key`` is the terminal molecule."""
    best: dict = {}
    with open(path, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        i_child, i_reward = header.index("child_key"), header.index("reward")
        for row in rd:
            key = row[i_child]
            if not key:
                continue
            try:
                reward = float(row[i_reward])
            except ValueError:
                continue
            if reward != reward:  # NaN never enters the pool
                continue
            prev = best.get(key)
            if prev is None or (reward > prev) == higher_is_better:
                best[key] = reward
    smiles = list(best)
    return smiles, [best[s] for s in smiles]


# ----------------------------------------------------------------------------- fingerprint cache
class MolCache:
    """Lazy per-molecule ECFP + Murcko cache. The sweep scores 44 nested subsets x N replicates, so
    the same molecule is re-scored dozens of times; RDKit work happens once per distinct SMILES."""

    def __init__(self):
        self._fp: dict = {}
        self._scaffold: dict = {}

    def fps(self, smiles):
        for s in smiles:
            if s not in self._fp:
                self._fp[s] = ecfp(s)
        return [self._fp[s] for s in smiles]

    def scaffolds(self, smiles):
        for s in smiles:
            if s not in self._scaffold:
                self._scaffold[s] = murcko_scaffold(s)
        return [self._scaffold[s] for s in smiles]

    def __len__(self):
        return len(self._fp)


# ----------------------------------------------------------------------------- the sweep
def draw(order, rewards, cutoff: float, n: int) -> list:
    """First ``n`` indices in ``order`` whose reward clears ``cutoff`` — a uniform random size-``n``
    subset of the gated set when ``order`` is a random permutation (fewer, if the set is smaller).
    """
    picked = []
    for i in order:
        if rewards[i] >= cutoff:
            picked.append(i)
            if len(picked) == n:
                break
    return picked


def score_subset(smiles, rewards, cache: MolCache) -> dict:
    """The four diversity readouts on one subsample. Modes are the canonical greedy sphere-exclusion
    count, ordered best-reward-first; the reward gate is already applied by the draw, so
    ``reward_threshold`` stays ``None`` (double-gating would be the same molecules anyway).

    Butina sits beside modes on purpose: greedy exclusion is order-dependent and here the order IS the
    reward, so a reward-blind clustering (centres by neighbourhood density) is the honest cross-check
    for a question about reward vs diversity."""
    fps = cache.fps(smiles)
    scafs = cache.scaffolds(smiles)
    n_modes = len(
        mode_representatives(smiles, rewards, similarity_threshold=MODE_SIMILARITY, fps=fps)
    )
    n_butina = count_butina_clusters(smiles, fps=fps, similarity_threshold=MODE_SIMILARITY)
    return {
        "mean_pairwise_similarity": mean_pairwise_similarity(smiles, fps=fps),
        "modes_per_mol": n_modes / len(smiles),
        "butina_per_mol": (n_butina / len(smiles)) if n_butina is not None else None,
        "scaffolds_per_mol": unique_scaffolds(smiles, scaffolds=scafs) / len(smiles),
        "n_modes": n_modes,
        "n_butina_clusters": n_butina,
        "n_scored": len(smiles),
    }


def sweep_pool(name: str, smiles, rewards, cutoffs, *, subsample: int, replicates: int, seed: int):
    """One pool, every cutoff. Returns per-cutoff aggregate rows + the per-replicate detail."""
    cache = MolCache()
    orders = [
        random.Random(seed + 1000 * r).sample(range(len(smiles)), len(smiles))
        for r in range(replicates)
    ]
    rows, detail = [], []
    for cutoff in cutoffs:
        n_above = sum(1 for v in rewards if v >= cutoff)
        if n_above < 2:
            print(
                f"  [{name}] cutoff {cutoff}: only {n_above} molecules above — skipped", flush=True
            )
            continue
        reps = []
        for r, order in enumerate(orders):
            idx = draw(order, rewards, cutoff, subsample)
            got = score_subset([smiles[i] for i in idx], [rewards[i] for i in idx], cache)
            got.update(replicate=r, cutoff=cutoff, set=name)
            reps.append(got)
            detail.append(got)
            if len(idx) == n_above:
                break  # gated set smaller than the subsample: every replicate is the same full set
        row = {
            "set": name,
            "cutoff": cutoff,
            "n_above_cutoff": n_above,
            "frac_above_cutoff": n_above / len(smiles),
            "n_scored": reps[0]["n_scored"],
            "n_replicates": len(reps),
            "subsampled": reps[0]["n_scored"] < n_above,
        }
        for metric in (
            "mean_pairwise_similarity",
            "modes_per_mol",
            "butina_per_mol",
            "scaffolds_per_mol",
        ):
            vals = [rep[metric] for rep in reps if rep[metric] is not None]
            row[metric] = statistics.fmean(vals) if vals else None
            row[metric + "_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
        rows.append(row)
        print(
            f"  [{name}] cutoff {cutoff:>4}: n>={cutoff} {n_above:>7} | scored {row['n_scored']:>5}"
            f" x{row['n_replicates']} | pairwise {row['mean_pairwise_similarity']:.4f}"
            f" | modes/mol {row['modes_per_mol']:.4f} | butina/mol {row['butina_per_mol']:.4f}"
            f" | scaffolds/mol {row['scaffolds_per_mol']:.4f}",
            flush=True,
        )
    return rows, detail, len(cache)


def spearman(xs, ys):
    """Rank correlation without scipy (ties averaged). The one-number summary of "does diversity
    track the cutoff at all?" across the swept range."""
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 3:
        return None

    def ranks(vals):
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs])
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = sum((a - mx) ** 2 for a in rx) ** 0.5 * sum((b - my) ** 2 for b in ry) ** 0.5
    return num / den if den else None


# ----------------------------------------------------------------------------- outputs
# BOTH axes of these panels are objectives: we want molecules that score high (x, further right) AND a
# set that stays diverse (y). So each panel carries ONE DIAGONAL marker at the desirable corner —
# ``pareto_marker`` — appended to the plain-language hint, not to the axis label. A single "(↓)" on the
# y-label would only speak for diversity and silently drop the reward half of what we want.
#
# Read the corner as: "does the set still look good when we demand more reward?" The x-axis is a filter
# we choose, not an outcome the model achieved, so a point near the corner means the set SURVIVED a
# stricter demand — it does not mean the generator scored better.
# The pool-size panel gets NO marker on purpose: a bigger or smaller surviving pool is context, neither
# good nor bad, and an arrow there would assert a preference we do not hold.
_DIVERSE_UP = pareto_marker(x="higher", y="higher")  # "(↗)" — more reward, more distinct families
_DIVERSE_DOWN = pareto_marker(x="higher", y="lower")  # "(↘)" — more reward, less self-similarity
METRICS = (
    (
        "mean_pairwise_similarity",
        "Mean pairwise Tanimoto",
        f"lower = more diverse {_DIVERSE_DOWN}",
    ),
    (
        "modes_per_mol",
        f"Modes per molecule (T {MODE_SIMILARITY})",
        f"reward-ordered · higher = more diverse {_DIVERSE_UP}",
    ),
    (
        "butina_per_mol",
        f"Butina clusters per molecule (T {MODE_SIMILARITY})",
        f"reward-blind · higher = more diverse {_DIVERSE_UP}",
    ),
    (
        "scaffolds_per_mol",
        "Murcko scaffolds per molecule",
        f"higher = more diverse {_DIVERSE_UP}",
    ),
)
CSV_COLUMNS = (
    "set",
    "cutoff",
    "n_above_cutoff",
    "frac_above_cutoff",
    "n_scored",
    "n_replicates",
    "subsampled",
    "mean_pairwise_similarity",
    "mean_pairwise_similarity_sd",
    "modes_per_mol",
    "modes_per_mol_sd",
    "butina_per_mol",
    "butina_per_mol_sd",
    "scaffolds_per_mol",
    "scaffolds_per_mol_sd",
)


def write_csv(path: Path, rows) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})


def read_csv(path: Path) -> list:
    """Committed sweep CSV -> rows, for ``--plot-only`` (iterate on the figure without re-sweeping)."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            row = {"set": r["set"], "subsampled": r["subsampled"] == "True"}
            for k, v in r.items():
                if k in ("set", "subsampled"):
                    continue
                row[k] = None if v == "" else (int(v) if k.startswith("n_") else float(v))
            rows.append(row)
    return rows


def plot(path: Path, rows, tag: str) -> None:
    """Five panels sharing the cutoff axis: the four diversity readouts + the pool size each readout
    was taken from (so a reader can see where the tail gets thin). One line per molecule set. The
    sixth cell of the 2x3 grid is hidden rather than filled with filler."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = [s for s in SET_STYLE if any(r["set"] == s for r in rows)]
    fig, axes = plt.subplots(2, 3, figsize=(14.6, 7.8), sharex=True)
    fig.suptitle(
        "Does the reward cutoff change intrinsic diversity?", fontsize=14, y=0.985, color="#1a1a18"
    )
    fig.text(
        0.5,
        0.938,
        f"SCENT sEH surrogate — {int(rows[0]['n_scored']):,} molecules scored at every cutoff, "
        f"{int(max(r['n_replicates'] for r in rows))} draws each",
        ha="center",
        fontsize=9.5,
        color="#7a7a72",
    )

    handles: list = []
    panels = list(METRICS) + [("frac_above_cutoff", "Fraction of set above cutoff", "pool size")]
    assert len(panels) <= axes.size, f"{len(panels)} panels will not fit {axes.size} axes"
    for ax, (metric, ylabel, hint) in zip(axes.flat, panels):
        for bar in CAMPAIGN_BARS:  # recessive orientation marks, behind the data
            ax.axvline(bar, color="#b9b9b4", lw=0.8, ls=(0, (1, 3)), zorder=1)
        for name in sets:
            style = dict(SET_STYLE[name])
            label = style.pop("label")
            pts = [r for r in rows if r["set"] == name and r.get(metric) is not None]
            xs = [r["cutoff"] for r in pts]
            ys = [r[metric] for r in pts]
            sd = [r.get(metric + "_sd") or 0.0 for r in pts]
            (line,) = ax.plot(xs, ys, lw=2.0, ms=3.4, mew=0, zorder=3, label=label, **style)
            if ax is axes[0][0]:
                handles.append(line)
            if any(sd):  # replicate spread as a band, not error bars (44 points would be a hedge)
                ax.fill_between(
                    xs,
                    [y - s for y, s in zip(ys, sd)],
                    [y + s for y, s in zip(ys, sd)],
                    color=style["color"],
                    alpha=0.18,
                    lw=0,
                    zorder=2,
                )
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(hint, fontsize=9, color="#7a7a72", loc="right", pad=3)
        ax.grid(axis="y", color="#e6e6e1", lw=0.7, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if metric == "frac_above_cutoff":
            ax.set_yscale("log")
    for ax in axes.flat[len(panels) :]:  # no filler panels
        ax.set_visible(False)
    # x-label on every axis with nothing drawn beneath it (bottom row + any top-row orphan)
    for col in range(axes.shape[1]):
        below = axes[1][col]
        target = below if below.get_visible() else axes[0][col]
        target.set_xlabel("sEH reward cutoff τ  (molecules kept: reward ≥ τ)", fontsize=10)
        if not below.get_visible():  # sharex hides the orphan's ticks; put them back
            target.tick_params(labelbottom=True)
    # One figure-level legend (read once, applies to all four panels) instead of an in-panel box that
    # collides with the data; solid-vs-dashed carries identity too, so nothing rests on hue alone.
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.912),
        ncol=len(handles),
        frameon=False,
        fontsize=9.5,
    )
    fig.text(
        0.008,
        0.012,
        "Dotted verticals: the reward bars the campaign reports at (5 / 6 / 7).  Every point scored on "
        "the SAME 2,500 randomly drawn molecules (band = SD over 3 draws), so mode and scaffold counts "
        "are not confounded by how many molecules survive the cutoff.",
        fontsize=7.5,
        color="#666660",
    )
    fig.tight_layout(rect=(0, 0.028, 1, 0.895))
    for ext in ("png", "pdf"):
        fig.savefig(path.with_suffix("." + ext), dpi=200, bbox_inches="tight")
    plt.close(fig)


def summarize(rows, cutoffs) -> dict:
    """Machine-readable verdict: the rank correlation of each readout against τ over the swept range,
    plus the readouts at the campaign's own bars and at the extremes."""
    out: dict = {}
    for name in {r["set"] for r in rows}:
        pts = sorted((r for r in rows if r["set"] == name), key=lambda r: r["cutoff"])
        entry = {"n_cutoffs": len(pts), "spearman_vs_cutoff": {}, "at_cutoff": {}}
        for metric, _, _ in METRICS:
            entry["spearman_vs_cutoff"][metric] = spearman(
                [p["cutoff"] for p in pts], [p.get(metric) for p in pts]
            )
        for bar in list(CAMPAIGN_BARS) + [min(cutoffs), max(cutoffs)]:
            hit = next((p for p in pts if abs(p["cutoff"] - bar) < 1e-9), None)
            if hit:
                entry["at_cutoff"][f"{bar:g}"] = {
                    k: hit.get(k)
                    for k in (
                        "n_above_cutoff",
                        "mean_pairwise_similarity",
                        "modes_per_mol",
                        "butina_per_mol",
                        "scaffolds_per_mol",
                    )
                }
        out[name] = entry
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample-records", type=Path, help="records.csv from the sampling stage")
    ap.add_argument("--enum-records", type=Path, help="enumerated_records.csv from the enumeration")
    ap.add_argument("--cutoffs", default="1,2,3,4.0:8.0:0.1")
    ap.add_argument(
        "--subsample",
        type=int,
        default=2500,
        help="molecules scored per cutoff (identical at every cutoff, so mode/scaffold counts are "
        "not confounded by set size)",
    )
    ap.add_argument("--replicates", type=int, default=3, help="independent random draws per cutoff")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="scent_seh")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figure from results/<tag>/reward_diversity.csv without re-sweeping",
    )
    args = ap.parse_args()

    cutoffs = parse_cutoffs(args.cutoffs)
    out_dir = HERE / "results" / args.tag
    if args.plot_only:
        rows = read_csv(out_dir / "reward_diversity.csv")
        plot(out_dir / "reward_diversity", rows, args.tag)
        print(f"redrew {out_dir}/reward_diversity.{{png,pdf}} from {len(rows)} committed rows")
        return

    if not args.sample_records and not args.enum_records:
        ap.error("give at least one of --sample-records / --enum-records")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, detail, meta = [], [], {}
    for name, path in (("sample", args.sample_records), ("enumerated", args.enum_records)):
        if not path:
            continue
        t0 = time.time()
        smiles, rewards = load_pool(Path(path))
        t_load = time.time() - t0
        print(
            f"[{name}] {path}\n  {len(smiles)} unique molecules loaded in {t_load:.1f}s "
            f"(reward {min(rewards):.2f}-{max(rewards):.2f})",
            flush=True,
        )
        t0 = time.time()
        r, d, n_cached = sweep_pool(
            name,
            smiles,
            rewards,
            cutoffs,
            subsample=args.subsample,
            replicates=args.replicates,
            seed=args.seed,
        )
        rows += r
        detail += d
        meta[name] = {
            "records": str(path),
            "n_unique_molecules": len(smiles),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "n_molecules_fingerprinted": n_cached,
            "load_s": round(t_load, 1),
            "sweep_s": round(time.time() - t0, 1),
        }
        print(f"  [{name}] swept in {meta[name]['sweep_s']}s ({n_cached} molecules fingerprinted)")

    write_csv(out_dir / "reward_diversity.csv", rows)
    payload = {
        "tag": args.tag,
        "model": "scent",
        "reward_name": "seh",
        "higher_is_better": True,
        "cutoffs": cutoffs,
        "subsample": args.subsample,
        "replicates": args.replicates,
        "seed": args.seed,
        "mode_similarity_threshold": MODE_SIMILARITY,
        "fingerprint": "Morgan r=3, 2048 bits, no features/chirality",
        "pools": meta,
        "summary": summarize(rows, cutoffs),
        "per_replicate": detail,
    }
    with open(out_dir / "reward_diversity.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    if not args.no_plot:
        plot(out_dir / "reward_diversity", rows, args.tag)
    print(f"\nwrote {out_dir}/reward_diversity.{{csv,json,png,pdf}}")
    for name, entry in payload["summary"].items():
        rho = entry["spearman_vs_cutoff"]
        # "n/a" is a real outcome, not a bug: a readout that is constant across the sweep (e.g. a
        # saturated scaffolds/mol) has no rank correlation to report.
        fmt = lambda v: "  n/a" if v is None else f"{v:+.3f}"  # noqa: E731
        print(
            f"  {name}: spearman(τ, ·) pairwise {fmt(rho['mean_pairwise_similarity'])} | "
            f"modes/mol {fmt(rho['modes_per_mol'])} | butina/mol {fmt(rho['butina_per_mol'])} | "
            f"scaffolds/mol {fmt(rho['scaffolds_per_mol'])}"
        )


if __name__ == "__main__":
    main()
