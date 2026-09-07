#!/usr/bin/env python
"""Accept or reject one ``benchmark_v2`` cell. Runbook §6, as a single gating command.

EVERY CHECK HERE IS A FAILURE THAT ACTUALLY HAPPENED, and every one of them produced a confident
wrong answer rather than an error. That is the class of problem this file exists for: an empty
``routes.json`` made SPARROW price the EMPTY library as trivially ``Optimal`` at zero cost; a
partially-populated enumeration priced a MIXTURE (some children at their true molecule, the rest at
their hub) and still returned ``Optimal``; a snapshot from a different model read 100% complete while
only 53% of the run's fragments were expandable. None of them raised.

THE ONE-LINE RULE, from the runbook: every one of those was an EXISTENCE check where the real
question was a MATCH or a CONTENT check. The file was present, well-formed, and wrong. When adding a
check here, ask what it would take for it to pass on bad data -- and check that instead.

CHECKS ARE REUSED, NOT REIMPLEMENTED. The route/enumeration/recipe checks import
``matrix16/check_route_readiness``, because those functions carry corrections that are not obvious
and would be lost in a rewrite -- most importantly that enumeration coverage must be judged on the
MERGED ``enum_children.json`` alone, never the merge plus the slices it superseded, which once
condemned a finished cell by counting its own discarded inputs (rgfn_clpp s43 read "84% partial"
when the merged artifact was 100% complete).

TWO STAGES, because "verified" means different things at different points:

  --stage train     the artifact a re-run cannot recreate: checkpoint, trace, arm metadata.
                    Applies to ALL NINE generators. This is what gates freezing.
  --stage campaign  the artifacts a chemist needs: routes, per-child reactions, expandable recipes.
                    Applies to the reaction-GFN pipeline only.

⚠ CONTRACT FOR THE TRAINING RUNNERS (agent A implements, this file enforces). A cell's train dir
must contain ``arm_meta.json``:

    {"arm": "a", "budget_calls": 10000, "n_scored_at_checkpoint": 10004,
     "checkpoint": "checkpoint.pt", "pythonhashseed": "0", "generator": "scent",
     "batch_size": 64, "commit": "<git sha>"}

``n_scored_at_checkpoint`` is the load-bearing field and it must be READ FROM THE TRACE, never
computed as batch x steps: the three reaction-GFNs have three different per-step call counts (RGFN
100, SCENT 64, RxnFlow 64) and replay buffers make that arithmetic unsettleable. Recording it is what
makes "this checkpoint is the 10,000-call one" an auditable claim rather than an assumption.

On success writes ``.verified.json`` beside the artifacts; ``freeze_cell.sh`` refuses to freeze a
cell without one. Exit code is non-zero on any failure, so a driver can gate on it.

    python experiments/benchmark_v2/tools/verify_cell.py --cell scent/seh/42 --arm a
    python experiments/benchmark_v2/tools/verify_cell.py --cell scent/seh/42 --stage campaign
    python experiments/benchmark_v2/tools/verify_cell.py --all --stage train        # sweep
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO_ROOT = V2_ROOT.parents[1]

# TWO MODULES ARE NAMED `manifest` -- this tree's and matrix16's -- so a plain sys.path import
# resolves to whichever directory happens to sit earlier, which is decided by import ORDER rather
# than intent. It silently picked the wrong one once already (matrix16's `select` has a different
# signature, so the failure surfaced as `TypeError: 'int' object is not iterable` several frames
# away from the actual cause). Load the v1 helpers from their FILE PATH instead: explicit, and
# immune to whatever any imported module does to sys.path afterwards.
sys.path.insert(0, str(HERE))
from manifest import Cell, get_cell, select  # noqa: E402  -- this tree's, unambiguously


def _load_from_path(name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reused, never reimplemented: these carry corrections that are invisible in a rewrite -- above all
# that enumeration coverage is judged on the MERGED enum_children.json alone, never the merge plus
# the slices it superseded.
_crr = _load_from_path(
    "_v1_check_route_readiness",
    REPO_ROOT / "experiments" / "lsd_hubs" / "matrix16" / "check_route_readiness.py",
)
_enum_rxn_coverage = _crr._enum_rxn_coverage
_n_routes = _crr._n_routes
_recipe_health = _crr._recipe_health

TRACE_FIELDS = ["n_scored", "n_distinct", "phase", "step", "smiles", "raw_score", "elapsed_s"]
# A trace may legitimately overshoot its budget (a batch straddles the boundary) but must not fall
# meaningfully short. 5% is one batch at any of our batch sizes.
BUDGET_TOLERANCE = 0.95


class Result:
    def __init__(self) -> None:
        self.checks: List[Tuple[str, bool, str]] = []

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append((name, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c[1] for c in self.checks)

    def render(self, indent: str = "  ") -> str:
        return "\n".join(
            f"{indent}{'PASS' if ok else 'FAIL'}  {name:<28} {detail}"
            for name, ok, detail in self.checks
        )


def _declared_promoted_count(cell: Cell, arm: str) -> Optional[int]:
    """How many dynamic-library fragments this run promoted, as the RUN ITSELF declares it.

    Read from the worker's own ``meta.json`` (``n_promoted_fragments``), never inferred from whether
    a snapshot file happens to exist -- because "no snapshot" is ambiguous between "promoted
    nothing" (fine) and "recipes were never logged" (unrecoverable, and the reason six v1 cells are
    dead). ``None`` means the run did not declare it, which is itself a failure: it makes the two
    indistinguishable.
    """
    for sub in ("enum", "sample"):
        p = cell.scratch_campaign_dir(arm) / sub / "meta.json"
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        if "n_promoted_fragments" in d:
            return int(d["n_promoted_fragments"])
    return None


# ---------------------------------------------------------------------------- train stage
def verify_train(cell: Cell, arm: str) -> Result:
    r = Result()
    d = cell.train_dir(arm)
    if not d.is_dir():
        r.add("train dir", False, f"missing: {d}")
        return r
    r.add("train dir", True, str(d))

    meta_p = d / "arm_meta.json"
    meta = None
    if not meta_p.is_file():
        r.add("arm_meta.json", False, "missing -- the runner must record which call count this "
                                     "checkpoint sits at (see module docstring)")
    else:
        try:
            meta = json.loads(meta_p.read_text())
            r.add("arm_meta.json", True, "")
        except Exception as e:
            r.add("arm_meta.json", False, f"unparseable: {e}")

    # -- the checkpoint itself
    ckpt_name = (meta or {}).get("checkpoint", "checkpoint.pt")
    ckpt = d / ckpt_name
    r.add("checkpoint", ckpt.is_file(),
          f"{ckpt_name} ({ckpt.stat().st_size/1e6:.0f} MB)" if ckpt.is_file() else f"missing {ckpt_name}")

    # -- the trace: present, well-formed, monotone, and long enough
    budget = cell.arm_calls(arm) or 0
    tp = cell.trace_path(arm)
    if not tp.is_file():
        # A missing trace does not make the cell WRONG -- it costs Stage 2 its free pool of
        # already-scored molecules. But the cell is never ACCEPTED without one, because the arm's
        # budget is then an assertion with no evidence behind it.
        r.add("trace.csv", False, "missing -- arm budget cannot be evidenced, and Stage 2 loses "
                                  "its free-pool harvest")
        return r

    n_rows = last_scored = 0
    prev = 0
    monotone = True
    phases = set()
    max_distinct = 0
    n_train_rows = 0
    try:
        with open(tp) as fh:
            rd = csv.DictReader(fh)
            missing = [c for c in ("n_scored", "phase") if c not in (rd.fieldnames or [])]
            if missing:
                r.add("trace schema", False, f"missing columns {missing}; want {TRACE_FIELDS}")
                return r
            r.add("trace schema", True, ",".join(rd.fieldnames or []))
            for row in rd:
                n_rows += 1
                try:
                    ns = int(row["n_scored"])
                except (TypeError, ValueError):
                    continue
                if ns < prev:
                    monotone = False
                prev = last_scored = ns
                ph = (row.get("phase") or "").strip()
                phases.add(ph)
                if ph == "train":
                    n_train_rows += 1
                try:
                    max_distinct = max(max_distinct, int(row.get("n_distinct") or 0))
                except ValueError:
                    pass
    except OSError as e:
        r.add("trace.csv", False, f"unreadable: {e}")
        return r

    # A HEADER-ONLY TRACE IS THE CHARACTERISTIC STUB, NOT MERELY AN EMPTY FILE. Re-invoking a
    # runner truncates trace.csv to its 59-byte header while the real history survives as
    # `trace.csv.1` (what the TraceWriter rotation in 3281bce buys). Nine of ten cells first read as
    # historyless turned out to have complete 10k-12k-row traces in their rotations. So a copy that
    # grabs `trace.csv` blindly carries the stub -- and if that stub were allowed to pass here, Stage 2
    # would pay to re-sample a pool we already hold. Fail it loudly and name the rotations.
    if n_rows == 0:
        rotations = sorted(tp.parent.glob(tp.name + ".*"))
        hint = (f" -- but {len(rotations)} rotation(s) exist ({', '.join(p.name for p in rotations)}); "
                f"the real history is probably in one of them") if rotations else ""
        r.add("trace rows", False, f"HEADER-ONLY stub ({tp.stat().st_size} bytes){hint}")
        return r
    r.add("trace rows", True, f"{n_rows:,} rows, final n_scored={last_scored:,}")
    r.add("trace monotone", monotone,
          "n_scored never decreases" if monotone else "n_scored DECREASES -- rows are interleaved "
          "or the file was appended to by two writers")
    # phase is load-bearing: some generators score outside the training loop (S3-GFN's evaluate()
    # scores a 1,000-molecule sample), and counting those inflates the budget AND puts molecules on
    # the learning curve the policy never learned from.
    r.add("trace phase column", "train" in phases,
          f"phases present: {sorted(p for p in phases if p)}")

    # THE BUDGET IS THE COUNT OF TRAINING ROWS, NOT THE FINAL COUNTER. `n_scored` is a single
    # cumulative counter SHARED across phases, and at least one entrant interleaves evaluation with
    # training rather than appending it: S3-GFN scores a 2,000-molecule eval sample mid-run, so on
    # s3gfn_drd2/42 the counter ends at 12,048 while the actual training budget is 10,048 -- three
    # phase switches, eval rows spanning n_scored 65..12,048. Reading the counter would let a cell
    # that is genuinely SHORT on training pass because evaluation padded it over the line, which is
    # the exact failure this gate exists to prevent. (Both figures are reported: for the five
    # entrants with no eval phase they agree, and where they disagree the gap is the contamination.)
    eval_pad = last_scored - n_train_rows
    r.add("budget reached (train rows)", n_train_rows >= budget * BUDGET_TOLERANCE,
          f"{n_train_rows:,} / {budget:,} ({100*n_train_rows/max(budget,1):.0f}%)"
          + (f"   [final counter {last_scored:,}, {eval_pad:,} non-train rows]" if eval_pad else ""))
    if max_distinct:
        # The gap between the two counters is itself a mode-collapse signal, so it is reported
        # rather than merely bounded.
        r.add("distinct <= scored", max_distinct <= last_scored,
              f"{max_distinct:,} distinct of {last_scored:,} scored "
              f"({100*max_distinct/max(last_scored,1):.0f}% unique)")

    # -- the checkpoint sits where the metadata claims
    if meta and "n_scored_at_checkpoint" in meta:
        at = int(meta["n_scored_at_checkpoint"])
        ok = at >= budget * BUDGET_TOLERANCE and at <= last_scored
        r.add("checkpoint placement", ok,
              f"recorded at n_scored={at:,} (budget {budget:,}, trace ends {last_scored:,})")
    else:
        r.add("checkpoint placement", False,
              "arm_meta.json does not record n_scored_at_checkpoint")

    # -- determinism. Without PYTHONHASHSEED a sample is a one-of-a-kind artifact recoverable only
    # from backup: --seed alone gave 377 vs 387 routes; with it, 730/730 byte-identical.
    if meta is not None:
        hs = str(meta.get("pythonhashseed", ""))
        r.add("PYTHONHASHSEED", hs == "0", f"recorded as {hs!r}" if hs else "not recorded")
    return r


# ------------------------------------------------------------------------- campaign stage
def verify_campaign(cell: Cell, arm: str) -> Result:
    r = Result()
    if not cell.is_hub_batching:
        r.add("pipeline", True, "competitor cell -- campaign stage is n/a")
        return r

    sample = cell.sample_dir(arm)
    enum = cell.enum_dir(arm)

    # §6.1 routes were emitted. Non-zero for rgfn/rxnflow/scent; FragGFN never reaches here.
    n_routes = _n_routes(sample)
    if cell.route_bearing:
        r.add("routes.json (§6.1)", bool(n_routes),
              f"{n_routes:,} routes" if n_routes else
              "EMPTY or missing -- downstream every hub is skipped and SPARROW prices the empty "
              "library as trivially Optimal at zero cost")
        rs = sample / "route_status.json"
        r.add("route_status.json", rs.is_file(),
              "written by the write-time contract check" if rs.is_file() else "missing")
    else:
        r.add("routes.json (§6.1)", True, "n/a -- attachments, not reactions; empty is CORRECT")

    # §6.2 every enumerated child carries its reaction. Anything strictly between 0 and 1 is MORE
    # dangerous than 0: it prices a mixture and still returns Optimal.
    cov = _enum_rxn_coverage(enum)
    if cov is None:
        r.add("children[].reaction (§6.2)", False, f"no enumeration under {enum}")
    else:
        n_child, n_rxn, n_hubs = cov
        frac = n_rxn / n_child if n_child else 0.0
        r.add("children[].reaction (§6.2)", n_child > 0 and frac >= 1.0,
              f"{n_rxn:,}/{n_child:,} = {frac:.4f} over {n_hubs:,} hubs"
              + ("" if frac >= 1.0 else "  <- a partial artifact prices SOME children at their hub"))

    # §6.3 recipes exist AND belong to this run.
    #
    # THREE STATES, NOT TWO -- and the third is the one that matters. "This run promoted nothing, so
    # there is nothing to expand" is a legitimate answer, but it must be DECLARED with a reason and
    # read off the run's own artifact, never inferred from a missing file. That distinction is one
    # the project has already paid for once: FragGFN's empty routes.json is CORRECT (its move is an
    # attachment, not a reaction) and for months looked identical on disk to "nobody implemented
    # it". Same shape here.
    #
    # WHY IT BITES NOW, AND HARD. SCENT's DynamicLibrary promotes on
    # `every_n_iterations = 1000` (verified in the clone: n_iterations_schedule starts at 1000), and
    # SCENT's batch is 64 -- so the FIRST promotion needs 64,000 oracle calls. Arm A's whole budget
    # is 10,000, about 156 iterations. **Every SCENT arm-A cell therefore promotes zero fragments**,
    # has no `additional_fragments/fragments_<N>.json` at all, and every route bottoms out directly
    # at base stock. Treating that as 0% coverage would fail every one of them. (v1 at 5,000
    # iterations recorded n_promoted_fragments=1600 = 4 of 10 promotions x 400, so the arithmetic
    # is consistent in both directions.)
    n_promoted = _declared_promoted_count(cell, arm)
    rh = _recipe_health(cell.scratch_campaign_dir(arm))
    if cell.generator != "scent":
        r.add("recipes (§6.3)", True, "n/a -- no dynamic library, routes bottom out at stock")
    elif n_promoted is None:
        r.add("recipes (§6.3)", False,
              "cannot tell: the run declares no n_promoted_fragments in its meta.json, so "
              "'nothing to expand' and 'recipes lost' are indistinguishable")
    elif n_promoted == 0:
        r.add("recipes (§6.3)", True,
              "n/a -- this run promoted 0 fragments (declared), so every route bottoms out at "
              "base stock and there is nothing to expand")
    elif rh is None:
        r.add("recipes (§6.3)", False,
              f"run declares {n_promoted:,} promoted fragments but its snapshot could not be "
              f"resolved from meta.json -- the recipes may be the unrecoverable kind")
    else:
        covr, snap = rh
        r.add("recipes (§6.3)", covr >= 1.0,
              f"coverage {covr:.1%} of this run's {n_promoted:,} promoted fragments "
              f"({Path(snap).name})"
              + ("" if covr >= 1.0 else "  <- the rest are BOUGHT, not built"))
    return r


def _write_marker(cell: Cell, arm: str, stage: str, res: Result) -> Path:
    p = cell.verified_marker(arm) if stage == "train" else cell.enum_dir(arm) / ".verified.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if p.is_file():
        try:
            prev = json.loads(p.read_text())
        except Exception:
            prev = {}
    prev[stage] = {
        "verified_at": _dt.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in res.checks],
    }
    p.write_text(json.dumps(prev, indent=2, sort_keys=True))
    return p


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cell", metavar="GEN/TARGET/SEED")
    g.add_argument("--all", action="store_true", help="sweep every cell in the grid")
    ap.add_argument("--arm", default="a", choices=("a", "b"))
    ap.add_argument("--stage", default="train", choices=("train", "campaign", "both"))
    ap.add_argument("--phase", type=int, default=None, help="with --all: restrict to a phase")
    ap.add_argument("--no-marker", action="store_true",
                    help="report only; do not write .verified.json")
    a = ap.parse_args()

    if a.cell:
        gen, tgt, seed = a.cell.split("/")
        cells = [get_cell(gen, tgt, int(seed))]
    else:
        cells = [c for c in select(phase=a.phase) if c.has_arm(a.arm)]

    stages = ["train", "campaign"] if a.stage == "both" else [a.stage]
    n_ok = n_total = 0
    for cell in cells:
        if not cell.has_arm(a.arm):
            continue
        for stage in stages:
            if stage == "campaign" and not cell.is_hub_batching:
                continue
            res = verify_train(cell, a.arm) if stage == "train" else verify_campaign(cell, a.arm)
            n_total += 1
            n_ok += bool(res.ok)
            head = f"{cell.tag}  arm{a.arm}  stage={stage}"
            print(f"\n{'=' * len(head)}\n{head}\n{'=' * len(head)}")
            print(res.render())
            if res.ok:
                if not a.no_marker:
                    p = _write_marker(cell, a.arm, stage, res)
                    print(f"  -> ACCEPTED, marker at {p}")
                else:
                    print("  -> would be ACCEPTED (marker suppressed)")
            else:
                print("  -> REJECTED")

    print(f"\n{n_ok}/{n_total} checks passed")
    return 0 if n_ok == n_total else 1


if __name__ == "__main__":
    raise SystemExit(main())
