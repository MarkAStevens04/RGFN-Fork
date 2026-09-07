#!/usr/bin/env python
"""The ONE place that resolves a ``benchmark_v2`` cell — for BOTH pipelines, BOTH arms.

A "cell" is one (generator x target x seed). This module joins four things into a single
:class:`Cell` every driver, verifier and analysis consumes:

  1. the plan (``grid.csv``: class, role, pipeline, phase, arm budgets, intended origin),
  2. the per-target science (``matrix16/targets.py``: gate value AND direction, oracle),
  3. resolved paths for every stage of whichever pipeline the cell runs through, and
  4. **live filesystem status**, computed at load time so the manifest never goes stale.

THREE DEFECTS IN THE v1 MANIFEST THAT THIS ONE FIXES BY CONSTRUCTION. They are recorded here
because each cost real work, and a successor that reintroduces any of them will not be obvious:

  * **The tag was not seed-aware.** v1's tag was ``(generator, target)`` with the seed nowhere in
    it, so a second seed reusing the manifest collided in both the scratch and results trees. The
    workaround was three env vars that had to be redirected TOGETHER (redirect only the scratch one
    and the campaign still wrote to the committed results -- which is how a 24-hub harvest test
    overwrote a real cell's curve). Here the seed is IN the tag, so three seeds cannot collide and
    no env var is load-bearing for correctness.
  * **``submit_cell.sh`` resolved ``SAMPLE_DIR`` from the manifest via ``eval``, clobbering any env
    override.** A caller who set ``SAMPLE_DIR=<somewhere safe>`` was silently ignored and the real
    cell was overwritten in place. Here every stage path is a pure function of (tag, arm), and the
    ONE override that exists redirects the whole tree at its root -- so it is impossible to redirect
    half of it.
  * **Status lived in a CSV.** A status column in a plan file is stale the moment anything runs.
    ``grid.csv`` holds only the plan; everything about reality is computed here, every load.

STDLIB ONLY, ON PURPOSE. A bare SLURM batch shell has no python on PATH until a conda env is
activated, yet this module is what tells the launcher WHICH env the cell needs. conda base always
has python, so the launcher bootstraps with base, emits the spec, then activates the cell's env.
Importing anything heavier would break that ordering (it already did once in v1).

CLI
    python experiments/benchmark_v2/tools/manifest.py                  # live status table
    python experiments/benchmark_v2/tools/manifest.py --status         # + plan-vs-disk reconciliation
    python experiments/benchmark_v2/tools/manifest.py --emit scent seh 42 --arm a
    python experiments/benchmark_v2/tools/manifest.py --list --phase 1 --role hub_batching
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
V2_ROOT = HERE.parent
REPO_ROOT = V2_ROOT.parents[1]

sys.path.insert(0, str(REPO_ROOT / "experiments" / "lsd_hubs" / "matrix16"))
from targets import Target, get_target  # noqa: E402

GRID_CSV = Path(os.environ.get("BENCHMARK_V2_GRID", str(V2_ROOT / "grid.csv")))
# ONE override, redirecting the WHOLE heavy tree. Deliberately not three: v1's separate scratch and
# results knobs made it possible to redirect half a run, and that is exactly what went wrong.
SCRATCH_ROOT = Path(os.environ.get("BENCHMARK_V2_SCRATCH", "/scratch/markymoo/rgfn_runs/v2"))
RESULTS_ROOT = Path(os.environ.get("BENCHMARK_V2_RESULTS", str(V2_ROOT / "results")))

ARMS = ("a", "b")

# THREE POOL VALUES, AND THE THIRD IS RESTRICTED ON PURPOSE (settled 2026-09-07).
#
#   naive      top-N by reward, whatever diversity the generator's own machinery produced
#   pruned     N guaranteed-distinct molecules, our sphere-exclusion rule applied to its output
#   full_pool  routes over the WHOLE emitted pool -- a superset of both, not a variant of either
#
# `full_pool` exists because TANGO and SynFormer ship routes by construction: syntheseus searches
# each target INDEPENDENTLY, so a route set over the full pool genuinely CONTAINS the routes for any
# subset. Route once, subset per variant at read time -- cheaper, and strictly more informative than
# two overlapping runs.
#
# It is REFUSED for the route-planned generators, and that refusal is the point. MultiAiZ is
# SET-BASED: planning 500 diverse molecules converges on different shared intermediates than planning
# 500 near-duplicates, so its naive and pruned results are not subsets of a larger run and never can
# be. Someone later "optimising" the pipeline by planning once and subsetting would silently price a
# library on intermediates the planner never actually shared -- a wrong answer that looks like a
# saving. Encoding it here rather than in a README on one directory is what makes it a guard.
POOLS = ("naive", "pruned", "full_pool")
ROUTE_NATIVE = {"synformer", "tango"}

# Which conda env and entry point each generator runs under. Mirrors
# experiments/fixed_reward/scale5k/submit_baseline.sh -- the two must not drift, and the awkward
# rows are the informative ones: REINVENT's env is `reinvent4`, and TANGO is not a fifth model but
# Saturn's agent with an extra oracle, so it shares Saturn's env AND runner.
_ENV = {"reinvent": "reinvent4", "tango": "saturn"}
_RUNNER = {
    "reinvent": "validation/generators/reinvent/run_reinvent_fixed.py",
    "saturn": "validation/generators/saturn/run_saturn_fixed.py",
    "tango": "validation/generators/saturn/run_saturn_fixed.py",
    "synformer": "validation/generators/synformer/run_synformer_fixed.py",
    "s3gfn": "validation/generators/s3gfn/run_s3gfn_fixed.py",
    "fraggfn": "validation/generators/fraggfn/run_fraggfn_fixed.py",
    "rxnflow": "validation/generators/rxnflow/run_rxnflow_fixed.py",
    "scent": "validation/generators/scent/run_scent_fixed.py",
    # RGFN is the odd one out: it needs no cross-env bridge because it runs in the same env as
    # `glue` and reaches its oracle in-process through gin.
    "rgfn": "scripts/fixed_reward.py",
}
# Generators whose moves ARE synthesis steps, so a sample stage must emit routes.json. Mirrors
# validation/lsdflow/adapters/workers/_routes.py ROUTE_CONTRACT -- FragGFN's empty routes.json is
# CORRECT (attachments, not reactions) and must never be "fixed".
_ROUTE_BEARING = {"rgfn", "rxnflow", "scent"}


@dataclass
class Cell:
    """One (generator x target x seed): plan + resolved target + resolved paths + live status."""

    generator: str
    gen_class: str
    role: str          # hub_batching | competitor
    pipeline: str      # ours | competitor
    target_name: str
    seed: int
    phase: int
    arm_a_calls: int
    arm_b_calls: Optional[int]
    train_plan: str    # generate | copy -- a PLAN, see build_grid.py
    note: str

    # -- identity ---------------------------------------------------------------
    @property
    def tag(self) -> str:
        """Stable cell id. The seed is IN it, which is the v1 collision fix."""
        return f"{self.generator}_{self.target_name}_s{self.seed}"

    @property
    def target(self) -> Target:
        """Raises on an unknown target -- deliberately. A silently-defaulted target is how a new
        system inherits the wrong gate direction."""
        return get_target(self.target_name)

    @property
    def is_hub_batching(self) -> bool:
        return self.role == "hub_batching"

    @property
    def route_bearing(self) -> bool:
        return self.generator in _ROUTE_BEARING

    @property
    def conda_env(self) -> str:
        return _ENV.get(self.generator, self.generator)

    @property
    def runner(self) -> str:
        return _RUNNER[self.generator]

    @property
    def worker(self) -> str:
        """Per-env sample/enumerate worker. Hub-batching cells only; competitors have no hub stage."""
        return f"validation/lsdflow/adapters/workers/{self.generator}_worker.py"

    def arm_calls(self, arm: str) -> Optional[int]:
        return self.arm_a_calls if arm == "a" else self.arm_b_calls

    def has_arm(self, arm: str) -> bool:
        """Arm B is the reaction-GFNs' continuation only -- a competitor never has one."""
        return self.arm_calls(arm) is not None

    # -- paths: every stage, a pure function of (tag, arm) -----------------------
    def train_dir(self, arm: str = "a") -> Path:
        return SCRATCH_ROOT / "train" / self.tag / f"arm{arm}"

    def scratch_campaign_dir(self, arm: str = "a") -> Path:
        """The parent holding ``sample/`` and ``enum/``. ``_recipe_health`` resolves a run's own
        fragment snapshot from the ``meta.json`` under one of those, so it wants this level."""
        return SCRATCH_ROOT / "campaign" / self.tag / f"arm{arm}"

    def sample_dir(self, arm: str = "a") -> Path:
        return self.scratch_campaign_dir(arm) / "sample"

    def enum_dir(self, arm: str = "a") -> Path:
        return self.scratch_campaign_dir(arm) / "enum"

    def _check_pool(self, pool: str) -> str:
        """Validate a pool value against this generator. Raises rather than returning a path.

        `full_pool` is permitted ONLY for generators that ship routes by construction -- see the
        POOLS comment. A route-planned generator asking for it is a category error, not a typo.
        """
        if pool not in POOLS:
            raise ValueError(f"unknown pool {pool!r}; known: {list(POOLS)}")
        if pool == "full_pool" and self.generator not in ROUTE_NATIVE:
            raise ValueError(
                f"{self.generator!r} is route-PLANNED (MultiAiZ), so it has no 'full_pool': "
                f"MultiAiZ is set-based, and its naive/pruned results are not subsets of a larger "
                f"run. Planning once and subsetting would price the library on intermediates the "
                f"planner never shared. Use 'naive' or 'pruned'. "
                f"(full_pool is for {sorted(ROUTE_NATIVE)}.)"
            )
        return pool

    def pool_dir(self, arm: str = "a", pool: str = "naive") -> Path:
        return SCRATCH_ROOT / "pools" / self.tag / f"arm{arm}" / self._check_pool(pool)

    def routes_dir(self, arm: str = "a", pool: str = "naive") -> Path:
        return SCRATCH_ROOT / "routes" / self.tag / f"arm{arm}" / self._check_pool(pool)

    def selection_dir(self, arm: str = "a", pool: str = "naive") -> Path:
        return SCRATCH_ROOT / "selection" / self.tag / f"arm{arm}" / self._check_pool(pool)

    def pools_for(self) -> tuple:
        """The pool variants this generator legitimately has. Drivers should iterate THIS, not POOLS."""
        return POOLS if self.generator in ROUTE_NATIVE else ("naive", "pruned")

    def results_dir(self, arm: str = "a") -> Path:
        return RESULTS_ROOT / self.tag / f"arm{arm}"

    def trace_path(self, arm: str = "a") -> Path:
        return self.train_dir(arm) / "trace.csv"

    def verified_marker(self, arm: str = "a") -> Path:
        return self.train_dir(arm) / ".verified.json"

    # -- live status -------------------------------------------------------------
    def train_exists(self, arm: str = "a") -> bool:
        return self.train_dir(arm).is_dir()

    def trace_rows(self, arm: str = "a") -> Optional[int]:
        """Final ``n_scored`` from the trace's last line -- a seek, not a scan, so a 108-cell status
        table stays instant. ``None`` when the file is absent or holds only a header.

        This is the number that decides where arm A's checkpoint belongs, which is why it is read
        from the trace rather than computed as batch x steps: the three reaction-GFNs have three
        different per-step call counts and replay buffers make that arithmetic unsettleable."""
        p = self.trace_path(arm)
        if not p.is_file():
            return None
        try:
            with open(p, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                if size == 0:
                    return None
                back = min(size, 8192)
                fh.seek(-back, os.SEEK_END)
                tail = fh.read().decode("utf-8", "replace").strip().splitlines()
            if len(tail) < 2 and size <= 8192:
                return None  # header only
            last = tail[-1].split(",")
            return int(last[0])
        except (OSError, ValueError, IndexError):
            return None

    def frozen(self, arm: str = "a") -> bool:
        """A frozen train dir is READ-ONLY. This is the structural guard against the failure that
        destroyed s3gfn_seh/seed43's training history: re-invoking a runner overwrites trace.csv,
        candidates.csv, timing.json and run_config.yaml in place."""
        d = self.train_dir(arm)
        return d.is_dir() and not os.access(d, os.W_OK)

    def verified(self, arm: str = "a") -> bool:
        return self.verified_marker(arm).is_file()

    def status(self, arm: str = "a") -> str:
        if not self.has_arm(arm):
            return "n/a"
        if not self.train_exists(arm):
            return "not-started"
        rows = self.trace_rows(arm)
        if rows is None:
            # The cell may still be perfectly usable -- a missing trace costs Stage 2 its free pool,
            # not correctness -- but it is never ACCEPTED without one, because the arm's budget
            # cannot be evidenced.
            return "no-trace"
        budget = self.arm_calls(arm) or 0
        if rows < budget * 0.95:
            return f"short-trace:{rows}/{budget}"
        if not self.verified(arm):
            return "unverified"
        return "frozen" if self.frozen(arm) else "verified"


def load_grid(path: Path = GRID_CSV) -> List[Cell]:
    if not path.is_file():
        raise SystemExit(
            f"no grid at {path}\n"
            f"  regenerate it:  python {HERE / 'build_grid.py'}"
        )
    cells: List[Cell] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            b = r.get("arm_b_calls", "").strip()
            cells.append(Cell(
                generator=r["generator"].strip(),
                gen_class=r["gen_class"].strip(),
                role=r["role"].strip(),
                pipeline=r["pipeline"].strip(),
                target_name=r["target"].strip(),
                seed=int(r["seed"]),
                phase=int(r["phase"]),
                arm_a_calls=int(r["arm_a_calls"]),
                arm_b_calls=int(b) if b else None,
                train_plan=r["train_plan"].strip(),
                note=r.get("note", "").strip(),
            ))
    return cells


def select(cells: Optional[List[Cell]] = None, *, generators=None, targets=None,
           seeds=None, phase=None, role=None, pipeline=None) -> List[Cell]:
    """Filter the grid. All criteria AND together; None = no constraint."""
    cells = cells if cells is not None else load_grid()
    out = []
    for c in cells:
        if generators and c.generator not in generators:
            continue
        if targets and c.target_name not in targets:
            continue
        if seeds and c.seed not in seeds:
            continue
        if phase is not None and c.phase != int(phase):
            continue
        if role and c.role != role:
            continue
        if pipeline and c.pipeline != pipeline:
            continue
        out.append(c)
    return out


def get_cell(generator: str, target: str, seed: int, cells=None) -> Cell:
    hits = select(cells, generators=[generator], targets=[target], seeds=[int(seed)])
    if not hits:
        raise KeyError(f"no cell for generator={generator!r} target={target!r} seed={seed!r}")
    return hits[0]


def emit_shell(cell: Cell, arm: str = "a") -> str:
    """Shell-sourceable ``KEY=VALUE`` view -- the bridge for the drivers
    (``eval "$(python manifest.py --emit <gen> <target> <seed> --arm a)"``).

    NOTE what is NOT here: nothing a caller might want to override. v1 emitted SAMPLE_DIR and the
    launcher ``eval``'d it, which silently clobbered any env override and overwrote live cells. Every
    path below is a pure function of (tag, arm); to run somewhere else, redirect
    ``BENCHMARK_V2_SCRATCH`` and the WHOLE tree moves together."""
    t = cell.target
    kv: Dict[str, object] = {
        "CELL_TAG": cell.tag,
        "GENERATOR": cell.generator,
        "GEN_CLASS": cell.gen_class,
        "ROLE": cell.role,
        "PIPELINE": cell.pipeline,
        "TARGET": cell.target_name,
        "SEED": cell.seed,
        "PHASE": cell.phase,
        "ARM": arm,
        "ARM_CALLS": cell.arm_calls(arm) if cell.has_arm(arm) else "",
        "REWARD_NAME": t.reward_name,
        "REWARD_TYPE": t.reward_type,
        "HIGHER_IS_BETTER": "true" if t.higher_is_better else "false",
        # The gate is resolved here, from targets.py, and never defaulted in a caller. Four scripts
        # once carried silent 7.0/0.0 defaults and produced mode counts on a bar nothing else used.
        "MODE_REWARD_THRESHOLD": t.mode_reward_threshold,
        "ORACLE": (t.oracle if t.is_docking else ""),
        "CONDA_ENV": cell.conda_env,
        "RUNNER": cell.runner,
        "WORKER": cell.worker if cell.is_hub_batching else "",
        "ROUTE_BEARING": "true" if cell.route_bearing else "false",
        "TRAIN_DIR": str(cell.train_dir(arm)),
        "TRACE": str(cell.trace_path(arm)),
        "RESULTS_DIR": str(cell.results_dir(arm)),
        "STATUS": cell.status(arm),
        "FROZEN": "true" if cell.frozen(arm) else "false",
        "TRAIN_PLAN": cell.train_plan,
    }
    if cell.is_hub_batching:
        kv["SAMPLE_DIR"] = str(cell.sample_dir(arm))
        kv["ENUM_DIR"] = str(cell.enum_dir(arm))
    else:
        # Only the pools this generator legitimately has -- a route-planned generator never gets a
        # FULL_POOL variable, so a driver cannot accidentally reference one.
        kv["POOLS"] = " ".join(cell.pools_for())
        for pool in cell.pools_for():
            kv[f"POOL_DIR_{pool.upper()}"] = str(cell.pool_dir(arm, pool))
            kv[f"ROUTES_DIR_{pool.upper()}"] = str(cell.routes_dir(arm, pool))
            kv[f"SELECTION_DIR_{pool.upper()}"] = str(cell.selection_dir(arm, pool))
    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in kv.items())


def _print_table(cells: List[Cell], arm: str) -> None:
    hdr = (f"{'cell':<26}{'class':<20}{'role':<14}{'ph':>3}{'gate':>9}  "
           f"{'plan':<9}{'trace':>9}  status")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        t = c.target
        gate = f"{'>' if t.higher_is_better else '<'}{t.mode_reward_threshold:g}"
        rows = c.trace_rows(arm)
        print(f"{c.tag:<26}{c.gen_class:<20}{c.role:<14}{c.phase:>3}{gate:>9}  "
              f"{c.train_plan:<9}{(rows if rows is not None else '-'):>9}  {c.status(arm)}")
    n = len(cells)
    done = sum(1 for c in cells if c.status(arm) in ("verified", "frozen"))
    print(f"\n{done}/{n} accepted (verified or frozen) on arm {arm}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--emit", nargs=3, metavar=("GEN", "TARGET", "SEED"),
                    help="shell-sourceable spec for one cell")
    ap.add_argument("--arm", default="a", choices=ARMS)
    ap.add_argument("--list", action="store_true", help="print tags only, one per line")
    ap.add_argument("--phase", type=int, default=None)
    ap.add_argument("--role", default=None, choices=("hub_batching", "competitor"))
    ap.add_argument("--generator", action="append", default=None)
    ap.add_argument("--target", action="append", default=None)
    ap.add_argument("--status", action="store_true",
                    help="status table plus a plan-vs-disk reconciliation summary")
    a = ap.parse_args()

    if a.emit:
        gen, tgt, seed = a.emit
        cell = get_cell(gen, tgt, int(seed))
        if not cell.has_arm(a.arm):
            raise SystemExit(f"{cell.tag} has no arm {a.arm} (arm B is reaction-GFNs only)")
        print(emit_shell(cell, a.arm))
        return 0

    cells = select(phase=a.phase, role=a.role, generators=a.generator, targets=a.target)
    cells = [c for c in cells if c.has_arm(a.arm)]

    if a.list:
        for c in cells:
            print(c.tag)
        return 0

    _print_table(cells, a.arm)

    if a.status:
        # The reconciliation the copy-forward step owns: grid.csv records what we INTENDED, the
        # filesystem records what happened. Printing them together is what stops a stale plan from
        # being mistaken for evidence.
        print("\nplan vs disk:")
        for plan in ("generate", "copy"):
            grp = [c for c in cells if c.train_plan == plan]
            started = [c for c in grp if c.train_exists(a.arm)]
            print(f"  train_plan={plan:<9} {len(grp):>3} cells, {len(started):>3} present on disk")
        print("  (train_plan is a PLAN, not an observation -- reconcile before relying on it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
