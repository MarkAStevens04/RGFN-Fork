#!/usr/bin/env python
"""The 16-cell LSD-Flow matrix manifest — the ONE place that resolves every cell's spec.

A "cell" = one (generator x target) run of the publication-scale 4x4 matrix (Logs/030).
This module joins three things into a single :class:`Cell` object every driver consumes:

  1. the static per-cell spec (``manifest.csv``: generator, target, seed, config, checkpoint,
     guidance sidecar),
  2. the per-target science (``targets.py``: reward gate value + direction + enumeration cost
     class), and
  3. **live filesystem status** (checkpoint present? SCENT sidecar present? how many candidates
     has training emitted?), computed at load time so the manifest never goes stale.

Output locations (the organization scheme — proposal §3 + Logs/030 layout):
  * heavy scratch artifacts (sampled DAG + enumeration):
    ``$SCRATCH/rgfn_runs/lsdflow/matrix16/<gen>_<target>/{sample,enum}/``
  * committed small results (campaign readouts, summaries):
    ``experiments/lsd_hubs/matrix16/results/<gen>_<target>/``

Run policy: ``surrogate`` cells (sEH/DRD2) are **active** now; ``docking`` cells (6TD3/ClpP) are
**deferred** (they need the GPU-docking enumeration path + finished training). ``Cell.run_stage``
exposes this; ``select(..., stage="active")`` filters to the runnable set.

CLI: ``python experiments/lsd_hubs/matrix16/manifest.py`` prints a live status table.
"""

from __future__ import annotations

import csv
import glob
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # allow `import targets` whether run as script or module
    sys.path.insert(0, str(HERE))
from targets import Target, get_target  # noqa: E402

REPO_ROOT = HERE.parents[2]  # experiments/lsd_hubs/matrix16 -> repo root
MANIFEST_CSV = HERE / "manifest.csv"
RESULTS_ROOT = HERE / "results"
SCRATCH_ROOT = Path(
    os.environ.get("MATRIX16_SCRATCH", "/scratch/markymoo/rgfn_runs/lsdflow/matrix16")
)


@dataclass
class Cell:
    """One (generator x target) matrix cell: static spec + resolved target + live status."""

    generator: str
    target_name: str
    seed: int
    config: str  # repo-relative
    checkpoint: str  # absolute (scratch)
    guidance_sidecar: str  # absolute (scent only) or ""

    @property
    def target(self) -> Target:
        return get_target(self.target_name)

    @property
    def tag(self) -> str:
        """Stable cell id used for scratch/results dir names + logs."""
        return f"{self.generator}_{self.target_name}"

    @property
    def reward_type(self) -> str:
        return self.target.reward_type

    @property
    def run_stage(self) -> str:
        """'active' (surrogate, run now) | 'deferred' (docking, wire but hold)."""
        return "deferred" if self.target.is_docking else "active"

    @property
    def conda_env(self) -> str:
        """The conda env the generator's worker runs in (== generator name for all four)."""
        return self.generator

    @property
    def worker(self) -> str:
        """Repo-relative path to the generator's per-env worker (sample + enumerate)."""
        return f"validation/lsdflow/adapters/workers/{self.generator}_worker.py"

    # -- resolved paths ---------------------------------------------------------
    @property
    def config_path(self) -> Path:
        return REPO_ROOT / self.config

    @property
    def scratch_dir(self) -> Path:
        return SCRATCH_ROOT / self.tag

    @property
    def sample_dir(self) -> Path:
        return self.scratch_dir / "sample"

    @property
    def enum_dir(self) -> Path:
        return self.scratch_dir / "enum"

    @property
    def results_dir(self) -> Path:
        return RESULTS_ROOT / self.tag

    # -- live filesystem status (computed each load, never frozen in the CSV) ----
    @property
    def checkpoint_exists(self) -> bool:
        return bool(self.checkpoint) and os.path.exists(self.checkpoint)

    @property
    def guidance_ok(self) -> bool:
        """SCENT needs the P_B sidecar (entry 024) for exact flow recovery; N/A for others."""
        if self.generator != "scent":
            return True
        return bool(self.guidance_sidecar) and os.path.exists(self.guidance_sidecar)

    @property
    def n_candidates(self) -> int:
        """Max candidates any candidates.csv in THIS checkpoint's own run dir holds — a cheap
        training-completion proxy (surrogate ~1000, docking ~200 when done).

        Derived from the checkpoint path, not a guessed ``<tag>_5k`` dir, so a cell may point at any
        run location (e.g. the cap-6 re-run ``fraggfn_drd2_maxfrag6/<timestamp>/``) and still resolve.
        Walks up from ``.../checkpoints/last_gfn.pt`` — the run root is 1 level up for
        fraggfn/rxnflow and 2 for rgfn/scent (which nest a ``train/`` dir)."""
        if not self.checkpoint:
            return 0
        ck = Path(self.checkpoint)
        best = 0
        for up in (2, 3, 4):  # checkpoints/ -> run root, allowing the extra train/ level
            if len(ck.parents) <= up:
                break
            for h in glob.glob(f"{ck.parents[up]}/**/candidates.csv", recursive=True):
                try:
                    with open(h) as fh:
                        best = max(best, sum(1 for _ in fh) - 1)
                except OSError:
                    pass
            if best:
                break
        return best

    @property
    def docking_wired(self) -> bool:
        """Whether this generator's worker can ACTUALLY score this target.

        ``ready`` used to mean only "the checkpoint and its sidecar exist", which for a docking cell
        is necessary but not sufficient: rgfn_worker and fraggfn_worker have no docking path at all
        (fraggfn raises "reward not wired", rgfn never references a docking bridge). Those cells
        still reported ``ready``, so ``submit_docking_cell.sh`` -- which gates on exactly that --
        would launch, construct the oracle (~35-44 s), run a real preflight dock, and only then die
        in the worker. Cheap, but it reads as a cluster problem rather than a missing feature, and it
        misled a cross-cluster hand-off into planning 8 docking cells when only 4 can run.

        Detected from the worker source rather than a hand-maintained list, so wiring a generator
        flips it automatically. Surrogate targets are always wired (in-process proxy, no bridge).

        Detects the REFUSAL, not the capability. A first attempt looked for the cross-env bridge
        classes and gave a FALSE NEGATIVE on RGFN, which needs no bridge at all: it runs in the same
        env as ``glue`` and reaches the oracle in-process through gin (``@OracleRewardProxy`` wrapping
        ``@DockingClpPOracle``), so the class names never appear in its worker. Every worker that
        cannot dock says so explicitly with a "not wired" SystemExit, and that is the reliable
        signal."""
        if not self.target.is_docking:
            return True
        try:
            src = (REPO_ROOT / self.worker).read_text()
        except OSError:
            return False
        # An explicit refusal naming this target's reward is the negative signal.
        for line in src.splitlines():
            if "not wired" in line and ("6td3" in line or "clpp" in line or "reward_name" in line):
                # a guard exists; wired iff the worker ALSO builds a docking reward
                return any(
                    k in src
                    for k in ("DockingBridgeReward", "DockingBridgeProxy", "OracleRewardProxy")
                )
        return True

    @property
    def ready(self) -> bool:
        """Analysis-ready = trained checkpoint present + (SCENT) sidecar present + training has
        emitted candidates. NOTE: candidate-presence is a proxy for "training finished"; verify
        ``torch.load(ckpt)['metrics']['epoch']`` == target iters before a *headline* run
        (verify-checkpoint-trained memory)."""
        return (
            self.checkpoint_exists
            and self.guidance_ok
            and self.n_candidates > 0
            and self.docking_wired
        )

    def status(self) -> str:
        if not self.checkpoint_exists:
            return "no-checkpoint"
        if not self.guidance_ok:
            return "missing-sidecar"
        if self.n_candidates == 0:
            return "training"
        if not self.docking_wired:
            return "worker-not-wired"  # trained + present, but this worker cannot dock
        return "ready"


def load_manifest(path: Path = MANIFEST_CSV) -> List[Cell]:
    cells: List[Cell] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            cells.append(
                Cell(
                    generator=r["generator"].strip(),
                    target_name=r["target"].strip(),
                    seed=int(r["seed"]),
                    config=r["config"].strip(),
                    checkpoint=r["checkpoint"].strip(),
                    guidance_sidecar=r.get("guidance_sidecar", "").strip(),
                )
            )
    return cells


def select(
    cells: Optional[List[Cell]] = None,
    *,
    generators: Optional[List[str]] = None,
    targets: Optional[List[str]] = None,
    stage: Optional[str] = None,  # "active" | "deferred"
    reward_type: Optional[str] = None,  # "surrogate" | "docking"
    only_ready: bool = False,
) -> List[Cell]:
    """Filter the manifest. All criteria AND together; None = no constraint."""
    cells = cells if cells is not None else load_manifest()
    out = []
    for c in cells:
        if generators and c.generator not in generators:
            continue
        if targets and c.target_name not in targets:
            continue
        if stage and c.run_stage != stage:
            continue
        if reward_type and c.reward_type != reward_type:
            continue
        if only_ready and not c.ready:
            continue
        out.append(c)
    return out


def get_cell(generator: str, target: str, cells: Optional[List[Cell]] = None) -> Cell:
    hits = select(cells, generators=[generator], targets=[target])
    if not hits:
        raise KeyError(f"No manifest cell for generator={generator!r} target={target!r}")
    return hits[0]


def _emit_shell(cell: Cell) -> str:
    """Shell-sourceable ``KEY=VALUE`` view of a cell — the bridge for ``submit_cell.sh``
    (``eval "$(python manifest.py --emit <gen> <target>)"``). Keeps the manifest the single
    source of truth; bash never re-derives paths."""
    t = cell.target
    kv = {
        "CELL_TAG": cell.tag,
        "GENERATOR": cell.generator,
        "TARGET": cell.target_name,
        "REWARD_NAME": cell.target.reward_name,
        "SEED": cell.seed,
        "CONFIG": cell.config,
        "CHECKPOINT": cell.checkpoint,
        "GUIDANCE": cell.guidance_sidecar,
        "HIGHER_IS_BETTER": "true" if t.higher_is_better else "false",
        "MODE_REWARD_THRESHOLD": t.mode_reward_threshold,
        "REWARD_TYPE": cell.reward_type,
        "RUN_STAGE": cell.run_stage,
        "CONDA_ENV": cell.conda_env,
        "WORKER": cell.worker,
        "SAMPLE_DIR": str(cell.sample_dir),
        "ENUM_DIR": str(cell.enum_dir),
        "RESULTS_DIR": str(cell.results_dir),
        "STATUS": cell.status(),
    }
    import shlex

    return "\n".join(f"{k}={shlex.quote(str(v))}" for k, v in kv.items())


def _print_table() -> None:
    cells = load_manifest()
    hdr = (
        f"{'cell':<16}{'stage':<10}{'reward_type':<12}{'gate':>7}  {'dir':>4}  {'#cand':>6}  status"
    )
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        t = c.target
        gate = f"{'>' if t.higher_is_better else '<'}{t.mode_reward_threshold:g}"
        print(
            f"{c.tag:<16}{c.run_stage:<10}{c.reward_type:<12}{gate:>7}  "
            f"{'ok' if c.checkpoint_exists else 'NO':>4}  {c.n_candidates:>6}  {c.status()}"
        )
    ready = [c.tag for c in cells if c.ready]
    active_ready = [c.tag for c in select(cells, stage="active", only_ready=True)]
    print(f"\nready ({len(ready)}/16): {', '.join(ready)}")
    print(f"active + ready ({len(active_ready)}/8 surrogate): {', '.join(active_ready)}")


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="LSD-Flow 16-cell matrix manifest")
    ap.add_argument(
        "--emit",
        nargs=2,
        metavar=("GENERATOR", "TARGET"),
        help="print a shell-sourceable KEY=VALUE spec for one cell (for submit_cell.sh)",
    )
    ap.add_argument(
        "--list",
        choices=["all", "active", "deferred", "active-ready"],
        help="print 'generator<TAB>target' per matching cell (for launch scripts)",
    )
    a = ap.parse_args()
    if a.emit:
        print(_emit_shell(get_cell(a.emit[0], a.emit[1])))
    elif a.list:
        stage = None if a.list in ("all", "active-ready") else a.list
        only_ready = a.list == "active-ready"
        if a.list == "active-ready":
            stage = "active"
        for c in select(stage=stage, only_ready=only_ready):
            print(f"{c.generator}\t{c.target_name}")
    else:
        _print_table()


if __name__ == "__main__":
    _main()
