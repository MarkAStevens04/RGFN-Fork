"""``PhaseTimer`` — per-phase wall-clock timing for the active-learning loop.

Motivation (experiment 009): the first end-to-end run left us unable to say where
the loop spent its time. Only rough per-step rates were eyeballed during the
session, and the run was killed by ``SIGXCPU`` *at* the oracle step — so the most
expensive phase has no number at all. This module closes that gap so every run
records, per round, how long each of the four phases took:

    ``fit_proxy``    — refit the MPNN proxy M on D_{i-1}
    ``train_gfn``    — train pi_theta against r(x) = M(x)^beta
    ``sample_batch`` — sample the query batch B ~ pi_theta
    ``oracle_score`` — label B with the expensive oracle O

Design — one ``with`` per phase, three outputs on exit:
    1. **Print** (``[AL]`` prefix, ``flush=True``) so the timing shows up live in
       the job log, matching how the loop already narrates itself.
    2. **Log** to the trainer logger under the ``timing`` prefix, so durations land
       in wandb alongside the round metrics.
    3. **Append** to ``<run_dir>/phase_timings.csv``. The append happens the moment
       a phase finishes, *before* the next phase can run — so a mid-run crash (the
       SIGXCPU that ended experiment 009) still leaves a complete record of every
       phase that did finish, which is exactly when we most want to know where the
       time went.

The final :meth:`report_total` ranks phases by share of total wall-clock — the
"where can we save the most" answer the timing exists to give.
"""

import csv
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


def _fmt(seconds: float) -> str:
    """Human-readable duration: ``45.2s``, ``3m 05s``, or ``1h 02m 03s``."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class PhaseTimer:
    """Times the named phases of each active-learning round, reporting as it goes.

    Wrap each phase in :meth:`phase`; on exit the elapsed wall-clock is printed,
    logged, and appended to CSV (see module docstring). :meth:`report_round`
    prints the round's breakdown; :meth:`report_total` prints the cross-round
    summary ranked by share of total time.
    """

    # Canonical ordering for the per-round breakdown line.
    PHASES = ("fit_proxy", "train_gfn", "sample_batch", "oracle_score")

    def __init__(self, logger=None, csv_path: Optional[Path] = None):
        """
        Args:
            logger: a trainer ``LoggerBase`` (or ``None`` to skip metric logging);
                durations are logged under the ``timing`` prefix.
            csv_path: where to append the per-phase log; ``None`` disables the CSV.
        """
        self.logger = logger
        self.csv_path = Path(csv_path) if csv_path else None
        # phase -> cumulative seconds across all rounds (for the final breakdown).
        self._totals: Dict[str, float] = {}
        # phase -> seconds within the current round (cleared by report_round).
        self._current: Dict[str, float] = {}
        if self.csv_path is not None and not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as fh:
                csv.writer(fh).writerow(["round", "phase", "seconds"])

    @contextmanager
    def phase(self, name: str, rnd: int):
        """Time a named phase of round ``rnd``; records even if the body raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self._record(name, rnd, time.perf_counter() - start)

    # ----------------------------------------------------------------- internals
    def _record(self, name: str, rnd: int, elapsed: float) -> None:
        self._totals[name] = self._totals.get(name, 0.0) + elapsed
        self._current[name] = elapsed
        print(f"[AL] round {rnd}: {name} took {_fmt(elapsed)}", flush=True)
        if self.logger is not None:
            self.logger.log_metrics(
                metrics={f"{name}_seconds": elapsed, "al_round": rnd},
                prefix="timing",
            )
        if self.csv_path is not None:
            with open(self.csv_path, "a", newline="") as fh:
                csv.writer(fh).writerow([rnd, name, f"{elapsed:.3f}"])

    # ------------------------------------------------------------------- reports
    def report_round(self, rnd: int) -> None:
        """Print this round's per-phase breakdown, then reset for the next round."""
        total = sum(self._current.values())
        parts = "  ".join(
            f"{p}={_fmt(self._current[p])}" for p in self.PHASES if p in self._current
        )
        print(f"[AL] round {rnd} timing: {parts}  (total {_fmt(total)})", flush=True)
        self._current = {}

    def report_total(self) -> None:
        """Print the cross-round breakdown ranked by share of total wall-clock."""
        grand = sum(self._totals.values())
        if grand <= 0:
            return
        print("[AL] timing summary (phase / total / share):", flush=True)
        for phase, secs in sorted(self._totals.items(), key=lambda kv: -kv[1]):
            print(f"[AL]   {phase:<13} {_fmt(secs):>12}  {100 * secs / grand:5.1f}%", flush=True)
        print(f"[AL]   {'TOTAL':<13} {_fmt(grand):>12}  100.0%", flush=True)

    def total_for(self, name: str) -> float:
        """Cumulative seconds recorded for a phase across all rounds (0 if never timed)."""
        return self._totals.get(name, 0.0)


class DockAccountant:
    """Accumulates wall-clock spent calling the expensive docking oracle during training.

    ``PhaseTimer`` times *phases* (the whole ``train_gfn`` pass); this times the docking that
    happens *inside* that pass — every step, the reward generator docks the step's molecules.
    A docking reward generator (:class:`glue.proxies.OracleRewardProxy`, or a baseline's
    cross-env bridge) wraps each oracle call in :meth:`time`. The fixed-reward pipeline then
    divides ``dock_seconds`` by the ``train_gfn`` phase wall-clock to report the **docking
    share** — the number that decides whether extra random seeds are ~free (a mostly-idle
    docker whose spare capacity a second seed can use) or costly (campaign Logs/030). It is
    the per-run companion to the persistent docking server's own busy/idle utilization log.

    Deliberately dependency-free (only ``time``/``contextlib``) so the baselines, which run in
    their own conda envs and cannot import ``glue``, can carry an identical copy.
    """

    def __init__(self) -> None:
        self.dock_seconds = 0.0
        self.n_calls = 0  # number of oracle-score batches (~= training steps that docked)
        self.n_mols = 0  # molecules actually docked (cache misses)

    def record(self, seconds: float, n_mols: int) -> None:
        self.dock_seconds += float(seconds)
        self.n_calls += 1
        self.n_mols += int(n_mols)

    @contextmanager
    def time(self, n_mols: int):
        """Time a docking call over ``n_mols`` molecules; records even if the body raises."""
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(time.perf_counter() - start, n_mols)

    def summary(self, train_seconds: Optional[float] = None) -> Dict[str, float]:
        """Totals, plus the docking share of ``train_seconds`` when provided."""
        out: Dict[str, float] = {
            "dock_seconds": round(self.dock_seconds, 3),
            "n_dock_calls": self.n_calls,
            "n_mols_docked": self.n_mols,
            "s_per_mol": round(self.dock_seconds / self.n_mols, 4) if self.n_mols else 0.0,
        }
        if train_seconds is not None and train_seconds > 0:
            out["train_gfn_seconds"] = round(float(train_seconds), 3)
            out["docking_share"] = round(self.dock_seconds / float(train_seconds), 4)
        return out
