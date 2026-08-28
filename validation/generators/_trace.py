"""One uniform per-molecule trace per run, written by every generator adapter.

WHY THIS EXISTS. Audited 2026-08-20, the five entrants each recorded something different and two
recorded nothing usable: S3-GFN kept only a 1,000-row final eval sample (its whole training history
discarded), and SynFormer's history lived in SLURM stdout mixed with worker chatter at ~36 h/cell to
regenerate. Neither could answer "how many modes had been found by oracle call N" without retraining.
Saturn and REINVENT *could*, but only through two bespoke parsers over two unrelated formats.

WHAT IT BUYS. The headline is the RGFN paper's **modes vs normalized iterations** curve: with
``n_scored`` and ``smiles`` on every row, the number of distinct modes discovered by any prefix of a
run is a retrospective query, for every entrant, with one parser. ``elapsed_s`` answers "where did the
time go" on the same rows. Neither needs the run repeated.

WHAT IT DELIBERATELY DOES NOT STORE. Routes. Per-step SPARROW pricing was considered and dropped: for
the route-less entrants it would mean a MultiAiZ run per checkpoint (~3-6 h each), and the questions
we actually want answered -- modes, diversity, reward over time -- need only *which molecules existed
when*. SynFormer and TANGO still emit routes at the end of a run (``routes.jsonl`` /
``route_0.pkl``), and our own generators keep full routes in ``paths.csv``; this file is orthogonal
to all of that.

COLUMNS

``n_scored``    cumulative count of scoring events, counting repeats. This is the honest denominator
                for "oracle calls" when a generator re-scores a molecule it has seen.
``n_distinct``  cumulative count of DISTINCT molecules scored. This is what Saturn's own
                ``oracle_calls`` counts (it runs ``allow_oracle_repeats: false``), so it is the column
                to use when comparing against a generator's declared budget.
``phase``       ``train`` or ``eval``. LOAD-BEARING: some generators score molecules outside the
                training loop (S3-GFN's ``evaluate()`` scores a 1,000-molecule sample), and counting
                those as oracle calls both inflates the budget and puts molecules on the
                modes-vs-calls curve that the policy never learned from. Measured on a 20-step smoke:
                3,280 scored, of which only ~1,280 were training. Filter to ``phase == "train"`` for
                any budget or learning-curve claim.
``step``        training step, where the generator exposes one; blank otherwise.
``smiles``      as the generator emitted it -- NOT canonicalised here, so the trace stays a faithful
                record. Canonicalise at analysis time.
``raw_score``   the ORACLE's own value, before any shaping. For the surrogates that is the sEH MPNN
                output or the DRD2 probability (higher-is-better). **For docking it is raw Vina
                kcal/mol, LOWER-is-better** -- not the clip(-vina) value the generator trains on --
                because the mode gates are defined on the raw energy (ClpP -8.0, Logs/045). Providers
                that transform their oracle expose ``raw_scores()``; the writers prefer it.
                Consequence: this column's sign convention depends on the target, so read the run's
                config (or manifest.json's score_units) before comparing across cells.
``elapsed_s``   seconds since the adapter started the run.

Keeping both counters matters: the gap between them is itself a mode-collapse signal (REINVENT scored
127,997 rows for 124,587 distinct; Saturn 10,020 for 10,020).
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

FIELDS = ["n_scored", "n_distinct", "phase", "step", "smiles", "raw_score", "elapsed_s"]


class TraceWriter:
    """Append-only trace. Flushes every row: a walltime kill must not cost the history.

    Usage::

        tr = TraceWriter(run_dir / "trace.csv")
        ...
        tr.add_many(smiles_list, scores, step=step)
        ...
        tr.close()
    """

    def __init__(self, path: Path | str, t0: Optional[float] = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time() if t0 is None else t0
        self._seen: set[str] = set()
        self.n_scored = 0
        self._fh = open(self.path, "w", newline="")
        self._w = csv.writer(self._fh)
        self._w.writerow(FIELDS)
        self._fh.flush()

    def add(
        self, smiles: str, raw_score: float, step: Optional[int] = None, phase: str = "train"
    ) -> None:
        self.n_scored += 1
        self._seen.add(smiles)
        self._w.writerow(
            [
                self.n_scored,
                len(self._seen),
                phase,
                "" if step is None else int(step),
                smiles,
                "" if raw_score is None else float(raw_score),
                round(time.time() - self.t0, 3),
            ]
        )

    def add_many(
        self,
        smiles: Sequence[str],
        scores: Sequence[float],
        step: Optional[int] = None,
        phase: str = "train",
    ) -> None:
        if len(smiles) != len(scores):
            raise ValueError(f"smiles/scores length mismatch: {len(smiles)} vs {len(scores)}")
        for s, v in zip(smiles, scores):
            self.add(s, v, step=step, phase=phase)
        # One flush per batch rather than per row: a batch is the unit a generator can lose anyway.
        self._fh.flush()

    @property
    def n_distinct(self) -> int:
        return len(self._seen)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "TraceWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def unshape_docking(norm: float = 1.0):
    """Return a callable turning a docking VALUE back into raw Vina, for the post-hoc converters.

    WHY THIS IS NEEDED. ``trace_from_saturn`` reads ``glue_surrogate_raw_values`` and
    ``trace_from_reinvent`` reads ``<name> (raw)``. Both are "raw" from the ORACLE COMPONENT's point
    of view -- which is right for the sEH/DRD2 surrogates, and wrong for docking, where the component
    returns the shaped ``clip(-vina/norm, 0, inf)`` the GFN trains on. Consequence measured
    2026-08-28: nine ClpP traces (REINVENT, Saturn and TANGO x3 seeds) carried positive 0..17 values
    with a median near 10, and **not one row cleared the -8.0 gate** -- so the whole training history
    of those cells looked empty when the ClpP gate was applied to it.

    The transform is invertible where it is not clipped: ``raw = -value * norm`` for ``value > 0``.
    ``value == 0`` is CENSORED (the clip floor), meaning raw was >= 0 or the dock failed; either way
    it cannot clear a negative gate, so it is returned as NaN rather than a fabricated 0.0. Verified
    against candidates.csv, which carries both columns: 1984/2000 rows satisfy raw == -score exactly,
    and all 16 exceptions are score == 0.

    S3-GFN and SynFormer are unaffected -- their adapters call ``provider.raw_scores()`` directly.
    """

    def _f(value):
        try:
            v = float(value)
        except (TypeError, ValueError):
            return float("nan")
        if v <= 0.0:
            return float("nan")
        return -v * float(norm)

    return _f


def write_trace_rows(
    path: Path | str,
    rows: Iterable[tuple],
    t0_elapsed: Optional[Sequence[float]] = None,
) -> int:
    """Write a trace from already-collected ``(smiles, raw_score, step)`` rows, in emission order.

    For the post-hoc converters below, where the generator's own log is the source of truth and the
    adapter did not hold the loop. ``t0_elapsed`` supplies per-row elapsed seconds when the source log
    carries timestamps; rows get a blank ``elapsed_s`` when it does not, which is honest rather than
    interpolated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(FIELDS)
        for i, (smi, score, step) in enumerate(rows):
            n += 1
            seen.add(smi)
            el = ""
            if t0_elapsed is not None and i < len(t0_elapsed) and t0_elapsed[i] is not None:
                el = round(float(t0_elapsed[i]), 3)
            w.writerow([n, len(seen), "train", "" if step is None else step, smi, score, el])
    return n


def write_timing(path: Path | str, phases: dict, total_s: Optional[float] = None) -> None:
    """Persist the phase breakdown the adapters already measure but only ever printed.

    ``phases`` is a plain ``{name: seconds}`` map -- typically ``train`` / ``sample`` / ``score``.
    Deliberately not a fixed schema: the phases differ per generator and a forced common vocabulary
    would misrepresent what was actually timed. ``total_s`` is the adapter's own wall-clock, kept
    separately so the unaccounted remainder is visible rather than hidden in a rounding gap.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"phases": {k: round(float(v), 3) for k, v in phases.items()}}
    if total_s is not None:
        body["total_s"] = round(float(total_s), 3)
        body["unaccounted_s"] = round(float(total_s) - sum(phases.values()), 3)
    path.write_text(json.dumps(body, indent=2))


# --------------------------------------------------------------------------------------------------
# Post-hoc converters.
#
# Saturn and REINVENT already record everything the trace needs, in their own formats. Converting
# after the run is strictly safer than hooking their training loops: no patched upstream, no risk of
# perturbing the run we are trying to measure, and it works on the SIX cells already trained. Only
# S3-GFN and SynFormer -- which record nothing usable -- need a live hook.
#
# ONE CAVEAT ON THE DOCKING CELLS. These two converters read the generator's own log, and what that
# log stores for a docking run is the scoring component's output -- the higher-is-better VALUE
# clip(-vina/norm, 0, inf) -- not raw Vina. (Measured: REINVENT's `docking (raw)` column reads 6.7 /
# 8.4 where Vina gave -6.7 / -8.4.) The live hooks above avoid this by calling `raw_scores()`; the
# post-hoc path cannot, because the raw number was never written down.
#
# This does NOT compromise the mode curve. With norm = 1, `value >= 8.0` is exactly `raw <= -8.0` for
# every molecule the ClpP gate could admit; the clip at 0 is lossy only for molecules with raw >= 0,
# which are far worse than any gate and can never be modes. So gate a docking trace from these two
# converters on `raw_score >= 8.0`, and read the column as a value rather than an energy. The
# authoritative raw energies for the emitted pool are in `candidates.csv`'s `raw_score`, which the
# adapters populate from `provider.raw_scores()`.
# --------------------------------------------------------------------------------------------------


def _log_elapsed_by_step(log_path: Path, pattern: str, groups: int = 1):
    """Map a counter parsed out of a log line to seconds since the log's first timestamp."""
    import re

    ts_re = re.compile(r"^(?:(\d{4}-\d{2}-\d{2})[ T])?(\d{2}):(\d{2}):(\d{2})")
    key_re = re.compile(pattern)
    out: dict[int, float] = {}
    t0 = None
    with open(log_path, errors="replace") as fh:
        for line in fh:
            m = ts_re.match(line)
            if not m:
                continue
            h, mi, sec = int(m.group(2)), int(m.group(3)), int(m.group(4))
            t = h * 3600 + mi * 60 + sec
            if t0 is None:
                t0 = t
            el = t - t0
            if el < 0:  # midnight rollover
                el += 86400
            k = key_re.search(line)
            if k:
                out[int(k.group(groups))] = float(el)
    return out


def trace_from_saturn(
    run_dir: Path | str,
    out_path: Optional[Path | str] = None,
    unshape=None,
) -> int:
    """Saturn -> trace.csv, from ``oracle_history.csv`` + timestamps in ``saturn.log``.

    ``oracle_history.csv`` carries ``oracle_calls`` (Saturn's own DISTINCT-molecule counter) and
    ``glue_surrogate_raw_values`` (the raw sEH/DRD2 score, NOT the shaped ``reward`` column -- reading
    ``reward`` instead returns the 0-1 transform and nothing clears a gate of 7.0).
    """
    run_dir = Path(run_dir)
    hist = run_dir / "oracle_history.csv"
    out_path = Path(out_path) if out_path else run_dir / "trace.csv"
    log = run_dir / "saturn.log"
    el_by_calls = _log_elapsed_by_step(log, r"Oracle calls: (\d+)/") if log.exists() else {}

    rows, elapsed = [], []
    with open(hist, newline="") as fh:
        for r in csv.DictReader(fh):
            smi = (r.get("smiles") or "").strip()
            if not smi:
                continue
            raw = r.get("glue_surrogate_raw_values") or r.get("reward")
            calls = r.get("oracle_calls")
            rows.append((smi, unshape(raw) if unshape else raw, None))
            # Saturn logs a line every batch; attribute each row the elapsed time of its batch.
            elapsed.append(el_by_calls.get(int(calls)) if calls and calls.isdigit() else None)
    return write_trace_rows(out_path, rows, elapsed)


def trace_from_reinvent(
    run_dir: Path | str,
    out_path: Optional[Path | str] = None,
    unshape=None,
) -> int:
    """REINVENT -> trace.csv, from ``staged_learning_1.csv`` + one stamped line per step in the log.

    Uses the ``<name> (raw)`` column, not the shaped score: REINVENT's ``Score`` is the aggregated,
    transformed objective and is not comparable to another entrant's raw oracle value.
    """
    run_dir = Path(run_dir)
    src = next(iter(sorted(run_dir.glob("staged_learning*.csv"))), None)
    if src is None:
        raise FileNotFoundError(f"no staged_learning*.csv in {run_dir}")
    out_path = Path(out_path) if out_path else run_dir / "trace.csv"
    log = run_dir / "staged_learning.log"
    # REINVENT's step lines are not numbered, so elapsed is indexed by order of "Score" lines.
    el_by_step: dict[int, float] = {}
    if log.exists():
        import re

        ts_re = re.compile(r"^(\d{2}):(\d{2}):(\d{2}).*<INFO> Score")
        t0, i = None, 0
        with open(log, errors="replace") as fh:
            for line in fh:
                m = ts_re.match(line)
                if not m:
                    continue
                t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                if t0 is None:
                    t0 = t
                el = t - t0
                if el < 0:
                    el += 86400
                i += 1
                el_by_step[i] = float(el)

    rows, elapsed = [], []
    with open(src, newline="") as fh:
        rdr = csv.DictReader(fh)
        raw_col = next((c for c in (rdr.fieldnames or []) if c.endswith("(raw)")), None)
        for r in rdr:
            smi = (r.get("SMILES") or "").strip()
            if not smi:
                continue
            step = r.get("step")
            step_i = int(step) if step and step.strip().isdigit() else None
            _v = r.get(raw_col) if raw_col else r.get("Score")
            rows.append((smi, unshape(_v) if unshape else _v, step_i))
            elapsed.append(el_by_step.get(step_i) if step_i is not None else None)
    return write_trace_rows(out_path, rows, elapsed)
