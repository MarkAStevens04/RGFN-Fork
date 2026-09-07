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
        # NEVER CLOBBER AN EXISTING TRACE. Opening "w" truncates, and a runner is now re-invoked
        # routinely -- Stage 2 upsampling calls it with a larger --n-samples, and a resumed run calls
        # it again after a failure. On 2026-08-28 that destroyed s3gfn_seh/seed43's entire training
        # history (10,162 rows, 50 modes of free pool) the moment a Stage-2 job started: the harvest
        # reads the trace, then the runner it invokes truncates the very file being harvested. The
        # data was unrecoverable -- S3-GFN keeps only a 1,000-row final eval sample beside it.
        #
        # Rotating instead of truncating makes re-invocation safe and costs a rename. Readers that
        # want the FULL history across rounds should concatenate trace.csv with its .N siblings;
        # readers that want only this round get trace.csv unchanged.
        if self.path.exists() and self.path.stat().st_size > 0:
            n = 1
            while (rotated := self.path.with_suffix(f".csv.{n}")).exists():
                n += 1
            self.path.replace(rotated)
            print(f"[trace] preserved previous history as {rotated.name}", flush=True)
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


# --------------------------------------------------------------------------------------------
# Arm-A checkpointing + the two wrapper shapes the reaction-GFNs need
#
# WHY THIS LIVES HERE. `benchmark_v2` defines its two training budgets on the ORACLE-CALL axis,
# never the step axis (docs/RETRAIN_RUNBOOK.md sec 1): the three reaction-GFNs have three different
# per-step call counts (RGFN 100, SCENT 64, RxnFlow 64) and replay buffers make the arithmetic
# unsettleable. So "arm A" is defined as "the checkpoint taken when the trace's n_scored first
# reaches 10,000" -- which means the trace, not a step counter, has to trigger it. One helper here
# rather than three per-runner copies, because three copies of a budget rule is how the four
# hardcoded ("6td3","clpp") target lists happened.
# --------------------------------------------------------------------------------------------


def _arm_a_budget() -> int:
    """The arm-A budget, overridable ONLY for smokes via ``BENCHMARK_V2_ARM_A_CALLS``.

    A 6-iteration smoke scores ~600 molecules, so at the real 10,000 the checkpoint path would
    never execute and the smoke would prove nothing about the thing it exists to prove. The
    override is read once at import; a production run leaves it unset and gets 10,000. It is
    recorded in ``arm_a.json`` as ``budget_oracle_calls``, so a cell accidentally trained under an
    override is self-identifying rather than silently off-budget.
    """
    import os

    raw = os.environ.get("BENCHMARK_V2_ARM_A_CALLS")
    if not raw:
        return 10_000
    try:
        val = int(raw)
    except ValueError:
        print(
            f"[trace] WARNING BENCHMARK_V2_ARM_A_CALLS={raw!r} is not an int; using 10000",
            flush=True,
        )
        return 10_000
    print(f"[trace] arm-A budget OVERRIDDEN to {val} calls (smoke only)", flush=True)
    return val


ARM_A_ORACLE_CALLS = _arm_a_budget()


class BudgetCheckpointer:
    """Fires ``save(label)`` ONCE, at the first iteration boundary at or after ``budget`` calls.

    WHY AT A BOUNDARY AND NOT THE EXACT ROW. A GFlowNet scores a whole minibatch inside one
    training iteration, so the 10,000th oracle call lands *mid-iteration*, where no coherent
    optimizer/replay state exists to check-point. Saving at the next boundary gives a genuinely
    resumable checkpoint whose recorded ``epoch`` is real, and overshoots the budget by at most one
    batch (64-100 molecules, <= 1%). The exact crossing row stays in ``trace.csv``, so any analysis
    that wants the budget honoured to the molecule slices the trace, and only the *weights* carry
    the rounding. Recording both is the point: ``crossed_at_n_scored`` is the truth,
    ``saved_at_iteration`` is where the weights are.

    ``save`` is called as ``save(iteration_idx)`` and must be exception-safe on the caller's side;
    a failure here is logged and swallowed, because losing a run to a checkpointing bug costs more
    than losing the arm-A checkpoint (which a re-read of the trace can always re-derive by
    resuming from the nearest periodic checkpoint).
    """

    def __init__(self, trace: "TraceWriter", budget: int, save, tag: str = "trace") -> None:
        self.trace = trace
        self.budget = int(budget)
        self._save = save
        self.tag = tag
        self.fired = False
        self.crossed_at_n_scored: Optional[int] = None
        self.saved_at_iteration: Optional[int] = None

    def note_iteration(self, iteration_idx: int) -> None:
        """Call at every training-iteration boundary. Saves when the budget has been reached."""
        if self.fired or self.trace.n_scored < self.budget:
            return
        self.fired = True
        self.crossed_at_n_scored = self.trace.n_scored
        self.saved_at_iteration = int(iteration_idx)
        try:
            self._save(int(iteration_idx))
            print(
                f"[{self.tag}] ARM A: checkpointed at iteration {iteration_idx} "
                f"(n_scored={self.trace.n_scored} >= {self.budget}, "
                f"n_distinct={self.trace.n_distinct})",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - never kill a training run over a checkpoint
            self.fired = False  # let the next boundary retry
            print(f"[{self.tag}] WARNING arm-A checkpoint failed ({exc}); will retry", flush=True)

    def manifest(self) -> dict:
        return {
            "budget_oracle_calls": self.budget,
            "fired": self.fired,
            "crossed_at_n_scored": self.crossed_at_n_scored,
            "saved_at_iteration": self.saved_at_iteration,
        }


class TracedReward:
    """Wrap a *shaped-reward* generator (``reward``/``predict``/``raw_scores``) so every evaluation
    lands in the trace. The RxnFlow / FragGFN shape.

    RECORDS THE RAW ORACLE VALUE, never the shaped one. For docking, ``reward()`` is
    ``exp(clip(-vina))`` and ``predict()`` is ``clip(-vina)``, while the mode gates are defined on
    raw Vina (ClpP -9.1). Getting this wrong is not hypothetical: nine ClpP traces recorded the
    shaped value and NOT ONE row cleared the gate, so those cells' whole training histories read as
    empty (fixed 2026-08-28, commit 736c8e0). ``raw_scores()`` is preferred wherever the provider
    exposes it, and is cached per SMILES inside the docking bridge, so tracing costs no extra docks.

    ``phase`` stays "train": these runners have no separate evaluation pass, so every call is
    training signal. (S3-GFN's ``evaluate()`` is the counter-example, which is why the column exists.)
    """

    def __init__(self, inner, trace: Optional["TraceWriter"], tag: str = "trace") -> None:
        self._inner = inner
        self._trace = trace
        self._tag = tag
        # Stamped by the caller's per-iteration hook so this shape's trace carries `step` too --
        # otherwise RxnFlow's file would be the only one of the three without it, and any per-step
        # reading would silently cover two generators out of three.
        self._step = None
        self._phase = "train"

    def set_step(self, step) -> None:
        self._step = None if step is None else int(step)

    def set_phase(self, phase: str) -> None:
        self._phase = str(phase)

    def _record(self, smiles) -> None:
        if self._trace is None or not smiles:
            return
        try:
            if hasattr(self._inner, "raw_scores"):
                vals = list(self._inner.raw_scores(smiles))
            else:
                vals = list(self._inner.predict(smiles))
            self._trace.add_many(list(smiles), vals, step=self._step, phase=self._phase)
        except Exception as exc:  # noqa: BLE001 - a trace failure must never kill a run
            print(f"[{self._tag}] WARNING: trace write failed ({exc})", flush=True)

    def reward(self, smiles):
        self._record(smiles)
        return self._inner.reward(smiles)

    def predict(self, smiles):
        return self._inner.predict(smiles)

    def __getattr__(self, name):  # set_device, fit, raw_scores, dock_accountant, ...
        return getattr(self._inner, name)


def attach_proxy_trace(
    proxy,
    trace: Optional["TraceWriter"],
    *,
    tag: str = "trace",
    budget_checkpointer: Optional["BudgetCheckpointer"] = None,
):
    """Instrument a ``ProxyBase``-shaped reward IN PLACE, by patching the instance's bound methods.

    THE RGFN / SCENT SHAPE, and why it is a patch rather than a wrapper. Both build their proxy
    through gin as ``%train_proxy`` and hand the SAME instance to the trainer's reward and to the
    pipeline. Substituting a wrapper would mean rebinding every gin reference after construction;
    patching two bound methods on the instance reaches every holder for free. Same technique SCENT's
    own ``recipe_logging.enable_recipe_logging()`` already uses on ``DynamicLibrary``.

    Patches TWO methods:

    * ``compute_proxy_output`` -- the single surface every reward evaluation crosses. Patching the
      OUTER (caching) method rather than ``_compute_proxy_output`` is deliberate: the outer one is
      called with every requested state including cache hits, which is exactly ``n_scored``
      ("oracle calls, counting repeats"), while the inner one would silently report only cache
      misses. ``n_distinct`` covers the other question.
    * ``on_end_sampling`` -- ``ProxyBase`` inherits ``TrainingHooksMixin``, so the proxy is handed
      the live ``iteration_idx`` once per training iteration. That is what lets the arm-A checkpoint
      land on a real iteration boundary without inferring a step count from batch arithmetic.

    Returns the proxy (patched in place) so callers can chain.
    """
    if trace is None:
        return proxy

    inner_compute = proxy.compute_proxy_output
    # Mutable so the two hooks below can stamp rows with the iteration that is actually running.
    # A list rather than a nonlocal because these are plain closures over a patched instance.
    current_step = [None]
    # PHASE IS LOAD-BEARING, not decoration. RGFN and SCENT score the final candidate batch through
    # the SAME compute_proxy_output the training loop uses, so without this flip those rows land in
    # the trace labelled "train" and inflate the cell's apparent oracle-call count -- measured on a
    # 6-iteration smoke: 600 training calls followed by 185 scoring calls, all indistinguishable.
    # Any "did this cell hit its 10,000-call budget" or modes-vs-calls reading would then be wrong,
    # and wrong in the direction that makes a cell look further trained than it is. Callers flip it
    # via the returned handle the moment training returns.
    current_phase = ["train"]

    def _traced_compute(states, _inner=inner_compute):
        output = _inner(states)
        try:
            smiles = [_state_smiles(s) for s in states]
            values = [float(v) for v in output.value.detach().cpu().reshape(-1).tolist()]
            keep = [(s, v) for s, v in zip(smiles, values) if s is not None]
            if keep:
                trace.add_many(
                    [s for s, _ in keep],
                    [v for _, v in keep],
                    step=current_step[0],
                    phase=current_phase[0],
                )
        except Exception as exc:  # noqa: BLE001 - a trace failure must never kill a run
            print(f"[{tag}] WARNING: trace write failed ({exc})", flush=True)
        return output

    proxy.compute_proxy_output = _traced_compute

    # ``step`` is stamped from on_START_sampling and the budget checked from on_END_sampling, and
    # the split is deliberate. Reward evaluation happens DURING sampling, so a step read at the end
    # would label every row with the previous iteration -- off by one for the whole file. The budget
    # check wants the opposite: firing at the end means the iteration is complete and its optimizer
    # / replay state is coherent enough to check-point.
    inner_start = proxy.on_start_sampling

    def _traced_on_start_sampling(iteration_idx, recursive=True, _inner=inner_start):
        current_step[0] = int(iteration_idx)
        return _inner(iteration_idx, recursive=recursive)

    proxy.on_start_sampling = _traced_on_start_sampling

    if budget_checkpointer is not None:
        inner_hook = proxy.on_end_sampling

        def _traced_on_end_sampling(iteration_idx, trajectories, recursive=True, _inner=inner_hook):
            out = _inner(iteration_idx, trajectories, recursive=recursive)
            budget_checkpointer.note_iteration(iteration_idx)
            return out

        proxy.on_end_sampling = _traced_on_end_sampling

    return _ProxyTraceHandle(proxy, current_phase)


class _ProxyTraceHandle:
    """What :func:`attach_proxy_trace` returns: the patched proxy plus the phase switch.

    The proxy itself is patched IN PLACE, so callers that only need the side effect can ignore this.
    Callers that run a scoring pass after training must call :meth:`set_phase` first.
    """

    def __init__(self, proxy, phase_holder) -> None:
        self.proxy = proxy
        self._phase = phase_holder

    def set_phase(self, phase: str) -> None:
        self._phase[0] = str(phase)


def _state_smiles(state) -> Optional[str]:
    """Best-effort SMILES for a reaction state. ``None`` -> the row is skipped, never fabricated.

    Terminal states carry ``state.molecule.smiles``; the fallbacks exist because a proxy may be
    handed a state shape this module has not seen, and dropping one untraceable row is strictly
    better than either crashing a multi-day training run or writing a ``str(obj)`` repr into a
    column that downstream code will canonicalise as a molecule.
    """
    mol = getattr(state, "molecule", None)
    smi = getattr(mol, "smiles", None) if mol is not None else None
    if isinstance(smi, str) and smi:
        return smi
    smi = getattr(state, "smiles", None)
    return smi if isinstance(smi, str) and smi else None
