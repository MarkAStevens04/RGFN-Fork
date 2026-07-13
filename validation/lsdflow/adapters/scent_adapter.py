"""Cross-env SCENT adapter — the in-process client mirror of the scent-env worker (§4b, §10.3).

SCENT is our cost-aware, synthesizable peer (entry 017); its package is also named ``rgfn`` and
it lives in the ``scent`` conda env, so — unlike the in-process :class:`RGFNAdapter` — it cannot
co-import with this (``rgfn``-env) process. This adapter is the client half of the subprocess
bridge (the ``scripts/score_batch.py`` shape): :meth:`sample_flow_records` shells to
``validation/lsdflow/adapters/workers/scent_worker.py`` under ``conda activate scent`` (with the
scent env's torch-bundled CUDA libs on ``LD_LIBRARY_PATH`` for dgl/graphbolt, the way
``~/bin/rgfn-smoke-env.sh`` does for the rgfn env), then reads the worker's canonical
``records.csv`` + ``visit_counts.json`` + ``meta.json`` back into a
:class:`~validation.lsdflow.adapters.base.FlowSample` — identical downstream contract to RGFN.

The worker loads the trained backward policy from the ``guidance_models.pt`` sidecar beside the
checkpoint (entry 024), so the recovered ``P_B`` — and hence ``F_hat``/``U(h)`` — is SCENT's real,
cost-tilted policy (§5): the recovered flow is *cost-aware for free*, the synergy the paper sells.

**Checkpoint requirement:** only the patched re-runs (entry 024: ``2026-07-07``/``2026-07-08``
dirs) carry the sidecar; pre-fix runs are flagged ``BACKWARD_POLICY_NOT_SAVED.txt`` and their P_B
is irrecoverable — do not analyze them.

Phase-2 ``enumerate_hub_children`` (frozen dynamic library, §4b) is not built yet — the
sampled-trajectory vertical slice is the phase-1 target (§10.3).
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import subprocess
import tempfile
from glob import glob
from pathlib import Path
from typing import List, Optional

from glue.samplers.lsdflow.records import FlowRecord
from validation.lsdflow.adapters.base import FlowSample, GFNAdapter

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKER = REPO_ROOT / "validation" / "lsdflow" / "adapters" / "workers" / "scent_worker.py"

_HIGHER_IS_BETTER_BY_REWARD = {"seh": True, "drd2": True, "clpp": False, "6td3": False}


class SCENTAdapter(GFNAdapter):
    model_name = "scent"

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        *,
        reward_name: str = "seh",
        device: str = "auto",
        batch_size: int = 100,
        guidance_path: str = "",
        seed: int = 0,
        strip_stereo: bool = True,
        freeze: bool = True,
        freeze_snapshot: str = "",
        conda_env: str = "scent",
        conda_base: str = "",
        run_dir: str = "/tmp/lsdflow_scent",
        **_ignored,
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.reward_name = reward_name
        self.device = device
        self.batch_size = batch_size
        self.guidance_path = guidance_path
        self.seed = seed
        self.strip_stereo = strip_stereo
        # Freeze the dynamic library to the full trained vocabulary (§4b) — our "faithful full
        # SCENT" substrate choice; applies to both sampling and enumeration.
        self.freeze = freeze
        self.freeze_snapshot = freeze_snapshot
        self.conda_env = conda_env
        self.run_dir = run_dir
        self._tmpdir: Optional[str] = None

        self._conda_base = Path(conda_base) if conda_base else _default_conda_base()

        # Sample path overrides both from the worker's meta.json; these defaults only serve the
        # --from-records path (where visitation is unavailable anyway, so log_z is unused).
        self.higher_is_better = _HIGHER_IS_BETTER_BY_REWARD.get(reward_name, True)
        self.log_z = 0.0

    def _common_args(self) -> List[str]:
        """Worker args shared by sample + enumerate (config, checkpoint, freeze, device)."""
        args = [
            "--config",
            self.config_path,
            "--checkpoint",
            self.checkpoint_path,
            "--reward-name",
            self.reward_name,
            "--model-name",
            self.model_name,
            "--seed",
            str(self.seed),
            "--device",
            self.device,
            "--run-dir",
            self.run_dir,
        ]
        if self.guidance_path:
            args += ["--guidance", self.guidance_path]
        if not self.strip_stereo:
            args += ["--no-strip-stereo"]
        if self.freeze:
            if self.freeze_snapshot:
                args += ["--freeze-snapshot", self.freeze_snapshot]
        else:
            args += ["--no-freeze"]
        return args

    # ---------------------------------------------------------------- phase 1 sampling
    def sample_flow_records(self, n_trajectories: int) -> FlowSample:
        self._tmpdir = self._tmpdir or tempfile.mkdtemp(prefix="lsdflow_scent_")
        out_dir = Path(self._tmpdir)

        worker_args = self._common_args() + [
            "--mode",
            "sample",
            "--n-trajectories",
            str(n_trajectories),
            "--batch-size",
            str(self.batch_size),
            "--out-dir",
            str(out_dir),
        ]
        self._run_worker(worker_args)

        records = _read_records(out_dir / "records.csv")
        visit_counts = _read_json(out_dir / "visit_counts.json", default={})
        compositions = _read_json(out_dir / "compositions.json", default={})
        meta = _read_json(out_dir / "meta.json", default={})
        self.log_z = float(meta.get("log_z", 0.0))
        self.higher_is_better = bool(meta.get("higher_is_better", self.higher_is_better))
        terminal_depths = [r.hub_depth + 1 for r in records]  # x is one reaction past hub h

        print(
            f"[SCENTAdapter] worker returned {meta.get('n_trajectories', '?')} trajectories -> "
            f"{len(records)} records, {len(visit_counts)} nodes, {len(compositions)} compositions; "
            f"frozen={meta.get('frozen')} (+{meta.get('n_promoted_fragments', 0)} frags); "
            f"higher_is_better={self.higher_is_better}, logZ={self.log_z:.4f}",
            flush=True,
        )
        return FlowSample(
            records=records,
            visit_counts=visit_counts,
            n_trajectories=int(meta.get("n_trajectories", 0)),
            terminal_depths=terminal_depths,
            log_z=self.log_z,
            higher_is_better=self.higher_is_better,
            model=self.model_name,
            reward_name=self.reward_name,
            compositions=compositions,
        )

    # ---------------------------------------------------------------- phase 2 enumeration
    def enumerate_hub_children(self, hubs, *, max_children: int = 4000):
        """Enumerate each hub's one-reaction terminal children in the scent env (§4b, §6).

        ``hubs``: iterable of ``(stereo_smiles, depth)`` (the harness passes the hub's stereo-aware
        SMILES + build depth). Freezes the full trained library first so promoted-fragment children
        are enumerated too. Returns ``(records, per_hub)`` — ``records`` a list of
        :class:`FlowRecord` (mergeable into the DAG), matching :class:`RGFNAdapter`."""
        enum_dir = Path(tempfile.mkdtemp(prefix="lsdflow_scent_enum_"))
        hubs_csv = enum_dir / "hubs.csv"
        with open(hubs_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["smiles", "depth"])
            for smiles, depth in hubs:
                w.writerow([smiles, int(depth)])

        worker_args = self._common_args() + [
            "--mode",
            "enumerate",
            "--hubs-file",
            str(hubs_csv),
            "--enum-max-children",
            str(max_children),
            "--out-dir",
            str(enum_dir),
        ]
        self._run_worker(worker_args)

        records = _read_records(enum_dir / "enumerated_records.csv")
        per_hub = _read_json(enum_dir / "enum_per_hub.json", default={}).get("per_hub", [])
        import shutil

        shutil.rmtree(enum_dir, ignore_errors=True)
        print(
            f"[SCENTAdapter] enumerated {len(per_hub)} hubs -> {len(records)} records", flush=True
        )
        return records, per_hub

    def close(self) -> None:
        if self._tmpdir:
            import shutil

            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    # ---------------------------------------------------------------- subprocess plumbing
    def _run_worker(self, worker_args: List[str]) -> None:
        conda_sh = self._conda_base / "etc" / "profile.d" / "conda.sh"
        scent_prefix = self._conda_base / "envs" / self.conda_env
        nvlibs = ":".join(
            sorted(
                glob(
                    str(scent_prefix / "lib" / "python*" / "site-packages" / "nvidia" / "*" / "lib")
                )
            )
        )
        if not conda_sh.exists():
            raise RuntimeError(
                f"[SCENTAdapter] conda.sh not found at {conda_sh}. Pass conda_base=<miniconda root>."
            )

        quoted = " ".join(shlex.quote(a) for a in [str(WORKER), *worker_args])
        script = (
            "set -e\n"
            f'source "{conda_sh}"\n'
            f"conda activate {shlex.quote(self.conda_env)}\n"
            # dgl/graphbolt need torch's bundled CUDA libs on LD_LIBRARY_PATH (cluster-agnostic,
            # no `module load` — the ~/bin/rgfn-smoke-env.sh trick, applied to the scent env).
            f'export LD_LIBRARY_PATH="{nvlibs}:${{LD_LIBRARY_PATH:-}}"\n'
            "export PYTHONUNBUFFERED=1\n"
            f"exec python {quoted}\n"
        )
        print(f"[SCENTAdapter] launching scent-env worker ({self.conda_env}) ...", flush=True)
        proc = subprocess.run(["bash", "-lc", script], cwd=str(REPO_ROOT), env=os.environ.copy())
        if proc.returncode != 0:
            raise RuntimeError(
                f"[SCENTAdapter] scent worker failed (exit {proc.returncode}). "
                "Check the worker stderr above (env, checkpoint, or guidance sidecar)."
            )


def _default_conda_base() -> Path:
    exe = os.environ.get("CONDA_EXE")
    if exe:
        return Path(exe).resolve().parents[1]
    return Path.home() / "miniconda3"


def _read_records(path: Path) -> List[FlowRecord]:
    recs: List[FlowRecord] = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            recs.append(
                FlowRecord(
                    hub_key=r["hub_key"],
                    child_key=r["child_key"],
                    reward=float(r["reward"]),
                    log_reward=float(r["log_reward"]),
                    log_pf_move=float(r["log_pf_move"]),
                    log_pb_move=float(r["log_pb_move"]),
                    log_pf_stop=float(r["log_pf_stop"]),
                    hub_depth=int(r["hub_depth"]),
                    hub_stereo_key=r.get("hub_stereo_key") or r["hub_key"],
                    child_stereo_key=r.get("child_stereo_key") or r["child_key"],
                )
            )
    return recs


def _read_json(path: Path, *, default):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return default
