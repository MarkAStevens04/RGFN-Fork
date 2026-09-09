"""Frozen reward generators for the SynFormer entrant.

Per the repo convention this is a **per-adapter copy** (mirrors ``validation/generators/{rxnflow,
fraggfn,s3gfn,reinvent,saturn}/fixed_reward.py``) so the ``synformer`` env stays self-contained. The
classes are byte-for-byte the REINVENT and Saturn adapters' — verified by AST comparison, because the
benchmark's central fairness claim is that every entrant optimizes the *same* frozen reward, and
identical code is the cheapest way to keep that true. It is also checked numerically: the sEH MPNN
returns 0.0273 for ethanol in the ``synformer``, ``saturn``, ``reinvent4`` and ``s3gfn`` envs alike.

- :class:`SEHFrozenReward` — the Bengio-2021 sEH MPNN.
- :class:`DRD2FrozenReward` — the cached TDC DRD2 activity oracle (``oracle/drd2_current.pkl``),
  loaded from the pickle rather than through ``tdc.Oracle`` on purpose: TDC self-downloads into
  ``./oracle`` on first use, which fails on a compute node. The ``synformer`` env has no ``pytdc``
  at all for that reason.

:meth:`predict` returns the RAW oracle value — the scale the per-target mode gate uses (sEH ``> 7.0``,
DRD2 ``> 0.5``) — and is the only method the GA driver calls. ``reward()`` exists for interface parity
with the GFlowNet entrants and is unused here.
"""

import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from rdkit import Chem


class SEHFrozenReward:
    """Frozen sEH MPNN reward generator. ``predict`` returns the RAW proxy value (mode-gate scale)."""

    higher_is_better = True

    def __init__(self, device: str = "cpu", clip: float = 10.0, batch_size: int = 128):
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
        # IMPORTED LAZILY, AND THAT IS LOAD-BEARING FOR THE WHOLE ADAPTER.
        # `from gflownet.models import bengio2021flow` opens SIX /dev/nvidia* descriptors at IMPORT
        # time -- measured 2026-08-27: 0 fds after `import torch`, 6 immediately after this import,
        # while `torch.cuda.is_initialized()` stays False the entire time. A parent holding those
        # descriptors cannot fork a child that initializes CUDA: the child dies with
        # `RuntimeError: CUDA error: initialization error`. At module scope this poisoned fork for
        # EVERY target, including DRD2 (sklearn pickle) and ClpP (a socket path) which never touch
        # this module -- job 75066 forked its first pool fine, then lost every rebuilt worker.
        # Keeping it inside the one class that uses it means the DRD2 and ClpP parents stay
        # fork-clean, so upstream's per-generation pool teardown works there.
        # NOT wrapped in CUDA_VISIBLE_DEVICES / thread-limit guards any more. Three such guards were
        # tried on 2026-08-27 and each fixed its own measured symptom while the fork still failed:
        # blanking the GPU stopped the 6 descriptors, constraining threads stopped the 128-191 pool,
        # and device_count.cache_clear() stopped the stale zero. Job 75080 then tested the premise
        # directly, with SynFormer removed entirely -- fork a child before the proxy loads, and one
        # after:
        #
        #     fork BEFORE any sEH load   child -> OK  device_count=1
        #     after sEH build+predict    fds=0 threads=33
        #     fork AFTER sEH load        child -> FAIL RuntimeError: No CUDA GPUs are available
        #
        # The parent measured CLEAN on every metric the probe reports and the fork failed anyway, so
        # those metrics are necessary but not sufficient and the guards were treating symptoms. Any
        # in-parent sEH load is incompatible with forking a CUDA child; the fix has to keep this
        # module out of the parent entirely (subprocess scoring, as the docking path already does),
        # not make its footprint smaller. Removed rather than left in place, because dead guards that
        # look like a fix are how the next person loses another day.
        from gflownet.models import bengio2021flow

        self._bf = bengio2021flow
        self.model = bengio2021flow.load_original_model()
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def predict(self, smiles: List[str]) -> List[float]:
        """Raw sEH proxy value per SMILES (higher = better); ``nan`` for invalid molecules."""
        out: List[float] = []
        for start in range(0, len(smiles), self.batch_size):
            chunk = smiles[start : start + self.batch_size]
            graphs, valid = [], []
            for s in chunk:
                mol = Chem.MolFromSmiles(s) if s else None
                g = None
                if mol is not None:
                    try:
                        g = self._bf.mol2graph(mol)
                    except Exception:
                        # Atom/feature outside the sEH featurizer's set. An unconstrained SMILES
                        # generator can emit these; treat as invalid, exactly as the S3-GFN adapter
                        # and native mol2seh do. Silently scoring them 0 would instead teach the
                        # policy that exotic elements are merely bad rather than unscoreable.
                        g = None
                valid.append(g is not None)
                if g is not None:
                    graphs.append(g)
            preds: List[float] = []
            if graphs:
                batch = self._bf.mols2batch(graphs).to(self.device)
                preds = self.model(batch).view(-1).cpu().numpy().tolist()
            it = iter(preds)
            for ok in valid:
                out.append(float(next(it)) if ok else float("nan"))
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Interface parity with the GFlowNet entrants; UNUSED on the REINVENT path (see module doc)."""
        return [
            float(np.exp(-self.clip))
            if v != v
            else float(np.exp(np.clip(v, -self.clip, self.clip)))
            for v in self.predict(smiles)
        ]

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, device: str) -> None:
        self.device = device
        self.model.to(device)


class DRD2FrozenReward:
    """Frozen DRD2 activity oracle. Reproduces ``tdc.Oracle("DRD2")`` bit-for-bit (cached sklearn
    model + TDC's count-Morgan/FCFP6 featurization), matching the reaction-GFNs' ``DRD2Proxy``.

    Loaded from the cached pickle rather than through ``tdc.Oracle`` on purpose: TDC self-downloads
    into ``./oracle`` on first use, which fails on a compute node ($HOME read-only, no internet)."""

    higher_is_better = True

    def __init__(self, model_path: str, clip: float = 10.0, **_ignored):
        import pickle

        with open(model_path, "rb") as fh:
            self.model = pickle.load(fh)  # nosec - trusted local TDC oracle
        self.clip = float(clip)

    @staticmethod
    def _fp(mol):
        from rdkit.Chem import AllChem

        f = AllChem.GetMorganFingerprint(mol, 3, useCounts=True, useFeatures=True)
        nfp = np.zeros((1, 2048), np.int32)
        for idx, v in f.GetNonzeroElements().items():
            nfp[0, idx % 2048] += int(v)
        return nfp

    def predict(self, smiles: List[str]) -> List[float]:
        """DRD2 activity probability in [0, 1] per SMILES; ``nan`` for invalid molecules."""
        mols = [Chem.MolFromSmiles(s) if s else None for s in smiles]
        valid_idx = [i for i, m in enumerate(mols) if m is not None]
        out = [float("nan")] * len(smiles)
        if valid_idx:
            X = np.concatenate([self._fp(mols[i]) for i in valid_idx], axis=0)
            probs = self.model.predict_proba(X)[:, 1]
            for j, i in enumerate(valid_idx):
                out[i] = float(probs[j])
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Interface parity with the GFlowNet entrants; UNUSED on the REINVENT path (see module doc)."""
        return [
            float(np.exp(-self.clip))
            if v != v
            else float(np.exp(np.clip(v, -self.clip, self.clip)))
            for v in self.predict(smiles)
        ]

    def fit(self, *args, **kwargs) -> dict:
        return {}

    def set_device(self, *args, **kwargs) -> None:
        pass


def _load_dock_server_client(repo_root):
    """Path-import the stdlib ``DockingServerClient`` from ``glue/oracles/docking_server.py`` and
    return ``client_from_env()`` (client if ``RGFN_DOCK_SOCKET`` set, else ``None``). Imports the file
    directly (not the ``glue`` package the s3gfn env can't import); the stdlib-only client half is safe
    here."""
    import importlib.util

    p = Path(repo_root) / "glue" / "oracles" / "docking_server.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("_rgfn_dockserver", str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.client_from_env()


class SEHBridgeReward:
    """The frozen sEH proxy, scored IN A CHILD PROCESS so the parent never loads the MPNN.

    Interface-identical to :class:`SEHFrozenReward` -- same ``predict`` semantics (raw proxy value,
    ``nan`` for unscoreable), same ``reward``, same ``higher_is_better`` -- so the run driver cannot
    tell them apart. Only the process in which the model lives differs.

    WHY. SynFormer forks workers every generation, and job 75080 (entry [074]) established that
    loading the sEH model in the COORDINATING process makes every later spawn fail with
    ``No CUDA GPUs are available``, with the parent measuring clean on both descriptor and thread
    counts. Descriptor/thread hygiene is necessary but not sufficient; the model simply must not be
    in the parent. That left the three sEH cells blocked because no out-of-process entry point for
    the surrogate existed. ``score_seh_subprocess.py`` is that entry point and this is its client.

    UNLIKE :class:`DockingBridgeReward` this does NOT cross an env boundary -- the synformer env can
    import ``bengio2021flow`` (the rgfn env cannot), so the child runs in the same env and only the
    PROCESS differs. Nothing here needs conda.

    Caches per canonical SMILES: the GA re-scores the same survivors every generation, and a cache
    hit costs nothing while a miss costs a model load.
    """

    higher_is_better = True

    def __init__(
        self,
        repo_root: str,
        device: str = "cpu",
        clip: float = 10.0,
        batch_size: int = 128,
        workdir: Optional[str] = None,
        timeout_s: int = 1800,
    ):
        self.repo_root = Path(repo_root)
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
        self.timeout_s = int(timeout_s)
        self.workdir = Path(workdir) if workdir else (self.repo_root / "reward_bridge")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, float] = {}
        self._step = 0
        print(
            "[seh-bridge] sEH proxy scored out-of-process (parent never loads the MPNN)", flush=True
        )

    def predict(self, smiles: List[str]) -> List[float]:
        """Raw sEH proxy value per SMILES, ``nan`` where unscoreable. Order is preserved."""
        need = [s for s in dict.fromkeys(smiles) if s not in self._cache]
        if need:
            self._step += 1
            d = self.workdir / f"seh_step{self._step:05d}"
            d.mkdir(parents=True, exist_ok=True)
            smi_path, out_path = d / "in.smi", d / "scores.csv"
            smi_path.write_text("\n".join(need) + "\n")
            cmd = [
                sys.executable, "-m", "validation.generators.synformer.score_seh_subprocess",
                "--smiles", str(smi_path), "--out", str(out_path),
                "--device", self.device, "--clip", str(self.clip),
                "--batch-size", str(self.batch_size),
            ]  # fmt: skip
            proc = subprocess.run(cmd, cwd=str(self.repo_root), timeout=self.timeout_s)
            if proc.returncode != 0 or not out_path.exists():
                # Do NOT fall back to a default score. A silent 0 would be indistinguishable from a
                # genuinely bad molecule and would train the policy on a scoring failure.
                raise RuntimeError(
                    f"sEH scoring child failed (rc={proc.returncode}); see {d}. Refusing to "
                    "substitute a default score -- that would be trained on as if it were real."
                )
            with out_path.open() as fh:
                for row in csv.DictReader(fh):
                    self._cache[row["smiles"]] = float(row["raw_score"])
        return [self._cache.get(s, float("nan")) for s in smiles]

    def reward(self, smiles: List[str]) -> List[float]:
        return [
            float(np.exp(-self.clip))
            if v != v
            else float(np.exp(np.clip(v, -self.clip, self.clip)))
            for v in self.predict(smiles)
        ]

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, device: str) -> None:
        self.device = device


class DockingBridgeReward:
    """Per-step GPU **docking** as a fixed reward, reached across the env boundary via
    ``scripts/score_batch.py`` under the ``rgfn`` env (interface-ready for 6TD3/ClpP). Per-adapter
    twin of the RxnFlow/FragGFN ``DockingBridgeReward``; docks each step (QV2-GPU + gnina), converts
    the lower-is-better raw score to the GFN VALUE ``clip(-raw/norm, 0, inf)`` and ``reward =
    exp(clip(value))``, frees torch's GPU cache before each dock (Logs/014), and caches per canonical
    SMILES. ``raw_scores`` exposes the raw docking value for the mode gate. Docking is per-step
    expensive → docking targets train FEWER steps than the proxy targets."""

    higher_is_better = True  # the recorded VALUE (clip(-raw/norm)) is higher-is-better

    def __init__(
        self,
        oracle: str,
        repo_root: str,
        norm: float = 1.0,
        failed_score: float = 0.0,
        clip: float = 10.0,
        conda_env: str = "rgfn",
        oracle_args: Optional[Dict] = None,
        workdir: Optional[str] = None,
    ):
        self.oracle = oracle
        self.repo_root = Path(repo_root)
        self.norm = float(norm)
        self.failed_score = float(failed_score)
        self.clip = float(clip)
        self.conda_env = conda_env
        self.oracle_args = dict(oracle_args or {})
        self.workdir = Path(workdir) if workdir else (self.repo_root / "reward_bridge")
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, float] = {}  # canonical smiles -> raw docking score
        self._step = 0
        self._client = _load_dock_server_client(self.repo_root)
        if self._client is not None:
            if not self._client.wait_until_ready(timeout=900):
                raise RuntimeError(
                    "RGFN_DOCK_SOCKET is set but the docking server never became ready "
                    f"({self._client.socket_path})."
                )
            print(
                f"[dock-bridge] persistent docking server @ {self._client.socket_path}", flush=True
            )
        else:
            print(
                "[dock-bridge] no RGFN_DOCK_SOCKET -> per-step score_batch.py subprocess",
                flush=True,
            )

    def predict(self, smiles: List[str]) -> List[float]:
        """The GFN VALUE per SMILES = ``clip(-raw/norm, 0, inf)`` (higher = better)."""
        return [self._value(r) for r in self._dock(smiles)]

    def raw_scores(self, smiles: List[str]) -> List[float]:
        """Raw docking score per SMILES (Vina/dvina, lower = better; ``nan`` on failure) — the
        mode-gate scale for docking targets."""
        return self._dock(smiles)

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(clip(value))`` (β applied later → ``exp(value·β)``)."""
        return [float(np.exp(min(v, self.clip))) for v in self.predict(smiles)]

    def fit(self, *args, **kwargs) -> dict:
        return {}

    def set_device(self, *args, **kwargs) -> None:
        pass

    def _value(self, raw: float) -> float:
        if raw is None or raw != raw:
            return self.failed_score
        return max(-float(raw) / self.norm, 0.0)

    def _dock(self, smiles: List[str]) -> List[float]:
        canons = [self._canonical(s) for s in smiles]
        todo = [c for c in dict.fromkeys(canons) if c and c not in self._cache]
        if todo:
            self._step += 1
            self._free_gpu_cache()  # free this proc's cache so the docking GPU can allocate (Logs/014)
            if self._client is not None:
                lab_raw, _ = self._client.dock(todo)
                labels = [float(x) if x is not None else float("nan") for x in lab_raw]
            else:
                smi_path = self.workdir / f"step_{self._step:05d}.smi"
                lbl_path = self.workdir / f"step_{self._step:05d}_labels.csv"
                smi_path.write_text("\n".join(todo) + "\n")
                cmd = [
                    "conda", "run", "--no-capture-output", "-n", self.conda_env,
                    "python", "scripts/score_batch.py",
                    "--oracle", self.oracle, "--in", str(smi_path), "--out", str(lbl_path),
                ]  # fmt: skip
                for k, v in self.oracle_args.items():
                    cmd += ["--oracle-arg", f"{k}={v}"]
                subprocess.run(cmd, check=True, cwd=str(self.repo_root))
                labels = self._read_labels(lbl_path, todo)
            for c, lab in zip(todo, labels):
                self._cache[c] = lab
            if todo and all(lab != lab for lab in labels):
                print(
                    f"[dock-bridge] WARNING step {self._step}: all {len(todo)} docks failed (nan) -- "
                    "flat reward this step (check GPU/OpenCL).",
                    flush=True,
                )
        return [self._cache.get(c, float("nan")) if c else float("nan") for c in canons]

    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return Chem.MolToSmiles(mol) if mol is not None else None

    @staticmethod
    def _read_labels(path: Path, batch: List[str]) -> List[float]:
        by_smi: Dict[str, float] = {}
        if path.exists():
            with open(path, newline="") as fh:
                for row in csv.DictReader(fh):
                    try:
                        by_smi[row["smiles"]] = float(row["label"])
                    except (KeyError, TypeError, ValueError):
                        by_smi[row.get("smiles", "")] = float("nan")
        return [by_smi.get(s, float("nan")) for s in batch]

    @staticmethod
    def _free_gpu_cache() -> None:
        try:
            import gc

            gc.collect()
            # GATE ON is_initialized(), NEVER is_available(). Measured 2026-08-27:
            # `torch.cuda.is_available()` OPENS six /dev/nvidia* descriptors (fds 0 -> 6) while
            # is_initialized() opens none. A parent holding those descriptors cannot fork a worker
            # that initializes CUDA, so this one call -- reached every docking batch from _dock --
            # poisoned SynFormer's per-generation pool rebuild on the ClpP cells, and only on ClpP,
            # because DRD2 never docks. Caught by _cuda_probe printing
            # "nvidia_fds=6 <-- FORK IS POISONED" on job 75089 before any worker died.
            #
            # The intent (Logs/014: free this process's VRAM so the docking subprocess can allocate)
            # is preserved exactly where it applies. If CUDA was never initialized here there is no
            # cache to free, so the call was doing nothing but harm -- and asking whether a GPU is
            # available is what created the harm.
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
        except Exception:
            pass


def build_provider(
    reward_type: str,
    device: str = "cpu",
    model_path: Optional[str] = None,
    clip: float = 10.0,
    batch_size: int = 128,
    oracle: Optional[str] = None,
    repo_root: Optional[str] = None,
    norm: float = 1.0,
    failed_score: float = 0.0,
    oracle_args: Optional[Dict] = None,
    workdir: Optional[str] = None,
    subprocess_scoring: bool = False,
):
    """One place that maps a config ``reward.type`` onto a provider, shared by the run driver and
    the scoring component so the two can never disagree about what 'seh_proxy' means."""
    if reward_type == "drd2":
        return DRD2FrozenReward(model_path=model_path or "oracle/drd2_current.pkl", clip=clip)
    if reward_type == "seh_proxy":
        # IN-PROCESS BY DEFAULT, out-of-process on request. SynFormer is the ONE entrant that
        # cannot load this model in its coordinating process (entry [074], job 75080), so its
        # config sets `subprocess: true`; every other caller keeps the cheaper in-process path.
        if subprocess_scoring:
            return SEHBridgeReward(
                repo_root=str(repo_root or Path(__file__).resolve().parents[3]),
                device=device,
                clip=clip,
                batch_size=batch_size,
                workdir=str(workdir) if workdir else None,
            )
        return SEHFrozenReward(device=device, clip=clip, batch_size=batch_size)
    if reward_type == "docking":
        # GPU docking reached ACROSS THE ENV BOUNDARY -- this entrant's env has no docking stack, so
        # the work goes to scripts/score_batch.py, or to the persistent docking server when
        # RGFN_DOCK_SOCKET is set. predict() returns the higher-is-better VALUE clip(-raw/norm,0,inf);
        # raw_scores() returns raw Vina kcal/mol, which is what the ClpP mode gate (-8.0, Logs/045)
        # is applied to. Identical class and semantics to the S3-GFN/FragGFN/RxnFlow entrants.
        return DockingBridgeReward(
            oracle=oracle or "docking_clpp",
            repo_root=str(repo_root or Path(__file__).resolve().parents[3]),
            norm=norm,
            failed_score=failed_score,
            clip=clip,
            oracle_args=dict(oracle_args or {}),
            workdir=str(workdir) if workdir else None,
        )
    raise SystemExit(
        f"unknown reward type {reward_type!r} for the REINVENT entrant; "
        "expected 'seh_proxy', 'drd2' or 'docking'."
    )
