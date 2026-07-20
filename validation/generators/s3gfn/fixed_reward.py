"""Fixed reward generators for the S3-GFN entrant — the CUSTOM oracle wiring.

S3-GFN (`[kim2026s3gfn]`) is the marquee non-reaction baseline for the LSD-Flow library-efficiency
benchmark. Its native ``scoring_function.get_scores`` computes sEH from ``bengio2021flow`` and docking
from ``rxnflow.tasks.unidock_vina`` — but for an apples-to-apples benchmark S3-GFN must optimize the
SAME reward the reaction-GFN entrants do. So, exactly like the RxnFlow / FragGFN entrants, this module
provides swappable **reward generators sharing one interface** (``predict`` / ``reward`` / ``fit`` /
``set_device``) that ``run_s3gfn_fixed.py`` injects in place of S3-GFN's native ``get_scores``:

- :class:`SEHFrozenReward` — the frozen Bengio-2021 sEH MPNN. Our benchmark's sEH proxy IS this exact
  model (``rgfn/gfns/reaction_gfn/proxies/seh_proxy.py`` loads the same ``bengio2021flow`` weights),
  and so is S3-GFN's own sEH path — so this is the **verification** oracle: wiring it in should
  reproduce S3-GFN's published sEH result, proving the injection is faithful.
- :class:`DRD2FrozenReward` — the TDC DRD2 activity oracle (interface-ready; the reaction-GFNs' third
  proxy).
- :class:`DockingBridgeReward` — per-step GPU **docking** reached across the env boundary via
  ``scripts/score_batch.py`` / the persistent docking server (``docking_6td3_gpu`` / ``docking_clpp``
  / …, already registered centrally). Interface-ready so 6TD3/ClpP slot in with no S3-GFN-specific
  work — just a different ``--oracle`` name and (for docking) fewer training steps.

Per the repo convention this is a **per-adapter copy** (mirrors ``validation/generators/rxnflow/
fixed_reward.py`` / ``fraggfn/fixed_reward.py``) so the ``s3gfn`` env stays self-contained. All three
providers import only torch + rdkit + numpy + (for sEH) ``gflownet.models.bengio2021flow`` — all
present in the ``s3gfn`` env. sEH is implemented + used now; DRD2/docking are the modular seam.

The reward math is bit-for-bit the RxnFlow/FragGFN twins': ``reward = exp(clip(value))`` with β
applied later by the sampler's temperature; :func:`predict` returns the RAW oracle value (the scale
our per-target mode gate uses — sEH ``> 7`` on the raw value, NOT S3-GFN's internal value/8 reward).
"""

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from gflownet.models import bengio2021flow
from rdkit import Chem


class SEHFrozenReward:
    """Frozen sEH MPNN reward generator (VERIFICATION oracle — the same bengio2021flow model our
    benchmark + S3-GFN both use). ``predict`` returns the RAW proxy value (mode-gate scale)."""

    higher_is_better = True

    def __init__(self, device: str = "cpu", clip: float = 10.0, batch_size: int = 128):
        self.device = device
        self.clip = float(clip)
        self.batch_size = int(batch_size)
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
                        g = bengio2021flow.mol2graph(mol)
                    except Exception:
                        # unsupported atom/feature (e.g. exotic elements from an unconstrained
                        # SMILES generator like S3-GFN) -> treat as invalid, exactly as native
                        # S3-GFN's mol2seh does (train.py try/except). RGFN/RxnFlow never hit this
                        # (curated building blocks), but the SMILES-GFN entrant can.
                        g = None
                valid.append(g is not None)
                if g is not None:
                    graphs.append(g)
            preds: List[float] = []
            if graphs:
                batch = bengio2021flow.mols2batch(graphs).to(self.device)
                preds = self.model(batch).view(-1).cpu().numpy().tolist()
            it = iter(preds)
            for ok in valid:
                out.append(float(next(it)) if ok else float("nan"))
        return out

    def reward(self, smiles: List[str]) -> List[float]:
        """Positive GFN reward ``exp(clip(value))`` per SMILES (β applied later → ``exp(value·β)``)."""
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:  # NaN (invalid)
                rewards.append(float(np.exp(-self.clip)))
            else:
                rewards.append(float(np.exp(np.clip(v, -self.clip, self.clip))))
        return rewards

    def fit(self, *args, **kwargs) -> dict:  # interface parity; never called (fixed reward)
        return {}

    def set_device(self, device: str) -> None:
        self.device = device
        self.model.to(device)


class DRD2FrozenReward:
    """Frozen DRD2 activity oracle (interface-ready). Reproduces ``tdc.Oracle("DRD2")`` bit-for-bit
    (cached sklearn model + TDC's count-Morgan/FCFP6 featurization). Matches the reaction-GFNs'
    ``DRD2Proxy``. Per-adapter copy (mirrors the RxnFlow/FragGFN twins)."""

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
        rewards: List[float] = []
        for v in self.predict(smiles):
            if v != v:
                rewards.append(float(np.exp(-self.clip)))
            else:
                rewards.append(float(np.exp(np.clip(v, -self.clip, self.clip))))
        return rewards

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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
