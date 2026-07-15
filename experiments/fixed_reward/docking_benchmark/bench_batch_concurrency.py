#!/usr/bin/env python
"""Batch-size sweep + 1-vs-2-concurrent-process benchmark for QuickVina2-GPU docking.

Answers two throughput-tuning questions for the A100 docking oracle (Logs/036),
for the two systems that dock live in the 16-cell run (entry 030):

  Part A -- throughput vs QuickVina2-GPU batch size.
    How does seconds/molecule change as we hand MORE SMILES to one QuickVina2-GPU
    process?  Bigger batches amortise the per-process fixed cost (OpenCL context +
    receptor-grid load) over more ligands.  The batch lever differs by oracle:
      * ClpP  (DockingClpPOracle, single-target): score() chunks the batch by the
        oracle's ``docking_batch_size`` -> we sweep that attribute.
      * 6TD3  (Docking6TD3GpuOracle, differential): score_detailed() docks the
        WHOLE call in one QuickVina2-GPU process (ignores docking_batch_size)
        -> the *call size* IS the batch, so we chunk the pool into calls of size b.
    The SAME molecule pool is docked at every batch size, so only the chunking
    (number of QuickVina2-GPU processes = ceil(N/b)) differs.

  Part B -- GPU headroom: 1 vs 2 concurrent QuickVina2-GPU processes on one A100.
    Prepare a fixed ligand set ONCE (production-identical Meeko pdbqt), then dock
    it (a) as one QuickVina2-GPU process over the whole set, and (b) split across
    two processes launched concurrently on the same GPU (CUDA_VISIBLE_DEVICES=0).
    If a single process already saturates the card, wall-clock is ~unchanged and
    aggregate throughput is flat; if there is headroom, two processes finish the
    same total work faster.  Run at the raw-binary level (reusing the oracle's
    receptor/box/thread/opencl settings) so only the GPU search step is measured,
    not the shared CPU prep / gnina rescoring.  GPU utilisation is sampled with
    nvidia-smi throughout both parts.

Run (rgfn env, dedicated GPU node -- see submit_batch_concurrency.sh):
    python experiments/fixed_reward/docking_benchmark/bench_batch_concurrency.py \
        --out-dir "$SCRATCH/rgfn_runs/docking_benchmark/<jobid>" \
        --n 200 --batch-sizes 25 50 100 200 --concurrency-n 96 --repeats 2
"""

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[3]


# ------------------------------------------------------------------- SMILES io
def read_smiles(path: Path, n: int) -> List[str]:
    """Read up to ``n`` SMILES from a CSV with a ``smiles`` column (or 1-per-line)."""
    lines = path.read_text().splitlines()
    out: List[str] = []
    if lines and "smiles" in lines[0].lower() and "," in lines[0]:
        with open(path, newline="") as fh:
            reader = csv.DictReader(fh)
            col = "smiles" if "smiles" in reader.fieldnames else reader.fieldnames[0]
            for row in reader:
                s = (row.get(col) or "").strip()
                if s:
                    out.append(s)
                if len(out) >= n:
                    break
    else:
        for ln in lines:
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                out.append(ln.split()[0])
            if len(out) >= n:
                break
    return out


def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


# ------------------------------------------------------------- GPU util sampler
class GpuSampler:
    """Poll ``nvidia-smi`` for GPU utilisation + used memory in a background thread.

    The job has exactly one visible GPU (``--gpus-per-node=1``), so the first
    query row is our card.  Samples are collected between :meth:`start`/:meth:`stop`.
    """

    def __init__(self, interval: float = 0.25):
        self.interval = interval
        self._util: List[float] = []
        self._mem: List[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                first = out.stdout.strip().splitlines()[0]
                u, m = first.split(",")
                self._util.append(float(u))
                self._mem.append(float(m))
            except (ValueError, IndexError, subprocess.SubprocessError):
                pass
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._util, self._mem = [], []
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return self.summary()

    def summary(self) -> Dict:
        if not self._util:
            return {"util_mean": None, "util_max": None, "mem_max_mib": None, "n_samples": 0}
        return {
            "util_mean": round(statistics.mean(self._util), 1),
            "util_median": round(statistics.median(self._util), 1),
            "util_max": round(max(self._util), 1),
            "mem_max_mib": round(max(self._mem), 1),
            "n_samples": len(self._util),
        }


# --------------------------------------------------------------- oracle helpers
def _build_oracle(target: str, docking_batch_size: int):
    """Construct the live oracle for ``target`` (imports glue lazily)."""
    import glue  # noqa: F401  (registers oracles)
    from glue.oracles import Docking6TD3GpuOracle, DockingClpPOracle

    if target == "clpp":
        return DockingClpPOracle(docking_batch_size=docking_batch_size)
    if target == "6td3":
        return Docking6TD3GpuOracle(docking_batch_size=docking_batch_size)
    raise ValueError(f"unknown target {target!r}")


def _score_pool(oracle, target: str, pool: List[str], batch: int):
    """Dock ``pool`` at QuickVina2-GPU batch ``batch``; return (n_ok, n_processes).

    ClpP: mutate the oracle's docking_batch_size (score() chunks by it).
    6TD3: chunk the pool into calls of size ``batch`` (each call = one QV2 process).
    """
    n_procs = math.ceil(len(pool) / batch)
    if target == "clpp":
        oracle.docking_batch_size = batch
        scores = oracle.score(pool)
        n_ok = sum(1 for s in scores if s is not None and s == s)
    else:  # 6td3
        n_ok = 0
        for chunk in _chunks(pool, batch):
            det = oracle.score_detailed(list(chunk))
            n_ok += sum(1 for d in det if d.get("status") == "ok")
    return n_ok, n_procs


# =========================================================== Part A: batch sweep
def part_a(target: str, pool: List[str], batch_sizes: List[int], sampler: GpuSampler) -> List[Dict]:
    print(f"\n########## Part A: batch-size sweep -- {target} ##########", flush=True)
    oracle = _build_oracle(target, docking_batch_size=batch_sizes[0])

    # Warm up: build proxy + init GPU/OpenCL on a tiny batch (not timed).
    print(f"[A:{target}] warming up (proxy build + GPU/OpenCL init)...", flush=True)
    warm = pool[: min(4, len(pool))]
    if target == "clpp":
        oracle.score(warm)
    else:
        oracle.score_detailed(warm)

    rows: List[Dict] = []
    for b in batch_sizes:
        sampler.start()
        t0 = time.perf_counter()
        n_ok, n_procs = _score_pool(oracle, target, pool, b)
        dt = time.perf_counter() - t0
        util = sampler.stop()
        per_mol = dt / len(pool)
        row = {
            "target": target,
            "batch": b,
            "n_mols": len(pool),
            "n_processes": n_procs,
            "seconds": round(dt, 2),
            "s_per_mol": round(per_mol, 3),
            "mols_per_s": round(len(pool) / dt, 2),
            "n_ok": n_ok,
            "gpu": util,
        }
        rows.append(row)
        print(
            f"[A:{target}] batch={b:>3} ({n_procs} proc) {dt:7.1f}s  {per_mol:5.3f}s/mol  "
            f"{row['mols_per_s']:5.2f}mol/s  ok={n_ok}/{len(pool)}  "
            f"gpu_util~{util['util_mean']}%(max{util['util_max']}) mem{util['mem_max_mib']}MiB",
            flush=True,
        )
    return rows


# ==================================================== Part B: concurrency (raw QV2)
def _prepare_ligands(oracle, target: str, smiles: List[str], prep_dir: Path) -> List[Path]:
    """Meeko-prepare ``smiles`` to persistent pdbqt files (production-identical prep).

    Reuses the built oracle's proxy preparator so the ligands match what the oracle
    itself would dock.  Returns the pdbqt paths that were successfully written.
    """
    prep_dir.mkdir(parents=True, exist_ok=True)
    proxy = oracle._build_proxy()
    vd = proxy.docking_module_gpu
    preparator = vd.preparator  # MeekoLigandPreparator (n_conformers == 1 here)
    paths = [prep_dir / f"lig_{i}.pdbqt" for i in range(len(smiles))]
    ligand_paths_by_smiles = [[str(p)] for p in paths]
    preparator(smiles, ligand_paths_by_smiles)
    return [p for p in paths if p.exists() and p.stat().st_size > 0]


def _qv2_settings(oracle, target: str) -> Dict:
    """Pull the exact production QuickVina2-GPU settings from the built proxy."""
    vd = oracle._build_proxy().docking_module_gpu
    return {
        "vina_cmd": vd.vina_cmd,
        "vina_cwd": vd.vina_cwd,
        "receptor": os.path.abspath(vd.receptor_pdbqt_file),
        "center": list(vd.center_pos),
        "size": list(vd.size),
        "extra": dict(vd.additional_vina_args),  # thread, opencl_binary_path, num_modes
    }


def _write_qv2_config(cfg_path: Path, s: Dict, ligand_dir: Path, output_dir: Path) -> None:
    lines = [
        f"receptor = {s['receptor']}",
        f"center_x = {s['center'][0]}",
        f"center_y = {s['center'][1]}",
        f"center_z = {s['center'][2]}",
        f"size_x = {s['size'][0]}",
        f"size_y = {s['size'][1]}",
        f"size_z = {s['size'][2]}",
    ]
    for k, v in s["extra"].items():
        lines.append(f"{k} = {v}")
    lines.append(f"ligand_directory = {os.path.abspath(ligand_dir)}")
    lines.append(f"output_directory = {os.path.abspath(output_dir)}")
    cfg_path.write_text("\n".join(lines) + "\n")


def _launch_qv2(s: Dict, cfg_path: Path):
    """Start ONE QuickVina2-GPU process (non-blocking Popen), GPU 0, output silenced."""
    cmd = f"CUDA_VISIBLE_DEVICES=0 {s['vina_cmd']} --config {cfg_path} > /dev/null 2>&1"
    return subprocess.Popen(cmd, shell=True, cwd=s["vina_cwd"])


def _stage_ligands(src_paths: List[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in src_paths:
        shutil.copy(p, dst_dir / p.name)


def _run_config(s: Dict, ligand_sets: List[List[Path]], work: Path, sampler: GpuSampler) -> Dict:
    """Dock ``len(ligand_sets)`` ligand groups as that many CONCURRENT QV2 processes.

    ligand_sets == [all]        -> 1 process over the whole set.
    ligand_sets == [half, half] -> 2 concurrent processes on the same GPU.
    Returns wall-clock (launch -> all finished) + output-pose count + GPU util.
    """
    # Reset a clean work area for this measurement.
    if work.exists():
        shutil.rmtree(work)
    procs_meta = []
    for j, ligs in enumerate(ligand_sets):
        ldir = work / f"p{j}" / "lig"
        odir = work / f"p{j}" / "out"
        _stage_ligands(ligs, ldir)
        odir.mkdir(parents=True, exist_ok=True)
        cfg = work / f"p{j}" / "config"
        _write_qv2_config(cfg, s, ldir, odir)
        procs_meta.append({"cfg": cfg, "odir": odir, "n_in": len(ligs)})

    sampler.start()
    t0 = time.perf_counter()
    running = [(_launch_qv2(s, m["cfg"]), m) for m in procs_meta]  # all launched ~together
    for proc, _m in running:
        proc.wait()
    dt = time.perf_counter() - t0
    util = sampler.stop()

    n_out = sum(len(list(m["odir"].glob("*.pdbqt"))) for m in procs_meta)
    n_in = sum(m["n_in"] for m in procs_meta)
    # A dual run can exceed 40 GB (each QV2 process ~20 GB at thread=8000) and OOM
    # -> a fast wall-clock with missing outputs. Only trust throughput when every
    # ligand produced an output pose.
    complete = n_out >= n_in
    return {
        "n_processes": len(ligand_sets),
        "n_ligands": n_in,
        "seconds": round(dt, 2),
        "ligands_per_s": round(n_in / dt, 2) if dt > 0 else None,
        "n_out_poses": n_out,
        "complete": complete,
        "gpu": util,
    }


def part_b(
    target: str, pool: List[str], p_n: int, repeats: int, out_dir: Path, sampler: GpuSampler
) -> Dict:
    print(f"\n########## Part B: concurrency (raw QV2) -- {target} ##########", flush=True)
    oracle = _build_oracle(target, docking_batch_size=25)
    settings = _qv2_settings(oracle, target)
    print(
        f"[B:{target}] QV2 thread={settings['extra'].get('thread')} "
        f"num_modes={settings['extra'].get('num_modes')} "
        f"box={settings['size']} receptor={Path(settings['receptor']).name}",
        flush=True,
    )

    # Prepare P ligands ONCE (production-identical Meeko pdbqt).
    prep_dir = out_dir / f"prep_{target}"
    ligs = _prepare_ligands(oracle, target, pool[:p_n], prep_dir)
    print(f"[B:{target}] prepared {len(ligs)}/{p_n} ligands -> {prep_dir}", flush=True)
    if len(ligs) < 4:
        print(f"[B:{target}] too few prepared ligands ({len(ligs)}); skipping.", flush=True)
        return {"target": target, "error": "too_few_ligands", "n_prepared": len(ligs)}

    work = out_dir / f"concwork_{target}"
    half = len(ligs) // 2
    splits = {
        "single": [ligs],  # 1 process, all ligands
        "dual": [ligs[:half], ligs[half:]],  # 2 concurrent processes
    }

    # Warm-up run (single) to guarantee the OpenCL kernel binary is cached before timing.
    print(f"[B:{target}] warm-up run (populate OpenCL kernel cache)...", flush=True)
    _run_config(settings, [ligs[: min(8, len(ligs))]], work, sampler)

    results: Dict[str, List[Dict]] = {"single": [], "dual": []}
    for mode, sets in splits.items():
        for r in range(repeats):
            res = _run_config(settings, sets, work, sampler)
            results[mode].append(res)
            u = res["gpu"]
            flag = "" if res["complete"] else "  INCOMPLETE(OOM?)"
            print(
                f"[B:{target}] {mode:>6} rep={r} {res['n_processes']}proc "
                f"{res['n_ligands']}lig {res['seconds']:7.2f}s  "
                f"{res['ligands_per_s']:5.2f}lig/s  poses={res['n_out_poses']}/{res['n_ligands']}  "
                f"gpu_util~{u['util_mean']}%(max{u['util_max']}) mem{u['mem_max_mib']}MiB{flag}",
                flush=True,
            )

    def _best(mode):  # best (max) throughput across COMPLETE repeats only
        vals = [r["ligands_per_s"] for r in results[mode] if r["ligands_per_s"] and r["complete"]]
        return max(vals) if vals else None

    single_tp, dual_tp = _best("single"), _best("dual")
    dual_complete = any(r["complete"] for r in results["dual"])
    speedup = round(dual_tp / single_tp, 2) if (single_tp and dual_tp) else None
    if not dual_complete:
        verdict = "dual OOM (mem-bound: 2x QV2 > 40 GB at thread=8000)"
    elif speedup and speedup >= 1.15:
        verdict = "headroom (2 procs faster)"
    elif speedup:
        verdict = "saturated (no gain)"
    else:
        verdict = "n/a"
    print(
        f"[B:{target}] best single={single_tp} lig/s  dual={dual_tp} lig/s  "
        f"speedup={speedup}x -> {verdict}",
        flush=True,
    )
    # tidy the (large) staged ligand copies; keep the prep dir + json.
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    return {
        "target": target,
        "settings": {k: settings[k] for k in ("center", "size", "extra", "receptor")},
        "n_ligands_prepared": len(ligs),
        "results": results,
        "best_single_lig_s": single_tp,
        "best_dual_lig_s": dual_tp,
        "dual_vs_single_speedup": speedup,
        "verdict": verdict,
    }


# ------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", nargs="+", default=["clpp", "6td3"])
    ap.add_argument("--smiles-csv", default="experiments/active_learning/6td3/seed_6td3.csv")
    ap.add_argument("--n", type=int, default=200, help="Part A pool size")
    ap.add_argument("--batch-sizes", nargs="+", type=int, default=[25, 50, 100, 200])
    ap.add_argument("--concurrency-n", type=int, default=96, help="Part B ligand count")
    ap.add_argument("--repeats", type=int, default=2, help="Part B repeats per mode")
    ap.add_argument("--mode", choices=["both", "a", "b"], default="both")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    need = max(args.n, args.concurrency_n, max(args.batch_sizes))
    pool = read_smiles(Path(args.smiles_csv), need)
    if len(pool) < max(args.batch_sizes):
        raise SystemExit(f"need >= {max(args.batch_sizes)} SMILES, got {len(pool)}")
    print(
        f"[bench] {len(pool)} SMILES from {args.smiles_csv}; targets={args.targets}; "
        f"batch_sizes={args.batch_sizes}; concurrency_n={args.concurrency_n}",
        flush=True,
    )

    sampler = GpuSampler()
    results: Dict[str, Dict] = {}
    for target in args.targets:
        entry: Dict = {}
        if args.mode in ("both", "a"):
            entry["part_a"] = part_a(target, pool[: args.n], args.batch_sizes, sampler)
        if args.mode in ("both", "b"):
            entry["part_b"] = part_b(
                target, pool, args.concurrency_n, args.repeats, out_dir, sampler
            )
        results[target] = entry
        # flush after each target so a late failure still leaves partial data.
        (out_dir / "benchmark_results.json").write_text(json.dumps(results, indent=2))

    print(f"\n[bench] wrote {out_dir / 'benchmark_results.json'}", flush=True)


if __name__ == "__main__":
    main()
