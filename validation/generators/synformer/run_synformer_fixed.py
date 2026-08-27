#!/usr/bin/env python
"""Entry point for the SynFormer **fixed-reward** run — the REACTION-AWARE, non-GFN entrant.

WHY THIS ONE IS DIFFERENT. REINVENT, Saturn and S3-GFN emit molecule strings and need a planner to
recover routes afterwards. SynFormer (`[gao2025synformer]`) generates molecules AS SYNTHETIC PATHWAYS
over reaction templates and purchasable building blocks, so every molecule arrives with a route
(``has_route=1``), just as ours do. It is therefore the cell that separates the two things the
headline otherwise conflates: is the advantage the FLOW FIELD, or merely reaction-grounding? No
ablation on our own generators can answer that.

Two consequences for the pipeline:
  * **No MultiAiZ.** Its routes already bottom out in purchasable stock, so it skips the ~2.25 h
    route-discovery stage the route-less entrants pay and prices directly through
    ``sparrow_select_frontier.py --route-source external``.
  * **No recipe expansion.** Unlike our native routes, nothing in its tree has to be built before it
    can be used — every leaf is buyable.

THE OPTIMIZER IS GraphGA-SF, upstream's own (``experiments/graphga_sf_opt.py``): a Graph GA proposes
molecules, SynFormer *projects* each into synthesizable space, and the oracle scores the projections.
The loop below is theirs, with the same population/offspring/mutation settings; only two things
change, both deliberate:

  1. **The oracle is our frozen reward**, so SynFormer optimizes exactly what every other entrant
     does. Upstream's ``Oracle`` class is not reused because it constructs ``tdc.Oracle("SA")`` and
     ``tdc.Evaluator("Diversity")`` in ``__init__`` — TDC self-downloads into ``./oracle`` on first
     use, which fails on a compute node ($HOME read-only, no internet). The three helpers we do need
     (``sanitize``, ``make_mating_pool``, ``reproduce``) are reproduced verbatim from their script,
     and ``crossover``/``mutate`` are imported from upstream unchanged.
  2. **A fixed oracle budget instead of their patience-based early stop.** Their loop halts when the
     top-100 mean stops improving, which would make the training budget depend on how easy the target
     is and stop the cells being comparable. We run to a fixed budget, like every other entrant.

Run (repo root, synformer env; candidate emission shells to the rgfn env):
    conda run -p /scratch/markymoo/conda_envs/synformer python \
        validation/generators/synformer/run_synformer_fixed.py \
        --cfg validation/configs/synformer_seh_fixed.yaml --root-dir $SCRATCH/rgfn_runs/experiments
"""

import argparse
import csv
import datetime
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLONE = _REPO_ROOT / "external" / "synformer"
# `experiments/` must be importable flat: upstream's mutate.py does `import crossover as co`.
for _p in (str(_REPO_ROOT), str(_CLONE), str(_CLONE / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Imported AFTER the sys.path bootstrap above: these adapters run as scripts, so
# `validation` is not importable until the repo root is on the path.
from validation.generators._trace import TraceWriter, write_timing

MINIMUM = 1e-10  # upstream's make_mating_pool constant


def _cuda_probe(tag: str) -> None:
    """Report whether THIS process has a CUDA context, and how it can tell.

    WHY. A parent that holds a CUDA context cannot fork a worker that initializes CUDA -- the child
    dies with `RuntimeError: CUDA error: initialization error`. Measured 2026-08-27 on a DRD2 smoke
    (job 75066, whose parent holds NO torch model): the first two pool forks succeeded, every
    rebuild after that failed, and the parent had 6 open /dev/nvidia* descriptors. So the parent
    ACQUIRES a context partway through the run and something in our own loop is doing it. The fd
    count is the reliable signal: `torch.cuda.is_initialized()` stays False for a probe that merely
    enumerated devices, while the descriptors are already open and fork is already poisoned.
    """
    import os

    n = 0
    try:
        for fd in os.listdir(f"/proc/{os.getpid()}/fd"):
            try:
                if "nvidia" in os.readlink(f"/proc/{os.getpid()}/fd/{fd}"):
                    n += 1
            except OSError:
                continue
    except OSError:
        pass
    # THREADS MATTER AS MUCH AS DESCRIPTORS, and reporting only the latter cost a whole smoke.
    # Two independent hazards poison fork, with different symptoms: open /dev/nvidia* descriptors
    # make the child DIE (`CUDA error: initialization error`), while a large parent thread pool makes
    # it HANG (2.9 GiB, futex_do_wait, forever). Loading the sEH MPNN takes the parent from 64 to
    # 128-191 threads; DRD2 forks happily at 64. So both numbers belong on the same line.
    #
    # DELIBERATELY does NOT call torch.cuda.device_count()/is_available(): those OPEN the very
    # descriptors this is measuring, so probing would create the fault it reports.
    try:
        threads = len(os.listdir(f"/proc/{os.getpid()}/task"))
    except OSError:
        threads = -1
    init = "?"
    if "torch" in sys.modules:
        try:
            init = str(sys.modules["torch"].cuda.is_initialized())
        except Exception:  # noqa: BLE001
            init = "err"
    warn = ""
    if n or threads > 70:
        warn = "  <-- FORK IS POISONED: rebuilt workers will die or hang"
    print(
        f"[SF-CUDA] {tag}: nvidia_fds={n} threads={threads} "
        f"torch_imported={'torch' in sys.modules} cuda_initialized={init}{warn}",
        flush=True,
    )


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# --- The three GA helpers, verbatim from experiments/graphga_sf_opt.py -----------------------------
# Copied rather than imported because that module does `import tdc` at module level (see docstring).


def sanitize(mol_list):
    from rdkit import Chem

    new_mol_list, smiles_set = [], set()
    for mol in mol_list:
        if mol is not None:
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles is not None and smiles not in smiles_set:
                    smiles_set.add(smiles)
                    new_mol_list.append(mol)
            except ValueError:
                print("bad smiles")
    return new_mol_list


def make_mating_pool(population_mol, population_scores, offspring_size: int):
    population_scores = [s + MINIMUM for s in population_scores]
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    return np.random.choice(population_mol, p=population_probs, size=offspring_size, replace=True)


# Module-level so the count survives across calls without threading state through the GA loop.
_reproduce_failures = [0]


def _rss_report(tag: str) -> str:
    """Parent RSS plus the sum over children, in GiB.

    Added while diagnosing an OOM: four cells were killed at ~260 GB MaxRSS, which SLURM reports at
    the JOB level as TIMEOUT while only the batch STEP says OUT_OF_MEMORY -- so the runs read as slow
    rather than as leaking. Splitting parent from children says which side is growing: the parent
    holds `scored`/`routes` (plain dicts, tens of MB at most), each worker holds its own copy of the
    ~4 GB index plus a torch model.
    """
    import os

    def _kb(pid):
        try:
            for line in open(f"/proc/{pid}/status"):
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        except OSError:
            pass
        return 0

    me = os.getpid()
    parent = _kb(me)
    kids = 0
    try:
        for t in os.listdir(f"/proc/{me}/task"):
            for c in open(f"/proc/{me}/task/{t}/children").read().split():
                kids += _kb(int(c))
                # one level deeper: the pool's workers fork their own helpers
                try:
                    for tt in os.listdir(f"/proc/{c}/task"):
                        for g in open(f"/proc/{c}/task/{tt}/children").read().split():
                            kids += _kb(int(g))
                except OSError:
                    pass
    except OSError:
        pass
    return f"[SF-MEM] {tag}: parent {parent/1048576:.1f} GiB | children {kids/1048576:.1f} GiB"


def reproduce(mating_pool, mutation_rate):
    import crossover as co
    import mutate as mu

    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    try:
        new_child = co.crossover(parent_a, parent_b)
        if new_child is not None:
            new_child = mu.mutate(new_child, mutation_rate)
        return new_child
    except Exception:  # noqa: BLE001
        # Upstream catches only ValueError here, which is not enough: their own
        # `crossover.crossover_non_ring` does `rxn.RunReactants((fa, fb))[0]`
        # (experiments/crossover.py:146) and raises IndexError whenever the reaction yields no
        # product. That killed a seed-42 run 12 minutes in, AFTER a full 100-molecule projection
        # (~12 min of GPU) had been paid for. Falling back to `parent_a` is upstream's own semantics
        # for a failed cross -- widening the catch changes which failures reach it, not what happens
        # when one does. Counted below so a systematically broken mating pool cannot hide as silence.
        _reproduce_failures[0] += 1
        return parent_a


# --- Projection: the SynFormer half of GraphGA-SF -------------------------------------------------


class Projector:
    """SynFormer projection with the model held in memory across GA generations.

    ``backend="parallel"`` (default) is upstream's own ``WorkerPool`` — the code path
    ``experiments/graphga_sf_opt.py`` runs — so the baseline gets its published implementation.
    ``backend="inprocess"`` mirrors upstream's ``run_sampling_one_cpu`` instead; algorithmically the
    same StatePool/evolve/TimeLimit loop, kept as a fallback.

    ================= THE FORK-AFTER-TORCH DEADLOCK — READ BEFORE REORDERING =================
    ``WorkerPool`` forks its workers. If the PARENT has already instantiated a torch model, the
    forked child deadlocks: it stays ``alive=True, exitcode=None`` forever, produces nothing, and
    the parent blocks in ``fetch() -> Queue.get(block=True)`` with no timeout and an empty stderr.
    That is what killed jobs 73620 and 73621 (45 and 60 min, zero projections), and it was NOT the
    4 GB index and NOT the SLURM cpuset — both were measured innocent:

        pool alone, login  : 12.4 s, 75 rows
        pool alone, SLURM  : 15.3 s, 59 rows  (affinity 32 of 128, sched_setaffinity still fine)
        sEH model in parent, THEN pool, SLURM : hung at 246 s, worker alive, exitcode None

    So **the pool must be constructed before the reward provider**, and ``main()`` is ordered that
    way deliberately. If you move ``build_provider`` earlier, this hangs again with no error.
    ==========================================================================================

    The pool is also held open across generations rather than rebuilt per call as upstream does.
    That is a pure compute saving — each rebuild re-reads the 4 GB fingerprint index per worker —
    and changes nothing about the per-molecule computation.
    """

    def __init__(self, model_path: str, sf_c: dict):
        import time as _time

        self._backend = sf_c.get("backend", "parallel")
        self._n_calls = 0
        self._time_limit = int(sf_c.get("time_limit", 180))
        self._max_evolve = int(sf_c.get("max_evolve_steps", 12))
        self._max_results = int(sf_c.get("max_results", 100))
        self._opt = {
            "factor": int(sf_c.get("search_width", 24)),
            "max_active_states": int(sf_c.get("exhaustiveness", 64)),
            "sort_by_score": True,
        }
        t0 = _time.time()
        if self._backend == "parallel":
            from synformer.sampler.analog.parallel import WorkerPool, _count_gpus

            n_gpus = int(sf_c.get("num_gpus", -1))
            n_gpus = n_gpus if n_gpus > 0 else _count_gpus()
            self._nw = int(sf_c.get("num_workers_per_gpu", 2)) * n_gpus
            # Kept so the pool can be rebuilt mid-run; see recycle().
            self._pool_kwargs = dict(
                gpu_ids=list(range(n_gpus)),
                num_workers_per_gpu=int(sf_c.get("num_workers_per_gpu", 2)),
                task_qsize=0,
                result_qsize=0,
                model_path=model_path,
                state_pool_opt=self._opt,
                time_limit=self._time_limit,
            )
            self._WorkerPool = WorkerPool
            self._pool = WorkerPool(**self._pool_kwargs)
            print(
                f"[SF-FR] projector: upstream WorkerPool, {self._nw} worker(s) on {n_gpus} GPU(s), "
                f"forked in {_time.time()-t0:.1f}s (BEFORE any torch model exists in this process)",
                flush=True,
            )
            _cuda_probe("after pool fork")
        else:
            self._load_inprocess(model_path)
            print(f"[SF-FR] projector: in-process, ready in {_time.time()-t0:.1f}s", flush=True)

    def _load_inprocess(self, model_path: str) -> None:
        import pickle

        import torch
        from omegaconf import OmegaConf as _OC
        from synformer.models.synformer import Synformer

        ckpt = torch.load(model_path, map_location="cpu")
        cfg = _OC.create(ckpt["hyper_parameters"]["config"])
        self._fpindex = pickle.load(open(cfg.chem.fpindex, "rb"))
        self._rxn_matrix = pickle.load(open(cfg.chem.rxn_matrix, "rb"))
        m = Synformer(cfg.model)
        m.load_state_dict({k[6:]: v for k, v in ckpt["state_dict"].items()})
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = m.eval().to(self._device)

    def _project_parallel(self, smiles_list):
        import queue as _q
        import time as _time

        from synformer.chem.mol import Molecule

        frames, t0 = [], _time.time()

        # SUBMIT NEEDS THE GUARD MORE THAN FETCH DOES. Upstream's `submit` is
        # `JoinableQueue.put(block=True, timeout=None)` on a BOUNDED queue, so once the workers stop
        # draining it the parent blocks in `put` forever -- before it ever reaches the guarded fetch
        # loop below. That is exactly how the 2026-08-23 ClpP cells were lost: SLURM logged 1-3
        # `oom_kill` events per step, the workers died, and each job then sat in `put` for ~60 h of
        # its 72 h walltime having written nothing since hour 10. The fetch timeout could not fire
        # because control never got there. So: verify the pool is alive, and recycle it if not.
        self._ensure_workers_alive()

        # POLL, DO NOT PARK. One blocking fetch(timeout=budget) cannot tell "this molecule is genuinely
        # taking 40 minutes" from "the worker holding it was OOM-killed and no answer is ever coming",
        # so it has to wait out the full budget before it can say anything -- and then the only honest
        # thing left to say is SystemExit, losing a ~19 h cell to one dead worker. Polling on a short
        # timeout and re-checking liveness separates the two cases in seconds, and the outstanding
        # molecules can simply be resubmitted to a fresh pool. Worth the extra bookkeeping: SLURM
        # logged 1-3 oom_kill events on EVERY ClpP job, so this is the common failure, not the rare one.
        poll_s = 30.0
        budget = self._time_limit * self._max_evolve + 300  # per-molecule patience, unchanged
        pending = {s: None for s in smiles_list}  # insertion-ordered; value unused
        for smi in smiles_list:
            self._pool.submit(Molecule(smi))

        n_done = 0
        n_lost = 0
        recycles = 0
        deadline = _time.time() + budget
        total = len(smiles_list)
        while n_done + n_lost < total:
            try:
                task, df = self._pool.fetch(block=True, timeout=poll_s)
            except _q.Empty:
                dead = [
                    (i, w.exitcode) for i, w in enumerate(self._pool._workers) if not w.is_alive()
                ]
                if dead:
                    # Recycle and resubmit whatever never came back. The tasks the dead workers held
                    # are gone from the queue with them, so without the resubmit those molecules would
                    # never be fetched and this loop would spin to the deadline.
                    if recycles >= 3:
                        print(
                            f"[SF-FR] giving up on this generation after {recycles} recycles; "
                            f"dropping {len(pending)} unprojected molecules and continuing.",
                            flush=True,
                        )
                        n_lost += len(pending)
                        pending.clear()
                        break
                    recycles += 1
                    print(
                        f"[SF-FR] {len(dead)} worker(s) died mid-generation (index, exitcode) = "
                        f"{dead}; recycling and resubmitting {len(pending)} outstanding molecule(s) "
                        f"[{n_done}/{total} already in hand].",
                        flush=True,
                    )
                    self.recycle(force=True)
                    for smi in list(pending):
                        self._pool.submit(Molecule(smi))
                    deadline = _time.time() + budget
                    continue
                if _time.time() > deadline:
                    alive = [(w.is_alive(), w.exitcode) for w in self._pool._workers]
                    raise SystemExit(
                        f"[SF-FR] worker pool produced nothing for {budget}s after {n_done}/"
                        f"{total} results; workers (alive, exitcode) = {alive}. "
                        "If they are alive with exitcode None this is the fork-after-torch "
                        "deadlock — see the Projector docstring."
                    )
                continue

            # A result landed: the pool is making progress, so the patience window restarts.
            deadline = _time.time() + budget
            pending.pop(getattr(task, "smiles", None), None)
            n_done += 1
            if len(df):
                frames.append(df)
            if n_done % 20 == 0 or n_done == total:
                el = _time.time() - t0
                print(
                    f"[SF-FR]   projected {n_done}/{total} in {el:.0f}s "
                    f"({el/n_done:.1f}s/molecule)",
                    flush=True,
                )
        if n_lost:
            print(
                f"[SF-FR]   generation lost {n_lost}/{total} molecules to worker deaths.",
                flush=True,
            )
        return frames

    def _project_inprocess(self, smiles_list):
        import time as _time

        from synformer.chem.fpindex import FingerprintOption
        from synformer.chem.mol import Molecule
        from synformer.sampler.analog.state_pool import StatePool, TimeLimit

        frames, t0 = [], _time.time()
        for i, smi in enumerate(smiles_list):
            try:
                mol = Molecule(smi)
                sampler = StatePool(
                    fpindex=self._fpindex,
                    rxn_matrix=self._rxn_matrix,
                    mol=mol,
                    model=self._model,
                    **self._opt,
                )
                tl = TimeLimit(self._time_limit)
                for _ in range(self._max_evolve):
                    sampler.evolve(gpu_lock=None, show_pbar=False, time_limit=tl)
                    sims = [
                        p.molecule.sim(mol, FingerprintOption.morgan_for_tanimoto_similarity())
                        for p in sampler.get_products()
                    ]
                    if max(sims or [-1]) == 1.0:
                        break
                df = sampler.get_dataframe()[: self._max_results]
                if len(df):
                    frames.append(df)
            except Exception as exc:  # one bad molecule must not sink a generation
                print(
                    f"[SF-FR]   projection failed for molecule {i}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            if (i + 1) % 20 == 0 or (i + 1) == len(smiles_list):
                el = _time.time() - t0
                print(
                    f"[SF-FR]   projected {i+1}/{len(smiles_list)} in {el:.0f}s "
                    f"({el/(i+1):.1f}s/molecule)",
                    flush=True,
                )
        return frames

    def __call__(self, smiles_list):
        """Project SMILES into synthesizable space; returns {product_smiles: route_steps}."""
        import pandas as pd

        from validation.generators.synformer.route_convert import ROUTE_COLUMN

        self._n_calls += 1
        frames = (
            self._project_parallel(smiles_list)
            if self._backend == "parallel"
            else self._project_inprocess(smiles_list)
        )
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset="target", keep="first")  # best projection per input

        out = {}
        for _, row in df.iterrows():
            raw = row.get(ROUTE_COLUMN, "") if ROUTE_COLUMN in df.columns else ""
            if not raw:
                # A projected molecule with no serializable route cannot be priced. Dropping it is
                # conservative: keeping it would let a molecule enter the pool as if it were free.
                continue
            try:
                out[row["smiles"]] = json.loads(raw)
            except json.JSONDecodeError:
                continue
        return out

    def _ensure_workers_alive(self) -> None:
        """Recycle the pool if any worker has died, so an OOM kill costs a fork, not the walltime.

        The OOM killer reaps a WORKER, never the parent -- the parent is the small process (0.6 GiB
        against the workers' tens of GiB). Nothing in upstream notices: the queues stay valid, the
        parent stays healthy, and the run wedges silently at the next `submit`. Checking `is_alive()`
        is the whole detection, and `recycle()` is already the repair.
        """
        if self._backend != "parallel":
            return
        dead = [(i, w.exitcode) for i, w in enumerate(self._pool._workers) if not w.is_alive()]
        if not dead:
            return
        print(
            f"[SF-FR] WARNING: {len(dead)}/{len(self._pool._workers)} projection workers are dead "
            f"(index, exitcode) = {dead} -- almost certainly OOM-killed. Recycling the pool. "
            "Results already fetched are kept; the tasks those workers held are lost.",
            flush=True,
        )
        self.recycle(force=True)

    def recycle(self, force: bool = False) -> bool:
        """Tear the worker pool down and fork a fresh one, releasing whatever it accumulated.

        THE WORKERS LEAK. Measured 2026-08-25 on a live cell: parent RSS is a flat 0.6 GiB while the
        two workers start at ~13 GiB each and climb ~0.4 GiB/min between them. Four cells were killed
        at ~260 GiB MaxRSS -- the node's entire memory -- losing 2 to 3 days of GPU each. SLURM
        reports those at the JOB level as TIMEOUT and only the batch STEP as OUT_OF_MEMORY, which is
        why they read as merely slow for days.

        The leak is inside the worker process, not in anything this file owns: `scored` and `routes`
        hold plain dicts worth tens of MB, and the worker rebuilds its StatePool per molecule. Rather
        than chase it through upstream's sampler, recycling resets the workers to their ~13 GiB
        baseline. The cost is one pool fork plus a model reload; the benefit is that a cell can reach
        its full budget instead of dying at 60% of it.

        Returns False for the in-process backend, which has no pool.
        """
        if self._backend != "parallel":
            return False
        # `force` MUST skip end(). Upstream's end() puts one sentinel per worker on the BOUNDED task
        # queue and then calls JoinableQueue.join() -- so against a pool whose workers are already
        # dead it blocks on a full queue and then waits forever for task_done() calls that will never
        # come. The graceful path is only graceful when the workers are alive to be graceful with.
        if force:
            try:
                self._pool.kill()
            except Exception:  # noqa: BLE001
                pass
        else:
            try:
                self._pool.end()
            except Exception:  # noqa: BLE001
                try:
                    self._pool.kill()
                except Exception:  # noqa: BLE001
                    pass
        import gc
        import time as _t

        gc.collect()
        _cuda_probe("before rebuilding the pool")
        t0 = _t.time()
        self._pool = self._WorkerPool(**self._pool_kwargs)
        print(f"[SF-FR] recycled worker pool in {_t.time()-t0:.1f}s", flush=True)
        return True

    def close(self):
        if self._backend == "parallel":
            try:
                self._pool.end()
            except Exception:
                try:
                    self._pool.kill()
                except Exception:
                    pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--run-dir", default=None, help="EXACT run dir (stable, no timestamp)")
    ap.add_argument("--budget", type=int, default=None, help="override the oracle budget (smoke)")
    ap.add_argument("--n-samples", type=int, default=None, help="override pool size (smoke)")
    ap.add_argument(
        "--time-limit",
        type=int,
        default=None,
        help="override the PER-MOLECULE projection ceiling (seconds). Upstream's 180 s is sized for "
        "analog projection of one specific target; in a GA a molecule that will not project quickly "
        "can simply be dropped, and 100 molecules x 180 s on one worker is 5 hours.",
    )
    ap.add_argument(
        "--population",
        type=int,
        default=None,
        help="override population AND offspring size (smoke). The budget alone does not bound a "
        "smoke's cost: the FIRST projection runs over the whole starting population before a single "
        "molecule is scored, and projection is the expensive stage.",
    )
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    run_c = OmegaConf.to_container(cfg.get("run", {}), resolve=True) or {}
    fr_c = OmegaConf.to_container(cfg.get("fixed_reward", {}), resolve=True) or {}
    reward_c = OmegaConf.to_container(cfg.get("reward", {}), resolve=True) or {}
    sf_c = OmegaConf.to_container(cfg.get("synformer", {}), resolve=True) or {}

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    if args.budget is not None:
        fr_c["budget"] = args.budget
    if args.n_samples is not None:
        fr_c["n_samples"] = args.n_samples
    if args.time_limit is not None:
        sf_c["time_limit"] = args.time_limit
    if args.population is not None:
        sf_c["population_size"] = args.population
        sf_c["offspring_size"] = args.population

    budget = int(fr_c.get("budget", 10000))
    n_samples = int(fr_c.get("n_samples", 2000))
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")
    pop_size = int(sf_c.get("population_size", 100))
    off_size = int(sf_c.get("offspring_size", 100))
    mut_rate = float(sf_c.get("mutation_rate", 0.1))

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
        run_dir = root / run_c.get("name", "fixed_reward/synformer_seh") / _timestamp()
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")

    model_path = sf_c.get("model_path") or str(
        _CLONE / "data" / "trained_weights" / "sf_ed_default.ckpt"
    )
    if not Path(model_path).exists():
        raise SystemExit(
            f"[SF-FR] checkpoint not found at {model_path} — run external/setup_synformer.sh"
        )

    random.seed(seed)
    np.random.seed(seed)

    # PATCH BEFORE ANY FORK. The sampler's workers are mp.Process under Linux's default fork start
    # method, so they inherit this; applying it later would leave the route column empty.
    from validation.generators.synformer.route_convert import patch_get_dataframe

    patch_get_dataframe()

    print(
        f"[SF-FR] run_dir={run_dir} seed={seed} budget={budget} reward={reward_c.get('type')} "
        f"system={system} pop={pop_size} off={off_size}",
        flush=True,
    )

    # --- oracle bookkeeping: one score per DISTINCT molecule, capped at the budget --------------
    scored: dict = {}  # smiles -> raw reward
    routes: dict = {}  # smiles -> steps
    run_t0 = time.time()
    # 0 disables recycling; see Projector.recycle for why the default is not 0.
    recycle_every = int(sf_c.get("recycle_workers_every_gens", 10) or 0)
    # run_dir, NOT out_dir: `out_dir = run_dir / "fixed_reward"` is not defined until the emit stage
    # far below, and the trace has to exist before the first molecule is scored.
    trace = TraceWriter(run_dir / "trace.csv")
    print(f"[SF-FR] trace -> {run_dir / 'trace.csv'}", flush=True)

    def score(smiles_list):
        """Score, charging the budget only for molecules never seen before."""
        todo = [s for s in dict.fromkeys(smiles_list) if s not in scored]
        room = budget - len(scored)
        if room <= 0:
            todo = []
        elif len(todo) > room:
            todo = todo[:room]
        if todo:
            raw = list(provider.predict(todo))
            # For docking, predict() returns clip(-vina); the TRACE wants the oracle's own
            # value, since the mode gate is defined on raw Vina. Surrogates have no
            # raw_scores() and are unaffected. No re-dock: results are cached per SMILES.
            traced = list(provider.raw_scores(todo)) if hasattr(provider, "raw_scores") else raw
            # THE ONLY DURABLE RECORD OF SYNFORMER'S SEARCH. Before this hook the history existed
            # solely in SLURM stdout, interleaved with per-worker chatter, at ~36 h/cell to
            # regenerate (audit 2026-08-20). RAW values, pre-clamp: the clamp below exists to protect
            # a probability distribution, not to describe the oracle.
            trace.add_many(todo, traced)
            for s, v in zip(todo, raw):
                # Clamp at 0. `make_mating_pool` turns scores into selection PROBABILITIES
                # (`p / sum_scores`), so a single negative value silently corrupts the whole
                # distribution — and the sEH proxy can return small negatives for poor molecules.
                # NaN (an unscoreable molecule) lands at the same floor.
                scored[s] = 0.0 if v != v else max(float(v), 0.0)
        return [scored.get(s, 0.0) for s in smiles_list]

    from rdkit import Chem

    start_file = sf_c.get("starting_population") or str(_CLONE / "data" / "chembl_filtered_1k.txt")
    with open(start_file) as fh:
        all_smiles = [ln.strip() for ln in fh if ln.strip() and ln.strip() != "SMILES"]
    print(
        f"[SF-FR] starting population drawn from {start_file} ({len(all_smiles)} molecules)",
        flush=True,
    )
    starting_population = list(np.random.choice(all_smiles, pop_size, replace=False))

    # CHDIR INTO THE CLONE. The checkpoint stores `chem.fpindex` / `chem.rxn_matrix` as RELATIVE
    # paths ("data/processed/comp_2048/...") and resolves them against cwd, so the projector can only
    # find the 4 GB index from inside the clone. Everything else this driver holds is absolute, and
    # the ingest subprocess passes cwd=_REPO_ROOT explicitly, so this is safe for the whole run.
    _cwd = os.getcwd()
    os.chdir(_CLONE)
    # ORDER IS LOAD-BEARING: fork the worker pool FIRST, while this process still has no torch model
    # in it. Building the reward provider before this point deadlocks every worker — measured, see
    # the Projector docstring. Do not reorder these two statements.
    projector = Projector(model_path, sf_c)

    from validation.generators.synformer.fixed_reward import build_provider

    provider = build_provider(
        reward_type=reward_c.get("type", "seh_proxy"),
        device=reward_c.get("device", "cpu"),
        model_path=reward_c.get("model_path") or None,
        # Docking passthrough. Inert for the surrogate rewards (build_provider ignores them
        # unless reward_type == "docking"), so one call site serves all three targets.
        oracle=reward_c.get("oracle") or None,
        repo_root=str(_REPO_ROOT),
        norm=float(reward_c.get("norm", 1.0)),
        failed_score=float(reward_c.get("failed_score", 0.0)),
        oracle_args=dict(reward_c.get("oracle_args") or {}),
        workdir=str(run_dir / "reward_bridge"),
    )
    print(
        f"[SF-FR] reward provider ready ({reward_c.get('type')}) — built AFTER the fork", flush=True
    )
    _cuda_probe("after build_provider")
    try:
        t0 = time.time()
        projected = projector(starting_population)
        _cuda_probe("after initial projection")
        routes.update(projected)
        population_smiles = list(projected)
        if not population_smiles:
            raise SystemExit(
                "[SF-FR] the initial projection returned nothing — check the checkpoint/data paths."
            )
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = score([Chem.MolToSmiles(m) for m in population_mol])

        gen = 0
        last_ckpt_bucket = 0
        while len(scored) < budget:
            gen += 1
            # Per-phase timing. Projection is only ~8 min of an observed ~60 min generation on the
            # real cells, and its s/molecule is FLAT across a run, so the remaining ~52 min is
            # somewhere else -- most likely the memory pressure documented in Projector.recycle,
            # but that is a hypothesis until this says so.
            _phase_t = {}
            _gen_t0 = time.time()
            _t = time.time()
            mating_pool = make_mating_pool(population_mol, population_scores, pop_size)
            offspring_mol = [reproduce(mating_pool, mut_rate) for _ in range(off_size)]
            _phase_t["ga_s"] = time.time() - _t

            _t = time.time()
            population_mol = sanitize(population_mol + offspring_mol)
            _phase_t["sanitize_s"] = time.time() - _t

            _t = time.time()
            projected = projector([Chem.MolToSmiles(m) for m in population_mol])
            _phase_t["project_s"] = time.time() - _t
            if not projected:
                print(f"[SF-FR] gen {gen}: projection returned nothing; stopping early", flush=True)
                break
            routes.update(projected)
            population_mol = [Chem.MolFromSmiles(s) for s in projected]

            _t = time.time()
            population_scores = score([Chem.MolToSmiles(m) for m in population_mol])
            _phase_t["score_s"] = time.time() - _t
            ranked = sorted(
                zip(population_scores, population_mol), key=lambda t: t[0], reverse=True
            )[:pop_size]
            population_mol = [t[1] for t in ranked]
            population_scores = [t[0] for t in ranked]
            print(
                f"[SF-FR] gen {gen}: {len(scored)}/{budget} scored | routed={len(routes)} | "
                f"best={max(population_scores):.3f} mean={float(np.mean(population_scores)):.3f}"
                + (
                    f" | crossover fallbacks={_reproduce_failures[0]}"
                    if _reproduce_failures[0]
                    else ""
                ),
                flush=True,
            )

            # PERIODIC CHECKPOINT = THE POPULATION, not model weights. SynFormer's transformer is
            # FROZEN (`sf_ed_default.ckpt` is loaded and never updated); the only state that evolves
            # is the GA population, so dumping it is the exact analogue of the other entrants'
            # weight checkpoints -- it is what a later run would have to be resumed from. Cadence is
            # ~1,000 SCORED molecules to match Saturn (oracle calls) and S3-GFN (16 steps x 64).
            print(_rss_report(f"gen {gen}"), flush=True)

            # Recycle before the workers can reach the node's memory ceiling. Every 10 generations
            # is well short of the ~260 GiB that killed the earlier cells, and costs one pool fork.
            if recycle_every > 0 and gen % recycle_every == 0:
                projector.recycle()

            ckpt_bucket = len(scored) // 1000
            if ckpt_bucket > last_ckpt_bucket:
                last_ckpt_bucket = ckpt_bucket
                ck_dir = run_dir / "population_checkpoints"
                ck_dir.mkdir(parents=True, exist_ok=True)
                with open(ck_dir / f"pop_{ckpt_bucket * 1000}.csv", "w", newline="") as fh:
                    w = csv.writer(fh)
                    w.writerow(["smiles", "score", "n_scored", "generation"])
                    for sc, m in zip(population_scores, population_mol):
                        w.writerow([Chem.MolToSmiles(m), sc, len(scored), gen])

            _phase_t["gen_wall_s"] = time.time() - _gen_t0
            _named = sum(v for k, v in _phase_t.items() if k != "gen_wall_s")
            print(
                "[SF-TIME] gen %d: " % gen
                + " ".join(f"{k}={v:.0f}" for k, v in _phase_t.items())
                + f" RESIDUAL={_phase_t['gen_wall_s'] - _named:.0f}s",
                flush=True,
            )

    finally:
        projector.close()
        os.chdir(_cwd)
    print(
        f"[SF-FR] optimization done in {time.time() - t0:.1f}s ({len(scored)} scored, {len(routes)} routed)",
        flush=True,
    )

    # --- Emit the pool: the top-N routed molecules by RAW reward. -------------------------------
    # Unlike the route-less entrants there is nothing to sample from — SynFormer's output IS the set
    # of projections it produced, each with its route. Taking the best `n_samples` of them is the
    # analogue of their post-training sample.
    have = [(s, scored[s]) for s in routes if s in scored]
    have.sort(key=lambda t: -t[1])
    pool = have[:n_samples]
    print(
        f"[SF-FR] pool: {len(pool)} routed+scored molecules (of {len(routes)} routed)", flush=True
    )
    if len(pool) < 0.25 * n_samples:
        raise SystemExit(
            f"[SF-FR] only {len(pool)}/{n_samples} molecules are both routed and scored — that is a "
            "broken run, not a property of the method."
        )

    out_dir = run_dir / "fixed_reward"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.csv"
    # RAW SCORE IS LOAD-BEARING FOR THE DOCKING CELLS. `scores` is the provider's higher-is-better
    # VALUE -- for docking that is clip(-vina), a positive number. The ClpP mode gate is -8.0 on the
    # RAW Vina energy (Logs/045), so without this column a docking pool cannot be gated at all.
    # None for the surrogates, where predict() already returns the raw oracle value.
    raws = provider.raw_scores([s for s, _ in pool]) if hasattr(provider, "raw_scores") else None
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"] + (["raw_score"] if raws is not None else []))
        for i, (smi, sc) in enumerate(pool):
            w.writerow([smi, sc] + ([raws[i]] if raws is not None else []))

    # routes.jsonl in the schema scripts/ingest_candidates.py joins by SMILES
    routes_path = out_dir / "routes.jsonl"
    with open(routes_path, "w") as fh:
        for smi, _ in pool:
            steps = routes[smi]
            fh.write(
                json.dumps(
                    {
                        "smiles": smi,
                        "product_smiles": smi,
                        "num_reactions": len(steps),
                        "steps": steps,
                    }
                )
                + "\n"
            )
    n_buyable = sum(1 for smi, _ in pool if len(routes[smi]) == 0)
    print(f"[SF-FR] wrote {len(pool)} routes -> {routes_path}", flush=True)

    trace.close()
    # Only a total here. SynFormer's wall-clock is dominated by PROJECTION (the transformer mapping
    # each GA proposal into synthesizable space), which happens inside the worker pool and is not
    # separately timed in this process; a train/oracle split would be a guess. The per-molecule
    # elapsed_s column in trace.csv is the finer-grained record.
    write_timing(
        run_dir / "timing.json",
        {"total_run_s": time.time() - run_t0},
        total_s=time.time() - run_t0,
    )
    print(
        f"[SF-FR] trace closed: {trace.n_scored} scored / {trace.n_distinct} distinct "
        f"-> {run_dir / 'trace.csv'}",
        flush=True,
    )
    if n_buyable:
        # A ZERO-STEP route is not a failure: SynFormer projected the molecule onto a catalogue
        # building block, so it costs ZERO reactions — which under a fixed reaction budget makes it
        # the most valuable kind of library member there is.
        #
        # TRAP FOR A LATER READER: `ingest_candidates.py` counts steps, so these rows come out of
        # candidates.csv with `has_route=0` even though the manifest says has_routes=True and the
        # route exists. Anything that filters on `has_route == 1` will silently discard SynFormer's
        # FREE molecules and overstate its cost. The pricing path is safe — `--route-source external`
        # reads routes.jsonl directly, and build_network handles a step-less route by making the
        # molecule a compound node with no producing reaction — but do not re-derive the pool from
        # the CSV's has_route column.
        print(
            f"[SF-FR]   of which {n_buyable} ({n_buyable/len(pool):.0%}) are ZERO-STEP: purchasable "
            f"outright, 0 reactions. They appear in candidates.csv as has_route=0 — see the code "
            f"comment before filtering on that column.",
            flush=True,
        )

    ingest_cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "rgfn",
        "python",
        "scripts/ingest_candidates.py",
        "--pairs",
        str(pairs_path),
        "--routes",
        str(routes_path),  # <-- has_route=1: the point of this entrant
        "--out-dir",
        str(out_dir / "candidates"),
        "--generator",
        "synformer",
        "--reward-name",
        reward_name,
        "--system",
        system,
        "--seed",
        str(seed),
        "--score-higher-is-better",
        "--score-units",
        score_units,
        "--source",
        str(run_dir),
    ]
    ingest_env = os.environ.copy()
    _ingest_ld = os.environ.get("RGFN_INGEST_LD_LIBRARY_PATH")
    if _ingest_ld:
        ingest_env["LD_LIBRARY_PATH"] = _ingest_ld
    print(f"[SF-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True, cwd=str(_REPO_ROOT), env=ingest_env)

    print(f"[SF-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
