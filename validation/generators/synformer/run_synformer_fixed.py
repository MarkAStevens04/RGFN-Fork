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

MINIMUM = 1e-10  # upstream's make_mating_pool constant


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
    except ValueError:
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
            self._pool = WorkerPool(
                gpu_ids=list(range(n_gpus)),
                num_workers_per_gpu=int(sf_c.get("num_workers_per_gpu", 2)),
                task_qsize=0,
                result_qsize=0,
                model_path=model_path,
                state_pool_opt=self._opt,
                time_limit=self._time_limit,
            )
            print(
                f"[SF-FR] projector: upstream WorkerPool, {self._nw} worker(s) on {n_gpus} GPU(s), "
                f"forked in {_time.time()-t0:.1f}s (BEFORE any torch model exists in this process)",
                flush=True,
            )
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
        for smi in smiles_list:
            self._pool.submit(Molecule(smi))
        # A TIMEOUT, unlike upstream's blocking fetch. A worker that dies or deadlocks must surface
        # as an error in minutes, not consume the job's entire walltime in silence.
        budget = self._time_limit * self._max_evolve + 300
        for i in range(len(smiles_list)):
            try:
                _, df = self._pool.fetch(block=True, timeout=budget)
            except _q.Empty:
                alive = [(w.is_alive(), w.exitcode) for w in self._pool._workers]
                raise SystemExit(
                    f"[SF-FR] worker pool produced nothing for {budget}s after {i}/"
                    f"{len(smiles_list)} results; workers (alive, exitcode) = {alive}. "
                    "If they are alive with exitcode None this is the fork-after-torch deadlock — "
                    "see the Projector docstring."
                )
            if len(df):
                frames.append(df)
            if (i + 1) % 20 == 0 or (i + 1) == len(smiles_list):
                el = _time.time() - t0
                print(
                    f"[SF-FR]   projected {i+1}/{len(smiles_list)} in {el:.0f}s "
                    f"({el/(i+1):.1f}s/molecule)",
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

    def score(smiles_list):
        """Score, charging the budget only for molecules never seen before."""
        todo = [s for s in dict.fromkeys(smiles_list) if s not in scored]
        room = budget - len(scored)
        if room <= 0:
            todo = []
        elif len(todo) > room:
            todo = todo[:room]
        if todo:
            for s, v in zip(todo, provider.predict(todo)):
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
    )
    print(
        f"[SF-FR] reward provider ready ({reward_c.get('type')}) — built AFTER the fork", flush=True
    )
    try:
        t0 = time.time()
        projected = projector(starting_population)
        routes.update(projected)
        population_smiles = list(projected)
        if not population_smiles:
            raise SystemExit(
                "[SF-FR] the initial projection returned nothing — check the checkpoint/data paths."
            )
        population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
        population_scores = score([Chem.MolToSmiles(m) for m in population_mol])

        gen = 0
        while len(scored) < budget:
            gen += 1
            mating_pool = make_mating_pool(population_mol, population_scores, pop_size)
            offspring_mol = [reproduce(mating_pool, mut_rate) for _ in range(off_size)]

            population_mol = sanitize(population_mol + offspring_mol)
            projected = projector([Chem.MolToSmiles(m) for m in population_mol])
            if not projected:
                print(f"[SF-FR] gen {gen}: projection returned nothing; stopping early", flush=True)
                break
            routes.update(projected)
            population_mol = [Chem.MolFromSmiles(s) for s in projected]

            population_scores = score([Chem.MolToSmiles(m) for m in population_mol])
            ranked = sorted(
                zip(population_scores, population_mol), key=lambda t: t[0], reverse=True
            )[:pop_size]
            population_mol = [t[1] for t in ranked]
            population_scores = [t[0] for t in ranked]
            print(
                f"[SF-FR] gen {gen}: {len(scored)}/{budget} scored | routed={len(routes)} | "
                f"best={max(population_scores):.3f} mean={float(np.mean(population_scores)):.3f}",
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
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        w.writerows(pool)

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
    print(f"[SF-FR] wrote {len(pool)} routes -> {routes_path}", flush=True)

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
