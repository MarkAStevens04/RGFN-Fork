#!/usr/bin/env python
"""CONTROL: run SynFormer's OWN GraphGA-SF loop and see whether it survives on Balam.

WHY THIS EXISTS. Nine production cells died after ~10 h, and we have three of our own divergences
from the authors in the frame at once: we hold the worker pool open across generations (upstream
tears it down every call), we run 2 workers/GPU (upstream 1), and we monkeypatch StatePool to emit a
route column. Debugging with all three live is how the last week went. So: establish the authors'
own configuration first, change nothing, and find out whether the base implementation is even sound
in this environment. Only then reintroduce our divergences, one at a time, each with a number.

WHAT IS UPSTREAM HERE, EXACTLY. Everything that touches the failure:
  * ``projection()`` -- imported from their experiments/graphga_sf_opt.py, unmodified. It builds a
    WorkerPool, projects the batch, and calls pool.end(). Per generation. At num_workers_per_gpu=1.
  * ``make_mating_pool`` / ``reproduce`` / ``sanitize`` -- their GA operators, imported.
  * population 100, offspring 100, mutation 0.1 -- their config dict.
Importing that module is safe: its TDC/ZINC setup lives under `if __name__ == "__main__"`.

WHAT DIFFERS, AND WHY IT CANNOT MATTER. The oracle is QED from rdkit instead of a TDC oracle, and
the starting population is their bundled ChEMBL file instead of a downloaded ZINC sample. Both are
substitutions of taste, not mechanism: neither touches the worker pool, and compute nodes have no
internet to fetch the originals. NO route patch is applied -- that is deliberately the next step,
not this one.

WHAT IT MEASURES. Per generation: wall-clock, molecules projected, and the RSS of the parent and of
every child. The child RSS series is the whole point. If upstream's teardown bounds memory, it is
flat across generations and our persistent pool is the bug. If it climbs anyway, the leak is
upstream's and the teardown cadence is not the answer.

Run (synformer env, one GPU):
    python experiments/synformer_baseline/upstream_control.py --generations 12
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CLONE = _REPO_ROOT / "external" / "synformer"
for _p in (str(_REPO_ROOT), str(_CLONE), str(_CLONE / "experiments")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _rss_gib(pid: int) -> float:
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return 0.0


def _children(pid: int) -> list[int]:
    out: list[int] = []
    try:
        for t in os.listdir(f"/proc/{pid}/task"):
            with open(f"/proc/{pid}/task/{t}/children") as fh:
                out += [int(c) for c in fh.read().split()]
    except OSError:
        pass
    return out


def _mem_line(tag: str) -> str:
    me = os.getpid()
    kids = _children(me)
    deep: list[int] = list(kids)
    for k in kids:
        deep += _children(k)
    tot = sum(_rss_gib(k) for k in deep)
    return f"[CTL-MEM] {tag}: parent {_rss_gib(me):.1f} GiB | {len(deep)} child(ren) {tot:.1f} GiB"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--population", type=int, default=100)
    ap.add_argument("--offspring", type=int, default=100)
    ap.add_argument("--mutation-rate", type=float, default=0.1)
    ap.add_argument(
        "--model-path",
        default=str(_CLONE / "data" / "trained_weights" / "sf_ed_default.ckpt"),
        help="upstream's projection() defaults to a logs/ path that does not exist in this clone",
    )
    # --- ONE DIVERGENCE PER FLAG. Default = pure upstream. Each of these reintroduces exactly one
    # of our three deviations so it can be blamed or cleared on its own, which is the thing a week
    # of debugging with all three live could not do.
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="workers per GPU. 1 = upstream's projection(). >1 calls their SAME helper with the "
        "count changed, which is our only wall-clock divergence (3.9 vs 11.4 s/mol).",
    )
    ap.add_argument(
        "--routes",
        action="store_true",
        help="apply patch_get_dataframe() so StatePool emits the route column, and ASSERT it is "
        "actually populated. Spawn passed every other check while silently emptying this column.",
    )
    ap.add_argument(
        "--torch-in-parent",
        action="store_true",
        help="build the sEH proxy in this process BEFORE projecting, reproducing the one structural "
        "difference between our harness and upstream's. Tests fork-after-torch directly.",
    )
    args = ap.parse_args()

    import joblib
    import numpy as np
    from joblib import delayed
    from rdkit import Chem
    from rdkit.Chem import QED

    # STUB `tdc` BEFORE IMPORTING THEIR MODULE. graphga_sf_opt.py does a bare `import tdc` at line 8,
    # and PyTDC is not in the synformer env. Every actual USE of it is out of our path: tdc.Oracle /
    # tdc.Evaluator are constructed inside their `Oracle` class (we never instantiate it -- we score
    # with QED), and MolGen("ZINC") sits under their `__main__`. So the dependency is import-time
    # only, and a stub satisfies it without touching the clone or perturbing an env that nine
    # production cells run in. If the stub is ever actually CALLED, that is a bug in this control and
    # it will say so loudly rather than silently scoring nothing.
    if "tdc" not in sys.modules:
        import types

        def _tdc_unavailable(*a, **k):
            raise RuntimeError(
                "the tdc stub was called -- this control is supposed to score with QED and never "
                "touch TDC. Something is using upstream's Oracle class."
            )

        _stub = types.ModuleType("tdc")
        _stub.Oracle = _tdc_unavailable
        _stub.Evaluator = _tdc_unavailable
        sys.modules["tdc"] = _stub

    # Upstream's own loop pieces, imported unmodified.
    from graphga_sf_opt import make_mating_pool, projection, reproduce, sanitize

    print(f"[CTL] upstream projection() from {sys.modules['graphga_sf_opt'].__file__}", flush=True)
    print(
        f"[CTL] model={args.model_path} | workers/gpu={args.workers} | routes={args.routes} "
        f"| torch_in_parent={args.torch_in_parent}",
        flush=True,
    )

    if args.routes:
        # Must precede every fork, exactly as the adapter's own comment insists.
        from validation.generators.synformer.route_convert import (
            ROUTE_COLUMN,
            patch_get_dataframe,
        )

        patch_get_dataframe()
        print(f"[CTL] route patch applied; expecting column {ROUTE_COLUMN!r}", flush=True)

    if args.torch_in_parent:
        # THE one structural difference between our harness and upstream's. Upstream's parent scores
        # with a TDC oracle (sklearn/rdkit); ours must hold the surrogate this benchmark is defined
        # against. If fork-after-torch is really the blocker, the very next projection hangs.
        from validation.generators.synformer.fixed_reward import build_provider

        _prov = build_provider(reward_type="seh_proxy", device="cpu")
        print(
            f"[CTL] sEH proxy built in the PARENT ({type(_prov).__name__}) — now forking",
            flush=True,
        )

    # Their projection() throws the dataframe away and returns df.smiles, so it can neither vary the
    # worker count nor show whether the route column survived. When either flag is set we call the
    # SAME helper it calls (run_parallel_sampling_return_smiles, which builds the pool and end()s it
    # exactly as before) and keep the frame. Default path stays their projection(), untouched.
    route_stats = {"rows": 0, "with_route": 0}
    if args.workers != 1 or args.routes:
        from synformer.chem.mol import Molecule as _Mol
        from synformer.sampler.analog.parallel import (
            run_parallel_sampling_return_smiles,
        )

        _route_col = None
        if args.routes:
            from validation.generators.synformer.route_convert import (
                ROUTE_COLUMN as _route_col,
            )

        def projection(smiles_list, model_path=args.model_path):  # noqa: F811
            df = run_parallel_sampling_return_smiles(
                input=[_Mol(s) for s in smiles_list],
                model_path=model_path,
                search_width=24,
                exhaustiveness=64,
                num_gpus=-1,
                num_workers_per_gpu=args.workers,
                task_qsize=0,
                result_qsize=0,
                time_limit=180,
                sort_by_scores=True,
            )
            df.drop_duplicates(subset="target", inplace=True, keep="first")
            if _route_col is not None:
                route_stats["rows"] += len(df)
                if _route_col in df.columns:
                    route_stats["with_route"] += int((df[_route_col].fillna("") != "").sum())
            return df.smiles.to_list()

    # CHDIR INTO THE CLONE, for the same reason run_synformer_fixed.py does at its line 633: the
    # checkpoint stores chem.fpindex / chem.rxn_matrix as RELATIVE paths ("data/processed/comp_2048/
    # fpindex.pkl") and each worker resolves them against cwd, so from the repo root every worker
    # dies with FileNotFoundError and the parent then waits on results that will never come. Every
    # path this script holds is absolute, so the chdir is safe.
    os.chdir(_CLONE)
    print(f"[CTL] cwd -> {os.getcwd()} (checkpoint stores relative data paths)", flush=True)

    start_file = _CLONE / "data" / "chembl_filtered_1k.txt"
    all_smiles = [l.strip() for l in open(start_file) if l.strip()]
    rng = np.random.default_rng(42)
    population_smiles = list(rng.choice(all_smiles, args.population, replace=False))
    print(f"[CTL] starting population {len(population_smiles)} from {start_file}", flush=True)

    def oracle(smis):
        vals = []
        for s in smis:
            m = Chem.MolFromSmiles(s)
            vals.append(QED.qed(m) if m is not None else 0.0)
        return vals

    pool = joblib.Parallel(n_jobs=8)

    run_t0 = time.time()
    t0 = time.time()
    population_smiles = projection(population_smiles, model_path=args.model_path)
    print(
        f"[CTL] gen 0 (initial projection): {len(population_smiles)} projected in "
        f"{time.time()-t0:.0f}s",
        flush=True,
    )
    print(_mem_line("gen 0"), flush=True)
    if not population_smiles:
        raise SystemExit("[CTL] initial projection returned nothing — check model/data paths.")

    population_mol = [Chem.MolFromSmiles(s) for s in population_smiles]
    population_mol = [m for m in population_mol if m is not None]
    population_scores = oracle([Chem.MolToSmiles(m) for m in population_mol])

    for gen in range(1, args.generations + 1):
        g0 = time.time()
        mating_pool = make_mating_pool(population_mol, population_scores, args.population)
        offspring_mol = pool(
            delayed(reproduce)(mating_pool, args.mutation_rate) for _ in range(args.offspring)
        )
        population_mol += [m for m in offspring_mol if m is not None]
        population_mol = sanitize(population_mol)

        p0 = time.time()
        projected = projection(
            [Chem.MolToSmiles(m) for m in population_mol], model_path=args.model_path
        )
        proj_s = time.time() - p0

        population_mol = [Chem.MolFromSmiles(s) for s in projected]
        population_mol = [m for m in population_mol if m is not None]
        if not population_mol:
            print(f"[CTL] gen {gen}: projection returned nothing; stopping", flush=True)
            break
        population_scores = oracle([Chem.MolToSmiles(m) for m in population_mol])

        # TRUNCATE TO population_size, which upstream does at their line 332 and this control did
        # not. Without it the population grows by ~offspring_size every generation (measured
        # 100 -> 200 -> 299), so each generation projects more than the last and cost climbs
        # quadratically -- a 16-generation run needs ~30 h rather than ~7.5. It also means the
        # memory series would have been read under a growing workload instead of upstream's steady
        # one. run_synformer_fixed.py already truncates at its line 707; the control was the
        # unfaithful one.
        ranked = sorted(zip(population_scores, population_mol), key=lambda t: t[0], reverse=True)[
            : args.population
        ]
        population_mol = [t[1] for t in ranked]
        population_scores = [t[0] for t in ranked]

        print(
            f"[CTL] gen {gen}: {len(projected)} projected | pop {len(population_mol)} | "
            f"project {proj_s:.0f}s | "
            f"gen {time.time()-g0:.0f}s | best {max(population_scores):.3f} | "
            f"elapsed {(time.time()-run_t0)/3600:.2f} h",
            flush=True,
        )
        print(_mem_line(f"gen {gen}"), flush=True)

    if args.routes:
        r, w = route_stats["rows"], route_stats["with_route"]
        pct = (100.0 * w / r) if r else 0.0
        print(f"[CTL-ROUTE] {w}/{r} projected rows carry a route ({pct:.1f}%)", flush=True)
        # THE assertion this variant exists for. Spawn projected perfectly and emptied this column,
        # and every other check passed. A run that "finished" is not evidence routes work.
        if r == 0 or pct < 50.0:
            raise SystemExit(
                f"[CTL-ROUTE] FAIL: route column populated on only {pct:.1f}% of rows. "
                "Routes are the entire reason this baseline is worth running."
            )
        print("[CTL-ROUTE] PASS", flush=True)

    print(
        f"[CTL] DONE {args.generations} generations in {(time.time()-run_t0)/3600:.2f} h",
        flush=True,
    )


if __name__ == "__main__":
    main()
