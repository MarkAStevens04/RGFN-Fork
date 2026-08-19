#!/usr/bin/env python
"""Entry point for the Saturn **fixed-reward** (single-shot) run — a ROUTE-LESS SMILES baseline for
the LSD-Flow library-efficiency benchmark, in the same class as S3-GFN and REINVENT 4.

Saturn (`[guo2026saturn]`) is a Mamba-based SMILES language model trained with Augmented Memory,
built for SAMPLE EFFICIENCY: few oracle calls per unit of reward. It has no reaction model, so its
molecules carry no route — emitted with ``has_route=0``, and a *library* built from them must have
its shared intermediates recovered post-hoc (MultiAiZ -> SPARROW).

Its sample efficiency is the axis worth watching. It is the strongest case for "you do not need many
oracle calls", which is precisely the axis this benchmark LOSES on and reports honestly (Logs/056's
candidates-you-must-score trade). A strong Saturn showing there is a result, not a problem.

THE PROTOCOL IS THE ONE EVERY ROUTE-LESS ENTRANT USES: train once against the frozen reward, then
sample the candidate pool from the TRAINED agent (`Generator.load_from_file` + `sample_smiles`), and
emit it in the standard candidate-dataset format. Sampling from the final agent — rather than
harvesting the molecules the run happened to score — is what keeps this pool comparable to S3-GFN's
2,000 post-training samples and REINVENT's. See the note on the oracle history below.

TWO INJECTIONS, BOTH BECAUSE SATURN HAS NO PLUGIN SEAM.
  1. ``oracles/utils.py`` imports EVERY oracle eagerly, including GEAM's, which needs openbabel.
     ``_stubs.py`` stubs it so we can skip a heavy conda dependency we never execute.
  2. ``construct_oracle_component`` is a hard-coded if/elif chain, so our component is injected by
     replacing ``oracles.oracle.construct_oracle_component`` with a wrapper that handles
     ``glue_surrogate`` and defers everything else to the original. Same technique as
     ``run_s3gfn_fixed.py``'s ``get_scores`` replacement. The clone is never edited.

BUDGET. Saturn stops on its ORACLE BUDGET (``while not oracle.budget_exceeded()``), not on a step
count — that is the whole design. Its published default is 10,000 calls, an order of magnitude below
REINVENT's ~128,000 and S3-GFN's ~320,000. All three are their authors' own defaults; the benchmark
reports the asymmetry rather than forcing a parity that would misrepresent every one of them.

Run (from repo root, saturn env; candidate emission shells to the rgfn env):
    conda run -p /scratch/markymoo/conda_envs/saturn python \
        validation/generators/saturn/run_saturn_fixed.py \
        --cfg validation/configs/saturn_seh_fixed.yaml --root-dir $SCRATCH/rgfn_runs/experiments
"""

import argparse
import csv
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CLONE = _REPO_ROOT / "external" / "saturn"
for _p in (str(_REPO_ROOT), str(_CLONE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from validation.generators.saturn._stubs import stub_unused_oracle_deps  # noqa: E402


def _timestamp() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _inject_component() -> None:
    """Route ``name == "glue_surrogate"`` to our component; defer everything else upstream."""
    import oracles.oracle as saturn_oracle

    from validation.generators.saturn.oracle_component import (
        GLUE_SURROGATE_NAME,
        GlueSurrogateOracle,
    )

    original = saturn_oracle.construct_oracle_component

    def _construct(params):
        if params.name == GLUE_SURROGATE_NAME:
            return GlueSurrogateOracle(params)
        return original(params)

    saturn_oracle.construct_oracle_component = _construct
    print(f"[SAT-FR] injected oracle component '{GLUE_SURROGATE_NAME}'", flush=True)


def _sample_pool(agent, n_target: int, batch_size: int, max_batches: int):
    """Unique, RDKit-valid, canonical SMILES from the TRAINED agent (mirrors the S3-GFN driver)."""
    from rdkit import Chem

    seen: dict = {}
    for nb in range(1, max_batches + 1):
        smiles, _ = agent.sample_smiles(num=batch_size, batch_size=batch_size)
        for s in smiles:
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is None:
                continue
            seen.setdefault(Chem.MolToSmiles(mol), None)
            if len(seen) >= n_target:
                return list(seen)[:n_target]
        if nb % 20 == 0:
            print(
                f"[SAT-FR] sampling: {len(seen)}/{n_target} unique valid (batch {nb})", flush=True
            )
    return list(seen)[:n_target]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cfg", required=True, help="YAML run config (validation/configs/saturn_*.yaml)"
    )
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--root-dir", default=None)
    ap.add_argument("--run-dir", default=None, help="EXACT run dir (stable, no timestamp)")
    ap.add_argument("--budget", type=int, default=None, help="override the oracle budget (smoke)")
    ap.add_argument("--n-samples", type=int, default=None, help="override pool size (smoke)")
    ap.add_argument("--device", default=None, help="override device (e.g. cpu for a login smoke)")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.cfg)
    run_c = OmegaConf.to_container(cfg.get("run", {}), resolve=True) or {}
    fr_c = OmegaConf.to_container(cfg.get("fixed_reward", {}), resolve=True) or {}
    reward_c = OmegaConf.to_container(cfg.get("reward", {}), resolve=True) or {}
    sat_c = OmegaConf.to_container(cfg.get("saturn", {}), resolve=True) or {}

    seed = args.seed if args.seed is not None else int(run_c.get("seed", 42))
    if args.budget is not None:
        fr_c["budget"] = args.budget
    if args.n_samples is not None:
        fr_c["n_samples"] = args.n_samples
    if args.device is not None:
        sat_c["device"] = args.device

    budget = int(fr_c.get("budget", 10000))
    n_samples = int(fr_c.get("n_samples", 2000))
    system = fr_c.get("system", "seh")
    reward_name = fr_c.get("reward_name", "seh_proxy")
    score_units = fr_c.get("score_units", f"{reward_name} (higher is better)")
    arch = sat_c.get("model_architecture", "mamba")

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        root = Path(args.root_dir or run_c.get("root_dir", "experiments"))
        run_dir = root / run_c.get("name", "fixed_reward/saturn_seh") / _timestamp()
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "run_config.yaml")
    # ...and the EFFECTIVE config. The line above writes the file as LOADED, so any CLI
    # override (--budget, --n-samples, ...) was absent from the run's own provenance —
    # which for a benchmark means a smoke and a campaign cell look identical on disk.
    OmegaConf.save(
        OmegaConf.create({"run": run_c, "fixed_reward": fr_c, "reward": reward_c, "saturn": sat_c}),
        run_dir / "run_config_effective.yaml",
    )

    prior = Path(
        sat_c.get("prior")
        or (
            _CLONE
            / "experimental_reproduction"
            / "checkpoint_models"
            / "chembl-33-mamba-epoch-18.prior"
        )
    ).resolve()
    if not prior.exists():
        raise SystemExit(
            f"[SAT-FR] prior not found at {prior} — run external/setup_saturn.sh first."
        )

    # Stub BEFORE any `oracles` import; inject AFTER the module exists but before Oracle is built.
    stubbed = stub_unused_oracle_deps()
    _inject_component()

    import torch
    from utils.utils import set_seed_everywhere

    device = sat_c.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    # Saturn seeds correctly through its own helper (unlike REINVENT, which needs a launcher).
    set_seed_everywhere(seed, device)

    print(
        f"[SAT-FR] run_dir={run_dir} device={device} seed={seed} arch={arch} "
        f"budget={budget} reward={reward_c.get('type')} system={system} stubbed={stubbed}",
        flush=True,
    )

    from beam_enumeration.dataclass import BeamEnumerationParameters
    from diversity_filter.dataclass import DiversityFilterParameters
    from experience_replay.dataclass import ExperienceReplayParameters
    from goal_directed_generation.dataclass import (
        GoalDirectedGenerationConfiguration,
        ReinforcementLearningParameters,
    )
    from goal_directed_generation.reinforcement_learning import (
        ReinforcementLearningAgent,
    )
    from hallucinated_memory.dataclass import HallucinatedMemoryParameters
    from oracles.dataclass import OracleConfiguration
    from oracles.oracle import Oracle

    # A plain DICT, not an OracleComponentParameters: `Oracle.construct_oracle` does
    # `OracleComponentParameters(**component)` on each entry, so it must be a mapping. (Saturn's own
    # entry point hands it raw JSON, which is why the dataclass in the signature is misleading.)
    component = {
        "name": "glue_surrogate",
        "weight": 1.0,
        "preliminary_check": False,
        "specific_parameters": {
            "reward_type": reward_c.get("type", "seh_proxy"),
            "model_path": reward_c.get("model_path", ""),
            "device": reward_c.get("device", "cpu"),
        },
        "reward_shaping_function_parameters": dict(
            reward_c.get("reward_shaping") or {"transformation_function": "no_transformation"}
        ),
    }
    oracle = Oracle(
        OracleConfiguration(
            components=[component],
            budget=budget,
            allow_oracle_repeats=bool(fr_c.get("allow_oracle_repeats", False)),
            aggregator=fr_c.get("aggregator", "product"),
        )
    )

    rl_c = sat_c.get("reinforcement_learning") or {}
    configuration = GoalDirectedGenerationConfiguration(
        seed=seed,
        model_architecture=arch,
        reinforcement_learning=ReinforcementLearningParameters(
            prior=str(prior), agent=str(prior), **rl_c
        ),
        experience_replay=ExperienceReplayParameters(**(sat_c.get("experience_replay") or {})),
        diversity_filter=DiversityFilterParameters(**(sat_c.get("diversity_filter") or {})),
        hallucinated_memory=HallucinatedMemoryParameters(
            **(sat_c.get("hallucinated_memory") or {"execute_hallucinated_memory": False})
        ),
        beam_enumeration=BeamEnumerationParameters(**(sat_c.get("beam_enumeration") or {})),
    )

    # Keyword args throughout: this signature gained a leading `logging_frequency` between the
    # Saturn and TANGO hashes, and positional calls would have silently bound the log path to it.
    agent_runner = ReinforcementLearningAgent(
        logging_frequency=int(sat_c.get("logging_frequency", 5000)),
        logging_path=str(run_dir / "saturn.log"),
        model_checkpoints_dir=str(run_dir / "checkpoints"),
        oracle=oracle,
        configuration=configuration,
        device=device,
    )
    # RUN WITH CWD INSIDE THE RUN DIR. Beam Enumeration writes `substructures_<n>.smi`,
    # `entire_pool.json` and `filter_history.txt` with BARE RELATIVE paths, i.e. into the process's
    # cwd — which for a SLURM job is the repo root on read-only $HOME, so it dies with
    # `PermissionError: 'substructures_304.smi'` at the very END of training, after the whole run
    # (job 73865). Same family as the Triton cache trap. Chdir'ing is safe because every path this
    # driver holds is already absolute, and it has the side benefit of collecting Saturn's own
    # substructure-pool artifacts as run provenance instead of scattering them.
    _cwd = os.getcwd()
    os.chdir(run_dir)
    t0 = time.time()
    try:
        agent_runner.run()
    finally:
        os.chdir(_cwd)
    print(
        f"[SAT-FR] training done in {time.time() - t0:.1f}s "
        f"({oracle.calls} oracle calls of a {budget} budget)",
        flush=True,
    )

    # --- Sample the candidate pool from the TRAINED agent. -------------------------------------
    # Saturn also writes its oracle HISTORY (every molecule it scored). We deliberately do not use
    # that as the pool: it is a training trace, already filtered by the run's own accept decisions,
    # and it would not be the same object the other route-less entrants contribute. Sampling the
    # final agent is the identical operation performed for S3-GFN and REINVENT.
    from models.generator import Generator

    ckpt = run_dir / "checkpoints" / f"final_{arch}_agent.ckpt"
    if not ckpt.exists():
        raise SystemExit(
            f"[SAT-FR] no final agent checkpoint at {ckpt} — training did not complete."
        )
    trained = Generator.load_from_file(str(ckpt), device=device, sampling_mode=True)

    t0 = time.time()
    pool = _sample_pool(
        trained,
        n_target=n_samples,
        batch_size=int(fr_c.get("sample_batch_size", 256)),
        max_batches=int(fr_c.get("max_sample_batches", 4000)),
    )
    print(
        f"[SAT-FR] sampled {len(pool)} unique valid candidates in {time.time() - t0:.1f}s",
        flush=True,
    )

    # A short pool is a real property of a sample-efficient, reward-maximizing model, not something
    # to paper over — but a pool a small fraction of target is a broken run. The difference matters
    # because the next stage costs ~2.25 h of route discovery.
    if len(pool) < n_samples:
        frac = len(pool) / n_samples
        msg = f"[SAT-FR] pool is {len(pool)}/{n_samples} unique valid ({frac:.0%})"
        if frac < 0.25:
            raise SystemExit(msg + " — aborting: that is a broken run, not mode collapse.")
        print(msg + " — CHECK MODE SATURATION before spending route discovery on it.", flush=True)

    # --- Re-score on the RAW value (the mode-gate scale) and emit the candidate dataset. --------
    from validation.generators.saturn.fixed_reward import build_provider

    provider = build_provider(
        reward_type=reward_c.get("type", "seh_proxy"),
        device=reward_c.get("rescore_device", "cpu"),
        model_path=reward_c.get("model_path") or None,
    )
    scores = provider.predict(pool)

    out_dir = run_dir / "fixed_reward"
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.csv"
    with open(pairs_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["smiles", "score"])
        w.writerows(zip(pool, scores))

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
        "--out-dir",
        str(out_dir / "candidates"),
        "--generator",
        "saturn",
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
    ]  # NOTE: no --routes -> has_route=0 (Saturn carries no route structure)
    ingest_env = os.environ.copy()
    _ingest_ld = os.environ.get("RGFN_INGEST_LD_LIBRARY_PATH")
    if _ingest_ld:
        ingest_env["LD_LIBRARY_PATH"] = _ingest_ld
    print(f"[SAT-FR] ingest -> {' '.join(ingest_cmd)}", flush=True)
    subprocess.run(ingest_cmd, check=True, cwd=str(_REPO_ROOT), env=ingest_env)

    print(f"[SAT-FR] done. candidates at {out_dir / 'candidates'}", flush=True)


if __name__ == "__main__":
    main()
