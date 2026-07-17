"""Run-spec dataclass for the LSD-Flow analysis harness (``docs/LSD_FLOW_PROPOSAL.md`` §3b).

One config = one (model x reward) analysis: where the trained checkpoint + gin config live,
how many trajectories to sample, which hub strategies to report, and where to write results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


def _default_hub_strategies() -> List[str]:
    return [
        "highest_terminating_flow",
        "highest_flow",
        "most_modes",
        "highest_visitation",
        "lowest_uncertainty",
        "parent_of_topk",
    ]


@dataclass
class LSDFlowRunConfig:
    model: str = "rgfn"
    config_path: str = "configs/glue/fixed_reward_seh_proxy_stdlib.gin"
    checkpoint_path: str = ""
    reward_name: str = "seh"
    run_id: Optional[str] = None

    n_trajectories: int = 2000
    sample_batch_size: int = 100
    device: str = "auto"

    hub_strategies: List[str] = field(default_factory=_default_hub_strategies)
    report_top_hubs: int = 10
    # Mode definition (paper-comparable, matches TanimotoSimilarityModes / dataset_metrics):
    # greedy ECFP r=3 sphere-exclusion at this Tanimoto threshold, gated by a reward/binding
    # cutoff so a "mode" is a distinct *high-affinity* product (§11). reward_threshold=None
    # counts structure-only modes; set it (orientation from the reward) for paper-comparable hits.
    mode_similarity_threshold: float = 0.7
    mode_reward_threshold: Optional[float] = None
    min_children_for_hub: int = 2  # a diversifying hub has >=2 distinct children
    seed: int = 0

    # Reuse a previously-persisted DAG (records.csv) instead of re-sampling — RGFN sampling on
    # the stdlib library is CPU-bound and slow (Logs/020), so enumeration/iteration should not
    # pay it repeatedly. Empty = sample fresh.
    from_records: str = ""

    # Phase-2 exhaustive child enumeration (proposal §4b/§6). 0 = off (sampled DAG only).
    enumerate_top_hubs: int = 0  # enumerate children of this many selected hubs
    enumerate_max_children: int = 4000  # per-hub enumeration cap (docking budget guard)

    out_dir: str = "validation/lsdflow/results/seh_rgfn_pilot"
