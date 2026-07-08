"""Run-spec dataclass for the LSD-Flow analysis harness (``docs/LSD_FLOW_PROPOSAL.md`` §3b).

One config = one (model x reward) analysis: where the trained checkpoint + gin config live,
how many trajectories to sample, which hub/molecule strategies to report, and where to write
results. The full model x reward x strategy x metric matrix (``matrix.py``) composes these.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


def _default_hub_strategies() -> List[str]:
    return [
        "highest_terminating_flow",
        "highest_flow",
        "most_modes",
        "highest_visitation",
        "lowest_uncertainty",
        "parent_of_topk",
    ]


def _default_combos() -> List[Tuple[str, str]]:
    # (hub_strategy, molecule_strategy) pairs to run end-to-end through acquisition + cost.
    return [
        ("highest_terminating_flow", "topk_reward"),
        ("highest_terminating_flow", "uniform_random"),
        ("lowest_uncertainty", "prob_weighted"),
        ("parent_of_topk", "topk_reward"),  # the control baseline
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
    combos: List[Tuple[str, str]] = field(default_factory=_default_combos)
    report_top_hubs: int = 10
    out_batch_size: int = 96  # molecules a selection should return
    per_hub: int = 8  # concurrency: products per shared hub
    mode_cutoff: float = 0.65  # Butina Tanimoto cutoff for "modes" (§11)
    min_children_for_hub: int = 2  # a diversifying hub has >=2 distinct children
    seed: int = 0

    out_dir: str = "validation/lsdflow/results/seh_rgfn_pilot"
