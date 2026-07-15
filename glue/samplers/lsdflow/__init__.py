"""LSD-Flow acquisition primitives (production side) — ``docs/LSD_FLOW_PROPOSAL.md`` §3a.

Post-hoc hub selection over a trained reaction GFlowNet, exposed as a batch-selection
strategy the active-learning loop can consume. The heavy comparative/analysis machinery
(cross-env adapters, the persisted rich DAG, severe tests, the RGFN-vs-SCENT study) lives on
the validation axis under ``validation/lsdflow/`` and imports these primitives (allowed by the
one-way rule); the pipeline never imports back.

Importing this package registers the ``@gin.configurable`` hub strategies and exposes the
campaign strategies (the AL-facing entry points), so ``glue.registry`` pulls them in.
"""

from glue.samplers.lsdflow import hub  # noqa: F401
from glue.samplers.lsdflow.campaign import (  # noqa: F401
    BestCandidateStrategy,
    CampaignPoint,
    CampaignResult,
    CampaignStrategy,
    Candidate,
    EnumChild,
    EnumeratedHub,
    HubBatchingStrategy,
    fragment_fanout,
    rank_fragments,
)
from glue.samplers.lsdflow.child_select import (  # noqa: F401
    ChildSelectionPolicy,
    FreeFragChildPolicy,
    RewardChildPolicy,
    available_child_policies,
    make_child_policy,
)
from glue.samplers.lsdflow.dag import ChildEstimate, Hub, LiteHubDAG  # noqa: F401
from glue.samplers.lsdflow.records import FlowRecord  # noqa: F401
