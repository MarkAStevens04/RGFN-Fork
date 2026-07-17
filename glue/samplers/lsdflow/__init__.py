"""LSD-Flow primitives (production side) — ``docs/LSD_FLOW_PROPOSAL.md`` §3a.

Post-hoc hub selection over a trained reaction GFlowNet. These are the reusable pieces the
library-cost campaign (``experiments/lsd_hubs/campaign/``) and the validation-axis analysis
(``validation/lsdflow/``) build on: the flow-record schema, the lightweight in-env DAG, the
hub-selection strategies, and the two campaign selection strategies (``BestCandidateStrategy`` /
``HubBatchingStrategy``) with their within-hub child policies and mode selector. The heavy
comparative/analysis machinery lives under ``validation/lsdflow/`` and imports these primitives
(allowed by the one-way rule); the pipeline never imports back.

The campaign strategies emit a uniform ``CampaignResult`` and are written to be AL-ready — an
active-learning acquisition that consumes hubs as a batch-selection sampler is planned but not yet
wired (the loop has no pluggable-sampler hook yet).

Importing this package registers the ``@gin.configurable`` hub strategies and exposes the campaign
strategies, so ``glue.registry`` pulls them in.
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
