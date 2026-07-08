"""LSD-Flow acquisition primitives (production side) — ``docs/LSD_FLOW_PROPOSAL.md`` §3a.

Post-hoc hub selection over a trained reaction GFlowNet, exposed as a batch-selection
strategy the active-learning loop can consume. The heavy comparative/analysis machinery
(cross-env adapters, the persisted rich DAG, severe tests, the RGFN-vs-SCENT study) lives on
the validation axis under ``validation/lsdflow/`` and imports these primitives (allowed by the
one-way rule); the pipeline never imports back.

Importing this package registers every ``@gin.configurable`` hub/molecule strategy and the
``LSDFlowAcquisition`` entry point, so ``glue.registry`` pulls it in.
"""

from glue.samplers.lsdflow import hub, molecule  # noqa: F401
from glue.samplers.lsdflow.acquisition import LSDFlowAcquisition  # noqa: F401
from glue.samplers.lsdflow.dag import ChildEstimate, Hub, LiteHubDAG  # noqa: F401
from glue.samplers.lsdflow.records import FlowRecord  # noqa: F401
