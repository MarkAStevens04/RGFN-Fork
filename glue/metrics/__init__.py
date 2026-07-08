"""Metrics — crash-safe / glue-specific variants of upstream training metrics.

Upstream metrics live in ``rgfn/trainer/metrics/``. We do not edit them; when one
needs a fix or a glue-specific behaviour, subclass it here and reference the
subclass from ``configs/glue/``. Imported below so ``glue.registry`` registers
them with gin.
"""

# LSD-Flow flow recovery + uncertainty (docs/LSD_FLOW_PROPOSAL.md §2). Pure functions (no gin
# classes), imported here so the module is on the glue.registry startup path per the proposal.
from glue.metrics import lsdflow_flow, uncertainty  # noqa: F401
from glue.metrics.safe_num_scaffolds import SafeNumScaffoldsFound  # noqa: F401
