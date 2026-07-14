"""Samplers — batch-selection strategies for training.

Upstream samplers live in `rgfn/shared/samplers/` (random, sequential) and are
selected via `configs/samplers/*.gin`. If/when we expand batch selection
(diversity-aware batches, active-learning acquisition over the expensive docking
oracle, curriculum schedules, ...), the implementations go here and subclass the
upstream `SamplerBase` (`rgfn/api/sampler_base.py`).

Import new sampler modules below so `glue.registry` registers them.
"""

# LSD-Flow post-hoc hub selection (docs/LSD_FLOW_PROPOSAL.md §3a): hub selection IS a
# batch-selection strategy, so it lives here. Importing registers the @gin.configurable hub
# strategies + exposes the campaign strategies (hub-batching / best-candidate, Logs/028).
from glue.samplers import lsdflow  # noqa: F401

# from glue.samplers.diversity_sampler import DiversitySampler  # noqa: F401
