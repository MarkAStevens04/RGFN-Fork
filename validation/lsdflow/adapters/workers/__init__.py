"""Per-env worker entrypoints for the cross-env adapter bridge (``docs/LSD_FLOW_PROPOSAL.md``
§4b).

Each non-RGFN target (SCENT, FragGFN, RxnFlow) runs in its own conda env and cannot co-import
with ``rgfn`` in one process; its worker runs under ``conda run -n <env> ...`` and exchanges
SMILES + action-ids + logprobs over the bridge (the ``scripts/score_batch.py`` shape), which
the in-process client (``validation/lsdflow/adapters/client.py``) rebuilds into a
:class:`~validation.lsdflow.adapters.base.FlowSample`.

Not built in the phase-1 vertical slice (RGFN is in-process). Add ``scent_worker.py`` /
``fraggfn_worker.py`` / ``rxnflow_worker.py`` when their build-order phase begins (§10 steps
3-4); the RGFN flow-extraction logic they mirror is the reusable, model-agnostic
``glue.samplers.lsdflow.rgfn_extract`` composed at *reaction* granularity per §4b.
"""
