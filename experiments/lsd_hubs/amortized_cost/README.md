# `amortized_cost/` — nested dynamic-fragment cost for SCENT hub batches (Logs/028)

The reactions-per-mode / amortization cost (entry `025`/`026`) treats every building block as a
free purchasable base fragment. SCENT breaks that: it promotes high-reward intermediates into its
library and attaches them in one step, so a molecule's `num_reactions` **hides** the reactions that
built each promoted fragment. `cost.py` adds those back honestly — each **distinct** promoted
fragment is built + charged **exactly once** per costed library (nested via its logged route), in
both the hub-amortized and independent plans (see
`validation/lsdflow/metrics/cost/dynamic_amortization.py`).

Run (rgfn env, CPU only) on a completed SCENT analysis dir + its recipe snapshot:

```bash
source ~/bin/rgfn-smoke-env.sh
python experiments/lsd_hubs/amortized_cost/cost.py \
    --analysis-dir /scratch/.../lsdflow/scent_seh_70189 \
    --snapshot /scratch/.../scent_seh/<ts>/additional_fragments/fragments_4000.json \
    --log-z 74.33 --reward-threshold 7.0 --tag scent_seh_70189
```

`--reward-threshold` is the mode "hit" bar and is **target-specific**: sEH ~7.0; DRD2 ~0.5 (the TDC
"active" cutoff — DRD2's proxy saturates near 1.0, so the gate barely filters).

## Committed results (flow acquisition = highest_terminating_flow × topk_reward, 30k-traj DAGs)

| target | modes | promoted frags (closure) | base rxn/mode (hub / indep) | +promoted build /mode | **augmented rxn/mode (hub / indep)** |
|---|---|---|---|---|---|
| sEH (70189) | 68 | 77 | 1.15 / 1.77 | +1.44 | **2.59 / 3.21** |
| DRD2 (70190) | 85 | 25 | 1.66 / 2.49 | +0.59 | **2.25 / 3.08** |

**Reading it:** charging SCENT's promoted intermediates ~doubles the true per-mode cost the base
metric reports (sEH 1.15 → 2.59), because those "cheap" one-step attachments hide 1–2 reactions of
intermediate synthesis each (sEH: 438/1600 promoted fragments are themselves built from *other*
promoted fragments — real nesting). The hub-vs-independent **saving is unchanged** (the shared
promoted-fragment build is identical in both plans, so it cancels) — batching still wins; it's the
*absolute* cost that was undercounted.

**Scope:** costs the **sampled** acquisition library (mode reps are sampled hub children, which have
per-molecule composition). The exhaustively-**enumerated** neighborhoods (`enumerated_records.csv`)
would need enumerate-mode composition capture to augment — a follow-up.

Result JSONs: `cost_scent_seh_70189.json`, `cost_scent_drd2_70190.json`.
