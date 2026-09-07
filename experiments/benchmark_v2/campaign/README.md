# the reaction-GFN pipeline — sample, pick_hubs, enumerate, campaign

RGFN, RxnFlow and SCENT only. Hub-batching vs best-candidate on identical chemistry: same enumeration,
same gate, same budget — only the chooser differs.

Configuration, identical across all three: `--pool all --child-policy free_frag --prebuild-k 0`.

`--pool all` removes v1's reward pre-filter on the hub set (v1 ranked flow only over the parents of
the top-1000 candidates by reward). It is free — the flow estimate is read off the record's
log-terms, 0.17 s over all 20,874 eligible hubs — and it removes a confound from the ordering
ablation, which otherwise measures "flow ordering wins" on a pool that was itself reward-selected.
Measured effect: 60% of the walked hub set changes, the outcome moves ~1 mode (Logs/053).

**It raises depth-0 exposure 4x** (8 of the first 40 walked hubs vs 2), because a bought scaffold
costs 0 reactions and still carries thousands of children, so it ranks high by flow. That is legitimate
and honestly priced, but depth-0 catalogue picking is the metric's named degenerate optimum, so every
cell MUST report the walked hubs' depth distribution and the share of delivered modes sitting on
depth-0 hubs, and `--min-hub-depth 1` runs as a labelled sensitivity arm. See runbook 2.2 and 7.5.
`free_frag` is inert on RGFN/RxnFlow (no dynamic library), so it costs nothing and removes a
per-generator special case. K=0 because pre-select-K's up-front reactions are the entire 12.69%
count-once-vs-SPARROW gap on DRD2; at K=0 that gap is exactly 0.00%.

Width knobs, stated so they can be read beside the competitor's stage-2 cap: `n_hubs=200`,
`n_traj=30,000`. Under `--pool all`, `n_hubs` is the ONLY cap — eligibility is ~20,000 hubs, where
v1's `TOPK=1000` pre-filter capped it at ~575 and made `n_hubs` a soft limit.

Emit 50/100/150/200/300 reactions; headline **100**.
