# the reaction-GFN pipeline — sample, pick_hubs, enumerate, campaign

RGFN, RxnFlow and SCENT only. Hub-batching vs best-candidate on identical chemistry: same enumeration,
same gate, same budget — only the chooser differs.

Configuration, identical across all three: `--child-policy free_frag --prebuild-k 0`.
`free_frag` is inert on RGFN/RxnFlow (no dynamic library), so it costs nothing and removes a
per-generator special case. K=0 because pre-select-K's up-front reactions are the entire 12.69%
count-once-vs-SPARROW gap on DRD2; at K=0 that gap is exactly 0.00%.

Width knobs, stated so they can be read beside the competitor's stage-2 cap: `n_hubs=200`,
`n_traj=30,000`.

Emit 50/100/150/200/300 reactions; headline **100**.
