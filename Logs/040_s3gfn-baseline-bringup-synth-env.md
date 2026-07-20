# S3-GFN (non-reaction baseline) — env bring-up, reward injection, and the synthesizability-library decision
**Date:** 2026-07-20, ~3pm

## Question

Can we run S3-GFN — the SMILES-based (non-reaction) generator — against *our* sEH reward, with a
synthesizability signal it can actually learn from, given we can't use the building-block library the
S3-GFN paper used?

## Context & Summary

The LSD-Flow library-efficiency benchmark (`docs/LSD_FLOW_BENCHMARK_PLAN.md`, T3) needs a strong
**non-reaction** baseline. S3-GFN (`[kim2026s3gfn]`) is that baseline: it generates molecules as raw
SMILES strings (post-training the GP-MolFormer language model) and makes them synthesizable by *soft
regularization* rather than by building them from reactions. So its molecules are individually
synthesizable but share **no route structure** — exactly the foil for our headline that a *reaction*
generator plus hub-batching wins on reactions-per-mode.

For the comparison to be fair, S3-GFN must optimize the **same** sEH reward as the reaction-GFN
entrants (RGFN/RxnFlow/SCENT), and its own synthesizability machinery must actually work. This entry
records standing the generator up: getting its environment to import, injecting our frozen sEH proxy
in place of its native scoring, and — the main scientific decision — choosing the building-block
library + reaction templates that give S3-GFN a synthesizability signal it can bootstrap from, since
the paper's choice (Enamine's commercial catalog) isn't available to us.

## Answer

The full S3-GFN pipeline works end-to-end: our sEH reward is injected faithfully (the training reward
matches S3-GFN's native sEH formula exactly), the model trains, and it emits a candidate pool in our
standard format marked as route-less (`has_route=0`), ready for post-hoc routing. On the
synthesizability question: with the **public ZINCFrag** building-block library and the **fuller
105-reaction template set**, the fraction of generated molecules that are synthesizable climbs from
**4.7% to over 90%** across just 200 training steps, and the emitted pool is **96% synthesizable** —
clearing the T3.1 bar (≥95%). The paper's 71-template set gives only **0.8%** against ZINCFrag's small
fragments, which is too weak to bootstrap, so we use the 105-template set. Crucially, this library
choice shapes only *what S3-GFN generates* — the benchmark's actual score (reactions-per-mode) is
computed afterward by a neutral referee (AiZynth→SPARROW) that's independent of it.

## Relevance to our Publication

The benchmark's headline claim — a reaction generator with hub-batching beats every library-assembly
method on reactions-per-mode — is only convincing if the non-reaction baseline was given a genuine
chance. Reviewers (NeurIPS/ICLR-style) will ask two things: was S3-GFN optimizing our reward (not its
own toy objective)? and was its synthesizability method actually functioning? This entry answers both
(same frozen sEH proxy; synthesizability climbing to 90%+; a 96%-synthesizable pool), and documents
the one forced deviation from the paper (ZINCFrag instead of Enamine) transparently, with the reason
it doesn't bias the comparison.

## Next Experiments

**Refining for publication**
- Run the full-scale S3-GFN (5000 steps, matched to the other entrants), route its pool through
  AiZynth→SPARROW, and place it on the reactions-per-mode frontier (T3.2).
- Extend across systems via the modular reward seam (6TD3 / ClpP / DRD2), reusing the same wiring.

**Next steps in project**
- Finish the native-route SPARROW cross-check: the SCENT native re-run (job 70974) produced sample
  routes + hubs but its hub-enumeration output (`enum_children.json`) didn't land — needs repair.
- Finalize the T2 compute frontier once the timed route-recovery cache (job 70976) completes.

---

# Re-creation

### Relevant Files

Root: repository root unless noted. `/external/s3gfn/...` is a git-ignored upstream clone; its retro
env is rebuilt from the committed build script, not committed itself.

**Scripts**
- `./external/setup_s3gfn.sh` — builds the `s3gfn` conda env. Key bring-up fixes baked in: install
  recursion `gflownet` with `--no-deps` (its metadata hard-pins `torch==2.1.2`, which otherwise
  silently downgrades the 2.5.1 stack and breaks the pyg `+pt25` extensions); pin `setuptools<81`
  (PyTDC's `pkg_resources`); copy RDKit's `fpscores.pkl.gz` into `gflownet/utils` (SA scoring).
- `./experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh` — builds the retro synthesizability env
  (`zincfrag_hb105`): gdown ZINCFrag, stereo-strip + dedup canonicalize the blocks (required — S3-GFN
  indexes blocks by exact string and queries with stereo stripped), copy the 105-template file,
  validate that in-stock blocks score 1.0. **Its header documents the library decision in full.**
- `./validation/generators/s3gfn/run_s3gfn_fixed.py` — fixed-reward driver. Stubs
  `rxnflow.tasks.unidock_vina` (needs openbabel; docking is bridged instead), calls S3-GFN's runtime-
  dep loader, then injects our reward by replacing the module-global `get_scores`; builds the trainer,
  trains, samples a pool, and shells to `ingest_candidates.py` under the `rgfn` env (no `--routes` →
  `has_route=0`). Passes the rgfn CUDA libs to the ingest child via `RGFN_INGEST_LD_LIBRARY_PATH` only
  (keeps the s3gfn parent's torch cu121 clean).
- `./validation/generators/s3gfn/fixed_reward.py` — swappable reward providers sharing one interface
  (`SEHFrozenReward` implemented; `DRD2FrozenReward` / `DockingBridgeReward` the modular seam).
  `SEHFrozenReward.predict` guards `bengio2021flow.mol2graph` in try/except (S3-GFN's unconstrained
  SMILES can contain atoms outside the sEH featurizer's set — matches native `mol2seh`'s guard).
- `./validation/configs/s3gfn_seh_fixed.yaml` — run config; `retro_env: zincfrag_hb105` with the
  decision documented inline.

**Datasets**
- `/external/s3gfn/data/building_blocks/zincfrag.smi.gz` — ZINCFrag (public, ~200k fragments;
  gdown id `16N8Xyxr9a-CifjIofgdH3ssFukC4Eh_V`). S3-GFN's reproducible alternative to Enamine; noted
  as included in AiZynthFinder's ZINC stock (aligns with our post-hoc referee).
- `/external/s3gfn/data/envs/zincfrag_hb105/{building_block.smi, template.txt}` — the retro env we
  use: 178,622 unique stereo-stripped blocks + 105 templates (`hb.txt`).
- `/external/s3gfn/data/envs/zincfrag/` — same blocks + the paper's 71 templates (`hb_edited.txt`);
  kept for reference (the 0.8% comparison).

**Results**
- De-risk run outputs were written to session scratch (ephemeral) — the numbers below are the record.

### Relevant Versions

Uncommitted (branch `Hub-Analysis`). Files to commit: `external/setup_s3gfn.sh`,
`experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh`, `validation/generators/s3gfn/`,
`validation/configs/s3gfn_seh_fixed.yaml`, `Logs/references/{references.bib,README.md}` (S3-GFN entry),
this log. `[TODO — add commit hash after pushing]`

### Relevant Resources

**Sources**
- `[kim2026s3gfn]` — S3-GFN (SMILES GFlowNet + soft synthesizability); PDF in `Logs/references/pdfs/`.
- `[bengio2021flow]` / `[bengio2021gflownet]` — the Bengio-2021 sEH MPNN; the SAME proxy our benchmark
  uses (`rgfn/.../seh_proxy.py`) and S3-GFN's native sEH scoring load — the fairness anchor.
- S3-GFN repo `data/README.md` — building-block/template prep; ZINCFrag as the public library.

**Packages**
- `transformers` (GP-MolFormer `ibm-research/GP-MoLFormer-Uniq`) — the pretrained SMILES LM.
- recursion `gflownet` (`bengio2021flow`) — sEH proxy; used by `fixed_reward.py`.
- PyTDC (`tdc.Oracle`) — S3-GFN's QED/SA/DRD2/… oracles; needs `setuptools<81`.
- vendored `rxnflow.envs.retrosynthesis` (inside the S3-GFN clone) — the retro analyzer used by
  `s3gfn.synthesizability`; reads the two env text files.

### Method

1. **Env** — `bash external/setup_s3gfn.sh`; repaired in place: restore torch 2.5.1 + tg 2.6.1,
   reinstall `gflownet --no-deps`, `setuptools<81`, copy `fpscores.pkl.gz`. Verified imports: torch
   2.5.1+cu121 (CUDA ok), `bengio2021flow`, `s3gfn`, `transformers`, `xformers`, `tdc`, SA.
2. **Retro env** — `bash experiments/lsd_hubs/campaign/build_s3gfn_retro_env.sh` → `zincfrag_hb105`
   (178,622 blocks + 105 templates). Validated in-stock blocks score 1.0 (exact-string + stereo).
3. **Initial synth_ratio** — sampled 256 molecules from the GP-MolFormer prior, scored each env with
   the retro analyzer (retro_steps 3 and 4).
4. **De-risk training** — `conda run -n s3gfn python validation/generators/s3gfn/run_s3gfn_fixed.py
   --cfg validation/configs/s3gfn_seh_fixed.yaml --n-train-steps 200 --n-samples 50 --retro-env
   zincfrag_hb105` on the login A100. Also a 3-step end-to-end smoke through ingest (has_route=0).

### Results

**Initial synth_ratio on the GP-MolFormer prior (256 samples):**

| Retro env | Templates | Blocks | retro_steps | synth_ratio |
|---|---|---|---|---|
| `zincfrag` | 71 (`hb_edited`, paper's) | 178,622 | 3 / 4 | **0.008** (2/256) |
| `zincfrag_hb105` | 105 (`hb.txt`) | 178,622 | 3 / 4 | **0.047** (12/256) |

(retro_steps 3 vs 4 identical → use 3.)

**200-step de-risk (ZINCFrag + 105 templates), sEH reward injected:**

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 199 |
|---|---|---|---|---|---|---|---|---|---|
| synth_ratio | .047 | .063 | .111 | .766 | .688 | .844 | .859 | .797 | **.938** |
| avg sEH reward | .46 | .47 | .47 | .70 | .69 | .74 | .76 | .76 | .72 |
| positive buffer | 3 | 59 | 165 | 671 | 1169 | 1646 | 2131 | 2517 | 2879 |

- Final eval: synth_ratio 0.90, avg reward 0.75, avg SA 2.49 (drug-like), top-100 reward 0.94.
- Emitted pool (50 unique valid): **96% internally synthesizable (48/50)** → meets T3.1 (≥95%).
- Loss finite throughout (positive buffer non-empty from step 0); no crashes after the `mol2graph`
  guard.
- Timing: ~3 s/step effective (628 s for 200 steps incl. model load + 2 evals) → full 5000-step run
  ≈ 3.5–4 h GPU.
- 3-step smoke: reward injected (`reward_scale=0.125`=1/8, matching native `mol2seh`), pool sampled,
  ingest wrote candidates with `has_route=0`, conformance OK.
