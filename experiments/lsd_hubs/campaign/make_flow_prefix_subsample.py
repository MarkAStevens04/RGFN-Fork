#!/usr/bin/env python
"""Cut a flow-prefix subsample out of a frozen enumeration, so both arms can be run on the EXACT
same candidates.

Why a flow PREFIX rather than a random sample
---------------------------------------------
The question this arm answers is "how well does SPARROW do if we hand it the same candidates
hub-batching gets?".  A uniform random draw does not answer that: it thins every hub by the same
factor (measured on seed 43, gate 7.0: median 423 qualifying children per hub down to ~71), and
per-hub density is precisely the thing hub-batching exploits -- it amortises one hub prefix over
many children.  Random subsampling therefore attacks the mechanism under test instead of levelling
the field.

So we take the pool hub-batching actually REACHES: walk the hubs in flow-descending order, keeping
ALL of each hub's children, until the qualifying count crosses the target.  At a 100-reaction budget
hub-batching only touches ~2 hubs, so a prefix of 6-9 hubs is already generous relative to what our
arm spends -- and every candidate in the file is one our arm would have seen.

Where the flow ordering comes from
----------------------------------
NOT from ``hub_scores.csv``.  On the scent_seh cells that file was written at 16:52 by an earlier
``pick_hubs`` run while the frozen ``enum_children.json`` dates from 21:31; 42 of its 200 rows have
no enum record and -- for seed 44 -- the rank-#1 row is absent from the enumeration entirely, so
"highest flow first" would silently start at the wrong hub.  That is the same stale-artifact class of
bug that cost us Results 13-15 (Logs/062).

Instead the ordering is read off the frozen artifact itself: ``pick_hubs`` writes hubs in walk order
and the enumerator preserves it, so ``enum_children.json``'s own hub sequence IS flow-descending.
Verified against the surviving overlap on both seeds: 12,403/12,403 and 13,041/13,041 concordant
pairs, i.e. 100% agreement on every hub the two runs share.  ``--verify-order`` re-checks this when a
scores file is available; the subsample never depends on it.

What is kept verbatim
---------------------
Every selected hub record is copied whole -- ``hub_input``, ``hub_key``, ``depth``, ``uncertainty``,
``n_effective`` and ALL children, sub-gate ones included.  Sub-gate children are dropped by the mode
selector anyway, but ``campaign.py`` charges ``cum_calls += len(hub.children)``, so removing them
would silently understate the oracle cost hub-batching actually paid.  The gate is applied only to
decide WHERE to stop, never to filter the file.

``meta.json``, ``compositions.json`` and ``routes.json`` are copied in beside the output because
``sparrow_select_frontier.py`` reads its provenance and fragment-coverage checks from the pool's
parent directory -- without them those guards silently pass.

Run:
  conda run -n rgfn python experiments/lsd_hubs/campaign/make_flow_prefix_subsample.py \
      --snapshot /scratch/.../_enum_snapshot_20260820/seh_seed43 \
      --gate 5.0 --target 20000 --out /scratch/.../_enum_subsample_flow20k/seh_seed43
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SIDECARS = ("meta.json", "compositions.json", "routes.json")


def md5(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def qualifies(reward: float, gate: float, higher_is_better: bool) -> bool:
    return reward >= gate if higher_is_better else reward <= gate


def verify_order(hubs, scores_path: Path) -> dict:
    """Concordance between the enum file's own hub order and an external flow ranking.

    Reported, never enforced: the external file may cover a different hub set (that is exactly why we
    do not join against it), so the useful statistic is agreement on the OVERLAP.
    """
    import csv

    rank = {r["smiles"]: int(r["rank"]) for r in csv.DictReader(open(scores_path))}
    pos = [rank[h["hub_key"]] for h in hubs if h["hub_key"] in rank]
    n = len(pos)
    conc = sum(1 for a in range(n) for b in range(a + 1, n) if pos[a] < pos[b])
    tot = n * (n - 1) // 2
    return {
        "scores_file": str(scores_path),
        "overlap_hubs": n,
        "scores_rows": len(rank),
        "concordant_pairs": conc,
        "total_pairs": tot,
        "concordance": (conc / tot) if tot else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--snapshot", required=True, help="frozen snapshot dir holding enum_children.json"
    )
    p.add_argument("--out", required=True)
    p.add_argument("--gate", type=float, required=True)
    p.add_argument(
        "--target", type=int, default=20000, help="stop once distinct qualifying >= this"
    )
    p.add_argument("--higher-is-better", dest="hib", action="store_true", default=True)
    p.add_argument("--lower-is-better", dest="hib", action="store_false")
    p.add_argument(
        "--verify-order", default=None, help="optional hub_scores.csv for a concordance report"
    )
    a = p.parse_args()

    snap, out = Path(a.snapshot), Path(a.out)
    src = snap / "enum_children.json"
    if not src.exists():
        raise SystemExit(f"ABORT: no enum_children.json in {snap}")
    src_md5 = md5(src)
    data = json.load(open(src))
    hubs = data.get("hubs", [])
    if not hubs:
        raise SystemExit(f"ABORT: {src} has no hubs")

    # Walk in file order == flow-descending walk order. Whole hubs only: stopping mid-hub would thin
    # the density this arm is meant to preserve.
    kept, seen, n_children = [], set(), 0
    for h in hubs:
        kept.append(h)
        n_children += len(h["children"])
        for c in h["children"]:
            if qualifies(float(c["reward"]), a.gate, a.hib):
                seen.add(c["smiles"])
        if len(seen) >= a.target:
            break

    total_q = len(
        {
            c["smiles"]
            for h in hubs
            for c in h["children"]
            if qualifies(float(c["reward"]), a.gate, a.hib)
        }
    )
    if len(seen) < a.target:
        print(
            f"[subsample] WARNING: whole pool holds only {len(seen):,} qualifying candidates, "
            f"below the {a.target:,} target -- keeping all {len(kept)} hubs."
        )

    out.mkdir(parents=True, exist_ok=True)
    dst = out / "enum_children.json"
    json.dump({**{k: v for k, v in data.items() if k != "hubs"}, "hubs": kept}, open(dst, "w"))
    out_md5 = md5(dst)
    (out / "enum_children.md5").write_text(f"{out_md5}  enum_children.json\n")

    copied = []
    for name in SIDECARS:
        s = snap / name
        if s.exists():
            shutil.copy2(s, out / name)
            copied.append(name)
        else:
            print(
                f"[subsample] NOTE: {name} absent from the snapshot; guards that read it will no-op"
            )

    manifest = {
        "kind": "flow_prefix_subsample",
        "source_snapshot": str(snap),
        "source_enum_children_md5": src_md5,
        "output_enum_children_md5": out_md5,
        "gate": a.gate,
        "higher_is_better": a.hib,
        "target_qualifying": a.target,
        "hubs_total": len(hubs),
        "hubs_kept": len(kept),
        "qualifying_kept": len(seen),
        "qualifying_pool_total": total_q,
        "qualifying_fraction_of_pool": (len(seen) / total_q) if total_q else None,
        "children_kept": n_children,
        "children_total": sum(len(h["children"]) for h in hubs),
        "hub_order_source": "enum_children.json file order (pick_hubs walk order == flow-descending)",
        "sidecars_copied": copied,
    }
    if a.verify_order:
        manifest["order_verification"] = verify_order(kept, Path(a.verify_order))
    json.dump(manifest, open(out / "subsample_manifest.json", "w"), indent=2)

    print(
        f"[subsample] {snap.name}: kept {len(kept)}/{len(hubs)} hubs | "
        f"{len(seen):,} qualifying (>= gate {a.gate}) = {len(seen)/total_q:.1%} of the "
        f"{total_q:,} in the full pool | {n_children:,} children written"
    )
    print(f"[subsample] src md5 {src_md5[:12]} -> out md5 {out_md5[:12]}  ({dst})")


if __name__ == "__main__":
    main()
