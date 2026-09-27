"""Publish the stage-1 cascade queue for a finished v5 run.

    python -m src.v5_cascade eval  --bucket B --source SRC --prefix DST [--ns 10,15,20,30]
    python -m src.v5_cascade score --bucket B --source SRC --prefix DST --n N --threshold T

`eval` measures stage-1 cut-offs on the tune partition (survivor oracle U and macro F0.5
with a re-selected threshold). `score` copies the source model, records the cut-off and
threshold in its manifest and queues one cascade task per test feature shard plus the
final assembly. The final candidate set is exactly the survivors scored by the model.
"""
from __future__ import annotations

import argparse

from .v5_dist import Store, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["eval", "score"])
    parser.add_argument("--bucket")
    parser.add_argument("--root")
    parser.add_argument("--source", required=True, help="prefix of the finished run")
    parser.add_argument("--prefix", required=True, help="prefix of the new cascade queue")
    parser.add_argument("--ns", default="10,15,20,25,30,40")
    parser.add_argument("--ranks", default="h_rank,bscore_rank")
    parser.add_argument("--rank", default="h_rank")
    parser.add_argument("--n", type=int)
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()
    source, target = Store(args.bucket, args.source, args.root), Store(args.bucket, args.prefix, args.root)
    feats = {name[:-5]: source.get_json(f"tasks/{name}") for name in source.list("tasks") if name.startswith("feat-")}
    if args.step == "eval":
        plan = source.get_json("plan.json")
        target.put_json("plan.json", dict(plan, plan_sha=digest({"cascade_of": args.source})))
        target.put_json("tasks/ceval.json", {
            "id": "ceval", "kind": "ceval", "source_prefix": args.source, "priority": 0, "min_mem_gb": 14,
            "inputs": sorted(t for t, v in feats.items() if v["split"] == "train"),
            "ns": [int(x) for x in args.ns.split(",")], "ranks": args.ranks.split(",")})
        return
    if args.n is None or args.threshold is None:
        parser.error("score needs --n and --threshold")
    manifest = source.get_json("model/model_manifest.json")
    manifest.update(threshold=args.threshold, cascade={"rank": args.rank, "n": args.n}, source_prefix=args.source)
    target.put_bytes("model/model.txt", source.get_bytes("model/model.txt"))
    target.put_json("model/model_manifest.json", manifest)
    ids = []
    for name, task in sorted(feats.items()):
        if task["split"] == "test":
            ids.append(name.replace("feat-", "cascade-"))
            target.put_json(f"tasks/{ids[-1]}.json", {"id": ids[-1], "kind": "cascade", "input": name,
                                                      "source_prefix": args.source, "priority": 1, "min_mem_gb": 8})
    target.put_json("tasks/final.json", {"id": "final", "kind": "final", "inputs": ids, "requires": ids,
                                         "priority": 2, "min_mem_gb": 60})


if __name__ == "__main__":
    main()
