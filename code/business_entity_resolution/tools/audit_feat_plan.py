"""Publish feature tasks for the locked audit S1 rows of a finished v5 run.

    python tools/audit_feat_plan.py --bucket B --source shared/er-v51-20260927 --prefix shared/er-v51-audit-20260927 --data DIR

Same plan settings (normaliser, channels, anchors) as the source run, so its models apply
unchanged. Only retrieval and features run here; no labels are read, so the audit stays
unopened until one champion is evaluated on it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v4_train import partitions  # noqa: E402
from src.v5_dist import Store, digest, log  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--root")
    parser.add_argument("--source", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    source, target = Store(args.bucket, args.source, args.root), Store(args.bucket, args.prefix, args.root)
    if target.exists("plan.json"):
        parser.error("queue exists")
    plan = source.get_json("plan.json")
    split, fit, stop, tune, heldout = partitions(plan["train_queries"], plan["fit"], plan["stop"], plan["tune"])
    audit = np.sort(split.audit).astype(np.uint32)
    if np.isin(audit, np.concatenate([fit, stop, tune])).any():
        raise SystemExit("audit rows overlap the fit/stop/tune rows")
    train = pl.read_csv(Path(args.data) / "train" / "train_source1.tsv", separator="\t", quote_char=None,
                        columns=["country"], infer_schema=False).with_row_index("row")
    if len(train) != plan["train_queries"]:
        raise SystemExit("dataset differs from the source plan")
    memory = {}
    for name in source.list("tasks"):
        if name.startswith("feat-train-"):
            task = source.get_json(f"tasks/{name}")
            memory[task["country"]] = (task["min_mem_gb"], task["catalog_rows"])
    chosen = train.filter(pl.col("row").is_in(pl.Series(audit).implode()))
    tasks = []
    for (country,), group in sorted(chosen.partition_by("country", as_dict=True).items()):
        rows = group["row"].sort().to_list()
        for i in range(0, len(rows), plan["shard"]):
            tasks.append({"id": f"feat-train-{country}-audit-{i // plan['shard']:03d}".replace(" ", "_"), "kind": "feat",
                          "split": "train", "country": country, "rows": rows[i:i + plan["shard"]], "priority": 0,
                          "min_mem_gb": memory[country][0], "catalog_rows": memory[country][1]})
    settings = dict(plan, audit_of=args.source, audit_rows=len(audit), audit_sha=digest(audit.tolist()),
                    task_count=len(tasks))
    settings["plan_sha"] = digest({"audit_of": args.source, "source_plan_sha": plan["plan_sha"], "audit_sha": settings["audit_sha"]})
    target.put_json("plan.json", settings)
    for task in tasks:
        target.put_json(f"tasks/{task['id']}.json", task)
    log(f"published {len(tasks)} audit feature tasks for {len(audit):,} S1 rows")


if __name__ == "__main__":
    main()
