"""Tune evaluation of an average of v6 models trained on the same survivors.

    python tools/v6_ensemble_eval.py --bucket B --prefixes P1,P2[,P3] [--data DIR] [--out FILE]

Downloads model/tune_scores.parquet of each queue, averages p over the models (the rows
are the same tune survivors in every queue), then reports each eval cut of the first plan
with the threshold swept as in training, next to v5.1 at the same cut. Labels come from
the supplied train ground truth; the audit partition is not touched.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.v5_dist import Store, truth_for  # noqa: E402
from src.v6_train import evaluate_cut  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket")
    parser.add_argument("--root")
    parser.add_argument("--prefixes", required=True)
    parser.add_argument("--data", default=str(Path(__file__).resolve().parents[3] / "student_resource" / "dataset"))
    parser.add_argument("--out")
    args = parser.parse_args()
    prefixes = args.prefixes.split(",")
    stores = [Store(args.bucket, p, args.root) for p in prefixes]
    plan = stores[0].get_json("plan.json")
    frames = []
    for i, store in enumerate(stores):
        frame = pl.read_parquet(store.get_bytes("model/tune_scores.parquet"))
        frames.append(frame.sort("s1", "t").rename({"p": f"p_{i}"}))
    base = frames[0]
    for i, frame in enumerate(frames[1:], 1):
        if not frame.select("s1", "t").equals(base.select("s1", "t")):
            raise SystemExit(f"{prefixes[i]} scored different tune survivors")
        base = base.with_columns(frame[f"p_{i}"])
    models = [f"p_{i}" for i in range(len(frames))]
    scores = base.with_columns(pl.mean_horizontal(models).cast(pl.Float32).alias("p"))
    tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
    truth, _ = truth_for(args.data, {"tune": tune}, np.array([], dtype=np.uint32))
    gt = truth.select("s1", pl.col("target_id").alias("t"))
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    report = {"prefixes": prefixes, "cuts": {}}
    for spec in plan["eval_cuts"]:
        key = f"n{spec['n']}_f{spec['floor']}"
        result, _ = evaluate_cut(scores, spec, tune, gt, country)
        report["cuts"][key] = result
        d = result["paired_delta"]
        print(f"{key:12s} cand/S1={result['pairs_per_query']:.2f} U={result['oracle_u']:.5f} "
              f"ens={result['v6']['macro_f05']:.5f}@{result['v6']['threshold']:.4f} "
              f"v51={result['v51']['macro_f05']:.5f}@{result['v51']['threshold']:.4f} "
              f"delta={d['mean']:+.5f}+-{d['se']:.5f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
