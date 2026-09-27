"""Evaluate a saved model on a fixed development subset; never opens the audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import Blocker
from .config import CHUNK, SEED, WORK_DIR, WORKERS
from .features import FEATURES, build, matrix
from .metric import decide, per_entity
from .predict import validate_retrieval_config
from .prepare import load, load_truth
from .splits import make_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    meta = json.loads((WORK_DIR / "meta.json").read_text())
    validate_retrieval_config(meta)
    if meta["features"] != FEATURES:
        raise ValueError("model feature mismatch")
    s1, tg = load("train")
    rows = make_split(len(s1), 1, args.queries, SEED).validation
    gt = load_truth(s1, tg).filter(pl.col("s1").is_in(pl.Series(rows).implode()))
    model = lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    blocker = Blocker(s1, tg)
    outputs = []
    for offset in range(0, len(rows), CHUNK):
        cand = blocker.candidates(rows[offset:offset+CHUNK])
        feats = build(cand, s1, tg)
        outputs.append(cand.select("s1", "t").with_columns(
            pl.Series("p", model.predict(matrix(feats), num_threads=WORKERS), dtype=pl.Float32)))
    scored = pl.concat(outputs)
    pred = decide(scored, meta["threshold"])
    per = per_entity(rows, gt, pred)
    ceiling = per_entity(rows, gt, scored.select("s1", "t").join(gt, on=["s1", "t"]))
    per = per.join(ceiling.select("s1", pl.col("f").alias("oracle_f05")), on="s1").join(
        s1.select(pl.col("row").alias("s1"), "country"), on="s1")
    report = dict(queries=len(rows), full_target_catalog=len(tg), macro_f05=per["f"].mean(),
                  oracle_f05=per["oracle_f05"].mean(), threshold=meta["threshold"],
                  pairs=len(scored), seconds=time.perf_counter()-start,
                  model_sha256=hashlib.sha256((WORK_DIR/"model.txt").read_bytes()).hexdigest(),
                  query_rows_sha256=hashlib.sha256(rows.astype("<u4").tobytes()).hexdigest(),
                  partition="development; not locked audit or leaderboard",
                  countries=per.group_by("country").agg(pl.len().alias("queries"),
                      pl.col("f").mean().alias("macro_f05"), pl.col("oracle_f05").mean()).to_dicts())
    args.out.mkdir(parents=True, exist_ok=True)
    per.write_parquet(args.out/"per_query.parquet")
    (args.out/"metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
