"""Block, score and decide on the test set; write both submission files.

OUT_DIR/candidate_pairs.tsv   every candidate the model scored (last stage before the model)
OUT_DIR/matching_results.tsv  unique_assign + threshold from meta.json

Run:  python -m src.predict
"""
from __future__ import annotations

import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import Blocker
from .config import CHUNK, OUT_DIR, WORK_DIR, WORKERS
from .features import FEATURES, build, matrix
from .metric import decide
from .prepare import load

warnings.filterwarnings("ignore", category=DeprecationWarning)


def write_lists(path, s1: pl.DataFrame, tg: pl.DataFrame, pairs: pl.DataFrame, header: str) -> None:
    """One row per S1 entity in test-file order; ids by descending probability; '' when none."""
    ids = (pairs.join(tg.select(pl.col("row").alias("t"), pl.col("id").alias("tid")), on="t")
           .sort(["s1", "p", "t"], descending=[False, True, False])
           .group_by("s1", maintain_order=True).agg(pl.col("tid").str.join(",").alias(header)))
    out = (s1.select(pl.col("row").alias("s1"), pl.col("id").alias("source1_entity_id"))
           .join(ids, on="s1", how="left").with_columns(pl.col(header).fill_null("")))
    out.select("source1_entity_id", header).write_csv(path, separator="\t", quote_style="never")


def main() -> None:
    t0 = time.time()
    meta = json.loads((WORK_DIR / "meta.json").read_text())
    if meta["features"] != FEATURES:
        raise SystemExit("feature list changed since training: retrain first")
    model = lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    s1, tg = load("test")
    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"blocking index built in {time.time() - t:.0f}s; keys dropped over cap: {blocker.dropped}",
          flush=True)
    rows = np.arange(len(s1), dtype=np.uint32)
    scored = []
    for i in range(0, len(rows), CHUNK):
        t = time.time()
        cand = blocker.candidates(rows[i:i + CHUNK])
        feats = build(cand, s1, tg)
        p = model.predict(matrix(feats), num_threads=WORKERS)
        scored.append(feats.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32)))
        print(f"  scored rows {i:,}+{len(rows[i:i + CHUNK]):,}: {len(cand):,} pairs "
              f"in {time.time() - t:.0f}s", flush=True)
        del feats, cand
    del blocker
    scored = pl.concat(scored)
    matches = decide(scored, meta["threshold"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lists(OUT_DIR / "candidate_pairs.tsv", s1, tg, scored, "candidate_entity_ids")
    write_lists(OUT_DIR / "matching_results.tsv", s1, tg, matches, "matched_entity_ids")
    n_with = matches["s1"].n_unique()
    stats = {
        "test_entities": len(s1), "candidate_pairs": len(scored),
        "candidates_per_entity": len(scored) / len(s1),
        "zero_candidate_entities": len(s1) - scored["s1"].n_unique(),
        "matches": len(matches), "empty_share": 1 - n_with / len(s1),
        "threshold": meta["threshold"], "seconds": time.time() - t0,
    }
    (WORK_DIR / "predict_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
