"""Run blocking + the trained matcher over the full test set and write both output files.

Outputs (in OUT_DIR):
  candidate_pairs.tsv   every candidate the model scored, per Source 1 entity
  matching_results.tsv  final matches (subset of the candidates)
Run:  python -m src.predict
"""
import json
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import prepare_blocking
from .metric import decide
from .pipeline import iter_scored_chunks
from .prepare import load_split


def write_lists(s1, pairs, s23, col, path):
    """Write one row per Source 1 entity with a comma-joined list of Source 2/3 ids."""
    lists = (pairs.join(s23.select(pl.col("idx").alias("idx2"), pl.col("entity_id").alias("eid")), on="idx2")
             .unique(["idx1", "eid"]).sort(["idx1", "eid"])
             .group_by("idx1").agg(pl.col("eid").str.join(",").alias(col)))
    out = (s1.select(pl.col("idx").alias("idx1"), pl.col("entity_id").alias("source1_entity_id"))
           .join(lists, on="idx1", how="left").with_columns(pl.col(col).fill_null(""))
           .sort("idx1").select("source1_entity_id", col))
    assert out.height == s1.height and out["source1_entity_id"].n_unique() == s1.height
    out.write_csv(path, separator="\t", quote_style="never")
    return out


def main():
    t0 = time.time()
    meta = json.loads((config.WORK_DIR / "meta.json").read_text())
    model = lgb.Booster(model_file=str(config.WORK_DIR / "model.txt"))
    features = meta["features"]

    print("Loading and normalising test data...", flush=True)
    s1, s23 = load_split("test")
    print(f"  S1={s1.height:,} S2+S3={s23.height:,}; countries: "
          f"{dict(s1['country'].value_counts().sort('count', descending=True).iter_rows())}", flush=True)

    print("Building blocking keys...", flush=True)
    keys1, keys23 = prepare_blocking(s1, s23)

    print("Scoring candidates...", flush=True)
    scored = []
    for feats in iter_scored_chunks(np.arange(s1.height), keys1, keys23, s1, s23):
        p = model.predict(feats.select(features).to_numpy(), num_threads=config.N_WORKERS)
        scored.append(feats.select("idx1", "idx2").with_columns(pl.Series("p", p, dtype=pl.Float32)))
    scored = pl.concat(scored)

    matches = decide(scored, meta["threshold"], meta["unique_assign"])
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cand = write_lists(s1, scored.select("idx1", "idx2"), s23, "candidate_entity_ids",
                       config.OUT_DIR / "candidate_pairs.tsv")
    res = write_lists(s1, matches, s23, "matched_entity_ids", config.OUT_DIR / "matching_results.tsv")

    n_empty = (res["matched_entity_ids"] == "").sum()
    print(f"Wrote {config.OUT_DIR}: {res.height:,} rows, {matches.height:,} matches, "
          f"{n_empty:,} empty ({n_empty / res.height:.1%}); candidates {scored.height:,}; "
          f"threshold {meta['threshold']}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
