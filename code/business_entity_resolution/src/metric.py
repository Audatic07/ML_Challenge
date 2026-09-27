"""Competition metric and the match decision rule.

Macro F0.5 per S1 entity, averaged over every evaluated S1 entity:
  true matches G non-empty: F = 5*TP / (4*|P| + |G|)
  G empty (singleton):      F = 1 if nothing predicted else 0
True matches lost at blocking count as misses, singletons count, so this mirrors the
leaderboard.
"""
from __future__ import annotations

import polars as pl


def per_entity(s1_rows, truth: pl.DataFrame, pred: pl.DataFrame) -> pl.DataFrame:
    """s1_rows: every evaluated S1 row; truth/pred: (s1, t) pairs restricted to them."""
    q = pl.DataFrame({"s1": pl.Series(s1_rows, dtype=pl.UInt32)})
    g = truth.group_by("s1").agg(pl.len().alias("g"))
    p = pred.group_by("s1").agg(pl.len().alias("p"))
    tp = (pred.select("s1", "t").join(truth.select("s1", "t"), on=["s1", "t"])
          .group_by("s1").agg(pl.len().alias("tp")))
    out = (q.join(g, on="s1", how="left").join(p, on="s1", how="left").join(tp, on="s1", how="left")
           .with_columns(pl.col("g", "p", "tp").fill_null(0).cast(pl.Int64)))
    f = (pl.when(pl.col("g") == 0).then((pl.col("p") == 0).cast(pl.Float64))
         .otherwise(5 * pl.col("tp") / (4 * pl.col("p") + pl.col("g"))))
    return out.with_columns(f.alias("f"))


def f05_macro(s1_rows, truth: pl.DataFrame, pred: pl.DataFrame) -> float:
    return float(per_entity(s1_rows, truth, pred)["f"].mean())


def oracle(s1_rows, truth: pl.DataFrame, cand: pl.DataFrame) -> float:
    """Best achievable score with this candidate set (predict exactly the retrieved truths)."""
    hit = cand.select("s1", "t").join(truth.select("s1", "t"), on=["s1", "t"])
    return f05_macro(s1_rows, truth, hit)


def unique_assign(scored: pl.DataFrame) -> pl.DataFrame:
    """Keep each S2/S3 record only for the S1 entity that gives it the highest probability.
    Every S2/S3 record belongs to at most one S1 entity in the training data."""
    return (scored.sort(["t", "p", "s1"], descending=[False, True, False])
            .group_by("t", maintain_order=True).head(1))


def decide(scored: pl.DataFrame, threshold: float) -> pl.DataFrame:
    return unique_assign(scored).filter(pl.col("p") >= threshold).select("s1", "t", "p")
