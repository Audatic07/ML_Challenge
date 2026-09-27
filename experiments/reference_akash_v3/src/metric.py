"""Macro-averaged F0.5 exactly as the challenge defines it (singletons included)."""
import polars as pl


def macro_f05(pred, truth, s1_idx):
    """pred, truth: frames with (idx1, idx2) pairs. s1_idx: every Source 1 idx being scored.

    Per entity: no true matches -> 1.0 if nothing predicted else 0.0;
    otherwise F0.5 from precision and recall (0 when no true positive).
    """
    base = pl.DataFrame({"idx1": s1_idx}).cast(pl.UInt32)
    npred = pred.group_by("idx1").agg(pl.len().alias("np"))
    ntrue = truth.group_by("idx1").agg(pl.len().alias("nt"))
    tp = pred.join(truth, on=["idx1", "idx2"], how="inner").group_by("idx1").agg(pl.len().alias("tp"))
    d = (base.join(npred, on="idx1", how="left").join(ntrue, on="idx1", how="left")
         .join(tp, on="idx1", how="left").fill_null(0))
    p = pl.col("tp") / pl.col("np")
    r = pl.col("tp") / pl.col("nt")
    f = (pl.when(pl.col("nt") == 0).then((pl.col("np") == 0).cast(pl.Float64))
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * p * r / (0.25 * p + r)))
    return d.select(f.mean()).item()


def decide(scored, threshold, unique_assign):
    """Turn scored pairs (idx1, idx2, p) into predicted matches.

    If unique_assign, each Source 2/3 record is kept only for the Source 1 entity
    that gives it the highest probability (Source 1 is deduplicated, so a record
    can belong to at most one entity)."""
    s = scored
    if unique_assign:
        s = s.filter(pl.col("p") == pl.col("p").max().over("idx2"))
    return s.filter(pl.col("p") >= threshold).select("idx1", "idx2")
