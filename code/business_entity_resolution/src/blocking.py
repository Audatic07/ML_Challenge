"""Key-based blocking.

Seven key types (see config.KEY_TYPES), each prefixed with the country and hashed to
UInt64. "Rarest" token = lowest document frequency over S1 + S2 + S3 of the same split.
A key shared by more S2/S3 records than its cap is dropped. Each S1 entity's candidates
are ranked by the summed weight of the key types they share; the top TOP_K are kept.
"""
from __future__ import annotations

import polars as pl

from .config import KEY_TYPES, TOP_K

KEYS = list(KEY_TYPES)


def _tokens(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return (df.select("row", pl.col(col).str.split(" ").list.unique().alias("tok"))
            .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")))


def doc_freq(frames: list[pl.DataFrame], col: str) -> pl.DataFrame:
    parts = [_tokens(f, col).select("tok") for f in frames]
    return pl.concat(parts).group_by("tok").agg(pl.len().cast(pl.UInt32).alias("df"))


def _rarest(df: pl.DataFrame, col: str, dfreq: pl.DataFrame, n: int) -> pl.DataFrame:
    toks = (_tokens(df, col).join(dfreq, on="tok", how="left")
            .sort(["row", "df", "tok"])
            .group_by("row", maintain_order=True).agg(pl.col("tok").head(n)))
    return toks


def make_keys(df: pl.DataFrame, name_df: pl.DataFrame, addr_df: pl.DataFrame) -> pl.DataFrame:
    """One row per record with a UInt64 key column per key type (null when unavailable)."""
    rn = _rarest(df, "skel", name_df, 2).select(
        "row", pl.col("tok").list.get(0, null_on_oob=True).alias("r1"),
        pl.col("tok").list.get(1, null_on_oob=True).alias("r2"))
    ra = _rarest(df, "addr_tok", addr_df, 1).select(
        "row", pl.col("tok").list.get(0, null_on_oob=True).alias("ra"))
    x = (df.select("row", "country", "postcode", "house", "concat")
         .join(rn, on="row", how="left").join(ra, on="row", how="left"))

    def key(tag: str, *parts: pl.Expr) -> pl.Expr:
        return pl.concat_str([pl.lit(tag), pl.col("country"), *parts], separator="\x1f").hash(seed=11)

    lo, hi = pl.min_horizontal("r1", "r2"), pl.max_horizontal("r1", "r2")
    pair = pl.when(pl.col("r2").is_not_null()).then(pl.concat_str([lo, hi], separator=" "))
    d8 = pl.when(pl.col("concat").str.len_chars() >= 4).then(pl.col("concat").str.slice(0, 8))
    return x.select(
        "row",
        key("A", pair).alias("A"),
        key("B", pl.col("r1"), pl.col("postcode")).alias("B"),
        key("C", pl.col("r1"), pl.col("ra")).alias("C"),
        key("D", d8).alias("D"),
        key("E", pl.col("r1")).alias("E"),
        key("F", pl.col("postcode"), pl.col("house")).alias("F"),
        key("G", pl.col("ra"), pl.col("house")).alias("G"),
    )


class Blocker:
    """Holds the capped target-side key index of one split."""

    def __init__(self, s1: pl.DataFrame, tg: pl.DataFrame):
        name_df = doc_freq([s1, tg], "skel")
        addr_df = doc_freq([s1, tg], "addr_tok")
        self.q_keys = make_keys(s1, name_df, addr_df)
        t_keys = make_keys(tg, name_df, addr_df)
        self.index = {}
        self.dropped = {}
        for k, (_, cap) in KEY_TYPES.items():
            tbl = t_keys.select(pl.col(k).alias("key"), pl.col("row").alias("t")).drop_nulls("key")
            cnt = tbl.group_by("key").agg(pl.len().alias("n"))
            keep = cnt.filter(pl.col("n") <= cap).select("key")
            self.dropped[k] = int((cnt["n"] > cap).sum())
            self.index[k] = tbl.join(keep, on="key", how="semi")
        del t_keys

    def candidates(self, s1_rows, top_k: int = TOP_K) -> pl.DataFrame:
        """(s1, t, bscore, kmask) for the given S1 rows, at most top_k per S1."""
        q = self.q_keys.filter(pl.col("row").is_in(pl.Series(s1_rows, dtype=pl.UInt32)))
        parts = []
        for i, (k, (w, _)) in enumerate(KEY_TYPES.items()):
            qk = q.select(pl.col(k).alias("key"), pl.col("row").alias("s1")).drop_nulls("key")
            parts.append(qk.join(self.index[k], on="key").select(
                "s1", "t", pl.lit(w, pl.UInt8).alias("w"), pl.lit(1 << i, pl.UInt8).alias("m")))
        pairs = (pl.concat(parts).group_by("s1", "t")
                 .agg(pl.col("w").sum().cast(pl.UInt8).alias("bscore"),
                      pl.col("m").sum().cast(pl.UInt8).alias("kmask")))
        return (pairs.sort(["s1", "bscore", "t"], descending=[False, True, False])
                .group_by("s1", maintain_order=True).head(top_k))
