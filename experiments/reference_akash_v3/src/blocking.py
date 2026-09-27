"""Candidate generation (blocking).

Each record gets up to seven blocking keys built from its rarest name tokens,
postcode, house number and rarest address token, always prefixed with the
country so records from different countries never compare. Keys shared by too
many Source 2/3 records are dropped. For every Source 1 entity the Source 2/3
records sharing at least one key are scored by the weighted number of shared
keys and the TOP_K best are kept.
"""
import polars as pl

from . import config

KT_NAMES = list(config.KEY_TYPES)                       # ["A", ..., "G"]
KT_CODE = {k: i for i, k in enumerate(KT_NAMES)}
KT_WEIGHT = [config.KEY_TYPES[k][0] for k in KT_NAMES]
KT_CAP = [config.KEY_TYPES[k][1] for k in KT_NAMES]


def token_freq(frames, col):
    """Document frequency of each space-separated token of `col` across frames."""
    parts = [f.select(pl.col(col).str.split(" ").list.unique().alias("tok")).explode("tok") for f in frames]
    return (pl.concat(parts).filter(pl.col("tok").str.len_chars() > 0)
            .group_by("tok").agg(pl.len().alias("freq")))


def rarest_tokens(df, col, freq, k):
    """For each idx, the k rarest tokens of `col` (ties broken alphabetically)."""
    return (df.select("idx", pl.col(col).str.split(" ").alias("tok")).explode("tok")
            .filter(pl.col("tok").str.len_chars() > 0).unique(["idx", "tok"])
            .join(freq, on="tok", how="left").with_columns(pl.col("freq").fill_null(0))
            .sort(["idx", "freq", "tok"])
            .group_by("idx", maintain_order=True).agg(pl.col("tok").head(k).alias(col + "_rare")))


def build_keys(df, name_freq, addr_freq):
    """Return a long frame (idx, key:UInt64, kt:UInt8) of blocking keys for df."""
    nulled = lambda c: pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c))
    r = (df.select("idx", "country", nulled("postcode").alias("pc"), nulled("house").alias("hs"), "concat",
                   pl.col("skel").str.split(" ").list.first().alias("s_first"),
                   pl.col("addr_tok").str.split(" ").list.last().alias("a_last"))
         .join(rarest_tokens(df, "skel", name_freq, 2), on="idx", how="left")
         .join(rarest_tokens(df, "addr_tok", addr_freq, 2), on="idx", how="left"))
    c = pl.col("country")
    r1 = pl.col("skel_rare").list.get(0, null_on_oob=True)
    r2 = pl.col("skel_rare").list.get(1, null_on_oob=True)
    a1 = pl.col("addr_tok_rare").list.get(0, null_on_oob=True)
    a2 = pl.col("addr_tok_rare").list.get(1, null_on_oob=True)
    cat = lambda *parts: pl.concat_str(list(parts), separator="|")
    exprs = {
        "A": cat(pl.lit("A"), c, r1, r2),
        "B": cat(pl.lit("B"), c, r1, pl.col("pc")),
        "C": cat(pl.lit("C"), c, r1, a1),
        "D": pl.when(pl.col("concat").str.len_chars() >= 5)
               .then(cat(pl.lit("D"), c, pl.col("concat").str.slice(0, 8))),
        "E": pl.when(r1.str.len_chars() >= 2).then(cat(pl.lit("E"), c, r1)),
        "F": cat(pl.lit("F"), c, pl.col("pc"), pl.col("hs")),
        "G": cat(pl.lit("G"), c, a1, pl.col("hs")),
        "H": cat(pl.lit("H"), c, a1, a2),
        "I": cat(pl.lit("I"), c, pl.col("pc"), a1),
        "J": cat(pl.lit("J"), c, a2, pl.col("hs")),
        "K": pl.when(pl.col("s_first").str.len_chars() >= 2).then(cat(pl.lit("K"), c, pl.col("s_first"), pl.col("hs"))),
        "L": pl.when((pl.col("a_last").str.len_chars() >= 2) & (pl.col("concat").str.len_chars() >= 1))
               .then(cat(pl.lit("L"), c, pl.col("hs"), pl.col("concat").str.slice(0, 1), pl.col("a_last"))),
    }
    parts = [r.select("idx", e.alias("key"), pl.lit(KT_CODE[k], dtype=pl.UInt8).alias("kt"))
             for k, e in exprs.items()]
    return (pl.concat(parts).drop_nulls("key")
            .with_columns(pl.col("key").hash(seed=7))
            .unique(["idx", "key"]))


def cap_blocks(keys):
    """Drop keys whose block holds more records than the cap for that key type."""
    cap = pl.col("kt").replace_strict(dict(enumerate(KT_CAP)), return_dtype=pl.UInt32)
    return (keys.with_columns(pl.len().over("key").alias("n"))
            .filter(pl.col("n") <= cap)
            .select("idx", "key"))


def candidates(keys1, keys23, top_k=config.TOP_K):
    """Join Source 1 keys to capped Source 2/3 keys and keep the top_k pairs per Source 1 idx.

    Returns (idx1, idx2, bscore, nkeys, kbits) where kbits is a bitmask of the key
    types the pair shares.
    """
    pairs = (keys1.rename({"idx": "idx1"})
             .join(keys23.rename({"idx": "idx2"}), on="key", how="inner")
             .with_columns(
                 pl.col("kt").replace_strict(dict(enumerate(KT_WEIGHT)), return_dtype=pl.Float32).alias("w"),
                 pl.col("kt").replace_strict({i: 1 << i for i in range(len(KT_NAMES))},
                                             return_dtype=pl.UInt16).alias("bit")))
    return (pairs.group_by(["idx1", "idx2"])
            .agg(pl.col("w").sum().alias("bscore"), pl.len().cast(pl.UInt8).alias("nkeys"),
                 pl.col("bit").unique().sum().alias("kbits"))
            .sort(["idx1", "bscore", "idx2"], descending=[False, True, False])
            .group_by("idx1", maintain_order=True).head(top_k))


def prepare_blocking(s1, s23):
    """Build token frequencies and keys for both sides. Returns (keys1, keys23_capped)."""
    name_freq = token_freq([s1, s23], "skel")
    addr_freq = token_freq([s1, s23], "addr_tok")
    keys1 = build_keys(s1, name_freq, addr_freq)
    keys23 = cap_blocks(build_keys(s23, name_freq, addr_freq))
    return keys1, keys23
