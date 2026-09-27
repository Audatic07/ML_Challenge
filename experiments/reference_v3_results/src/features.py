"""Pair features, computed in bulk with rapidfuzz.process.cpdist (element-wise, multithreaded).

Address similarities are set to -1 when either address is empty (rapidfuzz returns 100
for two empty strings). Exact-match flags use -1 when a side is missing.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import KEYS
from .config import WORKERS

VIEWS = ["name_norm", "core", "skel", "concat", "addr_norm", "addr_tok", "postcode", "house", "country"]

# (feature, view, scorer, masked-when-empty)
STRING_FEATURES = [
    ("n_ratio", "name_norm", fuzz.ratio, False),
    ("n_partial", "name_norm", fuzz.partial_ratio, False),
    ("n_tsort", "name_norm", fuzz.token_sort_ratio, False),
    ("n_tset", "name_norm", fuzz.token_set_ratio, False),
    ("c_ratio", "core", fuzz.ratio, False),
    ("c_tset", "core", fuzz.token_set_ratio, False),
    ("c_jw", "core", JaroWinkler.normalized_similarity, False),
    ("sk_ratio", "skel", fuzz.ratio, False),
    ("sk_tset", "skel", fuzz.token_set_ratio, False),
    ("cc_ratio", "concat", fuzz.ratio, False),
    ("cc_partial", "concat", fuzz.partial_ratio, False),
    ("a_ratio", "addr_norm", fuzz.ratio, True),
    ("a_partial", "addr_norm", fuzz.partial_ratio, True),
    ("a_tsort", "addr_norm", fuzz.token_sort_ratio, True),
    ("a_tset", "addr_norm", fuzz.token_set_ratio, True),
    ("at_ratio", "addr_tok", fuzz.ratio, True),
    ("at_tset", "addr_tok", fuzz.token_set_ratio, True),
]
GROUP_BASE = ["n_tset", "a_tset", "cc_ratio", "at_tset", "bscore"]

FEATURES = (
    [f for f, *_ in STRING_FEATURES]
    + ["n_exact", "c_exact", "pc_eq", "hs_eq", "cty_eq",
       "n_len1", "n_len2", "a_len1", "a_len2", "ntok_r", "src", "bscore"]
    + [f"k{k}" for k in KEYS]
    + ["n_cand"] + [f"{b}_gap" for b in GROUP_BASE] + [f"{b}_rank" for b in GROUP_BASE]
)


def _eq(a: pl.Expr, b: pl.Expr) -> pl.Expr:
    return (pl.when(a.is_null() | b.is_null() | (a == "") | (b == "")).then(-1)
            .otherwise((a == b).cast(pl.Int8))).cast(pl.Float32)


def build(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """pairs: (s1, t, bscore, kmask). Returns pairs with every column in FEATURES added."""
    a = s1.select(VIEWS).gather(pairs["s1"]).with_columns(pl.col(VIEWS[:6]).fill_null(""))
    b = tg.select(VIEWS + ["src"]).gather(pairs["t"]).with_columns(pl.col(VIEWS[:6]).fill_null(""))
    cols = {}
    lists = {}
    for _, view, _, _ in STRING_FEATURES:
        if view not in lists:
            lists[view] = (a[view].to_list(), b[view].to_list())
    empty_a = (a["addr_norm"] == "").to_numpy() | (b["addr_norm"] == "").to_numpy()
    empty_t = (a["addr_tok"] == "").to_numpy() | (b["addr_tok"] == "").to_numpy()
    for name, view, scorer, masked in STRING_FEATURES:
        x, y = lists[view]
        v = process.cpdist(x, y, scorer=scorer, workers=WORKERS, dtype=np.float32)
        if scorer is JaroWinkler.normalized_similarity:
            v = v * 100
        if masked:
            v[empty_t if view == "addr_tok" else empty_a] = -1
        cols[name] = v
    del lists
    ab = pl.concat([a.select(pl.all().name.prefix("a_")), b.select(pl.all().name.prefix("b_"))],
                   how="horizontal")
    ntok_a = pl.col("a_name_norm").str.count_matches(" ") + 1
    ntok_b = pl.col("b_name_norm").str.count_matches(" ") + 1
    extra = ab.select(
        (pl.col("a_name_norm") == pl.col("b_name_norm")).cast(pl.Float32).alias("n_exact"),
        ((pl.col("a_core") == pl.col("b_core")) & (pl.col("a_core") != "")).cast(pl.Float32).alias("c_exact"),
        _eq(pl.col("a_postcode"), pl.col("b_postcode")).alias("pc_eq"),
        _eq(pl.col("a_house"), pl.col("b_house")).alias("hs_eq"),
        _eq(pl.col("a_country"), pl.col("b_country")).alias("cty_eq"),
        pl.col("a_name_norm").str.len_chars().cast(pl.Float32).alias("n_len1"),
        pl.col("b_name_norm").str.len_chars().cast(pl.Float32).alias("n_len2"),
        pl.col("a_addr_norm").str.len_chars().cast(pl.Float32).alias("a_len1"),
        pl.col("b_addr_norm").str.len_chars().cast(pl.Float32).alias("a_len2"),
        (pl.min_horizontal(ntok_a, ntok_b) / pl.max_horizontal(ntok_a, ntok_b)).cast(pl.Float32).alias("ntok_r"),
        pl.col("b_src").cast(pl.Float32).alias("src"),
    )
    del ab
    out = pairs.with_columns(
        [pl.Series(k, v) for k, v in cols.items()]
        + extra.get_columns()
        + [((pl.col("kmask").cast(pl.Int32) // (1 << i)) % 2).cast(pl.Float32).alias(f"k{k}")
           for i, k in enumerate(KEYS)]
    ).with_columns(pl.col("bscore").cast(pl.Float32))
    grp = [pl.len().over("s1").cast(pl.Float32).alias("n_cand")]
    for base in GROUP_BASE:
        grp.append((pl.col(base).max().over("s1") - pl.col(base)).alias(f"{base}_gap"))
        grp.append(pl.col(base).rank("min", descending=True).over("s1").cast(pl.Float32).alias(f"{base}_rank"))
    return out.with_columns(grp)


def matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).to_numpy().astype(np.float32, copy=False)
