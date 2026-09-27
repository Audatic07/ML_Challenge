"""Stage 1: cheap learned ranking that shortlists blocking candidates.

Wide blocking (rescue keys, up to 300 per query) finds most true matches but yields
about 130 candidates per query, too many to score with the full feature set on the
whole test set. A small LightGBM on cheap features ranks them; the top SHORTLIST per
query are passed to the final matcher. That shortlist is the candidate_pairs.tsv set.
"""
from __future__ import annotations

import os

import lightgbm as lgb
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .blocking import KEYS
from .config import SEED, WORKERS

SHORTLIST = int(os.environ.get("ER_SHORTLIST", 40))

CHEAP = (["bscore", "nkeys", "n_cand", "bscore_rank", "bscore_gap"] + [f"k{k}" for k in KEYS]
         + ["n_tset", "n_ratio", "sk_tset", "a_tset", "a_ratio", "hs_eq", "pc_eq", "a_empty",
            "n_tset_rank", "a_tset_rank", "sum_rank", "n_tset_gap", "a_tset_gap"])

PARAMS = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "min_data_in_leaf": 200,
          "feature_fraction": 0.9, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": SEED,
          "verbose": -1, "metric": "binary_logloss"}


def _sim(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=WORKERS, dtype=np.float32)


def cheap_features(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    cols = ["name_norm", "skel", "addr_norm", "house", "postcode"]
    a = s1.select(cols).gather(pairs["s1"])
    b = tg.select(cols).gather(pairs["t"])
    an, bn = a["name_norm"].fill_null("").to_list(), b["name_norm"].fill_null("").to_list()
    aa, ba = a["addr_norm"].fill_null("").to_list(), b["addr_norm"].fill_null("").to_list()
    ask, bsk = a["skel"].fill_null("").to_list(), b["skel"].fill_null("").to_list()
    empty = (a["addr_norm"].fill_null("") == "").to_numpy() | (b["addr_norm"].fill_null("") == "").to_numpy()
    a_tset = _sim(aa, ba, fuzz.token_set_ratio)
    a_ratio = _sim(aa, ba, fuzz.ratio)
    a_tset[empty] = -1
    a_ratio[empty] = -1

    def eq(x: pl.Series, y: pl.Series) -> np.ndarray:
        return np.where(x.is_null().to_numpy() | y.is_null().to_numpy(), -1,
                        (x == y).fill_null(False).cast(pl.Int8).to_numpy()).astype(np.float32)

    df = pairs.with_columns(
        pl.Series("n_tset", _sim(an, bn, fuzz.token_set_ratio)),
        pl.Series("n_ratio", _sim(an, bn, fuzz.ratio)),
        pl.Series("sk_tset", _sim(ask, bsk, fuzz.token_set_ratio)),
        pl.Series("a_tset", a_tset), pl.Series("a_ratio", a_ratio),
        pl.Series("hs_eq", eq(a["house"], b["house"])),
        pl.Series("pc_eq", eq(a["postcode"], b["postcode"])),
        pl.Series("a_empty", empty.astype(np.float32)),
        *[((pl.col("kmask").cast(pl.Int32) // (1 << i)) % 2).cast(pl.Float32).alias(f"k{k}")
          for i, k in enumerate(KEYS)],
    )
    df = df.with_columns(pl.sum_horizontal([f"k{k}" for k in KEYS]).alias("nkeys"),
                         pl.col("bscore").cast(pl.Float32))
    rk = lambda c: pl.col(c).rank("min", descending=True).over("s1").cast(pl.Float32)
    df = df.with_columns(
        pl.len().over("s1").cast(pl.Float32).alias("n_cand"),
        rk("bscore").alias("bscore_rank"), rk("n_tset").alias("n_tset_rank"), rk("a_tset").alias("a_tset_rank"),
        (pl.col("bscore").max().over("s1") - pl.col("bscore")).alias("bscore_gap"),
        (pl.col("n_tset").max().over("s1") - pl.col("n_tset")).alias("n_tset_gap"),
        (pl.col("a_tset").max().over("s1") - pl.col("a_tset")).alias("a_tset_gap"),
    )
    return df.with_columns((pl.col("n_tset_rank") + pl.col("a_tset_rank")).alias("sum_rank"))


def train(df: pl.DataFrame, rounds: int = 300) -> lgb.Booster:
    x = df.select(CHEAP).to_numpy().astype(np.float32)
    return lgb.train(dict(PARAMS, num_threads=WORKERS), lgb.Dataset(x, df["y"].to_numpy(), feature_name=CHEAP),
                     num_boost_round=rounds)


def shortlist(df: pl.DataFrame, model: lgb.Booster, n: int = SHORTLIST) -> pl.DataFrame:
    """Keep the n best-ranked candidates per query; returns (s1, t, bscore, kmask, p1)."""
    p = model.predict(df.select(CHEAP).to_numpy().astype(np.float32), num_threads=WORKERS)
    keep = (df.select("s1", "t", "bscore", "kmask").with_columns(pl.Series("p1", p, dtype=pl.Float32))
            .sort(["s1", "p1", "t"], descending=[False, True, False])
            .group_by("s1", maintain_order=True).head(n))
    return keep.with_columns(pl.col("bscore").cast(pl.UInt16), pl.col("kmask").cast(pl.UInt16))
