"""v3 pair features: v1/v2 features plus evidence aimed at the two measured error types.

1. Heavily-noised true variants were rejected (largest loss). True variants often have
   truncated / off-by-one house numbers, extra filler words, typos, and transliterated
   names (handled by translit.py before this step).
2. Decoys: near-copies of a real business with a changed name word and a nearby house
   number. A decoy is a separate entity, so its deviation (odd house number, odd name
   word) is usually shared by other candidate records of the same query, while noise
   on true variants is independent per record. Group features count that support.

Everything is computed from the supplied records; IDF comes from the target catalog of
the same split.
"""
from __future__ import annotations

import math

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from . import features as base
from .config import WORKERS

NEW_FEATURES = [
    # numbers / house
    "num_jac", "num_q_in_t", "num_t_in_q", "hs_in_t", "h_prefix", "h_lev", "h_absdiff",
    "h_sortdig_eq", "h_len_a", "h_len_b",
    # IDF-weighted name tokens (core)
    "nm_cov_q", "nm_cov_t", "nm_extra_idf", "nm_extra_max", "nm_miss_idf", "nm_miss_max",
    "nm_true_extra", "nm_true_extra_max", "nm_true_miss", "nm_true_miss_max",
    # IDF-weighted address tokens
    "ad_cov_q", "ad_cov_t", "ad_extra_idf", "ad_miss_idf",
    # record flags
    "t_translit", "t_addr_empty", "q_translit",
    # group / decoy-cluster evidence within the query's candidate set
    "g_same_core", "g_same_core_goodaddr", "g_same_core_badaddr", "g_same_house",
    "g_same_addr", "g_extra_support", "g_core_eq_q", "g_house_eq_q",
    "nm_cov_q_rank", "nm_cov_q_gap", "num_jac_rank",
]


def idf_table(tg: pl.DataFrame, col: str) -> pl.DataFrame:
    n = len(tg)
    df = (tg.select(pl.col(col).str.split(" ").list.unique().alias("tok")).explode("tok")
          .filter(pl.col("tok").is_not_null() & (pl.col("tok") != ""))
          .group_by("tok").agg(pl.len().alias("df")))
    return df.select("tok", (math.log(n + 1) - pl.col("df").cast(pl.Float64).add(1).log()).cast(pl.Float32).alias("idf"))


def _toks(pid: pl.Series, text: pl.Series) -> pl.DataFrame:
    return (pl.DataFrame({"pid": pid, "tok": text.fill_null("").str.split(" ")})
            .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique())


def _token_stats(pid: pl.Series, qa: pl.Series, tb: pl.Series, idf: pl.DataFrame, max_idf: float,
                 prefix: str, fuzzy: bool) -> pl.DataFrame:
    q = _toks(pid, qa).join(idf, on="tok", how="left").with_columns(pl.col("idf").fill_null(max_idf))
    t = _toks(pid, tb).join(idf, on="tok", how="left").with_columns(pl.col("idf").fill_null(max_idf))
    shared = q.join(t.select("pid", "tok"), on=["pid", "tok"])
    q_only = q.join(t.select("pid", "tok"), on=["pid", "tok"], how="anti")
    t_only = t.join(q.select("pid", "tok"), on=["pid", "tok"], how="anti")
    agg = (pl.DataFrame({"pid": pid})
           .join(q.group_by("pid").agg(pl.col("idf").sum().alias("q_idf")), on="pid", how="left")
           .join(t.group_by("pid").agg(pl.col("idf").sum().alias("t_idf")), on="pid", how="left")
           .join(shared.group_by("pid").agg(pl.col("idf").sum().alias("s_idf")), on="pid", how="left")
           .join(t_only.group_by("pid").agg(pl.col("idf").sum().alias("x_idf"), pl.col("idf").max().alias("x_max")),
                 on="pid", how="left")
           .join(q_only.group_by("pid").agg(pl.col("idf").sum().alias("m_idf"), pl.col("idf").max().alias("m_max")),
                 on="pid", how="left")
           .fill_null(0))
    out = agg.select(
        "pid",
        pl.when(pl.col("q_idf") > 0).then(pl.col("s_idf") / pl.col("q_idf")).otherwise(-1).alias(f"{prefix}_cov_q"),
        pl.when(pl.col("t_idf") > 0).then(pl.col("s_idf") / pl.col("t_idf")).otherwise(-1).alias(f"{prefix}_cov_t"),
        pl.col("x_idf").alias(f"{prefix}_extra_idf"), pl.col("x_max").alias(f"{prefix}_extra_max"),
        pl.col("m_idf").alias(f"{prefix}_miss_idf"), pl.col("m_max").alias(f"{prefix}_miss_max"))
    if not fuzzy:
        return out, None
    # An extra token that is a typo of a missing token is noise, not a different word.
    cross = t_only.rename({"tok": "x", "idf": "xi"}).join(q_only.rename({"tok": "m", "idf": "mi"}), on="pid")
    if len(cross):
        cross = cross.with_columns(pl.Series("r", process.cpdist(
            cross["x"].to_list(), cross["m"].to_list(), scorer=fuzz.ratio, workers=WORKERS, dtype=np.float32)))
        best_x = cross.group_by("pid", "x").agg(pl.col("r").max())
        best_m = cross.group_by("pid", "m").agg(pl.col("r").max())
    else:
        best_x = pl.DataFrame(schema={"pid": pl.UInt32, "x": pl.String, "r": pl.Float32})
        best_m = pl.DataFrame(schema={"pid": pl.UInt32, "m": pl.String, "r": pl.Float32})
    true_x = (t_only.rename({"tok": "x"}).join(best_x, on=["pid", "x"], how="left")
              .filter(pl.col("r").fill_null(0) < 75))
    true_m = (q_only.rename({"tok": "m"}).join(best_m, on=["pid", "m"], how="left")
              .filter(pl.col("r").fill_null(0) < 75))
    out = (out.join(true_x.group_by("pid").agg(pl.len().alias(f"{prefix}_true_extra"),
                                               pl.col("idf").max().alias(f"{prefix}_true_extra_max")), on="pid", how="left")
           .join(true_m.group_by("pid").agg(pl.len().alias(f"{prefix}_true_miss"),
                                            pl.col("idf").max().alias(f"{prefix}_true_miss_max")), on="pid", how="left")
           .fill_null(0))
    return out, true_x.select("pid", pl.col("x").alias("tok"))


def _nums(c: pl.Expr) -> pl.Expr:
    return (c.fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def build(pairs: pl.DataFrame, s1: pl.DataFrame, tg: pl.DataFrame, idf_name: pl.DataFrame,
          idf_addr: pl.DataFrame) -> pl.DataFrame:
    out = base.build(pairs, s1, tg).with_row_index("pid")
    pid = out["pid"]
    cols = ["core", "addr_norm", "addr_tok", "house", "translit"]
    a = s1.select(cols).gather(out["s1"])
    b = tg.select(cols).gather(out["t"])
    max_n = float(idf_name["idf"].max())
    max_a = float(idf_addr["idf"].max())
    nm, extra_toks = _token_stats(pid, a["core"], b["core"], idf_name, max_n, "nm", True)
    ad, _ = _token_stats(pid, a["addr_tok"], b["addr_tok"], idf_addr, max_a, "ad", False)

    x = pl.DataFrame({"pid": pid, "ha": a["house"], "hb": b["house"], "aa": a["addr_norm"], "ab": b["addr_norm"]})
    x = x.with_columns(_nums(pl.col("aa")).alias("na"), _nums(pl.col("ab")).alias("nb"))
    inter = pl.col("na").list.set_intersection("nb").list.len()
    union = pl.col("na").list.set_union("nb").list.len()
    la, lb = pl.col("na").list.len(), pl.col("nb").list.len()
    both = pl.col("ha").is_not_null() & pl.col("hb").is_not_null()
    sortd = lambda c: c.str.split("").list.sort().list.join("")
    num = x.select(
        "pid",
        pl.when((la == 0) | (lb == 0)).then(-1.0).otherwise(inter / union).alias("num_jac"),
        pl.when(la == 0).then(-1.0).otherwise(inter / la).alias("num_q_in_t"),
        pl.when(lb == 0).then(-1.0).otherwise(inter / lb).alias("num_t_in_q"),
        pl.when(pl.col("ha").is_null() | (lb == 0)).then(-1).otherwise(
            pl.col("nb").list.contains(pl.col("ha")).cast(pl.Int8)).alias("hs_in_t"),
        pl.when(~both).then(-1).otherwise(
            ((pl.col("ha") != pl.col("hb")) & (pl.col("ha").str.starts_with(pl.col("hb"))
             | pl.col("hb").str.starts_with(pl.col("ha")) | pl.col("ha").str.ends_with(pl.col("hb"))
             | pl.col("hb").str.ends_with(pl.col("ha")))).cast(pl.Int8)).alias("h_prefix"),
        pl.when(~both).then(-1.0).otherwise(
            (pl.col("ha").cast(pl.Int64, strict=False) - pl.col("hb").cast(pl.Int64, strict=False))
            .abs().cast(pl.Float64).log1p()).alias("h_absdiff"),
        pl.when(~both).then(-1).otherwise(
            ((pl.col("ha") != pl.col("hb")) & (sortd(pl.col("ha")) == sortd(pl.col("hb")))).cast(pl.Int8)).alias("h_sortdig_eq"),
        pl.col("ha").str.len_chars().fill_null(0).alias("h_len_a"),
        pl.col("hb").str.len_chars().fill_null(0).alias("h_len_b"),
    )
    hl = process.cpdist(x["ha"].fill_null("").to_list(), x["hb"].fill_null("").to_list(),
                        scorer=Levenshtein.distance, workers=WORKERS, dtype=np.float32)
    num = num.with_columns(pl.Series("h_lev", np.where(x["ha"].is_null().to_numpy() | x["hb"].is_null().to_numpy(), -1, hl)))

    flags = pl.DataFrame({"pid": pid,
                          "t_translit": b["translit"].cast(pl.Float32),
                          "q_translit": a["translit"].cast(pl.Float32),
                          "t_addr_empty": (b["addr_norm"].fill_null("") == "").cast(pl.Float32)})

    # group evidence within each query's candidate set
    g = pl.DataFrame({"pid": pid, "s1": out["s1"], "tc": b["core"].fill_null(""), "qc": a["core"].fill_null(""),
                      "th": b["house"], "qh": a["house"], "ta": b["addr_norm"].fill_null(""),
                      "a_tset": out["a_tset"]})
    grp = g.select(
        "pid",
        pl.when(pl.col("tc") != "").then(pl.len().over("s1", "tc") - 1).otherwise(0).alias("g_same_core"),
        pl.when(pl.col("tc") != "").then((pl.col("a_tset") >= 80).sum().over("s1", "tc")
                                         - (pl.col("a_tset") >= 80).cast(pl.Int64)).otherwise(0).alias("g_same_core_goodaddr"),
        pl.when(pl.col("tc") != "").then(((pl.col("a_tset") >= 0) & (pl.col("a_tset") < 60)).sum().over("s1", "tc")
                                         - ((pl.col("a_tset") >= 0) & (pl.col("a_tset") < 60)).cast(pl.Int64))
        .otherwise(0).alias("g_same_core_badaddr"),
        pl.when(pl.col("th").is_not_null()).then(pl.len().over("s1", "th") - 1).otherwise(-1).alias("g_same_house"),
        pl.when(pl.col("ta") != "").then(pl.len().over("s1", "ta") - 1).otherwise(-1).alias("g_same_addr"),
        (pl.col("tc") == pl.col("qc")).sum().over("s1").alias("g_core_eq_q"),
        (pl.col("th") == pl.col("qh")).fill_null(False).sum().over("s1").alias("g_house_eq_q"),
    )
    if extra_toks is not None and len(extra_toks):
        sup = (extra_toks.join(g.select("pid", "s1"), on="pid")
               .with_columns((pl.len().over("s1", "tok") - 1).alias("n"))
               .group_by("pid").agg(pl.col("n").max().alias("g_extra_support")))
        grp = grp.join(sup, on="pid", how="left").with_columns(pl.col("g_extra_support").fill_null(-1))
    else:
        grp = grp.with_columns(pl.lit(-1).alias("g_extra_support"))

    res = out.join(nm, on="pid").join(ad, on="pid").join(num, on="pid").join(flags, on="pid").join(grp, on="pid")
    res = res.with_columns(
        pl.col("nm_cov_q").rank("min", descending=True).over("s1").alias("nm_cov_q_rank"),
        (pl.col("nm_cov_q").max().over("s1") - pl.col("nm_cov_q")).alias("nm_cov_q_gap"),
        pl.col("num_jac").rank("min", descending=True).over("s1").alias("num_jac_rank"),
    )
    return res.with_columns([pl.col(c).cast(pl.Float32) for c in NEW_FEATURES]).drop("pid")


FEATURES = base.FEATURES + NEW_FEATURES


def matrix(df: pl.DataFrame) -> np.ndarray:
    return df.select(FEATURES).to_numpy().astype(np.float32, copy=False)
