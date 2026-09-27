"""V7 F2: decoy clusters and token differences (plan revision 2 section 4, source v3 features3.py at 44cb55c).

A decoy is a separate business that copies a real one with a changed name word or a nearby house number. Its
deviation is usually shared by other candidates of the same S1, while noise on true variants is independent per
record. The features are: directional IDF-weighted name/address token coverage, extra and missing token mass (with
typo-level differences, fuzzy ratio >= 75, not counted as true extra/missing words), number and house structure,
and leave-one-out counts of the S1's other survivors sharing a name core, house number, address or extra word.
IDF comes from the full target catalog of the split (label-free). The v3 translit flags are omitted: v5 has no such
view. Missing evidence is -1 as in the source, never agreement.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import Levenshtein

from .text_norm_v5 import normalise

FUZZY_CUTOFF = 75
F2_COLUMNS = [
    "num_jac", "num_q_in_t", "num_t_in_q", "hs_in_t", "h_prefix", "h_lev", "h_absdiff", "h_sortdig_eq", "h_len_a", "h_len_b",
    "nm_cov_q", "nm_cov_t", "nm_extra_idf", "nm_extra_max", "nm_miss_idf", "nm_miss_max",
    "nm_true_extra", "nm_true_extra_max", "nm_true_miss", "nm_true_miss_max",
    "ad_cov_q", "ad_cov_t", "ad_extra_idf", "ad_miss_idf",
    "g_same_core", "g_same_core_goodaddr", "g_same_core_badaddr", "g_same_house", "g_same_addr", "g_extra_support",
    "g_core_eq_q", "g_house_eq_q", "nm_cov_q_rank", "nm_cov_q_gap", "num_jac_rank",
]
VIEW_COLS = ["core", "addr_norm", "addr_tok", "house"]


@dataclass
class DecoyContext:
    idf_name: pl.DataFrame
    idf_addr: pl.DataFrame


def _views(raw: pl.DataFrame) -> pl.DataFrame:
    norm = normalise(raw.with_columns(pl.col(pl.String).fill_null("")))
    return norm.select(pl.col("core").fill_null(""), pl.col("addr_norm").fill_null(""), pl.col("addr_tok").fill_null(""),
                       pl.col("house"))


def _idf(views: pl.DataFrame, col: str) -> pl.DataFrame:
    n = len(views)
    df = (views.select(pl.col(col).str.split(" ").list.unique().alias("tok")).explode("tok")
          .filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).group_by("tok").agg(pl.len().alias("df")))
    return df.select("tok", (math.log(n + 1) - (pl.col("df").cast(pl.Float64) + 1).log()).cast(pl.Float32).alias("idf"))


def build_context(t_raw: pl.DataFrame) -> DecoyContext:
    """IDF over the full target catalog (S2 then S3) of one split."""
    views = _views(t_raw)
    return DecoyContext(idf_name=_idf(views, "core"), idf_addr=_idf(views, "addr_tok"))


def _toks(pid: pl.Series, text: pl.Series) -> pl.DataFrame:
    return (pl.DataFrame({"pid": pid, "tok": text.fill_null("").str.split(" ")})
            .explode("tok").filter(pl.col("tok").is_not_null() & (pl.col("tok") != "")).unique())


def _token_stats(pid, qa, tb, idf, prefix: str, fuzzy: bool, threads: int):
    max_idf = float(idf["idf"].max())
    q = _toks(pid, qa).join(idf, on="tok", how="left").with_columns(pl.col("idf").fill_null(max_idf))
    t = _toks(pid, tb).join(idf, on="tok", how="left").with_columns(pl.col("idf").fill_null(max_idf))
    shared = q.join(t.select("pid", "tok"), on=["pid", "tok"])
    q_only = q.join(t.select("pid", "tok"), on=["pid", "tok"], how="anti")
    t_only = t.join(q.select("pid", "tok"), on=["pid", "tok"], how="anti")
    agg = (pl.DataFrame({"pid": pid})
           .join(q.group_by("pid").agg(pl.col("idf").sum().alias("q_idf")), on="pid", how="left")
           .join(t.group_by("pid").agg(pl.col("idf").sum().alias("t_idf")), on="pid", how="left")
           .join(shared.group_by("pid").agg(pl.col("idf").sum().alias("s_idf")), on="pid", how="left")
           .join(t_only.group_by("pid").agg(pl.col("idf").sum().alias("x_idf"), pl.col("idf").max().alias("x_max")), on="pid", how="left")
           .join(q_only.group_by("pid").agg(pl.col("idf").sum().alias("m_idf"), pl.col("idf").max().alias("m_max")), on="pid", how="left")
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
        cross = cross.with_columns(pl.Series("r", process.cpdist(cross["x"].to_list(), cross["m"].to_list(),
                                                                scorer=fuzz.ratio, workers=threads, dtype=np.float32)))
        best_x = cross.group_by("pid", "x").agg(pl.col("r").max())
        best_m = cross.group_by("pid", "m").agg(pl.col("r").max())
    else:
        best_x = pl.DataFrame(schema={"pid": pl.UInt32, "x": pl.String, "r": pl.Float32})
        best_m = pl.DataFrame(schema={"pid": pl.UInt32, "m": pl.String, "r": pl.Float32})
    true_x = t_only.rename({"tok": "x"}).join(best_x, on=["pid", "x"], how="left").filter(pl.col("r").fill_null(0) < FUZZY_CUTOFF)
    true_m = q_only.rename({"tok": "m"}).join(best_m, on=["pid", "m"], how="left").filter(pl.col("r").fill_null(0) < FUZZY_CUTOFF)
    out = (out.join(true_x.group_by("pid").agg(pl.len().alias(f"{prefix}_true_extra"),
                                               pl.col("idf").max().alias(f"{prefix}_true_extra_max")), on="pid", how="left")
           .join(true_m.group_by("pid").agg(pl.len().alias(f"{prefix}_true_miss"),
                                            pl.col("idf").max().alias(f"{prefix}_true_miss_max")), on="pid", how="left")
           .fill_null(0))
    return out, true_x.select("pid", pl.col("x").alias("tok"))


def _nums(c: pl.Expr) -> pl.Expr:
    return (c.fill_null("").str.extract_all(r"\d+").list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter(pl.element() != "")).list.unique())


def decoy_features(pairs: pl.DataFrame, s1_raw: pl.DataFrame, t_raw: pl.DataFrame, ctx: DecoyContext, a_tset: pl.Series,
                   threads: int) -> pl.DataFrame:
    """F2 for (s1, t) pairs (the survivors of each S1 together): s1, t plus F2_COLUMNS. `a_tset` is the q-vs-t
    address token-set ratio on the 0-100 scale (-1 when missing), aligned with `pairs`."""
    pairs = pairs.select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32)).with_row_index("pid")
    pid = pairs["pid"]
    qrows, trows = pairs["s1"].unique().sort(), pairs["t"].unique().sort()
    qv = _views(s1_raw.gather(qrows)).with_columns(pl.Series("s1", qrows))
    tv = _views(t_raw.gather(trows)).with_columns(pl.Series("t", trows))
    a = pairs.select("s1").join(qv, on="s1", how="left", maintain_order="left")
    b = pairs.select("t").join(tv, on="t", how="left", maintain_order="left")
    nm, extra = _token_stats(pid, a["core"], b["core"], ctx.idf_name, "nm", True, threads)
    ad, _ = _token_stats(pid, a["addr_tok"], b["addr_tok"], ctx.idf_addr, "ad", False, threads)

    x = pl.DataFrame({"pid": pid, "ha": a["house"], "hb": b["house"], "aa": a["addr_norm"], "ab": b["addr_norm"]})
    x = x.with_columns(_nums(pl.col("aa")).alias("na"), _nums(pl.col("ab")).alias("nb"))
    inter = pl.col("na").list.set_intersection("nb").list.len()
    union = pl.col("na").list.set_union("nb").list.len()
    la, lb = pl.col("na").list.len(), pl.col("nb").list.len()
    both = pl.col("ha").is_not_null() & pl.col("hb").is_not_null()

    def sortd(c):
        return c.str.split("").list.sort().list.join("")

    num = x.select(
        "pid",
        pl.when((la == 0) | (lb == 0)).then(-1.0).otherwise(inter / union).alias("num_jac"),
        pl.when(la == 0).then(-1.0).otherwise(inter / la).alias("num_q_in_t"),
        pl.when(lb == 0).then(-1.0).otherwise(inter / lb).alias("num_t_in_q"),
        pl.when(pl.col("ha").is_null() | (lb == 0)).then(-1).otherwise(pl.col("nb").list.contains(pl.col("ha")).cast(pl.Int8)).alias("hs_in_t"),
        pl.when(~both).then(-1).otherwise(
            ((pl.col("ha") != pl.col("hb")) & (pl.col("ha").str.starts_with(pl.col("hb")) | pl.col("hb").str.starts_with(pl.col("ha"))
             | pl.col("ha").str.ends_with(pl.col("hb")) | pl.col("hb").str.ends_with(pl.col("ha")))).cast(pl.Int8)).alias("h_prefix"),
        pl.when(~both).then(-1.0).otherwise((pl.col("ha").cast(pl.Int64, strict=False) - pl.col("hb").cast(pl.Int64, strict=False))
                                            .abs().cast(pl.Float64).log1p()).alias("h_absdiff"),
        pl.when(~both).then(-1).otherwise(((pl.col("ha") != pl.col("hb")) & (sortd(pl.col("ha")) == sortd(pl.col("hb")))).cast(pl.Int8)).alias("h_sortdig_eq"),
        pl.col("ha").str.len_chars().fill_null(0).alias("h_len_a"),
        pl.col("hb").str.len_chars().fill_null(0).alias("h_len_b"))
    hl = process.cpdist(x["ha"].fill_null("").to_list(), x["hb"].fill_null("").to_list(), scorer=Levenshtein.distance,
                        workers=threads, dtype=np.float32)
    num = num.with_columns(pl.Series("h_lev", np.where(x["ha"].is_null().to_numpy() | x["hb"].is_null().to_numpy(), -1, hl)))

    # Leave-one-out evidence within each S1's candidate set.
    g = pl.DataFrame({"pid": pid, "s1": pairs["s1"], "tc": b["core"], "qc": a["core"], "th": b["house"], "qh": a["house"],
                      "ta": b["addr_norm"], "a_tset": a_tset.cast(pl.Float32)})
    good, bad = pl.col("a_tset") >= 80, (pl.col("a_tset") >= 0) & (pl.col("a_tset") < 60)
    grp = g.select(
        "pid",
        pl.when(pl.col("tc") != "").then(pl.len().over("s1", "tc") - 1).otherwise(0).alias("g_same_core"),
        pl.when(pl.col("tc") != "").then(good.sum().over("s1", "tc") - good.cast(pl.Int64)).otherwise(0).alias("g_same_core_goodaddr"),
        pl.when(pl.col("tc") != "").then(bad.sum().over("s1", "tc") - bad.cast(pl.Int64)).otherwise(0).alias("g_same_core_badaddr"),
        pl.when(pl.col("th").is_not_null()).then(pl.len().over("s1", "th") - 1).otherwise(-1).alias("g_same_house"),
        pl.when(pl.col("ta") != "").then(pl.len().over("s1", "ta") - 1).otherwise(-1).alias("g_same_addr"),
        ((pl.col("tc") == pl.col("qc")) & (pl.col("tc") != "")).sum().over("s1").alias("g_core_eq_q"),
        (pl.col("th") == pl.col("qh")).fill_null(False).sum().over("s1").alias("g_house_eq_q"))
    if extra is not None and len(extra):
        sup = (extra.join(g.select("pid", "s1"), on="pid").with_columns((pl.len().over("s1", "tok") - 1).alias("n"))
               .group_by("pid").agg(pl.col("n").max().alias("g_extra_support")))
        grp = grp.join(sup, on="pid", how="left").with_columns(pl.col("g_extra_support").fill_null(-1))
    else:
        grp = grp.with_columns(pl.lit(-1).alias("g_extra_support"))
    res = pairs.join(nm, on="pid").join(ad, on="pid").join(num, on="pid").join(grp, on="pid").sort("pid")
    res = res.with_columns(pl.col("nm_cov_q").rank("min", descending=True).over("s1").alias("nm_cov_q_rank"),
                           (pl.col("nm_cov_q").max().over("s1") - pl.col("nm_cov_q")).alias("nm_cov_q_gap"),
                           pl.col("num_jac").rank("min", descending=True).over("s1").alias("num_jac_rank"))
    out = res.select("s1", "t", *[pl.col(c).cast(pl.Float32) for c in F2_COLUMNS])
    if len(out) != len(pairs):
        raise ValueError("decoy features do not cover every pair exactly once")
    return out
