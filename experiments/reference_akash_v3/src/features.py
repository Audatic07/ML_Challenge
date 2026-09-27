"""Pairwise features for (Source 1, Source 2/3) candidate pairs.

String similarities are computed with rapidfuzz.process.cpdist, which scores
aligned lists element-wise in C++ across several threads, so tens of millions
of pairs stay practical on a CPU instance.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from . import config
from .blocking import KT_NAMES

TEXT_COLS = ["name_norm", "core", "skel", "concat", "addr_norm", "addr_tok", "postcode", "house", "country"]


def _sim(a, b, scorer):
    return cpdist(a, b, scorer=scorer, workers=config.N_WORKERS, dtype=np.float32)


def _eq(a, b):
    """1 if both non-empty and equal, 0 if both non-empty and different, -1 if either missing."""
    a, b = np.asarray(a, dtype=object), np.asarray(b, dtype=object)
    present = (a != "") & (b != "")
    return np.where(present, (a == b).astype(np.int8), -1).astype(np.int8)


def pair_features(pairs, s1, s23):
    """Add feature columns to a candidate frame (idx1, idx2, bscore, nkeys, kbits)."""
    L = s1.select(TEXT_COLS)[pairs["idx1"].to_numpy()]
    R = s23.select(TEXT_COLS + ["src"])[pairs["idx2"].to_numpy()]
    l = {c: L[c].to_list() for c in TEXT_COLS}
    r = {c: R[c].to_list() for c in TEXT_COLS}

    f = {
        "n_ratio": _sim(l["name_norm"], r["name_norm"], fuzz.ratio),
        "n_tset": _sim(l["name_norm"], r["name_norm"], fuzz.token_set_ratio),
        "n_tsort": _sim(l["name_norm"], r["name_norm"], fuzz.token_sort_ratio),
        "n_partial": _sim(l["name_norm"], r["name_norm"], fuzz.partial_ratio),
        "c_ratio": _sim(l["core"], r["core"], fuzz.ratio),
        "c_jw": _sim(l["core"], r["core"], JaroWinkler.normalized_similarity) * 100,
        "s_ratio": _sim(l["skel"], r["skel"], fuzz.ratio),
        "s_tset": _sim(l["skel"], r["skel"], fuzz.token_set_ratio),
        "cc_ratio": _sim(l["concat"], r["concat"], fuzz.ratio),
        "cc_partial": _sim(l["concat"], r["concat"], fuzz.partial_ratio),
    }
    a_l = np.array([x != "" for x in l["addr_norm"]])
    a_r = np.array([x != "" for x in r["addr_norm"]])
    both = a_l & a_r
    for name, col, scorer in [("a_tset", "addr_norm", fuzz.token_set_ratio),
                              ("a_ratio", "addr_norm", fuzz.ratio),
                              ("a_partial", "addr_norm", fuzz.partial_ratio),
                              ("at_tset", "addr_tok", fuzz.token_set_ratio)]:
        f[name] = np.where(both, _sim(l[col], r[col], scorer), -1).astype(np.float32)
    f["pc_eq"] = _eq(l["postcode"], r["postcode"])
    f["hs_eq"] = _eq(l["house"], r["house"])
    f["country_eq"] = (np.asarray(l["country"], dtype=object) == np.asarray(r["country"], dtype=object)).astype(np.int8)
    f["a_miss_l"] = (~a_l).astype(np.int8)
    f["a_miss_r"] = (~a_r).astype(np.int8)
    f["len_l"] = np.array([len(x) for x in l["core"]], dtype=np.int16)
    f["len_r"] = np.array([len(x) for x in r["core"]], dtype=np.int16)
    f["ntok_l"] = np.array([x.count(" ") + 1 if x else 0 for x in l["core"]], dtype=np.int8)
    f["ntok_r"] = np.array([x.count(" ") + 1 if x else 0 for x in r["core"]], dtype=np.int8)
    f["src"] = R["src"].to_numpy()

    out = pairs.with_columns(**{k: pl.Series(v) for k, v in f.items()})
    out = out.with_columns(
        [((pl.col("kbits") & (1 << i)) > 0).cast(pl.Int8).alias(f"kb_{k}") for i, k in enumerate(KT_NAMES)])
    # Group features: how this candidate compares to the other candidates of the same Source 1 entity
    g = "idx1"
    return out.with_columns(
        pl.len().over(g).cast(pl.Int16).alias("n_cand"),
        pl.col("n_tset").rank("min", descending=True).over(g).cast(pl.Int16).alias("n_tset_rank"),
        (pl.col("n_tset") - pl.col("n_tset").max().over(g)).alias("n_tset_gap"),
        (pl.col("a_tset") - pl.col("a_tset").max().over(g)).alias("a_tset_gap"),
        pl.col("bscore").rank("min", descending=True).over(g).cast(pl.Int16).alias("bscore_rank"),
        (pl.col("bscore") - pl.col("bscore").max().over(g)).alias("bscore_gap"),
    )


FEATURES = [
    "bscore", "nkeys", "n_ratio", "n_tset", "n_tsort", "n_partial", "c_ratio", "c_jw", "s_ratio",
    "s_tset", "cc_ratio", "cc_partial", "a_tset", "a_ratio", "a_partial", "at_tset", "pc_eq",
    "hs_eq", "country_eq", "a_miss_l", "a_miss_r", "len_l", "len_r", "ntok_l", "ntok_r", "src",
    *[f"kb_{k}" for k in KT_NAMES],
    "n_cand", "n_tset_rank", "n_tset_gap", "a_tset_gap", "bscore_rank", "bscore_gap",
]
