"""V7 F3: coherent whole-population rivals (plan revision 3, section 4). Label-free.

Every labelled target has exactly one owner, so the deciding evidence for many residual errors is whether some
*other* S1 record resembles the target better than q does. The population is every S1 row of the split. Six
normalized views give exact-match keys `country \\x1f view \\x1f text` (text of at least 5 characters). A key with more
than 20 S1 is an overflow key: it is not expanded, and `t_overflow` records the censored search. R(t) is the union of
the postings of t's other keys. For each pair (q, t) the rivals are R(t) minus q, and three rivals are kept apart:
J (best joint name/address similarity), N (best name) and A (best address). Ties go to the lower S1 row. Missing
evidence is NaN, never 0 and never agreement. Truth is never opened here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from .text_norm_v5 import normalise

VIEWS = ("core", "core_sorted", "concat", "skel", "addr_norm", "house_addr")
MAX_POSTINGS = 20
MIN_LEN = 5
SIMS = ("n_tset", "n_ratio", "a_tset", "a_ratio", "joint")
F3_COLUMNS = [
    *[f"o_{s}" for s in SIMS],
    *[f"rJ_{s}" for s in SIMS], "rJ_present", "gap_joint", "gap_J_name", "gap_J_addr",
    "rN_n_ratio", "rN_n_tset", "rN_a_ratio", "rN_present", "gap_N_name", "rN_is_J",
    "rA_a_ratio", "rA_a_tset", "rA_n_ratio", "rA_present", "gap_A_addr", "rA_is_J",
    "t_overflow", "t_n_keys", "q_addr_missing", "t_addr_missing",
]


@dataclass
class PopulationIndex:
    views: pl.DataFrame     # s1 row -> core, concat, addr_norm (similarity inputs)
    keys: pl.DataFrame      # every (key, s1) of the population, sorted
    postings: pl.DataFrame  # (key, s1) of keys with at most MAX_POSTINGS S1
    overflow: pl.DataFrame  # keys with more than MAX_POSTINGS S1


def _views(raw: pl.DataFrame) -> pl.DataFrame:
    norm = normalise(raw.with_columns(pl.col(pl.String).fill_null("")))
    return norm.select(
        pl.col("country").fill_null(""),
        pl.col("core").fill_null(""),
        pl.col("core").fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.sort()
        .list.join(" ").alias("core_sorted"),
        pl.col("concat").fill_null(""), pl.col("skel").fill_null(""), pl.col("addr_norm").fill_null(""),
        pl.when((pl.col("house").fill_null("") != "") & (pl.col("addr_tok").fill_null("") != ""))
        .then(pl.col("house") + " " + pl.col("addr_tok")).otherwise(pl.lit("")).alias("house_addr"))


def _keys(views: pl.DataFrame, id_col: str) -> pl.DataFrame:
    parts = [views.select(id_col, (pl.col("country") + "\x1f" + view + "\x1f" + pl.col(view)).alias("key"),
                          pl.col(view).str.len_chars().alias("_n")).filter(pl.col("_n") >= MIN_LEN).drop("_n")
             for view in VIEWS]
    return pl.concat(parts).unique()


def build_index(s1_raw: pl.DataFrame) -> PopulationIndex:
    views = _views(s1_raw).with_row_index("s1").with_columns(pl.col("s1").cast(pl.UInt32))
    keys = _keys(views, "s1").sort("key", "s1")
    counts = keys.group_by("key").len()
    small = counts.filter(pl.col("len") <= MAX_POSTINGS).select("key")
    return PopulationIndex(views=views.select("s1", "core", "concat", "addr_norm"), keys=keys,
                           postings=keys.join(small, on="key"), overflow=counts.filter(pl.col("len") > MAX_POSTINGS).select("key"))


def index_digest(index: PopulationIndex) -> str:
    """SHA-256 of the sorted (key, s1_row) stream, one `key<TAB>row` line each."""
    h = hashlib.sha256()
    lines = index.keys.select(pl.concat_str([pl.col("key"), pl.col("s1").cast(pl.String)], separator="\t"))["key"]
    for start in range(0, len(lines), 1_000_000):
        h.update(("\n".join(lines.slice(start, 1_000_000).to_list()) + "\n").encode())
    return h.hexdigest()


def _sims(a: pl.DataFrame, b: pl.DataFrame, threads: int) -> dict[str, np.ndarray]:
    """Similarities of aligned rows a[i] vs b[i], scaled to [0, 1]; NaN where either side is empty."""
    def cp(x, y, scorer):
        return process.cpdist(a[x].to_list(), b[y].to_list(), scorer=scorer, workers=threads, dtype=np.float32) / 100

    no_name = ((a["core"] == "") | (b["core"] == "")).to_numpy()
    no_concat = ((a["concat"] == "") | (b["concat"] == "")).to_numpy()
    no_addr = ((a["addr_norm"] == "") | (b["addr_norm"] == "")).to_numpy()
    out = {"n_tset": cp("core", "core", fuzz.token_set_ratio), "n_ratio": cp("concat", "concat", fuzz.ratio),
           "a_tset": cp("addr_norm", "addr_norm", fuzz.token_set_ratio), "a_ratio": cp("addr_norm", "addr_norm", fuzz.ratio)}
    out["n_tset"][no_name] = np.nan
    out["n_ratio"][no_concat] = np.nan
    out["a_tset"][no_addr] = np.nan
    out["a_ratio"][no_addr] = np.nan
    out["joint"] = np.sqrt(out["n_ratio"] * out["a_ratio"])  # NaN without a valid name or address
    return out


def _best_two(rivals: pl.DataFrame, by: str, keep: list[str]) -> pl.DataFrame:
    """Per target, the two best rivals by `by` (non-NaN), ties to the lower S1 row: t, r1, r2 and their `keep` values."""
    ranked = (rivals.filter(pl.col(by).is_not_nan()).sort(["t", by, "r"], descending=[False, True, False])
              .group_by("t", maintain_order=True).head(2).with_columns(pl.int_range(pl.len()).over("t").alias("_k")))
    cols = ["r", *keep]
    first = ranked.filter(pl.col("_k") == 0).select("t", *[pl.col(c).alias(f"{c}_1") for c in cols])
    second = ranked.filter(pl.col("_k") == 1).select("t", *[pl.col(c).alias(f"{c}_2") for c in cols])
    return first.join(second, on="t", how="left")


def _pick(pairs: pl.DataFrame, best: pl.DataFrame, keep: list[str], prefix: str) -> pl.DataFrame:
    """For each (s1, t): the best rival other than s1 itself."""
    x = pairs.join(best, on="t", how="left", maintain_order="left")
    self_first = pl.col("r_1") == pl.col("s1")
    out = [pl.when(self_first).then(pl.col("r_2")).otherwise(pl.col("r_1")).alias(f"{prefix}_r")]
    out += [pl.when(self_first).then(pl.col(f"{c}_2")).otherwise(pl.col(f"{c}_1")).alias(f"{prefix}_{c}") for c in keep]
    return x.select("s1", "t", *out)


def rival_features(pairs: pl.DataFrame, s1_raw: pl.DataFrame, t_raw: pl.DataFrame, index: PopulationIndex,
                   threads: int) -> pl.DataFrame:
    """F3 for the (s1, t) pairs of one split: s1, t plus F3_COLUMNS in that order."""
    pairs = pairs.select(pl.col("s1").cast(pl.UInt32), pl.col("t").cast(pl.UInt32))
    tu = pairs["t"].unique().sort()
    tv = _views(t_raw.gather(tu)).with_columns(pl.Series("t", tu))
    tk = _keys(tv, "t")
    n_keys = tk.group_by("t").len().rename({"len": "t_n_keys"})
    over = tk.join(index.overflow, on="key", how="semi").select("t").unique().with_columns(pl.lit(1.0).alias("t_overflow"))
    rivals = tk.join(index.postings, on="key").select("t", pl.col("s1").alias("r")).unique().sort("t", "r")
    tsel = tv.select("t", "core", "concat", "addr_norm")
    a = rivals.select("t").join(tsel, on="t", how="left", maintain_order="left")
    b = rivals.select(pl.col("r").alias("s1")).join(index.views, on="s1", how="left", maintain_order="left")
    rivals = rivals.with_columns([pl.Series(k, v) for k, v in _sims(a, b, threads).items()])
    # Own similarities q vs t, same scorers and views.
    qa = pairs.select("t").join(tsel, on="t", how="left", maintain_order="left")
    qb = pairs.select("s1").join(index.views, on="s1", how="left", maintain_order="left")
    own = _sims(qa, qb, threads)
    feat = pairs.with_columns([pl.Series(f"o_{k}", v) for k, v in own.items()])
    j = _pick(pairs, _best_two(rivals, "joint", list(SIMS)), list(SIMS), "rJ")
    n = _pick(pairs, _best_two(rivals, "n_ratio", ["n_ratio", "n_tset", "a_ratio"]), ["n_ratio", "n_tset", "a_ratio"], "rN")
    a_ = _pick(pairs, _best_two(rivals, "a_ratio", ["a_ratio", "a_tset", "n_ratio"]), ["a_ratio", "a_tset", "n_ratio"], "rA")
    feat = (feat.join(j, on=["s1", "t"], how="left", maintain_order="left")
            .join(n, on=["s1", "t"], how="left", maintain_order="left")
            .join(a_, on=["s1", "t"], how="left", maintain_order="left")
            .join(n_keys, on="t", how="left", maintain_order="left").join(over, on="t", how="left", maintain_order="left"))
    f32 = pl.Float32
    feat = feat.with_columns(
        pl.col("rJ_r").is_not_null().cast(f32).alias("rJ_present"),
        pl.col("rN_r").is_not_null().cast(f32).alias("rN_present"),
        pl.col("rA_r").is_not_null().cast(f32).alias("rA_present"),
        (pl.col("o_joint") - pl.col("rJ_joint")).alias("gap_joint"),
        (pl.col("o_n_ratio") - pl.col("rJ_n_ratio")).alias("gap_J_name"),
        (pl.col("o_a_ratio") - pl.col("rJ_a_ratio")).alias("gap_J_addr"),
        (pl.col("o_n_ratio") - pl.col("rN_n_ratio")).alias("gap_N_name"),
        (pl.col("o_a_ratio") - pl.col("rA_a_ratio")).alias("gap_A_addr"),
        pl.when(pl.col("rN_r").is_null() | pl.col("rJ_r").is_null()).then(None)
        .otherwise((pl.col("rN_r") == pl.col("rJ_r")).cast(f32)).alias("rN_is_J"),
        pl.when(pl.col("rA_r").is_null() | pl.col("rJ_r").is_null()).then(None)
        .otherwise((pl.col("rA_r") == pl.col("rJ_r")).cast(f32)).alias("rA_is_J"),
        pl.col("t_overflow").fill_null(0.0), pl.col("t_n_keys").fill_null(0).cast(f32),
        pl.col("o_a_ratio").is_nan().cast(f32).alias("_addr_nan"))
    q_addr = pairs.select("s1").join(index.views.select("s1", "addr_norm"), on="s1", how="left", maintain_order="left")["addr_norm"]
    t_addr = pairs.select("t").join(tsel.select("t", "addr_norm"), on="t", how="left", maintain_order="left")["addr_norm"]
    feat = feat.with_columns(pl.Series("q_addr_missing", (q_addr == "").cast(f32)),
                             pl.Series("t_addr_missing", (t_addr == "").cast(f32)))
    out = feat.select("s1", "t", *[pl.col(c).cast(f32).fill_null(float("nan")) for c in F3_COLUMNS])
    if len(out) != len(pairs):
        raise ValueError("rival features do not cover every pair exactly once")
    return out
