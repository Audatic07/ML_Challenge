"""v6 cascade: stage-1 model ranking cut, then a sibling-aware stage-2 matcher.

Stage 1 is the released v5.1 LightGBM pair model. It scores every retrieved candidate
(about 256 per S1) and keeps the top K per S1 by its probability. Those K survivors are
the candidate set: the stage-2 model scores exactly them, and nothing else.

Stage 2 adds evidence that a pair model cannot see. Most S1 entities have several true
variants among their candidates, and the variants agree with each other. A variant whose
name was replaced by an unrelated word still shares the address of its confident
siblings. A variant without an address still shares their name. A near-duplicate
negative (next house number, one name word changed) usually disagrees with all of them.
Features therefore compare each survivor with the other survivors, weighted by their
stage-1 probability. Every feature uses only the S1 record, the K survivors and their
stage-1 scores; no labels or external data.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

TEXT = ["core", "concat", "name_norm", "addr_norm", "addr_tok", "house", "postcode"]
SIBLINGS = 5  # other survivors compared with each candidate (highest stage-1 p first)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).astype(np.float32)


def survivors(scored: pl.DataFrame, k: int, floor: float = 0.0) -> pl.DataFrame:
    """Top-k candidates of each S1 by stage-1 p (ties broken by target row), dropping
    candidates below `floor`. An S1 whose candidates are all below the floor keeps none."""
    top = (scored.sort(["s1", "p", "t"], descending=[False, True, False])
           .group_by("s1", maintain_order=True).head(k))
    if floor > 0:
        top = top.filter(pl.col("p") >= floor)
    return top.with_columns(pl.int_range(pl.len()).over("s1").cast(pl.UInt8).alias("rank"))


def _sim(a, b, scorer, workers):
    return process.cpdist(a, b, scorer=scorer, workers=workers, dtype=np.float32)


def _eq(a: pl.Series, b: pl.Series) -> np.ndarray:
    a, b = a.fill_null(""), b.fill_null("")
    out = (a == b).cast(pl.Float32).to_numpy().copy()
    out[((a == "") | (b == "")).to_numpy()] = -1
    return out


def _tokens(s: pl.Series) -> pl.Series:
    return s.fill_null("").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique()


def pair_block(a: pl.DataFrame, b: pl.DataFrame, workers: int, prefix: str) -> dict[str, np.ndarray]:
    """Text similarities of aligned record pairs a[i] vs b[i] (normalized views)."""
    na, nb = a["core"].fill_null("").to_list(), b["core"].fill_null("").to_list()
    ca, cb = a["concat"].fill_null("").to_list(), b["concat"].fill_null("").to_list()
    aa, ab = a["addr_norm"].fill_null("").to_list(), b["addr_norm"].fill_null("").to_list()
    ta, tb = a["addr_tok"].fill_null("").to_list(), b["addr_tok"].fill_null("").to_list()
    no_addr = ((a["addr_norm"].fill_null("") == "") | (b["addr_norm"].fill_null("") == "")).to_numpy()
    no_name = ((a["core"].fill_null("") == "") | (b["core"].fill_null("") == "")).to_numpy()
    out = {
        "n_tset": _sim(na, nb, fuzz.token_set_ratio, workers),
        "n_tsort": _sim(na, nb, fuzz.token_sort_ratio, workers),
        "cc_ratio": _sim(ca, cb, fuzz.ratio, workers),
        "cc_jw": 100 * _sim(ca, cb, JaroWinkler.normalized_similarity, workers),
        "a_tset": _sim(aa, ab, fuzz.token_set_ratio, workers),
        "a_ratio": _sim(aa, ab, fuzz.ratio, workers),
        "at_tset": _sim(ta, tb, fuzz.token_set_ratio, workers),
        "n_eq": _eq(a["core"], b["core"]),
        "a_eq": _eq(a["addr_norm"], b["addr_norm"]),
        "hs_eq": _eq(a["house"], b["house"]),
        "pc_eq": _eq(a["postcode"], b["postcode"]),
    }
    for key in ("n_tset", "n_tsort", "cc_ratio", "cc_jw"):
        out[key][no_name] = -1
    for key in ("a_tset", "a_ratio"):
        out[key][no_addr] = -1
    out["at_tset"][((a["addr_tok"].fill_null("") == "") | (b["addr_tok"].fill_null("") == "")).to_numpy()] = -1
    ta_, tb_ = _tokens(a["core"]), _tokens(b["core"])
    shared = pl.DataFrame({"x": ta_, "y": tb_}).select(pl.col("x").list.set_intersection(pl.col("y")).list.len())
    out["n_shared"] = shared.to_series().cast(pl.Float32).to_numpy()
    return {f"{prefix}{k}": v for k, v in out.items()}


QUERY_P = ["p_max", "p_2", "p_3", "p_sum", "n_p50", "n_p90", "n_surv"]


def features(surv: pl.DataFrame, s1_text: pl.DataFrame, tg_text: pl.DataFrame, workers: int) -> pl.DataFrame:
    """Stage-2 features for survivors (s1, t, target_id, p, rank).

    s1_text / tg_text: normalized views indexed by position = s1 row / target row
    (columns TEXT plus `src`, `alen`, `nlen`)."""
    surv = surv.sort("s1", "rank")
    p = surv["p"].to_numpy()
    q = surv.with_columns(
        pl.col("p").max().over("s1").alias("p_max"),
        pl.col("p").sort(descending=True).implode().list.get(1, null_on_oob=True).over("s1").fill_null(0).alias("p_2"),
        pl.col("p").sort(descending=True).implode().list.get(2, null_on_oob=True).over("s1").fill_null(0).alias("p_3"),
        pl.col("p").sum().over("s1").alias("p_sum"),
        (pl.col("p") >= 0.5).sum().over("s1").cast(pl.Float32).alias("n_p50"),
        (pl.col("p") >= 0.9).sum().over("s1").cast(pl.Float32).alias("n_p90"),
        pl.len().over("s1").cast(pl.Float32).alias("n_surv"),
        pl.col("p").shift(-1).over("s1").fill_null(0).alias("p_next"),
        pl.col("p").shift(1).over("s1").fill_null(1).alias("p_prev"))
    cols = {"p1": p.astype(np.float32), "p1_logit": logit(p), "rank": surv["rank"].cast(pl.Float32).to_numpy(),
            "gap_max": (q["p_max"] - q["p"]).to_numpy().astype(np.float32),
            "gap_next": (q["p"] - q["p_next"]).to_numpy().astype(np.float32),
            "gap_prev": (q["p_prev"] - q["p"]).to_numpy().astype(np.float32),
            "others_sum": (q["p_sum"] - q["p"]).to_numpy().astype(np.float32),
            "others_p90": (q["n_p90"] - (q["p"] >= 0.9).cast(pl.Float32)).to_numpy()}
    for c in QUERY_P:
        cols[c] = q[c].cast(pl.Float32).to_numpy()
    a = s1_text.gather(surv["qi"] if "qi" in surv.columns else surv["s1"])
    b = tg_text.gather(surv["ti"] if "ti" in surv.columns else surv["t"])
    cols.update(pair_block(a, b, workers, "q_"))
    cols["src"] = b["src"].cast(pl.Float32).to_numpy()
    cols["t_nlen"] = b["nlen"].cast(pl.Float32).to_numpy()
    cols["t_alen"] = b["alen"].cast(pl.Float32).to_numpy()
    cols["q_nlen"] = a["nlen"].cast(pl.Float32).to_numpy()
    cols["q_alen"] = a["alen"].cast(pl.Float32).to_numpy()
    cols.update(sibling_features(surv, tg_text, workers))
    return surv.select("s1", "t", "target_id").with_columns(
        [pl.Series(k, v, dtype=pl.Float32) for k, v in cols.items()])


SIB_SIMS = ["n_tset", "cc_ratio", "a_tset", "at_tset", "a_eq", "hs_eq", "n_eq"]


def sibling_features(surv: pl.DataFrame, tg_text: pl.DataFrame, workers: int) -> dict[str, np.ndarray]:
    """Compare each survivor with the SIBLINGS highest-p other survivors of its S1."""
    base = surv.select("s1", pl.col("ti" if "ti" in surv.columns else "t").alias("t"), "p", "rank").with_row_index("i")
    top = base.filter(pl.col("rank") <= SIBLINGS).select("s1", pl.col("t").alias("ts"), pl.col("p").alias("ps"),
                                                          pl.col("rank").alias("rs"))
    pairs = (base.join(top, on="s1").filter(pl.col("t") != pl.col("ts"))
             .sort("i", "rs").group_by("i", maintain_order=True).head(SIBLINGS)
             .with_columns(pl.int_range(pl.len()).over("i").alias("j")))
    sims = pair_block(tg_text.gather(pairs["t"]), tg_text.gather(pairs["ts"]), workers, "")
    pairs = pairs.with_columns([pl.Series(k, sims[k]) for k in SIB_SIMS])
    ps = pl.col("ps")
    conf = ps >= 0.5
    agg = [
        pl.col("ps").max().alias("sib_pmax"),
        *[pl.col(k).filter(conf).max().alias(f"sibc_{k}") for k in SIB_SIMS],
        *[(pl.col(k).clip(lower_bound=0) * ps).max().alias(f"sibw_{k}") for k in SIB_SIMS],
        ((ps >= 0.9) & (pl.col("a_tset") >= 90)).sum().alias("sib_conf_addr"),
        ((ps >= 0.9) & (pl.col("n_tset") >= 90)).sum().alias("sib_conf_name"),
        ((ps >= 0.9) & (pl.col("hs_eq") == 1)).sum().alias("sib_conf_house"),
        ((ps >= 0.9) & (pl.col("hs_eq") == 0)).sum().alias("sib_conf_house_diff"),
        ((ps >= 0.9) & (pl.col("a_eq") == 1)).sum().alias("sib_conf_aeq"),
        ((ps >= 0.9) & (pl.col("n_eq") == 1)).sum().alias("sib_conf_neq"),
    ]
    first = pairs.filter(pl.col("j") == 0).select("i", *[pl.col(k).alias(f"sib1_{k}") for k in SIB_SIMS],
                                                  pl.col("ps").alias("sib1_p"))
    per = pairs.group_by("i").agg(agg).join(first, on="i", how="left")
    out = base.select("i").join(per, on="i", how="left").sort("i")
    return {c: out[c].cast(pl.Float32).fill_null(-1).to_numpy() for c in out.columns if c != "i"}


def text_views(frame: pl.DataFrame, norm: str = "v5") -> pl.DataFrame:
    """Normalized views of raw rows (entity_id, business_name, business_address, country)."""
    if norm == "v5":
        from .text_norm_v5 import normalise
    else:
        from .text_norm import normalise
    out = normalise(frame.with_columns(pl.col(pl.String).fill_null("")))
    return out.select(*TEXT, pl.col("core").str.len_chars().alias("nlen"),
                      pl.col("addr_norm").str.len_chars().alias("alen"))
