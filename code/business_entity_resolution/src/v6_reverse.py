"""Reverse retrieval: which S1 record does each candidate target resemble most?

Every S2/S3 record belongs to at most one S1 entity. A candidate that closely resembles a
different S1 record is probably that record's variant, even when it also looks plausible
for the current S1 (same building, shared words). For each surviving candidate target we
search the complete S1 population of its country with the combined name+address
character n-gram view (IDF over that S1 population, no labels) and record how the
current S1 compares with the best other S1. Train and test use the same procedure over
their own complete S1 files, so the feature means the same thing in both.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import polars as pl

SPEC = {"analyzer": "char_wb", "ngram": [3, 4], "max_df": 0.03}
WORD = {"analyzer": "word", "ngram": [1, 2], "max_df": 0.05}
DIMS = 2 ** 20
BLOCK = 250_000
REV_FEATURES = ["rev_self", "rev_best_other", "rev_gap", "rev_rank", "rev_n_close",
                "revw_self", "revw_best_other", "revw_gap", "revw_rank"]


def combo(frame: pl.DataFrame) -> list[str]:
    return frame.select(pl.concat_str([pl.col("core").fill_null(""), pl.col("addr_norm").fill_null("")],
                                      separator=" ")).to_series().to_list()


def _weighted(matrix, idf):
    from sklearn.preprocessing import normalize
    matrix.data *= idf[matrix.indices]
    matrix.eliminate_zeros()
    return normalize(matrix, copy=False).tocsr()


def _search(queries, population, top, workers, batch=8192):
    """Top-`top` population rows for each query row: DataFrame(tq, s, score)."""
    from sparse_dot_topn import sp_matmul_topn
    found = []
    for first in range(0, population.shape[0], BLOCK):
        block = population[first:first + BLOCK].T.tocsr()
        for start in range(0, queries.shape[0], batch):
            part = queries[start:start + batch]
            if part.nnz == 0:
                continue
            hits = sp_matmul_topn(part, block, top_n=top, threshold=1e-6, sort=False, n_threads=workers).tocoo()
            found.append(pl.DataFrame({"tq": (hits.row + start).astype(np.uint32),
                                       "s": (hits.col + first).astype(np.uint32),
                                       "score": hits.data.astype(np.float32)}))
        del block
    if not found:
        return pl.DataFrame(schema={"tq": pl.UInt32, "s": pl.UInt32, "score": pl.Float32})
    return (pl.concat(found).sort(["tq", "score", "s"], descending=[False, True, False])
            .group_by("tq", maintain_order=True).head(top))


def _channel(pairs, s1_country, tg_country, spec, top, workers):
    """(pair index i, self cosine, best other cosine, rank of self, count of others within 0.05)."""
    from .v5_vec import vectorize
    started = time.monotonic()
    pop = vectorize("rev", combo(s1_country), DIMS, workers, spec=spec)
    print(f"  rev {spec['analyzer']}: population {pop.shape[0]:,} nnz={pop.nnz:,} t={time.monotonic()-started:.0f}s", flush=True)
    n = pop.shape[0]
    count = np.bincount(pop.indices, minlength=DIMS)
    idf = (1 + np.log((1 + n) / (1 + count))).astype(np.float32)
    idf[(count == 0) | (count > spec["max_df"] * n)] = 0
    pop = _weighted(pop, idf)
    qry = _weighted(vectorize("rev", combo(tg_country), DIMS, workers, spec=spec), idf)
    print(f"  rev {spec['analyzer']}: queries {qry.shape[0]:,} nnz={qry.nnz:,} t={time.monotonic()-started:.0f}s", flush=True)
    hits = _search(qry, pop, top, workers)
    print(f"  rev {spec['analyzer']}: searched hits={len(hits):,} t={time.monotonic()-started:.0f}s", flush=True)
    # self cosine for every pair, computed directly (the S1 may be outside the top list)
    a = qry[pairs["tq"].to_numpy()]
    b = pop[pairs["sq"].to_numpy()]
    self_cos = np.asarray(a.multiply(b).sum(axis=1)).ravel().astype(np.float32)
    del a, b, pop, qry
    gc.collect()
    frame = pairs.select("i", "tq", "sq").with_columns(pl.Series("self", self_cos))
    other = (frame.join(hits, on="tq", how="inner").filter(pl.col("s") != pl.col("sq"))
             .group_by("i").agg(pl.col("score").max().alias("best"),
                                (pl.col("score") > pl.col("self")).sum().alias("above"),
                                (pl.col("score") >= pl.col("self") - 0.05).sum().alias("close")))
    frame = frame.join(other, on="i", how="left").with_columns(pl.col("best", "above", "close").fill_null(0))
    return (frame["i"].to_numpy(), frame["self"].to_numpy(), frame["best"].cast(pl.Float32).to_numpy(),
            (1 + frame["above"]).cast(pl.Float32).to_numpy(), frame["close"].cast(pl.Float32).to_numpy())


def reverse_features(pairs: pl.DataFrame, s1_pop: pl.DataFrame, targets: pl.DataFrame, workers: int,
                     top: int = 6) -> pl.DataFrame:
    """pairs: (sq, tq) = S1 position in s1_pop and target position in `targets`.
    s1_pop: the complete normalized S1 file of the split (core, addr_norm, country).
    targets: normalized candidate targets (core, addr_norm, country). Returns REV_FEATURES per pair."""
    pairs = pairs.select("sq", "tq").with_row_index("i")
    out = {name: np.full(len(pairs), -1, dtype=np.float32) for name in REV_FEATURES}
    tcountry = targets["country"].fill_null("")
    for country in sorted(tcountry.unique().to_list()):
        s_idx = np.where((s1_pop["country"].fill_null("") == country).to_numpy())[0]
        t_idx = np.where((tcountry == country).to_numpy())[0]
        if not len(s_idx) or not len(t_idx):
            continue
        s_pos = np.full(len(s1_pop), -1, dtype=np.int64)
        s_pos[s_idx] = np.arange(len(s_idx))
        t_pos = np.full(len(targets), -1, dtype=np.int64)
        t_pos[t_idx] = np.arange(len(t_idx))
        sub = pairs.with_columns(pl.Series("tq", t_pos[pairs["tq"].to_numpy()]),
                                 pl.Series("sq", s_pos[pairs["sq"].to_numpy()]))
        sub = sub.filter((pl.col("tq") >= 0) & (pl.col("sq") >= 0))
        s_frame, t_frame = s1_pop[s_idx], targets[t_idx]
        for prefix, spec in (("rev", SPEC), ("revw", WORD)):
            i, selfv, best, rank, close = _channel(sub, s_frame, t_frame, spec, top, workers)
            out[f"{prefix}_self"][i] = selfv
            out[f"{prefix}_best_other"][i] = best
            out[f"{prefix}_gap"][i] = selfv - best
            out[f"{prefix}_rank"][i] = rank
            if prefix == "rev":
                out["rev_n_close"][i] = close
            gc.collect()
    return pl.DataFrame({k: pl.Series(k, v, dtype=pl.Float32) for k, v in out.items()})
