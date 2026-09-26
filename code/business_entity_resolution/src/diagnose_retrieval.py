"""Compare retrieval on fixed development queries against the full target catalog.

This measures candidate recall and oracle F0.5, not achieved model performance.
Only aggregate statistics are exported; true links never enter candidate generation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import polars as pl

from .blocking import Blocker
from .config import SEED, BLOCKING_PROFILE, RETRIEVAL_RANK, RETRIEVAL_VERSION, KEY_TYPES
from .metric import oracle, per_entity
from .prepare import load, load_truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=int, default=5000)
    parser.add_argument("--top-k", type=int, nargs="+", default=[30, 60, 100])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.queries <= 100000:
        parser.error("queries must be between 1 and 100000")
    start = time.perf_counter()
    s1, tg = load("train")
    truth = load_truth(s1, tg)
    # Stable canonical v1 development interval, independent of training sample size.
    perm = np.random.default_rng(SEED).permutation(len(s1))
    rows = np.sort(perm[200000:200000 + args.queries]).astype(np.uint32)
    gt = truth.filter(pl.col("s1").is_in(pl.Series(rows)))
    del truth
    ids_hash = hashlib.sha256("\n".join(s1["id"].gather(rows).to_list()).encode()).hexdigest()
    print(f"Loaded {len(s1):,} queries and ALL {len(tg):,} targets; evaluation {len(rows):,} queries", flush=True)
    blocker = Blocker(s1, tg)
    print(f"Index ready in {time.perf_counter()-start:.1f}s", flush=True)
    reports = []
    for k in args.top_k:
        tick = time.perf_counter()
        cand = blocker.candidates(rows, top_k=k)
        hits = cand.select("s1", "t").join(gt, on=["s1", "t"])
        per = per_entity(rows, gt, hits).join(s1.select(pl.col("row").alias("s1"), "country"), on="s1")
        report = dict(top_k=k, queries=len(rows), target_catalog=len(tg), true_pairs=len(gt),
                      retrieved_true_pairs=len(hits), recall=len(hits)/max(1, len(gt)),
                      oracle_f05=oracle(rows, gt, cand), pairs=len(cand), pairs_per_query=len(cand)/len(rows),
                      zero_candidates=len(rows)-cand["s1"].n_unique(), seconds=time.perf_counter()-tick,
                      country_oracle=per.group_by("country").agg(pl.len().alias("queries"), pl.col("f").mean().alias("oracle_f05")).to_dicts())
        reports.append(report)
        print(json.dumps(report), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(query_ids_sha256=ids_hash, seed=SEED,
                                      profile=BLOCKING_PROFILE, rank=RETRIEVAL_RANK,
                                      retrieval_version=RETRIEVAL_VERSION, key_types=KEY_TYPES,
                                      definition="candidate oracle, not model score", reports=reports,
                                      elapsed_seconds=time.perf_counter()-start), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
