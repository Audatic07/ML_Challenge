"""Reverse-retrieval features (src/v6_reverse.py) for the tune survivors.

    python tools/v6_analysis/reverse_tune.py [tag]

Searches every survivor target against the complete train S1 population of its country and
writes WORK/s2_rev_tune_{tag}.parquet for stage2_cv.py --rev. Status on 27 Sep: only the
India character search finished (727 s on 20 threads locally); no CV result yet.
"""
import sys
import time

import numpy as np
import polars as pl

from _common import DATA, WORK, read_tsv
from src.text_norm_v5 import normalise
from src.v6_reverse import reverse_features


def views(raw):
    return normalise(raw.with_columns(pl.col(pl.String).fill_null(""))).select("core", "addr_norm", "country")


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "K10_f0"
    started = time.time()
    feat = pl.read_parquet(WORK / f"s2_tune_{tag}.parquet", columns=["s1", "t", "target_id"])
    population = views(read_tsv(DATA / "train" / "train_source1.tsv"))
    rows = feat["t"].unique().sort()
    targets = views(pl.concat([read_tsv(DATA / "train" / f"train_source{s}.tsv") for s in (2, 3)]).gather(rows))
    print(f"S1 population={len(population):,} targets={len(targets):,} t={time.time()-started:.0f}s", flush=True)
    pairs = feat.join(pl.DataFrame({"t": rows, "tq": pl.Series(np.arange(len(rows), dtype=np.uint32))}), on="t")
    pairs = pairs.select(pl.col("s1").alias("sq"), "tq", "s1", "target_id")
    rev = reverse_features(pairs, population, targets, workers=20)
    pairs.select("s1", "target_id").with_columns(rev.get_columns()).write_parquet(WORK / f"s2_rev_tune_{tag}.parquet")
    print(f"done t={time.time()-started:.0f}s")


if __name__ == "__main__":
    main()
