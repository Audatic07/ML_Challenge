"""Stage-2 features for stage-1 survivors (top K per S1 by v5.1 probability, optional floor).

    python tools/v6_analysis/stage2_features.py tune [K] [floor]
    python tools/v6_analysis/stage2_features.py test [K] [floor]    (not run yet)

tune: reads WORK/tune_scores.parquet and the labels from loss.py and writes
      WORK/s2_tune_K{K}_f{floor}.parquet with a label column `y`.
test: reads WORK/test_scores/*.parquet and writes WORK/test_s2_K{K}_f{floor}/feat-NNN.parquet
      in chunks of 100k S1, without labels.
"""
import sys
import time

import numpy as np
import polars as pl

from _common import DATA, THREADS, WORK, read_tsv
from src.v6_stage2 import features, survivors, text_views


def target_text(split, rows):
    """Normalized views of the given global target rows (source 2 rows first, then source 3)."""
    first = read_tsv(DATA / split / f"{split}_source2.tsv")
    n2 = len(first)
    frame = pl.concat([first, read_tsv(DATA / split / f"{split}_source3.tsv")]).with_row_index("row")
    del first
    raw = frame.gather(rows).with_columns((pl.col("row") >= n2).cast(pl.UInt8).add(2).alias("src"))
    del frame
    return text_views(raw).with_columns(raw["src"])


def main():
    split = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    started = time.time()
    if split == "tune":
        surv = survivors(pl.read_parquet(WORK / "tune_scores.parquet"), k, floor)
    else:
        parts = [survivors(pl.read_parquet(path, columns=["s1", "t", "target_id", "p"]), k, floor)
                 for path in sorted((WORK / "test_scores").glob("*.parquet"))]
        surv = pl.concat(parts)
    print(f"{split} survivors={len(surv):,} S1={surv['s1'].n_unique():,} t={time.time()-started:.0f}s", flush=True)
    data_split = "train" if split == "tune" else "test"
    queries = read_tsv(DATA / data_split / f"{data_split}_source1.tsv").with_row_index("row")
    q_text = text_views(queries)  # indexed by S1 row
    del queries
    rows = surv["t"].unique().sort()
    t_text = target_text(data_split, rows)
    surv = surv.join(pl.DataFrame({"t": rows, "ti": pl.Series(np.arange(len(rows), dtype=np.uint32))}), on="t")
    print(f"normalized S1={len(q_text):,} targets={len(t_text):,} t={time.time()-started:.0f}s", flush=True)
    tag = f"K{k}_f{floor:g}"
    if split == "tune":
        feat = features(surv, q_text, t_text, workers=THREADS)
        labels = pl.read_parquet(WORK / "tune_labeled.parquet", columns=["s1", "t", "y"]).rename({"t": "target_id"})
        feat = feat.join(labels, on=["s1", "target_id"], how="left").with_columns(pl.col("y").fill_null(0))
        feat.write_parquet(WORK / f"s2_tune_{tag}.parquet")
        print(f"wrote {len(feat):,} rows x {len(feat.columns)} columns t={time.time()-started:.0f}s")
        return
    out = WORK / f"test_s2_{tag}"
    out.mkdir(exist_ok=True)
    chunk = 100_000
    for first in range(0, int(surv["s1"].max()) + 1, chunk):
        path = out / f"feat-{first // chunk:03d}.parquet"
        part = surv.filter((pl.col("s1") >= first) & (pl.col("s1") < first + chunk))
        if path.exists() or not len(part):
            continue
        features(part, q_text, t_text, workers=THREADS).write_parquet(path)
        print(f"chunk {first // chunk}: {len(part):,} rows t={time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
