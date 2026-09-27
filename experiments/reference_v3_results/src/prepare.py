"""Read the raw TSVs, normalise them and cache the result as parquet in WORK_DIR.

The cache is keyed only by file name: delete WORK_DIR/*.parquet after changing
text_norm.py or this file, otherwise the old normalisation is reused silently.
"""
from __future__ import annotations

import time

import polars as pl

from .config import CACHE_DIR, DATA_DIR, WORK_DIR
from .text_norm import normalise

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
KEEP = ["row", "id", "src", "country", "name_norm", "core", "skel", "concat",
        "addr_norm", "addr_tok", "postcode", "house"]


def read_tsv(path) -> pl.DataFrame:
    """Explicit tab separator, no quote handling (names contain quotes), all strings,
    empty cells as ''."""
    df = pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)
    return df.with_columns(pl.all().fill_null(""))


def _load_source(split: str, s: int) -> pl.DataFrame:
    df = read_tsv(DATA_DIR / split / f"{split}_source{s}.tsv")
    if df.columns != COLUMNS:
        raise ValueError(f"unexpected columns in {split} source {s}: {df.columns}")
    return df.with_columns(pl.lit(s, dtype=pl.UInt8).alias("src"))


def load(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return (s1, tg): Source 1 records and combined Source 2+3 records.

    `row` is the position in each frame and is what every other module uses.
    """
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f1, ft = CACHE_DIR / f"{split}_s1.parquet", CACHE_DIR / f"{split}_tg.parquet"
    if f1.exists() and ft.exists():
        return pl.read_parquet(f1), pl.read_parquet(ft)
    out = []
    for sources, path in (((1,), f1), ((2, 3), ft)):
        t = time.time()
        df = pl.concat([_load_source(split, s) for s in sources])
        df = normalise(df).rename({"entity_id": "id"})
        df = df.with_row_index("row").with_columns(pl.col("row").cast(pl.UInt32))
        df = df.select(KEEP)
        df.write_parquet(path)
        print(f"prepared {split} sources {sources}: {len(df):,} rows in {time.time() - t:.0f}s", flush=True)
        out.append(df)
    return out[0], out[1]


def load_truth(s1: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """Training links as (s1, t) row pairs."""
    gt = read_tsv(DATA_DIR / "train" / "train_ground_truth.tsv")
    gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
          .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))
    gt = (gt.join(s1.select(pl.col("id").alias("source1_entity_id"), pl.col("row").alias("s1")),
                  on="source1_entity_id")
          .join(tg.select(pl.col("id").alias("matched_entity_ids"), pl.col("row").alias("t")),
                on="matched_entity_ids"))
    return gt.select("s1", "t")


if __name__ == "__main__":
    import sys
    for sp in sys.argv[1:] or ["train", "test"]:
        a, b = load(sp)
        print(sp, a.shape, b.shape)
