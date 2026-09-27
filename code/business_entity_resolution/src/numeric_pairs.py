"""Numeric rescue channels J/K/L (from Astra's v3-rescue review) as a memory-light step.

J: country + house number + each name skeleton token
K: country + house number + each address token
L: country + postcode + each name skeleton token
A key shared by more than its cap of targets in the complete catalog is dropped (cap
from config.KEY_TYPES). Only 6 columns are loaded, one channel is processed at a time and
queries are joined in batches; pairs are written as parquet parts sorted by query row so
the main pipeline can read just the rows of its current chunk.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl

from .blocking import NUMERIC, numeric_postings
from .config import CACHE_DIR, KEY_TYPES

COLS = ["row", "country", "skel", "addr_tok", "house", "postcode"]
BATCH = 250_000


def build(split: str, out: Path, query_rows: np.ndarray | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.txt").exists():
        return {}
    t0 = time.time()
    s1 = pl.read_parquet(CACHE_DIR / f"{split}_s1.parquet", columns=COLS)
    if query_rows is not None:
        s1 = s1.filter(pl.col("row").is_in(pl.Series(query_rows, dtype=pl.UInt32).implode()))
    tg = pl.read_parquet(CACHE_DIR / f"{split}_tg.parquet", columns=COLS)
    stats = {}
    for tag, tok, num in NUMERIC:
        if tag not in KEY_TYPES:
            continue
        cap = KEY_TYPES[tag][1]
        tp = numeric_postings(tg, tag, tok, num)
        keep = tp.group_by("key").len().filter(pl.col("len") <= cap).select("key")
        tp = tp.join(keep, on="key", how="semi").rename({"row": "t"})
        del keep
        n = 0
        for b, off in enumerate(range(0, len(s1), BATCH)):
            qp = numeric_postings(s1.slice(off, BATCH), tag, tok, num).rename({"row": "s1"})
            pairs = qp.join(tp, on="key").select("s1", "t").unique().sort("s1", "t")
            pairs.write_parquet(out / f"{tag}_{b:04d}.parquet", row_group_size=200_000)
            n += len(pairs)
        stats[tag] = n
        del tp
        print(f"  numeric {split} {tag}: {n:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    (out / "done.txt").write_text(repr(stats))
    return stats


def load(out: Path, rows: np.ndarray) -> pl.DataFrame:
    """(s1, t, tag) numeric pairs for the given query rows."""
    lo, hi = int(rows.min()), int(rows.max())
    parts = []
    for tag, _, _ in NUMERIC:
        files = sorted(out.glob(f"{tag}_*.parquet"))
        if not files:
            continue
        x = (pl.scan_parquet(files).filter((pl.col("s1") >= lo) & (pl.col("s1") <= hi)).collect()
             .filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode())))
        parts.append(x.with_columns(pl.lit(tag).alias("tag")))
    if not parts:
        return pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "tag": pl.String})
    return pl.concat(parts)
