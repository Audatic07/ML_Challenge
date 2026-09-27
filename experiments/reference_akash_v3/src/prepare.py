"""Load the raw TSV files, normalise every record once and cache the result as parquet."""
import time

import polars as pl

from . import config
from .text_norm import COLUMNS, normalize_records


def read_tsv(path):
    """Read a challenge TSV with every column as a string and empty cells as ''."""
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema=False,
        missing_utf8_is_empty_string=True, truncate_ragged_lines=True,
    )


def load_source(split, n):
    """Return the normalised frame for {split}_source{n}.tsv, building the cache if needed.

    Columns: entity_id, country, src, plus the normalised views in text_norm.COLUMNS.
    """
    cache = config.WORK_DIR / f"{split}_s{n}.parquet"
    if cache.exists():
        return pl.read_parquet(cache)
    t0 = time.time()
    raw = read_tsv(config.DATA_DIR / split / f"{split}_source{n}.tsv")
    rows = normalize_records(raw["business_name"].to_list(), raw["business_address"].to_list(),
                             workers=config.N_WORKERS)
    norm = pl.DataFrame(rows, schema=COLUMNS, orient="row")
    df = pl.concat([
        raw.select(
            "entity_id",
            pl.col("country").str.strip_chars().str.to_lowercase().alias("country"),
            pl.lit(n, dtype=pl.UInt8).alias("src"),
        ),
        norm,
    ], how="horizontal")
    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(cache)
    print(f"  normalised {split}_source{n}: {df.height:,} rows in {time.time() - t0:.0f}s", flush=True)
    return df


def load_split(split):
    """Return (s1, s23) for a split. Both get a UInt32 'idx' row index;
    s23 stacks Source 2 and Source 3 into one frame."""
    s1 = load_source(split, 1).with_row_index("idx")
    s23 = pl.concat([load_source(split, 2), load_source(split, 3)]).with_row_index("idx")
    return s1, s23
