"""Build the v7 caches from the finished v3 caches: re-run text_norm.derive on them.

derive() now also maps look-alike digits inside name words back to letters
(`mi1ler` -> `miller`). Every other step of derive() is idempotent on v3 caches, so
this only changes names that had such typos. House number, postcode, the address and
the transliteration flag are copied unchanged. Much faster than prepare + prepare_v3.

Run:  ER_BASE_CACHE=/path/to/er_v3 ER_CACHE_DIR=/path/to/er_v7 python -m src.prepare_v7
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl

from .config import CACHE_DIR
from .text_norm import derive

BASE = Path(os.environ.get("ER_BASE_CACHE", CACHE_DIR))


def main() -> None:
    if BASE.resolve() == CACHE_DIR.resolve():
        raise SystemExit("set ER_BASE_CACHE (input) and ER_CACHE_DIR (output) to different folders")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        for name in ("s1", "tg"):
            t = time.time()
            df = pl.read_parquet(BASE / f"{split}_{name}.parquet")
            out = derive(df).select(df.columns)
            changed = int((out["name_norm"] != df["name_norm"]).sum())
            out.write_parquet(CACHE_DIR / f"{split}_{name}.parquet")
            print(f"  {split} {name}: {len(out):,} rows, {changed:,} names changed ({time.time() - t:.0f}s)", flush=True)
            del df, out


if __name__ == "__main__":
    main()
