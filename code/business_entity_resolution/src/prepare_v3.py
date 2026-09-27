"""Build the v3 caches: learn the transliteration dictionary and apply it.

Reads the base caches written by prepare.py from ER_BASE_CACHE, learns the dictionary
from labelled pairs of fit queries only (reserved validation and audit queries are
excluded), applies it to train and test, recomputes the derived views and writes
{split}_s1.parquet / {split}_tg.parquet to ER_CACHE_DIR.

Run:  python -m src.prepare_v3
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import polars as pl

from .config import CACHE_DIR, SEED, TRAIN_SAMPLE, VAL_SAMPLE
from .prepare import load_truth
from .splits import make_split
from .text_norm import derive
from .translit import apply_view, learn

BASE = Path(os.environ.get("ER_BASE_CACHE", CACHE_DIR))


def _read(split: str):
    return pl.read_parquet(BASE / f"{split}_s1.parquet"), pl.read_parquet(BASE / f"{split}_tg.parquet")


def _map(df: pl.DataFrame, maps: dict) -> pl.DataFrame:
    name = apply_view(df, "name_norm", maps["name_norm"])
    addr = apply_view(df, "addr_norm", maps["addr_norm"])
    df = df.with_columns((name != df["name_norm"]).alias("translit"), name, addr)
    return derive(df)


def main() -> None:
    if BASE.resolve() == CACHE_DIR.resolve():
        raise SystemExit("set ER_BASE_CACHE (input) and ER_CACHE_DIR (output) to different folders")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    s1, tg = _read("train")
    truth = load_truth(s1, tg)
    held = make_split(len(s1), TRAIN_SAMPLE, VAL_SAMPLE, SEED).held_out
    links = truth.filter(~pl.col("s1").is_in(pl.Series(held, dtype=pl.UInt32).implode()))
    maps = learn(s1, tg, links)
    print(f"dictionary: {len(maps['name_norm']):,} name tokens, {len(maps['addr_norm']):,} address tokens "
          f"from {len(links):,} fit links in {time.time() - t:.0f}s", flush=True)
    for split in ("train", "test"):
        t = time.time()
        if split == "test":
            s1, tg = _read("test")
        for df, name in ((s1, "s1"), (tg, "tg")):
            out = _map(df, maps)
            out.write_parquet(CACHE_DIR / f"{split}_{name}.parquet")
            print(f"  {split} {name}: {len(out):,} rows, translit {int(out['translit'].sum()):,}", flush=True)
        print(f"{split} done in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
