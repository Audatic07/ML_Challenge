"""Diagnose why true matches are missed at blocking, for one country.

Takes up to 3,000 validation entities of the chosen country, builds blocking keys for
them and for their true Source 2/3 matches only, and reports:
  - share of true pairs that share at least one key before block caps
    (pairs sharing none cannot be found by any setting; they need new keys or normalisation)
  - which key types the found pairs share
  - 40 example pairs that share no key, in normalised form
Light on memory: it never joins against the full catalogue.
Run:  python -m src.diag_misses india      (or us)
"""
import sys

import numpy as np
import polars as pl

from . import config
from .blocking import KT_NAMES, build_keys, token_freq
from .prepare import load_split
from .train import load_truth


def main(country="india", n=3000):
    s1, s23 = load_split("train")
    truth = load_truth(s1, s23)
    perm = np.random.default_rng(config.SEED).permutation(s1.height)
    va = pl.Series(perm[config.TRAIN_S1_SAMPLE:config.TRAIN_S1_SAMPLE + config.VAL_S1_SAMPLE]).cast(pl.UInt32)
    q = s1.filter(pl.col("idx").is_in(va.implode()) & (pl.col("country") == country)).head(n)
    t = truth.join(q.select(pl.col("idx").alias("idx1")), on="idx1")
    name_freq = token_freq([s1, s23], "skel")
    addr_freq = token_freq([s1, s23], "addr_tok")
    k1 = build_keys(q, name_freq, addr_freq).rename({"idx": "idx1"})
    k2 = build_keys(s23.filter(pl.col("idx").is_in(t["idx2"].implode())), name_freq, addr_freq)
    shared = t.join(k1, on="idx1").join(k2.rename({"idx": "idx2"}).drop("kt"), on=["idx2", "key"])
    found = shared.select("idx1", "idx2").unique()
    print(f"{country}: {q.height:,} entities, {t.height:,} true pairs; "
          f"share of pairs sharing >=1 key before caps: {found.height / max(t.height, 1):.4f}")
    by_kt = shared.group_by("kt").agg(pl.len().alias("pairs")).sort("kt")
    print("pairs sharing each key type:",
          ", ".join(f"{KT_NAMES[r['kt']]}={r['pairs']:,}" for r in by_kt.iter_rows(named=True)))
    miss = t.join(found, on=["idx1", "idx2"], how="anti")
    cols = ["entity_id", "name_norm", "addr_norm", "postcode", "house"]
    ex = (miss.head(40)
          .join(s1.select(pl.col("idx").alias("idx1"), *[pl.col(c).alias("s1_" + c) for c in cols]), on="idx1")
          .join(s23.select(pl.col("idx").alias("idx2"), *[pl.col(c).alias("t_" + c) for c in cols]), on="idx2"))
    print(f"\n{miss.height:,} pairs share no key. Examples (S1 | S2/S3):")
    for r in ex.iter_rows(named=True):
        print(f"- NAME  {r['s1_name_norm']!r}  |  {r['t_name_norm']!r}")
        print(f"  ADDR  {r['s1_addr_norm']!r}  |  {r['t_addr_norm']!r}")
        print(f"  pc/house {r['s1_postcode']}/{r['s1_house']}  |  {r['t_postcode']}/{r['t_house']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "india")
