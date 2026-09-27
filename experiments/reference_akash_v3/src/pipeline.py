"""Shared chunked loop: blocking -> features for a list of Source 1 indices."""
import time

import numpy as np
import polars as pl

from . import config
from .blocking import candidates
from .features import pair_features


def iter_scored_chunks(s1_idx, keys1, keys23, s1, s23):
    """Yield feature frames chunk by chunk so memory stays bounded."""
    s1_idx = np.asarray(s1_idx)
    for start in range(0, len(s1_idx), config.CHUNK):
        t0 = time.time()
        chunk = pl.Series(s1_idx[start:start + config.CHUNK]).cast(pl.UInt32)
        k1 = keys1.filter(pl.col("idx").is_in(chunk.implode()))
        cand = candidates(k1, keys23)
        feats = pair_features(cand, s1, s23) if cand.height else None
        print(f"  chunk {start // config.CHUNK + 1}/{-(-len(s1_idx) // config.CHUNK)}: "
              f"{len(chunk):,} S1, {cand.height:,} pairs, {time.time() - t0:.0f}s", flush=True)
        if feats is not None:
            yield feats
