"""Per-country decision thresholds for the matcher trained by exp_v3.

Uses the same fixed tune/check halves of the validation queries as exp_v3: thresholds
are picked on `tune` only, and kept only if they also do not hurt the untouched `check`
half. Countries without labels (France) use the global threshold from result.json.
Writes WORK_DIR/exp/thresholds.json, which assemble_v6 reads.

Run:  python -m src.thresholds_v7
"""
from __future__ import annotations

import json

import numpy as np
import polars as pl

from .config import CACHE_DIR, WORK_DIR
from .exp_v3 import query_sets
from .metric import f05_macro, unique_assign
from .prepare import load_truth

EXP = WORK_DIR / "exp"
GRID = [round(x, 3) for x in np.arange(0.3, 0.951, 0.01)]


def main() -> None:
    res = json.loads((EXP / "result.json").read_text())
    s1 = pl.read_parquet(CACHE_DIR / "train_s1.parquet", columns=["row", "id", "country"])
    tg = pl.read_parquet(CACHE_DIR / "train_tg.parquet", columns=["row", "id"])
    va_rows = query_sets(len(s1))["val"]
    perm = np.random.default_rng(7).permutation(va_rows)  # the exp_v3 halves
    parts = {"tune": np.sort(perm[: len(perm) // 2]), "check": np.sort(perm[len(perm) // 2:])}
    truth = load_truth(s1, tg)
    ua = unique_assign(pl.read_parquet(EXP / "val_scored.parquet"))
    country = dict(zip(s1["row"].to_list(), s1["country"].to_list()))

    def score(rows: np.ndarray, thr_of) -> float:
        rs = pl.Series(rows, dtype=pl.UInt32).implode()
        pred = ua.filter(pl.col("s1").is_in(rs)).with_columns(
            pl.col("s1").replace_strict(country, return_dtype=pl.String).alias("c"))
        pred = pred.filter(pl.col("p") >= pl.col("c").replace_strict(thr_of, default=res["threshold"],
                                                                     return_dtype=pl.Float64))
        return f05_macro(rows, truth.filter(pl.col("s1").is_in(rs)), pred)

    by_country = {}
    for c in sorted({country[r] for r in parts["tune"]}):
        rows_c = np.array([r for r in parts["tune"] if country[r] == c], dtype=np.uint32)
        by_country[c] = max(GRID, key=lambda x: score(rows_c, {c: x}))
    out = {"global": res["threshold"]}
    for name, thr in (("global", {}), ("by_country", by_country)):
        out[name + "_scores"] = {k: score(rows, thr) for k, rows in
                                 (("tune", parts["tune"]), ("check", parts["check"]), ("all", va_rows))}
    keep = out["by_country_scores"]["check"] >= out["global_scores"]["check"]
    out["by_country"] = by_country if keep else {}
    out["picked"] = "by_country" if keep else "global"
    (EXP / "thresholds.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
