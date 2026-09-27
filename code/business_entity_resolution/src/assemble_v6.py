"""Rebuild matching_results.tsv from saved v6 test scores with per-country thresholds.

Thresholds were picked on the tune half of the validation queries and confirmed on the
untouched check half (check 0.9737 -> 0.9741). Countries without labels (France) use
the global threshold. The candidate file is unchanged: the same scored pairs.

Run:  python -m src.assemble_v6 [--mode single]
"""
from __future__ import annotations

import json
import sys

import polars as pl

from .config import CACHE_DIR, OUT_DIR, WORK_DIR
from .metric import unique_assign
from .predict import write_lists

GLOBAL = 0.72
BY_COUNTRY = {"US": 0.79, "India": 0.69}
_TUNED = WORK_DIR / "exp" / "thresholds.json"  # written by thresholds_v7 for a retrained matcher
if _TUNED.exists():
    _t = json.loads(_TUNED.read_text())
    GLOBAL, BY_COUNTRY = _t["global"], _t["by_country"]


def decide_by_country(scored: pl.DataFrame, countries: pl.DataFrame) -> pl.DataFrame:
    """unique owner first (as in metric.decide), then the query country's threshold."""
    thr = pl.col("country").replace_strict(BY_COUNTRY, default=GLOBAL, return_dtype=pl.Float64)
    return (unique_assign(scored).join(countries, on="s1", how="left")
            .filter(pl.col("p") >= thr).select("s1", "t", "p"))


def main(argv: list[str]) -> None:
    mode = argv[1] if len(argv) > 1 and argv[0] == "--mode" else "single"
    chunks = sorted((WORK_DIR / "exp" / f"test_scored_{mode}").glob("chunk_*.parquet"))
    scored = pl.concat([pl.read_parquet(p) for p in chunks])
    s1 = pl.read_parquet(CACHE_DIR / "test_s1.parquet", columns=["row", "id", "country"])
    tg = pl.read_parquet(CACHE_DIR / "test_tg.parquet", columns=["row", "id"])
    if scored["s1"].max() >= len(s1) or scored.select("s1", "t").is_duplicated().any():
        raise SystemExit("scored chunks are incomplete or overlap")
    matches = decide_by_country(scored, s1.select(pl.col("row").alias("s1"), "country"))
    write_lists(OUT_DIR / "matching_results.tsv", s1, tg, matches, "matched_entity_ids")
    per = (matches.join(s1.select(pl.col("row").alias("s1"), "country"), on="s1")
           .group_by("country").agg(pl.col("s1").n_unique().alias("with_match"), pl.len().alias("matches")))
    stats = {"chunks": len(chunks), "scored_pairs": len(scored), "matches": len(matches),
             "thresholds": {**BY_COUNTRY, "other": GLOBAL}, "per_country": per.to_dicts()}
    (WORK_DIR / "exp" / "assemble_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
