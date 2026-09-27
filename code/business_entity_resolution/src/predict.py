"""Block, score and decide on the test set; write both submission files.

OUT_DIR/candidate_pairs.tsv   every candidate the model scored (last stage before the model)
OUT_DIR/matching_results.tsv  unique_assign + threshold from meta.json

Run:  python -m src.predict
"""
from __future__ import annotations

import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import Blocker
from .config import CHUNK, KEY_TYPES, OUT_DIR, TOP_K, WORK_DIR, WORKERS
from .features import FEATURES, build, matrix
from .metric import decide
from .prepare import load

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Exact settings of the supplied, tested incumbent. Missing version metadata is
# compatible only with this original policy and feature schema.
LEGACY_V1_KEYS = {"A": [3, 200], "B": [3, 200], "C": [2, 200],
                  "D": [2, 100], "E": [1, 50], "F": [2, 50], "G": [2, 100]}
LEGACY_V1_FEATURES = (
    "n_ratio n_partial n_tsort n_tset c_ratio c_tset c_jw sk_ratio sk_tset "
    "cc_ratio cc_partial a_ratio a_partial a_tsort a_tset at_ratio at_tset "
    "n_exact c_exact pc_eq hs_eq cty_eq n_len1 n_len2 a_len1 a_len2 ntok_r "
    "src bscore kA kB kC kD kE kF kG n_cand n_tset_gap a_tset_gap cc_ratio_gap "
    "at_tset_gap bscore_gap n_tset_rank a_tset_rank cc_ratio_rank at_tset_rank bscore_rank"
).split()


def validate_retrieval_config(meta: dict) -> None:
    """Reject a saved model whose candidate policy differs from active settings."""
    expected_keys = {key: list(settings) for key, settings in KEY_TYPES.items()}
    saved_keys = {key: list(settings) for key, settings in meta.get("key_types", {}).items()}
    if meta.get("top_k") != TOP_K or saved_keys != expected_keys:
        raise ValueError("retrieval settings changed since training (TOP_K or KEY_TYPES): retrain first")
    version = getattr(config, "RETRIEVAL_VERSION", "2")
    rank = getattr(config, "RETRIEVAL_RANK", "weight")
    if "retrieval_version" not in meta:
        legacy_compatible = (
            getattr(config, "BLOCKING_PROFILE", "v1") == "v1"
            and rank == "weight" and meta.get("retrieval_rank", "weight") == "weight"
            and TOP_K == 30 and expected_keys == LEGACY_V1_KEYS
            and FEATURES == LEGACY_V1_FEATURES and meta.get("features") == LEGACY_V1_FEATURES
        )
        if not legacy_compatible:
            raise ValueError("missing retrieval version is supported only for original v1/weight: retrain first")
    elif meta["retrieval_version"] != version or meta.get("retrieval_rank") != rank:
        raise ValueError("retrieval version or ranking changed since training: retrain first")


def write_lists(path, s1: pl.DataFrame, tg: pl.DataFrame, pairs: pl.DataFrame, header: str) -> None:
    """One row per S1 entity in test-file order; ids by descending probability; '' when none."""
    ids = (pairs.join(tg.select(pl.col("row").alias("t"), pl.col("id").alias("tid")), on="t")
           .sort(["s1", "p", "t"], descending=[False, True, False])
           .group_by("s1", maintain_order=True).agg(pl.col("tid").str.join(",").alias(header)))
    out = (s1.select(pl.col("row").alias("s1"), pl.col("id").alias("source1_entity_id"))
           .join(ids, on="s1", how="left").with_columns(pl.col(header).fill_null("")))
    out.select("source1_entity_id", header).write_csv(path, separator="\t", quote_style="never")


def main() -> None:
    t0 = time.time()
    meta = json.loads((WORK_DIR / "meta.json").read_text())
    if meta["features"] != FEATURES:
        raise SystemExit("feature list changed since training: retrain first")
    validate_retrieval_config(meta)
    model = lgb.Booster(model_file=str(WORK_DIR / "model.txt"))
    s1, tg = load("test")
    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"blocking index built in {time.time() - t:.0f}s; keys dropped over cap: {blocker.dropped}",
          flush=True)
    rows = np.arange(len(s1), dtype=np.uint32)
    scored = []
    for i in range(0, len(rows), CHUNK):
        t = time.time()
        cand = blocker.candidates(rows[i:i + CHUNK])
        feats = build(cand, s1, tg)
        p = model.predict(matrix(feats), num_threads=WORKERS)
        scored.append(feats.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32)))
        print(f"  scored rows {i:,}+{len(rows[i:i + CHUNK]):,}: {len(cand):,} pairs "
              f"in {time.time() - t:.0f}s", flush=True)
        del feats, cand
    del blocker
    scored = pl.concat(scored)
    matches = decide(scored, meta["threshold"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lists(OUT_DIR / "candidate_pairs.tsv", s1, tg, scored, "candidate_entity_ids")
    write_lists(OUT_DIR / "matching_results.tsv", s1, tg, matches, "matched_entity_ids")
    n_with = matches["s1"].n_unique()
    stats = {
        "test_entities": len(s1), "candidate_pairs": len(scored),
        "candidates_per_entity": len(scored) / len(s1),
        "zero_candidate_entities": len(s1) - scored["s1"].n_unique(),
        "matches": len(matches), "empty_share": 1 - n_with / len(s1),
        "threshold": meta["threshold"], "seconds": time.time() - t0,
    }
    (WORK_DIR / "predict_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
