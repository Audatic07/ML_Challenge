"""Train the LightGBM pair matcher on training candidates and pick the threshold.

Train on TRAIN_SAMPLE queries outside the fixed canonical validation interval and
locked audit reserve. VAL_SAMPLE selects a nested subset of the original v1
validation queries, independently of TRAIN_SAMPLE (see splits.py).
Validation F0.5 counts singletons and true matches lost at blocking.
Writes WORK_DIR/model.txt, WORK_DIR/meta.json and WORK_DIR/val_s1_ids.csv.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import Blocker
from .config import (CHUNK, KEY_TYPES, LGB_PARAMS, ROUNDS, SEED, THRESH_GRID, TOP_K,
                     TRAIN_SAMPLE, VAL_SAMPLE, WORK_DIR, WORKERS)
from .features import FEATURES, build, matrix
from .metric import decide, f05_macro, oracle, per_entity
from .prepare import load, load_truth
from .splits import make_split, split_manifest

warnings.filterwarnings("ignore", category=DeprecationWarning)


def labelled_pairs(blocker: Blocker, rows: np.ndarray, s1, tg, truth,
                   excluded_targets: pl.DataFrame | None = None) -> pl.DataFrame:
    out = []
    for i in range(0, len(rows), CHUNK):
        t = time.time()
        cand = blocker.candidates(rows[i:i + CHUNK])
        out.append(build(cand, s1, tg))
        print(f"  features rows {i:,}+{len(rows[i:i + CHUNK]):,}: {len(cand):,} pairs "
              f"in {time.time() - t:.0f}s", flush=True)
    df = pl.concat(out)
    lab = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
    labelled = df.join(lab, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))
    # Build query-context features on the full retrieval union, then keep held-out
    # target records out of supervised fitting, including negative examples.
    if excluded_targets is not None:
        labelled = labelled.join(excluded_targets, on="t", how="anti")
    return labelled


def held_out_targets(truth: pl.DataFrame, fit_rows: np.ndarray,
                     reserved_rows: np.ndarray) -> pl.DataFrame:
    """Exclude every target linked to reserved queries; fail on a split business."""
    excluded = (truth.filter(pl.col("s1").is_in(pl.Series(reserved_rows, dtype=pl.UInt32).implode()))
                .select("t").unique())
    shared = (truth.filter(pl.col("s1").is_in(pl.Series(fit_rows, dtype=pl.UInt32).implode()))
              .join(excluded, on="t", how="semi"))
    if len(shared):
        raise ValueError("labelled targets cross fit/held-out queries; create business-group splits first")
    return excluded


def report_blocking(name: str, rows: np.ndarray, pairs: pl.DataFrame, truth: pl.DataFrame) -> dict:
    tr = truth.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32)))
    found = int(pairs["y"].sum())
    n_cand = pairs.group_by("s1").len()
    stats = {
        "entities": len(rows), "true_pairs": len(tr), "found": found,
        "recall": found / max(len(tr), 1), "pairs": len(pairs),
        "pairs_per_entity": len(pairs) / len(rows),
        "zero_candidate_entities": len(rows) - len(n_cand),
        "oracle_f05": oracle(rows, tr, pairs.filter(pl.col("y") == 1)),
    }
    print(f"blocking {name}: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                             for k, v in stats.items()}), flush=True)
    return stats


def main() -> None:
    t0 = time.time()
    s1, tg = load("train")
    truth = load_truth(s1, tg)
    print(f"loaded {len(s1):,} S1 queries and full {len(tg):,} target catalog", flush=True)
    split = make_split(len(s1), TRAIN_SAMPLE, VAL_SAMPLE, SEED)
    tr_rows, va_rows = split.train, split.validation
    manifest = split_manifest(split, s1["id"])
    excluded_targets = held_out_targets(truth, tr_rows, split.held_out)
    (WORK_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    s1.select("id").gather(pl.Series(va_rows)).rename({"id": "source1_entity_id"}).write_csv(
        WORK_DIR / "val_s1_ids.csv")

    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"blocking index built in {time.time() - t:.0f}s; keys dropped over cap: {blocker.dropped}",
          flush=True)
    tr = labelled_pairs(blocker, tr_rows, s1, tg, truth, excluded_targets=excluded_targets)
    va = labelled_pairs(blocker, va_rows, s1, tg, truth)
    del blocker
    stats_tr = report_blocking("train", tr_rows, tr, truth)
    stats_va = report_blocking("val", va_rows, va, truth)
    truth_va = truth.filter(pl.col("s1").is_in(pl.Series(va_rows)))
    countries = s1.select(pl.col("row").alias("s1"), "country")
    excluded_target_count = len(excluded_targets)
    del s1, tg, truth, excluded_targets

    t = time.time()
    dtr = lgb.Dataset(matrix(tr), tr["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(matrix(va), va["y"].to_numpy(), reference=dtr)
    params = dict(LGB_PARAMS, num_threads=WORKERS)
    booster = lgb.train(params, dtr, num_boost_round=ROUNDS, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {booster.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)

    p = booster.predict(matrix(va), num_iteration=booster.best_iteration, num_threads=WORKERS)
    scored = va.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32))
    grid = [(thr, f05_macro(va_rows, truth_va, decide(scored, thr))) for thr in THRESH_GRID]
    best_thr, best_f = max(grid, key=lambda x: x[1])
    for thr, f in grid:
        print(f"  threshold {thr:.3f}: val F0.5 {f:.4f}")
    print(f"best threshold {best_thr} -> val macro F0.5 {best_f:.4f}", flush=True)
    per = per_entity(va_rows, truth_va, decide(scored, best_thr))
    per_oracle = per_entity(va_rows, truth_va, va.filter(pl.col("y") == 1))
    per = per.join(per_oracle.select("s1", pl.col("f").alias("oracle_f05")), on="s1").join(
        countries, on="s1")
    per.write_parquet(WORK_DIR / "val_per_query.parquet")

    gain = booster.feature_importance("gain")
    top = sorted(zip(FEATURES, gain), key=lambda x: -x[1])[:15]
    print("top features by gain:", [f for f, _ in top])

    booster.save_model(str(WORK_DIR / "model.txt"), num_iteration=booster.best_iteration)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    meta = {
        "partition": "development; not locked audit or leaderboard",
        "source_sha256": source_hashes,
        "model_sha256": hashlib.sha256((WORK_DIR / "model.txt").read_bytes()).hexdigest(),
        "country_validation": per.group_by("country").agg(pl.len().alias("queries"),
            pl.col("f").mean().alias("macro_f05"), pl.col("oracle_f05").mean()).to_dicts(),
        "features": FEATURES, "threshold": best_thr, "val_f05": best_f,
        "threshold_grid": grid, "best_iteration": booster.best_iteration,
        "params": params, "top_k": TOP_K, "key_types": KEY_TYPES,
        "retrieval_version": getattr(config, "RETRIEVAL_VERSION", "2"),
        "retrieval_rank": getattr(config, "RETRIEVAL_RANK", "weight"),
        "train_sample": TRAIN_SAMPLE, "val_sample": VAL_SAMPLE, "seed": SEED,
        "split": manifest, "excluded_held_out_targets": excluded_target_count,
        "blocking_train": stats_tr, "blocking_val": stats_va,
        "feature_gain": {f: float(g) for f, g in zip(FEATURES, gain)},
        "seconds": time.time() - t0,
    }
    (WORK_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
