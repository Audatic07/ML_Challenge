"""Train the LightGBM pair matcher on training candidates and pick the threshold.

Train on candidates of TRAIN_SAMPLE random train S1 entities, validate on a disjoint
VAL_SAMPLE: perm[:TRAIN_SAMPLE] and perm[TRAIN_SAMPLE:TRAIN_SAMPLE+VAL_SAMPLE] of
np.random.default_rng(42).permutation(len(train_source1)) in file order.
Validation F0.5 counts singletons and true matches lost at blocking.
Writes WORK_DIR/model.txt, WORK_DIR/meta.json and WORK_DIR/val_s1_ids.csv.

Run:  python -m src.train
"""
from __future__ import annotations

import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from .blocking import Blocker
from .config import (CHUNK, KEY_TYPES, LGB_PARAMS, ROUNDS, SEED, THRESH_GRID, TOP_K,
                     TRAIN_SAMPLE, VAL_SAMPLE, WORK_DIR, WORKERS)
from .features import FEATURES, build, matrix
from .metric import decide, f05_macro, oracle
from .prepare import load, load_truth

warnings.filterwarnings("ignore", category=DeprecationWarning)


def labelled_pairs(blocker: Blocker, rows: np.ndarray, s1, tg, truth) -> pl.DataFrame:
    out = []
    for i in range(0, len(rows), CHUNK):
        t = time.time()
        cand = blocker.candidates(rows[i:i + CHUNK])
        out.append(build(cand, s1, tg))
        print(f"  features rows {i:,}+{len(rows[i:i + CHUNK]):,}: {len(cand):,} pairs "
              f"in {time.time() - t:.0f}s", flush=True)
    df = pl.concat(out)
    lab = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
    return df.join(lab, on=["s1", "t"], how="left").with_columns(pl.col("y").fill_null(0))


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
    perm = np.random.default_rng(SEED).permutation(len(s1))
    tr_rows = np.sort(perm[:TRAIN_SAMPLE]).astype(np.uint32)
    va_rows = np.sort(perm[TRAIN_SAMPLE:TRAIN_SAMPLE + VAL_SAMPLE]).astype(np.uint32)
    s1.select("id").gather(pl.Series(va_rows)).rename({"id": "source1_entity_id"}).write_csv(
        WORK_DIR / "val_s1_ids.csv")

    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"blocking index built in {time.time() - t:.0f}s; keys dropped over cap: {blocker.dropped}",
          flush=True)
    tr = labelled_pairs(blocker, tr_rows, s1, tg, truth)
    va = labelled_pairs(blocker, va_rows, s1, tg, truth)
    del blocker
    stats_tr = report_blocking("train", tr_rows, tr, truth)
    stats_va = report_blocking("val", va_rows, va, truth)

    t = time.time()
    dtr = lgb.Dataset(matrix(tr), tr["y"].to_numpy(), feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(matrix(va), va["y"].to_numpy(), reference=dtr)
    params = dict(LGB_PARAMS, num_threads=WORKERS)
    booster = lgb.train(params, dtr, num_boost_round=ROUNDS, valid_sets=[dva], valid_names=["val"],
                        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    print(f"trained {booster.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)

    p = booster.predict(matrix(va), num_iteration=booster.best_iteration, num_threads=WORKERS)
    scored = va.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32))
    truth_va = truth.filter(pl.col("s1").is_in(pl.Series(va_rows)))
    grid = [(thr, f05_macro(va_rows, truth_va, decide(scored, thr))) for thr in THRESH_GRID]
    best_thr, best_f = max(grid, key=lambda x: x[1])
    for thr, f in grid:
        print(f"  threshold {thr:.3f}: val F0.5 {f:.4f}")
    print(f"best threshold {best_thr} -> val macro F0.5 {best_f:.4f}", flush=True)

    gain = booster.feature_importance("gain")
    top = sorted(zip(FEATURES, gain), key=lambda x: -x[1])[:15]
    print("top features by gain:", [f for f, _ in top])

    booster.save_model(str(WORK_DIR / "model.txt"), num_iteration=booster.best_iteration)
    meta = {
        "features": FEATURES, "threshold": best_thr, "val_f05": best_f,
        "threshold_grid": grid, "best_iteration": booster.best_iteration,
        "params": params, "top_k": TOP_K, "key_types": KEY_TYPES,
        "train_sample": TRAIN_SAMPLE, "val_sample": VAL_SAMPLE, "seed": SEED,
        "blocking_train": stats_tr, "blocking_val": stats_va,
        "feature_gain": {f: float(g) for f, g in zip(FEATURES, gain)},
        "seconds": time.time() - t0,
    }
    (WORK_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
