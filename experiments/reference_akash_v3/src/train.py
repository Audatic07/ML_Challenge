"""Train the pair classifier and tune the decision threshold.

Steps
  1. Normalise train sources (cached) and build blocking keys.
  2. Sample disjoint train / validation sets of Source 1 entities.
  3. Generate candidates + features for both, label with the ground truth.
  4. Fit LightGBM with early stopping on the validation pairs.
  5. Pick the probability threshold that maximises macro F0.5 on validation,
     counting singletons and matches lost at blocking.
Run:  python -m src.train
"""
import json
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from . import config
from .blocking import prepare_blocking
from .features import FEATURES
from .metric import decide, macro_f05
from .pipeline import iter_scored_chunks
from .prepare import load_split, read_tsv

warnings.filterwarnings("ignore", category=DeprecationWarning)


def load_truth(s1, s23):
    """Ground truth as (idx1, idx2) pairs."""
    gt = read_tsv(config.DATA_DIR / "train" / "train_ground_truth.tsv")
    pairs = (gt.filter(pl.col("matched_entity_ids") != "")
             .with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != ""))
    return (pairs.join(s1.select(pl.col("entity_id").alias("source1_entity_id"), pl.col("idx").alias("idx1")),
                       on="source1_entity_id")
            .join(s23.select(pl.col("entity_id").alias("matched_entity_ids"), pl.col("idx").alias("idx2")),
                  on="matched_entity_ids")
            .select("idx1", "idx2"))


def build_pairs(s1_idx, keys1, keys23, s1, s23, truth):
    """Candidates + features + label for a set of Source 1 indices."""
    parts = list(iter_scored_chunks(s1_idx, keys1, keys23, s1, s23))
    df = pl.concat(parts)
    return (df.join(truth.with_columns(pl.lit(1, dtype=pl.Int8).alias("y")), on=["idx1", "idx2"], how="left")
            .with_columns(pl.col("y").fill_null(0)))


def report_ceiling(va, truth, va_idx, s1):
    """Candidate oracle ceiling U: the macro F0.5 a perfect matcher would get on these candidates.

    For an entity with g true matches of which r were retrieved, the best achievable score is
    5r / (4r + g) (1.0 for singletons). If U is low, fix blocking before tuning the model.
    Also prints blocking recall per country, to show where matches are lost.
    """
    base = pl.DataFrame({"idx1": va_idx}).cast(pl.UInt32)
    vt = truth.join(base, on="idx1")
    g = vt.group_by("idx1").agg(pl.len().alias("g"))
    r = va.filter(pl.col("y") == 1).group_by("idx1").agg(pl.len().alias("r"))
    d = (base.join(g, on="idx1", how="left").join(r, on="idx1", how="left").fill_null(0)
         .join(s1.select(pl.col("idx").alias("idx1"), "country"), on="idx1", how="left"))
    u = pl.when(pl.col("g") == 0).then(1.0).otherwise(5 * pl.col("r") / (4 * pl.col("r") + pl.col("g")))
    d = d.with_columns(u.alias("u"))
    print(f"  val candidate oracle ceiling U = {d['u'].mean():.4f}", flush=True)
    by = (d.group_by("country").agg(pl.len().alias("n"), pl.col("r").sum(), pl.col("g").sum(), pl.col("u").mean())
          .with_columns((pl.col("r") / pl.col("g")).alias("recall")).sort("n", descending=True))
    for row in by.iter_rows(named=True):
        print(f"    {row['country']}: n={row['n']:,} recall={row['recall']:.4f} U={row['u']:.4f}", flush=True)


def main():
    t0 = time.time()
    print("Loading and normalising train data...", flush=True)
    s1, s23 = load_split("train")
    truth = load_truth(s1, s23)

    n_per = truth.group_by("idx1").len()
    singleton_share = 1 - n_per.height / s1.height
    multi = truth.group_by("idx2").len().filter(pl.col("len") > 1).height
    unique_assign = multi / max(truth.height, 1) < 0.001
    print(f"  S1={s1.height:,} S2+S3={s23.height:,} true pairs={truth.height:,}")
    print(f"  singleton share={singleton_share:.3f}; matches per non-singleton: "
          f"mean={n_per['len'].mean():.2f} max={n_per['len'].max()}")
    print(f"  S2/S3 records linked to >1 S1: {multi:,} -> unique_assign={unique_assign}", flush=True)

    print("Building blocking keys...", flush=True)
    keys1, keys23 = prepare_blocking(s1, s23)
    print(f"  keys: S1 {keys1.height:,}, S2/S3 after cap {keys23.height:,} ({time.time() - t0:.0f}s)", flush=True)

    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(s1.height)
    tr_idx = perm[:config.TRAIN_S1_SAMPLE]
    va_idx = perm[config.TRAIN_S1_SAMPLE:config.TRAIN_S1_SAMPLE + config.VAL_S1_SAMPLE]

    print(f"Train pairs ({len(tr_idx):,} S1)...", flush=True)
    tr = build_pairs(tr_idx, keys1, keys23, s1, s23, truth)
    print(f"Validation pairs ({len(va_idx):,} S1)...", flush=True)
    va = build_pairs(va_idx, keys1, keys23, s1, s23, truth)

    for name, idx, df in [("train", tr_idx, tr), ("val", va_idx, va)]:
        n_true = truth.filter(pl.col("idx1").is_in(pl.Series(idx).cast(pl.UInt32).implode())).height
        print(f"  {name}: {df.height:,} pairs, positives {int(df['y'].sum()):,}/{n_true:,} "
              f"-> blocking recall {df['y'].sum() / max(n_true, 1):.4f}", flush=True)
    report_ceiling(va, truth, va_idx, s1)

    print("Training LightGBM...", flush=True)
    dtr = lgb.Dataset(tr.select(FEATURES).to_numpy(), tr["y"].to_numpy(), feature_name=FEATURES)
    dva = lgb.Dataset(va.select(FEATURES).to_numpy(), va["y"].to_numpy(), reference=dtr)
    params = dict(config.LGB_PARAMS, num_threads=config.N_WORKERS, seed=config.SEED)
    model = lgb.train(params, dtr, config.LGB_ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    va_scored = va.select("idx1", "idx2").with_columns(
        pl.Series("p", model.predict(va.select(FEATURES).to_numpy(), num_threads=config.N_WORKERS)))
    va_truth = truth.filter(pl.col("idx1").is_in(pl.Series(va_idx).cast(pl.UInt32).implode()))
    best_t, best_f = 0.5, -1.0
    for t in np.round(np.arange(0.20, 0.96, 0.025), 3):
        f = macro_f05(decide(va_scored, t, unique_assign), va_truth, va_idx)
        if f > best_f:
            best_t, best_f = float(t), f
    print(f"Validation macro F0.5 = {best_f:.4f} at threshold {best_t}", flush=True)

    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda x: -x[1])[:12]
    print("Top features by gain:", ", ".join(f"{k}" for k, _ in imp))

    config.WORK_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.WORK_DIR / "model.txt"))
    meta = {"threshold": best_t, "val_f05": best_f, "unique_assign": unique_assign, "top_k": config.TOP_K,
            "features": FEATURES, "best_iteration": model.best_iteration}
    (config.WORK_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model to {config.WORK_DIR}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
