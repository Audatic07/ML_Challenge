"""2-fold cross-fitted stage-2 LightGBM on the tune survivors; compares with stage 1 at the same K.

    python tools/v6_analysis/stage2_cv.py [tag] [--rev] [--drop f1,f2]

tag defaults to K10_f0 (see stage2_features.py). --rev joins WORK/s2_rev_tune_{tag}.parquet
from reverse_tune.py. Folds split S1 by hash, so no S1 is scored by a model that saw it.
Writes WORK/s2_oof_{tag}[_rev].parquet (s1, t, target_id, y, p1, p2) and a JSON summary.
"""
import json
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from _common import WORK, macro

args = [a for a in sys.argv[1:] if not a.startswith("--")]
tag = args[0] if args else "K10_f0"
use_rev = "--rev" in sys.argv
drop = set(sys.argv[sys.argv.index("--drop") + 1].split(",")) if "--drop" in sys.argv else set()
started = time.time()
feat = pl.read_parquet(WORK / f"s2_tune_{tag}.parquet")
if use_rev:
    feat = feat.join(pl.read_parquet(WORK / f"s2_rev_tune_{tag}.parquet"), on=["s1", "target_id"], how="left")
per = pl.read_parquet(WORK / "tune_per.parquet").select("s1", "country", "g")
names = [c for c in feat.columns if c not in ("s1", "t", "target_id", "y") and c not in drop]
fold = (feat["s1"].hash(seed=7) % 2).to_numpy()
X = feat.select(names).to_numpy().astype(np.float32)
y = feat["y"].to_numpy()
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=20, seed=1)
oof = np.zeros(len(y), dtype=np.float32)
iterations = []
for f in (0, 1):
    train, test = np.where(fold != f)[0], np.where(fold == f)[0]
    inner = (pl.Series(feat["s1"].to_numpy()[train]).hash(seed=11) % 10 == 0).to_numpy()
    dtrain = lgb.Dataset(X[train[~inner]], y[train[~inner]], feature_name=names)
    dvalid = lgb.Dataset(X[train[inner]], y[train[inner]], reference=dtrain)
    booster = lgb.train(params, dtrain, 4000, valid_sets=[dvalid], callbacks=[lgb.early_stopping(150, verbose=False)])
    iterations.append(booster.best_iteration)
    oof[test] = booster.predict(X[test], num_iteration=booster.best_iteration)
    print(f"fold {f}: best_iteration={booster.best_iteration} t={time.time()-started:.0f}s", flush=True)
    if f == 0:
        gain = sorted(zip(booster.feature_importance("gain"), names), reverse=True)
        print("top gain (thousands):", [(n, round(v / 1e3)) for v, n in gain[:20]])
feat = feat.with_columns(pl.Series("p2", oof))
suffix = "_rev" if use_rev else ""
feat.select("s1", "t", "target_id", "y", "p1", "p2").write_parquet(WORK / f"s2_oof_{tag}{suffix}.parquet")

g = per.select("s1", "g", "country")


def sweep(col):
    best = (0.0, 0.0)
    for th in np.arange(0.30, 0.96, 0.01):
        score = macro(feat.filter(pl.col(col) >= th), g)[0].mean()
        best = max(best, (float(th), score), key=lambda item: item[1])
    for th in np.arange(best[0] - 0.01, best[0] + 0.0101, 0.001):
        score = macro(feat.filter(pl.col(col) >= th), g)[0].mean()
        best = max(best, (float(th), score), key=lambda item: item[1])
    return best


b1, b2 = sweep("p1"), sweep("p2")
print(f"{tag}{suffix}: stage 1 th={b1[0]:.3f} F={b1[1]:.5f} | stage 2 OOF th={b2[0]:.3f} F={b2[1]:.5f}")
f2, x = macro(feat.filter(pl.col("p2") >= b2[0]), g)
f1, _ = macro(feat.filter(pl.col("p1") >= b1[0]), g)
x = x.with_columns(pl.Series("stage2", f2), pl.Series("stage1", f1))
print(x.group_by("country").agg(pl.len(), pl.col("stage1").mean(), pl.col("stage2").mean()).sort("country"))
print(x.with_columns(pl.col("g").clip(upper_bound=5)).group_by("g").agg(
    pl.len(), pl.col("stage1").mean(), pl.col("stage2").mean()).sort("g"))
qfold = (x["s1"].hash(seed=7) % 2).to_numpy()
for fo in (0, 1):
    print(f"fold {fo}: stage 1 {f1[qfold == fo].mean():.5f} stage 2 {f2[qfold == fo].mean():.5f}")
for col, th in (("p1", b1[0]), ("p2", b2[0])):
    chosen = feat.filter(pl.col(col) >= th)
    print(f"{col}: FP={int((chosen['y'] == 0).sum()):,} TP={int(chosen['y'].sum()):,} "
          f"retrieved but rejected={int(feat['y'].sum() - chosen['y'].sum()):,}")
(WORK / f"s2_cv_{tag}{suffix}.json").write_text(json.dumps(
    {"tag": tag + suffix, "stage1": b1, "stage2": b2, "iterations": iterations, "features": names}, indent=1))
print(f"done t={time.time()-started:.0f}s")
