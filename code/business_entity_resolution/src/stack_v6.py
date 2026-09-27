"""v6 matcher experiments on cached v3-style features (exp_v3 output).

Compares, on the same shortlisted candidates:
  lgb      LightGBM on the v3 features
  xgb      XGBoost (hist) on the same features
  blend    mean of lgb and xgb probabilities
  sibling  second-stage LightGBM that adds sibling features built from first-pass
           out-of-fold probabilities: true variants of one business support each
           other, while a decoy competes with them.
The validation queries are split in two fixed halves: `tune` picks the threshold and
early stopping, `check` is only scored. Reported numbers are macro F0.5 over every
query (singletons and blocking misses included).

Run:  python -m src.stack_v6
"""
from __future__ import annotations

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl
import xgboost as xgb
from rapidfuzz import fuzz, process

from . import features3 as F3
from .config import LGB_PARAMS, SEED, WORK_DIR, WORKERS
from .exp_v3 import query_sets
from .metric import decide, f05_macro, per_entity
from .config import CACHE_DIR
from .prepare import load_truth

warnings.filterwarnings("ignore")
EXP = WORK_DIR / "exp"
OUT = EXP / "stack"
ROUNDS = int(os.environ.get("ER_ROUNDS", 2000))
BASE = F3.FEATURES + ["p1"]
SIB = ["p0", "p0_logit", "p0_max_other", "p0_sum_other", "p0_n_hi_other", "p0_rank", "p0_gap_top",
       "p0_second", "top_a_tset", "top_n_tset", "top_same_house", "top_same_core", "top_is_self",
       "p0_sum_all", "p0_expected_k"]
GRID = [round(x, 3) for x in np.arange(0.30, 0.951, 0.01)]


def xmat(df: pl.DataFrame, cols) -> np.ndarray:
    return df.select(cols).to_numpy().astype(np.float32)


def train_lgb(tr, cols, va=None, rounds=ROUNDS, params=None):
    p = dict(LGB_PARAMS, num_threads=WORKERS, **(params or {}))
    dtr = lgb.Dataset(xmat(tr, cols), tr["y"].to_numpy(), feature_name=cols)
    if va is None:
        return lgb.train(p, dtr, num_boost_round=rounds)
    dva = lgb.Dataset(xmat(va, cols), va["y"].to_numpy(), reference=dtr)
    return lgb.train(p, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(75, verbose=False), lgb.log_evaluation(500)])


def train_xgb(tr, cols, va):
    dtr = xgb.DMatrix(xmat(tr, cols), label=tr["y"].to_numpy(), feature_names=cols, nthread=WORKERS)
    dva = xgb.DMatrix(xmat(va, cols), label=va["y"].to_numpy(), feature_names=cols, nthread=WORKERS)
    params = {"objective": "binary:logistic", "eval_metric": "logloss", "tree_method": "hist",
              "eta": 0.05, "max_depth": 0, "grow_policy": "lossguide", "max_leaves": 127,
              "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.9, "lambda": 1.0,
              "max_bin": 256, "nthread": WORKERS, "seed": SEED}
    return xgb.train(params, dtr, num_boost_round=ROUNDS, evals=[(dva, "tune")],
                     early_stopping_rounds=75, verbose_eval=500)


def pred_xgb(model, df, cols):
    return model.predict(xgb.DMatrix(xmat(df, cols), feature_names=cols, nthread=WORKERS),
                         iteration_range=(0, model.best_iteration + 1))


def sibling_features(df: pl.DataFrame, tg: pl.DataFrame) -> pl.DataFrame:
    """df has s1, t, p0. Adds SIB columns computed inside each query's candidate set."""
    df = df.with_columns(pl.col("p0").rank("ordinal", descending=True).over("s1").alias("_r"))
    top = (df.filter(pl.col("_r") <= 2).select("s1", "t", "_r")
           .pivot(on="_r", index="s1", values="t").rename({"1": "t1", "2": "t2"}))
    if "t2" not in top.columns:
        top = top.with_columns(pl.lit(None, pl.UInt32).alias("t2"))
    df = df.join(top, on="s1", how="left")
    other = pl.when(pl.col("t") == pl.col("t1")).then(pl.col("t2")).otherwise(pl.col("t1"))
    df = df.with_columns(other.alias("_o"), (pl.col("t") == pl.col("t1")).cast(pl.Float32).alias("top_is_self"))
    has = df["_o"].is_not_null().to_numpy()
    o = df["_o"].fill_null(0)
    views = tg.select("addr_norm", "name_norm", "house", "core")
    a, b = views.gather(df["t"]), views.gather(o)

    def sim(col):
        v = process.cpdist(a[col].fill_null("").to_list(), b[col].fill_null("").to_list(),
                           scorer=fuzz.token_set_ratio, workers=WORKERS, dtype=np.float32)
        empty = (a[col].fill_null("") == "").to_numpy() | (b[col].fill_null("") == "").to_numpy()
        return np.where(~has | empty, -1, v)

    same = lambda c: np.where(~has | a[c].is_null().to_numpy() | b[c].is_null().to_numpy(), -1,
                              (a[c] == b[c]).fill_null(False).to_numpy().astype(np.float32))
    df = df.with_columns(pl.Series("top_a_tset", sim("addr_norm")), pl.Series("top_n_tset", sim("name_norm")),
                         pl.Series("top_same_house", same("house")), pl.Series("top_same_core", same("core")))
    p = pl.col("p0")
    n_hi = (p > 0.5).cast(pl.Float32)
    return df.with_columns(
        (p.clip(1e-6, 1 - 1e-6) / (1 - p.clip(1e-6, 1 - 1e-6))).log().alias("p0_logit"),
        pl.when(pl.len().over("s1") > 1).then(
            pl.when(pl.col("_r") == 1).then(p.top_k(2).min().over("s1")).otherwise(p.max().over("s1"))
        ).otherwise(0.0).alias("p0_max_other"),
        (p.sum().over("s1") - p).alias("p0_sum_other"),
        (n_hi.sum().over("s1") - n_hi).alias("p0_n_hi_other"),
        pl.col("_r").cast(pl.Float32).alias("p0_rank"),
        (p.max().over("s1") - p).alias("p0_gap_top"),
        pl.when(pl.len().over("s1") > 1).then(p.top_k(2).min().over("s1")).otherwise(0.0).alias("p0_second"),
        p.sum().over("s1").alias("p0_sum_all"),
        (p >= 0.5).sum().over("s1").cast(pl.Float32).alias("p0_expected_k"),
    ).drop("_r", "t1", "t2", "_o").with_columns([pl.col(c).cast(pl.Float32) for c in SIB])


def evaluate(name, scored, parts, truth, thr=None):
    """Pick the threshold on `tune` (unless given), report tune/check/all."""
    res = {}
    t_rows, c_rows = parts["tune"], parts["check"]
    sub = lambda rows: scored.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
    tr = lambda rows: truth.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
    if thr is None:
        st, tt = sub(t_rows), tr(t_rows)
        thr = max(GRID, key=lambda x: f05_macro(t_rows, tt, decide(st, x)))
    for k, rows in (("tune", t_rows), ("check", c_rows), ("all", np.concatenate([t_rows, c_rows]))):
        res[k] = f05_macro(rows, tr(rows), decide(sub(rows), thr))
    res["threshold"] = thr
    print(f"{name:10s} thr {thr:.2f} | tune {res['tune']:.4f} | check {res['check']:.4f} | all {res['all']:.4f}", flush=True)
    return res


def main() -> None:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    # Only the columns this step needs: the machine is memory-constrained.
    s1 = pl.read_parquet(CACHE_DIR / "train_s1.parquet", columns=["row", "id"])
    tg = pl.read_parquet(CACHE_DIR / "train_tg.parquet", columns=["row", "id", "addr_norm", "name_norm", "house", "core"])
    truth = load_truth(s1, tg)
    sets = query_sets(len(s1))
    va_rows = sets["val"]
    perm = np.random.default_rng(7).permutation(va_rows)
    parts = {"tune": np.sort(perm[: len(perm) // 2]), "check": np.sort(perm[len(perm) // 2:])}
    truth = truth.filter(pl.col("s1").is_in(pl.Series(va_rows, dtype=pl.UInt32).implode()))
    views = tg.select("addr_norm", "name_norm", "house", "core")
    del s1
    tr = pl.read_parquet(EXP / "feat_train.parquet")
    va = pl.read_parquet(EXP / "feat_val.parquet")
    in_tune = pl.col("s1").is_in(pl.Series(parts["tune"], dtype=pl.UInt32).implode())
    va_tune = va.filter(in_tune)
    results = {}

    # 1. LightGBM on base features (early stopping on the tune half)
    t = time.time()
    m_lgb = train_lgb(tr, BASE, va_tune)
    p_lgb = m_lgb.predict(xmat(va, BASE), num_iteration=m_lgb.best_iteration, num_threads=WORKERS)
    print(f"lgb: {m_lgb.best_iteration} rounds, {time.time() - t:.0f}s", flush=True)
    results["lgb"] = evaluate("lgb", va.select("s1", "t").with_columns(pl.Series("p", p_lgb, dtype=pl.Float32)), parts, truth)
    m_lgb.save_model(str(OUT / "lgb.txt"), num_iteration=m_lgb.best_iteration)

    # 2. XGBoost on the same features
    t = time.time()
    m_xgb = train_xgb(tr, BASE, va_tune)
    p_xgb = pred_xgb(m_xgb, va, BASE)
    print(f"xgb: {m_xgb.best_iteration + 1} rounds, {time.time() - t:.0f}s", flush=True)
    results["xgb"] = evaluate("xgb", va.select("s1", "t").with_columns(pl.Series("p", p_xgb, dtype=pl.Float32)), parts, truth)
    m_xgb.save_model(str(OUT / "xgb.json"))
    results["blend"] = evaluate("blend", va.select("s1", "t").with_columns(
        pl.Series("p", (p_lgb + p_xgb) / 2, dtype=pl.Float32)), parts, truth)

    # 3. Sibling second stage: out-of-fold first-pass p0 on train (2 folds by query)
    t = time.time()
    q = tr["s1"].unique().sort().to_numpy()
    fold_of = pl.DataFrame({"s1": q, "fold": (np.random.default_rng(11).permutation(len(q)) % 2).astype(np.int8)})
    tr = tr.join(fold_of, on="s1")
    p0_tr = np.zeros(len(tr), dtype=np.float32)
    p0_va = np.zeros(len(va), dtype=np.float32)
    rounds0 = max(200, m_lgb.best_iteration)
    for f in (0, 1):
        fit = tr.filter(pl.col("fold") != f)
        m = train_lgb(fit, BASE, rounds=rounds0)
        idx = (tr["fold"] == f).to_numpy()
        p0_tr[idx] = m.predict(xmat(tr.filter(pl.col("fold") == f), BASE), num_threads=WORKERS)
        p0_va += m.predict(xmat(va, BASE), num_threads=WORKERS) / 2
        m.save_model(str(OUT / f"first_fold{f}.txt"))
    print(f"out-of-fold first pass: {time.time() - t:.0f}s", flush=True)
    tr2 = sibling_features(tr.select("s1", "t").with_columns(pl.Series("p0", p0_tr)), views)
    va2 = sibling_features(va.select("s1", "t").with_columns(pl.Series("p0", p0_va)), views)
    tr = tr.join(tr2, on=["s1", "t"])
    va = va.join(va2, on=["s1", "t"])
    va_tune = va.filter(in_tune)
    results["first_pass_folds"] = evaluate("p0 folds", va.select("s1", "t", pl.col("p0").alias("p")), parts, truth)
    t = time.time()
    m_sib = train_lgb(tr, BASE + SIB, va_tune)
    p_sib = m_sib.predict(xmat(va, BASE + SIB), num_iteration=m_sib.best_iteration, num_threads=WORKERS)
    print(f"sibling: {m_sib.best_iteration} rounds, {time.time() - t:.0f}s", flush=True)
    results["sibling"] = evaluate("sibling", va.select("s1", "t").with_columns(pl.Series("p", p_sib, dtype=pl.Float32)), parts, truth)
    m_sib.save_model(str(OUT / "sibling.txt"), num_iteration=m_sib.best_iteration)
    results["sibling+xgb"] = evaluate("sib+xgb", va.select("s1", "t").with_columns(
        pl.Series("p", (p_sib + p_xgb) / 2, dtype=pl.Float32)), parts, truth)

    best = max(results, key=lambda k: results[k]["tune"])
    per = per_entity(parts["check"], truth.filter(pl.col("s1").is_in(pl.Series(parts["check"], dtype=pl.UInt32).implode())),
                     decide(va.select("s1", "t").with_columns(pl.Series("p", p_sib, dtype=pl.Float32))
                            .filter(pl.col("s1").is_in(pl.Series(parts["check"], dtype=pl.UInt32).implode())),
                            results["sibling"]["threshold"]))
    per.write_parquet(OUT / "check_per_query_sibling.parquet")
    va.select("s1", "t", "y", "p0").with_columns(pl.Series("p_lgb", p_lgb), pl.Series("p_xgb", p_xgb),
                                                  pl.Series("p_sib", p_sib)).write_parquet(OUT / "val_preds.parquet")
    gain = sorted(zip(BASE + SIB, m_sib.feature_importance("gain")), key=lambda x: -x[1])
    (OUT / "results.json").write_text(json.dumps({
        "results": results, "best_on_tune": best, "sibling_rounds": m_sib.best_iteration,
        "lgb_rounds": m_lgb.best_iteration, "xgb_rounds": m_xgb.best_iteration + 1, "first_pass_rounds": rounds0,
        "parts": {k: len(v) for k, v in parts.items()}, "top_gain_sibling": [g for g, _ in gain[:25]],
        "base_features": BASE, "sibling_features": SIB, "seconds": time.time() - t0}, indent=2))
    print(f"best on tune: {best}; total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
