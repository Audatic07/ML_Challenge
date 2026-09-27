"""v3 development experiment: rescue blocking -> stage-1 shortlist -> v3 features -> LightGBM.

Query sets (all from the training split, disjoint):
  s1fit  : fits the stage-1 shortlist model
  train  : fits the final matcher
  val    : prefix of the canonical validation interval perm[200000:300000]
Every stage is cached under WORK_DIR/exp so feature/model changes can be re-run alone.

Run:  python -m src.exp_v3 [blocking|shortlist|features|model]   (default: all missing stages)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from . import features3 as F3
from . import stage1
from . import numeric_pairs
from .blocking import NUMERIC, Blocker
from .config import CACHE_DIR, KEY_TYPES, LGB_PARAMS, SEED, THRESH_GRID, WORK_DIR, WORKERS
from .metric import decide, f05_macro, oracle, per_entity
from .prepare import load, load_truth
from .splits import make_split

warnings.filterwarnings("ignore", category=DeprecationWarning)
N_TRAIN = int(os.environ.get("ER_EXP_TRAIN", 60_000))
N_S1FIT = int(os.environ.get("ER_EXP_S1FIT", 20_000))
N_VAL = int(os.environ.get("ER_EXP_VAL", 20_000))
ROUNDS = int(os.environ.get("ER_ROUNDS", 1500))
CH = int(os.environ.get("ER_CHUNK", 10_000))
EXP = WORK_DIR / "exp"


def query_sets(n_rows: int) -> dict[str, np.ndarray]:
    sp = make_split(n_rows, N_TRAIN + N_S1FIT, N_VAL, SEED)
    rng = np.random.default_rng(SEED + 1)
    fit = rng.permutation(sp.train)
    return {"s1fit": np.sort(fit[:N_S1FIT]), "train": np.sort(fit[N_S1FIT:]), "val": sp.validation}


def label(df: pl.DataFrame, truth: pl.DataFrame) -> pl.DataFrame:
    return (df.join(truth.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "t"], how="left")
            .with_columns(pl.col("y").fill_null(0)))


def oracle_report(name, rows, df, truth) -> float:
    tr = truth.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
    u = oracle(rows, tr, df.filter(pl.col("y") == 1))
    print(f"  {name}: {len(df):,} pairs ({len(df) / len(rows):.1f}/query), recall "
          f"{df['y'].sum() / len(tr):.4f}, oracle U {u:.4f}", flush=True)
    return u


def main(stages: list[str]) -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_s1 = pl.scan_parquet(CACHE_DIR / "train_s1.parquet").select(pl.len()).collect().item()
    sets = query_sets(n_s1)
    use_numeric = any(tag in KEY_TYPES for tag, _, _ in NUMERIC)
    if use_numeric:  # before the full frames are loaded: this step is memory-light on its own
        numeric_pairs.build("train", EXP / "numeric", np.concatenate(list(sets.values())))
    s1, tg = load("train")
    truth = load_truth(s1, tg)

    if "blocking" in stages or not all((EXP / f"cand_{k}.parquet").exists() for k in sets):
        t = time.time()
        blocker = Blocker(s1, tg)
        print(f"blocking index {time.time() - t:.0f}s", flush=True)
        for k, rows in sets.items():
            c = pl.concat([blocker.candidates(
                rows[i:i + 20_000],
                extra=numeric_pairs.load(EXP / "numeric", rows[i:i + 20_000]) if use_numeric else None)
                for i in range(0, len(rows), 20_000)])
            c = label(c, truth)
            c.write_parquet(EXP / f"cand_{k}.parquet")
            oracle_report(f"blocking {k}", rows, c, truth)
        del blocker

    if "shortlist" in stages or not all((EXP / f"short_{k}.parquet").exists() for k in ("train", "val")):
        t = time.time()

        def cheap_chunks(k):
            c = pl.read_parquet(EXP / f"cand_{k}.parquet")
            for q in np.array_split(sets[k], max(1, len(sets[k]) // CH)):
                sub = c.filter(pl.col("s1").is_in(pl.Series(q, dtype=pl.UInt32).implode()))
                yield stage1.cheap_features(sub.drop("y"), s1, tg).join(sub.select("s1", "t", "y"), on=["s1", "t"])

        model = stage1.train(pl.concat(list(cheap_chunks("s1fit"))))
        model.save_model(str(EXP / "stage1.txt"))
        for k in ("train", "val"):
            # keep the 60 best per query only; the shortlist itself is the top SHORTLIST of these
            full = label(pl.concat([stage1.shortlist(ch.drop("y"), model, n=60) for ch in cheap_chunks(k)]), truth)
            for n in (20, 30, 40, 60):
                oracle_report(f"shortlist {k} top{n}", sets[k],
                              full.group_by("s1", maintain_order=True).head(n), truth)
            full.write_parquet(EXP / f"ranked_{k}.parquet")
            full.group_by("s1", maintain_order=True).head(stage1.SHORTLIST).write_parquet(EXP / f"short_{k}.parquet")
        print(f"shortlist stage {time.time() - t:.0f}s", flush=True)

    if "features" in stages or not all((EXP / f"feat_{k}" / "done.txt").exists() for k in ("train", "val")):
        t = time.time()
        idf_n, idf_a = F3.idf_table(tg, "core"), F3.idf_table(tg, "addr_tok")
        for k in ("train", "val"):
            sh = pl.read_parquet(EXP / f"short_{k}.parquet")
            (EXP / f"feat_{k}").mkdir(exist_ok=True)
            for j, q in enumerate(np.array_split(sets[k], max(1, len(sets[k]) // CH))):
                sub = sh.filter(pl.col("s1").is_in(pl.Series(q, dtype=pl.UInt32).implode()))
                f = F3.build(sub.select("s1", "t", "bscore", "kmask"), s1, tg, idf_n, idf_a)
                f.join(sub.select("s1", "t", "p1", "y"), on=["s1", "t"]).write_parquet(EXP / f"feat_{k}" / f"{j:04d}.parquet")
            (EXP / f"feat_{k}" / "done.txt").write_text("ok")
            print(f"  features {k} done {time.time() - t:.0f}s", flush=True)

    feats = F3.FEATURES + ["p1"]
    va_rows = sets["val"]
    perm = np.random.default_rng(7).permutation(va_rows)
    parts = {"tune": np.sort(perm[: len(perm) // 2]), "check": np.sort(perm[len(perm) // 2:])}
    truth_va = truth.filter(pl.col("s1").is_in(pl.Series(va_rows, dtype=pl.UInt32).implode()))
    countries = s1.select(pl.col("row").alias("s1"), "country")
    del s1, tg, truth
    read = lambda k: pl.scan_parquet(EXP / f"feat_{k}" / "*.parquet").select(["s1", "t", "y"] + feats).collect()
    tr = read("train")
    x_tr, y_tr = tr.select(feats).to_numpy().astype(np.float32), tr["y"].to_numpy()
    print(f"train rows {len(tr):,}, positives {int(y_tr.sum()):,}", flush=True)
    del tr
    va = read("val")
    in_tune = va["s1"].is_in(pl.Series(parts["tune"], dtype=pl.UInt32).implode()).to_numpy()
    x_va, y_va = va.select(feats).to_numpy().astype(np.float32), va["y"].to_numpy()
    t = time.time()
    dtr = lgb.Dataset(x_tr, y_tr, feature_name=feats, free_raw_data=True)
    dva = lgb.Dataset(x_va[in_tune], y_va[in_tune], reference=dtr)
    booster = lgb.train(dict(LGB_PARAMS, num_threads=WORKERS), dtr, num_boost_round=ROUNDS,
                        valid_sets=[dva], valid_names=["tune"],
                        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(250)])
    del dtr, x_tr
    print(f"trained {booster.best_iteration} rounds in {time.time() - t:.0f}s", flush=True)
    p = booster.predict(x_va, num_iteration=booster.best_iteration, num_threads=WORKERS)
    scored = va.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32))
    sub = lambda d, rows: d.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
    fine = [round(x, 3) for x in np.arange(0.3, 0.951, 0.01)]
    tt = sub(truth_va, parts["tune"])
    best_thr = max(fine, key=lambda x: f05_macro(parts["tune"], tt, decide(sub(scored, parts["tune"]), x)))
    res = {k: f05_macro(rows, sub(truth_va, rows), decide(sub(scored, rows), best_thr))
           for k, rows in (("tune", parts["tune"]), ("check", parts["check"]), ("all", va_rows))}
    best_f = res["all"]
    u = oracle(va_rows, truth_va, va.filter(pl.col("y") == 1))
    print(f"VAL tune {res['tune']:.4f} | CHECK {res['check']:.4f} | all {res['all']:.4f} at threshold {best_thr} "
          f"(picked on tune) | candidate oracle U {u:.4f}", flush=True)
    per = per_entity(va_rows, truth_va, decide(scored, best_thr)).join(countries, on="s1")
    print(f"VAL macro F0.5 {best_f:.4f} at threshold {best_thr} | candidate oracle U {u:.4f} "
          f"| matching loss {u - best_f:.4f}", flush=True)
    print(per.group_by("country").agg(pl.len(), pl.col("f").mean()).sort("country"))
    gain = sorted(zip(feats, booster.feature_importance("gain")), key=lambda x: -x[1])
    print("top gain:", [f for f, _ in gain[:25]])
    booster.save_model(str(EXP / "model_v3.txt"), num_iteration=booster.best_iteration)
    scored.write_parquet(EXP / "val_scored.parquet")
    (EXP / "result.json").write_text(json.dumps({"val_f05": best_f, "val_tune": res["tune"], "val_check": res["check"],
                                                 "threshold": best_thr, "oracle": u,
                                                 "best_iteration": booster.best_iteration,
                                                 "features": feats, "sets": {k: len(v) for k, v in sets.items()},
                                                 "seconds": time.time() - t0}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
