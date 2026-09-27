"""v6 test inference: numeric-rescue blocking -> stage-1 shortlist -> v3 features ->
matcher chosen by stack_v6 (lgb, xgb, blend, sibling or sibling+xgb) -> decision.

candidate_pairs.tsv is the stage-1 shortlist, exactly the pairs the matcher scores.
Resumable: finished chunks are kept in WORK_DIR/exp/test_scored_<mode>/.

Run:  python -m src.predict_v6 [mode]     (default: best mode on the tune half)
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
import xgboost as xgb

from . import features3 as F3
from . import numeric_pairs, stage1
from .blocking import NUMERIC, Blocker
from .config import CACHE_DIR, KEY_TYPES, OUT_DIR, WORK_DIR, WORKERS
from .metric import decide
from .predict import write_lists
from .prepare import load
from .stack_v6 import BASE, SIB, pred_xgb, sibling_features, xmat

warnings.filterwarnings("ignore")
EXP = WORK_DIR / "exp"
STACK = EXP / "stack"
CH = int(os.environ.get("ER_CHUNK", 10_000))


def main(argv: list[str]) -> None:
    t0 = time.time()
    if (argv and argv[0] == "single") or not (STACK / "results.json").exists():
        mode = "single"  # the exp_v3 LightGBM matcher and its tune-half threshold
        thr = json.loads((EXP / "result.json").read_text())["threshold"]
    else:
        res = json.loads((STACK / "results.json").read_text())
        mode = argv[0] if argv else res["best_on_tune"]
        thr = res["results"][mode]["threshold"]
    print(f"mode {mode}, threshold {thr}", flush=True)
    m1 = lgb.Booster(model_file=str(EXP / "stage1.txt"))
    if m1.feature_name() != stage1.CHEAP:
        raise SystemExit("stage-1 model does not match the current blocking keys")
    models = {}
    if mode == "single":
        models["single"] = lgb.Booster(model_file=str(EXP / "model_v3.txt"))
    if mode in ("lgb", "blend"):
        models["lgb"] = lgb.Booster(model_file=str(STACK / "lgb.txt"))
    if mode in ("xgb", "blend", "sibling+xgb"):
        models["xgb"] = xgb.Booster()
        models["xgb"].load_model(str(STACK / "xgb.json"))
    if mode.startswith("sibling"):
        models["folds"] = [lgb.Booster(model_file=str(STACK / f"first_fold{f}.txt")) for f in (0, 1)]
        models["sib"] = lgb.Booster(model_file=str(STACK / "sibling.txt"))
    use_numeric = any(tag in KEY_TYPES for tag, _, _ in NUMERIC)
    if use_numeric:  # memory-light, before the full frames are loaded
        numeric_pairs.build("test", EXP / "numeric_test")
    s1, tg = load("test")
    views = tg.select("addr_norm", "name_norm", "house", "core")
    idf_n, idf_a = F3.idf_table(tg, "core"), F3.idf_table(tg, "addr_tok")
    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"test blocking index {time.time() - t:.0f}s", flush=True)
    out_dir = EXP / f"test_scored_{mode.replace('+', '_')}"
    out_dir.mkdir(exist_ok=True)
    rows = np.arange(len(s1), dtype=np.uint32)
    for i in range(0, len(rows), CH):
        path = out_dir / f"chunk_{i:08d}.parquet"
        if path.exists():
            continue
        t = time.time()
        q = rows[i:i + CH]
        cand = blocker.candidates(q, extra=numeric_pairs.load(EXP / "numeric_test", q) if use_numeric else None)
        short = stage1.shortlist(stage1.cheap_features(cand, s1, tg), m1)
        f = F3.build(short.select("s1", "t", "bscore", "kmask"), s1, tg, idf_n, idf_a)
        f = f.join(short.select("s1", "t", "p1"), on=["s1", "t"])
        x = xmat(f, BASE)
        if mode == "single":
            p = models["single"].predict(x, num_threads=WORKERS)
        elif mode == "lgb":
            p = models["lgb"].predict(x, num_threads=WORKERS)
        elif mode == "xgb":
            p = pred_xgb(models["xgb"], f, BASE)
        elif mode == "blend":
            p = (models["lgb"].predict(x, num_threads=WORKERS) + pred_xgb(models["xgb"], f, BASE)) / 2
        else:
            p0 = sum(m.predict(x, num_threads=WORKERS) for m in models["folds"]) / 2
            sib = sibling_features(f.select("s1", "t").with_columns(pl.Series("p0", p0, dtype=pl.Float32)), views)
            f = f.join(sib, on=["s1", "t"])
            p = models["sib"].predict(xmat(f, BASE + SIB), num_threads=WORKERS)
            if mode == "sibling+xgb":
                p = (p + pred_xgb(models["xgb"], f, BASE)) / 2
        f.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32)).write_parquet(path)
        print(f"  rows {i:,}: {len(cand):,} blocked -> {len(f):,} scored in {time.time() - t:.0f}s", flush=True)
    del blocker
    scored = pl.concat([pl.read_parquet(p) for p in sorted(out_dir.glob("chunk_*.parquet"))])
    if scored.select("s1", "t").is_duplicated().any():
        raise SystemExit("scored chunks overlap: delete the chunk folder and rerun")
    matches = decide(scored, thr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lists(OUT_DIR / "candidate_pairs.tsv", s1, tg, scored, "candidate_entity_ids")
    write_lists(OUT_DIR / "matching_results.tsv", s1, tg, matches, "matched_entity_ids")
    stats = {"mode": mode, "test_entities": len(s1), "candidate_pairs": len(scored),
             "candidates_per_entity": len(scored) / len(s1),
             "zero_candidate_entities": len(s1) - scored["s1"].n_unique(),
             "matches": len(matches), "empty_share": 1 - matches["s1"].n_unique() / len(s1),
             "threshold": thr, "seconds": time.time() - t0}
    (EXP / f"predict_stats_{mode.replace('+', '_')}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
