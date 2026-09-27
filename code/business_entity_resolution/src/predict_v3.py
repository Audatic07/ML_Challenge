"""v3 test inference: rescue blocking -> stage-1 shortlist -> v3 features -> matcher -> decision.

candidate_pairs.tsv is the stage-1 shortlist: exactly the pairs the final matcher scores.
Needs the v3 caches (prepare_v3.py) and WORK_DIR/exp/{stage1.txt, model_v3.txt, result.json}
from exp_v3.py. Resumable: finished chunks are kept in WORK_DIR/exp/test_scored/.

Run:  python -m src.predict_v3
"""
from __future__ import annotations

import json
import os
import time
import warnings

import lightgbm as lgb
import numpy as np
import polars as pl

from . import features3 as F3
from . import stage1
from .blocking import Blocker
from .config import OUT_DIR, WORK_DIR, WORKERS
from .metric import decide
from .predict import write_lists
from .prepare import load

warnings.filterwarnings("ignore", category=DeprecationWarning)
EXP = WORK_DIR / "exp"
CH = int(os.environ.get("ER_CHUNK", 10_000))


def main() -> None:
    t0 = time.time()
    res = json.loads((EXP / "result.json").read_text())
    feats, thr = res["features"], res["threshold"]
    m1 = lgb.Booster(model_file=str(EXP / "stage1.txt"))
    m2 = lgb.Booster(model_file=str(EXP / "model_v3.txt"))
    if m2.feature_name() != feats or m1.feature_name() != stage1.CHEAP:
        raise SystemExit("saved models do not match the current feature lists")
    s1, tg = load("test")
    idf_n, idf_a = F3.idf_table(tg, "core"), F3.idf_table(tg, "addr_tok")
    t = time.time()
    blocker = Blocker(s1, tg)
    print(f"test blocking index {time.time() - t:.0f}s", flush=True)
    out_dir = EXP / "test_scored"
    out_dir.mkdir(exist_ok=True)
    rows = np.arange(len(s1), dtype=np.uint32)
    for i in range(0, len(rows), CH):
        path = out_dir / f"chunk_{i:08d}.parquet"
        if path.exists():
            continue
        t = time.time()
        cand = blocker.candidates(rows[i:i + CH])
        short = stage1.shortlist(stage1.cheap_features(cand, s1, tg), m1)
        f = F3.build(short.select("s1", "t", "bscore", "kmask"), s1, tg, idf_n, idf_a)
        f = f.join(short.select("s1", "t", "p1"), on=["s1", "t"])
        p = m2.predict(f.select(feats).to_numpy().astype(np.float32), num_threads=WORKERS)
        f.select("s1", "t").with_columns(pl.Series("p", p, dtype=pl.Float32)).write_parquet(path)
        print(f"  rows {i:,}: {len(cand):,} blocked -> {len(f):,} scored in {time.time() - t:.0f}s", flush=True)
    del blocker
    scored = pl.concat([pl.read_parquet(p) for p in sorted(out_dir.glob("chunk_*.parquet"))])
    if scored["s1"].n_unique() + 0 > len(s1) or scored.select("s1", "t").is_duplicated().any():
        raise SystemExit("scored chunks overlap: delete test_scored/ and rerun")
    matches = decide(scored, thr)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_lists(OUT_DIR / "candidate_pairs.tsv", s1, tg, scored, "candidate_entity_ids")
    write_lists(OUT_DIR / "matching_results.tsv", s1, tg, matches, "matched_entity_ids")
    stats = {"test_entities": len(s1), "candidate_pairs": len(scored),
             "candidates_per_entity": len(scored) / len(s1),
             "zero_candidate_entities": len(s1) - scored["s1"].n_unique(),
             "matches": len(matches), "empty_share": 1 - matches["s1"].n_unique() / len(s1),
             "threshold": thr, "seconds": time.time() - t0}
    (EXP / "predict_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
