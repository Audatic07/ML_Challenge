"""Re-score the saved v3 model's validation predictions without retraining (read-only)."""
import json

import polars as pl

from src.config import WORK_DIR
from src.exp_v3 import query_sets
from src.metric import decide, oracle, per_entity
from src.prepare import load, load_truth

EXP = WORK_DIR / "exp"
res = json.loads((EXP / "result.json").read_text())
print(f"saved result.json : val F0.5 {res['val_f05']:.4f} | threshold {res['threshold']} "
      f"| oracle U {res['oracle']:.4f} | rounds {res['best_iteration']} | sets {res['sets']}")

s1, tg = load("train")
truth = load_truth(s1, tg)
rows = query_sets(len(s1))["val"]
truth_va = truth.filter(pl.col("s1").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
scored = pl.read_parquet(EXP / "val_scored.parquet")
cand = pl.read_parquet(EXP / "feat_val.parquet", columns=["s1", "t", "y"])

per = per_entity(rows, truth_va, decide(scored, res["threshold"])).join(
    s1.select(pl.col("row").alias("s1"), "country"), on="s1")
f, u = per["f"].mean(), oracle(rows, truth_va, cand.filter(pl.col("y") == 1))
print(f"recomputed now    : val F0.5 {f:.4f} | oracle U {u:.4f} | matching loss {u - f:.4f} "
      f"| {len(rows):,} validation queries, {len(tg):,} target records")

print("\nBy country:")
print(per.group_by("country").agg(pl.len().alias("queries"), pl.col("f").mean().alias("F0.5")).sort("country"))

print("\nWhere the score is lost:")
lost = per.filter(pl.col("f") < 1).with_columns(
    pl.when(pl.col("g") == 0).then(pl.lit("predicted a match for a singleton"))
    .when(pl.col("p") == 0).then(pl.lit("predicted nothing, a true match exists"))
    .when(pl.col("tp") == 0).then(pl.lit("predicted only wrong records"))
    .otherwise(pl.lit("partly right (missed or extra records)")).alias("error"))
print(lost.group_by("error").agg(pl.len().alias("queries"),
                                 ((1 - pl.col("f")).sum() / len(rows)).alias("F0.5 lost"))
      .sort("F0.5 lost", descending=True))

print("\nThreshold sweep (the saved model uses %s):" % res["threshold"])
for thr in (0.5, 0.6, 0.65, 0.69, 0.72, 0.75, 0.8):
    print(f"  {thr:.2f} -> {per_entity(rows, truth_va, decide(scored, thr))['f'].mean():.4f}")
