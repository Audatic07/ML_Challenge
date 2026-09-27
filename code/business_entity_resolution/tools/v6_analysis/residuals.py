"""Categorize the errors left after stage 2 at its tuned threshold.

    python tools/v6_analysis/residuals.py [threshold] [tag]

Categories (label-free descriptions of each pair):
  name_disjoint   no shared core-name token between S1 and candidate
  addr_missing    S1 or candidate address empty
  house_diff      both house numbers present and different
  sib_addr_agree  a survivor sibling with stage-1 p >= 0.5 has address token-set ratio >= 90
  sib_name_agree  the same for the core name
"""
import sys

import polars as pl

from _common import WORK

threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.651
tag = sys.argv[2] if len(sys.argv) > 2 else "K10_f0"
oof = pl.read_parquet(WORK / f"s2_oof_{tag}.parquet")
feat = pl.read_parquet(WORK / f"s2_tune_{tag}.parquet").drop("y", "p1")
x = oof.join(feat, on=["s1", "t", "target_id"]).with_columns(
    (pl.col("q_n_shared") == 0).alias("name_disjoint"),
    (pl.col("q_a_tset") < 0).alias("addr_missing"),
    (pl.col("q_hs_eq") == 0).alias("house_diff"),
    (pl.col("sibc_a_tset") >= 90).alias("sib_addr_agree"),
    (pl.col("sibc_n_tset") >= 90).alias("sib_name_agree"))
groups = {"rejected": x.filter((pl.col("y") == 1) & (pl.col("p2") < threshold)),
          "false_pos": x.filter((pl.col("y") == 0) & (pl.col("p2") >= threshold)),
          "accepted": x.filter((pl.col("y") == 1) & (pl.col("p2") >= threshold)),
          "negatives": x.filter(pl.col("y") == 0)}
print({k: len(v) for k, v in groups.items()})
for c in ("name_disjoint", "addr_missing", "house_diff", "sib_addr_agree", "sib_name_agree"):
    print(f"{c:15s} " + " ".join(f"{k}={v[c].mean():.3f}" for k, v in groups.items()))
for k in ("rejected", "false_pos"):
    print(k, groups[k].group_by("name_disjoint", "addr_missing").len().sort("len", descending=True).rows())
    print(k, "p2 deciles 10/25/50/75/90:", [round(groups[k]["p2"].quantile(q), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
sub = x.filter(pl.col("name_disjoint") & ~pl.col("addr_missing"))
print(f"name-disjoint, address present: n={len(sub):,} positive rate={sub['y'].mean():.4f}")
print(sub.group_by("sib_addr_agree", (pl.col("q_a_tset") >= 90).alias("addr_same_as_s1")).agg(
    pl.len(), pl.col("y").mean()).sort("len", descending=True).rows())
sub = x.filter(pl.col("addr_missing"))
print(f"address missing: n={len(sub):,} positive rate={sub['y'].mean():.4f}")
print(sub.group_by((pl.col("q_n_tset") >= 90).alias("name_close"), "sib_name_agree").agg(
    pl.len(), pl.col("y").mean()).sort("len", descending=True).rows())
