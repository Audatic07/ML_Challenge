"""Label-free test checks on the v5.1 full-union stage-1 scores (WORK/test_scores).

    python tools/v6_analysis/test_checks.py [examples]

1. Target competition: how many pairs at or above the threshold lose their target to
   another S1 with a higher probability, by country.
2. Predicted set sizes per country within the top 10, and the candidate counts left by
   probability floors.
3. With a positive `examples` count, prints sampled France S1 records with their top
   candidates (local inspection only; never commit the output).
Writes WORK/test_top10.parquet (s1, t, target_id, p, country).
"""
import sys

import polars as pl

from _common import DATA, THRESHOLD, WORK, read_tsv

examples = int(sys.argv[1]) if len(sys.argv) > 1 else 0
country = read_tsv(DATA / "test" / "test_source1.tsv", columns=["country"]).with_row_index("s1").with_columns(
    pl.col("s1").cast(pl.UInt32))
total, high, top = 0, [], []
for path in sorted((WORK / "test_scores").glob("*.parquet")):
    x = pl.read_parquet(path, columns=["s1", "t", "target_id", "p"])
    total += len(x)
    high.append(x.filter(pl.col("p") >= 0.3).select("s1", "t", "p"))
    top.append(x.sort(["s1", "p"], descending=[False, True]).group_by("s1", maintain_order=True).head(10))
high = pl.concat(high).with_columns(pl.col("p").max().over("t").alias("pmax"), pl.len().over("t").alias("claims"))
above = high.filter(pl.col("p") >= THRESHOLD)
lost = above.filter(pl.col("p") < pl.col("pmax")).join(country, on="s1")
print(f"test pairs={total:,}; pairs p>={THRESHOLD}: {len(above):,}; lost to a higher claimant: {len(lost):,} "
      f"({len(lost) / len(above):.4%})")
print("lost by country:", lost.group_by("country").len().sort("country").rows())
print("lost by p band:", lost.with_columns(pl.col("p").cut([0.9, 0.99]).alias("band")).group_by("band").len().sort("band").rows())
top = pl.concat(top).join(country, on="s1")
top.write_parquet(WORK / "test_top10.parquet")
q = top.group_by("s1", "country").agg((pl.col("p") >= THRESHOLD).sum().alias("k"), pl.col("p").max().alias("pmax"))
print(q.group_by("country").agg(pl.len(), pl.col("k").mean().alias("mean_predicted"),
                                 (pl.col("k") == 0).mean().alias("empty"), (pl.col("k") >= 8).sum().alias("k>=8"),
                                 pl.col("pmax").mean()).sort("country"))
n = top["s1"].n_unique()
for floor in (0.0, 0.001, 0.002, 0.005):
    kept = top.filter(pl.col("p") >= floor)
    print(f"top10 floor={floor}: candidates per S1={len(kept) / n:.2f}",
          kept.group_by("country").len().sort("country").rows())
if examples:
    s1 = read_tsv(DATA / "test" / "test_source1.tsv").with_row_index("s1").with_columns(pl.col("s1").cast(pl.UInt32))
    tg = pl.concat([read_tsv(DATA / "test" / f"test_source{s}.tsv") for s in (2, 3)]).rename({"entity_id": "target_id"})
    for r in q.filter(pl.col("country") == "France").sample(examples, seed=3).iter_rows(named=True):
        a = s1.filter(pl.col("s1") == r["s1"]).row(0, named=True)
        print(f"--- S1 {a['business_name']} | {a['business_address']}  (predicted {r['k']})")
        for c in top.filter(pl.col("s1") == r["s1"]).head(7).join(tg, on="target_id", how="left").iter_rows(named=True):
            print(f"   p={c['p']:.3f} {c['business_name']} | {c['business_address']}")
