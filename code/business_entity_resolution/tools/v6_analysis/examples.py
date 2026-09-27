"""Print sampled tune errors with their records and true siblings (run loss.py first).

    python tools/v6_analysis/examples.py [count]

Prints supplied records to the terminal for local inspection. Never commit the output.
"""
import sys

import polars as pl

from _common import DATA, THRESHOLD, WORK, read_tsv

count = int(sys.argv[1]) if len(sys.argv) > 1 else 20
sc = pl.read_parquet(WORK / "tune_labeled.parquet")
s1 = read_tsv(DATA / "train" / "train_source1.tsv").with_row_index("row")
tg = pl.concat([read_tsv(DATA / "train" / f"train_source{s}.tsv") for s in (2, 3)])


def show(frame, title):
    print("=" * 20, title)
    for r in frame.sample(n=min(count, len(frame)), seed=1).iter_rows(named=True):
        a = s1.row(r["s1"], named=True)
        b = tg.filter(pl.col("entity_id") == r["t"]).row(0, named=True)
        print(f"p={r['p']:.3f} [{a['country']}] S1: {a['business_name']} | {a['business_address']}")
        print(f"      candidate {r['t']}: {b['business_name']} | {b['business_address']}")
        sib = sc.filter((pl.col("s1") == r["s1"]) & (pl.col("y") == 1) & (pl.col("t") != r["t"])).sort("p", descending=True)
        for o in sib.head(3).iter_rows(named=True):
            ob = tg.filter(pl.col("entity_id") == o["t"]).row(0, named=True)
            print(f"      true sibling p={o['p']:.3f}: {ob['business_name']} | {ob['business_address']}")


rejected = sc.filter((pl.col("y") == 1) & (pl.col("p") < THRESHOLD))
show(rejected.filter(pl.col("p") > 0.4), "rejected positives, 0.4 < p < threshold")
show(rejected.filter(pl.col("p") <= 0.1), "rejected positives, p <= 0.1")
show(sc.filter((pl.col("y") == 0) & (pl.col("p") >= THRESHOLD) & pl.col("owner")), "false positives")
