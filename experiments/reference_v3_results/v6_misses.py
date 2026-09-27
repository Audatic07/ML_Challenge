"""Read-only: what do v6's remaining retrieval misses look like? Loads only needed columns."""
import re
import sys
from collections import Counter

import polars as pl

from src.prepare import read_tsv

V3, V6, DATA = sys.argv[1], sys.argv[2], sys.argv[3]
cols = ["row", "id", "name_norm", "core", "house", "addr_norm"]
s1 = pl.read_parquet(f"{V3}/train_s1.parquet", columns=cols)
tg = pl.read_parquet(f"{V3}/train_tg.parquet", columns=cols)
cand = pl.read_parquet(f"{V6}/exp/cand_val.parquet", columns=["s1", "t"])
rows = cand["s1"].unique()
gt = read_tsv(f"{DATA}/train/train_ground_truth.tsv")
gt = (gt.with_columns(pl.col("matched_entity_ids").str.split(",")).explode("matched_entity_ids")
      .join(s1.select(pl.col("id").alias("source1_entity_id"), pl.col("row").alias("s1")), on="source1_entity_id")
      .filter(pl.col("s1").is_in(rows.implode()))
      .join(tg.select(pl.col("id").alias("matched_entity_ids"), pl.col("row").alias("t")), on="matched_entity_ids")
      .select("s1", "t"))
miss = gt.join(cand, on=["s1", "t"], how="anti")
print(f"v6 val: {len(rows):,} queries, {len(gt):,} true pairs, missed {len(miss):,} ({len(miss)/len(gt):.2%})")
q = s1.select(pl.col("row").alias("s1"), pl.col("name_norm").alias("qn"), pl.col("core").alias("qc"), pl.col("house").alias("qh"))
t = tg.select(pl.col("row").alias("t"), pl.col("name_norm").alias("tn"), pl.col("core").alias("tc"),
              pl.col("house").alias("th"), pl.col("addr_norm").alias("ta"))
m = miss.join(q, on="s1").join(t, on="t")
tok = lambda c: pl.col(c).str.split(" ").list.unique()
m = m.with_columns(
    digit_typo=pl.col("tn").str.contains(r"[a-z][0-9]|[0-9][a-z]"),
    addr_empty=pl.col("ta").fill_null("").str.len_chars() == 0,
    house_missing=pl.col("qh").is_null() | pl.col("th").is_null(),
    same_house=pl.col("qh") == pl.col("th"),
    same_core=pl.col("qc") == pl.col("tc"),
    no_shared=tok("qc").list.set_intersection(tok("tc")).list.len() == 0)
print(m.select(pl.col("digit_typo", "addr_empty", "house_missing", "same_house", "same_core", "no_shared").mean()))

# Which letter does each digit replace? Align same-length token pairs that differ only at digit positions.
sub = Counter()
for qn, tn in m.filter("digit_typo").select("qn", "tn").iter_rows():
    for a in tn.split():
        if re.search(r"[a-z]", a) and re.search(r"\d", a):
            for b in qn.split():
                if len(a) == len(b) and all(x == y or x.isdigit() for x, y in zip(a, b)):
                    sub.update((x, y) for x, y in zip(a, b) if x.isdigit())
                    break
print("digit -> letter counts:", sorted(sub.items(), key=lambda kv: -kv[1])[:20])
pl.Config.set_tbl_width_chars(220); pl.Config.set_fmt_str_lengths(45); pl.Config.set_tbl_rows(20)
print(m.sample(min(20, len(m)), seed=3).select("qn", "tn", "qh", "th", "ta"))
