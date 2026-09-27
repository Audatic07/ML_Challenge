"""Per-country thresholds on saved V7 test scores (the file behind leaderboard 0.972 and its stricter variants).

    python tools/v7_thresholds.py --scores OUT_B4 --data D --out DIR --france 0.7375,0.80,0.85 [--us-india 0.7375]

Reads the per-shard predict outputs (s1, t, target_id, s), applies one owner per target exactly as collect does (highest
s, ties to the lower S1 row), then a threshold per country (France vs US/India), and writes one matching_results TSV per
setting with label-free per-country counts and its SHA-256. France at 0.7375 with US/India 0.7375 reproduces the
unrouted B4 file 97ceedb9... Validate every output with the official and strict validators before uploading.
"""
import argparse
import glob
import hashlib
from pathlib import Path

import polars as pl


def read_tsv(path, **kwargs):
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--france", default="0.7375")
    ap.add_argument("--us-india", default="0.7375")
    a = ap.parse_args()
    q = read_tsv(Path(a.data) / "test/test_source1.tsv", columns=["entity_id", "country"]).with_row_index("s1")
    s = pl.concat([pl.read_parquet(p, columns=["s1", "t", "target_id", "s"]) for p in sorted(glob.glob(f"{a.scores}/*.parquet"))])
    owner = s.sort(["t", "s", "s1"], descending=[False, True, False]).group_by("t", maintain_order=True).head(1).join(
        q.select("s1", "country"), on="s1")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for ui in map(float, a.us_india.split(",")):
        for fr in map(float, a.france.split(",")):
            keep = owner.filter(pl.col("s") >= pl.when(pl.col("country") == "France").then(fr).otherwise(ui))
            rows = keep.group_by("s1").agg(pl.col("target_id").sort().str.join(",").alias("matched_entity_ids"))
            out = q.join(rows, on="s1", how="left").select(pl.col("entity_id").alias("source1_entity_id"),
                                                          pl.col("matched_entity_ids").fill_null(""))
            path = Path(a.out) / f"matching_ui{ui}_fr{fr}.tsv"
            out.write_csv(path, separator="\t", quote_style="never")
            k = (out.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element() != ""))
                                  .list.len().alias("k")).join(q, left_on="source1_entity_id", right_on="entity_id"))
            stats = {r["country"]: round(r["k"], 3) for r in k.group_by("country").agg(pl.col("k").mean()).to_dicts()}
            print(f"US/India {ui} France {fr}: mean predicted {stats} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)


if __name__ == "__main__":
    main()
