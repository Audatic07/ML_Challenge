"""Candidate oracle U and link recall for finished train feature shards (no model needed)."""
import sys
from pathlib import Path
import boto3
import polars as pl
B = "amazon-sagemaker-580857071542-ap-south-1-dn0trrxacr3ddt"
P = sys.argv[1]
shards = sys.argv[2:]
W = Path(r"C:\Users\Heroa\AppData\Local\Temp\claude\C--Ml-challenge\166c9404-5585-40bd-8634-27d523922360\scratchpad\shards")
W.mkdir(exist_ok=True)
D = Path(r"C:\Ml_challenge\student_resource\dataset\train")
s3 = boto3.client("s3", region_name="ap-south-1")
opts = dict(separator="\t", quote_char=None, infer_schema=False)
ids = pl.read_csv(D / "train_source1.tsv", columns=["entity_id"], **opts).with_row_index("s1")
gt_all = pl.read_csv(D / "train_ground_truth.tsv", **opts)
for shard in shards:
    path = W / f"{P.replace('/', '_')}_{shard}.parquet"
    if not path.exists():
        s3.download_file(B, f"{P}/feat/{shard}.parquet", str(path))
    f = pl.read_parquet(path, columns=["s1", "target_id", "bscore", "name_char", "address_char", "name_fold", "address_fold"])
    task = __import__("json").loads(s3.get_object(Bucket=B, Key=f"{P}/tasks/{shard}.json")["Body"].read())
    rows = pl.Series("s1", task["rows"], dtype=pl.UInt32)
    q = ids.filter(pl.col("s1").is_in(rows.implode()))
    gt = (gt_all.join(q.rename({"entity_id": "source1_entity_id"}), on="source1_entity_id")
          .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
          .filter(pl.col("matched_entity_ids") != "").select("s1", pl.col("matched_entity_ids").alias("target_id")))
    hit = f.select("s1", "target_id").join(gt, on=["s1", "target_id"])
    g = gt.group_by("s1").len().rename({"len": "g"})
    r = hit.group_by("s1").len().rename({"len": "r"})
    per = pl.DataFrame({"s1": rows}).join(g, on="s1", how="left").join(r, on="s1", how="left").fill_null(0)
    per = per.with_columns(pl.when(pl.col("g") == 0).then(1.0).otherwise(5 * pl.col("r") / (4 * pl.col("r") + pl.col("g"))).alias("u"))
    print(f"{shard}: queries={len(rows)} links={len(gt)} recall={len(hit)/max(1,len(gt)):.4f} U={per['u'].mean():.5f} "
          f"pairs/q={len(f)/len(rows):.1f} all-found={(per.filter(pl.col('g')>0)['r'] == per.filter(pl.col('g')>0)['g']).mean():.4f}")
    # per-channel contribution: which channels found the true links
    tp = f.join(gt.with_columns(pl.lit(1).alias("y")), on=["s1", "target_id"])
    for ch in ("name_char", "address_char", "name_fold", "address_fold"):
        only = tp.filter((pl.col(ch) > 0) & (pl.sum_horizontal([pl.col(c) > 0 for c in ("name_char", "address_char", "name_fold", "address_fold") if c != ch]) == 0))
        print(f"   {ch}: finds {tp.filter(pl.col(ch) > 0).height/len(gt):.4f} of links; sole finder of {len(only)/len(gt):.4f}")
