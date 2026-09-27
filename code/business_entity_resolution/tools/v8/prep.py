"""V8 step 1 (CPU notebook): SC-Block inputs. Training groups come from FIT S1 only (each S1 with its labelled S2/S3
matches, one owner per target, so groups are disjoint); tune and test pair lists with serialized texts.
Run from code/business_entity_resolution:  python tools/v8/prep.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "tools/v6_analysis")
import stage2_release as sr  # noqa: E402
from src.v4_train import partitions  # noqa: E402

W = Path.home() / "SageMaker/work"
D = W / "data/student_resource/dataset"
OUT = W / "v8"
OUT.mkdir(exist_ok=True)


def ser(raw: pl.DataFrame) -> pl.Series:
    """SC-Block serialization: [COL] attribute [VAL] value."""
    return raw.select((pl.lit("[COL] name [VAL] ") + pl.col("business_name").fill_null("") + pl.lit(" [COL] address [VAL] ")
                       + pl.col("business_address").fill_null("")).alias("text"))["text"]


plan = json.loads((W / "ens3/plan.json").read_text())
parts = partitions(plan["train_queries"], plan["fit"], plan["stop"], plan["tune"])
fit = np.sort(np.asarray(parts[1])).astype(np.uint32)
tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
assert not set(fit.tolist()) & set(tune.tolist()), "fit and tune overlap"
q_train, t_train = sr.read_tsv(D / "train/train_source1.tsv"), sr.target_frame(D, "train")
tid = t_train.select(pl.col("entity_id").alias("target_id")).with_row_index("t")
truth = sr.tune_truth(D, fit).join(tid, on="target_id")          # fit S1 labels only
rows = truth["s1"].unique().sort()
groups = pl.concat([pl.DataFrame({"g": rows, "text": ser(q_train.gather(rows))}),
                    pl.DataFrame({"g": truth["s1"], "text": ser(t_train.gather(truth["t"]))})])
groups.write_parquet(OUT / "train_groups.parquet")
print(f"train groups {rows.len():,} from {len(fit):,} fit S1; records {len(groups):,}", flush=True)

_, _, kept, _ = sr.load_ens3(W / "ens3", W / "members")
pairs = kept.select("s1", "t")
pairs.write_parquet(OUT / "tune_pairs.parquet")
qs, ts = pairs["s1"].unique().sort(), pairs["t"].unique().sort()
pl.DataFrame({"s1": qs, "text": ser(q_train.gather(qs))}).write_parquet(OUT / "tune_q.parquet")
pl.DataFrame({"t": ts, "text": ser(t_train.gather(ts))}).write_parquet(OUT / "tune_t.parquet")
print(f"tune pairs {len(pairs):,}, q {len(qs):,}, t {len(ts):,}", flush=True)
del q_train, t_train

pairs = pl.concat([pl.read_parquet(p, columns=["s1", "t"]) for p in sorted((W / "scores").glob("*.parquet"))])
assert len(pairs) == 9_266_800
pairs.write_parquet(OUT / "test_pairs.parquet")
q_test, t_test = sr.read_tsv(D / "test/test_source1.tsv"), sr.target_frame(D, "test")
qs, ts = pairs["s1"].unique().sort(), pairs["t"].unique().sort()
pl.DataFrame({"s1": qs, "text": ser(q_test.gather(qs))}).write_parquet(OUT / "test_q.parquet")
pl.DataFrame({"t": ts, "text": ser(t_test.gather(ts))}).write_parquet(OUT / "test_t.parquet")
print(f"test pairs {len(pairs):,}, q {len(qs):,}, t {len(ts):,}\nPREP_DONE", flush=True)
