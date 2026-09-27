"""Loss decomposition of the v5.1 model on the full retrieved union (100k tune S1).

    python tools/v6_analysis/loss.py

Writes WORK/tune_labeled.parquet (s1, t=target id, p, y, owner) and WORK/tune_per.parquet
(per-S1 g, p, tp, r and F), then prints the decomposition, target ownership statistics and
the candidate oracle of stage-1 probability cut-offs (top-K and probability floors).
"""
import numpy as np
import polars as pl

from _common import DATA, THRESHOLD, WORK, explode_truth, per_query_f, read_tsv, tune_frame, tune_truth

tune = tune_frame()
gt = tune_truth(tune)
sc = pl.read_parquet(WORK / "tune_scores.parquet").select("s1", pl.col("target_id").alias("t"), "p")
sc = (sc.join(gt.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["s1", "t"], how="left")
      .with_columns(pl.col("y").fill_null(0), (pl.col("p") == pl.col("p").max().over("t")).alias("owner")))
sc.write_parquet(WORK / "tune_labeled.parquet")
g = gt.group_by("s1").len().rename({"len": "g"})
print(f"tune S1={len(tune):,} truth pairs={len(gt):,} scored pairs={len(sc):,}")

pred = sc.filter(pl.col("owner") & (pl.col("p") >= THRESHOLD))
tp = pred.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "tp"})
pp = pred.group_by("s1").len().rename({"len": "p"})
r = sc.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "r"})
per = (tune.join(g, on="s1", how="left").join(pp, on="s1", how="left").join(tp, on="s1", how="left")
       .join(r, on="s1", how="left").with_columns(pl.col("g", "p", "tp", "r").fill_null(0)))
G, P, TP, R = (per[c].to_numpy() for c in ("g", "p", "tp", "r"))
f = per_query_f(TP, P, G)
u = per_query_f(R, R, G)
per = per.with_columns(pl.Series("f", f), pl.Series("u", u), (pl.col("p") - pl.col("tp")).alias("fp"),
                       (pl.col("r") - pl.col("tp")).alias("fn_rej"), (pl.col("g") - pl.col("r")).alias("fn_miss"))
per.write_parquet(WORK / "tune_per.parquet")
n = len(per)
nofp = per_query_f(TP, TP, G)
norej = per_query_f(R, P - TP + R, G)
no_assign = sc.filter(pl.col("p") >= THRESHOLD)
f_na = per_query_f(*(tune.join(g, on="s1", how="left").join(
    no_assign.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "tp"}), on="s1", how="left").join(
    no_assign.group_by("s1").len().rename({"len": "p"}), on="s1", how="left").fill_null(0)[c].to_numpy()
    for c in ("tp", "p", "g")))
print(f"threshold {THRESHOLD}: F={f.mean():.5f} (without one-owner assignment {f_na.mean():.5f})  U={u.mean():.5f}")
print(f"total loss {1 - f.mean():.5f}; counterfactual gains (not additive): remove every FP {(nofp - f).sum() / n:.5f}, "
      f"accept every retrieved positive {(norej - f).sum() / n:.5f}; retrieval loss 1-U {1 - u.mean():.5f}")
bucket = (pl.when(pl.col("g") == 0).then(pl.lit("0")).when(pl.col("g") <= 2).then(pl.lit("1-2"))
          .when(pl.col("g") <= 4).then(pl.lit("3-4")).otherwise(pl.lit("5+")))
with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=200, float_precision=5):
    print(per.with_columns(bucket.alias("gb")).group_by("country", "gb").agg(
        pl.len().alias("n"), pl.col("f").mean().alias("F"), pl.col("u").mean().alias("U"),
        ((1 - pl.col("f")).sum() / n).alias("loss_share"), pl.col("fp").sum().alias("FP"),
        pl.col("fn_rej").sum().alias("FN_rejected"), pl.col("fn_miss").sum().alias("FN_missed"),
        pl.col("g").sum().alias("G")).sort("country", "gb"))
    print(per.group_by("country").agg(pl.len(), pl.col("f").mean(), pl.col("u").mean(), pl.col("fp").sum(),
                                      pl.col("fn_rej").sum(), pl.col("fn_miss").sum(), pl.col("g").sum()))
    pos = (sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("prank"))
           .filter(pl.col("y") == 1))
    print("positive p histogram:", pos.with_columns(pl.col("p").cut([0.05, 0.2, 0.4, 0.6, 0.7, THRESHOLD, 0.9]).alias("bin"))
          .group_by("bin").len().sort("bin").rows())
    neg = sc.filter(pl.col("y") == 0)
    print("negatives at or above p:", [(b, int((neg["p"] >= b).sum())) for b in (0.4, 0.6, 0.7, THRESHOLD, 0.9)])
    rej = pos.filter(pl.col("p") < THRESHOLD)
    print("rejected positives by rank of p within S1:", rej["prank"].value_counts().sort("prank").head(8).rows())
    acc = pos.group_by("s1").agg((pl.col("p") >= THRESHOLD).sum().alias("acc"), (pl.col("p") < THRESHOLD).sum().alias("rej"))
    print("S1 with a rejected positive, by accepted positives:", acc.filter(pl.col("rej") > 0)["acc"].value_counts().sort("acc").rows())
print(f"S1 with FP={int((per['fp'] > 0).sum()):,} rejected positive={int((per['fn_rej'] > 0).sum()):,} "
      f"missed positive={int((per['fn_miss'] > 0).sum()):,} F<1={int((per['f'] < 1).sum()):,}")
print("positives above threshold lost to one-owner assignment:",
      len(sc.filter((pl.col("y") == 1) & (pl.col("p") >= THRESHOLD) & ~pl.col("owner"))))

# Target ownership: every labelled target belongs to at most one S1.
links = explode_truth(read_tsv(DATA / "train" / "train_ground_truth.tsv")).select(
    pl.col("source1_entity_id").alias("owner_sid"), pl.col("matched_entity_ids").alias("t"))
print(f"labelled links={len(links):,} distinct targets={links['t'].n_unique():,}")
fp = pred.filter(pl.col("y") == 0).join(links, on="t", how="left")
print(f"FP pairs={len(fp):,} whose target belongs to another S1={int(fp['owner_sid'].is_not_null().sum()):,}")
sample = sc.filter(pl.col("y") == 0).sample(n=200_000, seed=0).join(links, on="t", how="left")
print(f"negative candidates owned by some S1: {sample['owner_sid'].is_not_null().mean():.4f}")

# Candidate oracle when stage 1 keeps its top K (optionally above a probability floor).
sc = sc.with_columns(pl.col("p").rank("ordinal", descending=True).over("s1").alias("prank"))
gq = per.select("s1", "g")
for kmax in (3, 5, 8, 10, 12, 15, 20, 30):
    for floor in ((0.0,) if kmax not in (8, 10, 12, 15) else (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01)):
        kept = sc.filter((pl.col("prank") <= kmax) & (pl.col("p") >= floor))
        rr = kept.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "r"})
        x = gq.join(rr, on="s1", how="left").with_columns(pl.col("r").fill_null(0))
        uk = per_query_f(x["r"].to_numpy(), x["r"].to_numpy(), x["g"].to_numpy())
        pk = kept.filter(pl.col("owner") & (pl.col("p") >= THRESHOLD))
        yk = gq.join(pk.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "tp"}), on="s1", how="left").join(
            pk.group_by("s1").len().rename({"len": "pp"}), on="s1", how="left").fill_null(0)
        fk = per_query_f(yk["tp"].to_numpy(), yk["pp"].to_numpy(), yk["g"].to_numpy())
        print(f"top{kmax:<3d} floor={floor:<7} U={uk.mean():.5f} F@{THRESHOLD}={fk.mean():.5f} "
              f"mean candidates={len(kept) / n:.2f}")
