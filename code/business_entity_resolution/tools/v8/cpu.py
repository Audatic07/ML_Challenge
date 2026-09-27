"""V8 step 3 (CPU notebook): B5 = B4 features + SC-Block cosine features, the same 5 folds, compared with B4's own
out-of-fold predictions (each B4 fold model scores only its held-out fold). Then test scoring and final assembly.
    python tools/v8/cpu.py gate      (needs ~/SageMaker/work/v8/cos_tune.parquet)
    python tools/v8/cpu.py test      (needs cos_test.parquet and a B5 that passed)
"""
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, "tools/v6_analysis")
import stage2_release as sr  # noqa: E402

W = Path.home() / "SageMaker/work"
D = W / "data/student_resource/dataset"
V = W / "v8"
TH = int(os.environ.get("ER_V6_THREADS", 16))
ECOLS = ["e_cos", "e_rank", "e_gap", "e_best_other"]
MIN_GAIN = 0.0010  # pre-registered: B5 must beat B4's OOF by >= 0.0010 on the same folds


def efeat(cos: pl.DataFrame) -> pl.DataFrame:
    top = pl.col("cos").sort(descending=True)
    return cos.with_columns(
        pl.col("cos").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("e_rank"),
        (pl.col("cos").max().over("s1") - pl.col("cos")).alias("e_gap"),
        top.get(0).over("s1").alias("_m1"), top.get(1, null_on_oob=True).over("s1").alias("_m2"),
    ).with_columns(pl.when(pl.col("e_rank") == 1).then(pl.col("_m2")).otherwise(pl.col("_m1")).fill_null(float("nan"))
                   .alias("e_best_other")).drop("_m1", "_m2").rename({"cos": "e_cos"})


def gate():
    import lightgbm as lgb
    from src.v7_decoy import build_context
    from src.v7_population import build_index
    started = time.time()
    b4 = json.loads((W / "gate_b4/release.json").read_text())
    plan, manifest, kept, _ = sr.load_ens3(W / "ens3", W / "members")
    tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
    gt = sr.tune_truth(D, tune).select("s1", pl.col("target_id").alias("t"))
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    q_raw, t_raw = sr.read_tsv(D / "train/train_source1.tsv"), sr.target_frame(D, "train")
    feat = sr.build_features(kept, manifest["cascade"]["n"], q_raw, t_raw, TH, "F0F1F3F2", build_index(q_raw), build_context(t_raw))
    feat = feat.join(efeat(pl.read_parquet(V / "cos_tune.parquet")), on=["s1", "t"], how="left", maintain_order="left")
    assert feat["e_cos"].null_count() == 0, "missing tune cosines"
    labels = gt.rename({"t": "target_id"}).with_columns(pl.lit(1, pl.Int8).alias("y"))
    feat = feat.join(labels, on=["s1", "target_id"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    fold, inner = sr.folds_for(feat, q_raw)
    print(f"features {len(feat):,} x {len(b4['features']) + len(ECOLS)} t={time.time() - started:.0f}s", flush=True)
    xb = feat.select(b4["features"]).to_numpy().astype(np.float32)
    oof_b4 = np.zeros(len(feat))
    for f, m in enumerate(b4["models"]):  # B4 fold model f only scores fold f: honest OOF
        idx = np.where(fold == f)[0]
        oof_b4[idx] = lgb.Booster(model_file=str(W / "gate_b4" / m["file"])).predict(xb[idx])
    names = b4["features"] + ECOLS
    oof, models, _ = sr.cross_fit(feat, names, fold, inner, V / "model", "B5", TH)
    res = {}
    for tag, p7 in (("B4", oof_b4), ("B5", oof)):
        best = None
        for alpha in (0.5, 1.0):
            scored = feat.select("s1", "t", "target_id", "p6").with_columns(
                pl.Series("s", sr.blend(feat["p6"], p7.astype(np.float32), alpha)))
            th, f, per = sr.sweep(scored, "s", tune, gt)
            print(f"{tag} alpha={alpha}: threshold {th:.4f} F {f:.5f}", flush=True)
            if best is None or (f, alpha) > (best[0], best[1]):
                best = (f, alpha, th, per)
        res[tag] = best
    d = sr.paired(res["B4"][3], res["B5"][3], country)
    gain = res["B5"][0] - res["B4"][0]
    ok = bool(gain >= MIN_GAIN and d["mean"] - 1.645 * d["se"] > 0)
    cfg = {"features": names, "alpha": res["B5"][1], "threshold": res["B5"][2], "b5_oof": res["B5"][0], "b4_oof": res["B4"][0],
           "gain": gain, "paired": d, "pass": ok, "models": models, "encoder": "sentence-transformers/all-MiniLM-L6-v2 SupCon fit-only",
           "seconds": time.time() - started}
    (V / "b5.json").write_text(json.dumps(cfg, indent=1, default=str))
    print(f"B4 OOF {res['B4'][0]:.5f} (release says {b4['gate']['g0']} / 0.98002) | B5 OOF {res['B5'][0]:.5f} "
          f"gain {gain:+.5f} (paired se {d['se']:.5f}) -> {'PASS' if ok else 'FAIL'}", flush=True)


def test():
    import lightgbm as lgb
    cfg = json.loads((V / "b5.json").read_text())
    if not cfg["pass"]:
        raise SystemExit("B5 did not pass its tune gate; keep B4")
    boosters = [lgb.Booster(model_file=str(V / m["file"])) for m in cfg["models"]]
    cos = efeat(pl.read_parquet(V / "cos_test.parquet"))
    parts = []
    for path in sorted(glob.glob(str(W / "cache_all/*.parquet"))):
        feat = pl.read_parquet(path).join(cos, on=["s1", "t"], how="left", maintain_order="left")
        x = feat.select(cfg["features"]).to_numpy().astype(np.float32)
        p7 = np.mean([b.predict(x, num_threads=TH) for b in boosters], axis=0).astype(np.float32)
        parts.append(feat.select("s1", "t", "target_id").with_columns(pl.Series("s", sr.blend(feat["p6"], p7, cfg["alpha"]))))
    s = pl.concat(parts)
    assert len(s) == 9_266_800 and s["s"].is_finite().all(), (len(s), "pairs")
    q = sr.read_tsv(D / "test/test_source1.tsv", columns=["entity_id"]).with_row_index("s1")
    owner = s.sort(["t", "s", "s1"], descending=[False, True, False]).group_by("t", maintain_order=True).head(1)
    rows = owner.filter(pl.col("s") >= cfg["threshold"]).group_by("s1").agg(pl.col("target_id").sort().str.join(",").alias("matched_entity_ids"))
    out = q.join(rows, on="s1", how="left").select(pl.col("entity_id").alias("source1_entity_id"), pl.col("matched_entity_ids").fill_null(""))
    path = V / "matching_results.tsv"
    out.write_csv(path, separator="\t", quote_style="never")
    print(f"wrote {path} rows {len(out):,} matched pairs {len(owner.filter(pl.col('s') >= cfg['threshold'])):,} "
          f"sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}\nTEST_DONE", flush=True)


if __name__ == "__main__":
    {"gate": gate, "test": test}[sys.argv[1]]()
