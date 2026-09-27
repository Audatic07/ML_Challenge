"""v6: a survivor-specialist LightGBM behind the v5.1 stage-1 model.

    python -m src.v6_train plan  --bucket B --source SRC --prefix DST [--train-n 15 --n 12 --floor 0.001]
    python -m src.v6_train apply --bucket B --source DST --prefix DST2 --n N --floor F [--threshold T]
    python -m src.v6_train variant --bucket B --source DST --prefix DST3 --seed S [--leaves L ...]

Stage 1 is the released v5.1 pair model of the run at SRC. It scores every retrieved
candidate (about 256 per S1) and keeps the top `n` of each S1 by probability, ties broken
by target row, then drops candidates below `floor`. v6 is trained on such survivors only
and is the final model: it scores exactly the survivors, and they form candidate_pairs.tsv.
It reads the v5.1 feature shards of SRC unchanged, so retrieval and features are not rebuilt.

Queue tasks (this prefix P, source run S = plan["source_prefix"]):
  v6cut-<shard>    one train feature shard of S scored by the v5.1 model; the top
                   plan["train_n"] of each S1 go to P/surv/<shard>.parquet with v5.1 p as p1.
  v6train          LightGBM on fit survivors with early stopping on stop survivors, then tune
                   metrics for every cut in plan["eval_cuts"], each next to v5.1 at the same cut.
  v6score-<shard>  one test shard: v5.1 p from S/score/, the apply cut, v6 p -> P/score/.
  final            the v5 final assembly (one owner per target, threshold, both TSVs, validator).

The v5.1 model saw the fit rows in training, so its p only chooses survivors and is never a
v6 feature. Stop and tune rows are out of sample for both models. The audit stays closed.
`apply` republishes test scoring for another cut with the same trained model. `variant` trains
another model (seed or LightGBM settings) on the survivors of a finished queue, for ensembles.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import time

import numpy as np
import polars as pl

from .metric import per_entity, unique_assign
from .v5_dist import Store, digest, feature_list, log, mem_gb, truth_for

VERSION = "v6-survivor-1"
EVAL_CUTS = [{"n": 15, "floor": 0.0}, {"n": 15, "floor": 0.001}, {"n": 12, "floor": 0.001},
             {"n": 10, "floor": 0.001}, {"n": 8, "floor": 0.001}, {"n": 12, "floor": 0.005},
             {"n": 12, "floor": 0.01}]


def cut(frame, n, floor=0.0, p="p1"):
    """Top-n rows of each S1 by `p` (ties by target row t), then rows below `floor` dropped.
    Rows come out ordered by S1, then rank. Same rule as v6_stage2.survivors."""
    order = (frame.select("s1", p, "t").with_row_index("_i")
             .sort(["s1", p, "t"], descending=[False, True, False])
             .group_by("s1", maintain_order=True).head(n))
    if floor > 0:
        order = order.filter(pl.col(p) >= floor)
    return frame[order["_i"]]


def stage1(ctx, source):
    """The v5.1 booster and manifest of the source run, checksum-verified and cached."""
    import lightgbm as lgb
    cached = getattr(ctx, "_v6_stage1", None)
    if cached and cached[0] == source.prefix:
        return cached[1], cached[2]
    manifest = source.get_json("model/model_manifest.json")
    path = source.download("model/model.txt", ctx.work / "stage1_model.txt")
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
        raise ValueError("Stage-1 model checksum mismatch")
    booster = lgb.Booster(model_file=str(path))
    ctx._v6_stage1 = (source.prefix, booster, manifest)
    return booster, manifest


def source_store(store, plan):
    return Store(store.bucket, plan["source_prefix"], store.base_root)


def read_shard(store, name, ctx, columns=None):
    path = store.download(name, ctx.work / "in" / name.replace("/", "_"))
    frame = pl.read_parquet(path, columns=columns)
    path.unlink()
    return frame


def run_cut(task, store, ctx):
    source = source_store(store, ctx.plan)
    booster, manifest = stage1(ctx, source)
    if manifest["model_sha256"] != ctx.plan["stage1_model_sha256"]:
        raise ValueError("Source model differs from the one this plan was published for")
    frame = read_shard(source, f"feat/{task['input']}.parquet", ctx)
    p1 = booster.predict(frame.select(manifest["features"]).to_numpy(), num_threads=ctx.workers)
    frame = frame.with_columns(pl.Series("p1", p1.astype(np.float32)))
    kept = cut(frame, ctx.plan["train_n"])
    local = ctx.work / f"{task['id']}.parquet"
    kept.write_parquet(local, compression="zstd", compression_level=3)
    store.upload(local, f"surv/{task['input']}.parquet")
    local.unlink()
    return {"pairs": len(frame), "kept": len(kept), "queries": frame["s1"].n_unique(),
            "kept_queries": kept["s1"].n_unique()}


def best_threshold(scored, column, tune, gt):
    """One-owner assignment, then the threshold with the best tune macro F0.5 (coarse grid,
    then a 0.0025 grid around the best). Returns (threshold, macro F0.5, per-query frame)."""
    assigned = unique_assign(scored.select("s1", "t", pl.col("target_id"), pl.col(column).alias("p")))

    def score(th):
        pred = assigned.filter(pl.col("p") >= th).select("s1", pl.col("target_id").alias("t"))
        return float(per_entity(tune, gt, pred)["f"].mean())

    grid = [(float(th), score(th)) for th in np.unique(np.r_[np.arange(0.2, 0.961, 0.02), [0.97, 0.98, 0.99]])]
    th, best = max(grid, key=lambda item: item[1])
    fine = [(float(x), score(x)) for x in np.arange(max(0.05, th - 0.02), min(0.995, th + 0.021), 0.0025)]
    th, best = max(fine + [(th, best)], key=lambda item: item[1])
    pred = assigned.filter(pl.col("p") >= th).select("s1", pl.col("target_id").alias("t"))
    return th, best, per_entity(tune, gt, pred)


def evaluate_cut(scores, cut_spec, tune, gt, country):
    kept = cut(scores, cut_spec["n"], cut_spec["floor"])
    hit = kept.select("s1", pl.col("target_id").alias("t")).join(gt, on=["s1", "t"])
    oracle = per_entity(tune, gt, hit).select("s1", pl.col("f").alias("oracle"))
    out = {"n": cut_spec["n"], "floor": cut_spec["floor"], "pairs_per_query": len(kept) / len(tune),
           "pairs_p95": float(kept.group_by("s1").len()["len"].quantile(0.95) or 0),
           "oracle_u": float(oracle["oracle"].mean()), "link_recall": len(hit) / max(1, len(gt))}
    per = {}
    for model, column in (("v6", "p"), ("v51", "p1")):
        th, best, frame = best_threshold(kept, column, tune, gt)
        per[model] = frame
        out[model] = {"threshold": th, "macro_f05": best}
    both = (per["v6"].select("s1", "g", pl.col("f").alias("f_v6"))
            .join(per["v51"].select("s1", pl.col("f").alias("f_v51")), on="s1")
            .join(oracle, on="s1").join(country, on="s1", how="left"))
    delta = (both["f_v6"] - both["f_v51"]).to_numpy()
    out["paired_delta"] = {"mean": float(delta.mean()), "se": float(delta.std(ddof=1) / np.sqrt(len(delta))),
                           "queries_better": int((delta > 0).sum()), "queries_worse": int((delta < 0).sum())}
    out["by_country"] = both.group_by("country").agg(
        pl.len().alias("queries"), pl.col("f_v6").mean(), pl.col("f_v51").mean(),
        pl.col("oracle").mean().alias("oracle_u")).sort("country").to_dicts()
    out["singleton"] = both.filter(pl.col("g") == 0).select(pl.len().alias("queries"), pl.col("f_v6").mean(),
                                                           pl.col("f_v51").mean()).to_dicts()[0]
    return out, both


def run_train6(task, store, ctx):
    import lightgbm as lgb
    from .config import LGB_PARAMS
    from .v4_train import partitions
    started = time.monotonic()
    plan = ctx.plan
    names = feature_list(plan)
    split, fit, stop, tune, heldout = partitions(plan["train_queries"], plan["fit"], plan["stop"], plan["tune"])
    if digest(fit.tolist()) != plan["fit_sha"] or digest(tune.tolist()) != plan["tune_sha"]:
        raise ValueError("Train partitions differ from the published plan")
    truth, excluded = truth_for(ctx.data, {"fit": fit, "stop": stop, "tune": tune}, heldout)
    fit_s, stop_s, tune_s = (pl.Series(x.astype(np.uint32)) for x in (fit, stop, tune))
    positive = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
    surv = Store(store.bucket, plan.get("surv_prefix") or store.prefix, store.base_root)
    xf, yf, xs, ys, tune_parts = [], [], [], [], []
    for shard in task["inputs"]:
        frame = read_shard(surv, f"surv/{shard}.parquet", ctx)
        frame = frame.join(positive, on=["s1", "target_id"], how="left").with_columns(pl.col("y").fill_null(0))
        part = frame.filter(pl.col("s1").is_in(fit_s.implode()))
        if part.filter(pl.col("y") == 1).join(excluded, on="target_id", how="semi").height:
            raise ValueError("A labelled fit target is owned by a held-out query")
        part = part.join(excluded, on="target_id", how="anti")
        if len(part):
            xf.append(part.select(names).to_numpy().astype(np.float32, copy=False))
            yf.append(part["y"].to_numpy())
        part = frame.filter(pl.col("s1").is_in(stop_s.implode()))
        if len(part):
            xs.append(part.select(names).to_numpy().astype(np.float32, copy=False))
            ys.append(part["y"].to_numpy())
        part = frame.filter(pl.col("s1").is_in(tune_s.implode()))
        if len(part):
            tune_parts.append(part.select("s1", "t", "target_id", "y", "p1", *names))
        del frame, part
        gc.collect()
        log(f"loaded {shard}: fit rows={sum(len(y) for y in yf):,} mem avail={mem_gb()[1]:.1f}GB")
    y_fit = np.concatenate(yf)
    params = dict(LGB_PARAMS, **plan["lgb"], num_threads=ctx.workers, deterministic=True, force_col_wise=True)
    dtrain = lgb.Dataset(np.concatenate(xf), label=y_fit, feature_name=names, params=params, free_raw_data=True)
    dstop = lgb.Dataset(np.concatenate(xs), label=np.concatenate(ys), reference=dtrain, params=params)
    dtrain.construct()
    del xf, xs
    gc.collect()
    log(f"LightGBM fit rows={len(y_fit):,} positives={int(y_fit.sum()):,}; training")
    booster = lgb.train(params, dtrain, num_boost_round=plan["rounds"], valid_sets=[dstop], valid_names=["stop"],
                        callbacks=[lgb.early_stopping(plan["patience"]), lgb.log_evaluation(50)])
    model_path = ctx.work / "model.txt"
    booster.save_model(str(model_path))
    trained = time.monotonic() - started
    tune_frame = pl.concat(tune_parts)
    p = booster.predict(tune_frame.select(names).to_numpy(), num_threads=ctx.workers)
    scores = tune_frame.select("s1", "t", "target_id", "p1").with_columns(pl.Series("p", p.astype(np.float32)))
    del tune_frame
    gt = truth.filter(pl.col("s1").is_in(tune_s.implode())).select("s1", pl.col("target_id").alias("t"))
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    cuts, apply_per = {}, None
    for spec in plan["eval_cuts"]:
        key = f"n{spec['n']}_f{spec['floor']}"
        cuts[key], both = evaluate_cut(scores, spec, tune, gt, country)
        if spec == plan["apply_cut"]:
            apply_per = both
        log(f"tune cut {key}: " + str({k: v for k, v in cuts[key].items() if k not in ("by_country",)}))
    apply_key = f"n{plan['apply_cut']['n']}_f{plan['apply_cut']['floor']}"
    applied = cuts[apply_key]
    metrics = {"tune_queries": len(tune), "apply_cut": plan["apply_cut"], "macro_f05": applied["v6"]["macro_f05"],
               "v51_same_cut_f05": applied["v51"]["macro_f05"], "oracle_u": applied["oracle_u"],
               "pairs_per_query": applied["pairs_per_query"], "threshold": applied["v6"]["threshold"],
               "cuts": cuts, "best_iteration": booster.best_iteration, "fit_rows": len(y_fit),
               "fit_positives": int(y_fit.sum()), "train_seconds": trained, "audit_opened": False}
    apply_per.write_parquet(ctx.work / "tune_per_query.parquet")
    scores.write_parquet(ctx.work / "tune_scores.parquet")
    manifest = {"version": VERSION, "features": names, "threshold": applied["v6"]["threshold"],
                "cascade": {"stage1": "model", **plan["apply_cut"]}, "source_prefix": plan["source_prefix"],
                "stage1_model_sha256": plan["stage1_model_sha256"], "params": params,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "metrics": metrics,
                "plan_sha": plan["plan_sha"], "tree_count": booster.num_trees(),
                "model_license": "MIT (LightGBM); no pretrained weights"}
    dumped = booster.dump_model()["tree_info"]
    manifest["parameter_count"] = 2 * sum(t["num_leaves"] for t in dumped) - len(dumped)
    for name in ("tune_per_query.parquet", "tune_scores.parquet"):
        store.upload(ctx.work / name, f"model/{name}")
    store.upload(model_path, "model/model.txt")
    store.put_json("model/model_manifest.json", manifest)
    summary = {k: v for k, v in metrics.items() if k != "cuts"}
    log("TRAINING_COMPLETE " + str(summary))
    return summary


def run_score6(task, store, ctx):
    source = source_store(store, ctx.plan)
    manifest = ctx.model_manifest(store)
    booster = ctx.booster(store)
    names = manifest["features"]
    feat = read_shard(source, f"feat/{task['input']}.parquet", ctx, columns=["s1", "t", "target_id", *names])
    first = read_shard(source, f"score/{task['input'].replace('feat-', 'score-', 1)}.parquet", ctx)
    if len(feat) != len(first) or not (feat["s1"].equals(first["s1"]) and feat["t"].equals(first["t"])):
        raise ValueError("Stage-1 score shard is not aligned with its feature shard")
    kept = cut(feat.with_columns(first["p"].alias("p1")), manifest["cascade"]["n"], manifest["cascade"]["floor"])
    p = booster.predict(kept.select(names).to_numpy(), num_threads=ctx.workers) if len(kept) else np.zeros(0)
    out = kept.select("s1", "t", "target_id", "p1").with_columns(pl.Series("p", p.astype(np.float32)))
    local = ctx.work / f"{task['id']}.parquet"
    out.write_parquet(local)
    store.upload(local, f"score/{task['id']}.parquet")
    local.unlink()
    return {"pairs": len(feat), "kept": len(out), "queries": feat["s1"].n_unique(),
            "kept_queries": out["s1"].n_unique(), "above_threshold": int((out["p"] >= manifest["threshold"]).sum())}


def load_models(store, prefixes, ctx):
    """v6 boosters (checksum-verified) of the listed queues; all must share one feature list."""
    import lightgbm as lgb
    boosters, names = [], None
    for prefix in prefixes:
        queue = Store(store.bucket, prefix, store.base_root)
        manifest = queue.get_json("model/model_manifest.json")
        path = queue.download("model/model.txt", ctx.work / f"model-{len(boosters)}.txt")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["model_sha256"]:
            raise ValueError(f"Model checksum mismatch for {prefix}")
        if names not in (None, manifest["features"]):
            raise ValueError("Ensemble members use different features")
        names = manifest["features"]
        boosters.append((prefix, manifest["model_sha256"], lgb.Booster(model_file=str(path))))
    return boosters, names


def run_audit(task, store, ctx):
    """One-time evaluation of a champion chosen on tune, on the locked audit S1 rows.

    Every threshold is fixed in the task (chosen on tune); nothing is selected here. Reports
    the champion (mean p of task["models"] on the task cut) next to v5.1 at the same cut, the
    v5.1 full retrieved union and the released cosine top-30 cascade."""
    from .v4_train import partitions
    from .v5_dist import stage_rank
    started = time.monotonic()
    plan = ctx.plan
    source = source_store(store, plan)
    booster1, manifest1 = stage1(ctx, source)
    audit_store = Store(store.bucket, task["audit_prefix"], store.base_root)
    audit_plan = audit_store.get_json("plan.json")
    split = partitions(plan["train_queries"], plan["fit"], plan["stop"], plan["tune"])[0]
    audit = np.sort(split.audit).astype(np.uint32)
    if digest(audit.tolist()) != audit_plan["audit_sha"]:
        raise ValueError("Audit rows differ from the audit feature plan")
    boosters, names = load_models(store, task["models"], ctx)
    if names != manifest1["features"]:
        raise ValueError("v6 and v5.1 feature lists differ")
    truth, _ = truth_for(ctx.data, {"audit": audit}, np.array([], dtype=np.uint32))
    gt = truth.select("s1", pl.col("target_id").alias("t"))
    th = task["thresholds"]
    kept_parts, full_above, rel_above, full_hit, rel_hit, counts = [], [], [], [], [], {"full": 0, "release": 0}
    for name in sorted(n[:-5] for n in audit_store.list("tasks")):
        frame = read_shard(audit_store, f"feat/{name}.parquet", ctx)
        p1 = booster1.predict(frame.select(names).to_numpy(), num_threads=ctx.workers)
        frame = frame.with_columns(pl.Series("p1", p1.astype(np.float32)))
        base = frame.select("s1", "t", "target_id", pl.col("p1").alias("p"))
        counts["full"] += len(base)
        full_above.append(base.filter(pl.col("p") >= th["v51_full"]))
        full_hit.append(base.select("s1", pl.col("target_id").alias("t")).join(gt, on=["s1", "t"]))
        rel = frame.filter(stage_rank({"score": ["combo", "combo_word"]}) <= 30).select("s1", "t", "target_id",
                                                                                          pl.col("p1").alias("p"))
        counts["release"] += len(rel)
        rel_above.append(rel.filter(pl.col("p") >= th["release_cosine30"]))
        rel_hit.append(rel.select("s1", pl.col("target_id").alias("t")).join(gt, on=["s1", "t"]))
        kept = cut(frame, task["cut"]["n"], task["cut"]["floor"])
        x = kept.select(names).to_numpy()
        p = np.mean([b.predict(x, num_threads=ctx.workers) for _, _, b in boosters], axis=0) if len(kept) else np.zeros(0)
        kept_parts.append(kept.select("s1", "t", "target_id", "p1").with_columns(pl.Series("p", p.astype(np.float32))))
        del frame, base, rel, kept, x
        gc.collect()
        log(f"audit {name}: pairs so far full={counts['full']:,}")
    country = pl.read_csv(ctx.data / "train" / "train_source1.tsv", separator="	", quote_char=None,
                          columns=["country"], infer_schema=False).with_row_index("s1").with_columns(
        pl.col("s1").cast(pl.UInt32)).filter(pl.col("s1").is_in(pl.Series(audit).implode()))

    def report(pred_scored, threshold, hit, pairs):
        chosen = unique_assign(pred_scored).filter(pl.col("p") >= threshold).select("s1", pl.col("target_id").alias("t"))
        per = per_entity(audit, gt, chosen).join(per_entity(audit, gt, hit).select("s1", pl.col("f").alias("oracle")),
                                                  on="s1").join(country, on="s1", how="left")
        out = {"macro_f05": float(per["f"].mean()), "oracle_u": float(per["oracle"].mean()), "threshold": threshold,
               "pairs_per_query": pairs / len(audit), "singleton_f": float(per.filter(pl.col("g") == 0)["f"].mean() or 0),
               "by_country": per.group_by("country").agg(pl.len().alias("queries"), pl.col("f").mean().alias("macro_f05"),
                                                        pl.col("oracle").mean().alias("oracle_u")).sort("country").to_dicts()}
        return out, per.select("s1", "f")

    kept = pl.concat(kept_parts)
    kept_hit = kept.select("s1", pl.col("target_id").alias("t")).join(gt, on=["s1", "t"])
    results, per = {}, {}
    results["champion"], per["champion"] = report(kept.select("s1", "t", "target_id", "p"), th["champion"], kept_hit, len(kept))
    results["v51_same_cut"], per["v51_same_cut"] = report(kept.select("s1", "t", "target_id", pl.col("p1").alias("p")),
                                                          th["v51_same_cut"], kept_hit, len(kept))
    results["v51_full_union"], _ = report(pl.concat(full_above), th["v51_full"], pl.concat(full_hit), counts["full"])
    results["release_cosine30"], per["release"] = report(pl.concat(rel_above), th["release_cosine30"],
                                                         pl.concat(rel_hit), counts["release"])
    for other in ("v51_same_cut", "release"):
        d = per["champion"].join(per[other], on="s1", suffix="_o").select(pl.col("f") - pl.col("f_o"))["f"].to_numpy()
        results[f"delta_vs_{other}"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}
    results.update(audit_queries=len(audit), cut=task["cut"], models=[(p, sha) for p, sha, _ in boosters],
                   seconds=time.monotonic() - started, note="thresholds fixed from tune; evaluated once")
    store.put_json(f"audit/{task['id']}.json", results)
    log("AUDIT " + str({k: v for k, v in results.items() if k not in ("models",)}))
    return {k: (v["macro_f05"] if isinstance(v, dict) and "macro_f05" in v else v) for k, v in results.items()
            if k not in ("models", "cut")}


RUNNERS = {"v6cut": run_cut, "v6train": run_train6, "v6score": run_score6, "v6audit": run_audit}


def score_tasks(feats, source_prefix):
    tasks = [{"id": name.replace("feat-", "v6score-", 1), "kind": "v6score", "input": name, "requires": ["v6train"],
              "source_prefix": source_prefix, "priority": 1, "min_mem_gb": 8}
             for name, t in sorted(feats.items()) if t["split"] == "test"]
    final = {"id": "final", "kind": "final", "inputs": [t["id"] for t in tasks], "requires": [t["id"] for t in tasks],
             "priority": 2, "min_mem_gb": 60}
    return tasks + [final]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["plan", "apply", "variant"])
    parser.add_argument("--bucket")
    parser.add_argument("--root")
    parser.add_argument("--source", required=True, help="plan: finished v5.1 run; apply: finished v6 queue")
    parser.add_argument("--prefix", required=True, help="new queue prefix")
    parser.add_argument("--train-n", type=int, default=15)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--floor", type=float, default=0.001)
    parser.add_argument("--threshold", type=float, help="apply: override the tune-selected threshold")
    parser.add_argument("--rounds", type=int, default=5000)
    parser.add_argument("--patience", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--leaves", type=int, default=255)
    parser.add_argument("--min-leaf", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source, target = Store(args.bucket, args.source, args.root), Store(args.bucket, args.prefix, args.root)
    if target.exists("plan.json"):
        parser.error(f"{args.prefix} already has a plan; queues are immutable")
    base = source.get_json("plan.json")
    apply_cut = {"n": args.n, "floor": args.floor}
    if args.step == "apply":
        # Same model, another test cut. The threshold comes from the v6 tune evaluation of that cut.
        manifest = source.get_json("model/model_manifest.json")
        key = f"n{args.n}_f{args.floor}"
        threshold = args.threshold
        if threshold is None:
            if key not in manifest["metrics"]["cuts"]:
                parser.error(f"cut {key} was not evaluated on tune; pass --threshold")
            threshold = manifest["metrics"]["cuts"][key]["v6"]["threshold"]
        manifest.update(threshold=threshold, cascade={"stage1": "model", **apply_cut}, applied_from=args.source)
        target.put_json("plan.json", dict(base, apply_cut=apply_cut, plan_sha=digest({**base, "apply_cut": apply_cut})))
        target.put_bytes("model/model.txt", source.get_bytes("model/model.txt"))
        target.put_json("model/model_manifest.json", manifest)
        origin = Store(args.bucket, base["source_prefix"], args.root)
        feats = {n[:-5]: origin.get_json(f"tasks/{n}") for n in origin.list("tasks") if n.startswith("feat-")}
        for task in score_tasks(feats, base["source_prefix"]):
            task["requires"] = [r for r in task["requires"] if r != "v6train"]
            target.put_json(f"tasks/{task['id']}.json", task)
        return
    lgb_settings = {"learning_rate": args.learning_rate, "num_leaves": args.leaves, "min_data_in_leaf": args.min_leaf,
                    "feature_fraction": args.feature_fraction, "seed": args.seed}
    if args.step == "variant":
        # Another model on the survivors of a queue whose cut tasks are finished.
        cut_ids = [n[:-5] for n in source.list("tasks") if n.startswith("v6cut-")]
        missing = [c for c in cut_ids if not source.exists(f"done/{c}.json")]
        if missing:
            parser.error(f"{len(missing)} cut tasks of {args.source} are not finished")
        settings = {k: v for k, v in base.items() if k not in ("plan_sha", "task_count", "created")}
        settings.update(surv_prefix=args.source, variant_of=args.source, lgb=lgb_settings, rounds=args.rounds,
                        patience=args.patience)
        settings["plan_sha"] = digest(settings)
        settings["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        train = source.get_json("tasks/v6train.json")
        tasks = [dict(train, requires=[])] + score_tasks(
            {n[:-5]: {"split": "test"} for n in Store(args.bucket, base["source_prefix"], args.root).list("tasks")
             if n.startswith("feat-test-")}, base["source_prefix"])
        settings["task_count"] = len(tasks)
        target.put_json("plan.json", settings)
        for task in tasks:
            target.put_json(f"tasks/{task['id']}.json", task)
        log(f"published variant {args.prefix}: {len(tasks)} tasks, lgb {lgb_settings}")
        return
    manifest = source.get_json("model/model_manifest.json")
    settings = {k: v for k, v in base.items() if k not in ("plan_sha", "task_count", "created")}
    if feature_list(settings) != manifest["features"]:
        raise SystemExit("Source model features differ from the source plan's feature list")
    cuts = EVAL_CUTS + ([apply_cut] if apply_cut not in EVAL_CUTS else [])
    settings.update(version=VERSION, source_prefix=args.source, source_plan_sha=base["plan_sha"],
                    stage1_model_sha256=manifest["model_sha256"], train_n=args.train_n, apply_cut=apply_cut,
                    eval_cuts=cuts, rounds=args.rounds, patience=args.patience,
                    lgb=lgb_settings)
    settings["plan_sha"] = digest(settings)
    settings["created"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    feats = {n[:-5]: source.get_json(f"tasks/{n}") for n in source.list("tasks") if n.startswith("feat-")}
    cut_tasks = [{"id": name.replace("feat-", "v6cut-", 1), "kind": "v6cut", "input": name, "priority": 0,
                  "min_mem_gb": 14} for name, t in sorted(feats.items()) if t["split"] == "train"]
    train = {"id": "v6train", "kind": "v6train", "inputs": [t["input"] for t in cut_tasks],
             "requires": [t["id"] for t in cut_tasks], "priority": -1, "min_mem_gb": 60}
    tasks = cut_tasks + [train] + score_tasks(feats, args.source)
    settings["task_count"] = len(tasks)
    target.put_json("plan.json", settings)
    for task in tasks:
        target.put_json(f"tasks/{task['id']}.json", task)
    log(f"published {len(tasks)} tasks: {len(cut_tasks)} cut, train, {len(tasks) - len(cut_tasks) - 2} score, final")


if __name__ == "__main__":
    main()
