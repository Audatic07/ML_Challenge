"""Stage 2 on the frozen v6 ens3 survivors: tune gate, frozen release, shard-aware test inference, collector.

    python tools/v6_analysis/stage2_release.py gate    --ens3 E --members M --data D --out R [--expect-f 0.97061]
    python tools/v6_analysis/stage2_release.py assign  --release R/release.json --scores S --out A.json
                                                       --benchmark WORKER=SHARD,... [--rates WORKER=META.json,...]
    python tools/v6_analysis/stage2_release.py predict --release R/release.json --assignment A.json --worker W
                                                       --scores S --data D --out O [--shards SHARD,...]
    python tools/v6_analysis/stage2_release.py collect --release R/release.json --assignment A.json
                                                       --inputs O1,O2,O3 --data D --out F

Local copies of Aditya's queues, read-only (synced from S3):
  E  shared/er-v6ens3-20260927/{plan.json, model/model_manifest.json}
  M  one folder per ens3 member, named like the member prefix's last part, with model/{model_manifest.json,
     tune_scores.parquet}
  S  shared/er-v6ens3-20260927/score/*.parquet: the ens3 test survivors (mean p, v5.1 p1), one file per shard
  D  the supplied dataset directory (train/, test/; ../utils/validate_submission.py)

Stage 1 is the ens3 mean p, and the candidate set is exactly the ens3 survivors (cut read from the ens3
manifest). Stage 2 is v6_stage2.features on those survivors and the stage2_cv.py LightGBM recipe,
cross-fitted in 2 folds on the tune S1. The release model is the mean of the two fold models. Both sides of
the gate get the one-owner threshold sweep that chose ens3's threshold, on the same tune S1 and survivor
rows. The audit partition is never read. Test inference reads only the shards it is given.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from _common import PACKAGE, read_tsv
from src.metric import per_entity, unique_assign
from src.v5_dist import Context, Store, run_final
from src.v6_stage2 import features, survivors, text_views
from src.v6_train import best_threshold, cut

VERSION = "s2-ens3-1"
WORKERS = ("abhigyan-r7i4xl", "aditya-worker", "akash-r5xl")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1)  # stage2_cv.py
OUT_COLUMNS = ["s1", "t", "target_id", "p1", "p_ens", "p"]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # KiB on Linux
    except ImportError:
        return None


def code_identity() -> dict:
    here = Path(__file__).resolve()
    ident = {"stage2_release.py": sha256(here), "v6_stage2.py": sha256(PACKAGE / "src" / "v6_stage2.py")}
    try:
        ident["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True,
                                           text=True, check=True).stdout.strip()
        ident["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PACKAGE,
                                                 capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        ident["git_head"] = None
    return ident


def target_frame(data: Path, split: str) -> pl.DataFrame:
    """Raw S2 rows then S3 rows: position = the global target row `t` of the score files."""
    return pl.concat([read_tsv(data / split / f"{split}_source{s}.tsv").with_columns(pl.lit(s, pl.UInt8).alias("src"))
                      for s in (2, 3)])


def stage2_features(scored: pl.DataFrame, n: int, q_raw: pl.DataFrame, t_raw: pl.DataFrame, threads: int):
    """Stage-2 features of already-cut survivors (s1, t, target_id, p = stage-1 p), ranked by p."""
    surv = survivors(scored.select("s1", "t", "target_id", "p"), n)
    qrows, trows = surv["s1"].unique().sort(), surv["t"].unique().sort()
    q_text = text_views(q_raw.gather(qrows))
    t_sel = t_raw.gather(trows)
    t_text = text_views(t_sel).with_columns(t_sel["src"])
    surv = (surv.join(pl.DataFrame({"s1": qrows, "qi": np.arange(len(qrows), dtype=np.uint32)}), on="s1")
            .join(pl.DataFrame({"t": trows, "ti": np.arange(len(trows), dtype=np.uint32)}), on="t"))
    return features(surv, q_text, t_text, workers=threads)


# ------------------------------------------------------------------ tune gate
def tune_truth(data: Path, rows: np.ndarray) -> pl.DataFrame:
    """(s1 row, target_id) labels of the given train S1 rows: v4_train.labels without held-out rows."""
    ids = read_tsv(data / "train" / "train_source1.tsv", columns=["entity_id"])["entity_id"]
    qmap = pl.DataFrame({"source1_entity_id": ids.gather(pl.Series(rows)), "s1": pl.Series(rows, dtype=pl.UInt32)})
    truth = (read_tsv(data / "train" / "train_ground_truth.tsv").join(qmap, on="source1_entity_id")
             .with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids")
             .filter(pl.col("matched_entity_ids") != "").select("s1", pl.col("matched_entity_ids").alias("target_id")))
    if truth.n_unique() != len(truth):
        raise SystemExit("duplicate labelled query/target pairs")
    return truth


def load_ens3(ens3: Path, members: Path):
    """ens3 plan/manifest and its tune survivors: mean member p, cut by the manifest's cascade."""
    plan = json.loads((ens3 / "plan.json").read_text())
    manifest = json.loads((ens3 / "model" / "model_manifest.json").read_text())
    inputs = {"ens3_plan.json": sha256(ens3 / "plan.json"),
              "ens3_model_manifest.json": sha256(ens3 / "model" / "model_manifest.json")}
    frames = []
    for i, (prefix, model_sha) in enumerate(manifest["members"]):
        folder = members / Path(prefix).name / "model"
        if json.loads((folder / "model_manifest.json").read_text())["model_sha256"] != model_sha:
            raise SystemExit(f"{folder}: model_sha256 differs from the ens3 manifest")
        inputs[f"{Path(prefix).name}/tune_scores.parquet"] = sha256(folder / "tune_scores.parquet")
        frames.append(pl.read_parquet(folder / "tune_scores.parquet").sort("s1", "t").rename({"p": f"p_{i}"}))
    base = frames[0]
    for i, frame in enumerate(frames[1:], 1):
        if not frame.select("s1", "t").equals(base.select("s1", "t")):
            raise SystemExit(f"member {i} scored different tune survivors")
        base = base.with_columns(frame[f"p_{i}"])
    # Same average as tools/v6_ensemble_eval.py, which chose the ens3 threshold.
    scores = base.select("s1", "t", "target_id", "p1",
                         pl.mean_horizontal([f"p_{i}" for i in range(len(frames))]).cast(pl.Float32).alias("p"))
    cascade = manifest["cascade"]
    return plan, manifest, cut(scores, cascade["n"], cascade["floor"]), inputs


def cross_fit(feat: pl.DataFrame, names: list[str], model_dir: Path, threads: int):
    """2-fold OOF by S1 hash (stage2_cv.py); saves each fold model and checks that it reloads identically."""
    import lightgbm as lgb
    model_dir.mkdir(parents=True, exist_ok=True)
    fold = (feat["s1"].hash(seed=7) % 2).to_numpy()
    X = feat.select(names).to_numpy().astype(np.float32)
    y = feat["y"].to_numpy()
    oof, iterations, files = np.zeros(len(y), dtype=np.float64), [], []
    for f in (0, 1):
        train, test = np.where(fold != f)[0], np.where(fold == f)[0]
        inner = (pl.Series(feat["s1"].to_numpy()[train]).hash(seed=11) % 10 == 0).to_numpy()
        dtrain = lgb.Dataset(X[train[~inner]], y[train[~inner]], feature_name=names)
        dvalid = lgb.Dataset(X[train[inner]], y[train[inner]], reference=dtrain)
        booster = lgb.train(dict(PARAMS, num_threads=threads), dtrain, 4000, valid_sets=[dvalid],
                            callbacks=[lgb.early_stopping(150, verbose=False)])
        best = booster.best_iteration or booster.current_iteration()
        oof[test] = booster.predict(X[test], num_iteration=best)
        path = model_dir / f"fold{f}.txt"
        booster.save_model(str(path), num_iteration=best)
        check = test[:5000]
        diff = float(np.abs(lgb.Booster(model_file=str(path)).predict(X[check]) - oof[check]).max(initial=0))
        if diff > 1e-12:
            raise SystemExit(f"fold {f}: reloaded model differs from the trained one by {diff}")
        iterations.append(best)
        files.append({"file": f"model/{path.name}", "sha256": sha256(path), "best_iteration": best,
                      "reload_max_abs_diff": diff})
        print(f"fold {f}: best_iteration={best} reload diff={diff}", flush=True)
    return oof, files


def paired(per_a: pl.DataFrame, per_b: pl.DataFrame, country: pl.DataFrame) -> dict:
    both = (per_a.select("s1", "g", pl.col("f").alias("f_a")).join(per_b.select("s1", pl.col("f").alias("f_b")), on="s1")
            .join(country, on="s1", how="left"))
    d = (both["f_b"] - both["f_a"]).to_numpy()
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "queries": len(d),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "by_country": both.group_by("country").agg(pl.len().alias("queries"), pl.col("f_a").mean().alias("ens3"),
                                                        pl.col("f_b").mean().alias("stage2")).sort("country").to_dicts(),
            "singletons": both.filter(pl.col("g") == 0).select(pl.len().alias("queries"), pl.col("f_a").mean().alias("ens3"),
                                                               pl.col("f_b").mean().alias("stage2")).to_dicts()[0]}


def run_gate(args):
    started = time.time()
    out = Path(args.out)
    if (out / "gate.json").exists():
        raise SystemExit(f"{out} already holds a gate result; use a new output directory")
    plan, manifest, kept, inputs = load_ens3(Path(args.ens3), Path(args.members))
    cascade = manifest["cascade"]
    print(f"ens3 manifest: cascade={cascade} threshold={manifest['threshold']} members={len(manifest['members'])}; "
          f"tune survivors={len(kept):,}", flush=True)
    tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
    truth = tune_truth(Path(args.data), tune)
    gt = truth.select("s1", pl.col("target_id").alias("t"))
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    data = Path(args.data)
    feat = stage2_features(kept, cascade["n"], read_tsv(data / "train" / "train_source1.tsv"),
                           target_frame(data, "train"), args.threads)
    labels = gt.rename({"t": "target_id"}).with_columns(pl.lit(1, pl.Int8).alias("y"))
    feat = feat.join(labels, on=["s1", "target_id"], how="left").with_columns(pl.col("y").fill_null(0))
    names = [c for c in feat.columns if c not in ("s1", "t", "target_id", "y")]
    print(f"features {len(feat):,} rows x {len(names)} t={time.time() - started:.0f}s", flush=True)
    oof, models = cross_fit(feat, names, out / "model", args.threads)
    # "p1" in the feature frame is the stage-1 input, here the ens3 mean p.
    scored = feat.select("s1", "t", "target_id", pl.col("p1").alias("p_ens")).with_columns(pl.Series("p2", oof))
    th_e, f_e, per_e = best_threshold(scored, "p_ens", tune, gt)
    fixed = per_entity(tune, gt, best_threshold_pred(scored, "p_ens", manifest["threshold"]))["f"].mean()
    th_2, f_2, per_2 = best_threshold(scored, "p2", tune, gt)
    delta = paired(per_e, per_2, country)
    reproduced = abs(fixed - args.expect_f) <= 1e-4 if args.expect_f else None
    passed = bool(reproduced is not False and delta["mean"] > args.z * delta["se"])
    gate = {"version": VERSION, "tune_queries": len(tune), "survivor_rows": len(kept),
            "pairs_per_query": len(kept) / len(tune), "ens3_at_manifest_threshold": {"threshold": manifest["threshold"], "macro_f05": fixed},
            "ens3_reproduces_expected": reproduced, "expected_f": args.expect_f,
            "ens3_swept": {"threshold": th_e, "macro_f05": f_e}, "stage2_oof": {"threshold": th_2, "macro_f05": f_2},
            "paired_stage2_minus_ens3": delta, "rule": f"pass iff paired mean gain > {args.z} se and ens3 reproduced",
            "pass": passed, "seconds": time.time() - started, "peak_rss_gb": peak_rss_gb(), "audit_opened": False}
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(json.dumps({k: gate[k] for k in ("ens3_at_manifest_threshold", "ens3_swept", "stage2_oof", "pass")}), flush=True)
    print(f"paired gain {delta['mean']:+.5f} (se {delta['se']:.5f}); GATE {'PASS' if passed else 'FAIL'}", flush=True)
    if not passed:
        print("Gate failed: no release manifest written. Do not run full-test stage 2.", flush=True)
        return 2
    write_release(out, manifest, inputs, names, models, th_2, gate)
    return 0


def best_threshold_pred(scored, column, threshold):
    assigned = unique_assign(scored.select("s1", "t", "target_id", pl.col(column).alias("p")))
    return assigned.filter(pl.col("p") >= threshold).select("s1", pl.col("target_id").alias("t"))


def write_release(out: Path, manifest, inputs, names, models, threshold, gate):
    release = {"version": VERSION, "code": code_identity(),
               "ens3": {k: manifest[k] for k in ("version", "members", "model_sha256", "cascade", "threshold")},
               "candidate_policy": {"stage1": "ens3 mean p (v6 members)", "cut": "top n per S1 by v5.1 p1, ties by target row, "
                                    "then v5.1 p1 >= floor", **{k: manifest["cascade"][k] for k in ("n", "floor")}},
               "inputs_sha256": inputs, "features": names, "params": PARAMS, "models": models,
               "combine": "mean of the fold models' probabilities",
               "selection": {"rule": "one owner per target (highest p, ties to the lower S1 row), then p >= threshold",
                             "threshold": threshold},
               "gate": {k: gate[k] for k in ("stage2_oof", "ens3_at_manifest_threshold", "paired_stage2_minus_ens3", "pass")}}
    (out / "release.json").write_text(json.dumps(release, indent=1))
    print(f"release frozen: {out / 'release.json'} sha256 {sha256(out / 'release.json')}", flush=True)


# ------------------------------------------------------------------ assignment
def shard_rows(scores: Path) -> dict[str, int]:
    return {p.stem: pl.scan_parquet(p).select(pl.len()).collect().item() for p in sorted(scores.glob("*.parquet"))}


def run_assign(args):
    rows = shard_rows(Path(args.scores))
    if len(rows) != args.expect_shards:
        raise SystemExit(f"{args.scores}: {len(rows)} shard files, expected {args.expect_shards}")
    bench = dict(item.split("=", 1) for item in args.benchmark.split(","))
    if set(bench) - set(WORKERS) or len(set(bench.values())) != len(bench) or set(bench.values()) - set(rows):
        raise SystemExit(f"bad --benchmark {bench}")
    workers = {w: [s] for w, s in bench.items()}
    if args.rates:
        # Longest shard first to the worker that would finish it earliest (rows / measured rows per second).
        rate = {}
        for item in args.rates.split(","):
            w, path = item.split("=", 1)
            meta = json.loads(Path(path).read_text())
            rate[w] = meta["input_rows"] / meta["seconds"]
        load = {w: rows[bench[w]] / rate[w] for w in bench}
        for shard in sorted(set(rows) - set(bench.values()), key=lambda s: (-rows[s], s)):
            w = min(load, key=lambda k: (load[k] + rows[shard] / rate[k], k))
            workers[w].append(shard)
            load[w] += rows[shard] / rate[w]
        print({w: {"shards": len(s), "rows": sum(rows[x] for x in s), "projected_s": round(load[w])}
               for w, s in workers.items()}, flush=True)
    assignment = {"release_sha256": sha256(args.release), "shards": sorted(rows), "shard_rows": rows,
                  "workers": {w: sorted(s) for w, s in workers.items()}, "complete": bool(args.rates)}
    Path(args.out).write_text(json.dumps(assignment, indent=1))
    print(f"wrote {args.out} sha256 {sha256(args.out)}", flush=True)


def check_assignment(assignment: dict, release_sha: str):
    if assignment["release_sha256"] != release_sha:
        raise SystemExit("assignment was made for a different release manifest")
    listed = [s for shards in assignment["workers"].values() for s in shards]
    if len(listed) != len(set(listed)) or set(listed) - set(assignment["shards"]):
        raise SystemExit("assignment lists a shard twice or outside its shard set")
    if set(assignment["workers"]) - set(WORKERS):
        raise SystemExit(f"unknown worker in assignment: {set(assignment['workers']) - set(WORKERS)}")


# ------------------------------------------------------------------ worker
def predict_shard(frame: pl.DataFrame, release: dict, boosters, q_raw, t_raw, threads: int) -> pl.DataFrame:
    n, floor = release["candidate_policy"]["n"], release["candidate_policy"]["floor"]
    if frame.is_empty():
        return pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "target_id": pl.String, "p1": pl.Float32,
                                    "p_ens": pl.Float32, "p": pl.Float32})
    per_s1 = int(frame.group_by("s1").len()["len"].max())
    if float(frame["p1"].min()) < floor - 1e-6 or per_s1 > n or frame.select("s1", "t").is_duplicated().any():
        raise SystemExit(f"shard does not follow the ens3 cut (min p1 {frame['p1'].min()}, max {per_s1} per S1)")
    feat = stage2_features(frame, n, q_raw, t_raw, threads)
    if feat.columns[3:] != release["features"]:
        raise SystemExit("feature order differs from the release manifest")
    X = feat.select(release["features"]).to_numpy().astype(np.float32)
    p = np.mean([b.predict(X, num_threads=threads) for b in boosters], axis=0).astype(np.float32)
    out = (feat.select("s1", "t", "target_id").with_columns(pl.Series("p", p))
           .join(frame.select("s1", "t", "p1", pl.col("p").alias("p_ens")), on=["s1", "t"]))
    if len(out) != len(frame):
        raise SystemExit("stage 2 did not score every survivor exactly once")
    return out.select(OUT_COLUMNS).sort("s1", "t")


def load_release(path: Path):
    import lightgbm as lgb
    release = json.loads(path.read_text())
    if not release.get("gate", {}).get("pass"):
        raise SystemExit("release gate did not pass; full-test stage 2 is not allowed")
    boosters = []
    for model in release["models"]:
        file = path.parent / model["file"]
        if sha256(file) != model["sha256"]:
            raise SystemExit(f"{file}: checksum differs from the release manifest")
        boosters.append(lgb.Booster(model_file=str(file)))
    return release, boosters


def run_predict(args):
    if args.worker not in WORKERS:
        raise SystemExit(f"--worker must be one of {WORKERS}")
    release_path = Path(args.release)
    release, boosters = load_release(release_path)
    assignment = json.loads(Path(args.assignment).read_text())
    check_assignment(assignment, sha256(release_path))
    shards = args.shards.split(",") if args.shards else assignment["workers"].get(args.worker, [])
    if not shards or set(shards) - set(assignment["workers"].get(args.worker, [])):
        raise SystemExit(f"{args.worker} is not assigned {sorted(set(shards) - set(assignment['workers'].get(args.worker, [])))}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    q_raw = read_tsv(data / "test" / "test_source1.tsv")
    t_raw = target_frame(data, "test")
    country = q_raw["country"]
    for shard in shards:
        meta_path = out / f"{shard}.json"
        if meta_path.exists():
            print(f"{shard}: already done", flush=True)
            continue
        started = time.time()
        source = Path(args.scores) / f"{shard}.parquet"
        frame = pl.read_parquet(source, columns=["s1", "t", "target_id", "p1", "p"])
        result = predict_shard(frame, release, boosters, q_raw, t_raw, args.threads)
        path = out / f"{shard}.parquet"
        result.write_parquet(path)
        above = result.filter(pl.col("p") >= release["selection"]["threshold"])
        meta = {"shard": shard, "worker": args.worker, "release_sha256": sha256(release_path),
                "code": code_identity(), "input_sha256": sha256(source), "input_rows": len(frame),
                "rows": len(result), "s1_with_survivors": result["s1"].n_unique(), "output_sha256": sha256(path),
                "above_threshold_by_country": above.with_columns(country.gather(above["s1"]).alias("country"))
                .group_by("country").len().sort("country").to_dicts(),
                "seconds": time.time() - started, "peak_rss_gb": peak_rss_gb()}
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{shard}: {len(result):,} rows in {meta['seconds']:.0f}s peak {meta['peak_rss_gb']}", flush=True)


# ------------------------------------------------------------------ collector
def run_collect(args):
    release_path = Path(args.release)
    release, _ = load_release(release_path)
    release_sha = sha256(release_path)
    assignment = json.loads(Path(args.assignment).read_text())
    check_assignment(assignment, release_sha)
    found = {}
    for folder in args.inputs.split(","):
        for meta_path in sorted(Path(folder).glob("*.json")):
            meta = json.loads(meta_path.read_text())
            if meta["shard"] in found:
                raise SystemExit(f"{meta['shard']} is present twice ({found[meta['shard']][0]} and {folder})")
            found[meta["shard"]] = (folder, meta)
    missing, extra = set(assignment["shards"]) - set(found), set(found) - set(assignment["shards"])
    if missing or extra:
        raise SystemExit(f"shard set mismatch: missing {sorted(missing)} unexpected {sorted(extra)}")
    stage = Path(args.out) / "stage"
    seen = set()
    for shard, (folder, meta) in sorted(found.items()):
        path = Path(folder) / f"{shard}.parquet"
        if meta["release_sha256"] != release_sha or sha256(path) != meta["output_sha256"]:
            raise SystemExit(f"{shard}: release or output checksum mismatch")
        if meta["rows"] != meta["input_rows"] or meta["rows"] != assignment["shard_rows"][shard]:
            raise SystemExit(f"{shard}: row count differs from its ens3 input")
        s1 = set(pl.read_parquet(path, columns=["s1"])["s1"].unique().to_list())
        if s1 & seen:
            raise SystemExit(f"{shard}: S1 rows also present in another shard")
        seen |= s1
        (stage / "score").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, stage / "score" / f"{shard}.parquet")
    heads = {json.dumps(meta["code"].get("git_head")) for _, meta in found.values()}
    if len(heads) != 1:
        raise SystemExit(f"shards were produced by different code versions: {heads}")
    Store(root=stage).put_json("model/model_manifest.json", {"threshold": release["selection"]["threshold"],
                                                             "model_sha256": release_sha})
    # v5 final assembly: global one owner per target, threshold, every test S1 in both TSVs, validator.
    report = run_final({"inputs": sorted(found)}, Store(root=stage), Context(args.data, Path(args.out), 1, {}))
    output = Path(args.out) / "output"
    q = read_tsv(Path(args.data) / "test" / "test_source1.tsv", columns=["entity_id", "country"])
    match = read_tsv(output / "matching_results.tsv").join(q, left_on="source1_entity_id", right_on="entity_id")
    n = pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    report["label_free_by_country"] = (match.with_columns(n.alias("k")).group_by("country").agg(
        pl.len().alias("s1"), pl.col("k").mean().alias("mean_predicted"), (pl.col("k") == 0).mean().alias("empty"),
        (pl.col("k") >= 8).mean().alias("eight_plus")).sort("country").to_dicts())
    report["shards"] = len(found)
    report["release_sha256"] = release_sha
    report["tsv_sha256"] = {name: sha256(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")}
    (Path(args.out) / "collect_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "validator_output"}, indent=1, default=str), flush=True)
    if report.get("validator_exit") not in (0, None):
        raise SystemExit("official validator failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    threads = int(os.environ.get("ER_V6_THREADS", os.cpu_count() or 4))
    g = sub.add_parser("gate")
    for name in ("--ens3", "--members", "--data", "--out"):
        g.add_argument(name, required=True)
    g.add_argument("--expect-f", type=float, default=0.97061, help="ens3 tune F at its threshold (control); 0 to skip")
    g.add_argument("--z", type=float, default=2.0, help="required paired gain in standard errors")
    a = sub.add_parser("assign")
    for name in ("--release", "--scores", "--out", "--benchmark"):
        a.add_argument(name, required=True)
    a.add_argument("--rates")
    a.add_argument("--expect-shards", type=int, default=88)
    p = sub.add_parser("predict")
    for name in ("--release", "--assignment", "--worker", "--scores", "--data", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--shards")
    c = sub.add_parser("collect")
    for name in ("--release", "--assignment", "--inputs", "--data", "--out"):
        c.add_argument(name, required=True)
    for s in (g, p):
        s.add_argument("--threads", type=int, default=threads)
    args = parser.parse_args(argv)
    return {"gate": run_gate, "assign": run_assign, "predict": run_predict, "collect": run_collect}[args.step](args) or 0


if __name__ == "__main__":
    sys.exit(main())
