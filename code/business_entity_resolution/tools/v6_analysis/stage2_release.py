"""V7 head on the frozen V6 ens3 survivors (plan revision 3, WP2): gate, release, cached features,
shard-aware prediction and the collector.

    python tools/v6_analysis/stage2_release.py gate     --ens3 E --members M --data D --out R [--families F0F1,F0F1F3]
    python tools/v6_analysis/stage2_release.py assign   --scores S --out A.json --benchmark WORKER=SHARD,...
                                                        [--release R/release.json --rates WORKER=META.json,...]
    python tools/v6_analysis/stage2_release.py features --split test|audit --scores S --data D --out C
                                                        (--shards SHARD,... | --assignment A.json --worker W)
    python tools/v6_analysis/stage2_release.py predict  --release R/release.json --assignment A.json --worker W
                                                        --scores S --data D --out O [--cache C] [--shards SHARD,...]
    python tools/v6_analysis/stage2_release.py collect  --release R/release.json --assignment A.json
                                                        --inputs O1,O2,O3 --data D --out F --expect-full
                                                        --v6-matching V6/matching_results.tsv --v6-candidates V6/candidate_pairs.tsv

Local copies of Aditya's queues, read-only (synced from S3):
  E  shared/er-v6ens3-20260927/{plan.json, model/model_manifest.json}
  M  one folder per ens3 member, named like the member prefix's last part, with model/{model_manifest.json,
     tune_scores.parquet}
  S  survivor shards with s1, t, target_id, p1 (v5.1) and p6 (named p in the ens3 score shards)
  D  the supplied dataset directory (train/, test/; ../utils/validate_submission.py)

C(q) is the ens3 cut (top n by v5.1 p1, ties by target row, then p1 >= floor), read from the ens3 manifest and
never changed here. Families: F0 scores (p6 and the real v5.1 p), F1 siblings (v6_stage2.features), F3 population
rivals (src.v7_population, used once WP1 is merged). Rungs B1 = F0F1, B2 = F0F1F3, each 5-fold cross-fitted on
the tune S1 with folds from SHA-256 of the S1 entity ID. The deployed head is the mean of the five fold models;
s = sigmoid((1-alpha) logit p6 + alpha logit p7), then one owner per target and one global threshold. The audit
partition is never read here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

from _common import PACKAGE, read_tsv
from src.metric import per_entity, unique_assign
from src.v6_stage2 import features, logit, survivors, text_views
from src.v6_train import cut

VERSION = "v7-head-1"
WORKERS = ("abhigyan-r7i4xl", "aditya-worker", "akash-r5xl")
FAMILIES = ("F0F1", "F0F1F3")
K_FOLDS = 5
MONOTONE = ("p6", "p6_logit", "v51_p", "v51_logit")
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=40, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=1, deterministic=True,
              force_col_wise=True, monotone_constraints_method="intermediate")
ALPHAS = (0.5, 1.0)
G2 = {"min_gain": 0.0015, "z": 1.645}
B2_MIN_OVER_B1 = 0.0003
OUT_COLUMNS = ["s1", "t", "target_id", "p1", "p6", "p7", "s"]
EXPECTED = {"test_s1": 1_732_544, "test_pairs": 9_266_800, "test_zero_survivor_s1": 23_644}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_mod(keys, k: int, salt: str) -> np.ndarray:
    """SHA-256 of salt + key, mod k: the same on every machine (no Python or Polars hashing)."""
    return np.array([int.from_bytes(hashlib.sha256(f"{salt}{key}".encode()).digest()[:8], "little") % k
                     for key in keys], dtype=np.int64)


def pair_digest(s1, t, split: str) -> str:
    """SHA-256 of the sorted (s1, t) row pairs as little-endian uint32, after a header naming the row maps."""
    s1, t = np.asarray(s1, dtype=np.uint32), np.asarray(t, dtype=np.uint32)
    order = np.lexsort((t, s1))
    h = hashlib.sha256(f"er-v7 pairs; s1 = {split}_source1 row; t = {split} S2 rows then S3 rows\n".encode())
    h.update(np.stack([s1[order], t[order]], axis=1).astype("<u4").tobytes())
    return h.hexdigest()


def peak_rss_gb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20  # KiB on Linux
    except ImportError:
        return None


def code_identity() -> dict:
    files = {"stage2_release.py": Path(__file__).resolve(), "v6_stage2.py": PACKAGE / "src" / "v6_stage2.py",
             "text_norm_v5.py": PACKAGE / "src" / "text_norm_v5.py", "v7_population.py": PACKAGE / "src" / "v7_population.py"}
    ident = {name: sha256(path) for name, path in files.items() if path.exists()}
    try:
        ident["git_head"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PACKAGE, capture_output=True,
                                           text=True, check=True).stdout.strip()
        ident["git_dirty"] = bool(subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=PACKAGE,
                                                 capture_output=True, text=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        ident["git_head"] = None
    return ident


def feature_code_sha() -> str:
    ident = code_identity()
    return hashlib.sha256(json.dumps({k: v for k, v in ident.items() if k.endswith(".py")}, sort_keys=True).encode()).hexdigest()


def pip_freeze() -> list[str]:
    try:
        return subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split()
    except OSError:
        return []


def target_frame(data: Path, split: str) -> pl.DataFrame:
    """Raw S2 rows then S3 rows: position = the global target row `t` of the score files."""
    return pl.concat([read_tsv(data / split / f"{split}_source{s}.tsv").with_columns(pl.lit(s, pl.UInt8).alias("src"))
                      for s in (2, 3)])


def population():
    """WP1's module, or None until it is merged."""
    try:
        from src import v7_population
        return v7_population
    except ImportError:
        return None


# ------------------------------------------------------------------ features
def normalize_scores(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.rename({"p": "p6"}) if "p6" not in frame.columns else frame


def check_cut(frame: pl.DataFrame, n: int, floor: float):
    if frame.is_empty():
        return
    per_s1 = int(frame.group_by("s1").len()["len"].max())
    if float(frame["p1"].min()) < floor - 1e-6 or per_s1 > n or frame.select("s1", "t").is_duplicated().any():
        raise SystemExit(f"shard does not follow the ens3 cut (min p1 {frame['p1'].min()}, max {per_s1} per S1)")


def build_features(scored: pl.DataFrame, n: int, q_raw, t_raw, threads: int, families: str, index=None) -> pl.DataFrame:
    """Features of already-cut survivors (s1, t, target_id, p1 = v5.1, p6 = ens3 mean), one row per pair."""
    surv = survivors(scored.select("s1", "t", "target_id", pl.col("p6").alias("p")), n)  # rank by p6
    qrows, trows = surv["s1"].unique().sort(), surv["t"].unique().sort()
    q_text = text_views(q_raw.gather(qrows))
    t_sel = t_raw.gather(trows)
    t_text = text_views(t_sel).with_columns(t_sel["src"])
    surv = (surv.join(pl.DataFrame({"s1": qrows, "qi": np.arange(len(qrows), dtype=np.uint32)}), on="s1")
            .join(pl.DataFrame({"t": trows, "ti": np.arange(len(trows), dtype=np.uint32)}), on="t"))
    # v6_stage2 names its stage-1 input p1; here that input is p6. The real v5.1 p joins as v51_p.
    feat = features(surv, q_text, t_text, workers=threads).rename({"p1": "p6", "p1_logit": "p6_logit"})
    feat = feat.join(scored.select("s1", "t", pl.col("p1").alias("v51_p")), on=["s1", "t"], how="left",
                     maintain_order="left")
    feat = feat.with_columns(pl.Series("v51_logit", logit(feat["v51_p"].to_numpy())))
    if "F3" in families:
        rivals = population().rival_features(feat.select("s1", "t"), q_raw, t_raw, index, threads)
        feat = feat.join(rivals, on=["s1", "t"], how="left", maintain_order="left")
    if len(feat) != len(scored):
        raise SystemExit("features do not cover every survivor exactly once")
    return feat


def feature_names(feat: pl.DataFrame, families: str) -> list[str]:
    names = [c for c in feat.columns if c not in ("s1", "t", "target_id", "y")]
    if "F3" not in families and population() is not None:
        names = [c for c in names if c not in set(population().F3_COLUMNS)]
    return names


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
    return plan, manifest, cut(scores, cascade["n"], cascade["floor"]).rename({"p": "p6"}), inputs


def folds_for(feat: pl.DataFrame, q_raw: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Per-row fold (SHA-256 of the S1 entity ID mod 5; exact (country, core, addr_norm) duplicates follow their
    lowest-row member) and the inner 10% early-stopping flag (a second stable hash)."""
    rows = feat["s1"].unique().sort()
    raw = q_raw.gather(rows)
    views = text_views(raw)
    ids = raw["entity_id"].to_list()
    q = pl.DataFrame({"s1": rows, "country": raw["country"].fill_null(""), "core": views["core"].fill_null(""),
                      "addr": views["addr_norm"].fill_null(""), "fold": stable_mod(ids, K_FOLDS, "v7-fold:"),
                      "inner": stable_mod(ids, 10, "v7-inner:") == 0})
    fp = pl.concat_str(["country", "core", "addr"], separator="\x1f")
    q = q.with_columns(pl.when(pl.col("core") != "").then(pl.col("fold").sort_by("s1").first().over(fp))
                       .otherwise(pl.col("fold")).alias("fold"))
    joined = feat.select("s1").join(q.select("s1", "fold", "inner"), on="s1", how="left", maintain_order="left")
    return joined["fold"].to_numpy(), joined["inner"].to_numpy()


def cross_fit(feat, names, fold, inner, model_dir: Path, tag: str, threads: int):
    """K-fold OOF; saves each fold model and checks that it reloads to the same predictions (<= 1e-6)."""
    import lightgbm as lgb
    model_dir.mkdir(parents=True, exist_ok=True)
    monotone = [1 if n in MONOTONE else 0 for n in names]
    X = feat.select(names).to_numpy().astype(np.float32)
    y = feat["y"].to_numpy()
    oof, files = np.zeros(len(y), dtype=np.float64), []
    for f in range(K_FOLDS):
        train, test = np.where(fold != f)[0], np.where(fold == f)[0]
        stop = inner[train]
        dtrain = lgb.Dataset(X[train[~stop]], y[train[~stop]], feature_name=names)
        dvalid = lgb.Dataset(X[train[stop]], y[train[stop]], reference=dtrain)
        booster = lgb.train(dict(PARAMS, num_threads=threads, monotone_constraints=monotone), dtrain, 4000,
                            valid_sets=[dvalid], callbacks=[lgb.early_stopping(150, verbose=False)])
        best = booster.best_iteration or booster.current_iteration()
        oof[test] = booster.predict(X[test], num_iteration=best)
        path = model_dir / f"{tag}_fold{f}.txt"
        booster.save_model(str(path), num_iteration=best)
        check = test[:1000]
        diff = float(np.abs(lgb.Booster(model_file=str(path)).predict(X[check]) - oof[check]).max(initial=0))
        if diff > 1e-6:
            raise SystemExit(f"{tag} fold {f}: reloaded model differs from the trained one by {diff}")
        files.append({"file": f"model/{path.name}", "sha256": sha256(path), "best_iteration": best,
                      "rows": int(len(test)), "reload_max_abs_diff": diff})
        print(f"{tag} fold {f}: best_iteration={best} reload diff={diff}", flush=True)
    return oof, files, monotone


def blend(p6, p7, alpha: float) -> np.ndarray:
    z = (1 - alpha) * logit(np.asarray(p6)).astype(np.float64) + alpha * logit(np.asarray(p7)).astype(np.float64)
    return (1 / (1 + np.exp(-z))).astype(np.float32)


def decide(scored: pl.DataFrame, column: str, threshold: float) -> pl.DataFrame:
    assigned = unique_assign(scored.select("s1", "t", "target_id", pl.col(column).alias("p")))
    return assigned.filter(pl.col("p") >= threshold).select("s1", pl.col("target_id").alias("t"))


def sweep(scored, column, tune, gt):
    """One owner per target, then the global threshold (0.02 grid, then 0.0025 around the best); ties to the
    higher threshold. Returns (threshold, macro F0.5, per-query frame)."""
    assigned = unique_assign(scored.select("s1", "t", "target_id", pl.col(column).alias("p")))

    def score(th):
        pred = assigned.filter(pl.col("p") >= th).select("s1", pl.col("target_id").alias("t"))
        return float(per_entity(tune, gt, pred)["f"].mean())

    grid = [(float(th), score(th)) for th in np.unique(np.r_[np.arange(0.2, 0.961, 0.02), [0.97, 0.98, 0.99]])]
    th, _ = max(grid, key=lambda item: (item[1], item[0]))
    grid += [(float(x), score(x)) for x in np.arange(max(0.05, th - 0.02), min(0.995, th + 0.021), 0.0025)]
    th, best = max(grid, key=lambda item: (item[1], item[0]))
    return th, best, per_entity(tune, gt, decide(scored, column, th))


def paired(per_a: pl.DataFrame, per_b: pl.DataFrame, country: pl.DataFrame) -> dict:
    both = (per_a.select("s1", "g", pl.col("f").alias("f_a")).join(per_b.select("s1", pl.col("f").alias("f_b")), on="s1")
            .join(country, on="s1", how="left"))
    d = (both["f_b"] - both["f_a"]).to_numpy()

    def part(frame):
        x = (frame["f_b"] - frame["f_a"]).to_numpy()
        return {"queries": len(x), "base": float(frame["f_a"].mean() or 0), "new": float(frame["f_b"].mean() or 0),
                "delta": float(x.mean()) if len(x) else 0.0}

    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "queries": len(d),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "by_country": {c: part(both.filter(pl.col("country") == c)) for c in sorted(both["country"].unique().to_list())},
            "singletons": part(both.filter(pl.col("g") == 0)), "one_true_match": part(both.filter(pl.col("g") == 1))}


def run_gate(args):
    started = time.time()
    out = Path(args.out)
    if (out / "gate.json").exists():
        raise SystemExit(f"{out} already holds a gate result; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    data = Path(args.data)
    families = [f for f in args.families.split(",") if f]
    if not families or set(families) - set(FAMILIES) or ("F0F1F3" in families and population() is None):
        raise SystemExit(f"--families must be from {FAMILIES}; F0F1F3 needs src/v7_population.py")
    plan, manifest, kept, inputs = load_ens3(Path(args.ens3), Path(args.members))
    cascade = manifest["cascade"]
    tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
    gt = tune_truth(data, tune).select("s1", pl.col("target_id").alias("t"))
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    per_v6 = per_entity(tune, gt, decide(kept, "p6", manifest["threshold"]))
    g0 = float(per_v6["f"].mean())
    print(f"ens3 manifest: cascade={cascade} threshold={manifest['threshold']}; tune survivors={len(kept):,}; "
          f"V6 at {manifest['threshold']}: {g0:.5f}", flush=True)
    if args.expect_f and abs(g0 - args.expect_f) > 1e-4:
        raise SystemExit(f"G0 failed: V6 at its manifest threshold scores {g0:.5f}, expected {args.expect_f}")
    q_raw, t_raw = read_tsv(data / "train" / "train_source1.tsv"), target_frame(data, "train")
    index = population().build_index(q_raw) if "F0F1F3" in families else None
    feat = build_features(kept, cascade["n"], q_raw, t_raw, args.threads, families[-1], index)
    labels = gt.rename({"t": "target_id"}).with_columns(pl.lit(1, pl.Int8).alias("y"))
    feat = feat.join(labels, on=["s1", "target_id"], how="left", maintain_order="left").with_columns(pl.col("y").fill_null(0))
    fold, inner = folds_for(feat, q_raw)
    print(f"features {len(feat):,} rows x {len(feat.columns) - 4} t={time.time() - started:.0f}s", flush=True)
    rungs = {}
    for fam in families:
        tag = {"F0F1": "B1", "F0F1F3": "B2"}[fam]
        names = feature_names(feat, fam)
        oof, models, monotone = cross_fit(feat, names, fold, inner, out / "model", tag, args.threads)
        scored = feat.select("s1", "t", "target_id", "p6").with_columns(pl.Series("p7", oof.astype(np.float32)))
        options = []
        for alpha in ALPHAS:
            scored = scored.with_columns(pl.Series("s", blend(scored["p6"], scored["p7"], alpha)))
            th, f, per = sweep(scored, "s", tune, gt)
            options.append((f, alpha, th, per))
            print(f"{tag} alpha={alpha}: threshold {th:.4f} F {f:.5f}", flush=True)
        f, alpha, th, per = max(options, key=lambda o: (o[0], o[1], o[2]))
        rungs[tag] = {"families": fam, "features": names, "monotone": monotone, "models": models, "alpha": alpha,
                      "threshold": th, "macro_f05": f, "paired_vs_v6": paired(per_v6, per, country),
                      "options": [{"alpha": o[1], "threshold": o[2], "macro_f05": o[0]} for o in options]}
    chosen = "B1" if "B1" in rungs else "B2"
    if "B1" in rungs and "B2" in rungs and rungs["B2"]["macro_f05"] - rungs["B1"]["macro_f05"] >= B2_MIN_OVER_B1:
        chosen = "B2"
    d = rungs[chosen]["paired_vs_v6"]
    g2 = g2_pass(d)
    gate = {"version": VERSION, "tune_queries": len(tune), "survivor_rows": len(kept), "pairs_per_query": len(kept) / len(tune),
            "g0": {"v6_threshold": manifest["threshold"], "v6_macro_f05": g0, "expected": args.expect_f},
            "rungs": {k: {x: v[x] for x in ("families", "alpha", "threshold", "macro_f05", "paired_vs_v6", "options")}
                      for k, v in rungs.items()},
            "chosen": chosen, "g2_rule": G2, "g2_pass": g2, "pass": g2, "seconds": time.time() - started,
            "peak_rss_gb": peak_rss_gb(), "audit_opened": False,
            "index_digest": population().index_digest(index) if index is not None else None}
    (out / "gate.json").write_text(json.dumps(gate, indent=1))
    print(f"chosen {chosen}: F {rungs[chosen]['macro_f05']:.5f} vs V6 {g0:.5f}, gain {d['mean']:+.5f} "
          f"(se {d['se']:.5f}); G2 {'PASS' if g2 else 'FAIL'}", flush=True)
    if not g2:
        print("G2 failed: no release manifest written. V6 ships.", flush=True)
        return 2
    write_release(out, manifest, inputs, rungs[chosen], chosen, gate)
    return 0


def g2_pass(d: dict) -> bool:
    """Plan section 6, G2: gain >= +0.0015 and gain - 1.645 SE > 0 on the same tune S1; India and US deltas >= 0."""
    return bool(d["mean"] >= G2["min_gain"] and d["mean"] - G2["z"] * d["se"] > 0
                and all(d["by_country"][c]["delta"] >= 0 for c in ("India", "US") if c in d["by_country"]))


def write_release(out: Path, manifest, inputs, rung, tag, gate):
    release = {"version": VERSION, "code": code_identity(), "feature_code_sha256": feature_code_sha(),
               "ens3": {k: manifest[k] for k in ("version", "members", "model_sha256", "cascade", "threshold")},
               "candidate_policy": {"stage1": "ens3 mean p (v6 members) = p6", "cut": "top n per S1 by v5.1 p1, ties by "
                                    "target row, then v5.1 p1 >= floor", **{k: manifest["cascade"][k] for k in ("n", "floor")}},
               "inputs_sha256": inputs, "rung": tag, "families": rung["families"], "features": rung["features"],
               "monotone_constraints": rung["monotone"], "params": PARAMS, "models": rung["models"],
               "combine": "mean of the fold models' probabilities = p7",
               "selection": {"score": "s = sigmoid((1 - alpha) logit(p6) + alpha logit(p7)), logits clipped at 1e-6",
                             "alpha": rung["alpha"], "rule": "one owner per target (highest s, ties to the lower S1 "
                             "row), then s >= threshold", "threshold": rung["threshold"]},
               "index_digest": gate["index_digest"],
               "gate": {k: gate[k] for k in ("g0", "chosen", "g2_pass", "pass")}}
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
    assignment = {"release_sha256": sha256(args.release) if args.release else None, "shards": sorted(rows),
                  "shard_rows": rows, "workers": {w: sorted(s) for w, s in workers.items()}, "complete": bool(args.rates)}
    Path(args.out).write_text(json.dumps(assignment, indent=1))
    print(f"wrote {args.out} sha256 {sha256(args.out)}", flush=True)


def check_assignment(assignment: dict, release_sha: str | None):
    if release_sha and assignment["release_sha256"] != release_sha:
        raise SystemExit("assignment was made for a different release manifest (or none)")
    listed = [s for shards in assignment["workers"].values() for s in shards]
    if len(listed) != len(set(listed)) or set(listed) - set(assignment["shards"]):
        raise SystemExit("assignment lists a shard twice or outside its shard set")
    if set(assignment["workers"]) - set(WORKERS):
        raise SystemExit(f"unknown worker in assignment: {set(assignment['workers']) - set(WORKERS)}")


def worker_shards(args, assignment) -> list[str]:
    if args.worker not in WORKERS:
        raise SystemExit(f"--worker must be one of {WORKERS}")
    mine = assignment["workers"].get(args.worker, [])
    shards = args.shards.split(",") if args.shards else mine
    if not shards or set(shards) - set(mine):
        raise SystemExit(f"{args.worker} is not assigned {sorted(set(shards) - set(mine)) or 'any shard'}")
    return shards


# ------------------------------------------------------------------ feature cache
class Split:
    """Raw records, rival index and pins for one data split, loaded once per process."""

    def __init__(self, data: Path, split: str, threads: int):
        self.split = "train" if split in ("tune", "audit") else "test"
        self.q_raw = read_tsv(data / self.split / f"{self.split}_source1.tsv")
        self.t_raw = target_frame(data, self.split)
        pop = population()
        self.index = pop.build_index(self.q_raw) if pop else None
        self.families = "F0F1F3" if pop else "F0F1"
        self.index_digest = pop.index_digest(self.index) if pop else None
        self.threads = threads

    def pins(self, source: Path) -> dict:
        return {"input_sha256": sha256(source), "feature_code_sha256": feature_code_sha(),
                "index_digest": self.index_digest, "families": self.families, "split": self.split}


def cached_features(split: Split, source: Path, cache: Path | None, n: int, floor: float):
    """Features of one survivor shard, from the cache when every pin matches, else computed (and cached)."""
    frame = normalize_scores(pl.read_parquet(source))
    check_cut(frame, n, floor)
    if frame.is_empty():
        return frame, None
    pins = split.pins(source)
    if cache is not None:
        path, meta_path = cache / f"{source.stem}.parquet", cache / f"{source.stem}.feat.json"
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if {k: meta.get(k) for k in pins} == pins and meta.get("output_sha256") == sha256(path):
                return frame, pl.read_parquet(path)
    feat = build_features(frame, n, split.q_raw, split.t_raw, split.threads, split.families, split.index)
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        tmp = cache / f"{source.stem}.parquet.tmp"
        feat.write_parquet(tmp)
        os.replace(tmp, path)
        meta = dict(pins, columns=feat.columns, rows=len(feat), output_sha256=sha256(path))
        tmp = cache / f"{source.stem}.feat.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
    return frame, feat


def run_features(args):
    """Cache features for the given shards (test or audit survivors); the head is not needed."""
    if args.shards:
        shards = args.shards.split(",")
    else:
        assignment = json.loads(Path(args.assignment).read_text())
        check_assignment(assignment, None)
        shards = worker_shards(args, assignment)
    split = Split(Path(args.data), args.split, args.threads)
    for shard in shards:
        started = time.time()
        frame, feat = cached_features(split, Path(args.scores) / f"{shard}.parquet", Path(args.out), args.n, args.floor)
        print(f"{shard}: {len(frame):,} pairs, {'no' if feat is None else feat.width} feature columns "
              f"in {time.time() - started:.0f}s peak {peak_rss_gb()}", flush=True)


# ------------------------------------------------------------------ worker
def load_release(path: Path):
    import lightgbm as lgb
    release = json.loads(path.read_text())
    if not release.get("gate", {}).get("pass"):
        raise SystemExit("release gate did not pass; full-test prediction is not allowed")
    boosters = []
    for model in release["models"]:
        file = path.parent / model["file"]
        if sha256(file) != model["sha256"]:
            raise SystemExit(f"{file}: checksum differs from the release manifest")
        boosters.append(lgb.Booster(model_file=str(file)))
    return release, boosters


def apply_release(feat: pl.DataFrame, release: dict, boosters, threads: int) -> pl.DataFrame:
    """p7 = mean of the fold models, s = the frozen blend. Returns s1, t, target_id, p6, p7, s."""
    missing = set(release["features"]) - set(feat.columns)
    if missing:
        raise SystemExit(f"cached features lack {sorted(missing)}")
    X = feat.select(release["features"]).to_numpy().astype(np.float32)
    p7 = np.mean([b.predict(X, num_threads=threads) for b in boosters], axis=0).astype(np.float32)
    if not np.isfinite(p7).all():
        raise SystemExit("non-finite p7")
    s = blend(feat["p6"].to_numpy(), p7, release["selection"]["alpha"])
    return feat.select("s1", "t", "target_id", "p6").with_columns(pl.Series("p7", p7), pl.Series("s", s))


def run_predict(args):
    release_path = Path(args.release)
    release, boosters = load_release(release_path)
    assignment = json.loads(Path(args.assignment).read_text())
    check_assignment(assignment, sha256(release_path))
    shards = worker_shards(args, assignment)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    split = Split(Path(args.data), "test", args.threads)
    if "F3" in release["families"] and "F3" not in split.families:
        raise SystemExit("this machine lacks the F3 module the release needs")
    policy = release["candidate_policy"]
    country = split.q_raw["country"]
    for shard in shards:
        path, meta_path = out / f"{shard}.parquet", out / f"{shard}.json"
        source = Path(args.scores) / f"{shard}.parquet"
        if meta_path.exists() and path.exists():
            meta = json.loads(meta_path.read_text())
            if (meta.get("release_sha256") == sha256(release_path) and meta.get("output_sha256") == sha256(path)
                    and meta.get("input_sha256") == sha256(source)):
                print(f"{shard}: already done, pins verified", flush=True)
                continue
        started = time.time()
        frame, feat = cached_features(split, source, Path(args.cache) if args.cache else None, policy["n"], policy["floor"])
        if feat is None:
            result = pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "target_id": pl.String, "p1": pl.Float32,
                                          "p6": pl.Float32, "p7": pl.Float32, "s": pl.Float32})
        else:
            scored = apply_release(feat, release, boosters, args.threads)
            result = scored.join(frame.select("s1", "t", "p1"), on=["s1", "t"]).select(OUT_COLUMNS).sort("s1", "t")
        if len(result) != len(frame):
            raise SystemExit(f"{shard}: scored {len(result)} of {len(frame)} survivors")
        tmp = out / f"{shard}.parquet.tmp"
        result.write_parquet(tmp)
        os.replace(tmp, path)
        above = result.filter(pl.col("s") >= release["selection"]["threshold"])
        f3 = {}
        if feat is not None and "rJ_present" in feat.columns:
            by = feat.select("s1", "rJ_present", "t_overflow").with_columns(country.gather(feat["s1"]).alias("country"))
            f3 = {r["country"]: r for r in by.group_by("country").agg(
                pl.len().alias("pairs"), pl.col("rJ_present").cast(pl.Float64).sum(),
                pl.col("t_overflow").cast(pl.Float64).sum()).to_dicts()}
        meta = {"shard": shard, "worker": args.worker, "release_sha256": sha256(release_path),
                "assignment_sha256": sha256(args.assignment), "code": code_identity(),
                "input_sha256": sha256(source), "input_rows": len(frame), "rows": len(result),
                "pair_digest": pair_digest(result["s1"], result["t"], "test"), "s1_with_survivors": result["s1"].n_unique(),
                "index_digest": split.index_digest, "families": split.families, "output_sha256": sha256(path),
                "above_threshold_by_country": above.with_columns(country.gather(above["s1"]).alias("country"))
                .group_by("country").len().sort("country").to_dicts(), "f3_rates": f3,
                "seconds": time.time() - started, "peak_rss_gb": peak_rss_gb(), "pip_freeze": pip_freeze()}
        tmp = out / f"{shard}.json.tmp"
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, meta_path)
        print(f"{shard}: {len(result):,} rows in {meta['seconds']:.0f}s peak {meta['peak_rss_gb']}", flush=True)


# ------------------------------------------------------------------ collector
def france_out_of_envelope(metas, per_country) -> tuple[bool, dict]:
    """Plan section 6: France's F3 rates outside the US/India envelope, or V7's France output outside its bands."""
    rates = {}
    for meta in metas:
        for c, r in meta.get("f3_rates", {}).items():
            acc = rates.setdefault(c, {"pairs": 0, "rJ_present": 0.0, "t_overflow": 0.0})
            for k in acc:
                acc[k] += r[k]
    reasons = []
    for key in ("rJ_present", "t_overflow"):
        vals = {c: v[key] / v["pairs"] for c, v in rates.items() if v["pairs"]}
        if {"US", "India", "France"} <= set(vals):
            lo, hi = min(vals["US"], vals["India"]), max(vals["US"], vals["India"])
            if vals["France"] > max(1.5 * hi, hi + 0.05) or vals["France"] < min(lo / 1.5, lo - 0.05):
                reasons.append(f"{key} {vals['France']:.4f} outside the US/India envelope [{lo:.4f}, {hi:.4f}]")
    fr = per_country.get("France")
    if fr and not (3.1 <= fr["mean_predicted"] <= 3.7 and 0.03 <= fr["empty"] <= 0.09 and fr["eight_plus"] <= 0.02):
        reasons.append(f"France output outside its bands: {fr}")
    return bool(reasons), {"rates": rates, "reasons": reasons}


def label_free(matching: Path, q: pl.DataFrame) -> dict:
    match = read_tsv(matching).join(q, left_on="source1_entity_id", right_on="entity_id")
    n = pl.col("matched_entity_ids").fill_null("").str.split(",").list.eval(pl.element().filter(pl.element() != "")).list.len()
    rows = match.with_columns(n.alias("k")).group_by("country").agg(
        pl.len().alias("s1"), pl.col("k").mean().alias("mean_predicted"), (pl.col("k") == 0).mean().alias("empty"),
        (pl.col("k") >= 8).mean().alias("eight_plus")).to_dicts()
    return {r["country"]: r for r in rows}


def official_validator(validator: Path, output: Path, data: Path):
    run = subprocess.run([sys.executable, str(validator), "--matching", str(output / "matching_results.tsv"),
                          "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", str(data / "test"),
                          "--check-ids"], capture_output=True, text=True)
    return run.returncode, run.stdout[-4000:] + run.stderr[-2000:]


def run_collect(args):
    from src.v5_dist import Context, Store, run_final
    release_path = Path(args.release)
    release, _ = load_release(release_path)
    release_sha = sha256(release_path)
    assignment = json.loads(Path(args.assignment).read_text())
    check_assignment(assignment, release_sha)
    data = Path(args.data)
    validator = data.parent / "utils" / "validate_submission.py"
    if not validator.exists():
        raise SystemExit(f"official validator not found at {validator}")
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
    out = Path(args.out)
    stage = out / "stage"
    (stage / "score").mkdir(parents=True, exist_ok=True)
    seen, pairs = set(), []
    for shard, (folder, meta) in sorted(found.items()):
        path = Path(folder) / f"{shard}.parquet"
        if meta["release_sha256"] != release_sha or sha256(path) != meta["output_sha256"]:
            raise SystemExit(f"{shard}: release or output checksum mismatch")
        if meta["rows"] != meta["input_rows"] or meta["rows"] != assignment["shard_rows"][shard]:
            raise SystemExit(f"{shard}: row count differs from its ens3 input")
        frame = pl.read_parquet(path)
        if pair_digest(frame["s1"], frame["t"], "test") != meta["pair_digest"]:
            raise SystemExit(f"{shard}: pair digest mismatch")
        if frame["s"].is_null().any() or not np.isfinite(frame["s"].to_numpy()).all():
            raise SystemExit(f"{shard}: missing or non-finite scores")
        if frame.select("s1", "t").is_duplicated().any() or frame.select("t", "target_id").unique().height != frame["t"].n_unique():
            raise SystemExit(f"{shard}: duplicate pairs or a target row with two target IDs")
        s1 = set(frame["s1"].unique().to_list())
        if s1 & seen:
            raise SystemExit(f"{shard}: S1 rows also present in another shard")
        seen |= s1
        pairs.append(frame.select("s1", "t", "target_id"))
        frame.select("s1", "t", "target_id", pl.col("s").alias("p")).write_parquet(stage / "score" / f"{shard}.parquet")
    pins = {json.dumps({k: m["code"].get(k) for k in ("git_head", "stage2_release.py", "v6_stage2.py", "v7_population.py")}
                       | {"index": m.get("index_digest"), "families": m.get("families")}, sort_keys=True)
            for _, m in found.values()}
    if len(pins) != 1:
        raise SystemExit(f"shards were produced by different code, index or families: {pins}")
    allpairs = pl.concat(pairs)
    q = read_tsv(data / "test" / "test_source1.tsv", columns=["entity_id", "country"])
    if args.expect_full and (len(q) != EXPECTED["test_s1"] or len(allpairs) != EXPECTED["test_pairs"]
                             or len(q) - len(seen) != EXPECTED["test_zero_survivor_s1"]):
        raise SystemExit(f"invariants: {len(q)} S1, {len(allpairs)} pairs, {len(q) - len(seen)} zero-survivor S1")
    cross = (allpairs.join(pl.DataFrame({"s1": np.arange(len(q), dtype=np.uint32), "country": q["country"]}), on="s1")
             .group_by("target_id").agg(pl.col("country").n_unique().alias("n")).filter(pl.col("n") > 1))
    if len(cross):
        raise SystemExit(f"{len(cross)} targets are candidates of S1 in two countries; per-country ownership is not exact")
    Store(root=stage).put_json("model/model_manifest.json", {"threshold": release["selection"]["threshold"],
                                                             "model_sha256": release_sha})
    # v5 final assembly: global one owner per target, threshold, every test S1 in both TSVs, validator.
    report = run_final({"inputs": sorted(found)}, Store(root=stage), Context(str(data), out, 1, {}))
    output = out / "output"
    if args.v6_candidates:
        if sha256(output / "candidate_pairs.tsv") != sha256(args.v6_candidates):
            raise SystemExit("candidate_pairs.tsv differs from V6's file: the pair set changed")
        report["candidate_pairs_is_v6_bytes"] = True
    per_country = label_free(output / "matching_results.tsv", q)
    fallback, envelope = france_out_of_envelope([m for _, m in found.values()], per_country)
    report["france_envelope"] = envelope
    if fallback:
        if not args.v6_matching:
            raise SystemExit("France is out of envelope and --v6-matching was not given")
        v6 = read_tsv(Path(args.v6_matching)).join(q, left_on="source1_entity_id", right_on="entity_id", how="left",
                                                   maintain_order="left")
        mine = read_tsv(output / "matching_results.tsv")
        if not mine["source1_entity_id"].equals(v6["source1_entity_id"]):
            raise SystemExit("V6 matching rows are not in the same S1 order")
        france = (v6["country"] == "France").to_numpy()
        merged = np.where(france, v6["matched_entity_ids"].fill_null("").to_numpy(),
                          mine["matched_entity_ids"].fill_null("").to_numpy())
        mine.with_columns(pl.Series("matched_entity_ids", merged)).write_csv(
            output / "matching_results.tsv", separator="\t", quote_style="never")
        report["validator_exit"], report["validator_output"] = official_validator(validator, output, data)
        per_country = label_free(output / "matching_results.tsv", q)
    report["france_routed_to_v6"] = fallback
    if report.get("validator_exit") != 0:
        raise SystemExit(f"official validator exit {report.get('validator_exit')}: {report.get('validator_output', '')[-800:]}")
    strict = subprocess.run([sys.executable, "-m", "src.strict_validate", "--matching", str(output / "matching_results.tsv"),
                             "--candidate", str(output / "candidate_pairs.tsv"), "--test-dir", str(data / "test"),
                             "--expected-s1", str(len(q)), "--target-index", "memory", "--quiet"],
                            cwd=PACKAGE, capture_output=True, text=True)
    report["strict_validator_exit"] = strict.returncode
    if strict.returncode != 0:
        raise SystemExit(f"strict validator failed: {strict.stdout[-2000:]}{strict.stderr[-2000:]}")
    report.update({"label_free_by_country": per_country, "shards": len(found), "release_sha256": release_sha,
                   "tsv_sha256": {name: sha256(output / name) for name in ("matching_results.tsv", "candidate_pairs.tsv")},
                   "pair_digest": pair_digest(allpairs["s1"], allpairs["t"], "test")})
    (out / "collect_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "validator_output"}, indent=1, default=str), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    threads = int(os.environ.get("ER_V6_THREADS", os.cpu_count() or 4))
    g = sub.add_parser("gate")
    for name in ("--ens3", "--members", "--data", "--out"):
        g.add_argument(name, required=True)
    g.add_argument("--expect-f", type=float, default=0.97061, help="G0: V6 tune F at its threshold (control); 0 to skip")
    g.add_argument("--families", default="F0F1", help="rungs to run, in order: F0F1 (B1), F0F1F3 (B2)")
    a = sub.add_parser("assign")
    for name in ("--scores", "--out", "--benchmark"):
        a.add_argument(name, required=True)
    a.add_argument("--release")
    a.add_argument("--rates")
    a.add_argument("--expect-shards", type=int, default=88)
    f = sub.add_parser("features")
    for name in ("--split", "--scores", "--data", "--out"):
        f.add_argument(name, required=True)
    f.add_argument("--shards")
    f.add_argument("--assignment")
    f.add_argument("--worker")
    f.add_argument("--n", type=int, default=12, help="ens3 cut n (checked on every shard)")
    f.add_argument("--floor", type=float, default=0.01, help="ens3 cut floor (checked on every shard)")
    p = sub.add_parser("predict")
    for name in ("--release", "--assignment", "--worker", "--scores", "--data", "--out"):
        p.add_argument(name, required=True)
    p.add_argument("--shards")
    p.add_argument("--cache")
    c = sub.add_parser("collect")
    for name in ("--release", "--assignment", "--inputs", "--data", "--out"):
        c.add_argument(name, required=True)
    c.add_argument("--v6-matching")
    c.add_argument("--v6-candidates")
    c.add_argument("--expect-full", action="store_true", help="enforce the full-test invariants of plan section 2")
    for s in (g, f, p):
        s.add_argument("--threads", type=int, default=threads)
    args = parser.parse_args(argv)
    steps = {"gate": run_gate, "assign": run_assign, "features": run_features, "predict": run_predict, "collect": run_collect}
    return steps[args.step](args) or 0


if __name__ == "__main__":
    sys.exit(main())
