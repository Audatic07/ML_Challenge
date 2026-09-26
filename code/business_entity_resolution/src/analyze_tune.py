"""Read-only v5 tune diagnostics. No training, inference, S3 writes or job control.

Inputs: local immutable tune_scores.parquet, tune_per_query.parquet,
model_manifest.json; supplied train TSVs. Only tune labels are retained.
Outputs contain private examples: use --out outside Git; publish aggregates only.
Requires numpy and polars (already used by v5). No v5 module is imported.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import polars as pl


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def f05(g, tp, fp):
    return float(fp == 0) if g == 0 else 5 * tp / (4 * (tp + fp) + g)


def loss_parts(g, retrieved, tp, fp):
    """Ordered counterfactual split: remove FPs, then recover retrieved FNs."""
    if not (0 <= tp <= retrieved <= g and fp >= 0):
        raise ValueError("Inconsistent truth/retrieval/prediction counts")
    score = f05(g, tp, fp)
    no_fp = f05(g, tp, 0)
    oracle = f05(g, retrieved, 0)
    return {"f": score, "oracle": oracle, "retrieval_loss": 1 - oracle,
            "false_positive_loss": no_fp - score,
            "missed_retrieved_loss": oracle - no_fp}


def score_expr(tp, fp):
    return (pl.when(pl.col("g") == 0).then((fp == 0).cast(pl.Float64))
            .otherwise(5 * tp / (4 * (tp + fp) + pl.col("g"))))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def analyze(per, scores, truth, threshold):
    """Return per-query diagnostics and tagged pairs; validate against saved F/U.

    v5 run_train saves g,p,tp from the *oracle* and only f from actual predictions.
    Deliberately ignore saved p,tp. Recompute assignment over all tune queries.
    """
    require(math.isfinite(threshold) and 0 <= threshold <= 1, "Invalid threshold")
    require(len(per) > 0 and per["s1"].n_unique() == len(per), "Empty/duplicate per-query index")
    require(not per.select(pl.any_horizontal(pl.all().is_null()).any()).item(), "Null per-query values")
    require(truth.select("s1", "target_id").n_unique() == len(truth), "Duplicate truth pair")
    require(scores.select("s1", "t").n_unique() == len(scores), "Duplicate score pair")
    require(scores.select("s1", "target_id").n_unique() == len(scores), "Duplicate target ID per query")
    require(not scores.select(pl.any_horizontal(pl.all().is_null()).any()).item(), "Null score values")
    require(scores.filter(~pl.col("p").is_finite() | (pl.col("p") < 0) | (pl.col("p") > 1)).height == 0,
            "Scores must be finite probabilities")
    q = per.select("s1")
    require(scores.join(q, on="s1", how="anti").height == 0, "Scores outside tune index")
    require(truth.join(q, on="s1", how="anti").height == 0, "Truth outside tune index")
    mappings = scores.select("t", "target_id").unique()
    require(mappings["t"].n_unique() == len(mappings) and mappings["target_id"].n_unique() == len(mappings),
            "Target row/ID map is not one-to-one")
    owners = (scores.sort(["t", "p", "s1"], descending=[False, True, False])
              .unique(subset=["t"], keep="first", maintain_order=True)
              .select("s1", "t").with_columns(pl.lit(True).alias("owner")))
    tagged = (scores.join(truth.with_columns(pl.lit(True).alias("is_truth")), on=["s1", "target_id"], how="left")
              .join(owners, on=["s1", "t"], how="left")
              .with_columns(pl.col("is_truth", "owner").fill_null(False))
              .with_columns(((pl.col("p") >= threshold) & pl.col("owner")).alias("pred")))
    counts = tagged.group_by("s1").agg(
        pl.len().alias("candidate_count"), pl.col("is_truth").sum().alias("retrieved"),
        (pl.col("pred") & pl.col("is_truth")).sum().alias("tp"),
        (pl.col("pred") & ~pl.col("is_truth")).sum().alias("fp"),
        (pl.col("is_truth") & ~pl.col("owner") & (pl.col("p") >= threshold)).sum().alias("owner_lost"),
        (pl.col("is_truth") & (pl.col("p") < threshold)).sum().alias("below_threshold"))
    truth_counts = truth.group_by("s1").len().rename({"len": "g"})
    detail = (per.select("s1", "country", pl.col("g").alias("saved_g"),
                         pl.col("f").alias("saved_f"), pl.col("oracle").alias("saved_oracle"))
              .join(truth_counts, on="s1", how="left").join(counts, on="s1", how="left"))
    numeric = ["g", "candidate_count", "retrieved", "tp", "fp", "owner_lost", "below_threshold"]
    detail = detail.with_columns(pl.col(numeric).fill_null(0).cast(pl.Int64))
    require(detail.filter(pl.col("g") != pl.col("saved_g")).height == 0, "Truth counts disagree with saved per-query data")
    detail = detail.with_columns(
        score_expr(pl.col("tp"), pl.col("fp")).alias("f"),
        score_expr(pl.col("tp"), pl.lit(0)).alias("without_fp"),
        score_expr(pl.col("retrieved"), pl.lit(0)).alias("oracle"),
        (pl.col("g") - pl.col("retrieved")).alias("retrieval_misses"),
        (pl.col("retrieved") - pl.col("tp")).alias("missed_retrieved"))
    require(detail.filter((pl.col("retrieval_misses") < 0) | (pl.col("missed_retrieved") < 0)).height == 0,
            "Invalid pair counts")
    for actual, saved in (("f", "saved_f"), ("oracle", "saved_oracle")):
        require(detail.filter(~pl.col(saved).is_finite() | ((pl.col(actual) - pl.col(saved)).abs() > 1e-7)).height == 0,
                f"Recomputed {actual} disagrees with saved values; check artifact versions/threshold/row map")
    detail = detail.with_columns(
        (1 - pl.col("f")).alias("loss"), (1 - pl.col("oracle")).alias("retrieval_loss"),
        (pl.col("without_fp") - pl.col("f")).alias("false_positive_loss"),
        (pl.col("oracle") - pl.col("without_fp")).alias("missed_retrieved_loss"))
    require(detail.filter((pl.col("loss") - pl.sum_horizontal("retrieval_loss", "false_positive_loss", "missed_retrieved_loss")).abs() > 1e-10).height == 0,
            "Loss components do not reconcile")
    missed = truth.join(tagged.filter(pl.col("is_truth")).select("s1", "target_id"),
                        on=["s1", "target_id"], how="anti")
    return detail.sort("s1"), tagged, missed


def summarize(detail):
    n = len(detail)
    def expressions():
        return [pl.len().alias("queries"), pl.col("f").mean().alias("macro_f05"),
                pl.col("oracle").mean().alias("oracle_u"),
                *[pl.col(c).mean().alias(c) for c in ("loss", "retrieval_loss", "false_positive_loss", "missed_retrieved_loss")],
                *[(pl.col(c).sum() / n).alias(c + "_contribution") for c in ("loss", "retrieval_loss", "false_positive_loss", "missed_retrieved_loss")],
                *[pl.col(c).sum().alias(c) for c in ("g", "candidate_count", "tp", "fp", "retrieval_misses", "missed_retrieved", "owner_lost", "below_threshold")],
                (pl.col("candidate_count") == 0).sum().alias("zero_candidate_queries"),
                ((pl.col("g") == 0) & (pl.col("fp") > 0)).sum().alias("singleton_false_merges")]
    return {"overall": detail.select(expressions()).to_dicts()[0],
            "by_country": detail.group_by("country").agg(expressions()).sort("country").to_dicts(),
            "by_truth_count": detail.group_by("g").agg([e for e in expressions() if e.meta.output_name() != "g"]).sort("g").to_dicts(),
            "by_country_truth_count": detail.group_by("country", "g").agg([e for e in expressions() if e.meta.output_name() != "g"]).sort("country", "g").to_dicts()}


def load_tune_truth(train_dir, per):
    """Stream full reference files, keeping only indexed tune queries and labels."""
    wanted = set(per["s1"].to_list())
    qmap, queries = {}, {}
    with (train_dir / "train_source1.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row_number, row in enumerate(reader):
            if row_number in wanted:
                require(row["entity_id"] not in qmap, "Duplicate tune query ID")
                qmap[row["entity_id"]] = row_number
                queries[row_number] = row
    require(len(queries) == len(wanted), "Tune row outside supplied train source1")
    seen, rows = set(), []
    with (train_dir / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
            query = row["source1_entity_id"]
            if query not in qmap:
                continue
            require(query not in seen, "Duplicate ground-truth query row")
            seen.add(query)
            target_ids = row["matched_entity_ids"].split(",") if row["matched_entity_ids"] else []
            for target in target_ids:
                require(target.startswith(("S2-", "S3-")), "Invalid truth target ID")
                rows.append((qmap[query], target))
    require(len(seen) == len(wanted), "Missing tune ground-truth rows (including singletons)")
    for row in per.select("s1", "country").iter_rows(named=True):
        require(queries[row["s1"]]["country"] == row["country"], "Country/row map differs from saved tune data")
    truth = pl.DataFrame(rows, schema={"s1": per.schema["s1"], "target_id": pl.String}, orient="row")
    return truth, queries


def choose_examples(detail, limit):
    """Deterministic, severity-ranked round-robin across mechanism/country/count."""
    pools = []
    for mechanism in ("retrieval_loss", "false_positive_loss", "missed_retrieved_loss"):
        for part in detail.filter(pl.col(mechanism) > 0).partition_by("country", "g", maintain_order=True):
            pools.append(part.sort([mechanism, "s1"], descending=[True, False])["s1"].to_list())
    picked, seen = [], set()
    depth = 0
    while len(picked) < limit:
        progressed = False
        for pool in pools:
            if depth < len(pool):
                progressed = True
                query = pool[depth]
                if query not in seen:
                    seen.add(query)
                    picked.append(query)
                    if len(picked) == limit:
                        break
        if not progressed:
            break
        depth += 1
    return picked


def write_examples(out, detail, tagged, missed, queries, train_dir, limit):
    selected = choose_examples(detail, limit)
    wanted = pl.Series(selected, dtype=detail.schema["s1"])
    relevant = tagged.filter(pl.col("s1").is_in(wanted.implode()) & (pl.col("pred") | pl.col("is_truth")))
    absent = missed.filter(pl.col("s1").is_in(wanted.implode()))
    target_ids = set(relevant["target_id"].to_list()) | set(absent["target_id"].to_list())
    target_records = {}
    for source in (2, 3):
        with (train_dir / f"train_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t", quoting=csv.QUOTE_NONE):
                if row["entity_id"] in target_ids:
                    require(row["entity_id"] not in target_records, "Duplicate example target catalog ID")
                    target_records[row["entity_id"]] = row
    require(target_ids == set(target_records), "Example target missing from supplied train catalogs")
    examples = []
    for query in selected:
        records = []
        for row in relevant.filter(pl.col("s1") == query).sort(["p", "target_id"], descending=[True, False]).iter_rows(named=True):
            kind = ("TP" if row["pred"] else "missed_but_retrieved") if row["is_truth"] else "FP"
            records.append({"kind": kind, "score": row["p"], "owner": row["owner"], **target_records[row["target_id"]]})
        for row in absent.filter(pl.col("s1") == query).sort("target_id").iter_rows(named=True):
            records.append({"kind": "retrieval_miss", **target_records[row["target_id"]]})
        examples.append({"query": queries[query], "diagnostics": detail.filter(pl.col("s1") == query).to_dicts()[0], "targets": records})
    (out / "examples.private.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(examples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New private local directory, outside Git")
    parser.add_argument("--plan", type=Path, help="Optional immutable v5 plan.json for exact tune membership check")
    parser.add_argument("--examples", type=int, default=20)
    args = parser.parse_args(argv)
    require(args.examples >= 0, "examples must be nonnegative")
    require(not any((p / ".git").exists() for p in (args.out.resolve(), *args.out.resolve().parents)),
            "Diagnostic outputs contain private records; choose a directory outside Git")
    inputs = [args.model_dir / name for name in ("tune_scores.parquet", "tune_per_query.parquet", "model_manifest.json")]
    for path in inputs:
        require(path.is_file(), f"Not ready: missing {path.name}")
    started = time.monotonic()
    initial = {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in inputs}
    manifest = json.loads(inputs[2].read_text(encoding="utf-8"))
    per = pl.read_parquet(inputs[1])
    require(len(per) == manifest["metrics"]["tune_queries"], "Tune query count differs from manifest")
    if args.plan:
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        require(plan["plan_sha"] == manifest["plan_sha"], "Plan/model mismatch")
        require(set(plan["tune_rows"]) == set(per["s1"].to_list()), "Tune membership differs from plan")
    truth, queries = load_tune_truth(args.train_dir, per)
    scores = pl.read_parquet(inputs[0], columns=["s1", "t", "target_id", "p"])
    detail, tagged, missed = analyze(per, scores, truth, float(manifest["threshold"]))
    report = summarize(detail)
    for field in ("macro_f05", "oracle_u"):
        require(abs(report["overall"][field] - manifest["metrics"][field]) < 1e-7, f"{field} differs from model manifest")
    report.update(threshold=manifest["threshold"], model_sha256=manifest.get("model_sha256"),
                  plan_sha=manifest.get("plan_sha"), audit_read=False,
                  loss_order="remove false positives first, then recover missed-but-retrieved truths; retrieval loss is 1-U",
                  caveat="FP/FN counterfactual attribution is order-dependent; this order is exactly additive.",
                  input_files={p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in inputs})
    if args.plan:
        report["input_files"]["plan.json"] = {"sha256": sha256(args.plan), "bytes": args.plan.stat().st_size}
    # Hash supplied files for exact provenance, without retaining records.
    report["source_files"] = {f"train_source{s}.tsv": {"sha256": sha256(args.train_dir / f"train_source{s}.tsv")} for s in (1, 2, 3)}
    report["source_files"]["train_ground_truth.tsv"] = {"sha256": sha256(args.train_dir / "train_ground_truth.tsv")}
    require(all((p.stat().st_size, p.stat().st_mtime_ns) == state for p, state in initial.items()), "Input changed during analysis")
    args.out.mkdir(parents=True, exist_ok=False)
    detail.write_parquet(args.out / "detail.private.parquet")
    report["example_count"] = write_examples(args.out, detail, tagged, missed, queries, args.train_dir, args.examples)
    report["seconds"] = time.monotonic() - started
    (args.out / "aggregate.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "example_count": report["example_count"], "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
