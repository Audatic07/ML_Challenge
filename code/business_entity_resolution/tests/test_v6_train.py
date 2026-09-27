import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

from test_v5 import make_split

ROOT = Path(__file__).resolve().parents[1]


def test_cut_matches_stage2_survivors():
    from src.v6_stage2 import survivors
    from src.v6_train import cut
    rng = np.random.default_rng(5)
    n = 4000
    frame = pl.DataFrame({"s1": rng.integers(0, 300, n).astype(np.uint32),
                          "t": rng.permutation(n).astype(np.uint32),
                          # coarse values force ties, which must break by target row
                          "p1": (rng.integers(0, 40, n) / 40).astype(np.float32)})
    for k, floor in ((1, 0.0), (5, 0.0), (5, 0.3), (12, 0.001)):
        mine = cut(frame, k, floor).select("s1", "t")
        ref = survivors(frame.rename({"p1": "p"}), k, floor).select("s1", "t")
        assert mine.equals(ref), (k, floor)
    assert len(cut(frame.head(0), 3)) == 0


def run(args, **kw):
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True, **kw)


def test_v6_queue_end_to_end(tmp_path):
    rng = random.Random(11)
    channels = tmp_path / "channels.json"
    channels.write_text(json.dumps({
        "address_char": {"top_k": 30}, "name_fold": {"top_k": 10},
        "combo": {"view": "combo", "analyzer": "char_wb", "ngram": [3, 4], "max_df": 0.3, "top_k": 30},
        "combo_word": {"view": "combo", "analyzer": "word", "ngram": [1, 2], "max_df": 0.3, "top_k": 30}}))
    data = tmp_path / "student_resource" / "dataset"
    make_split(data, "train", ["US", "India"], 400, rng)
    make_split(data, "test", ["US", "India", "France"], 150, rng)
    validator = Path(os.environ.get("ER_VALIDATOR", ROOT.parents[1] / "student_resource" / "utils" / "validate_submission.py"))
    if validator.exists():
        (data.parent / "utils").mkdir(parents=True)
        (data.parent / "utils" / "validate_submission.py").write_bytes(validator.read_bytes())
    store = tmp_path / "store"

    def worker(prefix, name):
        run(["src.v5_dist", "worker", "--data", str(data), "--root", str(store), "--prefix", prefix,
             "--work", str(tmp_path / f"w-{name}"), "--name", name, "--workers", "2", "--mem-gb", "64"])
        failed = store / prefix / "failed"
        assert not failed.exists(), next(failed.glob("*.json")).read_text()

    # the v5.1-style source run: features, stage-1 model, full-union test scores
    run(["src.v5_dist", "plan", "--data", str(data), "--root", str(store), "--prefix", "v51", "--shard", "60",
         "--norm", "v5", "--anchors", "--channels-json", str(channels), "--neg-keep", "0.5"])
    worker("v51", "src")
    # v6 on its survivors: publish, run, then republish the test scoring for another cut
    run(["src.v6_train", "plan", "--root", str(store), "--source", "v51", "--prefix", "v6", "--train-n", "6",
         "--n", "4", "--floor", "0.001", "--rounds", "60", "--patience", "20", "--leaves", "15", "--min-leaf", "5"])
    plan = json.loads((store / "v6" / "plan.json").read_text())
    tasks = {p.stem for p in (store / "v6" / "tasks").glob("*.json")}
    assert plan["task_count"] == len(tasks) and "v6train" in tasks and "final" in tasks
    worker("v6", "v6")
    assert {p.stem for p in (store / "v6" / "done").glob("*.json")} == tasks

    manifest = json.loads((store / "v6" / "model" / "model_manifest.json").read_text())
    source = json.loads((store / "v51" / "model" / "model_manifest.json").read_text())
    assert manifest["features"] == source["features"]
    assert manifest["cascade"] == {"stage1": "model", "n": 4, "floor": 0.001}
    metrics = manifest["metrics"]
    assert metrics["audit_opened"] is False and metrics["tune_queries"] > 0
    for key, cut in metrics["cuts"].items():
        assert 0 <= cut["v6"]["macro_f05"] <= cut["oracle_u"] <= 1, key
        assert 0 <= cut["v51"]["macro_f05"] <= cut["oracle_u"], key
        assert cut["pairs_per_query"] <= cut["n"]

    # survivors keep the source model's own probabilities, and at most n per S1
    surv = pl.concat([pl.read_parquet(p) for p in (store / "v6" / "surv").glob("*.parquet")])
    assert surv.group_by("s1").len()["len"].max() <= 6
    tune = pl.read_parquet(store / "v51" / "model" / "tune_scores.parquet")
    joined = surv.join(tune, on=["s1", "t"])
    assert len(joined) and np.allclose(joined["p1"], joined["p"], atol=1e-6)
    scored = pl.concat([pl.read_parquet(p) for p in (store / "v6" / "score").glob("*.parquet")])
    first = pl.concat([pl.read_parquet(p) for p in (store / "v51" / "score").glob("*.parquet")])
    kept = first.sort(["s1", "p", "t"], descending=[False, True, False]).group_by("s1", maintain_order=True).head(4)
    kept = kept.filter(pl.col("p") >= 0.001)
    assert scored.select("s1", "t").sort("s1", "t").equals(kept.select("s1", "t").sort("s1", "t"))

    def check_final(prefix, n):
        final = store / prefix / "final"
        matching = pl.read_csv(final / "matching_results.tsv", separator="\t", infer_schema=False)
        cands = pl.read_csv(final / "candidate_pairs.tsv", separator="\t", infer_schema=False)
        assert len(matching) == len(cands) == 150
        assert matching["source1_entity_id"].equals(cands["source1_entity_id"])
        sizes = cands["candidate_entity_ids"].fill_null("").str.split(",").list.eval(
            pl.element().filter(pl.element() != "")).list.len()
        assert sizes.max() <= n
        for m, c in zip(matching["matched_entity_ids"].fill_null(""), cands["candidate_entity_ids"].fill_null("")):
            assert set(filter(None, m.split(","))) <= set(filter(None, c.split(",")))
        report = json.loads((final / "report.json").read_text())
        assert report["candidate_pairs"] == int(sizes.sum())
        if validator.exists():
            assert report["validator_exit"] == 0, report["validator_output"]

    check_final("v6", 4)
    run(["src.v6_train", "apply", "--root", str(store), "--source", "v6", "--prefix", "v6b", "--n", "3",
         "--floor", "0", "--threshold", "0.5"])
    worker("v6b", "v6b")
    assert json.loads((store / "v6b" / "model" / "model_manifest.json").read_text())["threshold"] == 0.5
    check_final("v6b", 3)

    # a seed variant trained on the same survivors, for ensembles
    run(["src.v6_train", "variant", "--root", str(store), "--source", "v6", "--prefix", "v6v", "--seed", "7",
         "--rounds", "60", "--patience", "20", "--leaves", "15", "--min-leaf", "5"])
    worker("v6v", "v6v")
    variant = json.loads((store / "v6v" / "model" / "model_manifest.json").read_text())
    assert variant["params"]["seed"] == 7 and not (store / "v6v" / "surv").exists()
    a = pl.read_parquet(store / "v6" / "model" / "tune_scores.parquet").sort("s1", "t")
    b = pl.read_parquet(store / "v6v" / "model" / "tune_scores.parquet").sort("s1", "t")
    assert a.select("s1", "t", "p1").equals(b.select("s1", "t", "p1"))
    check_final("v6v", 4)

    # locked-audit features, then the one-time audit evaluation of a two-model champion
    subprocess.run([sys.executable, "tools/audit_feat_plan.py", "--root", str(store), "--source", "v51", "--prefix", "aud",
                    "--data", str(data)], cwd=ROOT, check=True)
    worker("aud", "aud")
    audit_task = {"id": "v6audit-x", "kind": "v6audit", "priority": 0, "min_mem_gb": 1, "audit_prefix": "aud",
                  "models": ["v6", "v6v"], "cut": {"n": 4, "floor": 0.001},
                  "thresholds": {"champion": 0.5, "v51_same_cut": 0.5, "v51_full": 0.5, "release_cosine30": 0.5}}
    (store / "v6" / "tasks" / "v6audit-x.json").write_text(json.dumps(audit_task))
    worker("v6", "v6-audit")
    audit = json.loads((store / "v6" / "audit" / "v6audit-x.json").read_text())
    assert audit["audit_queries"] > 0 and len(audit["models"]) == 2
    for key in ("champion", "v51_same_cut", "v51_full_union", "release_cosine30"):
        assert 0 <= audit[key]["macro_f05"] <= audit[key]["oracle_u"] <= 1, key
    assert audit["champion"]["pairs_per_query"] <= 4 < audit["v51_full_union"]["pairs_per_query"]

    # ensemble of the two finished queues: averaged survivor scores, then the standard final
    run(["src.v6_train", "ensemble", "--root", str(store), "--source", "v6,v6v", "--prefix", "ens", "--threshold", "0.5"])
    worker("ens", "ens")
    a = pl.read_parquet(store / "v6" / "score" / "v6score-test-US-000.parquet")
    b = pl.read_parquet(store / "v6v" / "score" / "v6score-test-US-000.parquet")
    e = pl.read_parquet(store / "ens" / "score" / "v6score-test-US-000.parquet")
    assert np.allclose(e["p"], (a["p"] + b["p"]) / 2, atol=1e-6)
    check_final("ens", 4)
    # the same average with a tighter stage-1 floor keeps exactly the survivors above it
    run(["src.v6_train", "ensemble", "--root", str(store), "--source", "v6,v6v", "--prefix", "ens2", "--threshold", "0.5",
         "--floor", "0.3"])
    worker("ens2", "ens2")
    e2 = pl.read_parquet(store / "ens2" / "score" / "v6score-test-US-000.parquet")
    assert e2.equals(e.filter(pl.col("p1") >= 0.3))
    assert json.loads((store / "ens2" / "model" / "model_manifest.json").read_text())["cascade"]["floor"] == 0.3
    check_final("ens2", 4)
