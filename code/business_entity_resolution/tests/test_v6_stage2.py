import numpy as np
import polars as pl

from src.v6_stage2 import SIBLINGS, survivors, features, text_views


def records(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


def test_survivors_top_k_and_floor():
    scored = pl.DataFrame({"s1": [0, 0, 0, 0, 1, 1], "t": [5, 6, 7, 8, 5, 9],
                           "target_id": list("abcdef"), "p": [0.9, 0.2, 0.0005, 0.95, 0.0001, 0.0002]})
    top = survivors(scored, 2)
    assert top.filter(pl.col("s1") == 0)["t"].to_list() == [8, 5]
    assert top["rank"].to_list() == [0, 1, 0, 1]
    floored = survivors(scored, 3, floor=0.001)
    assert floored.filter(pl.col("s1") == 0)["t"].to_list() == [8, 5, 6]
    assert floored.filter(pl.col("s1") == 1).height == 0  # every candidate below the floor


def test_features_use_confident_siblings():
    s1 = text_views(records([("S1-1", "Rana Better Engineering Inc", "3445 Refugee Road, Columbus, OH", "US")]))
    targets = records([
        ("S2-1", "Rana Better Engineering Inc", "3445 Refugee Rd, Columbus, Ohio", "US"),   # confident sibling
        ("S3-1", "Pyrawex", "3445 Refugee Rd, Columbus, Ohio", "US"),                       # renamed variant
        ("S3-2", "Rana Better Engineering", "", "US"),                                      # no address
        ("S2-2", "Quinta Bakery", "12 Elm St, Dayton, OH", "US")])                          # unrelated
    t_text = text_views(targets).with_columns(pl.Series("src", [2, 3, 3, 2], dtype=pl.UInt8))
    scored = pl.DataFrame({"s1": [0, 0, 0, 0], "t": [0, 1, 2, 3], "target_id": targets["entity_id"],
                           "p": [0.99, 0.6, 0.5, 0.01]})
    feat = features(survivors(scored, 10), s1, t_text, workers=1)
    assert feat.height == 4 and feat["t"].to_list() == [0, 1, 2, 3]
    row = {r["target_id"]: r for r in feat.iter_rows(named=True)}
    assert row["S3-1"]["sibc_a_eq"] == 1 and row["S3-1"]["q_n_shared"] == 0
    assert row["S3-2"]["q_a_tset"] == -1 and row["S3-2"]["sibc_n_tset"] >= 90
    assert row["S2-2"]["sibc_a_eq"] == 0
    assert row["S2-1"]["sib1_p"] == np.float32(0.6)  # the best other survivor, never itself
    assert row["S2-1"]["others_p90"] == 0 and row["S3-1"]["others_p90"] == 1
    assert SIBLINGS == 5


def _scores(rng, n_s1, n_targets, truth, first=0):
    """Synthetic ens3-style survivors: each S1's true targets plus random others, p1 >= 0.01."""
    rows = []
    for s in range(first, first + n_s1):
        for t in sorted(set(truth.get(s, [])) | set(rng.sample(range(n_targets), 6))):
            y = t in truth.get(s, [])
            rows.append((s, t, y, min(0.999, 0.011 + 0.5 * y + 0.49 * rng.random())))
    return rows


def test_release_gate_predict_collect(tmp_path):
    import json
    import os
    import random
    import shutil
    import sys
    from pathlib import Path

    import pytest

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "v6_analysis"))
    import stage2_release as sr
    from test_v5 import make_split

    rng = random.Random(7)
    data = tmp_path / "student_resource" / "dataset"
    make_split(data, "train", ["US", "India"], 240, rng)
    make_split(data, "test", ["US", "India", "France"], 90, rng)
    if os.environ.get("ER_VALIDATOR"):
        (data.parent / "utils").mkdir()
        shutil.copyfile(os.environ["ER_VALIDATOR"], data.parent / "utils" / "validate_submission.py")

    def table(split, name):
        return pl.read_csv(data / split / f"{split}_{name}.tsv", separator="\t", quote_char=None, infer_schema=False)

    s1 = table("train", "source1")
    targets = pl.concat([table("train", "source2"), table("train", "source3")])["entity_id"].to_list()
    row = {e: i for i, e in enumerate(targets)}
    gt = dict(table("train", "ground_truth").iter_rows())
    truth = {i: [row[x] for x in (gt.get(e) or "").split(",") if x] for i, e in enumerate(s1["entity_id"])}

    ens3, members = tmp_path / "ens3", tmp_path / "members"
    (ens3 / "model").mkdir(parents=True)
    (ens3 / "plan.json").write_text(json.dumps({"tune_rows": list(range(len(s1))), "tune_country": s1["country"].to_list()}))
    (ens3 / "model" / "model_manifest.json").write_text(json.dumps({
        "version": "v6-survivor-1-mean", "members": [[f"shared/m{i}", f"sha{i}"] for i in range(3)], "model_sha256": "ens",
        "cascade": {"stage1": "model", "n": 12, "floor": 0.01}, "threshold": 0.72}))
    base = _scores(rng, len(s1), len(targets), truth)
    for i in range(3):
        folder = members / f"m{i}" / "model"
        folder.mkdir(parents=True)
        (folder / "model_manifest.json").write_text(json.dumps({"model_sha256": f"sha{i}"}))
        pl.DataFrame({"s1": pl.Series([r[0] for r in base], dtype=pl.UInt32), "t": pl.Series([r[1] for r in base], dtype=pl.UInt32),
                      "target_id": [targets[r[1]] for r in base], "p1": pl.Series([r[3] for r in base], dtype=pl.Float32),
                      "p": pl.Series([min(0.999, 0.3 * r[2] + 0.7 * rng.random()) for r in base], dtype=pl.Float32)}
                     ).write_parquet(folder / "tune_scores.parquet")

    release = tmp_path / "release"
    assert sr.main(["gate", "--ens3", str(ens3), "--members", str(members), "--data", str(data), "--out", str(release),
                    "--expect-f", "0", "--z", "-1e9", "--threads", "2"]) == 0  # forced pass: synthetic data
    gate = json.loads((release / "gate.json").read_text())
    rel = json.loads((release / "release.json").read_text())
    assert gate["survivor_rows"] == len(base) and gate["audit_opened"] is False
    assert rel["candidate_policy"]["n"] == 12 and len(rel["models"]) == 2 and rel["features"][0] == "p1"

    test_targets = pl.concat([table("test", "source2"), table("test", "source3")])["entity_id"].to_list()
    scores = tmp_path / "scores"
    scores.mkdir()
    shards = {"v6score-test-A-000": (0, 30), "v6score-test-A-001": (30, 30), "v6score-test-A-002": (60, 29)}  # S1 89: no survivors
    for name, (first, count) in shards.items():
        rows = _scores(rng, count, len(test_targets), {}, first)
        pl.DataFrame({"s1": pl.Series([r[0] for r in rows], dtype=pl.UInt32), "t": pl.Series([r[1] for r in rows], dtype=pl.UInt32),
                      "target_id": [test_targets[r[1]] for r in rows], "p1": pl.Series([r[3] for r in rows], dtype=pl.Float32),
                      "p": pl.Series([rng.random() for _ in rows], dtype=pl.Float32)}).write_parquet(scores / f"{name}.parquet")

    common = ["--release", str(release / "release.json"), "--scores", str(scores)]
    bench = "abhigyan-r7i4xl=v6score-test-A-000,akash-r5xl=v6score-test-A-001"
    sr.main(["assign", *common, "--out", str(tmp_path / "a0.json"), "--benchmark", bench, "--expect-shards", "3"])
    out = {w: tmp_path / w for w in ("abhigyan-r7i4xl", "akash-r5xl")}
    run = ["--release", str(release / "release.json"), "--scores", str(scores), "--data", str(data)]
    for worker in out:
        sr.main(["predict", *run, "--assignment", str(tmp_path / "a0.json"), "--worker", worker, "--out", str(out[worker])])
    with pytest.raises(SystemExit):  # a worker without an assignment processes nothing
        sr.main(["predict", *run, "--assignment", str(tmp_path / "a0.json"), "--worker", "aditya-worker", "--out", str(tmp_path / "x")])
    rates = ",".join(f"{w}={out[w] / (s + '.json')}" for w, s in (x.split("=") for x in bench.split(",")))
    sr.main(["assign", *common, "--out", str(tmp_path / "a1.json"), "--benchmark", bench, "--rates", rates, "--expect-shards", "3"])
    a1 = json.loads((tmp_path / "a1.json").read_text())
    assert sorted(s for v in a1["workers"].values() for s in v) == sorted(shards)
    for worker in out:
        sr.main(["predict", *run, "--assignment", str(tmp_path / "a1.json"), "--worker", worker, "--out", str(out[worker])])

    final = tmp_path / "final"
    sr.main(["collect", "--release", str(release / "release.json"), "--assignment", str(tmp_path / "a1.json"),
             "--inputs", ",".join(str(p) for p in out.values()), "--data", str(data), "--out", str(final)])
    match = pl.read_csv(final / "output" / "matching_results.tsv", separator="\t", infer_schema=False)
    cand = pl.read_csv(final / "output" / "candidate_pairs.tsv", separator="\t", infer_schema=False)
    assert match.height == cand.height == 90 and match["source1_entity_id"].n_unique() == 90
    last = table("test", "source1")["entity_id"][89]
    assert match.filter(pl.col("source1_entity_id") == last)["matched_entity_ids"].fill_null("").item() == ""
    assert cand.filter(pl.col("source1_entity_id") == last)["candidate_entity_ids"].fill_null("").item() == ""
    for m, c in zip(match["matched_entity_ids"].fill_null(""), cand["candidate_entity_ids"].fill_null("")):
        assert set(filter(None, m.split(","))) <= set(filter(None, c.split(",")))
    report = json.loads((final / "collect_report.json").read_text())
    assert report["shards"] == 3 and report.get("validator_exit") in (0, None)

    dup = tmp_path / "dup"  # the same shard delivered by two workers must be refused
    dup.mkdir()
    for f in out["akash-r5xl"].glob("v6score-test-A-000.*") or out["abhigyan-r7i4xl"].glob("v6score-test-A-000.*"):
        shutil.copy(f, dup)
    for f in out["abhigyan-r7i4xl"].glob("v6score-test-A-000.*"):
        shutil.copy(f, dup)
    with pytest.raises(SystemExit):
        sr.main(["collect", "--release", str(release / "release.json"), "--assignment", str(tmp_path / "a1.json"),
                 "--inputs", ",".join([*(str(p) for p in out.values()), str(dup)]), "--data", str(data), "--out", str(tmp_path / "f2")])
