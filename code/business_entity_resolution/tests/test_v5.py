import json
import random
import subprocess
import sys
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
STREETS = ["Cedar Willow", "Shorebird", "Preston", "Portland", "Melrose", "Covington", "Dupre", "Riverside",
           "Green Valley", "Bidhan", "Jhilmil", "Mahavir", "Pitampura", "Goregaon", "Nesco", "Maruti"]
WORDS = ["Allied", "Textile", "Resorts", "Philbin", "Eddington", "Sprout", "Navkar", "Olva", "Gomez", "Flint",
         "Liberty", "Beacon", "Ingaborg", "Federal", "Dutton", "Gyan", "Bili", "United", "Coalition", "Nova"]
SUFFIX = {"US": ["Inc", "LLC", "Corp", "LP"], "India": ["Pvt Ltd", "Private Limited", "Ltd"],
          "France": ["SARL", "SAS", "SA"]}


def noisy(text, rng):
    chars = list(text)
    if len(chars) > 4 and rng.random() < 0.5:
        i = rng.randrange(1, len(chars) - 1)
        chars[i], chars[i + 1] = chars[i + 1], chars[i]
    out = "".join(chars)
    return out.upper() if rng.random() < 0.3 else out


def business(i, country, rng):
    name = f"{rng.choice(WORDS)} {rng.choice(WORDS)}{i} {rng.choice(SUFFIX[country])}"
    address = f"{rng.randint(1, 9999)} {rng.choice(STREETS)} Road, City{i % 37}, {country}"
    return name, address


def write(path, rows, header):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\t".join(header) + "\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")


def make_split(root, split, countries, n, rng):
    s1, s2, s3, truth, sid = [], [], [], [], 0
    for i in range(n):
        country = countries[i % len(countries)]
        name, address = business(i, country, rng)
        qid = f"S1-{split}{i}"
        s1.append((qid, name, address, country))
        matches = []
        for _ in range(rng.choice([0, 1, 2, 3, 4])):
            sid += 1
            target = (noisy(name, rng), noisy(address, rng) if rng.random() > 0.1 else "", country)
            if rng.random() < 0.5:
                s2.append((f"S2-{split}{sid}", *target))
                matches.append(f"S2-{split}{sid}")
            else:
                s3.append((f"S3-{split}{sid}", *target))
                matches.append(f"S3-{split}{sid}")
        truth.append((qid, ",".join(matches)))
        sid += 1
        other = business(10_000 + i, country, rng)
        (s2 if i % 2 else s3).append((f"S{2 if i % 2 else 3}-{split}{sid}", *other, country))
    header = ["entity_id", "business_name", "business_address", "country"]
    write(root / split / f"{split}_source1.tsv", s1, header)
    write(root / split / f"{split}_source2.tsv", s2, header)
    write(root / split / f"{split}_source3.tsv", s3, header)
    if split == "train":
        write(root / split / "train_ground_truth.tsv", truth, ["source1_entity_id", "matched_entity_ids"])


import pytest


@pytest.mark.parametrize("variant", [[], ["--norm", "v5", "--anchors"]])
def test_distributed_pipeline_end_to_end(tmp_path, variant):
    rng = random.Random(7)
    data = tmp_path / "student_resource" / "dataset"
    make_split(data, "train", ["US", "India"], 400, rng)
    make_split(data, "test", ["US", "India", "France"], 150, rng)
    utils = tmp_path / "student_resource" / "utils"
    utils.mkdir(parents=True)
    import os
    validator = Path(os.environ.get("ER_VALIDATOR", ROOT.parents[1] / "student_resource" / "utils" / "validate_submission.py"))
    if validator.exists():
        (utils / "validate_submission.py").write_bytes(validator.read_bytes())
    store = tmp_path / "store"
    common = [sys.executable, "-m", "src.v5_dist"]
    subprocess.run(common + ["plan", "--data", str(data), "--root", str(store), "--shard", "60"] + variant,
                   cwd=ROOT, check=True)
    plan = json.loads((store / "plan.json").read_text())
    assert plan["test_queries"] == 150
    subprocess.run(common + ["worker", "--data", str(data), "--root", str(store), "--work", str(tmp_path / "w"),
                             "--name", "local", "--workers", "2", "--mem-gb", "64"], cwd=ROOT, check=True)
    failed = list((store / "failed").glob("*.json")) if (store / "failed").exists() else []
    assert not failed, failed[0].read_text()
    tasks = {p.stem for p in (store / "tasks").glob("*.json")}
    assert {p.stem for p in (store / "done").glob("*.json")} == tasks
    manifest = json.loads((store / "model" / "model_manifest.json").read_text())
    metrics = manifest["metrics"]
    assert 0 <= metrics["macro_f05"] <= metrics["oracle_u"] <= 1
    assert metrics["link_recall"] > 0.8
    matching = pl.read_csv(store / "final" / "matching_results.tsv", separator="\t", infer_schema=False)
    candidates = pl.read_csv(store / "final" / "candidate_pairs.tsv", separator="\t", infer_schema=False)
    assert matching.columns == ["source1_entity_id", "matched_entity_ids"]
    assert candidates.columns == ["source1_entity_id", "candidate_entity_ids"]
    assert len(matching) == len(candidates) == 150
    for m, c in zip(matching["matched_entity_ids"].fill_null(""), candidates["candidate_entity_ids"].fill_null("")):
        assert set(filter(None, m.split(","))) <= set(filter(None, c.split(",")))
    report = json.loads((store / "final" / "report.json").read_text())
    if validator.exists():
        assert report["validator_exit"] == 0, report["validator_output"]
