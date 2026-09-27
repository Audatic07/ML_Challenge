import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def _model(path, names, rng):
    import lightgbm as lgb
    X = rng.random((400, len(names)))
    y = (X[:, 0] + 0.3 * rng.random(400) > 0.7).astype(int)
    lgb.train({"objective": "binary", "verbose": -1, "num_leaves": 4}, lgb.Dataset(X, y, feature_name=names), 5).save_model(str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_survivors_are_label_free_and_cover_every_audit_s1(tmp_path):
    import v7_audit
    from src.v5_dist import digest

    rng = np.random.default_rng(3)
    names = ["f0", "f1", "f2"]
    store = tmp_path / "store"
    (store / "v51" / "model").mkdir(parents=True)
    v51_sha = _model(store / "v51" / "model" / "model.txt", names, rng)
    (store / "v51" / "model" / "model_manifest.json").write_text(json.dumps({"model_sha256": v51_sha, "features": names}))
    members = []
    for i in range(3):
        folder = store / f"m{i}" / "model"
        folder.mkdir(parents=True)
        sha = _model(folder / "model.txt", names, rng)
        (folder / "model_manifest.json").write_text(json.dumps({"model_sha256": sha, "features": names}))
        members.append([f"m{i}", sha])
    plan = {"train_queries": 240, "fit": 150, "stop": 10, "tune": 24, "source_prefix": "v51", "stage1_model_sha256": v51_sha}
    (store / "ens3" / "model").mkdir(parents=True)
    (store / "ens3" / "plan.json").write_text(json.dumps(plan))
    (store / "ens3" / "model" / "model_manifest.json").write_text(json.dumps(
        {"members": members, "model_sha256": "ens", "cascade": {"stage1": "model", "n": 12, "floor": 0.01}, "threshold": 0.72}))
    audit = v7_audit.audit_rows(plan)
    assert len(audit) == 24
    (store / "audit" / "tasks").mkdir(parents=True)
    (store / "audit" / "feat").mkdir()
    (store / "audit" / "plan.json").write_text(json.dumps({"audit_sha": digest(audit.tolist())}))
    retrieved = audit[:-1]  # the last audit S1 has no retrieved candidates at all
    for k, rows in enumerate(np.array_split(retrieved, 2)):
        s1 = np.repeat(rows, 20).astype(np.uint32)
        t = rng.choice(5000, len(s1), replace=False).astype(np.uint32)
        frame = pl.DataFrame({"s1": s1, "t": t, "target_id": [f"T{x}" for x in t],
                              **{n: rng.random(len(s1)) for n in names}})
        frame.write_parquet(store / "audit" / "feat" / f"feat-audit-{k:03d}.parquet")
        (store / "audit" / "tasks" / f"feat-audit-{k:03d}.json").write_text("{}")

    args = ["survivors", "--root", str(store), "--ens3", "ens3", "--audit", "audit", "--out", "v7/audit-survivors",
            "--data", str(tmp_path / "no-dataset"), "--work", str(tmp_path / "w"), "--threads", "1"]
    v7_audit.main(args)  # no dataset directory exists: extraction cannot read any label
    out = store / "v7" / "audit-survivors"
    summary = json.loads((out / "manifest.json").read_text())
    ledger = pl.read_parquet(out / "ledger.parquet")
    assert summary["labels_read"] is False and summary["audit_s1"] == 24 and len(ledger) == 24
    assert ledger.filter(pl.col("s1") == int(audit[-1]))["retrieved"].item() is False
    lean = pl.concat([pl.read_parquet(p) for p in sorted((out / "surv").glob("*.parquet")) if not p.name.endswith(".s1.parquet")])
    assert lean.columns == ["s1", "t", "target_id", "p1", "p6"] and len(lean) == summary["survivor_pairs"]
    assert lean["p1"].min() >= 0.01 and lean.group_by("s1").len()["len"].max() <= 12
    assert int(ledger["survivors"].sum()) == len(lean) and summary["zero_survivor_s1"] >= 1
    v7_audit.main(args)  # restart: shards already extracted are reused, the ledger is identical
    assert pl.read_parquet(out / "ledger.parquet").equals(ledger)

    bad = json.loads((store / "audit" / "plan.json").read_text())
    (store / "audit" / "plan.json").write_text(json.dumps({"audit_sha": "0" * 64}))
    with pytest.raises(SystemExit):
        v7_audit.main(args)
    (store / "audit" / "plan.json").write_text(json.dumps(bad))
