"""V7 plan revision 3, WP3: lean audit survivors (label-free) and the one-time G3 evaluator.

    python tools/v7_audit.py survivors --bucket B --ens3 shared/er-v6ens3-20260927 --audit shared/er-v51-audit-20260927
                                       --out shared/er-v7-20260927/audit-survivors --data D --work W [--root DIR]
    python tools/v7_audit.py gate      --survivors DIR --release R/release.json --data D --out G3.json [--cache C]

survivors: for every audit feature shard (the task list of the audit queue, as in v6_train.run_audit), v5.1 p1 from
the source run's model, the ens3 cut from the ens3 manifest, the three V6 members on the survivors and their mean
p6. Writes lean s1, t, target_id, p1, p6 shards, a manifest per shard and a query ledger of every audit S1 (zero-
survivor S1 included), checked against the audit plan's audit_sha. It never reads ground truth.

gate (once): the frozen release on the cached audit features, labels joined, V6 at its manifest threshold against
V7 on every audit S1, and the pre-registered G3 rule. Refuses to run if its output exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent / "v6_analysis"))
import stage2_release as sr  # noqa: E402  (also puts the package on sys.path)
from src.metric import per_entity  # noqa: E402
from src.v4_train import partitions  # noqa: E402
from src.v5_dist import Context, Store, digest  # noqa: E402
from src.v6_train import cut, load_models, read_shard, stage1  # noqa: E402

G3 = {"v6_expected": 0.97089, "tolerance": 1e-4, "min_gain": 0.0010, "z": 1.645, "slice_floor": -0.003}


def audit_rows(plan: dict) -> np.ndarray:
    return np.sort(partitions(plan["train_queries"], plan["fit"], plan["stop"], plan["tune"])[0].audit).astype(np.uint32)


def run_survivors(args):
    started = time.time()
    ens3 = Store(args.bucket, args.ens3, args.root)
    plan, manifest = ens3.get_json("plan.json"), ens3.get_json("model/model_manifest.json")
    ctx = Context(args.data, Path(args.work), args.threads, plan)
    Path(args.work).mkdir(parents=True, exist_ok=True)
    booster1, manifest1 = stage1(ctx, Store(args.bucket, plan["source_prefix"], args.root))
    if manifest1["model_sha256"] != plan["stage1_model_sha256"]:
        raise SystemExit("v5.1 model differs from the one the V6 plan names")
    audit_store = Store(args.bucket, args.audit, args.root)
    audit_plan = audit_store.get_json("plan.json")
    audit = audit_rows(plan)
    if digest(audit.tolist()) != audit_plan["audit_sha"]:
        raise SystemExit("audit rows differ from the audit feature plan")
    boosters, names = load_models(ens3, [m for m, _ in manifest["members"]], ctx)
    if [sha for _, sha, _ in boosters] != [sha for _, sha in manifest["members"]] or names != manifest1["features"]:
        raise SystemExit("V6 members or their feature list differ from the ens3 manifest")
    n, floor = manifest["cascade"]["n"], manifest["cascade"]["floor"]
    out = Store(args.bucket, args.out, args.root)
    audit_set = pl.Series(audit).implode()
    seen, counts = [], []
    for name in sorted(x[:-5] for x in audit_store.list("tasks")):
        if out.exists(f"surv/{name}.json"):
            meta = out.get_json(f"surv/{name}.json")
            seen.append(pl.read_parquet(out.get_bytes(f"surv/{name}.s1.parquet")))
            counts.append(meta["survivors"])
            print(f"{name}: already extracted", flush=True)
            continue
        frame = read_shard(audit_store, f"feat/{name}.parquet", ctx)
        if frame.filter(~pl.col("s1").is_in(audit_set)).height:
            raise SystemExit(f"{name}: rows outside the audit partition")
        p1 = booster1.predict(frame.select(names).to_numpy(), num_threads=args.threads)
        frame = frame.with_columns(pl.Series("p1", p1.astype(np.float32)))
        kept = cut(frame, n, floor)
        x = kept.select(names).to_numpy()
        p6 = np.mean([b.predict(x, num_threads=args.threads) for _, _, b in boosters], axis=0) if len(kept) else np.zeros(0)
        lean = kept.select("s1", "t", "target_id", "p1").with_columns(pl.Series("p6", p6.astype(np.float32))).sort("s1", "t")
        present = frame.select(pl.col("s1").unique().sort())
        local = Path(args.work) / f"{name}.parquet"
        lean.write_parquet(local)
        out.upload(local, f"surv/{name}.parquet")
        present.write_parquet(Path(args.work) / f"{name}.s1.parquet")
        out.upload(Path(args.work) / f"{name}.s1.parquet", f"surv/{name}.s1.parquet")
        meta = {"shard": name, "union_rows": len(frame), "survivors": len(lean), "s1_in_shard": len(present),
                "s1_with_survivors": lean["s1"].n_unique(), "sha256": sr.sha256(local),
                "pair_digest": sr.pair_digest(lean["s1"], lean["t"], "train")}
        out.put_json(f"surv/{name}.json", meta)
        seen.append(present)
        counts.append(len(lean))
        local.unlink()
        del frame, kept, x
        print(f"{name}: {meta['union_rows']:,} union rows -> {meta['survivors']:,} survivors "
              f"t={time.time() - started:.0f}s", flush=True)
    present = pl.concat(seen)
    if present["s1"].is_duplicated().any():
        raise SystemExit("an audit S1 appears in two feature shards")
    surv = {}
    for name in sorted(x[:-5] for x in audit_store.list("tasks")):
        lean = pl.read_parquet(out.get_bytes(f"surv/{name}.parquet"), columns=["s1"])
        for s, k in lean.group_by("s1").len().iter_rows():
            surv[s] = k
    ledger = pl.DataFrame({"s1": audit}).with_columns(
        pl.col("s1").replace_strict(surv, default=0, return_dtype=pl.UInt16).alias("survivors"),
        pl.col("s1").is_in(present["s1"].implode()).alias("retrieved"))
    local = Path(args.work) / "ledger.parquet"
    ledger.write_parquet(local)
    out.upload(local, "ledger.parquet")
    summary = {"audit_s1": len(audit), "audit_sha": audit_plan["audit_sha"], "survivor_pairs": int(sum(counts)),
               "zero_survivor_s1": int((ledger["survivors"] == 0).sum()), "cascade": manifest["cascade"],
               "ens3_model_sha256": manifest["model_sha256"], "members": manifest["members"],
               "stage1_model_sha256": manifest1["model_sha256"], "ledger_sha256": sr.sha256(local),
               "labels_read": False, "seconds": time.time() - started, "peak_rss_gb": sr.peak_rss_gb()}
    out.put_json("manifest.json", summary)
    print(json.dumps(summary), flush=True)


def g3_pass(v6: float, d: dict) -> bool:
    return bool(abs(v6 - G3["v6_expected"]) <= G3["tolerance"] and d["mean"] >= G3["min_gain"]
                and d["mean"] - G3["z"] * d["se"] > 0
                and all(d["by_country"][c]["delta"] >= 0 for c in ("India", "US") if c in d["by_country"])
                and d["singletons"]["delta"] >= G3["slice_floor"] and d["one_true_match"]["delta"] >= G3["slice_floor"])


def run_gate(args):
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: G3 runs once")
    started = time.time()
    release, boosters = sr.load_release(Path(args.release))
    folder = Path(args.survivors)
    summary = json.loads((folder / "manifest.json").read_text())
    if summary["ens3_model_sha256"] != release["ens3"]["model_sha256"] or summary["cascade"] != release["ens3"]["cascade"]:
        raise SystemExit("audit survivors were extracted for a different V6 or cut")
    ledger = pl.read_parquet(folder / "ledger.parquet")
    if sr.sha256(folder / "ledger.parquet") != summary["ledger_sha256"] or len(ledger) != summary["audit_s1"]:
        raise SystemExit("audit ledger checksum or size mismatch")
    split = sr.Split(Path(args.data), "audit", args.threads)
    policy = release["candidate_policy"]
    parts = []
    for meta_path in sorted((folder / "surv").glob("*.json")):
        meta = json.loads(meta_path.read_text())
        source = folder / "surv" / f"{meta['shard']}.parquet"
        if sr.sha256(source) != meta["sha256"]:
            raise SystemExit(f"{meta['shard']}: checksum mismatch")
        frame, feat = sr.cached_features(split, source, Path(args.cache) if args.cache else None, policy["n"], policy["floor"])
        if feat is not None:
            parts.append(sr.apply_release(feat, release, boosters, args.threads))
    scored = pl.concat(parts)
    if len(scored) != summary["survivor_pairs"]:
        raise SystemExit("scored audit pairs differ from the extracted survivors")
    audit = ledger["s1"].to_numpy()
    # The audit is opened here, once: labels are joined only after the release is frozen and every pin checked.
    gt = sr.tune_truth(Path(args.data), audit).select("s1", pl.col("target_id").alias("t"))
    country = sr.read_tsv(Path(args.data) / "train" / "train_source1.tsv", columns=["country"]).with_row_index("s1").with_columns(
        pl.col("s1").cast(pl.UInt32)).filter(pl.col("s1").is_in(pl.Series(audit).implode()))
    per_v6 = per_entity(audit, gt, sr.decide(scored, "p6", release["ens3"]["threshold"]))
    per_v7 = per_entity(audit, gt, sr.decide(scored, "s", release["selection"]["threshold"]))
    d = sr.paired(per_v6, per_v7, country)
    v6, v7 = float(per_v6["f"].mean()), float(per_v7["f"].mean())
    result = {"audit_s1": len(audit), "v6_macro_f05": v6, "v7_macro_f05": v7, "paired_v7_minus_v6": d,
              "g3_rule": G3, "pass": g3_pass(v6, d), "release_sha256": sr.sha256(args.release),
              "survivors_manifest": summary, "second_audit_look": "pre-registered single binary decision after V6's audit",
              "seconds": time.time() - started}
    with out.open("x") as stream:
        json.dump(result, stream, indent=1)
    print(f"G3: V6 {v6:.5f} V7 {v7:.5f} gain {d['mean']:+.5f} (se {d['se']:.5f}) {'PASS' if result['pass'] else 'FAIL'}",
          flush=True)
    return 0 if result["pass"] else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    threads = int(os.environ.get("ER_V6_THREADS", os.cpu_count() or 4))
    s = sub.add_parser("survivors")
    for name in ("--ens3", "--audit", "--out", "--data", "--work"):
        s.add_argument(name, required=True)
    s.add_argument("--bucket")
    s.add_argument("--root")
    g = sub.add_parser("gate")
    for name in ("--survivors", "--release", "--data", "--out"):
        g.add_argument(name, required=True)
    g.add_argument("--cache")
    for p in (s, g):
        p.add_argument("--threads", type=int, default=threads)
    args = parser.parse_args(argv)
    return {"survivors": run_survivors, "gate": run_gate}[args.step](args) or 0


if __name__ == "__main__":
    sys.exit(main())
