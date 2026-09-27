"""Build pair features by target shard, then recompute full-query context."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from .features import FEATURES as BASE_FEATURES, GROUP_BASE, build
from .v4_retrieval import CHANNELS, atomic_json, digest, sha256

EXTRA = ["fold_name_ratio", "fold_name_set", "fold_addr_ratio", "fold_addr_set",
         "name_token_precision", "name_token_recall", "name_extra_tokens", "name_missing_tokens",
         "number_jaccard", "number_precision", "number_recall", "number_missing",
         "name_missing", "address_missing"]
CONTEXT = list(CHANNELS) + ["fold_name_ratio", "fold_addr_ratio"]
FEATURES = BASE_FEATURES + list(CHANNELS) + EXTRA + [f"{x}_{suffix}" for x in CONTEXT for suffix in ("gap", "rank")]


def extra_features(q, t, workers):
    columns = []
    for view, prefix in [("name_fold", "fold_name"), ("address_fold", "fold_addr")]:
        a, b = q[view].fill_null("").to_list(), t[view].fill_null("").to_list()
        missing = np.array([not x or not y for x, y in zip(a, b)])
        for suffix, scorer in [("ratio", fuzz.ratio), ("set", fuzz.token_set_ratio)]:
            values = process.cpdist(a, b, scorer=scorer, dtype=np.float32, workers=workers)
            values[missing] = -1
            columns.append(pl.Series(f"{prefix}_{suffix}", values))
    frame = pl.DataFrame({"qn": q["core"], "tn": t["core"], "qa": q["addr_norm"], "ta": t["addr_norm"]})
    frame = frame.with_columns(
        pl.col("qn", "tn").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique(),
        pl.col("qa", "ta").str.extract_all(r"\d+").list.unique())
    ni = pl.col("qn").list.set_intersection(pl.col("tn")).list.len()
    di = pl.col("qa").list.set_intersection(pl.col("ta")).list.len()
    def fraction(num, den):
        return pl.when(den > 0).then(num / den).otherwise(-1).cast(pl.Float32)
    numbers_missing = (pl.col("qa").list.len() == 0) | (pl.col("ta").list.len() == 0)
    extra = frame.select(
        fraction(ni, pl.col("tn").list.len()).alias("name_token_precision"),
        fraction(ni, pl.col("qn").list.len()).alias("name_token_recall"),
        (pl.col("tn").list.len() - ni).alias("name_extra_tokens"),
        (pl.col("qn").list.len() - ni).alias("name_missing_tokens"),
        fraction(di, pl.col("qa").list.set_union(pl.col("ta")).list.len()).alias("number_jaccard"),
        fraction(di, pl.col("ta").list.len()).alias("number_precision"),
        fraction(di, pl.col("qa").list.len()).alias("number_recall"),
        numbers_missing.cast(pl.Float32).alias("number_missing"),
        ((pl.col("qn").list.len() == 0) | (pl.col("tn").list.len() == 0)).cast(pl.Float32).alias("name_missing"),
        ((q["addr_norm"] == "") | (t["addr_norm"] == "")).cast(pl.Float32).alias("address_missing"))
    return columns + extra.get_columns()


def build_features(queries, candidates, cache_dir, manifest, output_dir, workers=4):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qmap = queries.select(pl.col("row").alias("s1")).with_row_index("local_q")
    schema_pin = digest({"features": FEATURES, "code": sha256(__file__),
                        "base_code": sha256(Path(__file__).with_name("features.py"))})
    frames = []
    for part in manifest["parts"]:
        selected = candidates.filter((pl.col("t") >= part["offset"]) &
                                    (pl.col("t") < part["offset"] + part["rows"]))
        if not len(selected):
            continue
        path = output / part["file"]
        pin = digest({"schema": schema_pin, "target": part["sha256"],
                      "candidates": selected.hash_rows(seed=17).to_list()})
        done = path.with_suffix(".json")
        if path.exists() and done.exists():
            record = __import__("json").loads(done.read_text())
            if record["pin"] != pin or record["sha256"] != sha256(path):
                raise ValueError("Feature checkpoint has changed pins/checksum")
            frame = pl.read_parquet(path)
        else:
            target = pl.read_parquet(Path(cache_dir) / part["file"])
            selected = selected.join(qmap, on="s1", how="left").sort("s1", "t")
            local = selected.select(pl.col("local_q").alias("s1"),
                (pl.col("t") - part["offset"]).cast(pl.UInt32).alias("t"),
                (100 * pl.max_horizontal(list(CHANNELS))).cast(pl.Float32).alias("bscore"),
                pl.lit(0, pl.UInt16).alias("kmask"))
            frame = build(local, queries, target)
            frame = frame.with_columns(extra_features(queries.gather(local["s1"]),
                target.gather(local["t"]), workers) + selected.select(list(CHANNELS)).get_columns())
            frame = frame.with_columns(selected["s1"], selected["t"], target["id"].gather(local["t"]).alias("target_id"))
            frame = frame.select("s1", "t", "target_id", *[x for x in FEATURES if x in frame.columns])
            temp = path.with_suffix(".tmp.parquet")
            frame.write_parquet(temp)
            os.replace(temp, path)
            atomic_json(done, {"pin": pin, "sha256": sha256(path)})
            del target
        frames.append(frame)
        print(f"features {part['file']} pairs={len(frame):,}", flush=True)
    frame = pl.concat(frames)
    # Shard-local context is invalid; always recalculate after the complete union.
    context = [pl.len().over("s1").cast(pl.Float32).alias("n_cand")]
    for feature in GROUP_BASE + CONTEXT:
        context.extend([(pl.col(feature).max().over("s1") - pl.col(feature)).alias(f"{feature}_gap"),
                        pl.col(feature).rank("min", descending=True).over("s1").cast(pl.Float32).alias(f"{feature}_rank")])
    return frame.with_columns(context).with_columns(pl.col(FEATURES).cast(pl.Float32))
