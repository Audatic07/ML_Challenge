"""Distributed v5 pipeline: S3 task queue executed by SageMaker notebook-instance workers.

    python -m src.v5_dist plan   --data DIR --bucket B --prefix P
    python -m src.v5_dist worker --data DIR --bucket B --prefix P --work DIR --name NAME

Tasks are immutable JSON files under P/tasks/. A worker claims one with an S3
conditional write (If-None-Match) to P/claims/<id>, runs it and records
P/done/<id>.json. Failures go to P/failed/<id>.json and are never retried
automatically; delete the claim to retry after a fix.

Stages
  feat   independent name/address sparse retrieval of one query shard against the full
         target catalog of its country (both sources, per source/channel quota), then
         pair features; train and test shards alike. No labels enter this stage.
  train  LightGBM on the fixed fit partition, early stopping on the stop reserve and a
         global threshold chosen on the full-catalog tune partition. Audit stays closed.
  score  model probabilities for one test feature shard.
  final  global one-owner-per-target assignment, threshold, both TSVs, validation.

Compared with v4 this builds each country catalog once per worker, searches it with
large query batches instead of per-100k-target shards, computes features per query
chunk (context stays exact because every candidate of a query is in its chunk) and
never holds more than one shard of pairs in memory.
"""
from __future__ import annotations

import argparse
import datetime
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import polars as pl

from .metric import per_entity, f05_macro, unique_assign
from .v4_features import CONTEXT, FEATURES, extra_features
from .v4_retrieval import CHANNELS, folded, raw_batches
from .v5_vec import make_vectorizer, vectorize


def normalized(frame, norm="v4"):
    """v4_retrieval.normalized with a selectable text normaliser (plan setting `norm`)."""
    if norm == "v5":
        from .text_norm_v5 import normalise
    else:
        from .text_norm import normalise
    frame = normalise(frame.with_columns(pl.col(pl.String).fill_null("")))
    return frame.with_columns(folded(pl.col("core")).alias("name_fold"),
                              folded(pl.col("addr_tok")).alias("address_fold")).rename({"entity_id": "id"}).select(
        "row", "id", "src", "country", "name_norm", "core", "skel", "concat",
        "addr_norm", "addr_tok", "postcode", "house", "name_fold", "address_fold")

VERSION = "v5-dist-1"
REGION = os.environ.get("AWS_REGION", "ap-south-1")
PLAN_DEFAULTS = {"fit": 400_000, "stop": 10_000, "tune": 100_000, "shard": 20_000,
                 "top_k": 20, "dims": 2 ** 20, "max_df": {"char": 0.03, "word": 0.05},
                 "rounds": 2000, "learning_rate": 0.08}


def log(*args):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
    print(stamp, *args, flush=True)


def mem_gb():
    try:
        info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
        return (int(info["MemTotal"].split()[0]) / 2 ** 20, int(info["MemAvailable"].split()[0]) / 2 ** 20)
    except OSError:
        return (16.0, 8.0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


# ---------------------------------------------------------------- storage
class Store:
    """S3 bucket/prefix, or a local directory (tests)."""

    def __init__(self, bucket=None, prefix="", root=None):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.base_root = Path(root) if root else None
        self.root = (self.base_root / self.prefix if self.prefix else self.base_root) if root else None
        if bucket:
            import boto3
            from botocore.config import Config
            self.s3 = boto3.client("s3", region_name=REGION,
                                   config=Config(retries={"max_attempts": 10, "mode": "adaptive"}))

    def key(self, name):
        return f"{self.prefix}/{name}" if self.prefix else name

    def put_bytes(self, name, data):
        if self.root:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_name(path.name + ".tmp")
            temp.write_bytes(data)
            os.replace(temp, path)
        else:
            self.s3.put_object(Bucket=self.bucket, Key=self.key(name), Body=data,
                               ServerSideEncryption="AES256")

    def put_json(self, name, value):
        self.put_bytes(name, json.dumps(value, indent=1, default=str).encode())

    def get_bytes(self, name):
        if self.root:
            return (self.root / name).read_bytes()
        return self.s3.get_object(Bucket=self.bucket, Key=self.key(name))["Body"].read()

    def get_json(self, name):
        return json.loads(self.get_bytes(name))

    def exists(self, name):
        if self.root:
            return (self.root / name).exists()
        try:
            self.s3.head_object(Bucket=self.bucket, Key=self.key(name))
            return True
        except self.s3.exceptions.ClientError as error:
            if error.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, folder):
        if self.root:
            base = self.root / folder
            return sorted(p.name for p in base.iterdir() if not p.name.endswith(".tmp")) if base.exists() else []
        names, token = [], None
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self.key(folder) + "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.s3.list_objects_v2(**kwargs)
            names += [item["Key"].rsplit("/", 1)[1] for item in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return sorted(names)
            token = page["NextContinuationToken"]

    def create_exclusive(self, name, data):
        """Atomic create; False when the object already exists."""
        if self.root:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as stream:
                    stream.write(data)
                return True
            except FileExistsError:
                return False
        try:
            self.s3.put_object(Bucket=self.bucket, Key=self.key(name), Body=data,
                               IfNoneMatch="*", ServerSideEncryption="AES256")
            return True
        except self.s3.exceptions.ClientError as error:
            code = error.response["Error"]["Code"]
            status = error.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            if code in ("PreconditionFailed", "ConditionalRequestConflict") or status in (409, 412):
                return False
            raise

    def etag(self, name):
        try:
            if self.root:
                return str((self.root / name).stat().st_mtime_ns)
            return self.s3.head_object(Bucket=self.bucket, Key=self.key(name))["ETag"]
        except Exception:
            return None

    def delete(self, name):
        if self.root:
            (self.root / name).unlink(missing_ok=True)
        else:
            self.s3.delete_object(Bucket=self.bucket, Key=self.key(name))

    def upload(self, path, name):
        if self.root:
            self.put_bytes(name, Path(path).read_bytes())
        else:
            self.s3.upload_file(str(path), self.bucket, self.key(name),
                                ExtraArgs={"ServerSideEncryption": "AES256"})

    def download(self, name, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if self.root:
            Path(path).write_bytes((self.root / name).read_bytes())
        else:
            self.s3.download_file(self.bucket, self.key(name), str(path))
        return Path(path)


# ---------------------------------------------------------------- retrieval
def channel_df_cap(name, max_df):
    return max_df["char"] if CHANNELS[name][1].startswith("char") else max_df["word"]


def weigh(matrix, idf):
    from sklearn.preprocessing import normalize
    matrix.data *= idf[matrix.indices]
    matrix.eliminate_zeros()
    return normalize(matrix, copy=False).tocsr()


def channel_specs(plan):
    """Channel -> {view, analyzer, ngram, max_df, top_k}. Defaults reproduce the first v5 plan;
    plan["channels"] adds channels (e.g. the combined name+address view `combo`) or overrides."""
    specs = {}
    for name, (view, analyzer, ngram) in CHANNELS.items():
        cap = plan["max_df"]["char"] if analyzer.startswith("char") else plan["max_df"]["word"]
        specs[name] = {"view": view, "analyzer": analyzer, "ngram": list(ngram), "max_df": cap,
                       "top_k": plan["top_k"]}
    for name, spec in (plan.get("channels") or {}).items():
        specs[name] = {**specs.get(name, {}), **spec}
    return specs


def extra_channels(plan):
    return [c for c in channel_specs(plan) if c not in CHANNELS]


def with_views(frame, specs):
    if any(s["view"] == "combo" for s in specs.values()) and "combo" not in frame.columns:
        frame = frame.with_columns(pl.concat_str([pl.col("core"), pl.col("addr_norm")], separator=" ").alias("combo"))
    return frame


def load_catalog_frame(data_dir, split, country, norm):
    """Normalized targets of one country, source 2 rows then source 3, with global row ids."""
    frames, offset = [], 0
    for source in (2, 3):
        for raw in raw_batches(Path(data_dir) / split / f"{split}_source{source}.tsv"):
            raw = raw.with_row_index("row", offset=offset)
            offset += len(raw)
            raw = raw.filter(pl.col("country") == country)
            if len(raw):
                frames.append(normalized(raw.with_columns(pl.lit(source, pl.UInt8).alias("src")), norm))
    if not frames:
        raise ValueError(f"No {split} targets for country {country!r}")
    frame = pl.concat(frames).rechunk()
    src = frame["src"].to_numpy()
    if np.any(np.diff(src.astype(np.int16)) < 0):
        raise ValueError("Catalog rows must keep source 2 before source 3")
    bounds = {s: (int(np.searchsorted(src, s)), int(np.searchsorted(src, s, side="right"))) for s in (2, 3)}
    return frame, bounds


BLOCK = 250_000  # targets per column block: keeps sparse_dot_topn's dense accumulator in cache


def build_channel(frame, bounds, name, spec, dims, workers, block=None):
    """IDF over the searchable catalog (no labels), then per-source lists of (offset, transposed
    L2 block). Blocking changes speed only: top-k is merged exactly across blocks."""
    block = block or BLOCK
    n = len(frame)
    parts, count = {}, np.zeros(dims, dtype=np.int64)
    for s, (a, b) in bounds.items():
        if b > a:
            parts[s] = vectorize(name, frame[spec["view"]].slice(a, b - a).to_list(), dims, workers, spec=spec)
            count += np.bincount(parts[s].indices, minlength=dims)
    idf = (1 + np.log((1 + n) / (1 + count))).astype(np.float32)
    idf[(count == 0) | (count > spec["max_df"] * n)] = 0
    mats = {}
    for s in list(parts):
        weighted = weigh(parts.pop(s), idf)
        mats[s] = [(first, weighted[first:first + block].T.tocsr()) for first in range(0, weighted.shape[0], block)]
        del weighted
        gc.collect()
    return idf, mats


def search_channel(texts, idf, mats, bounds, name, spec, dims, workers, top_k=None, batch=4096):
    """Top-k of one channel within each source: DataFrame(q, t, score)."""
    from sparse_dot_topn import sp_matmul_topn
    k = top_k or spec["top_k"]
    query = weigh(make_vectorizer(name, dims, spec).transform(texts).tocsr(), idf)
    out = []
    for s, (a, b) in bounds.items():
        if s not in mats:
            continue
        found = []
        for offset, target in mats[s]:
            for first in range(0, query.shape[0], batch):
                block = query[first:first + batch]
                if block.nnz == 0:
                    continue
                hits = sp_matmul_topn(block, target, top_n=k, threshold=1e-6, sort=False, n_threads=workers).tocoo()
                found.append(pl.DataFrame({"q": (hits.row + first).astype(np.uint32),
                                           "t": (hits.col + a + offset).astype(np.uint32),
                                           "score": hits.data.astype(np.float32)}))
        if not found:
            continue
        found = pl.concat(found)
        if len(mats[s]) > 1:
            found = (found.sort(["q", "score", "t"], descending=[False, True, False])
                     .group_by("q", maintain_order=True).head(k))
        out.append(found)
    return pl.concat(out) if out else pl.DataFrame(schema={"q": pl.UInt32, "t": pl.UInt32, "score": pl.Float32})


class Catalog:
    """Every target of one country in one split, one IDF-weighted L2 matrix per channel/source.

    IDF uses the searchable catalog itself (both sources of that country); no labels.
    Row order: source 2 then source 3, preserving the global S2+S3 target row numbers.
    """

    def __init__(self, data_dir, split, country, dims, specs, workers, norm="v4"):
        started = time.monotonic()
        self.key = (split, country)
        self.frame, self.bounds = load_catalog_frame(data_dir, split, country, norm)
        self.frame = with_views(self.frame, specs)
        self.rows = self.frame["row"].to_numpy()
        self.dims, self.specs, self.idf, self.mats = dims, specs, {}, {}
        log(f"catalog {split}/{country}: {len(self.frame):,} targets normalized in {time.monotonic()-started:.0f}s")
        for name, spec in specs.items():
            self.idf[name], self.mats[name] = build_channel(self.frame, self.bounds, name, spec, dims, workers)
            nnz = sum(m.nnz for blocks in self.mats[name].values() for _, m in blocks)
            log(f"catalog {split}/{country}: channel {name} nnz={nnz:,} t={time.monotonic()-started:.0f}s")
        if "combo" in self.frame.columns:
            self.frame = self.frame.drop("combo")
        self.seconds = time.monotonic() - started

    def search(self, queries, workers):
        """Per channel and source top-k, union without a final cap: (q, t, channel scores)."""
        queries = with_views(queries, self.specs)
        parts = []
        for name, spec in self.specs.items():
            started = time.monotonic()
            hits = search_channel(queries[spec["view"]].to_list(), self.idf[name], self.mats[name], self.bounds,
                                  name, spec, self.dims, workers)
            parts.append(hits.with_columns(pl.lit(name).alias("channel")))
            log(f"   search {name}: {len(hits):,} hits in {time.monotonic()-started:.0f}s")
        pairs = pl.concat(parts)
        return (pairs.group_by("q", "t").agg([
            pl.col("score").filter(pl.col("channel") == c).max().fill_null(0).alias(c) for c in self.specs])
            .sort("q", "t"))


ANCHOR = ["h_score", "h_gap", "h_rank", "anc_name_top", "anc_addr_top", "anc_name_max", "anc_addr_max",
           "anc_h_top", "anc_support"]


def feature_list(plan):
    extra = extra_channels(plan)
    return (FEATURES + extra + [f"{c}_{suffix}" for c in extra for suffix in ("gap", "rank")]
            + (ANCHOR if plan.get("anchors") else []))


def anchor_features(frame, catalog, workers, anchors=3):
    """Similarity of each candidate to the query's strongest other candidates.

    A label-free heuristic h ranks candidates; the top `anchors` of each query are
    anchors. Variants of one business tend to agree with each other, so support from
    strong anchors helps; a candidate is never its own anchor.
    """
    from rapidfuzz import fuzz, process
    address = pl.when(pl.col("a_tset") >= 0).then(pl.col("a_tset")).otherwise(pl.col("c_tset"))
    frame = frame.with_columns(((pl.col("c_tset") + pl.col("n_tset") + 2 * address) / 4).cast(pl.Float32).alias("h_score"))
    frame = frame.with_columns(
        (pl.col("h_score").max().over("s1") - pl.col("h_score")).alias("h_gap"),
        pl.col("h_score").rank("ordinal", descending=True).over("s1").cast(pl.Float32).alias("h_rank"))
    top = (frame.filter(pl.col("h_rank") <= anchors)
           .select("s1", pl.col("t").alias("a"), pl.col("h_score").alias("a_h"), pl.col("h_rank").alias("a_rank")))
    pairs = frame.select("s1", "t").with_row_index("i").join(top, on="s1").filter(pl.col("t") != pl.col("a"))
    views = catalog.frame.select("core", "addr_norm")
    c = views.gather(pairs["t"])
    a = views.gather(pairs["a"])
    name_sim = process.cpdist(c["core"].to_list(), a["core"].to_list(), scorer=fuzz.token_set_ratio,
                              workers=workers, dtype=np.float32)
    addr_sim = process.cpdist(c["addr_norm"].to_list(), a["addr_norm"].to_list(), scorer=fuzz.token_set_ratio,
                              workers=workers, dtype=np.float32)
    empty = ((c["addr_norm"] == "") | (a["addr_norm"] == "")).to_numpy()
    addr_sim[empty] = -1
    pairs = pairs.with_columns(pl.Series("ns", name_sim), pl.Series("as", addr_sim))
    pairs = pairs.with_columns(((pl.col("ns") + pl.col("as").clip(lower_bound=0)) / 200 * pl.col("a_h")).alias("support"))
    first = pairs.sort("i", "a_rank").group_by("i", maintain_order=True).first()
    agg = pairs.group_by("i").agg(pl.col("ns").max().alias("anc_name_max"), pl.col("as").max().alias("anc_addr_max"),
                                  pl.col("support").max().alias("anc_support"))
    per = first.select("i", pl.col("ns").alias("anc_name_top"), pl.col("as").alias("anc_addr_top"),
                       pl.col("a_h").alias("anc_h_top")).join(agg, on="i")
    frame = frame.with_row_index("i").join(per, on="i", how="left").drop("i")
    return frame.with_columns(pl.col(ANCHOR).fill_null(-1).cast(pl.Float32))


def pair_features(queries, cand, catalog, workers, chunk_q=4000, anchors=False, plan=None):
    """Features for every candidate. Chunks hold complete queries, so context is exact."""
    from .features import build
    plan = plan or {"anchors": anchors, "max_df": PLAN_DEFAULTS["max_df"], "top_k": PLAN_DEFAULTS["top_k"]}
    names = feature_list({**plan, "anchors": anchors})
    channels = [c for c in cand.columns if c not in ("q", "t")]
    extra = [c for c in channels if c not in CHANNELS]
    frames = []
    q_rows = queries["row"].to_numpy()
    for first in range(0, len(queries), chunk_q):
        sel = cand.filter((pl.col("q") >= first) & (pl.col("q") < first + chunk_q))
        if not len(sel):
            continue
        local = sel.select(pl.col("q").alias("s1"), "t",
                           (100 * pl.max_horizontal(channels)).cast(pl.Float32).alias("bscore"),
                           pl.lit(0, pl.UInt16).alias("kmask"))
        frame = build(local, queries, catalog.frame)
        frame = frame.with_columns(extra_features(queries.gather(local["s1"]), catalog.frame.gather(local["t"]), workers)
                                   + sel.select(channels).get_columns())
        context = []
        for feature in CONTEXT + extra:
            context += [(pl.col(feature).max().over("s1") - pl.col(feature)).alias(f"{feature}_gap"),
                        pl.col(feature).rank("min", descending=True).over("s1").cast(pl.Float32).alias(f"{feature}_rank")]
        frame = frame.with_columns(context)
        if anchors:
            frame = anchor_features(frame, catalog, workers)
        t_local = local["t"].to_numpy()
        frame = frame.select(
            pl.Series("s1", q_rows[local["s1"].to_numpy()], dtype=pl.UInt32),
            pl.Series("t", catalog.rows[t_local], dtype=pl.UInt32),
            catalog.frame["id"].gather(local["t"]).alias("target_id"),
            *[pl.col(x).cast(pl.Float32) for x in names])
        frames.append(frame)
    if not frames:
        return pl.DataFrame(schema={"s1": pl.UInt32, "t": pl.UInt32, "target_id": pl.String,
                                    **{x: pl.Float32 for x in names}})
    return pl.concat(frames)


# ---------------------------------------------------------------- stages
def load_query_rows(data_dir, split, rows, norm="v4"):
    """Selected S1 rows (global file positions), normalized; only those rows enter memory."""
    wanted = pl.Series(np.asarray(rows, dtype=np.uint32))
    selected, offset = [], 0
    for raw in raw_batches(Path(data_dir) / split / f"{split}_source1.tsv"):
        raw = raw.with_row_index("row", offset=offset)
        offset += len(raw)
        raw = raw.filter(pl.col("row").is_in(wanted.implode()))
        if len(raw):
            selected.append(normalized(raw.with_columns(pl.lit(1, pl.UInt8).alias("src")), norm))
    return pl.concat(selected).sort("row")


def run_feat(task, store, ctx):
    started = time.monotonic()
    catalog = ctx.catalog(task["split"], task["country"])
    queries = load_query_rows(ctx.data, task["split"], task["rows"], ctx.plan.get("norm", "v4"))
    if len(queries) != len(task["rows"]) or set(queries["country"].unique()) != {task["country"]}:
        raise ValueError("Query shard does not match its planned rows/country")
    cand = catalog.search(queries, ctx.workers)
    searched = time.monotonic() - started
    feats = pair_features(queries, cand, catalog, ctx.workers, anchors=bool(ctx.plan.get("anchors")), plan=ctx.plan)
    path = ctx.work / f"{task['id']}.parquet"
    feats.write_parquet(path, compression="zstd", compression_level=3)
    store.upload(path, f"feat/{task['id']}.parquet")
    per_query = cand.group_by("q").len()["len"]
    info = {"queries": len(queries), "pairs": len(feats), "zero_candidate_queries": len(queries) - len(per_query),
            "cand_mean": float(per_query.mean() or 0), "cand_p95": float(per_query.quantile(0.95) or 0),
            "cand_max": int(per_query.max() or 0), "search_seconds": searched,
            "feature_seconds": time.monotonic() - started - searched, "catalog_seconds": catalog.seconds,
            "bytes": path.stat().st_size}
    path.unlink()
    return info


def truth_for(data_dir, rows_by_name, heldout):
    """(s1 row, target_id) labels for the selected rows and targets owned by held-out rows."""
    from .v4_train import labels
    ids = pl.read_csv(Path(data_dir) / "train" / "train_source1.tsv", separator="\t", quote_char=None,
                      columns=["entity_id"], infer_schema=False)
    rows = np.unique(np.concatenate(list(rows_by_name.values()))).astype(np.uint32)
    queries = pl.DataFrame({"id": ids["entity_id"].gather(pl.Series(rows)), "row": pl.Series(rows, dtype=pl.UInt32)})
    truth, excluded = labels(data_dir, ids, queries, heldout)
    return truth.with_columns(pl.col("s1").cast(pl.UInt32)), excluded


def run_train(task, store, ctx):
    import lightgbm as lgb
    from .config import LGB_PARAMS
    from .v4_train import partitions
    started = time.monotonic()
    plan = ctx.plan
    names = feature_list(plan)
    n = plan["train_queries"]
    split, fit, stop, tune, heldout = partitions(n, plan["fit"], plan["stop"], plan["tune"])
    if digest(fit.tolist()) != plan["fit_sha"] or digest(tune.tolist()) != plan["tune_sha"]:
        raise ValueError("Train partitions differ from the published plan")
    truth, excluded = truth_for(ctx.data, {"fit": fit, "stop": stop, "tune": tune}, heldout)
    log(f"truth pairs={len(truth):,} excluded held-out targets={len(excluded):,}")
    fit_s, stop_s, tune_s = (pl.Series(x.astype(np.uint32)) for x in (fit, stop, tune))
    positive = truth.with_columns(pl.lit(1, pl.Int8).alias("y"))
    xf, yf, wf, xs, ys, tune_parts = [], [], [], [], [], []
    neg_keep, hard_rank = plan.get("neg_keep"), plan.get("hard_rank", 40)
    rng = np.random.default_rng(20260927)
    for shard in task["inputs"]:
        path = store.download(f"feat/{shard}.parquet", ctx.work / "in" / f"{shard}.parquet")
        frame = pl.read_parquet(path)
        path.unlink()
        frame = frame.join(positive, on=["s1", "target_id"], how="left").with_columns(pl.col("y").fill_null(0))
        part = frame.filter(pl.col("s1").is_in(fit_s.implode()))
        if part.filter(pl.col("y") == 1).join(excluded, on="target_id", how="semi").height:
            raise ValueError("A labelled fit target is owned by a held-out query")
        part = part.join(excluded, on="target_id", how="anti")
        if len(part) and neg_keep:
            # Keep positives and hard negatives (top `hard_rank` by best channel score);
            # sample easy negatives and reweight them so the class balance is preserved.
            easy = ((part["y"] == 0) & (part["bscore_rank"] > hard_rank)).to_numpy()
            keep = ~easy | (rng.random(len(part)) < neg_keep)
            part = part.filter(pl.Series(keep)).with_columns(
                pl.Series("w", np.where(easy[keep], 1.0 / neg_keep, 1.0).astype(np.float32)))
        if len(part):
            xf.append(part.select(names).to_numpy().astype(np.float32, copy=False))
            yf.append(part["y"].to_numpy())
            wf.append(part["w"].to_numpy() if "w" in part.columns else np.ones(len(part), dtype=np.float32))
        part = frame.filter(pl.col("s1").is_in(stop_s.implode()))
        if len(part):
            xs.append(part.select(names).to_numpy().astype(np.float32, copy=False))
            ys.append(part["y"].to_numpy())
        part = frame.filter(pl.col("s1").is_in(tune_s.implode()))
        if len(part):
            tune_parts.append(part.select("s1", "t", "target_id", "y", *names))
        del frame, part
        gc.collect()
        log(f"loaded {shard}: fit rows={sum(len(y) for y in yf):,} mem avail={mem_gb()[1]:.1f}GB")
    y_fit = np.concatenate(yf)
    params = dict(LGB_PARAMS, learning_rate=plan["learning_rate"], num_threads=ctx.workers,
                  deterministic=True, force_col_wise=True)
    dtrain = lgb.Dataset(xf, label=y_fit, weight=np.concatenate(wf), feature_name=names, params=params,
                         free_raw_data=True)
    dstop = lgb.Dataset(np.concatenate(xs), label=np.concatenate(ys), reference=dtrain, params=params)
    dtrain.construct()
    del xf, xs
    gc.collect()
    log(f"LightGBM fit rows={len(y_fit):,} positives={int(y_fit.sum()):,}; training")
    booster = lgb.train(params, dtrain, num_boost_round=plan["rounds"], valid_sets=[dstop], valid_names=["stop"],
                        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(25)])
    model_path = ctx.work / "model.txt"
    booster.save_model(str(model_path))
    trained = time.monotonic() - started
    tune_frame = pl.concat(tune_parts)
    score = booster.predict(tune_frame.select(names).to_numpy(), num_threads=ctx.workers)
    scores = tune_frame.select("s1", "t", "target_id").with_columns(pl.Series("p", score.astype(np.float32)))
    assigned = unique_assign(scores)
    gt = truth.filter(pl.col("s1").is_in(tune_s.implode())).rename({"target_id": "tid"})
    def as_pred(frame):
        return frame.select("s1", pl.col("target_id").alias("t"))
    gt_pairs = gt.select("s1", pl.col("tid").alias("t"))
    grid = [(float(th), f05_macro(tune, gt_pairs, as_pred(assigned.filter(pl.col("p") >= th))))
            for th in np.unique(np.r_[np.arange(0.2, 0.961, 0.02), [0.97, 0.98, 0.99]])]
    threshold, best = max(grid, key=lambda item: item[1])
    fine = [(float(th), f05_macro(tune, gt_pairs, as_pred(assigned.filter(pl.col("p") >= th))))
            for th in np.arange(max(0.05, threshold - 0.02), min(0.995, threshold + 0.021), 0.0025)]
    threshold, best = max(fine + [(threshold, best)], key=lambda item: item[1])
    candidates = scores.select("s1", pl.col("target_id").alias("t"))
    hit = candidates.join(gt_pairs, on=["s1", "t"])
    oracle = per_entity(tune, gt_pairs, hit).rename({"f": "oracle"})
    actual = per_entity(tune, gt_pairs, as_pred(assigned.filter(pl.col("p") >= threshold)))
    per = oracle.join(actual.select("s1", "f"), on="s1")
    country = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    per = per.join(country, on="s1", how="left")
    metrics = {"tune_queries": len(tune), "macro_f05": float(per["f"].mean()), "oracle_u": float(per["oracle"].mean()),
               "matching_gap": float(per["oracle"].mean() - per["f"].mean()), "threshold": threshold,
               "link_recall": len(hit) / max(1, len(gt_pairs)), "truth_pairs": len(gt_pairs),
               "candidate_pairs": len(candidates),
               "singleton_queries": int((per["g"] == 0).sum()),
               "singleton_f": float(per.filter(pl.col("g") == 0)["f"].mean() or 0),
               "zero_candidate_queries": len(tune) - candidates["s1"].n_unique(),
               "by_country": per.group_by("country").agg(pl.len().alias("queries"), pl.col("f").mean().alias("macro_f05"),
                                                        pl.col("oracle").mean().alias("oracle_u")).to_dicts(),
               "best_iteration": booster.best_iteration, "fit_rows": len(y_fit), "train_seconds": trained,
               "grid": grid, "audit_opened": False}
    per.write_parquet(ctx.work / "tune_per_query.parquet")
    scores.write_parquet(ctx.work / "tune_scores.parquet")
    manifest = {"version": VERSION, "features": names, "threshold": threshold, "params": params,
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "metrics": metrics,
                "plan_sha": plan["plan_sha"], "tree_count": booster.num_trees(),
                "model_license": "MIT (LightGBM); no pretrained weights"}
    dumped = booster.dump_model()["tree_info"]
    manifest["parameter_count"] = 2 * sum(t["num_leaves"] for t in dumped) - len(dumped)
    for name in ("tune_per_query.parquet", "tune_scores.parquet"):
        store.upload(ctx.work / name, f"model/{name}")
    store.upload(model_path, "model/model.txt")
    store.put_json("model/model_manifest.json", manifest)
    log("TRAINING_COMPLETE " + json.dumps({k: v for k, v in metrics.items() if k != "grid"}))
    return {k: v for k, v in metrics.items() if k != "grid"}


def run_score(task, store, ctx):
    import lightgbm as lgb
    manifest = ctx.model_manifest(store)
    booster = ctx.booster(store)
    names = feature_list(ctx.plan)
    path = store.download(f"feat/{task['input']}.parquet", ctx.work / "in" / f"{task['input']}.parquet")
    frame = pl.read_parquet(path)
    path.unlink()
    if manifest["features"] != names:
        raise ValueError("Model feature contract differs from this code")
    p = booster.predict(frame.select(names).to_numpy(), num_threads=ctx.workers)
    out = frame.select("s1", "t", "target_id").with_columns(pl.Series("p", p.astype(np.float32)))
    local = ctx.work / f"{task['id']}.parquet"
    out.write_parquet(local)
    store.upload(local, f"score/{task['id']}.parquet")
    local.unlink()
    return {"pairs": len(out), "above_threshold": int((out["p"] >= manifest["threshold"]).sum())}


def run_final(task, store, ctx):
    manifest = ctx.model_manifest(store)
    threshold = manifest["threshold"]
    ids = pl.read_csv(Path(ctx.data) / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                      columns=["entity_id"], infer_schema=False)["entity_id"]
    # Stream shards: keep per-query candidate strings and only above-threshold pairs.
    cand_parts, above, total = [], [], 0
    for shard in task["inputs"]:
        path = store.download(f"score/{shard}.parquet", ctx.work / "in" / f"{shard}.parquet")
        frame = pl.read_parquet(path)
        path.unlink()
        total += len(frame)
        cand_parts.append(frame.group_by("s1").agg(
            pl.col("target_id").sort().str.join(",").alias("candidate_entity_ids")))
        above.append(frame.filter(pl.col("p") >= threshold).select("s1", "t", "target_id", "p"))
        del frame
        gc.collect()
    base = pl.DataFrame({"s1": pl.Series(np.arange(len(ids), dtype=np.uint32)), "source1_entity_id": ids})
    cand = pl.concat(cand_parts)
    del cand_parts
    # Filtering first is equivalent: a target's best claimant survives iff its p passes.
    chosen = unique_assign(pl.concat(above))
    del above
    match = chosen.group_by("s1").agg(pl.col("target_id").sort().str.join(",").alias("matched_entity_ids"))
    out = ctx.work / "output"
    out.mkdir(parents=True, exist_ok=True)
    base.join(match, on="s1", how="left").sort("s1").select(
        "source1_entity_id", pl.col("matched_entity_ids").fill_null("")).write_csv(
        out / "matching_results.tsv", separator="\t", quote_style="never")
    base.join(cand, on="s1", how="left").sort("s1").select(
        "source1_entity_id", pl.col("candidate_entity_ids").fill_null("")).write_csv(
        out / "candidate_pairs.tsv", separator="\t", quote_style="never")
    validator = Path(ctx.data).parent / "utils" / "validate_submission.py"
    report = {"test_queries": len(ids), "matched_queries": len(match), "matched_pairs": len(chosen),
              "candidate_pairs": total, "threshold": threshold, "model_sha256": manifest["model_sha256"]}
    if validator.exists():
        run = subprocess.run([sys.executable, str(validator), "--matching", str(out / "matching_results.tsv"),
                              "--candidate", str(out / "candidate_pairs.tsv"),
                              "--test-dir", str(Path(ctx.data) / "test"), "--check-ids"],
                             capture_output=True, text=True)
        report["validator_exit"] = run.returncode
        report["validator_output"] = run.stdout[-4000:] + run.stderr[-2000:]
    for name in ("matching_results.tsv", "candidate_pairs.tsv"):
        store.upload(out / name, f"final/{name}")
    store.put_json("final/report.json", report)
    return report


# ---------------------------------------------------------------- worker
class Context:
    def __init__(self, data, work, workers, plan, prefix=""):
        self.data, self.work, self.workers, self.plan = Path(data), Path(work), workers, plan
        self.prefix = prefix
        self._catalog = None
        self._booster = None
        self._manifest = None

    def use(self, prefix, plan):
        """Switch to another queue's plan; caches of a different queue are dropped lazily."""
        if prefix != self.prefix:
            self.prefix, self.plan = prefix, plan
            self._booster = self._manifest = None

    def catalog(self, split, country):
        key = (self.prefix, split, country)
        if self._catalog is None or getattr(self._catalog, "queue_key", None) != key:
            self._catalog = None
            gc.collect()
            self._catalog = Catalog(self.data, split, country, self.plan["dims"], channel_specs(self.plan),
                                    self.workers, self.plan.get("norm", "v4"))
            self._catalog.queue_key = key
        return self._catalog

    def drop_catalog(self):
        self._catalog = None
        gc.collect()

    def model_manifest(self, store):
        if self._manifest is None:
            self._manifest = store.get_json("model/model_manifest.json")
        return self._manifest

    def booster(self, store):
        import lightgbm as lgb
        if self._booster is None:
            path = store.download("model/model.txt", self.work / "model.txt")
            if hashlib.sha256(path.read_bytes()).hexdigest() != self.model_manifest(store)["model_sha256"]:
                raise ValueError("Downloaded model checksum mismatch")
            self._booster = lgb.Booster(model_file=str(path))
        return self._booster


def run_bench(task, store, ctx):
    """Retrieval-only comparison of channel variants on labelled development queries.

    Each channel is built on the full catalog of the country, searched once at its largest
    k, and evaluated at every k in `ks` (rank within query and source). Unions of named
    results give the candidate oracle U of a whole policy. No model is involved.
    """
    started = time.monotonic()
    norm = task.get("norm", ctx.plan.get("norm", "v4"))
    specs = task["channels"]
    ctx.drop_catalog()
    frame, bounds = load_catalog_frame(ctx.data, task["split"], task["country"], norm)
    frame = with_views(frame, specs)
    rows = np.asarray(task["rows"], dtype=np.uint32)
    queries = with_views(load_query_rows(ctx.data, task["split"], rows, norm), specs)
    truth, _ = truth_for(ctx.data, {"bench": rows}, np.array([], dtype=np.uint32))
    truth = truth.select("s1", "target_id")
    q_rows = queries["row"].to_numpy()
    ids = frame["id"]
    src3 = bounds[3][0]
    g = truth.group_by("s1").len().rename({"len": "g"})
    base = pl.DataFrame({"s1": pl.Series(rows)}).join(g, on="s1", how="left").fill_null(0)

    def evaluate(cand):
        hit = cand.join(truth, on=["s1", "target_id"])
        r = hit.group_by("s1").len().rename({"len": "r"})
        per = base.join(r, on="s1", how="left").fill_null(0).with_columns(
            pl.when(pl.col("g") == 0).then(1.0).otherwise(5 * pl.col("r") / (4 * pl.col("r") + pl.col("g"))).alias("u"))
        multi = per.filter(pl.col("g") > 0)
        return {"U": float(per["u"].mean()), "recall": len(hit) / max(1, len(truth)),
                "all_found": float((multi["r"] == multi["g"]).mean()), "pairs_per_query": len(cand) / len(rows)}

    results, cands = {}, {}
    for name, spec in specs.items():
        t0 = time.monotonic()
        idf, mats = build_channel(frame, bounds, name, spec, ctx.plan["dims"], ctx.workers)
        built = time.monotonic() - t0
        ks = sorted(set(spec.get("ks", [spec["top_k"]])))
        hits = search_channel(queries[spec["view"]].to_list(), idf, mats, bounds, name, spec, ctx.plan["dims"],
                              ctx.workers, top_k=max(ks))
        searched = time.monotonic() - t0 - built
        del mats
        gc.collect()
        hits = hits.with_columns((pl.col("t") >= src3).alias("s3"))
        hits = hits.with_columns(pl.col("score").rank("ordinal", descending=True).over("q", "s3").alias("k"))
        for k in ks:
            sel = hits.filter(pl.col("k") <= k)
            cand = pl.DataFrame({"s1": q_rows[sel["q"].to_numpy()], "target_id": ids.gather(sel["t"])}).with_columns(
                pl.col("s1").cast(pl.UInt32)).unique()
            cands[f"{name}@{k}"] = cand
            results[f"{name}@{k}"] = {**evaluate(cand), "build_s": round(built, 1), "search_s": round(searched, 1)}
            log(f"bench {name}@{k}: {results[f'{name}@{k}']}")
    for union in task.get("unions", []):
        cand = pl.concat([cands[x] for x in union]).unique()
        results[" + ".join(union)] = evaluate(cand)
        log(f"bench union {union}: {results[' + '.join(union)]}")
    store.put_json(f"bench/{task['id']}.json", {"results": results, "queries": len(rows), "truth": len(truth),
                                                 "seconds": time.monotonic() - started})
    return {"queries": len(rows), "variants": len(results)}


def stage_rank(policy):
    """Label-free stage-1 rank within each query: a rank feature, or the ordinal rank of
    the sum of retrieval cosine scores (a candidate missing from a channel scores 0 there)."""
    if "score" in policy:
        return pl.sum_horizontal([pl.col(c) for c in policy["score"]]).rank("ordinal", descending=True).over("s1")
    return pl.col(policy.get("rank", "h_rank"))


def survivors(frame, cascade):
    """Stage-1 blocking filter applied before the final model: keep the top `n` candidates
    of each query by the label-free stage-1 rank."""
    return frame.filter(stage_rank(cascade) <= cascade["n"])


def run_ceval(task, store, ctx):
    """Tune-set evaluation of stage-1 cut-offs: survivor oracle U and end-to-end macro F0.5
    with the threshold re-selected on survivors. Uses saved tune scores of the source run."""
    source = Store(store.bucket, task["source_prefix"], store.base_root)
    plan = ctx.plan
    tune = np.asarray(plan["tune_rows"], dtype=np.uint32)
    tune_s = pl.Series(tune)
    policies = task.get("policies") or [{"name": r, "rank": r} for r in task.get("ranks", ["h_rank"])]
    columns = sorted({c for p in policies for c in (p["score"] if "score" in p else [p["rank"]])})
    parts = []
    for shard in task["inputs"]:
        path = source.download(f"feat/{shard}.parquet", ctx.work / "in" / f"{shard}.parquet")
        parts.append(pl.read_parquet(path, columns=["s1", "target_id", *columns])
                     .filter(pl.col("s1").is_in(tune_s.implode())))
        path.unlink()
    rank = pl.concat(parts)
    scores = pl.read_parquet(source.download("model/tune_scores.parquet", ctx.work / "in" / "tune_scores.parquet"))
    scores = scores.join(rank, on=["s1", "target_id"], how="inner")
    truth, _ = truth_for(ctx.data, {"tune": tune}, np.array([], dtype=np.uint32))
    gt = truth.select("s1", pl.col("target_id").alias("t"))
    results = {}
    for policy in policies:
        column = policy["name"]
        ranked = scores.with_columns(stage_rank(policy).alias("_stage_rank"))
        for n in task["ns"]:
            kept = ranked.filter(pl.col("_stage_rank") <= n)
            hit = kept.select("s1", pl.col("target_id").alias("t")).join(gt, on=["s1", "t"])
            oracle = f05_macro(tune, gt, hit)
            assigned = unique_assign(kept)
            grid = [(float(th), f05_macro(tune, gt, assigned.filter(pl.col("p") >= th)
                                          .select("s1", pl.col("target_id").alias("t"))))
                    for th in np.arange(0.6, 0.951, 0.01)]
            threshold, best = max(grid, key=lambda item: item[1])
            results[f"{column}<={n}"] = {"macro_f05": best, "threshold": threshold, "oracle_u": oracle,
                                          "link_recall": len(hit) / max(1, len(gt)),
                                          "pairs_per_query": len(kept) / len(tune)}
            log(f"ceval {column}<={n}: {results[f'{column}<={n}']}")
    store.put_json(f"ceval/{task['id']}.json" if task["id"] != "ceval" else "ceval/result.json", results)
    return results


def run_cascade(task, store, ctx):
    """Final scoring of one test shard: stage-1 survivors only, scored by the final model."""
    source = Store(store.bucket, task["source_prefix"], store.base_root)
    manifest = ctx.model_manifest(store)
    booster = ctx.booster(store)
    names = manifest["features"]
    path = source.download(f"feat/{task['input']}.parquet", ctx.work / "in" / f"{task['input']}.parquet")
    frame = survivors(pl.read_parquet(path), manifest["cascade"])
    path.unlink()
    p = booster.predict(frame.select(names).to_numpy(), num_threads=ctx.workers)
    out = frame.select("s1", "t", "target_id").with_columns(pl.Series("p", p.astype(np.float32)))
    local = ctx.work / f"{task['id']}.parquet"
    out.write_parquet(local)
    store.upload(local, f"score/{task['id']}.parquet")
    local.unlink()
    return {"pairs": len(out), "queries": out["s1"].n_unique(),
            "above_threshold": int((out["p"] >= manifest["threshold"]).sum())}


RUNNERS = {"feat": run_feat, "train": run_train, "score": run_score, "final": run_final, "bench": run_bench,
           "ceval": run_ceval, "cascade": run_cascade}


def uploader(holder, name, logfile, state, stop):
    while not stop.wait(60):
        try:
            store = holder["store"]
            store.upload(logfile, f"logs/{name}.log")
            total, avail = mem_gb()
            store.put_json(f"logs/{name}.json", {**state, "mem_total_gb": total, "mem_avail_gb": avail,
                                                  "time": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        except Exception as error:  # keep working when S3 is briefly unavailable
            log(f"log upload warning: {type(error).__name__}")


def release_interrupted(store, name):
    """A task claimed by this worker name without a result was interrupted (restart/OOM).
    Release it for other workers and never retake it here."""
    finished = {x[:-5] for x in store.list("done")} | {x[:-5] for x in store.list("failed")}
    for claim in store.list("claims"):
        if claim not in finished:
            try:
                owner = json.loads(store.get_bytes(f"claims/{claim}")).get("worker")
            except Exception:
                continue
            if owner == name:
                store.put_json(f"attempts/{claim}--{name}.json", {"released": True})
                store.delete(f"claims/{claim}")
                log(f"released interrupted task {claim}")


class Queue:
    """One task queue (S3 prefix) with its immutable plan and cached task definitions."""

    def __init__(self, args, prefix):
        self.prefix = prefix
        self.store = Store(args.bucket, prefix, args.root)
        self.plan = self.store.get_json("plan.json")
        self.tasks, self.loaded_at, self.finished = {}, None, False
        release_interrupted(self.store, args.name)

    def ready(self, name, total_mem, current):
        if self.loaded_at is None or time.monotonic() - self.loaded_at > 60:
            for item in self.store.list("tasks"):
                if item.endswith(".json") and item[:-5] not in self.tasks:
                    self.tasks[item[:-5]] = self.store.get_json(f"tasks/{item}")
            self.loaded_at = time.monotonic()
        done = {x[:-5] for x in self.store.list("done")}
        claimed = set(self.store.list("claims"))
        mine = {x[:-5].rsplit("--", 1)[0] for x in self.store.list("attempts") if x[:-5].endswith("--" + name)}
        self.finished = bool(self.tasks) and len(done) >= len(self.tasks)
        ready = [t for t in self.tasks.values() if t["id"] not in claimed and t["id"] not in mine
                 and t.get("min_mem_gb", 0) <= total_mem and all(r in done for r in t.get("requires", []))]
        ready.sort(key=lambda t: (t["priority"], (self.prefix, t.get("split"), t.get("country")) != current, t["id"]))
        return ready


def worker(args):
    """Serve the queues listed (in priority order) by control/queues of the home prefix."""
    from .v6_train import RUNNERS as V6_RUNNERS  # v6 task kinds (imported late: v6_train imports this module)
    RUNNERS.update(V6_RUNNERS)
    home = Store(args.bucket, args.prefix, args.root)
    holder = {"store": home}
    args.work.mkdir(parents=True, exist_ok=True)
    ctx = Context(args.data, args.work, args.workers, None, prefix=None)
    total_mem = args.mem_gb or mem_gb()[0]
    state = {"worker": args.name, "task": None, "mem_total_gb": total_mem, "host": platform.node(), "queues": []}
    stop = threading.Event()
    if args.logfile:
        threading.Thread(target=uploader, args=(holder, args.name, args.logfile, state, stop), daemon=True).start()
    queues, idle_since = {}, None
    code_etag, code_checked, exit_code = home.etag("code/code.tar.gz"), time.monotonic(), 0
    log(f"worker {args.name} mem={total_mem:.1f}GB workers={args.workers} home={args.prefix}")
    while True:
        if home.exists("control/stop") or home.exists(f"control/stop-{args.name}"):
            log("stop flag found")
            break
        if args.reload and code_etag and time.monotonic() - code_checked > 30:
            code_checked = time.monotonic()
            if home.etag("code/code.tar.gz") not in (None, code_etag):
                log("new code published; exiting for reload")
                exit_code = 75
                break
        order = [home.prefix]
        for marker in (f"control/queues-{args.name}", "control/queues"):
            if home.exists(marker):
                try:
                    order = json.loads(home.get_bytes(marker)) or order
                except ValueError:
                    log(f"unreadable {marker}; using home queue")
                break
        state["queues"] = order
        chosen = None
        for prefix in order:
            if prefix not in queues:
                try:
                    queues[prefix] = Queue(args, prefix)
                    log(f"serving queue {prefix} plan={queues[prefix].plan['plan_sha'][:12]}")
                except Exception as error:
                    log(f"queue {prefix} unavailable: {error!r}")
                    continue
            queue = queues[prefix]
            current = getattr(ctx._catalog, "queue_key", None)
            for task in queue.ready(args.name, total_mem, current):
                if queue.store.create_exclusive(f"claims/{task['id']}", json.dumps(
                        {"worker": args.name, "at": datetime.datetime.now(datetime.timezone.utc).isoformat()}).encode()):
                    chosen = (queue, task)
                    break
            if chosen:
                break
        if chosen is None:
            active = [queues[p] for p in order if p in queues]
            if not args.stay and active and all(q.finished for q in active):
                log("all tasks done")
                break
            idle_since = idle_since or time.monotonic()
            if args.idle_exit and time.monotonic() - idle_since > args.idle_exit:
                log("idle limit reached")
                break
            time.sleep(20)
            continue
        idle_since = None
        queue, task = chosen
        ctx.use(queue.prefix, queue.plan)
        store = queue.store
        state["task"] = f"{queue.prefix}:{task['id']}"
        log(f"START {queue.prefix}:{task['id']}")
        started = time.monotonic()
        try:
            if task["kind"] in ("train", "final", "bench", "v6train"):
                ctx.drop_catalog()
            info = RUNNERS[task["kind"]](task, store, ctx)
            store.put_json(f"done/{task['id']}.json", {"worker": args.name, "seconds": time.monotonic() - started,
                                                        "info": info})
            log(f"DONE {task['id']} in {time.monotonic()-started:.0f}s {json.dumps(info, default=str)[:600]}")
        except Exception as error:
            store.put_json(f"failed/{task['id']}.json", {"worker": args.name, "error": repr(error),
                                                          "traceback": traceback.format_exc()})
            log(f"FAILED {task['id']}: {error!r}\n{traceback.format_exc()}")
            ctx.drop_catalog()
        state["task"] = None
    stop.set()
    if args.logfile:
        home.upload(args.logfile, f"logs/{args.name}.log")
    return exit_code


# ---------------------------------------------------------------- plan
def plan(args):
    from .v4_train import partitions
    store = Store(args.bucket, args.prefix, args.root)
    settings = dict(PLAN_DEFAULTS)
    for key in ("fit", "stop", "tune", "shard", "top_k"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    settings["norm"] = args.norm
    settings["anchors"] = bool(args.anchors)
    settings["max_df"] = {"char": args.max_df_char, "word": args.max_df_word}
    if args.channels_json:
        settings["channels"] = json.loads(Path(args.channels_json).read_text())
    if args.memory_scale is not None:
        settings["memory_scale"] = args.memory_scale
    if args.neg_keep is not None:
        settings["neg_keep"] = args.neg_keep
    train = pl.read_csv(Path(args.data) / "train" / "train_source1.tsv", separator="\t", quote_char=None,
                        columns=["entity_id", "country"], infer_schema=False).with_row_index("row")
    test = pl.read_csv(Path(args.data) / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                       columns=["entity_id", "country"], infer_schema=False).with_row_index("row")
    split, fit, stop, tune, heldout = partitions(len(train), settings["fit"], settings["stop"], settings["tune"])
    selected = np.unique(np.concatenate([fit, stop, tune]))
    tune_country = train["country"].gather(pl.Series(tune.astype(np.uint32)))
    settings.update(train_queries=len(train), test_queries=len(test), fit_sha=digest(fit.tolist()),
                    stop_sha=digest(stop.tolist()), tune_sha=digest(tune.tolist()), version=VERSION,
                    tune_rows=tune.tolist(), tune_country=tune_country.to_list(),
                    created=datetime.datetime.now(datetime.timezone.utc).isoformat())
    tasks = []
    catalog_rows = {}
    for split_name, frame, rows in (("train", train, selected), ("test", test, None)):
        chosen = frame if rows is None else frame.filter(pl.col("row").is_in(pl.Series(rows.astype(np.uint32)).implode()))
        for (country,), group in sorted(chosen.partition_by("country", as_dict=True).items()):
            group_rows = group["row"].sort().to_list()
            for i in range(0, len(group_rows), settings["shard"]):
                tasks.append({"id": f"feat-{split_name}-{country}-{i // settings['shard']:03d}".replace(" ", "_"),
                              "kind": "feat", "split": split_name, "country": country,
                              "rows": group_rows[i:i + settings["shard"]],
                              "priority": 0 if split_name == "train" else 1})
    for split_name in ("train", "test"):
        for source in (2, 3):
            path = Path(args.data) / split_name / f"{split_name}_source{source}.tsv"
            counts = pl.read_csv(path, separator="\t", quote_char=None, columns=["country"],
                                 infer_schema=False)["country"].value_counts()
            for country, count in counts.iter_rows():
                catalog_rows[(split_name, country)] = catalog_rows.get((split_name, country), 0) + count
    for task in tasks:
        size = catalog_rows[(task["split"], task["country"])] / 1e6
        task["min_mem_gb"] = round(min(60.0, 4 + settings.get("memory_scale", 1.4) * size), 1)
        task["catalog_rows"] = catalog_rows[(task["split"], task["country"])]
    train_inputs = [t["id"] for t in tasks if t["split"] == "train"]
    tasks.append({"id": "train", "kind": "train", "inputs": train_inputs, "requires": train_inputs,
                  "priority": -1, "min_mem_gb": 60})
    scores = []
    for t in [t for t in tasks if t.get("split") == "test"]:
        scores.append({"id": t["id"].replace("feat-", "score-"), "kind": "score", "input": t["id"],
                       "requires": ["train", t["id"]], "priority": 2, "min_mem_gb": 8})
    tasks += scores
    tasks.append({"id": "final", "kind": "final", "inputs": [t["id"] for t in scores],
                  "requires": [t["id"] for t in scores], "priority": 3, "min_mem_gb": 60})
    settings["plan_sha"] = digest({k: v for k, v in settings.items() if k != "created"})
    settings["task_count"] = len(tasks)
    store.put_json("plan.json", settings)
    for task in tasks:
        store.put_json(f"tasks/{task['id']}.json", task)
    summary = {t["id"]: len(t.get("rows", [])) for t in tasks if t["kind"] == "feat"}
    log(f"published {len(tasks)} tasks; feat shards={len(summary)}; train rows={len(selected):,}; test rows={len(test):,}")
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "worker"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--root", type=Path, help="local directory store instead of S3 (tests)")
    parser.add_argument("--work", type=Path, default=Path("work-v5"))
    parser.add_argument("--name", default=platform.node())
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--logfile", type=Path)
    parser.add_argument("--idle-exit", type=int, default=0)
    parser.add_argument("--mem-gb", type=float, help="override detected memory (tests)")
    parser.add_argument("--stay", action="store_true", help="keep polling after the queue is finished")
    parser.add_argument("--reload", action="store_true", help="exit with 75 between tasks when code.tar.gz changes")
    parser.add_argument("--norm", choices=["v4", "v5"], default="v4", help="text normaliser (plan)")
    parser.add_argument("--anchors", action="store_true", help="add anchor-support features (plan)")
    parser.add_argument("--channels-json", type=Path, help="channel specs added to/overriding the v4 four (plan)")
    parser.add_argument("--memory-scale", type=float, help="GB per million catalog rows for feat tasks (plan)")
    parser.add_argument("--neg-keep", type=float, help="fraction of easy fit negatives kept, reweighted (plan)")
    parser.add_argument("--max-df-char", type=float, default=0.03)
    parser.add_argument("--max-df-word", type=float, default=0.05)
    for key in ("fit", "stop", "tune", "shard", "top_k"):
        parser.add_argument(f"--{key.replace('_', '-')}", dest=key, type=int)
    args = parser.parse_args()
    if not args.bucket and not args.root:
        parser.error("--bucket or --root is required")
    if args.command == "plan":
        plan(args)
    else:
        sys.exit(worker(args))


if __name__ == "__main__":
    main()
