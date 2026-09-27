"""Bounded-memory, full-catalog independent text retrieval.

No labels enter this module. Four channels retain separate top-k per source;
their union has no final cap. Global catalog IDF makes shard scores comparable.
Hash collisions are possible and are recorded as part of this retrieval policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize
from sparse_dot_topn import sp_matmul_topn

from .text_norm import normalise, skeleton

VERSION = "v4-independent-sparse-1"
CHANNELS = {
    "name_char": ("core", "char_wb", (3, 4)),
    "address_char": ("addr_norm", "char_wb", (3, 4)),
    "name_fold": ("name_fold", "word", (1, 2)),
    "address_fold": ("address_fold", "word", (1, 2)),
}
PAIR_SCHEMA = {"s1": pl.UInt32, "t": pl.UInt32, "score": pl.Float32}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temp, path)


def folded(expr):
    # Akash's more aggressive phonetic view is auxiliary, never a replacement
    # for the raw-normalized names or addresses used by the matcher.
    expr = (expr.str.replace_all("bh", "v").str.replace_all("x", "ks")
            .str.replace_all("j", "g").str.replace_all(r"\By", ""))
    return skeleton(expr)


def normalized(frame):
    frame = normalise(frame.with_columns(pl.col(pl.String).fill_null("")))
    return frame.with_columns(
        folded(pl.col("core")).alias("name_fold"),
        folded(pl.col("addr_tok")).alias("address_fold"),
    ).rename({"entity_id": "id"}).select(
        "row", "id", "src", "country", "name_norm", "core", "skel", "concat",
        "addr_norm", "addr_tok", "postcode", "house", "name_fold", "address_fold")


def raw_batches(path, batch_size=100_000):
    reader = pl.read_csv_batched(path, separator="\t", quote_char=None,
        infer_schema_length=0, batch_size=batch_size, n_threads=2)
    while batches := reader.next_batches(1):
        yield from batches


def prepare_catalog(data_dir, cache_dir, split="train", batch_size=100_000):
    """Normalize once in restartable shards. Verify input and normalization pins."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    files = [Path(data_dir) / split / f"{split}_source{s}.tsv" for s in (2, 3)]
    manifest_path = cache / "catalog.json"
    identity = {"split": split, "batch_size": batch_size, "version": VERSION,
        "normalization_sha": sha256(Path(__file__).with_name("text_norm.py")),
        "retrieval_code_sha": sha256(__file__),
        "inputs": [{"source": s, "sha256": sha256(p)} for s, p in zip((2, 3), files)]}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Catalog cache belongs to different data/code; use a fresh cache directory")
        for part in previous["parts"]:
            path = cache / part["file"]
            if not path.exists() or sha256(path) != part["sha256"]:
                raise ValueError(f"Missing/corrupt catalog shard: {path}")
        return previous
    parts, offset = [], 0
    for source, path in zip((2, 3), files):
        for i, raw in enumerate(raw_batches(path, batch_size)):
            out = cache / f"source{source}-{i:05d}.parquet"
            side = out.with_suffix(".json")
            pin = {"identity": digest(identity), "offset": offset, "rows": len(raw)}
            if out.exists() and side.exists() and json.loads(side.read_text()).get("pin") == pin:
                saved = json.loads(side.read_text())
                if sha256(out) != saved["sha256"]:
                    raise ValueError(f"Corrupt normalized shard: {out}")
            else:
                frame = normalized(raw.with_row_index("row", offset=offset).with_columns(
                    pl.lit(source, pl.UInt8).alias("src")))
                temp = out.with_suffix(".tmp.parquet")
                frame.write_parquet(temp)
                os.replace(temp, out)
                saved = {"pin": pin, "sha256": sha256(out)}
                atomic_json(side, saved)
                del frame
            parts.append({"file": out.name, "source": source, "offset": offset,
                          "rows": len(raw), "sha256": saved["sha256"]})
            offset += len(raw)
            print(f"normalized source={source} shard={i} targets={offset:,}", flush=True)
    manifest = {"identity": identity, "rows": offset, "parts": parts}
    manifest["sha256"] = digest(manifest)
    atomic_json(manifest_path, manifest)
    return manifest


def load_queries(data_dir, rows=None, split="train", batch_size=100_000):
    """Rows are global S1 positions; only selected records enter memory."""
    selected, offset = [], 0
    for raw in raw_batches(Path(data_dir) / split / f"{split}_source1.tsv", batch_size):
        raw = raw.with_row_index("row", offset=offset)
        offset += len(raw)
        if rows is not None:
            raw = raw.filter(pl.col("row").is_in(pl.Series(rows, dtype=pl.UInt32).implode()))
        if len(raw):
            selected.append(normalized(raw.with_columns(pl.lit(1, pl.UInt8).alias("src"))))
    return pl.concat(selected).sort("row")


class SparseCatalog:
    def __init__(self, cache_dir, manifest, *, dimensions=2**19, top_k=30,
                 workers=4, max_df=0.1, query_batch=256, channels=None):
        self.cache = Path(cache_dir)
        self.manifest = manifest
        self.channels = channels or list(CHANNELS)
        self.top_k, self.workers = top_k, workers
        self.max_df, self.query_batch = max_df, query_batch
        self.vectorizers = {name: HashingVectorizer(n_features=dimensions,
            analyzer=CHANNELS[name][1], ngram_range=CHANNELS[name][2],
            alternate_sign=False, norm=None, binary=True, lowercase=False,
            dtype=np.float32, token_pattern=r"(?u)\b\w+\b") for name in self.channels}
        self.policy = {"version": VERSION, "channels": self.channels,
            "dimensions": dimensions, "top_k_per_source_channel": top_k,
            "max_df": max_df, "country_partition": True,
            "binary_tf": True, "global_idf": True, "catalog_sha": manifest["sha256"]}
        self.idf = {}

    def fit_idf(self):
        key = digest({k: v for k, v in self.policy.items() if k != "top_k_per_source_channel"})
        path = self.cache / f"idf-{key}.npz"
        if path.exists():
            with np.load(path) as saved:
                self.idf = {name: saved[name].copy() for name in self.channels}
            return
        counts = {n: np.zeros(v.n_features, dtype=np.int64) for n, v in self.vectorizers.items()}
        for part in self.manifest["parts"]:
            frame = pl.read_parquet(self.cache / part["file"])
            for name, vectorizer in self.vectorizers.items():
                matrix = vectorizer.transform(frame[CHANNELS[name][0]])
                counts[name] += np.asarray(matrix.getnnz(axis=0)).ravel()
            print(f"IDF {part['file']}", flush=True)
        for name, count in counts.items():
            weights = (1 + np.log((1 + self.manifest["rows"]) / (1 + count))).astype(np.float32)
            weights[(count == 0) | (count > self.max_df * self.manifest["rows"])] = 0
            self.idf[name] = weights
        temp = path.with_suffix(".tmp.npz")
        np.savez(temp, **self.idf)
        os.replace(temp, path)

    def matrix(self, name, texts):
        mat = self.vectorizers[name].transform(texts)
        mat.data *= self.idf[name][mat.indices]
        mat.eliminate_zeros()
        return normalize(mat, copy=False).tocsr()

    @staticmethod
    def merge_top(parts, k):
        if not parts:
            return pl.DataFrame(schema=PAIR_SCHEMA)
        return (pl.concat(parts).group_by("s1", "t").agg(pl.col("score").max())
                .sort(["s1", "score", "t"], descending=[False, True, False])
                .group_by("s1", maintain_order=True).head(k))

    def retrieve(self, queries, output_dir):
        """Search every target shard; checkpoint results independently per shard."""
        self.fit_idf()
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        pin = digest({"policy": self.policy, "queries": queries.select("row", "id").to_dicts()})
        meta = output / "retrieval.json"
        if meta.exists() and json.loads(meta.read_text())["pin"] != pin:
            raise ValueError("Retrieval checkpoint pins changed; use a new output directory")
        atomic_json(meta, {"pin": pin, "policy": self.policy, "queries": len(queries)})
        qparts = {key[0]: frame for key, frame in queries.partition_by("country", as_dict=True).items()}
        qm = {(country, name): self.matrix(name, frame[CHANNELS[name][0]])
              for country, frame in qparts.items() for name in self.channels}
        best = {(source, name): [] for source in (2, 3) for name in self.channels}
        for part in self.manifest["parts"]:
            start = time.monotonic()
            checkpoint = output / part["file"]
            done = checkpoint.with_suffix(".done.json")
            if checkpoint.exists() and done.exists():
                recorded = json.loads(done.read_text())
                if recorded["pin"] != pin or sha256(checkpoint) != recorded["sha256"]:
                    raise ValueError("Invalid retrieval shard checksum/pin")
                hits = pl.read_parquet(checkpoint)
            else:
                targets = pl.read_parquet(self.cache / part["file"])
                batches = []
                for country_tuple, target in targets.partition_by("country", as_dict=True).items():
                    country = country_tuple[0]
                    if country not in qparts:
                        continue
                    query = qparts[country]
                    target_rows = target["row"].to_numpy()
                    for name in self.channels:
                        tm = self.matrix(name, target[CHANNELS[name][0]]).T.tocsr()
                        for first in range(0, len(query), self.query_batch):
                            similarity = sp_matmul_topn(qm[country, name][first:first+self.query_batch],
                                tm, top_n=self.top_k, threshold=0.0, sort=True, n_threads=self.workers)
                            coo = similarity.tocoo()
                            batches.append(pl.DataFrame({
                                "s1": query["row"].to_numpy()[coo.row + first],
                                "t": target_rows[coo.col], "score": coo.data}, schema=PAIR_SCHEMA)
                                .with_columns(pl.lit(name).alias("channel")))
                        del tm
                hits = pl.concat(batches) if batches else pl.DataFrame(schema={**PAIR_SCHEMA, "channel": pl.String})
                temp = checkpoint.with_suffix(".tmp.parquet")
                hits.write_parquet(temp)
                os.replace(temp, checkpoint)
                atomic_json(done, {"pin": pin, "sha256": sha256(checkpoint)})
                del targets
            for name in self.channels:
                slot = best[part["source"], name]
                slot.append(hits.filter(pl.col("channel") == name).drop("channel"))
                best[part["source"], name] = [self.merge_top(slot, self.top_k)]
            print(f"searched {part['file']} in {time.monotonic()-start:.1f}s", flush=True)
        channels = [frame.with_columns(pl.lit(name).alias("channel"))
            for (_, name), frames in best.items() for frame in frames]
        combined = pl.concat(channels)
        # Each channel's protected quota survives even when another has many decoys.
        result = combined.group_by("s1", "t").agg([
            pl.col("score").filter(pl.col("channel") == name).max().fill_null(0).alias(name)
            for name in self.channels]).sort("s1", "t")
        result.write_parquet(output / "candidates.parquet")
        atomic_json(output / "complete.json", {"pin": pin, "pairs": len(result),
            "sha256": sha256(output / "candidates.parquet")})
        return result
