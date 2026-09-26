"""Stable query splits that preserve the original v1 validation population.

The canonical validation interval is always permutation[200000:300000]. Pilots use
a prefix of it, while training excludes the entire interval and the final 10% audit
partition. This assumes the documented one-S1-per-target property; train.py checks
for shared labelled targets across fit and held-out queries before model fitting.
Repeated identity fingerprints across different S1 IDs still need a separate audit.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class QuerySplit:
    train: np.ndarray
    validation: np.ndarray
    reserved_validation: np.ndarray
    audit: np.ndarray
    seed: int
    n_rows: int
    validation_start: int
    validation_size: int

    @property
    def held_out(self) -> np.ndarray:
        return np.sort(np.concatenate([self.reserved_validation, self.audit]))


def make_split(n_rows: int, train_sample: int, val_sample: int, seed: int = 42,
               *, validation_start: int = 200_000, validation_size: int = 100_000,
               audit_fraction: float = 0.10) -> QuerySplit:
    """Return deterministic, nested samples without moving the validation interval."""
    if n_rows <= 0 or train_sample <= 0 or not 0 < val_sample <= validation_size:
        raise ValueError("positive train sample and 1 <= val sample <= validation size required")
    if validation_start < 0 or validation_size <= 0 or not 0 <= audit_fraction < 1:
        raise ValueError("invalid validation interval or audit fraction")
    end = validation_start + validation_size
    audit_start = n_rows - int(n_rows * audit_fraction)
    if end > audit_start:
        raise ValueError("dataset is too small for the canonical validation interval and audit reserve")
    available = audit_start - validation_size
    if train_sample > available:
        raise ValueError(f"train sample {train_sample} exceeds {available} unreserved queries")
    perm = np.random.default_rng(seed).permutation(n_rows)
    fit_pool = np.concatenate([perm[:validation_start], perm[end:audit_start]])

    def ordered(rows):
        return np.sort(rows).astype(np.uint32)

    return QuerySplit(ordered(fit_pool[:train_sample]), ordered(perm[validation_start:validation_start + val_sample]),
                      ordered(perm[validation_start:end]), ordered(perm[audit_start:]),
                      seed, n_rows, validation_start, validation_size)


def split_manifest(split: QuerySplit, query_ids: Iterable[str]) -> dict:
    """Pin the source ID order and every selected/reserved row set without raw IDs."""
    source_hash = hashlib.sha256()
    count = 0
    for value in query_ids:
        encoded = value.encode("utf-8")
        source_hash.update(len(encoded).to_bytes(8, "little"))
        source_hash.update(encoded)
        count += 1
    if count != split.n_rows:
        raise ValueError("query ID count does not match split population")
    manifest = {
        "version": "v1-fixed-validation-1",
        "policy": "seeded file-order permutation; fixed validation interval; final permutation tail audit reserve",
        "seed": split.seed, "query_count": split.n_rows,
        "source1_id_order_sha256": source_hash.hexdigest(),
        "validation_start": split.validation_start,
        "validation_size": split.validation_size,
        "fit_excludes_targets_linked_to_all_reserved_queries": True,
        "repeated_identity_fingerprint_grouping": "not audited",
    }
    for name in ("train", "validation", "reserved_validation", "audit"):
        rows = getattr(split, name)
        manifest[name] = {"count": len(rows), "row_sha256": hashlib.sha256(rows.astype("<u4").tobytes()).hexdigest()}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    manifest["sha256"] = hashlib.sha256(payload).hexdigest()
    return manifest
