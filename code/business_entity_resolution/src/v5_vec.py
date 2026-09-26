"""Hashed n-gram vectorization for v5, parallel across spawned processes."""
from __future__ import annotations

import numpy as np

from .v4_retrieval import CHANNELS


def spec_of(name, spec=None):
    """(analyzer, ngram range) for a channel; v4 channel names need no explicit spec."""
    if spec:
        return spec["analyzer"], tuple(spec["ngram"])
    _, analyzer, ngram = CHANNELS[name]
    return analyzer, ngram


def make_vectorizer(name, dims, spec=None):
    from sklearn.feature_extraction.text import HashingVectorizer
    analyzer, ngram = spec_of(name, spec)
    return HashingVectorizer(n_features=dims, analyzer=analyzer, ngram_range=ngram,
                             alternate_sign=False, norm=None, binary=True, lowercase=False,
                             dtype=np.float32, token_pattern=r"(?u)\b\w+\b")


def _part(job):
    name, dims, spec, texts = job
    return make_vectorizer(name, dims, spec).transform(texts).tocsr()


def vectorize(name, texts, dims, workers, chunk=150_000, spec=None):
    """Binary hashed n-grams (CSR, float32). Identical output for any worker count."""
    from scipy import sparse
    if workers <= 1 or len(texts) <= chunk:
        return _part((name, dims, spec, texts))
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(name, dims, spec, texts[i:i + chunk]) for i in range(0, len(texts), chunk)]
    with ProcessPoolExecutor(min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")) as pool:
        parts = list(pool.map(_part, jobs))
    return sparse.vstack(parts, format="csr")
