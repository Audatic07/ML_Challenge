"""Per-S1 expected-F0.5 set selection on out-of-fold probabilities, against global thresholds.

    python tools/v6_analysis/expected_f.py [oof file] [column]

For each S1 the survivors are sorted by probability and every prefix k = 0..K is scored by
its expected F0.5, treating candidates as independent Bernoulli variables and ignoring
positives outside the survivors. The prefix with the highest expectation is predicted.
This is an approximation (correlated variants, unretrieved positives), measured end to end.
"""
import sys

import numpy as np
import polars as pl

from _common import WORK, per_query_f

name = sys.argv[1] if len(sys.argv) > 1 else "s2_oof_K10_f0.parquet"
col = sys.argv[2] if len(sys.argv) > 2 else "p2"
oof = pl.read_parquet(WORK / name).sort(["s1", col], descending=[False, True])
per = pl.read_parquet(WORK / "tune_per.parquet").select("s1", "g")
groups = oof.group_by("s1", maintain_order=True).agg(pl.col(col), pl.col("y"))
K = int(groups[col].list.len().max())
P = np.zeros((len(groups), K))
Y = np.zeros((len(groups), K))
for i, (pp, yy) in enumerate(zip(groups[col].to_list(), groups["y"].to_list())):
    P[i, :len(pp)], Y[i, :len(yy)] = pp, yy
g = groups.select("s1").join(per, on="s1", how="left")["g"].to_numpy()
assert len(per.join(groups.select("s1"), on="s1", how="anti")) == 0, "every tune S1 needs survivors here"


def poisson_binomial(P, reverse=False):
    """dist[q, k, a]: probability that a of the first k (or of the last K-k) candidates are true."""
    Q, K = P.shape
    dist = np.zeros((Q, K + 1, K + 1))
    order = range(K - 1, -1, -1) if reverse else range(K)
    start = K if reverse else 0
    dist[:, start, 0] = 1
    for k in order:
        src, dst = (k + 1, k) if reverse else (k, k + 1)
        p = P[:, k][:, None]
        dist[:, dst, :] = dist[:, src, :] * (1 - p)
        dist[:, dst, 1:] += dist[:, src, :-1] * p
    return dist


def expected_f(P):
    pre, suf = poisson_binomial(P), poisson_binomial(P, reverse=True)
    a = np.arange(K + 1)
    A, B = np.meshgrid(a, a, indexing="ij")
    out = np.zeros((len(P), K + 1))
    for k in range(K + 1):
        joint = pre[:, k, :, None] * suf[:, k, None, :]
        G = A + B
        f = (G == 0).astype(float) if k == 0 else np.where(G > 0, 5 * A / (4 * k + G), 0.0)
        out[:, k] = (joint * f[None]).sum(axis=(1, 2))
    return out


def score(k):
    tp = np.array([Y[i, :k[i]].sum() for i in range(len(k))])
    return per_query_f(tp, k, g)


k_ef = expected_f(np.clip(P, 0, 1)).argmax(axis=1)
f_ef = score(k_ef)
print(f"{name}:{col} expected-F selection F={f_ef.mean():.5f} mean predicted={k_ef.mean():.3f}")
best = None
for th in np.arange(0.50, 0.901, 0.05):
    k = (P >= th).sum(axis=1)
    f = score(k)
    print(f"threshold {th:.2f}: F={f.mean():.5f} mean predicted={k.mean():.3f}")
    if best is None or f.mean() > best[1].mean():
        best = (th, f)
table = pl.DataFrame({"g": np.minimum(g, 5), "expected_f": f_ef, f"threshold_{best[0]:.2f}": best[1]})
print(table.group_by("g").agg(pl.len(), pl.all().mean()).sort("g"))
