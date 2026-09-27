"""Word-level transliteration dictionary learned from the training labels.

Many Source 2/3 names and addresses are written in Devanagari, Bengali, Telugu, ...
and `unidecode` turns them into spellings like `sonphttveyr` for "software" or
`mhaaraassttr` for "maharashtra". Source 1 is always Latin script.

For every labelled (S1, target) pair of the fit queries, each target token that never
occurs in any S1 record ("foreign" token) is associated with the S1 tokens missing
from the target. A foreign token x is mapped to the S1 token y with the highest
Jaccard association |pairs(x, y)| / |pairs(x) U pairs(y)|, when that association is
strong enough. Only supplied training data is used. The dictionary is then applied
token by token to every record of every split (train and test).
"""
from __future__ import annotations

import json

import polars as pl

from .config import WORK_DIR

MIN_PAIRS = 3
MIN_JACCARD = 0.2


def _tok(df: pl.DataFrame, row: str, col: str) -> pl.DataFrame:
    return (df.select(pl.col("row").alias(row), pl.col(col).str.split(" ").list.unique().alias("w"))
            .explode("w").filter(pl.col("w").is_not_null() & (pl.col("w") != "")))


def learn_view(s1: pl.DataFrame, tg: pl.DataFrame, links: pl.DataFrame, col: str) -> dict[str, str]:
    """links: (s1, t) labelled pairs of fit queries only."""
    q_vocab = _tok(s1, "s1", col).select("w").unique()
    links = links.with_row_index("pid")
    qt = links.select("pid", "s1").join(_tok(s1, "s1", col), on="s1").select("pid", "w")
    tt = links.select("pid", "t").join(_tok(tg, "t", col), on="t").select("pid", "w")
    x = tt.join(qt, on=["pid", "w"], how="anti").join(q_vocab, on="w", how="anti")  # foreign target tokens
    y = qt.join(tt, on=["pid", "w"], how="anti")  # S1 tokens missing from the target
    occ_x = x.group_by("w").agg(pl.len().alias("nx"))
    occ_y = y.group_by("w").agg(pl.len().alias("ny"))
    xy = (x.rename({"w": "x"}).join(y.rename({"w": "y"}), on="pid")
          .group_by("x", "y").agg(pl.len().alias("n")))
    xy = (xy.join(occ_x.rename({"w": "x"}), on="x").join(occ_y.rename({"w": "y"}), on="y")
          .with_columns((pl.col("n") / (pl.col("nx") + pl.col("ny") - pl.col("n"))).alias("jac"))
          .filter((pl.col("n") >= MIN_PAIRS) & (pl.col("jac") >= MIN_JACCARD))
          .sort(["x", "jac", "n", "y"], descending=[False, True, True, False])
          .group_by("x", maintain_order=True).head(1))
    return dict(zip(xy["x"].to_list(), xy["y"].to_list()))


def learn(s1: pl.DataFrame, tg: pl.DataFrame, links: pl.DataFrame) -> dict:
    maps = {"name_norm": learn_view(s1, tg, links, "name_norm"),
            "addr_norm": learn_view(s1, tg, links, "addr_norm")}
    (WORK_DIR / "translit.json").write_text(json.dumps(maps, ensure_ascii=True, indent=0))
    return maps


def load_maps() -> dict:
    return json.loads((WORK_DIR / "translit.json").read_text())


def apply_view(df: pl.DataFrame, col: str, mapping: dict[str, str]) -> pl.Series:
    if not mapping:
        return df[col]
    m = pl.DataFrame({"w": list(mapping), "to": list(mapping.values())})
    toks = (df.select(pl.col("row"), pl.col(col).fill_null("").str.split(" ").alias("w"))
            .explode("w").with_row_index("pos")
            .join(m, on="w", how="left", maintain_order="left")
            .with_columns(pl.coalesce("to", "w").alias("w"))
            .group_by("row", maintain_order=True).agg(pl.col("w").str.join(" ")))
    out = df.select("row").join(toks, on="row", how="left", maintain_order="left")["w"]
    return out.fill_null("").str.strip_chars().alias(col)
