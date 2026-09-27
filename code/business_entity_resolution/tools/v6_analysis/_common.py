"""Shared paths and helpers for the v6 analysis scripts.

ER_V6_WORK  directory holding the downloaded run artifacts and the analysis outputs
            (default ./work-v6). Expected inputs, copied from the private bucket:
              plan.json, model_manifest.json, tune_scores.parquet   (shared/er-v51-20260927/{.,model}/)
              test_scores/score-test-*.parquet                      (shared/er-v51-20260927/score/)
ER_DATA_DIR the supplied dataset directory with train/ and test/ (default student_resource/dataset)

Outputs are written back into ER_V6_WORK. They contain record-level data and must stay out of git.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

PACKAGE = Path(__file__).resolve().parents[2]
if str(PACKAGE) not in sys.path:
    sys.path.insert(0, str(PACKAGE))

WORK = Path(os.environ.get("ER_V6_WORK", "work-v6"))
DATA = Path(os.environ.get("ER_DATA_DIR", "student_resource/dataset"))
THRESHOLD = 0.785  # v5.1 full-union threshold from model_manifest.json


def read_tsv(path, **kwargs) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False, **kwargs)


def explode_truth(frame: pl.DataFrame) -> pl.DataFrame:
    return (frame.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(","))
            .explode("matched_entity_ids").filter(pl.col("matched_entity_ids") != ""))


def tune_frame() -> pl.DataFrame:
    """Tune rows (train S1 row numbers), their country and S1 id, from the run plan."""
    plan = json.loads((WORK / "plan.json").read_text())
    tune = pl.DataFrame({"s1": pl.Series(plan["tune_rows"], dtype=pl.UInt32), "country": plan["tune_country"]})
    ids = read_tsv(DATA / "train" / "train_source1.tsv", columns=["entity_id"])["entity_id"]
    return tune.with_columns(ids.gather(tune["s1"]).alias("sid"))


def tune_truth(tune: pl.DataFrame) -> pl.DataFrame:
    """(s1, t) true pairs of the tune rows; t is the target entity id."""
    gt = read_tsv(DATA / "train" / "train_ground_truth.tsv")
    gt = gt.join(tune.select(pl.col("sid").alias("source1_entity_id"), "s1"), on="source1_entity_id")
    return explode_truth(gt).select("s1", pl.col("matched_entity_ids").alias("t"))


def per_query_f(tp, p, g) -> np.ndarray:
    """Competition F0.5 per S1: 5TP/(4|P|+|G|); an empty truth scores 1 only for an empty prediction."""
    tp, p, g = (np.asarray(x) for x in (tp, p, g))
    return np.where(g == 0, (p == 0).astype(float), 5 * tp / np.maximum(1, 4 * p + g))


def macro(pred: pl.DataFrame, g: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
    """pred: predicted (s1, y) rows; g: every evaluated S1 with its truth count `g`."""
    tp = pred.filter(pl.col("y") == 1).group_by("s1").len().rename({"len": "tp"})
    pp = pred.group_by("s1").len().rename({"len": "pp"})
    x = g.join(tp, on="s1", how="left").join(pp, on="s1", how="left").with_columns(pl.col("tp", "pp").fill_null(0))
    return per_query_f(x["tp"].to_numpy(), x["pp"].to_numpy(), x["g"].to_numpy()), x
