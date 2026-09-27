import numpy as np
import polars as pl

from src.v6_stage2 import SIBLINGS, survivors, features, text_views


def records(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


def test_survivors_top_k_and_floor():
    scored = pl.DataFrame({"s1": [0, 0, 0, 0, 1, 1], "t": [5, 6, 7, 8, 5, 9],
                           "target_id": list("abcdef"), "p": [0.9, 0.2, 0.0005, 0.95, 0.0001, 0.0002]})
    top = survivors(scored, 2)
    assert top.filter(pl.col("s1") == 0)["t"].to_list() == [8, 5]
    assert top["rank"].to_list() == [0, 1, 0, 1]
    floored = survivors(scored, 3, floor=0.001)
    assert floored.filter(pl.col("s1") == 0)["t"].to_list() == [8, 5, 6]
    assert floored.filter(pl.col("s1") == 1).height == 0  # every candidate below the floor


def test_features_use_confident_siblings():
    s1 = text_views(records([("S1-1", "Rana Better Engineering Inc", "3445 Refugee Road, Columbus, OH", "US")]))
    targets = records([
        ("S2-1", "Rana Better Engineering Inc", "3445 Refugee Rd, Columbus, Ohio", "US"),   # confident sibling
        ("S3-1", "Pyrawex", "3445 Refugee Rd, Columbus, Ohio", "US"),                       # renamed variant
        ("S3-2", "Rana Better Engineering", "", "US"),                                      # no address
        ("S2-2", "Quinta Bakery", "12 Elm St, Dayton, OH", "US")])                          # unrelated
    t_text = text_views(targets).with_columns(pl.Series("src", [2, 3, 3, 2], dtype=pl.UInt8))
    scored = pl.DataFrame({"s1": [0, 0, 0, 0], "t": [0, 1, 2, 3], "target_id": targets["entity_id"],
                           "p": [0.99, 0.6, 0.5, 0.01]})
    feat = features(survivors(scored, 10), s1, t_text, workers=1)
    assert feat.height == 4 and feat["t"].to_list() == [0, 1, 2, 3]
    row = {r["target_id"]: r for r in feat.iter_rows(named=True)}
    assert row["S3-1"]["sibc_a_eq"] == 1 and row["S3-1"]["q_n_shared"] == 0
    assert row["S3-2"]["q_a_tset"] == -1 and row["S3-2"]["sibc_n_tset"] >= 90
    assert row["S2-2"]["sibc_a_eq"] == 0
    assert row["S2-1"]["sib1_p"] == np.float32(0.6)  # the best other survivor, never itself
    assert row["S2-1"]["others_p90"] == 0 and row["S3-1"]["others_p90"] == 1
    assert SIBLINGS == 5
