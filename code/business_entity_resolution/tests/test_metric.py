import polars as pl
import pytest

from src.metric import decide, f05_macro, oracle, per_entity


def pairs(rows):
    return pl.DataFrame(rows, schema={"s1": pl.UInt32, "t": pl.UInt32}, orient="row")


def test_golden_cases():
    truth = pairs([(1, 10), (1, 11), (2, 20), (4, 40), (4, 41), (4, 42), (5, 50), (5, 51), (5, 52)])
    pred = pairs([(1, 10), (1, 11), (1, 12),          # PDF example: 2 of 2 plus 1 wrong -> 10/14
                  (3, 30),                            # singleton with a false match -> 0
                  (4, 40), (4, 41),                   # 2 of 3 -> 10/11
                  (5, 50), (5, 51), (5, 52), (5, 53)])  # 3 of 3 plus 1 wrong -> 15/19
    f = per_entity([1, 2, 3, 4, 5, 6], truth, pred).sort("s1")["f"].to_list()
    # entity 2: truth, nothing predicted -> 0; entity 6: singleton, nothing predicted -> 1
    assert f == pytest.approx([10 / 14, 0.0, 0.0, 10 / 11, 15 / 19, 1.0])
    assert f05_macro([1, 2, 3, 4, 5, 6], truth, pred) == pytest.approx(sum(f) / 6)


def test_oracle_counts_blocking_misses():
    truth = pairs([(1, 10), (1, 11), (1, 12)])
    cand = pairs([(1, 10), (1, 99)])
    assert oracle([1, 2], truth, cand) == pytest.approx((5 / 7 + 1.0) / 2)


def test_unique_assign_keeps_best_owner():
    scored = pl.DataFrame({"s1": [1, 2, 2], "t": [10, 10, 11], "p": [0.9, 0.95, 0.4]},
                          schema={"s1": pl.UInt32, "t": pl.UInt32, "p": pl.Float32})
    out = decide(scored, 0.5).sort("t")
    assert out.select("s1", "t").rows() == [(2, 10)]
