import polars as pl

from src import blocking
from src.text_norm import normalise


def records(names, addresses):
    return normalise(pl.DataFrame({"business_name": names, "business_address": addresses,
                                   "country": ["France"] * len(names)})).with_row_index("row")


def policy(monkeypatch, rescue):
    keys = {"A": (3, 200), "B": (3, 200), "C": (2, 200), "D": (2, 100),
            "E": (1, 50), "F": (2, 50), "G": (2, 100)}
    if rescue:
        keys.update({"H": (2, 200), "I": (1, 100)})
    monkeypatch.setattr(blocking, "KEY_TYPES", keys)


def test_address_rescue_without_shared_name_or_number(monkeypatch):
    q = records(["Amber Studio"], ["Cedar Willow Birch"])
    t = records(["Zinc Works"], ["Birch Cedar Willow"])
    policy(monkeypatch, False)
    assert len(blocking.Blocker(q, t).candidates([0])) == 0
    policy(monkeypatch, True)
    out = blocking.Blocker(q, t).candidates([0])
    assert out.select("s1", "t").rows() == [(0, 0)]
    # Three shared address keys contribute only one channel bit/weight.
    assert out["kmask"].to_list() == [128]
    assert out["bscore"].to_list() == [2]


def test_ninth_channel_does_not_overflow(monkeypatch):
    policy(monkeypatch, True)
    q = records(["Amber Studio"], [""])
    t = records(["Amber Studio"], [""])
    out = blocking.Blocker(q, t).candidates([0])
    assert int(out["kmask"][0]) & 256
    assert out.select("s1", "t").n_unique() == len(out)


def test_topk_deterministic_and_empty_queries(monkeypatch):
    policy(monkeypatch, True)
    q = records(["Amber Studio", ""], ["", ""])
    t = records(["Amber Studio"] * 3, [""] * 3)
    out = blocking.Blocker(q, t).candidates([0, 1], top_k=2)
    assert out.select("s1", "t").rows() == [(0, 0), (0, 1)]


def test_cap_counts_unique_target_records(monkeypatch):
    policy(monkeypatch, True)
    monkeypatch.setitem(blocking.KEY_TYPES, "H", (2, 1))
    q = records(["Amber Studio"], ["Cedar Willow Birch"])
    t = records(["Zinc Works", "Nickel Shop"], ["Cedar Willow Birch"] * 2)
    assert len(blocking.Blocker(q, t).candidates([0])) == 0


def test_text_ranking_preserves_address_only_rescue(monkeypatch):
    policy(monkeypatch, True)
    monkeypatch.setattr(blocking, "RETRIEVAL_RANK", "text")
    q = records(["Amber Studio"], ["Cedar Willow Birch"])
    t = records(["Amber Studio", "Zinc Works"], ["", "Cedar Willow Birch"])
    out = blocking.Blocker(q, t).candidates([0], top_k=1)
    assert out.select("s1", "t").rows() == [(0, 1)]
