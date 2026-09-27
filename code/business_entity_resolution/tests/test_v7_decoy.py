import polars as pl

from src.v7_decoy import F2_COLUMNS, build_context, decoy_features


def records(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


def test_decoy_support_typos_and_missing_houses():
    s1 = records([("S1-0", "Rana Better Engineering", "3445 Refugee Road, Columbus, OH", "US")])
    targets = records([
        ("S2-0", "Rana Beter Engineering", "3445 Refugee Rd, Columbus, OH", "US"),       # typo: not a true extra word
        ("S2-1", "Rana Bakery Engineering", "3447 Refugee Rd, Columbus, OH", "US"),      # decoy word "bakery"
        ("S2-2", "Rana Bakery Engineering Works", "3447 Refugee Rd, Columbus, OH", "US"),  # shares the decoy word
        ("S2-3", "Rana Better Engineering", "", "US"),                                   # no address, no house
    ])
    ctx = build_context(targets)
    pairs = pl.DataFrame({"s1": [0, 0, 0, 0], "t": [0, 1, 2, 3]}, schema={"s1": pl.UInt32, "t": pl.UInt32})
    f = decoy_features(pairs, s1, targets, ctx, pl.Series([95.0, 95.0, 95.0, -1.0]), threads=1)
    assert f.columns == ["s1", "t", *F2_COLUMNS] and len(f) == 4
    row = {r["t"]: r for r in f.iter_rows(named=True)}
    assert row[0]["nm_true_extra"] == 0 and row[0]["nm_extra_idf"] > 0          # "beter" is a typo of "better"
    assert row[1]["nm_true_extra"] == 1 and row[1]["g_extra_support"] == 1     # "bakery" also appears in S2-2, self excluded
    assert row[1]["g_same_house"] == 1 and row[2]["g_same_house"] == 1        # the decoy house 3447 is shared
    assert row[3]["hs_in_t"] == -1 and row[3]["h_lev"] == -1 and row[3]["g_same_addr"] == -1
    assert decoy_features(pairs, s1, targets, ctx, pl.Series([95.0, 95.0, 95.0, -1.0]), threads=2).equals(f)
