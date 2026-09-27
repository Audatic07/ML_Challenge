import math

import polars as pl

from src.v7_population import F3_COLUMNS, MAX_POSTINGS, build_index, index_digest, rival_features


def records(rows):
    return pl.DataFrame(rows, schema=["entity_id", "business_name", "business_address", "country"], orient="row")


def test_rivals_are_coherent_censored_and_never_self():
    s1 = records([
        ("S1-0", "Rana Better Engineering", "3445 Refugee Road, Columbus, OH", "US"),     # q
        ("S1-1", "Rana Better Engineering", "99 Other Street, Dayton, OH", "US"),         # name-only rival
        ("S1-2", "Quinta Bakery Works", "3445 Refugee Road, Columbus, OH", "US"),         # address-only rival
        ("S1-3", "Lonely Widget Makers", "", "US"),                                       # no address
        *[(f"S1-{i}", "Common Store Name", f"{i} Some Place, Town, OH", "US") for i in range(4, 5 + MAX_POSTINGS)],
    ])
    targets = records([
        ("S2-0", "Rana Better Engineering", "3445 Refugee Rd, Columbus, Ohio", "US"),
        ("S2-1", "Lonely Widget Makers", "", "US"),
        ("S2-2", "Common Store Name", "1 Unrelated Ave, City, OH", "US"),
    ]).with_columns(pl.lit(2, pl.UInt8).alias("src"))
    index = build_index(s1)
    pairs = pl.DataFrame({"s1": [0, 3, 4], "t": [0, 1, 2]}, schema={"s1": pl.UInt32, "t": pl.UInt32})
    f = rival_features(pairs, s1, targets, index, threads=1)
    assert f.columns == ["s1", "t", *F3_COLUMNS] and len(f) == 3
    row = {r["t"]: r for r in f.iter_rows(named=True)}
    t0 = row[0]
    # Name rival S1-1 and address rival S1-2 are separate witnesses; the joint rival is one real record.
    assert t0["rN_present"] == 1 and t0["rA_present"] == 1 and t0["rN_n_ratio"] >= 0.99 and t0["rA_a_ratio"] >= 0.99
    assert t0["rN_is_J"] + t0["rA_is_J"] <= 1
    assert t0["rJ_n_ratio"] < 0.99 or t0["rJ_a_ratio"] < 0.99  # no rival has both S1-1's name and S1-2's address
    # q itself is never its own rival: its own similarity is not reused as a rival.
    assert t0["o_n_ratio"] >= 0.99 and not (t0["rJ_n_ratio"] >= 0.99 and t0["rJ_a_ratio"] >= 0.99)
    t1 = row[1]  # missing address: NaN, never agreement
    assert math.isnan(t1["o_a_ratio"]) and math.isnan(t1["o_joint"]) and t1["t_addr_missing"] == 1 and t1["q_addr_missing"] == 1
    t2 = row[2]  # the shared name has more than MAX_POSTINGS S1: censored, and it adds no rival
    assert t2["t_overflow"] == 1 and t2["rN_present"] == 0
    # Deterministic and label-free: the same inputs give the same features and index digest.
    assert rival_features(pairs, s1, targets, build_index(s1), threads=2).equals(f, null_equal=True)
    assert index_digest(index) == index_digest(build_index(s1))
