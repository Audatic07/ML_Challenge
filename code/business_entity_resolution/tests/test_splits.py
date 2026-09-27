import numpy as np
import polars as pl
import pytest

from src.splits import make_split, split_manifest
from src import train


def small_split(train_sample=20, val_sample=5):
    return make_split(100, train_sample, val_sample, validation_start=20, validation_size=10)


def test_canonical_validation_matches_v1_and_does_not_move_with_larger_training():
    original = make_split(600_000, 200_000, 100_000)
    larger = make_split(600_000, 400_000, 100_000)
    perm = np.random.default_rng(42).permutation(600_000)
    np.testing.assert_array_equal(original.train, np.sort(perm[:200_000]))
    np.testing.assert_array_equal(original.validation, np.sort(perm[200_000:300_000]))
    np.testing.assert_array_equal(original.validation, larger.validation)
    assert not np.intersect1d(larger.train, original.validation).size
    assert not np.intersect1d(larger.train, larger.audit).size


def test_pilots_reserve_entire_validation_and_use_nested_samples():
    pilot = small_split(10, 3)
    larger = small_split(40, 8)
    assert set(pilot.train) <= set(larger.train)
    assert set(pilot.validation) <= set(larger.validation)
    assert len(pilot.reserved_validation) == 10
    assert not np.intersect1d(larger.train, pilot.held_out).size


def test_manifest_is_deterministic_and_pins_source_id_order():
    split = small_split()
    ids = [f"s1-{i}" for i in range(100)]
    first = split_manifest(split, ids)
    assert first == split_manifest(split, ids)
    assert first["sha256"] != split_manifest(split, reversed(ids))["sha256"]
    assert first["audit"]["count"] == 10
    with pytest.raises(ValueError, match="ID count"):
        split_manifest(split, ids[:-1])


@pytest.mark.parametrize("kwargs", [{"train_sample": 81}, {"val_sample": 11}, {"val_sample": 0}])
def test_invalid_sample_sizes_fail_instead_of_silently_changing_split(kwargs):
    with pytest.raises(ValueError):
        small_split(**kwargs)


def test_reserved_targets_cannot_be_used_as_fit_negatives(monkeypatch):
    truth = pl.DataFrame({"s1": [0, 1], "t": [10, 11]},
                         schema={"s1": pl.UInt32, "t": pl.UInt32})
    excluded = train.held_out_targets(truth, np.array([0]), np.array([1]))
    assert excluded["t"].to_list() == [11]

    class FixedBlocker:
        def candidates(self, rows):
            return pl.DataFrame({"s1": [0, 0, 0], "t": [10, 11, 12]},
                                schema={"s1": pl.UInt32, "t": pl.UInt32})

    monkeypatch.setattr(train, "build", lambda cand, *_: cand)
    fit = train.labelled_pairs(FixedBlocker(), np.array([0]), None, None, truth, excluded)
    assert fit.select("t", "y").sort("t").rows() == [(10, 1), (12, 0)]
    # Validation retrieval retains the entire target catalog.
    validation = train.labelled_pairs(FixedBlocker(), np.array([0]), None, None, truth)
    assert set(validation["t"]) == {10, 11, 12}


def test_shared_labelled_target_requires_business_group_split():
    truth = pl.DataFrame({"s1": [0, 1], "t": [10, 10]},
                         schema={"s1": pl.UInt32, "t": pl.UInt32})
    with pytest.raises(ValueError, match="business-group"):
        train.held_out_targets(truth, np.array([0]), np.array([1]))
