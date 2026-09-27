from copy import deepcopy

import pytest

from src import predict


def metadata():
    return {"top_k": predict.TOP_K, "key_types": {key: list(value) for key, value in predict.KEY_TYPES.items()},
            "features": list(predict.FEATURES),
            "retrieval_version": getattr(predict.config, "RETRIEVAL_VERSION", "2"),
            "retrieval_rank": getattr(predict.config, "RETRIEVAL_RANK", "weight")}


def test_saved_json_settings_match_active_policy():
    predict.validate_retrieval_config(metadata())


@pytest.mark.parametrize("change", ["top_k", "cap", "weight", "missing_key", "missing_policy"])
def test_changed_or_missing_retrieval_policy_requires_retraining(change):
    meta = deepcopy(metadata())
    key = next(iter(meta["key_types"]))
    if change == "top_k":
        meta["top_k"] += 1
    elif change == "cap":
        meta["key_types"][key][1] += 1
    elif change == "weight":
        meta["key_types"][key][0] += 1
    elif change == "missing_key":
        del meta["key_types"][key]
    else:
        del meta["key_types"]
    with pytest.raises(ValueError, match="retrain"):
        predict.validate_retrieval_config(meta)


@pytest.mark.parametrize("change", ["version", "rank", "missing_rank"])
def test_changed_version_or_rank_requires_retraining(change):
    meta = metadata()
    if change == "version":
        meta["retrieval_version"] = "different-version"
    elif change == "rank":
        meta["retrieval_rank"] = "text"
        monkey_rank = getattr(predict.config, "RETRIEVAL_RANK", "weight")
        if monkey_rank == "text":
            meta["retrieval_rank"] = "weight"
    else:
        del meta["retrieval_rank"]
    with pytest.raises(ValueError, match="retrain"):
        predict.validate_retrieval_config(meta)


def test_original_incumbent_without_version_remains_usable(monkeypatch):
    monkeypatch.setattr(predict.config, "BLOCKING_PROFILE", "v1", raising=False)
    monkeypatch.setattr(predict.config, "RETRIEVAL_RANK", "weight", raising=False)
    monkeypatch.setattr(predict, "KEY_TYPES", deepcopy(predict.LEGACY_V1_KEYS))
    monkeypatch.setattr(predict, "TOP_K", 30)
    monkeypatch.setattr(predict, "FEATURES", list(predict.LEGACY_V1_FEATURES))
    meta = metadata()
    del meta["retrieval_version"]
    del meta["retrieval_rank"]
    predict.validate_retrieval_config(meta)


@pytest.mark.parametrize("changed_policy", ["rescue", "text", "top_k", "features"])
def test_unversioned_metadata_rejected_outside_original_v1(monkeypatch, changed_policy):
    monkeypatch.setattr(predict.config, "BLOCKING_PROFILE", "v1", raising=False)
    monkeypatch.setattr(predict.config, "RETRIEVAL_RANK", "weight", raising=False)
    monkeypatch.setattr(predict, "KEY_TYPES", deepcopy(predict.LEGACY_V1_KEYS))
    monkeypatch.setattr(predict, "TOP_K", 30)
    monkeypatch.setattr(predict, "FEATURES", list(predict.LEGACY_V1_FEATURES))
    if changed_policy == "rescue":
        monkeypatch.setattr(predict.config, "BLOCKING_PROFILE", "rescue")
        monkeypatch.setattr(predict, "KEY_TYPES", {**predict.KEY_TYPES, "H": [2, 200], "I": [1, 100]})
    elif changed_policy == "text":
        monkeypatch.setattr(predict.config, "RETRIEVAL_RANK", "text")
    elif changed_policy == "top_k":
        monkeypatch.setattr(predict, "TOP_K", 60)
    else:
        monkeypatch.setattr(predict, "FEATURES", [*predict.FEATURES, "new_feature"])
    meta = metadata()  # All saved settings match active settings, but version is absent.
    del meta["retrieval_version"]
    with pytest.raises(ValueError, match="original v1"):
        predict.validate_retrieval_config(meta)
