import json
from pathlib import Path
import numpy as np
import polars as pl
from src.v4_retrieval import SparseCatalog, normalized, digest, sha256
from src.v4_features import build_features, FEATURES


def records(names, addresses, offset=0, source=2):
    return normalized(pl.DataFrame({"entity_id": [f"S{source}-{i+offset}" for i in range(len(names))],
        "business_name": names, "business_address": addresses, "country": ["France"] * len(names)})
        .with_row_index("row", offset=offset).with_columns(pl.lit(source, pl.UInt8).alias("src")))


def catalog(tmp_path, frames):
    parts = []
    for i, frame in enumerate(frames):
        path = tmp_path / f"part{i}.parquet"
        frame.write_parquet(path)
        parts.append({"file": path.name, "offset": int(frame['row'][0]), "rows": len(frame),
                      "source": int(frame['src'][0]), "sha256": sha256(path)})
    manifest = {"parts": parts, "rows": sum(len(x) for x in frames)}
    manifest["sha256"] = digest(manifest)
    return manifest


def test_independent_address_retrieval_and_full_query_features(tmp_path):
    queries = records(["Amber Studio", "Unfindable", ""], ["17 Cedar Willow", "", ""], offset=500, source=1)
    targets = [records(["Zinc Workshop", "Amber Studio"], ["17 Cedar Willow", "88 Unrelated Street"]),
               records(["Amber Studio"], ["17 Cedar Willow"], offset=2, source=3)]
    manifest = catalog(tmp_path, targets)
    index = SparseCatalog(tmp_path, manifest, dimensions=2048, top_k=2, max_df=1, workers=1)
    result = index.retrieve(queries, tmp_path / "retrieved")
    assert (500, 0) in result.select('s1', 't').rows()  # no common name
    assert 502 not in result['s1']  # two empty fields are never evidence
    again = index.retrieve(queries, tmp_path / "retrieved")
    assert result.equals(again)
    features = build_features(queries, result, tmp_path, manifest, tmp_path / "features", workers=1)
    assert all(x in features.columns for x in FEATURES)
    assert features.filter(pl.col('s1') == 500)['n_cand'].unique().to_list() == [3.0]
    assert features.select('s1','t').n_unique() == len(features)


def test_topk_merges_all_target_shards_and_protects_source_quota(tmp_path):
    queries = records(["Amber Studio"], ["17 Cedar Willow"], source=1)
    frames = [records(["Amber Studios"], ["17 Cedar Willow"]),
              records(["Amber Studio"], ["17 Cedar Willow"], offset=1),
              records(["Amber Studio"], ["17 Cedar Willow"], offset=2, source=3)]
    manifest = catalog(tmp_path, frames)
    index = SparseCatalog(tmp_path, manifest, dimensions=2048, top_k=1, max_df=1, workers=1)
    out = index.retrieve(queries, tmp_path / 'out')
    assert 1 in out['t'] and 2 in out['t']
    assert out.select('s1','t').n_unique() == len(out)


def test_folded_view_preserves_original(tmp_path):
    x = records(['Laxmi Management'], ['Maharashtra'])
    y = records(['Lkssmii Mainejmentt'], ['Mhaaraassttr'])
    assert x['name_fold'][0] == y['name_fold'][0]
    assert x['address_fold'][0] == y['address_fold'][0]
    assert x['name_norm'][0] != y['name_norm'][0]


def test_training_saves_model_and_metadata(tmp_path):
    from src.v4_train import train_model
    q = records(['Amber', 'Zinc', 'Nickel'], ['17 Cedar', '82 Willow', '93 Birch'], source=1)
    t = records(['Amber', 'Unrelated', 'Zinc', 'Nobody', 'Nickel', 'Other'],
                ['17 Cedar', '900 Elm', '82 Willow', '90 Elm', '93 Birch', '29 Elm'])
    m = catalog(tmp_path, [t])
    retrieval = SparseCatalog(tmp_path, m, dimensions=2048, top_k=6, max_df=1, workers=1)
    candidates = retrieval.retrieve(q, tmp_path/'retrieve')
    frame = build_features(q, candidates, tmp_path, m, tmp_path/'features', 1)
    truth = pl.DataFrame({'s1':[0,1,2], 'target_id':['S2-0','S2-2','S2-4']},
                         schema={'s1':pl.UInt32,'target_id':pl.String})
    excluded = pl.DataFrame({'target_id':['S2-2','S2-4']})
    result = train_model(frame,truth,excluded,np.array([0],dtype=np.uint32),
        np.array([1],dtype=np.uint32),np.array([2],dtype=np.uint32),q,tmp_path,2,1,{})
    assert (tmp_path/'model.txt').is_file()
    assert (tmp_path/'model_manifest.json').is_file()
    assert result['audit_opened'] is False
    assert result['metrics']['queries'] == 1


def test_v4_samples_are_nested_disjoint_and_keep_legacy_validation():
    from src.v4_train import partitions
    a = partitions(600000, 30000, 2000, 5000)
    b = partitions(600000, 300000, 5000, 10000)
    assert set(a[1]) <= set(b[1])
    assert set(a[2]) <= set(b[2])
    assert set(a[3]) <= set(b[3])
    assert not set(b[1]) & set(b[4])
    assert not set(b[2]) & set(b[3])
    assert set(a[3]) == set(np.random.default_rng(42).permutation(600000)[200000:205000])
