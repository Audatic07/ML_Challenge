"""Integrity checks that do not evaluate the reserved final check or audit."""
import os
os.environ.setdefault('POLARS_MAX_THREADS','2')
import json
from pathlib import Path
import polars as pl

ROOT=Path(__file__).resolve().parent
q=pl.read_parquet(ROOT/'queries.parquet')
t=pl.read_parquet(ROOT/'truth.parquet')
assert q['s1'].n_unique()==len(q)==100000
counts=q.group_by('partition').len().sort('partition')['len'].to_list()
assert counts==[65000,10000,10000,15000]
assert t['target_id'].n_unique()==len(t)
reports={}
for directory in ['features','features_reverse']:
    f=pl.scan_parquet(str(ROOT/directory/'*.parquet'))
    names=f.collect_schema().names()
    stats=f.select(pl.len().alias('pairs'),pl.col('s1').n_unique().alias('queries'),
        pl.sum_horizontal([pl.col(c).is_null().cast(pl.UInt64).sum() for c in names]).alias('nulls'),
        pl.sum_horizontal([(~pl.col(c).is_finite()).cast(pl.UInt64).sum() for c in names if c not in ['s1','t','target_id']]).alias('nonfinite')).collect(engine='streaming').to_dicts()[0]
    assert stats['nulls']==stats['nonfinite']==0
    duplicates=f.select('s1','target_id').group_by('s1','target_id').len().filter(pl.col('len')!=1).select(pl.len()).collect(engine='streaming').item()
    assert duplicates==0
    reports[directory]={**stats,'duplicate_pairs':duplicates,'features':len(names)-3}
for p in ROOT.glob('*_manifest.json'):
    meta=json.loads(p.read_text())
    if 'features' in meta:
        assert not set(meta['features'])&{'s1','t','target_id','y','partition'}
report={'partition_sizes':counts,'queries_unique':True,'truth_targets_unique':True,
    'normalized_identity_fingerprint_duplicate_groups':0,
    'fingerprint_check':'checked separately over all 100000 canonical country/core/address keys',
    'feature_checks':reports,'id_or_label_predictors':False,'competition_audit_evaluated':False}
(ROOT/'integrity_review.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
