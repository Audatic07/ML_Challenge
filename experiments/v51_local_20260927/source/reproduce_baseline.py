"""Reproduce exported development metrics using supplied labels; never read audit labels into evaluation."""
import os
os.environ.setdefault('POLARS_MAX_THREADS', '2')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parents[1] / 'student_resource' / 'dataset' / 'train'

def collect(lf):
    return lf.collect(engine='streaming')

def evaluate(base, truth, predictions):
    assert not predictions.select('s1', 'target_id').is_duplicated().any()
    pred = predictions.group_by('s1').len().rename({'len': 'pred_count'})
    hit = predictions.join(truth, on=['s1', 'target_id']).group_by('s1').len().rename({'len': 'tp_count'})
    per = base.join(pred, on='s1', how='left').join(hit, on='s1', how='left').fill_null(0)
    g, p, tp = [per[c].to_numpy().astype(np.float64) for c in ['truth_count', 'pred_count', 'tp_count']]
    f = np.zeros(len(g), dtype=np.float64)
    positive = g > 0
    f[positive] = 5 * tp[positive] / (4 * p[positive] + g[positive])
    f[~positive] = (p[~positive] == 0).astype(np.float64)
    return per.with_columns(pl.Series('reproduced_f', f))

def main():
    started = time.monotonic()
    plan = json.loads((ROOT / 'plan.json').read_text())
    model = json.loads((ROOT / 'model/model_manifest.json').read_text())
    assert hashlib.sha256(json.dumps(plan['tune_rows'], sort_keys=True).encode()).hexdigest() == plan['tune_sha']
    queries = pl.DataFrame({'s1': pl.Series(plan['tune_rows'], dtype=pl.UInt32), 'country': plan['tune_country']})
    assert queries['s1'].n_unique() == len(queries) == 100000
    ids = collect(pl.scan_csv(DATA / 'train_source1.tsv', separator='\t', quote_char=None, infer_schema=False)
                  .with_row_index('s1').select('s1', pl.col('entity_id').alias('source1_entity_id'))
                  .join(queries.lazy().select('s1'), on='s1', how='semi'))
    labels = collect(pl.scan_csv(DATA / 'train_ground_truth.tsv', separator='\t', quote_char=None, infer_schema=False)
                     .join(ids.lazy(), on='source1_entity_id', how='inner').select('s1', 'matched_entity_ids'))
    assert len(labels) == len(queries) and labels['s1'].n_unique() == len(queries)
    truth = (labels.with_columns(pl.col('matched_entity_ids').fill_null('').str.split(','))
             .explode('matched_entity_ids').filter(pl.col('matched_entity_ids') != '')
             .select('s1', pl.col('matched_entity_ids').alias('target_id')))
    assert not truth.is_duplicated().any()
    base = queries.join(truth.group_by('s1').len().rename({'len': 'truth_count'}), on='s1', how='left').fill_null(0)
    print('Loaded tuning labels:', len(queries), 'queries;', len(truth), 'links', flush=True)
    scores = pl.scan_parquet(ROOT / 'model/tune_scores.parquet')
    score_stats = collect(scores.select(pl.len().alias('pairs'), pl.col('s1').n_unique().alias('queries'),
        ((~pl.col('p').is_finite()) | pl.col('p').is_null()).sum().alias('invalid_scores'))).to_dicts()[0]
    assert score_stats['invalid_scores'] == 0
    threshold = model['threshold']
    # Thresholding before target assignment is equivalent at this fixed threshold:
    # any excluded score is below every surviving competitor's score.
    above = collect(scores.filter(pl.col('p') >= threshold))
    assert above.join(queries.select('s1'), on='s1', how='anti').is_empty()
    predictions = (above.sort(['t', 'p', 's1'], descending=[False, True, False])
                   .unique(subset=['t'], keep='first', maintain_order=True).select('s1', 'target_id'))
    per = evaluate(base, truth, predictions)
    print('Reproduced predictions:', len(predictions), 'macro F0.5:', per['reproduced_f'].mean(), flush=True)
    retrieved_truth = collect(scores.select('s1', 'target_id').join(truth.lazy(), on=['s1', 'target_id'], how='semi'))
    oracle = evaluate(base, truth, retrieved_truth)
    saved = pl.read_parquet(ROOT / 'model/tune_per_query.parquet')
    comparison = per.join(saved.select('s1', 'f'), on='s1')
    max_delta = comparison.select((pl.col('reproduced_f') - pl.col('f')).abs().max()).item()
    actual = float(per['reproduced_f'].mean())
    upper = float(oracle['reproduced_f'].mean())
    report = {'partition': 'existing 100000-query tuning set; not audit or independent test',
        'model_sha256': model['model_sha256'], 'plan_sha': plan['plan_sha'], 'tune_sha': plan['tune_sha'],
        'threshold': threshold, 'queries': len(queries), 'truth_pairs': len(truth), **score_stats,
        'predicted_pairs': len(predictions), 'macro_f05': actual, 'oracle_u': upper,
        'matching_gap': upper - actual, 'link_recall': len(retrieved_truth) / len(truth),
        'singleton_queries': per.filter(pl.col('truth_count') == 0).height,
        'singleton_f05': per.filter(pl.col('truth_count') == 0)['reproduced_f'].mean(),
        'max_saved_per_query_difference': max_delta,
        'by_country': per.group_by('country').agg(pl.len().alias('queries'), pl.col('reproduced_f').mean().alias('macro_f05')).to_dicts(),
        'raw_label_counts_match_saved': per.join(saved.select('s1', 'g'), on='s1').select((pl.col('truth_count') == pl.col('g')).all()).item(),
        'audit_evaluated': False, 'new_model_trained': False,
        'saved_count_column_note': 'Exported g/p/tp describe oracle predictions; actual decisions reconstructed here.',
        'seconds': time.monotonic() - started}
    assert max_delta < 1e-12
    assert abs(actual - model['metrics']['macro_f05']) < 1e-12
    assert abs(upper - model['metrics']['oracle_u']) < 1e-12
    assert report['raw_label_counts_match_saved']
    per.write_parquet(ROOT / 'baseline_reproduced_per_query.parquet')
    (ROOT / 'baseline_reproduction.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

if __name__ == '__main__':
    main()
