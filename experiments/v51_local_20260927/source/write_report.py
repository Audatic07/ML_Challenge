import hashlib,json,zipfile,importlib.metadata,platform
from pathlib import Path

ROOT=Path(__file__).resolve().parent
(ROOT/'environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':{name:importlib.metadata.version(name) for name in ['polars','numpy','lightgbm','xgboost','rapidfuzz','pytest']},'hardware':'Local Windows; 4 physical / 8 logical CPU; training limited to 3 threads','model_licenses':{'lightgbm':'MIT; verified installed license text','xgboost':importlib.metadata.distribution('xgboost').metadata.get('License')},'pretrained_weights_added':False},indent=2))
report=json.loads((ROOT/'final_check.json').read_text())
cal=json.loads((ROOT/'calibration.json').read_text())
champion=report['champion']
def components(name):
    if name.startswith('ensemble:'):return name.split(':')[1:]
    if '_rawblend_' in name:return [name.rsplit('_rawblend_',1)[0]]
    return [] if name=='baseline' else [name]
chosen=components(champion['name'])
text=f'''# Local v5.1 improvement: measured results

**Reserved-check macro per-S1 F0.5: {report['new_macro_f05']:.8f}.**
The original model scored {report['baseline_same_queries_macro_f05']:.8f} on the same
15,000 queries under the same within-partition target assignment rule.
Paired improvement: {report['paired_gain']:.8f} ({100*report['paired_gain']:.4f} percentage points).
Target 0.98 met: {report['target_098_met']}. Target 0.99 met: {report['target_099_met']}.

This is the competition's macro F0.5, not binary classification accuracy. Each query
counts once, including singletons and queries with no candidates. The check queries
were held out from all second-stage fitting, early stopping and model/policy selection.
They belonged to the original v5.1 tuning set, so this is **development evidence**, not
an untouched competition audit or a leaderboard result. The competition audit stayed closed.
No score is asserted for France or for the unlabeled test set.

## What changed

- Retained up to 16 candidates per query with original v5.1 probability >= 0.0001.
- Rebuilt 145 features from the supplied records and saved model scores: query/candidate
  similarities, directional token/number agreement, sibling agreement, and competition.
- Added 11 reverse-retrieval features from the complete 2,206,821-row supplied S1
  population. These use normalized name/address keys and similarities, never labels.
- Compared LightGBM, XGBoost, blends with original scores, and model ensembles.
- Compared fixed thresholds with expected-F0.5 set selection, including predicting no
  matches. Its Bernoulli-independence and no-positive-mass-outside-survivors assumptions
  are approximations; the reported metric is measured against actual complete truth.
- Selected **{champion['name']}**, policy **{champion['policy']}**, on a separate
  10,000-query calibration partition, then froze that choice before the reserved check.

Only the supplied train_source1/2/3.tsv and train_ground_truth.tsv, and the supplied
v5.1 export, were used. No external records, identity lookups, labels or pretrained
model weights were added. All work and artifacts stayed local; no commits or pushes.

## Evaluation

| Item | Result |
|---|---:|
| Original verified v5.1 score on all original tuning queries | 0.96793547 |
| Original model on the 15,000 reserved check queries | {report['baseline_same_queries_macro_f05']:.8f} |
| Selected model on those same check queries | {report['new_macro_f05']:.8f} |
| Candidate oracle ceiling on check | {report['candidate_oracle_u']:.8f} |
| Remaining matching/selection gap | {report['matching_gap']:.8f} |
| Check candidate pairs | {report['candidate_pairs']:,} |
| Check predicted pairs | {report['predicted_pairs']:,} |
| Check singleton queries | {report['singleton_queries']:,} |
| Check queries with no survivors | {report['zero_candidate_queries']} |

Score bootstrap 95% interval: {report['score_bootstrap_95_interval']}.
Paired-gain bootstrap 95% interval: {report['gain_bootstrap_95_interval']}.
Bootstrap resamples queries; it does not account for country shift or earlier reuse of
the original v5.1 tuning partition. Countries: {json.dumps(report['by_country'])}.

The candidate oracle uses ground truth only to calculate an upper bound. It is never
reported as model performance and its predictions are never used as model features.

## Reproduction and files

The scripts use paths relative to this experiment folder and the existing workspace.
Run using the workspace's existing `.venv/Scripts/python.exe`, from the workspace root:

```text
python -B artifacts/v51_improve_20260927_0820/prepare.py
python -B artifacts/v51_improve_20260927_0820/train.py lightgbm
python -B artifacts/v51_improve_20260927_0820/train.py xgboost
python -B artifacts/v51_improve_20260927_0820/reverse.py
python -B artifacts/v51_improve_20260927_0820/train.py lightgbm --tag lightgbm_reverse --features features_reverse
python -B artifacts/v51_improve_20260927_0820/train.py xgboost --tag xgboost_reverse --features features_reverse
```

Use `calibrate.py --models ...` with the model names in calibration.json to reproduce
model selection. `final_check.py` is deliberately guarded against reusing its saved
check for further tuning. Preserve the frozen model/policy; use a separate output
directory for an authorized reproducibility rerun.

- `final_check.json`: exact measured results and caveats.
- `champion_frozen.json`: chosen policy and model hashes before the check.
- `calibration.json`: model-selection comparisons.
- `*_manifest.json`: features, parameters, iterations and measured training time.
- `input_dataset_checksums.json`: supplied input file hashes.
- `integrity_review.json`: split and feature-integrity checks.
- `test_selection.py`: four passing metric/selection/edge-case tests.
- `final_check_per_query.parquet`: private per-query comparison for verification.

This is a trained local improvement artifact, **not a completed test submission**.
The export lacks the 88 full-union test score shards. Full-test feature rebuilding,
inference, global target assignment and strict TSV validation must be completed before
it can replace the released submission. Models and private data must not be committed
to a public repository merely because this report exists.
'''
(ROOT/'RESULTS.md').write_text(text,encoding='utf-8')
files=[p for p in [*ROOT.glob('*.py'),*ROOT.glob('*.json'),*ROOT.glob('LICENSE_*.txt'),ROOT/'RESULTS.md',*(ROOT/'stage2').glob('*.py')] if p.name!='handoff_checksums.json']
for name in chosen:
    meta=json.loads((ROOT/f'{name}_manifest.json').read_text())
    files.append(ROOT/(name+('.txt' if meta['model']=='lightgbm' else '.ubj')))
checks={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
(ROOT/'handoff_checksums.json').write_text(json.dumps(checks,indent=2))
files.append(ROOT/'handoff_checksums.json')
with zipfile.ZipFile(ROOT/'selected_model_and_code.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(set(files)):z.write(p,p.relative_to(ROOT))
print('Report:',ROOT/'RESULTS.md')
print('Selected model package:',ROOT/'selected_model_and_code.zip')
