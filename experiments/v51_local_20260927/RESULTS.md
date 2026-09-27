# Local v5.1 improvement: measured results

**Reserved-check macro per-S1 F0.5: 0.97647081.**
The original model scored 0.96776538 on the same
15,000 queries under the same within-partition target assignment rule.
Paired improvement: 0.00870543 (0.8705 percentage points).
Target 0.98 met: False. Target 0.99 met: False.

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
- Selected **ensemble:lightgbm_reverse:xgboost_reverse**, policy **expected_f05**, on a separate
  10,000-query calibration partition, then froze that choice before the reserved check.

Only the supplied train_source1/2/3.tsv and train_ground_truth.tsv, and the supplied
v5.1 export, were used. No external records, identity lookups, labels or pretrained
model weights were added. The experiment ran locally. This branch publishes its source and aggregate evidence; data, predictions and trained weights remain private.

## Evaluation

| Item | Result |
|---|---:|
| Original verified v5.1 score on all original tuning queries | 0.96793547 |
| Original model on the 15,000 reserved check queries | 0.96776538 |
| Selected model on those same check queries | 0.97647081 |
| Candidate oracle ceiling on check | 0.99486104 |
| Remaining matching/selection gap | 0.01839023 |
| Check candidate pairs | 155,381 |
| Check predicted pairs | 48,929 |
| Check singleton queries | 844 |
| Check queries with no survivors | 7 |

Score bootstrap 95% interval: [0.9750234728943488, 0.9780098915037999].
Paired-gain bootstrap 95% interval: [0.007600732307314507, 0.009971394791979506].
Bootstrap resamples queries; it does not account for country shift or earlier reuse of
the original v5.1 tuning partition. Countries: [{"country": "US", "queries": 8978, "macro_f05": 0.9819551673743789}, {"country": "India", "queries": 6022, "macro_f05": 0.968294365268979}].

The candidate oracle uses ground truth only to calculate an upper bound. It is never
reported as model performance and its predictions are never used as model features.

## Reproduction and files

Start with [README.md](README.md) to restore the archived source into an ignored local work directory. In this public folder, aggregate JSON files are under `metadata/`; private Parquet and model files are not included.

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
