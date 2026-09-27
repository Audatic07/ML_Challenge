# v5.1 local ensemble results

The frozen LightGBM + XGBoost ensemble achieved **0.97647081 macro per-S1 F0.5** on
15,000 reserved second-stage development queries. The original model scored
**0.96776538** on the same queries: **+0.8705 percentage points**.

**Neither 0.98 nor 0.99 was achieved.** This is not ordinary classification accuracy,
an untouched competition audit, a test-set score, or a leaderboard result. These
queries were originally part of v5.1's tuning set. The original audit remained closed.
No France score or completed final test submission is claimed.

| Evidence | Value |
|---|---:|
| Separate model-selection queries | 10,000 |
| Model-selection F0.5 | 0.97681871 |
| Reserved-check queries | 15,000 |
| Reserved-check F0.5 | 0.97647081 |
| Score bootstrap 95% interval | 0.97502347–0.97800989 |
| Paired gain bootstrap 95% interval | 0.00760073–0.00997139 |
| Candidate oracle ceiling on check | 0.99486104 |

See [the detailed report](RESULTS.md), [machine-readable results](metadata/final_check.json),
[model comparisons](metadata/calibration.json), and [the frozen model/policy hashes](metadata/champion_frozen.json).
The oracle ceiling assumes perfect use of retrieved truth; it is not an achieved model score.

## What is published

- `source/`: byte-for-byte copies of the measured experiment's Python source and four tests.
- `metadata/`: aggregate scores, parameters, split sizes, checksums, provenance and integrity checks.
- `published_manifest.json`: checksums of the published experiment source and evidence.
- `restore_experiment.py`: restores source and protocol into the repository's ignored `artifacts/` directory.

The original scripts were written for an isolated local artifact directory. **Restore
them before running them; do not run training directly from `source/`.** The source
snapshot and its original metadata retain the historical local experiment paths and
status. Publication does not change the recorded experiment or rerun its evaluation.

Datasets, query-level predictions, feature shards, model weights, private archives and
cloud credentials are not included. The trained ensemble remains in the original
private local artifact directory, with its hashes recorded here. The existing released
submission is not replaced by this branch.

## Reproduction

Use the package versions recorded in [environment.json](metadata/environment.json).
The inherited normalizer additionally requires `Unidecode`; pytest is needed for tests.
No external dataset or pretrained weights are needed or permitted by this experiment.

From the repository root:

```text
python experiments/v51_local_20260927/restore_experiment.py
```

This creates `artifacts/v51_improve_20260927_0820/` and refuses to overwrite an existing
experiment. It does not start training. Supply these private inputs locally:

1. The provided TSVs under `student_resource/dataset/train/`.
2. The verified v5.1 export under `artifacts/v51_received_20260927/`, including
   `plan.json`, `model/model_manifest.json`, and `model/tune_scores.parquet`.
3. For `final_check.py`'s historical comparison, the original
   `baseline_reproduced_per_query.parquet` in that export directory. It can be recreated
   with the baseline reproduction script in `source/reproduce_baseline.py`, copied into
   the private export directory and run there.

Match dataset and model hashes against the supplied metadata before comparing results.
Then run, in order:

```text
python artifacts/v51_improve_20260927_0820/prepare.py
python artifacts/v51_improve_20260927_0820/train.py lightgbm
python artifacts/v51_improve_20260927_0820/train.py xgboost
python artifacts/v51_improve_20260927_0820/reverse.py
python artifacts/v51_improve_20260927_0820/train.py lightgbm --tag lightgbm_reverse --features features_reverse
python artifacts/v51_improve_20260927_0820/train.py xgboost --tag xgboost_reverse --features features_reverse
python artifacts/v51_improve_20260927_0820/calibrate.py --models lightgbm xgboost lightgbm_reverse xgboost_reverse
python artifacts/v51_improve_20260927_0820/final_check.py
```

The split is fixed: 65,000 fit, 10,000 early-stop, 10,000 selection and 15,000 check
queries, seed 27092026. Check labels must not be reused for new tuning. The final-check
script refuses to overwrite a completed check. A reproduction is not a fresh evaluation
partition; further development needs a separately declared evaluation protocol.

Tests require no business records or model weights:

```text
python -B -m pytest experiments/v51_local_20260927/source/test_selection.py -q -p no:cacheprovider
```

## Method and limits

The original full-catalog candidate scores are reduced to top 16 per query with
probability >= 0.0001. Features compare each query with candidates and candidates
with their siblings. Reverse competition uses all 2,206,821 supplied S1 records,
normalized name/address keys and similarity comparisons, without label lookups.
The two 156-feature classifiers are averaged, then an expected-F0.5 set selector uses
logit bias -0.3 and temperature 1.0. Its independence assumptions are approximations;
all reported performance is scored against complete actual query truth.

The supplied export lacked full-test score shards. Full-test feature generation,
inference, global assignment and strict TSV validation remain required for a release.
Do not compare this reserved 15k development score directly with another run's full
100k tuning score or audit score as if the partitions were identical.
