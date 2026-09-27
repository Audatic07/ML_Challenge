# Results: v3, v6 and the v7 change

Macro F0.5 per Source 1 entity on 20,000 validation queries (prefix of the canonical
interval `perm[200000:300000]`, seed 42). Singletons and retrieval misses count. The
audit partition was never opened. Numbers come from `results/*/result.json` and logs.

| Version | Validation F0.5 | Check half (untouched) | Candidate oracle U | Candidates / S1 |
|---|---|---|---|---|
| v3 | 0.9637 (India 0.9576, US 0.9678) | - | 0.9817 | 35 |
| v6 | 0.9742 (India 0.9688, US 0.9777) | 0.9737 | 0.9904 | 37 |
| v6 + per-country thresholds | - | 0.9741 | 0.9904 | 37 |
| v7 (v6 + digit-typo fix) | not yet measured | - | - | - |

- v3 test submission (1,732,544 S1) passed the official validator with `--check-ids`.
  Stats: `results/v3/predict_stats.json`.
- v3 reproduced exactly in Google Colab (retrained: 763 rounds, 0.9637).
- Retrieval misses: v3 lost 4.94% of true validation pairs, v6 loses 2.66%.
  10.5% of v6's misses have look-alike digits in the target name. Measured map:
  0->o, 1->l, 5->s, 6->g, 8->b (`code/business_entity_resolution/v6_misses.py`).
- Model family is not the bottleneck: LightGBM, XGBoost and their blend are within 0.0005
  (`src/stack_v6.py`).

## v7 change (in `src/`)

- `text_norm.fix_digit_letters`, used by `derive`: `mi1ler` -> `miller`, only inside words
  that mix letters and digits. Tests: `tests/test_digit_letters.py` (42 tests pass).
- `prepare_v7.py`: re-derives the v3 caches with the fix.
- `thresholds_v7.py`: per-country thresholds picked on the tune half, kept only if the
  check half does not get worse. `assemble_v6.py` reads them.

## Reproduce

```bash
export ER_DATA_DIR=.../student_resource/dataset ER_BLOCKING_PROFILE=numeric ER_TOP_K=400
ER_BASE_CACHE=.../er_v3 ER_CACHE_DIR=.../er_v7 python -m src.prepare_v7
ER_CACHE_DIR=.../er_v7 ER_WORK_DIR=.../w7 ER_EXP_TRAIN=150000 python -m src.exp_v3
ER_CACHE_DIR=.../er_v7 ER_WORK_DIR=.../w7 ER_EXP_TRAIN=150000 python -m src.thresholds_v7
ER_CACHE_DIR=.../er_v7 ER_WORK_DIR=.../w7 ER_OUT_DIR=.../out python -m src.predict_v6 single
ER_CACHE_DIR=.../er_v7 ER_WORK_DIR=.../w7 ER_OUT_DIR=.../out python -m src.assemble_v6
```

Data, caches, models and prediction TSVs are not in git.
