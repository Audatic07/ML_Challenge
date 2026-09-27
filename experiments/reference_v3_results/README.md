# Business entity resolution

**v6 is the current best pipeline.** v1 is kept below as the documented baseline.

| Version | Validation macro F0.5 (check half, untouched) | Candidate oracle | Candidates / S1 |
|---|---|---|---|
| v1 | 0.8709 (100k queries, threshold picked on the same set) | 0.896 | 19.6 |
| v3 | 0.9646 (0.9633 on a fresh 20k holdout) | 0.982 | 34 |
| v3 + LightGBM/XGBoost blend | 0.9645 | 0.982 | 34 |
| v3 + sibling stage + XGBoost | 0.9646 | 0.982 | 34 |
| **v6** | **0.9737** (tune 0.9746, all 0.9742) | **0.990** | 37 |
| v6 + per-country thresholds | **0.9741** | 0.990 | 37 |

The validation queries are a prefix of the canonical interval `perm[200000:300000]`
(seed 42), split into fixed halves: `tune` picks early stopping and thresholds, `check`
is only scored. The locked audit partition was never opened.

## v6 pipeline

```
raw TSV -> prepare.py (normalise) -> prepare_v3.py (learned transliteration dictionary)
        -> blocking.py, profile "numeric": 9 capped key types (A-I)
           + numeric_pairs.py: channels J/K/L = country + house number/postcode + every token
        -> stage1.py: cheap LightGBM shortlist, top 40 per S1 (= candidate_pairs.tsv)
        -> features3.py (~95 features) -> LightGBM matcher -> unique owner
        -> per-country threshold (US 0.79, India 0.69, others incl. France 0.72)
```

- `translit.py`: word dictionary learned from labelled training pairs maps transliterated
  or misspelt target words to the Source 1 spelling (`sonphttveyr` -> `software`).
  Validation and audit queries are excluded when learning it.
- `numeric_pairs.py`: Astra's numeric rescue channels (from the v3 review), rebuilt as a
  memory-light, batched step and fed into blocking, so the shortlist ranker and matcher
  are retrained on them. They raise the oracle from 0.982 to 0.990.
- `features3.py`: house-number relations, number-set overlap, IDF-weighted token coverage,
  typo-aware extra/missing words and decoy-cluster group features.
- `stack_v6.py`: model comparison (LightGBM, XGBoost, blend, sibling second stage).
  The model family is not the bottleneck: all within 0.0005.

```bash
export ER_DATA_DIR=/path/to/student_resource/dataset
export ER_BLOCKING_PROFILE=numeric ER_TOP_K=400
ER_WORK_DIR=/path/to/base ER_CACHE_DIR=/path/to/base python -m src.prepare train test
ER_BASE_CACHE=/path/to/base ER_CACHE_DIR=/path/to/v3 ER_WORK_DIR=/path/to/v3 python -m src.prepare_v3
ER_CACHE_DIR=/path/to/v3 ER_WORK_DIR=/path/to/v6 ER_EXP_TRAIN=150000 python -m src.exp_v3
ER_CACHE_DIR=/path/to/v3 ER_WORK_DIR=/path/to/v6 ER_OUT_DIR=/path/to/out python -m src.predict_v6 single
ER_CACHE_DIR=/path/to/v3 ER_WORK_DIR=/path/to/v6 ER_OUT_DIR=/path/to/out python -m src.assemble_v6
```

Measured on a 4-core, 16 GB laptop: `prepare_v3` 25 min, `exp_v3` 60 min (150k training
queries, 5.6M pairs, 1,439 rounds), `predict_v6` about 4 h. Set `POLARS_MAX_THREADS=6`
if memory is tight.

## v1 pipeline

Blocking keys -> pair features -> LightGBM -> unique owner + threshold. The design and
the reasons behind it are in `v1_implementation.md` at the repository root.

```
raw TSV -> prepare.py (normalise, cache parquet) -> blocking.py (keys, caps, top-K)
        -> features.py (pair + group features) -> LightGBM -> metric.decide (unique owner, threshold)
        -> predict.py writes candidate_pairs.tsv and matching_results.tsv
```

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt      # Windows: .venv\Scripts\pip
```

Only the supplied challenge files are used. No external data, APIs or pretrained weights.
The model is LightGBM (MIT licence).

## Reproduce end to end

Run from this folder (`code/business_entity_resolution/`). Point the three directories
at the data, a scratch area and the output folder:

```bash
export ER_DATA_DIR=/path/to/student_resource/dataset   # contains train/ and test/
export ER_WORK_DIR=/path/to/er_work                     # parquet cache, model, meta
export ER_OUT_DIR=/path/to/output                       # submission TSVs

python -m src.prepare train test   # normalise + cache (optional, done on demand)
python -m src.train                # blocking + features + LightGBM + threshold on validation
python -m src.predict              # writes $ER_OUT_DIR/candidate_pairs.tsv and matching_results.tsv
python ../../student_resource/utils/validate_submission.py \
    --matching $ER_OUT_DIR/matching_results.tsv \
    --candidate $ER_OUT_DIR/candidate_pairs.tsv \
    --test-dir $ER_DATA_DIR/test --check-ids
```

`train.py` writes `model.txt`, `meta.json` (features, threshold, validation score,
blocking recall) and `val_s1_ids.csv` into `ER_WORK_DIR`. `predict.py` refuses to run
if the feature list changed since training.

Unit tests for the metric: `python -m pytest -q tests`.

## Configuration (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ER_DATA_DIR` | `~/student_resource/dataset` | folder with `train/` and `test/` |
| `ER_WORK_DIR` | `~/er_work` | parquet cache, model, meta |
| `ER_OUT_DIR` | `~/student_resource/output` | output TSVs |
| `ER_WORKERS` | CPU count | threads |
| `ER_TOP_K` | 30 | candidates kept per S1 entity |
| `ER_CHUNK` | 150000 | S1 entities per processing chunk (lower it if memory runs short) |
| `ER_TRAIN_SAMPLE` | 200000 | S1 entities used for training |
| `ER_VAL_SAMPLE` | 100000 | S1 entities used for validation |
| `ER_ROUNDS` | 1000 | max LightGBM rounds |

Block caps and key weights are `KEY_TYPES` in `src/config.py`.

Normalised data is cached in `ER_WORK_DIR/*.parquet`. After changing `text_norm.py` or
`prepare.py`, delete those files or the old normalisation is reused silently.

## Modules

| File | Role |
|---|---|
| `src/config.py` | paths, knobs, key types, LightGBM parameters |
| `src/text_norm.py` | name/address views: `name_norm`, `core`, `skel`, `concat`, `addr_norm`, `addr_tok`, `postcode`, `house` |
| `src/prepare.py` | TSV reading (explicit tab, no quoting), normalisation, parquet cache, ground-truth links |
| `src/blocking.py` | seven capped key types, candidates ranked by summed key weight, top-K |
| `src/features.py` | ~45 pair features with `rapidfuzz.process.cpdist`, key flags, per-entity group features |
| `src/metric.py` | exact macro F0.5 (singletons and blocking misses count), oracle, `unique_assign`, `decide` |
| `src/train.py` | sampling, LightGBM training with early stopping, threshold grid 0.20-0.95 |
| `src/predict.py` | test inference and the two TSV files |

## Licences

LightGBM: MIT. polars: MIT. RapidFuzz: MIT. NumPy: BSD-3-Clause.
Unidecode is GPL-2.0-or-later; it is a text-preprocessing dependency, not the model.
