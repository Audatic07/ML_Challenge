# Business entity resolution: reproduction guide

This folder regenerates `output/matching_results.tsv` and `output/candidate_pairs.tsv`
from the supplied `student_resource/dataset` files. It uses only the provided data. There
are no external lookups and no pretrained weights; the matcher is a LightGBM model (MIT
license) trained on the supplied labels.

## Pipeline

1. **Normalization** (`src/text_norm_v5.py`). Names and addresses are lowercased and
   transliterated. The canonical name drops legal suffixes, honorifics and alias
   prefixes (`aka`, `dba`), repairs digits inside words (`g0mez` to `gomez`) and removes
   repeated tokens. Addresses get canonical street types, ordinals and state codes, and
   PMB, PO Box, unit and `N/A` noise is removed.
2. **Candidate retrieval** (`src/v5_dist.py`, `Catalog`). For each country, every S2
   and S3 target is indexed with six sparse TF-IDF channels: name character 3-4 grams,
   address character 3-4 grams, folded name words, folded address words, and a combined
   name plus address view as character 3-4 grams and as words. IDF comes from the
   searchable catalog itself. Each channel keeps its own top-k per source (the k values
   are in `deploy/channels_v51.json`), and the union is kept. Address search does not
   depend on name similarity.
3. **Pair features** (`src/features.py`, `src/v4_features.py`, `anchor_features` in
   `src/v5_dist.py`). There are 92 features: fuzzy name and address similarities,
   number and token overlap, missing-field flags, channel scores, ranks and gaps within
   the query, and support from the query's strongest other candidates.
4. **Stage-1 filter** (`survivors` in `src/v5_dist.py`). Each query keeps its 30
   retrieved candidates with the highest sum of the two combined name+address cosine
   scores (`combo` + `combo_word`). This is a label-free retrieval score. The survivors
   are the final candidate set written to `candidate_pairs.tsv`. The cut-off was chosen
   on the tune partition (`src/v5_cascade.py eval`).
5. **Matching** (`run_train`, `run_cascade`). A LightGBM binary classifier scores only
   the survivors. Each target can belong to only one S1: the highest probability wins,
   and the pair is kept only if its probability reaches the threshold chosen on the tune
   partition. An S1 can end up with zero, one or many matches.

Splits: seed-42 permutation of train S1 rows. Rows 200,000-300,000 are the tune set,
300,000-310,000 the early-stopping set, the last 10% a locked audit set that was never
opened, and 400,000 of the remaining rows are the fit set. Targets owned by held-out
S1 rows are removed from fit pairs.

## Environment

Python 3.12 with the pinned `requirements.txt`:

```bash
python -m venv venv
venv/bin/pip install -r requirements.txt
```

## Run on one machine

Every stage runs through a small task queue. With `--root` the queue is a local folder.
From this directory, with `DATA` pointing at `student_resource/dataset`:

```bash
python -m src.v5_dist plan --data $DATA --root runs/v51 --norm v5 --anchors --channels-json deploy/channels_v51.json --memory-scale 2.0 --neg-keep 0.3
python -m src.v5_dist worker --data $DATA --root runs/v51 --work work --name local
python -m src.v5_cascade eval --root runs --source v51 --prefix v51c --ns 20,30,40,50,60,80 --ranks= --scores=combo+combo_word
python -m src.v5_dist worker --data $DATA --root runs --prefix v51c --work work --name local
python -m src.v5_cascade score --root runs --source v51 --prefix v51c --n 30 --score combo+combo_word --threshold 0.72
python -m src.v5_dist worker --data $DATA --root runs --prefix v51c --work work --name local
```

The first worker call retrieves candidates and builds features for 510,000 train and
1,732,544 test queries in 20,000-query shards, then trains the model and scores the tune
partition (`runs/v51/model/`). `eval` writes `runs/v51c/ceval/result.json`, which reports
the tune macro F0.5, the candidate oracle and candidates per query for each cut-off.
The submission used the top 30 and the threshold 0.72 that the evaluation selected for it. The last worker call writes both TSV files to
`runs/v51c/final/` and runs the supplied validator.

The first worker call on a single 8-core machine with 64 GB RAM takes many hours.
Training needs about 40 GB of memory.

## Run distributed (how the submitted files were made)

The same commands ran on eight SageMaker notebook instances. Use `--bucket` and
`--prefix` in place of `--root`. `deploy/onstart.sh` is the notebook lifecycle hook,
and `deploy/bootstrap.sh` installs Python, downloads the data and code and starts
`src.v5_dist worker`. Workers claim tasks with S3 conditional writes. The submitted run
used the prefixes `shared/er-v51-20260927` (retrieval, features, training) and
`shared/er-v51c-20260927` (stage-1 filter, test scoring, final files).

## Checks

```bash
python ../../student_resource/utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir $DATA/test --check-ids
python src/strict_validate.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir $DATA/test
python -m pytest tests -q
```

## Licenses

LightGBM, scikit-learn, SciPy, NumPy, Polars, RapidFuzz, sparse-dot-topn and boto3
use MIT, BSD or Apache-2.0 licenses. Unidecode (transliteration during normalization)
is GPL-2.0-or-later. It is a text-processing dependency and is not part of the model.
The model file holds 2,000 trees with 127 leaves each and has about 506,000
parameters.
