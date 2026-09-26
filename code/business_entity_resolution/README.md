# Business entity resolution pipeline

Blocking (key-based candidate generation, with name-based and name-independent address channels) followed by a LightGBM pair classifier
over string-similarity features, with the decision threshold tuned for macro F0.5.
No external data, APIs or pretrained models are used.

## Layout

```
src/config.py     paths (env-overridable) and parameters
src/text_norm.py  name / address normalisation, transliteration, consonant skeletons
src/prepare.py    TSV loading, parallel normalisation, parquet cache
src/blocking.py   blocking keys, block-size caps, top-K candidates per Source 1 entity
src/features.py   pairwise and per-group features (rapidfuzz cpdist)
src/metric.py     challenge macro F0.5 and the final match decision
src/pipeline.py   chunked blocking + features loop
src/train.py      training, validation, threshold tuning
src/predict.py    test inference, writes both output TSVs
```

## Reproduce

```bash
pip install -r requirements.txt
export ER_DATA_DIR=/path/to/student_resource/dataset   # contains train/ and test/
export ER_OUT_DIR=/path/to/output                        # where the two TSVs are written
export ER_WORK_DIR=/path/to/work                         # cache + model (default ~/er_work)
python -m src.train      # normalise train data, build candidates, fit model, tune threshold
python -m src.predict    # score test candidates, write candidate_pairs.tsv and matching_results.tsv
```

Defaults without the exports: data in `~/student_resource/dataset`, outputs in
`~/student_resource/output`, cache in `~/er_work`. Delete `~/er_work/*.parquet`
after changing `text_norm.py`, because normalised data is cached.

v3 defaults: twelve key types (A to L, see `src/config.py`), transliteration-aware skeletons for names and address words, TOP_K 60, chunks of 75,000, up to 2000 rounds.
`train.py` also prints the candidate oracle ceiling U and blocking recall per country.
`python -m src.diag_misses india` shows sample true pairs that share no blocking key.

Tested on a 4 vCPU / 32 GB instance (ml.r5.xlarge). Useful overrides:
`ER_WORKERS`, `ER_CHUNK` (lower it if memory runs out), `ER_TOP_K`,
`ER_TRAIN_SAMPLE`, `ER_VAL_SAMPLE`, `ER_ROUNDS`.

Validate the outputs with the organisers' script before uploading:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```
