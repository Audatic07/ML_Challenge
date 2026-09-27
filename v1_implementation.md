# AGENTS.md: handoff for teammates and AI coding agents

Read this before changing anything in `code/business_entity_resolution/` (run commands from that folder). It records what exists, why it was built this way, what must not break, and what is worth trying next. Give this whole file to your agent as context.

Status as of 26 Sep 2026, evening: v1 is complete, validated (organisers' validator: PASS) and ready for the portal. Challenge closes 27 Sep 2026, 11:59 PM IST.

## 1. Task in one paragraph

Three sources of business records (`entity_id`, `business_name`, `business_address`, `country`), no shared IDs. Source 1 is deduplicated. For each Source 1 (S1) entity, output every Source 2/3 (S2/S3) record that is the same real business. Metric is macro F0.5: computed per S1 entity, then averaged. An S1 entity with no true matches scores 1.0 only if we predict nothing, else 0.0. Precision is weighted 2x over recall.

## 2. Hard rules (breaking these means rejection or disqualification)

- No external data of any kind: no APIs, no geocoding, no business registries, no scraped or downloaded reference lists. Only the provided train/test files.
- The final model must be MIT or Apache 2.0 licensed and at most 8B parameters. Current model is LightGBM (MIT). If you add a pretrained embedding model, check its licence first.
- `country` is an open set. The test set has France, which is absent from training. Never one-hot, filter or hard-code countries. Every test S1 entity, French ones included, must get a row.
- Output files are TSV. One row per S1 entity, no duplicate rows, no duplicate IDs inside a list, only S2-/S3- IDs that exist in the test files, empty string for no match.
- `candidate_pairs.tsv` must be exactly the set the model scores (the last filtering stage before the model). Every matched ID must appear in it.
- Max 5 portal uploads per day per team. One person (Aadish) uploads, to avoid wasting them. Each member uses only their own login on one device.

## 3. Data facts (measured, not guessed)

| Split | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 |

- Train ground truth: 7,638,365 true pairs. Singleton share 5.6%. Matched entities have 3.67 matches on average, max 11.
- No S2/S3 record belongs to more than one S1 entity (checked on train). The pipeline relies on this (see `unique_assign`).
- Test countries: India 809,986; US 663,106; France 259,452.
- Raw TSVs total 2.4 GB. Ground truth file is 122 MB.

Noise actually seen in the files:

- Hindi (Devanagari) names in S2: `राम मार्केटिंग प्राइवेट लिमिटेड`
- Junk prefixes on names: `-- Holloway Peak Inc Seafood`, `<< Team Ecole`
- Websites as names: `wilfordhancock.com`
- Uppercase addresses: `105 ELM ST, MORGANTON, NC`
- Empty addresses: some S3 India rows have none
- Reordered address components (France): `Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac`
- Long Indian addresses with landmarks: `..., Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra`
- US addresses where a 5-digit number is the house number, not a ZIP: `17560 Ellis Road, Tahlequah, OK`

## 4. Pipeline as built

```
raw TSV -> prepare.py (normalise, cache parquet) -> blocking.py (keys, caps, top-K)
        -> features.py (pair + group features) -> LightGBM -> metric.decide (unique owner, threshold)
        -> predict.py writes candidate_pairs.tsv and matching_results.tsv
```

### Normalisation (`src/text_norm.py`)

Each record produces: `name_norm`, `core` (legal words removed), `skel` (consonant skeleton per core token), `concat` (core without spaces), `addr_norm`, `addr_tok` (informative address words), `postcode`, `house`.

Design choices and reasons:

- `unidecode` runs only on non-ASCII strings (speed). It gives rough Latin for Devanagari.
- Skeleton: keep first letter, drop vowels and `h`, map `c/q -> k`, `z -> s`, `ph -> f`, `ck -> k`, merge repeated letters. Purpose: `raam maarketting` and `ram marketing` both become `rm mrktng`. It absorbs many typos and transliteration variants.
- Legal words are also removed by skeleton (`LEGAL_SKEL`), because transliterated `praaivett limittedd` does not match the English word list directly.
- Postcode is only searched after the first comma, so a leading 5-digit house number is not mistaken for a ZIP. House number is the first number (up to 5 digits) anywhere that is not the postcode, because components are sometimes reordered.

### Blocking (`src/blocking.py`)

Seven key types, all prefixed with country, hashed to UInt64:

| Key | Content | Weight | Cap (v1) |
|---|---|---|---|
| A | two rarest skeleton name tokens | 3 | 200 |
| B | rarest name token + postcode | 3 | 200 |
| C | rarest name token + rarest address token | 2 | 200 |
| D | first 8 chars of `concat` | 2 | 100 |
| E | rarest name token alone | 1 | 50 |
| F | postcode + house number | 2 | 50 |
| G | rarest address token + house number | 2 | 100 |

"Rarest" uses document frequency over S1 + S2 + S3 of the same split. A key is dropped if more S2/S3 records than its cap share it. For each S1 entity, candidates are ranked by summed key weight and the top `TOP_K` (v1: 30) are kept.

### Features (`src/features.py`)

About 40 features, computed in bulk with `rapidfuzz.process.cpdist` (multithreaded, element-wise). Name similarities on several views, address similarities (set to -1 when either address is empty), exact-match flags for postcode / house / country (-1 when missing), lengths, source (2 or 3), which key types the pair shares, and group features comparing the pair to the other candidates of the same S1 entity (rank, gap to best).

### Model and decision (`src/train.py`, `src/metric.py`)

- Train on candidates of 200,000 random train S1 entities, validate on a disjoint 100,000 (seed 42).
- LightGBM binary, learning rate 0.05, 127 leaves, early stopping on validation logloss.
- Threshold grid 0.20 to 0.95, chosen by macro F0.5 on validation. Validation counts singletons and true matches lost at blocking, so it mirrors the leaderboard.
- `unique_assign`: each S2/S3 record is kept only for the S1 entity that gives it the highest probability, then the threshold is applied.

## 5. Results

| Version | Change | Blocking recall | Val F0.5 | Threshold | Leaderboard |
|---|---|---|---|---|---|
| v1 | baseline | 0.79 (train 0.7907, val 0.7913) | 0.8652 | 0.625 | fill in |
| v1 (repo) | same design, code committed in `code/business_entity_resolution/` | 0.79 (train 0.7916, val 0.7921) | 0.8709 | 0.675 | fill in |

The v1 (repo) row was measured locally (Windows, 4 cores, 16 GB) with the default settings and `ER_CHUNK=50000`: train 688 s. Candidate oracle (best achievable F0.5 with these candidates) is 0.8964 on validation, so blocking alone caps this design near 0.90.

v1 test output: 4,629,486 matches, 11.3% of entities empty, 33.6M candidate pairs, about 19 candidates per entity on average. Only 4,999 test entities (0.3%) got zero candidates.

Top features by gain: `a_tset`, `a_partial`, `cc_partial`, `n_tsort`, `hs_eq`, `at_tset`, `cc_ratio`, `a_tset_gap`, `n_partial`, `n_tset`, `n_cand`, `ntok_r`. Addresses carry more signal than names.

LightGBM had not early-stopped at 1000 rounds (validation logloss still falling).

Runtime on ml.r5.xlarge (4 vCPU, 32 GB): train 755 s, predict 1924 s. First-time normalisation of each split takes about 4 minutes and is cached afterwards. Peak memory stayed low (instance MEM indicator around 8% at rest).

## 6. Diagnosis: where the score is lost

Blocking recall of 0.79 is the main limit. With perfect precision, recall 0.79 caps F0.5 near 0.95. Average candidates per entity (about 19) sit below `TOP_K = 30`, so the cut that loses matches is the block-size caps, not top-K. Since almost every entity gets some candidates, the losses are entities that get some of their matches but not all.

Unknown so far (worth measuring): which key types the missed pairs fail on, and whether misses concentrate in India (Devanagari, landmark addresses) or the US.

## 7. Ranked improvement ideas

Work top-down. Always compare on the shared validation set (section 8).

1. **v2 blocking settings (ready, not yet run).** Caps A/B/C to 1000, D/G to 500, E/F to 300; `ER_TOP_K=60 ER_CHUNK=75000 ER_ROUNDS=2000`. Expect recall gain; watch runtime and pair counts.
2. **Look at missed pairs before inventing fixes.** In `train.py`, after `va` is built, take true pairs of validation entities that are not in `va` and print the raw S1 vs S2/S3 name and address for 50 of them, grouped by country. Decide new keys from what you see.
3. **New or looser keys**, depending on step 2:
   - first 4 characters of the rarest skeleton token + country (typos in the rare token)
   - sorted full skeleton string (word-order transpositions)
   - rarest address token + second rarest address token (names too different, e.g. DBA or bad transliteration)
   - postcode + first letter of core name
4. **Normalisation gaps:** Indian transliteration variants (`aa/a`, `ee/i`, `oo/u`, `w/v`, `sh/s`) in the skeleton; strip landmark phrases (`near ...`, `opp ...`) before address similarity; French street words (`rue`, `bd`/`boulevard`, `av`/`avenue`, `bis`, `ter`) canonicalised.
5. **Features:** IDF-weighted token overlap for names and addresses (IDF from the provided data only); numeric tokens inside names matching; count of candidates that share the same postcode.
6. **Model:** more training entities (`ER_TRAIN_SAMPLE=400000`), more rounds, tune leaves and minimum data in leaf. Keep the threshold tuned after every change.
7. **Post-processing:** relative rule (keep a pair only if its probability is within some margin of the best candidate for that entity) tuned on validation.
8. **Embeddings (only if time allows):** a small Apache/MIT sentence-embedding model for a nearest-neighbour blocking pass. 10M records on CPU is slow; test on a sample first and confirm the licence.

France cannot be validated (no French labels). After any change, eyeball 30 random French predictions from `matching_results.tsv` against the raw test rows.

## 8. How to experiment without fooling yourself

- Evaluate everyone on the same 100,000 validation S1 IDs (`val_s1_ids.csv`, shared by Akash). They are `perm[200_000:300_000]` of `np.random.default_rng(42).permutation(len(train_source1))` in file order, which is what `train.py` uses.
- Never train on those IDs.
- Report: blocking recall, validation macro F0.5, chosen threshold. Use `src/metric.py`, which counts singletons and blocking misses.
- Iterate on a sample first: `ER_TRAIN_SAMPLE=30000 ER_VAL_SAMPLE=20000` runs in a few minutes. For the final comparison use the shared 100,000.
- Log every run: version, exact change, recall, F0.5, threshold, leaderboard score if uploaded.
- Only upload to the portal when validation F0.5 beats the current best.

## 9. Gotchas already hit

- Read TSVs with an explicit tab separator. `prepare.read_tsv` uses polars with `quote_char=None`, all columns as strings, empty cells as `''`. Do not switch to default CSV parsing: names contain quotes and addresses contain commas.
- **Normalised data is cached** in `~/er_work/*.parquet`. After changing `text_norm.py` or `prepare.py`, delete those parquet files or the old normalisation is reused silently. `model.txt` and `meta.json` also live there.
- `rapidfuzz` returns 100 for two empty strings. Address similarities are masked to -1 when either side is empty; keep that masking for any new feature on optional fields.
- The feature list is stored in `meta.json`. `predict.py` uses the saved list, so retrain after adding or removing features.
- Polars prints a deprecation warning about `explode` in `train.py`. It is harmless and silenced.
- SageMaker quotas: ml.r5.2xlarge and larger were 0 on Akash's account. ml.r5.xlarge (32 GB) works. If memory runs short, lower `ER_CHUNK`.
- Run long jobs with `nohup python -m src.train > train.log 2>&1 &` so a browser disconnect does not kill them. `tail -f` only watches the log; Ctrl+C stops the viewer, not the job.
- Stop the SageMaker space when idle; it bills per hour.

## 10. Config knobs (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ER_DATA_DIR` | `~/student_resource/dataset` | folder with `train/` and `test/` |
| `ER_WORK_DIR` | `~/er_work` | parquet cache, model, meta |
| `ER_OUT_DIR` | `~/student_resource/output` | output TSVs |
| `ER_WORKERS` | CPU count | threads and processes |
| `ER_TOP_K` | 30 | candidates kept per S1 entity |
| `ER_CHUNK` | 150000 | S1 entities per processing chunk |
| `ER_TRAIN_SAMPLE` | 200000 | S1 entities used for training |
| `ER_VAL_SAMPLE` | 100000 | S1 entities used for validation |
| `ER_ROUNDS` | 1000 | max LightGBM rounds |

Block caps and key weights are in `KEY_TYPES` in `src/config.py`, not in environment variables.

## 11. Still owed before the deadline

- Upload v1 and record its leaderboard score.
- Fill in `Documentation_template.md` from `student_resource/` (methodology, blocking strategy, model and features, other notes). Sections 4 to 7 of this file are the raw material.
- Final zip: `<team_name>_submission.zip` with `output/` (both TSVs from the chosen version), `code/business_entity_resolution/` (src, README.md, requirements.txt) and the filled `Documentation_template.md`. Regenerate both TSVs with the committed code once to confirm it reproduces.
