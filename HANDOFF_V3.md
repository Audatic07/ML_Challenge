# Handoff: Akash's pipeline, v1 to v3

Branch: `akash/v3-pipeline`. Written 27 Sep 2026, just after midnight IST. Challenge closes 27 Sep 2026, 23:59 IST.

This file is for teammates and their coding agents. It covers what the pipeline does, what was measured, why v3 looks the way it does, what is still weak, and how to continue. `v1_implementation.md` on main describes v1 in more depth; this file supersedes it where they differ.

## 1. What is on this branch

```
HANDOFF_V3.md                        this file
code/business_entity_resolution/     v3 code (runnable), same layout as the final submission zip
  src/config.py                      paths, key types, caps, parameters (env-overridable)
  src/text_norm.py                   name / address normalisation, skeletons
  src/prepare.py                     TSV loading, parallel normalisation, parquet cache
  src/blocking.py                    blocking keys, caps, top-K candidates
  src/features.py                    pair and group features
  src/metric.py                      exact macro F0.5, final decision rule
  src/pipeline.py                    chunked blocking + features loop
  src/train.py                       training, validation, threshold, ceiling report
  src/predict.py                     test inference, writes both TSVs
  src/diag_misses.py                 shows true pairs that share no blocking key
  README.md, requirements.txt
results/akash/
  v1_train_log.txt                   v1 training + prediction numbers
  v2_train_log.txt                   v2 training log
  diag_india_v2.txt                  India miss diagnostic on v2
  SUBMISSION_LOG.md                  every version, its settings and scores
.gitignore                           adds caches, outputs, models, logs
```

Not committed: the dataset, parquet caches, `model.txt` / `meta.json`, output TSVs. Models are shared separately.

## 2. The approach in one screen

1. **Normalise** every record once (cached as parquet). Transliterate non-Latin text, lowercase, strip punctuation, unify abbreviations, drop legal suffixes, and build a consonant **skeleton** per name word so spelling variants collapse (`ram marketing` and `raam maarketting` both become `rm mrktng`). Extract postcode and house number.
2. **Block**: each record gets up to 12 keys (A to L), all prefixed with country. Keys shared by more S2/S3 records than a cap are dropped. Each S1 entity keeps its top 60 candidates by summed key weight. This set is exactly `candidate_pairs.tsv`.
3. **Match**: about 45 features per pair (rapidfuzz similarities on names and addresses, exact postcode / house / country flags, which keys the pair shares, rank of the pair among the entity's candidates). LightGBM binary classifier trained on 200k S1 entities, validated on a separate 100k.
4. **Decide**: each S2/S3 record goes only to the S1 entity that gives it the highest probability (the training labels confirm no record belongs to two S1s), then a single global threshold chosen by exact macro F0.5 on validation.

Validation counts singletons and matches lost at blocking, so it mirrors the leaderboard formula.

## 3. Results so far

| Version | What changed | Val blocking recall | Ceiling U | Val F0.5 |
|---|---|---|---|---|
| v1 | baseline, 7 keys | 0.791 | not measured | 0.865 |
| v2 | address-only keys H, I, J; caps about 5x; TOP_K 60; 2000 rounds | 0.882 | 0.948 | **0.912** |
| v3 | transliteration folds, skeleton address words, keys K and L | not run yet | | |

v2 per country: US recall 0.926 (U 0.972), India recall 0.816 (U 0.913). Leaderboard scores go in `results/akash/SUBMISSION_LOG.md` once uploaded.

**Ceiling U** is the macro F0.5 a perfect matcher would reach on the retrieved candidates. For an entity with g true matches of which r were retrieved it is 5r / (4r + g), and 1.0 for singletons. v2's gap splits roughly in half: 1 - U = 0.052 is lost at blocking, U - F = 0.036 is lost by the matcher and threshold.

## 4. Why v2 and v3 look like this

**v2** came from Aditya's labelled noise profile on main (`output/hackathon/labelled_noise_profile.json`): about 9% of true links share no name word (15% for India S3) while about 96% share an address word. v1 only searched by address when a house number or postcode was present. v2 added three address keys that ignore the name:

| Key | Content |
|---|---|
| H | two rarest address words |
| I | postcode + rarest address word |
| J | second rarest address word + house number |

**v3** came from `diag_misses.py` on 3,000 Indian validation entities (10,402 true pairs, output in `results/akash/diag_india_v2.txt`):

- 81.6% of true pairs are found by v2
- about 11% share a key with their S1 but are dropped by block caps or the top-60 cut
- 7.3% share no key at all

The no-key pairs show two patterns:

1. **Back-transliterated names.** `lotus properties pvt ltd` vs `lottaas prpaarttij praaibhett limittedd`, `laxmi` vs `lkssmii`, `management` vs `mainejmentt`, `investment` vs `inbhesttmentt`. v2's skeleton did not fold `bh`/`v`, `x`/`ks`, `j`/`g` or stray `y`.
2. **Shortened addresses.** One side keeps the full address, the other only house number + city + state, so the "rarest address word" differs between sides and address keys never line up. State names are transliterated too (`mhaaraassttr`, `krnaattk`, `hriyaannaa`). Indian records almost never carry a postcode (B, F, I keys matched only about 20 pairs each).

v3 changes:

- Skeleton folds `ph->f`, `bh->v`, `ck->k`, `x->ks`, `q->k`, `c->k`, `z->s`, `j->g`, and drops `y` as well as vowels and `h`. Checked on real misses: lotus/lottaas, laxmi/lkssmii, anand/aannd, management/mainejmentt, investment/inbhesttmentt, real/riyl now give identical skeletons.
- `addr_tok` (informative address words) is stored as skeletons: `maharashtra`/`mhaaraassttr` -> `mrstr`, `karnataka`/`krnaattk` -> `krntk`.
- Extra legal tokens dropped from names: `praa`, `pra`, `li`, `pvtltd`, `limted` (seen as `praa li` for pvt ltd).
- Two keys that do not depend on rarity:

| Key | Content | Reason |
|---|---|---|
| K | first skeleton name word + house number | first word usually survives transliteration; house number survives shortening |
| L | house number + first letter of core name + last address word (usually state) | works when names differ a lot |

- Caps stay at v2 values.

**Rejected:** wider caps (A/C 2000, G/J 1500, D/E/H 1000). On the synthetic check, recall fell 0.926 -> 0.901 and F0.5 0.960 -> 0.952, because weak candidates filled the top-60. The candidate ranking before top-K is too crude to benefit from bigger blocks (see limitations).

v3 has only been run end to end on a synthetic dataset, which has no transliteration, so its real effect on India is unmeasured. Ship v3 only if its validation F0.5 beats 0.912.

## 5. Known limitations

**Retrieval**
- Candidate ranking before the top-K cut is a weighted count of shared keys. It cannot tell a strong single key from a weak one, ties are broken by row index, and this is why wider caps hurt. About 11% of Indian true pairs are lost here.
- Keys use "rarest token" logic. When one record is a shortened copy of the other, rarest tokens differ. K and L address part of this; the rest is open.
- Names that are unrelated strings (`iriarc`, `tavozeph`, `irivio` for real company names) can only be found through the address. If the house number also differs (`143` vs `6 143`), nothing links them.
- House number is the first number (up to 5 digits) that is not the postcode. Indian addresses like `h no 14 1 211 622 4` or `plot no 6 143` often put a different number first.

**Normalisation**
- The v3 skeleton folds were chosen from Indian examples. `j->g` and dropping `y` also apply to US and French names and may create collisions (for example `jay` and `gay` share a skeleton). Watch US recall and precision in the v3 log.
- `unidecode` gives rough Latin for Devanagari; there is no proper Hindi transliteration.
- France: only generic handling (accent folding through unidecode, SARL/SAS/SASU/EURL removal, `de/la/le/du/des` stopwords). No French street-word canonicalisation. France cannot be validated because training has no French labels.

**Model and decision**
- LightGBM never early-stopped (v1 at 1000, v2 at 2000 rounds). More rounds or a higher learning rate may help a little.
- Training uses 200k of 2.2M S1 entities. More data may help; not tested.
- No hard-negative mining, no per-query set selection, one global threshold for all countries.
- `unique_assign` works globally at test time, but in validation competitors from outside the 100k sample are missing, so validation slightly underestimates its benefit.
- The validation split is by S1 entity. Because each S2/S3 record belongs to at most one S1, this is equivalent to a group split.

**Runtime**
- v2 prediction takes about 1.5 hours on ml.r5.xlarge. The per-chunk time printed in the log covers only blocking and features; scoring with 2000 trees adds about 3 minutes per chunk.
- v2 training about 26 minutes with the parquet cache; v3 needs a fresh cache (about 8 extra minutes across train and test).
- Larger instances (ml.r5.2xlarge and up) have a quota of 0 on Akash's account.

## 6. How to run v3

On SageMaker (ml.r5.xlarge, 32 GB), after v2 prediction has finished:

```bash
cd code/business_entity_resolution
pip install polars unidecode rapidfuzz lightgbm
export ER_DATA_DIR=~/student_resource/dataset
export ER_WORK_DIR=~/er_work_v3      # fresh folder: normalisation changed, do NOT copy v1/v2 parquet
export ER_OUT_DIR=~/output_v3
nohup python -m src.train > train.log 2>&1 &
tail -f train.log
```

Read these lines when it finishes:

```bash
grep -E "recall|ceiling|us:|india:|F0.5" train.log
```

Only if validation F0.5 > 0.9119:

```bash
nohup python -m src.predict > predict.log 2>&1 &
cd ~/student_resource
python3 utils/validate_submission.py --matching ~/output_v3/matching_results.tsv \
    --candidate ~/output_v3/candidate_pairs.tsv --test-dir dataset/test
```

Diagnose misses for any country (light on memory, can run beside a prediction):

```bash
python -m src.diag_misses india > diag_india.txt 2>&1
```

Useful overrides: `ER_TOP_K`, `ER_CHUNK` (lower if memory is short), `ER_TRAIN_SAMPLE`, `ER_VAL_SAMPLE`, `ER_ROUNDS`, `ER_WORKERS`. Caps and weights live in `KEY_TYPES` in `src/config.py`. Iterate on a sample first (`ER_TRAIN_SAMPLE=30000 ER_VAL_SAMPLE=20000`), then confirm on the full 100k validation set.

Gotchas: `export` only applies to the current terminal; run long jobs with `nohup`; Ctrl+C on `tail -f` stops the viewer, not the job; start a job only once (a double paste once launched two predictions writing the same files); delete parquet caches after any change to `text_norm.py`.

## 7. Suggested next steps, highest value first

1. **Run v3 and compare.** Watch India recall and U, and check US recall did not drop.
2. **Better ranking before top-K.** Take the top 150 to 200 pairs by key weight, score them with two cheap similarities (for example `token_set_ratio` on skeleton name and on `addr_tok`), keep the best 60. This is the way to recover the 11% lost to caps without flooding, and would let caps go up safely.
3. **Rerun `diag_misses` on v3** for India and US to see what still shares no key.
4. **Matcher gains** (U - F is 0.036): more training entities (`ER_TRAIN_SAMPLE=400000`), more rounds, IDF-weighted token overlap features, a query-level no-match gate.
5. **France sanity check:** open 30 random French rows of `matching_results.tsv` and compare with the raw test records.

Stop tuning with enough time left for the final zip: `output/` (both TSVs of the chosen version), `code/business_entity_resolution/`, and the filled `Documentation_template.md`.

## 8. Rules any change must keep

- Only the provided data. No APIs, geocoding, registries or downloaded reference lists.
- Final model MIT or Apache 2.0 and at most 8B parameters (LightGBM is MIT).
- Country is an open set; never one-hot or filter to US/India. Every test S1 gets a row.
- `candidate_pairs.tsv` is exactly the set the model scored; every match appears in it.
- Compare versions on the same validation IDs; upload to the portal only when validation improves.
