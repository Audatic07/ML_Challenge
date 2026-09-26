# Interface contract 2.0-proposed

This is the implementation target, not a claim that modules already exist. The first
session records adoption with the first vertical slice; no four-agent acknowledgment
barrier is required. Task claims determine temporary file ownership.

## Records and immutable versions

- Source columns: `entity_id`, `business_name`, `business_address`, `country`; preserve
  strings, raw text and true empty fields. Source derives from file/ID prefix.
- `query_row` and `target_row` are integer positions scoped to a dataset/row-map hash.
  Use one combined S2+S3 target map with unique target_row values, plus a source enum.
  If using source-local positions internally, include source in their keys and convert
  to the combined map before unioning pairs. Train/test row maps never mix.
- Dataset manifest: file SHA-256, schema, row counts, country/missingness counts, audit
  results, normalization version and row-map hashes.
- Split manifest: stable business-group ID, S1 ID, fit/stop/tune/audit designation,
  seed, policy and SHA-256. Audit is locked; optional transfer splits are separately named.

## Candidates, features and scores

- Candidate key: `(query_row, target_row)` unique within the pinned test/train catalog.
  Store channel flags, channel scores/ranks and retrieval-policy version.
- Preserve the full S1 index independently, including queries with no candidates.
- Feature manifest: names, order, dtypes, missing-value conventions, normalization/index
  dependencies, feature version and code SHA. Use batches/iterators, not full-pair frames.
- Pair scores: same pair keys, finite score, model version and optional routing flags.
- Model manifest: data/split/candidate/feature/code pins, seed, actual hyperparameters,
  thresholds/selection policy, licenses/parameter count, metrics and file checksums.
- A final candidate manifest and scored-input manifest describe the same pair set.
  Any adaptive expansion happens before that set is frozen and completely scored.
- Changing retrieval policy changes negatives and context features: regenerate affected
  training/evaluation candidates, retrain or explicitly validate model compatibility,
  and retune selection on development data before promotion.

## Metrics

- Per nonempty-truth query: `5*TP / (5*TP + 4*FP + FN)`; empty truth scores 1 only
  for empty predictions. Macro average over every expected S1, never over pairs.
- Candidate oracle: select only retrieved truth, empty for true singletons. Report
  `U`, actual score `S`, retrieval loss `1-U`, and matching loss `U-S`.
- Report partition/query count, singleton rate, slices, candidate counts, throughput,
  RAM, uncertainty method and whether results are sampled. No unmeasured values.

## Outputs and CLI

- `matching_results.tsv`: `source1_entity_id`, `matched_entity_ids`.
- `candidate_pairs.tsv`: `source1_entity_id`, `candidate_entity_ids`.
- UTF-8/tab separated; ID lists comma separated without quoting; one row per test S1,
  empty field allowed, no duplicate IDs, valid test S2/S3 only, matches subset candidates.
- Strict validation also compares candidate and scored-pair manifests and shard coverage.
- Proposed commands live in the implementation plan. Implement a single `python -m er`
  CLI with audit/split/index/retrieve/train/evaluate/benchmark/infer/validate/package.
  No command is claimed to exist before its tested implementation is committed.

Schema changes use a small control-state decision with version, affected tasks and cache
invalidation. Notify active consumers. Do not require idle sessions to acknowledge.
