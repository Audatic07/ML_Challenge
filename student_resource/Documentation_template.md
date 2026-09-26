# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** TBD

**Team Members:** TBD

**Submission Date:** TBD

**Status:** Documentation draft; champion, completed-run metrics and release hashes pending.

This draft describes DIST-V5 and the v5.1 experiment reported by
`coordination/evidence/v5-dist/claude-20260927-0340.md` at code commit
`87298e02d8d8fe5f4df8aad815b41758b4b7f976` on
`codex/v5-dist-claude-20260927`. Implementation details were checked against that
commit. The benchmark results below are attributed to that evidence, not
independently rerun here. A final selected model is **TBD**.

## 1. Executive Summary

Independent name and address retrieval generates a union of plausible S2/S3 records
for each S1 business. A LightGBM binary classifier scores engineered pair features;
a validation-selected threshold and one-owner-per-target rule produce zero, one or
many matches per query. Combined name-address retrieval and, in v5.1, support from
other strong candidates address difficult name variations without requiring a name
match before searching an address.

## 2. Methodology

### 2.1 Problem Analysis

The supplied data contains noisy names, abbreviations, transliterations, changed
word order, partial or missing addresses, and multiple target records for one
business. Common business names create misleading neighbors. Source 1 is the
deduplicated reference; the target catalog is the union of S2 and S3. Test queries
include France, although supplied training labels cover US and India.

Evaluation uses every query in the selected tune partition, including singletons
and queries with no candidates. With truth G and predictions P, the per-query
score is `5 TP / (4 |P| + |G|)` when G is nonempty. Empty truth earns 1 for empty
predictions and 0 otherwise. All reported end-to-end scores must be macro means.

### 2.2 Solution Strategy

**Approach Type:** Independent sparse blocking + supervised tree classifier +
target-ownership selection.

**Core contribution:** Separate quotas for multiple name and address views, followed
by deduplication of their union. Address retrieval runs independently of name
similarity. The model combines field agreement, missingness, retrieval context and
optional candidate-to-candidate support.

Catalogs contain all targets of the query's country and both target sources;
evaluation targets are not reduced to known positives. Country partitions are
constructed from observed labels, including France at test time. Country blocking
can still miss any cross-country truth; its safety is not proved by the small
benchmark alone. No truth is injected into evaluation candidates.

The implementation uses a fixed seed-42 S1 permutation with separate fit, stop,
tune and reserved audit rows; it is not represented here as an audited connected-
component split. The reported run configuration selects 400,000 fit, 10,000 stop
and 100,000 tune queries. Targets owned by held-out references are removed from
supervised fit pairs. The source evidence says the audit remains unopened.
Exact split, data and model manifests for the selected release: **TBD**.

## 3. Candidate Generation (Blocking)

Binary hashed n-grams use 2^20 dimensions, catalog-derived IDF, and L2 normalization.
Common hashed features are suppressed above their document-frequency limits
(character: 0.03; word: 0.05). Sparse cosine products search each source separately.
Target columns are processed in blocks of 250,000 and merged into per-source top-k
lists. Hash collisions are possible. All channel results are unioned and deduplicated
without a further final union cap in the documented v5 implementation.

| Channel | Input view and n-grams | v5 baseline k/source | v5.1 config k/source |
| --- | --- | ---: | ---: |
| name_char | Core name, character-within-word 3-4 | 20 | 20 |
| address_char | Normalized address, character-within-word 3-4 | 20 | 40 |
| name_fold | Auxiliary folded name, word 1-2 | 20 | 10 |
| address_fold | Auxiliary folded address tokens, word 1-2 | 20 | 10 |
| combo | Core name + normalized address, character-within-word 3-4 | Off | 40 |
| combo_word | Same combined view, word 1-2 | Off | 60 |

The baseline uses v4 normalization. v5.1 adds fixed transformations for alias/trade-
name markers, digit substitutions inside names, repeated tokens, address labels and
abbreviations. Normalization changes require new indexes/features and validation.
The deployed v5.1 JSON has `combo_word=60`; the third benchmark row below used 20.
These are distinct policies and their results must not be interchanged.

**Candidate pairs generated for final test release:** TBD.

**How retrieval loss is measured:** Candidate oracle U predicts exactly the retrieved
truths. For g true links and r retrieved true links, U per query is `5r/(4r+g)`;
singletons have oracle 1. Report link recall, oracle U and the matching gap U-S.

### Reported retrieval benchmark

4,000 tune queries **per country**, full training target catalogs, v5.1
normalization; k is per source. Values are copied from the cited DIST-V5 evidence.

| Policy | India U | US U | Mean pairs/query, India |
| --- | ---: | ---: | ---: |
| Four v5 channels at 20 | 0.9705 | 0.9926 | 125 |
| Add combined name-address character channel at 40 | 0.9789 | 0.9981 | 177 |
| Name 20, address 40, folds 10, combo 40, combo_word 20 | 0.9854 | 0.9986 | 194 |
| All channels at k 40-60 | 0.9876 | 0.9989 | 309 |

These are retrieval ceilings, not classifier or leaderboard scores. They do not
establish a 0.98 final macro score, and none reaches the 0.995 working oracle margin
on India. There is no labelled France benchmark. The evidence also reports
baseline India shard U=0.9725 and link recall 0.932 with v4 normalization; that is
a separate shard measurement, not the first row of this table.

## 4. Matching Model

**Features used:**

- Names: normalized/core/skeleton/concatenated string ratios, partial and token
  sort/set similarities, core Jaro-Winkler, exact agreement, lengths, folded-view
  comparisons, token precision/recall and extra/missing tokens.
- Addresses: normalized/token/folded similarities, postcode and house-number
  agreement, number overlap/precision/recall, lengths and explicit missingness.
  Address similarity is masked to -1 when an address is absent.
- Context: source indicator, country agreement, candidate count, per-channel
  cosine values, within-query ranks and gaps to the best candidate. The inherited
  blocking-key flags are zero in v5's sparse path; they do not imply extra key
  retrieval channels.
- v5.1: combined-channel scores/ranks/gaps and nine anchor-support features.
  Up to three candidates selected by a label-free heuristic support comparisons
  to other candidates; a candidate is excluded as its own anchor.

**Model type:** LightGBM binary gradient-boosted decision trees, trained from
provided labels with no pretrained weights. Source defaults use 127 leaves,
minimum 100 records per leaf, feature fraction 0.9, bagging fraction 0.8 every round,
L2 penalty 1.0, seed 42, and deterministic column-wise training. The v5 plan specifies
learning rate 0.08, at most 2,000 rounds, and early stopping after 75 rounds on the
stop partition's binary log loss. Actual tree count and parameters must be taken
from the selected `model_manifest.json`: **TBD**.

The v5.1 evidence reports retaining all retrieved positives and hard fit negatives;
easy negatives are retained with probability 0.30 and weighted by its inverse.
Stop/tune candidate distributions are not downsampled.

**Threshold selection method:** Score every tune candidate, apply global target
ownership, then maximize exact macro per-S1 F0.5 over a coarse threshold grid and
a 0.0025 refinement near its best value. **Selected threshold: TBD.**

**One-owner-per-target rule:** For each S2/S3 target, keep the S1 query with highest
model probability; ties go to the lower S1 row position. Accept that pair only if
its score is at least the chosen threshold. An S1 can own many targets or none.
This is target exclusivity, not a one-to-one assignment. Tune ownership is computed
within the tune queries; test ownership spans the full test S1 set.

## 5. Results & Error Analysis

| Result for the selected release | Value |
| --- | --- |
| Tune macro F0.5 | TBD |
| Full-tune candidate oracle U and U-S | TBD |
| Locked audit macro F0.5 | TBD |
| Public leaderboard score | TBD |
| Private leaderboard score | TBD |
| Final threshold / model hash / candidate-manifest hash | TBD |
| Total test candidate pairs / matched pairs / TSV bytes | TBD |

**Common false positives:** TBD from saved tune scores; do not infer prevalence
from retrieval examples.

**Common false negatives:** TBD, separated into unretrieved truths and retrieved
truths rejected by the threshold or ownership rule.

The prepared `src/analyze_tune.py` reconstructs predictions from saved tune scores,
checks its F and U against saved per-query values, and groups losses by country,
exact truth count, and their cross-product. Its explicit attribution removes
false positives first, then restores retrieved-but-rejected truths. This gives
`1-S = (1-U) + FP loss + retrieved-FN loss`; the latter two depend on that order.
About 20 deterministic examples are retained privately. Real analysis status and
aggregate results: **TBD**. No audit labels are selected for analysis.

## 6. Conclusion

Independent address retrieval and combined name-address channels improve the
reported candidate ceilings, especially in India. A supervised tree model and
explicit empty-set/ownership decisions complete the documented approach.
Final score, reproducibility and compliance acceptance remain pending; the
retrieval table alone does not establish the target score.

## Appendix

### A. Code Artefacts and Reproduction

The release must contain the complete selected pipeline under
`code/business_entity_resolution/`, pinned dependencies, data/split/feature/model
manifests, license notices and commands needed to regenerate both outputs. This
documentation branch contains support files; it does not itself contain the v5
pipeline from Claude's branch.

At the pinned source revision, entry points are `python -m src.v5_dist plan`
and `python -m src.v5_dist worker`, both requiring `--data` and either a private
`--bucket/--prefix` or a local `--root`. The v5.1 plan additionally uses
`--norm v5 --anchors --channels-json deploy/channels_v51.json --neg-keep 0.3`.
A fresh reproduction directory must be used; do not rerun a plan into a live queue.
The archived exact selected command, runtime, data pins and clean-run result are
**TBD**, so this draft does not claim a completed reproduction.

Release validation from the package directory:

~~~bash
python src/strict_validate.py --matching /release/output/matching_results.tsv --candidate /release/output/candidate_pairs.tsv --test-dir /data/test --scratch-dir /private/scratch --report /private/strict-report.json
~~~

The default enforces 1,732,544 S1 rows in **both** outputs, exact headers, no
duplicate list/row IDs, test target membership and matches-subset-candidates.
It streams candidates and reports byte sizes and SHA-256 hashes. Verify the files
extracted from the final archive. Separately reconcile all scored-input pairs and
shards; TSV validation cannot establish that semantic equality.

### B. Licenses and Data Use

LightGBM uses the [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE).
The model is a tree ensemble trained from the supplied labels; **no pretrained
weights** are used in the documented v5/v5.1 paths. The final manifest's reported
parameter-count convention and value must be checked against the <=8B rule:
**TBD**. No external entity records, registries, geocoding or hosted entity-resolution
calls are part of this documented method.

**Open release decision: Unidecode 1.4.0 is GPLv2-or-later**, as stated by its
[maintainer's package metadata](https://pypi.org/project/Unidecode/1.4.0/).
It is a normalization dependency in `requirements-v5.txt`; the pipeline cannot
be described as entirely MIT/Apache while that issue is unresolved. The release
owner must decide whether its use/distribution meets the competition rules, or
replace it with a permissible normalization approach and regenerate/revalidate
the affected artifacts. LightGBM's MIT license does not settle that dependency
question. No replacement or infrastructure change was made for this draft.

### C. Submission Size

See `coordination/evidence/submission-size/findings.md`. The supplied PDFs and
README state no numerical ZIP or candidate-file upload limit. That does not prove
the portal accepts arbitrary sizes. The separate candidate-generation notice
also makes candidate-set compactness relevant to final ranking. Any pruning
cascade must be implemented and measured before its survivors can be declared
the final candidate set.
