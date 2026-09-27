# Business entity resolution: approach summary

**Team / members / submission date:** TBD

**Draft status:** Final model, threshold and all end-to-end score fields remain TBD.
Based on Claude's DIST-V5 evidence at `87298e02d8d8fe5f4df8aad815b41758b4b7f976`.

## Approach and candidate generation

We resolve each deduplicated S1 business against S2 and S3 using independent
sparse name/address retrieval, engineered pair features and LightGBM. The output
can contain zero, one or many targets. Every test S1, including France, must
appear once in each output TSV. Only the supplied business data and labels are
used; the documented model uses no pretrained weights.

The baseline has four channels: core-name character 3-4-grams, normalized-address
character 3-4-grams, and word 1-2-grams of auxiliary folded name and address views.
Each channel retains its own top 20 per source. Address search runs independently
of name similarity. Binary hashing into 2^20 dimensions, catalog-derived IDF and
L2 normalization permit sparse cosine retrieval against full country catalogs.
Targets are searched in 250,000-column blocks and results merged. The channel
union is deduplicated without a final union cap.

The v5.1 experiment adds combined name-address character and word channels,
revises normalization of aliases and address forms, and increases address
retrieval. Its configured per-source quotas are name 20, address 40, folded views
10 each, combined character 40 and combined word 60. Country labels are discovered
from the data; France receives its own test catalog without requiring training
labels for that country.

## Reported experiments

The source evidence reports 4,000 tune queries **per country**, against full
training target catalogs with v5.1 normalization. U is the candidate oracle:
the macro F0.5 achievable by returning only retrieved true matches.

| Retrieval policy | India U | US U | Pairs/query, India |
| --- | ---: | ---: | ---: |
| Four baseline channels at 20 | 0.9705 | 0.9926 | 125 |
| Add combined character channel at 40 | 0.9789 | 0.9981 | 177 |
| Name 20, address 40, folds 10, combo 40, combo_word 20 | 0.9854 | 0.9986 | 194 |
| All channels at 40-60 | 0.9876 | 0.9989 | 309 |

These are attributed retrieval ceilings, not independently reproduced classifier
scores. The third row's combined-word quota is 20; the v5.1 deployment configuration
uses 60. Higher recall costs more candidate pairs. India remains below the 0.995
working oracle margin even in the widest reported policy. No labelled France
score is available.

<div style="page-break-after: always;"></div>

## Matching and selection

Features compare normalized names, core names, skeletons, concatenated names,
addresses and folded views using string ratios, token set/sort similarity and
Jaro-Winkler. Exact agreement, lengths, token and number overlap, house/postcode
agreement, source, missingness, channel scores, candidate counts, ranks and gaps
supply additional evidence. Missing addresses receive explicit flags and masked
similarities. v5.1 also compares each candidate with up to three strong alternative
candidates using label-free anchor-support features.

LightGBM learns a binary pair classifier. The reported plan reserves 400,000 fit,
10,000 early-stop and 100,000 tune queries under fixed seed-42 S1 splits; the
audit stays locked. Held-out-owned targets are excluded from fit pairs. Defaults
include 127 leaves, learning rate 0.08, at most 2,000 rounds and 75-round early
stopping. v5.1 retains hard negatives and positives, sampling easy fit negatives
at 30% with inverse-probability weights. Actual model/tree counts are TBD.

For each target, the highest-probability S1 is its sole owner; equal scores favor
the lower S1 row position. The pair is accepted only at or above a global
threshold chosen to maximize tune macro per-S1 F0.5. S1 can still own many targets.
Singletons and zero-candidate queries remain in the metric. This rule is evaluated
within tune queries and applied across all test queries at final inference.

## Results, limitations and release

**Tune macro F0.5: TBD. Locked audit: TBD. Public/private leaderboard: TBD.
Selected threshold and champion: TBD.** Error analysis will distinguish retrieval
misses, false positives and missed-but-retrieved truths by country and truth count,
with approximately 20 private examples. The retrieval results alone do not
establish the 0.98 target.

[LightGBM is MIT licensed](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE);
no pretrained weights are used. **Unidecode 1.4.0 is
[GPLv2-or-later](https://pypi.org/project/Unidecode/1.4.0/) and needs a release
decision.** Any normalization replacement requires regeneration and validation.
Final license review and parameter-count confirmation are pending.

The package must include both exact TSVs, runnable source, pinned dependencies
and the full documentation template. The strict streaming validator checks
1,732,544 rows per output, exact coverage/headers, existing targets, duplicates
and the subset relation. The release owner must also reconcile the candidate
file with the final scored-pair manifest. A smaller candidate set can affect
final ranking; the supplied documents specify no numerical upload-size limit.
