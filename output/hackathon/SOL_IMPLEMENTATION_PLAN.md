> Historical bootstrap plan. The current implementation/training specification is [V7 final iteration plan](V7_FINAL_ITERATION_PLAN.md). V6 and stage-two code now exist on main; the original no-model/no-code status and startup schedule below describe 26 September, not the current state. Read the current plan and live control branch before acting.

# Sol implementation plan: pursue 0.98 macro F0.5

**Decision date:** 26 September 2026, approximately 20:00 IST. **Official close:** 27 September, 23:59 IST. **Internal acceptance deadline:** 27 September, 21:30 IST. Recalculate the remaining time when execution starts; approximately 28 hours remained at research time. This replaces the earlier fixed-role plan and its September 25 schedule.

**Objective:** deliver a reproducible, compliant submission with the strongest defensible route to at least **0.9800 macro per-S1 F0.5**. No model or leaderboard score has been measured in this workspace. The target is not a forecast or guarantee. Paper benchmark scores are not evidence that this dataset will reach it.

**Execution:** whoever is available takes the highest-value unclaimed work. Sol can implement the entire critical path without waiting for three other agents. Other Codex/Claude sessions take bounded tasks or experiments from the queue. There are no permanent A/B/C/D responsibilities, equal-work quotas, or all-agent approval meetings. See `coordination/PROTOCOL.md` for brief claims and evidence-based handoffs.

## 1. The concrete strategy

Build **independent name and address retrieval -> supervised tree matcher -> query-aware selection**, then spend remaining time on the largest measured source of lost score. The likely first improvement is recovering badly changed names through addresses, followed by realistic hard negatives and better treatment of singletons. A small multilingual neural model is a conditional residual experiment, not a prerequisite.

```text
provided TSVs -> immutable row maps + safe normalized views
                        |
             full-catalog candidate indexes
             / name / address / exact keys /
                        |
             union, unique pairs, measured recall
                        |
           pair features -> LightGBM scores
                        |
       optional bounded candidate expansion / second-stage features
                        |
           freeze final candidate set and score every pair
                        |
           calibrated set selection, including empty set
                        |
      two complete TSVs -> strict validation -> reproducible package
```

The main engineering bet is to use the large supplied label set well and make the metric explicit. Do not spend the remaining day building a general agent platform, reproducing an entire research repository, or running an unbounded model search.

## 2. Evidence from these files, including a new local diagnostic

All three supplied PDFs were reviewed previously: 18 pages including the blank final page. The problem statement and guidelines were re-read for this revision. The resource README, template and validator were inspected. `data_profile.json` contains the full seven-TSV streaming profile.

| Observed quantity | Value | Consequence |
| --- | ---: | --- |
| Labelled S1 queries | 2,206,821 | Enough labels for a substantial supervised matcher and honest held-out evaluation. |
| Training S2 + S3 | 10,320,219 | Small-target-pool validation would be misleading. |
| Test S1 queries | 1,732,544 | Every one needs an output row. |
| Test S2 + S3 | 9,969,589 | No all-pairs materialization; roughly 17.27 trillion possible pairs. |
| Labelled links | 7,638,365 | Multiple matches are normal; maximum observed is 11, not a legal prediction cap. |
| Training singletons | 123,247 / 5.585% | False positives on empty-truth queries consume score directly. |
| Test France S1 | 259,452 / 14.975% | Unseen-country behavior materially affects the target. |

A new deterministic hash sample selected **22,034 S1 queries and 75,776 true links**. It examined only local supplied records and exported aggregate statistics to `labelled_noise_profile.json`. Reproduce with `python tools/profile_labelled_noise.py`. This script is a diagnostic, not a retrieval pipeline or model.

| Diagnostic on true links | Measured result | Implementation implication |
| --- | ---: | --- |
| Same normalized full name | 25.75% | Exact matching alone cannot be the solution. |
| Shared normalized name token | 85.72% | Require independent rescue channels. |
| Neither shared name token nor name trigram Jaccard >= 0.25 | 9.12% | Do not require name similarity before searching addresses. |
| Shared address token | 95.57% | Address retrieval is promising; common tokens alone do not establish identity. |
| Missing target address | 4.43% | Retain a strong name-only path and explicit missingness. |
| Both addresses contain numbers, but number sets are disjoint | 7.29% of all sampled links | Numbers are learned evidence, never a universal hard veto. |
| Name-edge diagnostic oracle macro score | 0.958622 | This particular name-only gate is incompatible with 0.98 on this sample. |
| Same oracle with true-label-assisted name bridges | 0.961167 | Name-based graph closure alone does not fix the observed gap. |

The name-edge oracle assumes perfect classification of all positives satisfying that diagnostic condition. It does **not** simulate an actual index, candidate cap, ranking, or negative population. The bridge oracle even uses ground truth to connect variants and must never be reported as achievable inference performance. Address overlap is not address retrieval recall or precision. Zero observed cross-country links in this sample does not prove zero across all training data.

The weak-name fraction is particularly high in the India sample: approximately 23.89% for S2 and 15.36% for S3, versus approximately 2.1-2.2% for US. Prioritize the India/address error slice before a generic neural upgrade. These are sample diagnostics, not conclusions about France.

## 3. Turn 0.98 into measurable gates

For query i, let G_i be the true set and P_i the predicted set. With nonempty G_i:

`F_i = 5 TP_i / (5 TP_i + 4 FP_i + FN_i) = 5 |P_i intersect G_i| / (4 |P_i| + |G_i|)`.

For empty G_i, score 1 when P_i is empty and 0 otherwise. Average over **all** S1 queries. A micro pair score, ordinary F1, classification accuracy, or a score excluding no-candidate queries is not the competition metric. [Problem statement pp3,6]

Examples: with three true matches, returning two correct matches scores 10/11 = 0.909091; returning all three plus one false match scores 15/19 = 0.789474. With one true match, adding one false match drops the score to 5/9. The target allows average loss of just 0.02.

For retrieved candidate set C_i, compute the exact **candidate oracle ceiling** by predicting `C_i intersect G_i`. For g_i > 0 and r_i = |C_i intersect G_i| its score is `5 r_i / (4 r_i + g_i)`; for singletons it is 1. The mean is U. The actual pipeline score S cannot exceed U with that candidate set.

Report the exact decomposition `1 - S = (1 - U) + (U - S)`. The first term is unavoidable loss from retrieval; the second is recoverable matching/selection loss on those candidates. This makes the next experiment a decision rather than a guess.

| Gate | Working target, not an organizer rule | Action if missed |
| --- | --- | --- |
| Candidate oracle U | >= 0.995 overall, inspect every material slice | If U < 0.98, fix retrieval before classifier tuning; between 0.98 and 0.995, little error margin remains. |
| Candidate link recall | Aim >= 99.5%, plus per-query all-links recall | Use macro oracle as the decisive gate; link recall alone can hide missed singleton-like queries. |
| Development score | Aim >= 0.985 to leave room for shift | Inspect lost-score totals by slice; do not call 0.985 a guarantee of private 0.98. |
| Locked audit | Point estimate >= 0.98 and one-sided 95% group-bootstrap lower bound >= 0.98 | Otherwise report target unproven and retain the strongest valid incumbent. |
| Transfer tests | Compare US->India and India->US against incumbent | These are stress tests, not estimates of France's labelled score. |
| Full run | Measured completion, verification and upload fit the deadline with buffer | Simplify or start earlier; never substitute an unmeasured ETA. |

Using full-test country proportions for illustration, if the non-France portion averages 0.985, France needs about **0.9516** for the combined score to reach 0.98; if the other portion averages 0.99, France needs about **0.9232**. The public/private country mix is unknown. Even very high domestic validation leaves meaningful transfer risk.

## 4. What the research supports, and what to adopt

These are method references only. Do not download their benchmark datasets, business records, external labels, gazetteers or demo-trained entity matchers into the solution.

| Primary source reviewed | Useful result or idea | Decision for this deadline |
| --- | --- | --- |
| [Ditto, PVLDB 2021, sections 2-5](https://arxiv.org/pdf/2004.00584) | Treat a pair of records as a classification input, preserve field boundaries, and train on difficult examples. Its company case combines blocking signals before a matcher. | Use field-specific features and hard negatives immediately; consider a compact pair encoder only for residual errors. Its reported F1 belongs to another dataset and metric. |
| [SC-Block, ESWC 2024, sections 3-5](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf) | Supervised contrastive retrieval and nearest neighbors; evaluate blocker cost at a specified recall level. Validation recall does not automatically transfer to test. | Adopt recall-versus-cost measurement now. A learned dense retriever is optional only if lexical/address misses dominate and full-catalog encoding fits time. |
| [Sudowoodo, methods and ablations](https://arxiv.org/pdf/2207.04122) | Contrastive representations and deliberately difficult, lexically related training examples. | Mine near-name/different-address and same-address/different-business negatives from supplied labels. Do not reproduce its whole pretraining system. |
| [Dembczynski et al., ICML 2013, sections 3-5](https://proceedings.mlr.press/v28/dembczynski13.pdf) | F-beta optimization is a set decision problem; choosing an empty set and choosing set size matter. Independence assumptions can fail. | First tune the exact macro metric; then test a query-level selector using held-out predictions. Do not assume 0.5 is optimal or call a simple expected-count ratio an exact expected F score. |
| [Splink term-frequency documentation](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html) | Agreement on a rare value carries different evidence from agreement on a common value. | Include frequency-weighted overlap and name/address collision counts in the supervised matcher; a full unsupervised linkage system is unnecessary with these labels. |

The pipeline below is our dataset-specific proposal, not a claimed reproduction of any paper. Paper code licenses and downloaded model-weight licenses are separate. In particular, the [SC-Block repository](https://github.com/wbsg-uni-mannheim/SC-Block) advertises BSD-3-Clause; its availability does not establish that a submitted final model meets the challenge's MIT/Apache requirement.

## 5. Build validation before optimizing

1. Audit IDs, duplicate rows, ground-truth references, cross-country positive links, and targets linked to more than one S1. S1 is officially deduplicated, but verify before using target exclusivity as evidence. Check identical normalized name/address records with inconsistent labels. Do not repair labels based on a model guess.
2. Form groups from each S1 and its labelled targets. Union components if a target occurs under multiple references. Group repeated exact identity fingerprints across S1 when warranted by the audit. Use deterministic hashes and store a split manifest. Never randomly split labelled pairs.
3. Allocate approximately 80% of groups to fit, 5% to early stopping, 5% to calibration/selection tuning, and 10% to locked audit. Stratify by country and match-count bucket when feasible. Every evaluation partition retains its natural singletons and zero-candidate queries.
4. Use nested fixed subsets for speed: initially 100k fit S1, 10k early-stop S1 and 20k tuning S1; expand fit to 300k only if the measured learning curve justifies it. Final audit should use the full reserved partition if time permits, otherwise explicitly report its fixed sample size.
5. Validation retrieval searches the full **10.32-million-record training target catalog**. Do not reduce it to just the held-out positives or inject ground-truth positives into validation candidates. Targets belonging to held-out groups may be searchable distractors, but exclude them from supervised fit pairs, including fit negatives.
6. Separate two kinds of statistics: label-derived normalizers/aliases and supervised models fit only on fit groups; an index may compute unsupervised document frequencies on its actual searchable catalog, including held-out/test targets. Apply that same index-building rule at validation and test time and document it. No test labels or external data enter fitting.
7. Tune and ablate on the development partitions. Open the audit only after choosing the champion. If it disappoints, report that fact; do not turn it into another tuning set. Retain one tested model instead of an unvalidated last-minute refit on every label.

Minimum scorer tests: both sets empty; nonempty truth/empty prediction; false match on singleton; two correct of three; three correct plus one wrong; the PDF example 10/14 = 0.714286; predictions containing unknown query IDs; missing queries; duplicate IDs. Compare set-based implementation with an independent scalar reference on generated fixtures. Missing rows must fail validation, not silently leave the mean.

For experiments, save per-query F, oracle F, candidate count, truth count, false positives, false negatives and slices. Compute paired deltas on the same queries. For uncertainty, resample whole business groups and calculate a query-weighted mean within each bootstrap replicate; use 1,000 replicates on saved scores. Country shift is not covered by this confidence interval.

## 6. Retrieval implementation: address rescue is mandatory

### 6.1 Data layout and text views

Parse UTF-8 TSV explicitly with `sep='\t'`; preserve IDs as strings and empty addresses as empty, not literal `nan`. Convert once into partitioned Parquet/Arrow plus integer row maps. Store train/test maps separately and checksum them. Use uint32 row positions where counts fit; never infer identity from ID digits, adjacent row order or source-file ordering.

Keep raw text, NFKC/casefolded text, punctuation/whitespace normalization, an auxiliary accent-folded view, name tokens, address tokens, number strings, name character n-grams and address character n-grams. Preserve accents in one view and numbers in all meaningful views. Legal suffix removal is an additional name view only; an empty or common suffix-only result cannot become a strong matching key. Use transformations justified by supplied examples or fit labels. No external address dictionaries or geocoding.

### 6.2 Independent candidate channels

| Channel | Initial implementation | Why it exists |
| --- | --- | --- |
| Exact full/name-address keys | Hash/SQL indexes on country + normalized name + normalized address, plus distinct auxiliary views | Cheap high-quality candidates; still run through the model. |
| Rare name tokens and token pairs | Inverted postings, retain document frequencies, cap costly common postings with recorded truncation | Handles reordering and partially changed names. |
| Name character search | TF-IDF character 3/4-grams, initial top 20 per target source | Typo and unseen-word recovery, including France. |
| Address search, independently of name | Rare address token pairs and character 3/4-gram search; initial top 20 per target source | Recovers the weak-name slice exposed by the diagnostic. A low name score must not prefilter this channel. |
| Distinctive number/address combinations | Number plus rare street/locality token, with a name-independent alternative | Efficient location retrieval without relying on arbitrary digit equality. |
| Missing-address/name fallback | Preserve name-only neighbors and collision metadata | Missing address is absent evidence, not proof of nonmatch. |

Start with dynamic country/source partitions after the full cross-country audit. Retain an explicit route for missing/unknown labels and measured cross-country exceptions. Never enumerate only US and India. All France rows use the same generic indexes and features.

Recommended first backend: sparse CSR TF-IDF with `sparse_dot_topn.sp_matmul_topn` for bounded top-k products, alongside exact/inverted keys. The [maintainer documentation](https://github.com/ing-bank/sparse_dot_topn) supports chunked products and Windows/Linux wheels. Use float32, query batches and target shards; merge shard results into global top-k with correct target offsets. Top-k limits output memory, not every intermediate operation or runtime: benchmark common-name tails. Avoid a dense cosine matrix or Python comparison of every query against every target.

Fit each catalog's index vocabulary on that catalog, so French n-grams are represented. An optional hashing backend must measure collision effects and IDF behavior; do not silently replace TF-IDF with an unweighted hash vector. Preserve index version, vocabulary/IDF hash, row map, channel configuration and hardware timings.

### 6.3 Recall gate and adaptive expansion

Union all channels and deduplicate by `(query_row, target_row)` using a single combined S2+S3 target-row map; source-local row numbers must not collide. Do not cap the final union without measuring the resulting oracle. Start by comparing top-k 20, 50 and 100 per source/channel on a fixed 20k tuning sample against full indexes. Report before/after each cap and filter.

Expand difficult queries when: a posting list was truncated; a search fills its k slots with near-tied scores; names are common/short; an address channel disagrees with a name channel; no credible candidate exists; or an initial matcher sees uncertainty. A high top score alone must not disable expansion: a query may already have one correct match while missing several others.

Candidate diagnostics: mean/p95/p99/max count, source and country coverage, link recall, fraction of nonsingletons with all links, missed-all frequency, and macro oracle U. Report absolute losses for weak-name, missing-address, common-name, one-match and many-match slices. If U < 0.98, stop model tuning and repair the highest-loss retrieval cases. If U >= 0.995, proceed to matching while keeping retrieval fixed.

Optional one-hop expansion: query the indexes using at most two independently strong candidate variants, union the new records, and score them against the original S1 with anchor-support features. Train/evaluate this using predictions from a model that did not fit those queries. Never use true peers as inference anchors, never copy an anchor's identity without scoring, and never repeatedly apply unrestricted transitive closure. The positive-only diagnostic suggests this is secondary to independent address search.

## 7. Matcher implementation and hard negatives

### 7.1 Feature contract

Implement a versioned feature list with explicit order/dtypes/missingness. Target roughly 50-80 numeric features initially, not hundreds of speculative transformations.

| Family | Concrete features |
| --- | --- |
| Names | Exact raw-normalized/accent-folded/suffix-light agreement; character cosine; normalized edit and Jaro-Winkler; token Jaccard and containment; weighted overlap; unmatched rare tokens; lengths and ratios. |
| Addresses | Character cosine/edit/token overlap; weighted rare-token overlap; exact nonempty agreement; number intersection/union; number conflicts; digit sequences in relative positions; token and character lengths. |
| Quality and ambiguity | Missing address flags; name frequency; identical name/address key frequency; number of S1 competitors for a target; short/common-name flags; normalization information loss. |
| Retrieval context | Channel flags, channel scores/ranks, counts per channel/source, top-score gaps, score distribution, truncation/expansion flags. |
| Interactions | Strong name + weak/conflicting location; weak name + rare address agreement; missing address + common name; strong location + multiple competing names. |

Use compiled/vectorized comparisons in batches. Cache normalized records once. Do not materialize a Python object/dictionary for every feature of tens of millions of pairs. Missing fields receive separate flags and well-defined similarity values. Never use raw entity IDs or row positions as predictive features. Source can be a feature; country must accept unseen strings and have a global fallback rather than a closed US/India one-hot gate.

### 7.2 Training recipe

Retrieve for fit queries with the production policy. Label their candidates from supplied truth, retaining singletons. Keep retrieved positives, hard negatives and a random negative sample with recorded inclusion probabilities. Useful hard negatives include same name/different location, same address/different business, nearly identical branch names and plausible competitors for the same target. Ground-truth positives missed by retrieval may train an auxiliary pair model only when explicitly tagged; they cannot inflate retrieval or end-to-end validation metrics.

First model: LightGBM binary classifier, provisional `learning_rate=0.05`, `num_leaves=63`, `min_data_in_leaf=100`, `feature_fraction=0.9`, `bagging_fraction=0.8`, `bagging_freq=1`, `lambda_l2=2`, maximum 1,200 rounds, early stopping after 75 non-improving rounds, fixed seed. Set worker threads to available cores and avoid nested oversubscription. These are starting values to benchmark, not optimal settings. Pin the installed version and retain the [MIT license](https://github.com/lightgbm-org/LightGBM/blob/main/LICENSE).

Start without `is_unbalance` or a guessed `scale_pos_weight`; class weighting and sampled negatives can distort probability estimates, as the [LightGBM documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html) notes. The selection threshold must be learned on the natural, unsampled candidate distribution. Compare unweighted training with per-query weights as a small ablation; neither loss is exactly macro F0.5.

After the first model, mine false positives and high-scoring nonmatches from **fit** groups. Add one hard-negative round, then retrain. Do not train on audit errors. Run a 100k->300k query learning-curve comparison before choosing extra data over improved features. Only try a second tree seed/configuration if it produces a complementary measured gain. A retrieval-policy change also changes negative examples and query-context features: regenerate affected candidates/features and retrain or explicitly validate compatibility, then retune the development selection policy before promotion.

## 8. Selection: optimize a set per query

**Baseline:** save candidate scores and sweep a global threshold to maximize the exact query macro metric on the selection-tuning partition. Use a coarse grid followed by refinement around the best region, including high-score quantiles. Include empty queries and singletons. Compare source- or missing-address-specific thresholds only with sufficient support, a meaningful paired improvement, and no clear transfer regression. France defaults to the validated global policy.

**First residual experiment:** train a query-level no-match gate from top scores, score gaps, collision counts, agreement across channels, source support and missingness. Its training inputs must be out-of-fold scores, not scores from a classifier trained on those same queries. Tune the gate jointly with match thresholds; do not impose the training singleton fraction on test.

**Second residual experiment, only when selection loss remains material:** a learned prefix-utility selector. For each query, sort the final candidate scores and construct choices k=0..K, preserving at least every prefix that a reasonable threshold would select. On out-of-fold fit predictions compute the actual F0.5 of each prefix using complete truth. Regress this utility from query summaries, k, prefix-score summaries and boundary gaps, weighting each query equally across its choices. Choose k with the largest predicted utility; retain the threshold policy as a fallback for unsupported cases. Calibrate/select on a separate partition and evaluate end-to-end.

This is an approximate engineering adaptation of set-level decision making. It is not the exact General F-measure Maximizer, which needs joint label/cardinality information. Do not use `1.25 * sum(p_top_k) / (0.25 * sum(p_all) + k)` as if it were exact expected F0.5; expectation of a ratio is not that ratio. Correlated duplicate variants and unretrieved positives invalidate naive independence assumptions. Start with thresholds because they are easier to validate under this deadline.

S1 deduplication suggests target competition can be useful. First audit that each labelled target has at most one S1. Use competing-reference scores as features; consider retaining only a sufficiently dominant claimant as a validated experiment. Do not run a one-to-one assignment across S1 and targets: S1 can have many targets. Never force a winner for an ambiguous target or force at least one match for a query.

## 9. Conditional neural upgrade and France

Run at most one neural experiment if all conditions hold: a valid incumbent exists; full retrieval is sufficiently strong; the residual error audit points to pair understanding; pretrained weights are established as permissible under the organizer rules; a suitable licensed checkpoint is verified; GPU access is available; and both training and routed inference fit the remaining deadline and recorded spend cap. If these conditions fail, continue the supplied-data tree path immediately.

A concrete candidate to investigate is Microsoft's [Multilingual-MiniLM-L12-H384](https://huggingface.co/microsoft/Multilingual-MiniLM-L12-H384). Its card lists MIT and approximately 117M parameters (21M Transformer + 96M embedding), well below 8B. It is a base encoder, not an entity matcher. The card specifically warns about its BERT architecture/XLM-R tokenizer combination; verify the tokenizer and two-record classification head on a fixture before training. Check the exact downloaded revision, license and parameter count. The challenge's model-license allowance does not by itself settle its broader pretrained-data wording.

Suggested bounded pilot: fine-tune a two-record classifier on 100k-200k supplied hard pairs, maximum length 192 or 256 with both fields represented, learning rate around 2e-5, one epoch first, stratified error evaluation. Infer on a measured uncertainty subset, combine its score with tree/context features using held-out data, and evaluate the complete cascade. Route based on observed scores/quality, never validation truth. Prefer real labelled variants; optional punctuation/order/address-drop augmentation uses only fit records and must preserve valid labels. Do not use external synonyms, business examples or LLM-generated identity labels.

A dense contrastive retriever is a different experiment: it must encode the whole relevant target catalog and search it. Do not start that simply because a pair classifier pilot is fast. Generic semantic embeddings can merge similar types of businesses; retain lexical/address evidence. No hosted LLM or external resolution service processes dataset records.

France checks without labels: complete row coverage; nonempty index features; accent-preserving versus folded views; candidate-count and score distributions; common-name collision rates; address-number behavior; source balance; and missingness. Run US-only fit -> India and the reverse on fixed samples as shift stress tests. Character features and field comparisons should remain useful without an external French dictionary. No synthetic or unlabelled France score may be presented as actual validation.

## 10. Sol's build order and task queue

These are implementation specifications. The `er` package and commands below do not exist yet. Existing deliverables are the plan, coordination documents, data profiles and local diagnostic script. Start with a runnable vertical slice and extend it.

| ID | Priority / dependencies | Deliverable and acceptance evidence |
| --- | --- | --- |
| P00 | First; can overlap local fixture work | Inspect existing work; create package skeleton, config and manifests; register sessions/task claims. Coordination setup timebox 20 minutes, no orchestration platform. |
| P01 | Critical / P00 | TSV audit, row maps, group split and exact scorer. Golden cases and profile/reference-integrity report pass. |
| P02 | Critical / P00 | Two-file writer and strict validator on invented fixtures. Reject missing/duplicate/foreign IDs and non-subset predictions. |
| P03 | Critical / P01 | Independent name/address retrieval; full-pool recall/oracle/cost curves and replayable candidates. |
| P04 | Critical / P01; production data from P03 | Batched features, first tree and saved dev scores; end-to-end metric and error slices. |
| P05 | Critical / P03, P04 | Global selection threshold, measured throughput and complete valid incumbent; preserve immutable artifacts. |
| P06 | Highest measured gain / P05 | Address/weak-name retrieval repair or hard-negative/features improvement selected by loss decomposition. |
| P07 | Conditional / P05 | OOF no-match or prefix-utility experiment. Promote only by paired development gain and transfer checks. |
| P08 | Conditional / P05 | One-hop rescue or optional neural pilot, chosen from residual evidence; never a release dependency. |
| P09 | Critical / P05 | Country-transfer audit, clean-run rehearsal, resource and license checks; can run while improvements proceed. |
| P10 | Critical / chosen champion, P09 | One locked audit; freeze manifests; final inference and strict validation. Optional tasks can be explicitly skipped. |
| P11 | Critical / P02, P10 | Exact submission ZIP, full template + concise summary, version ledger and authorized portal receipt. |

Any agent can claim any ready item or a nonoverlapping subtask. One fast Sol session may take P00-P05 end to end. If other agents arrive, good immediately separable work includes P02 fixtures, P01 scorer review, full-pool profiling, address-retrieval variants and P09 reproduction. Nobody waits merely to preserve equal contribution.

Suggested package layout:

```text
code/business_entity_resolution/
  pyproject.toml, requirements.txt, README.md
  configs/baseline.json
  src/er/
    cli.py, io.py, manifests.py, normalize.py, splits.py, metrics.py
    retrieval/{keys.py,sparse.py,union.py,diagnostics.py}
    matching/{features.py,train.py,select.py,context.py}
    pipeline/{infer.py,validate.py,package.py}
  tests/{test_metric.py,test_retrieval.py,test_outputs.py,test_pipeline.py}
```

Use Python 3.11 or another verified compatible installed environment; avoid a contest-time runtime upgrade. Core candidates are NumPy, Arrow/Parquet, DuckDB or equivalent chunked I/O, SciPy/scikit-learn, sparse_dot_topn, RapidFuzz, LightGBM and pytest. Install in a project environment and pin **actual tested versions**. Neural dependencies stay optional. The bundled planning runtime did not contain LightGBM, SciPy, DuckDB or RapidFuzz when checked; do not assume imports will work without setup.

Proposed CLI, implement each command and document its exact arguments:

```text
python -m er audit --data student_resource/dataset --out artifacts/audit
python -m er split --config configs/baseline.json
python -m er index --catalog train --config configs/baseline.json
python -m er retrieve --split tune --config configs/baseline.json
python -m er train --config configs/baseline.json
python -m er evaluate --split tune --config configs/baseline.json
python -m er benchmark --catalog test --queries 10000 --config configs/baseline.json
python -m er infer --catalog test --run-id RUN_ID --resume --config configs/champion.json
python -m er validate --run-id RUN_ID --strict
python -m er package --run-id RUN_ID
```

The benchmark must include indexes with **all relevant target records**, a representative country mixture and common-name tails. A 10k-query microbenchmark against a small target subset is not evidence of full-run throughput. No command may silently use test labels, fill validation misses with truth or regenerate a changed split.

## 11. Runtime, storage and spending

At 40 candidates per test S1 there are about 69.3M pairs; at 80 there are about 138.6M. A dense 80-feature float32 table alone would occupy approximately 22.2GB or 44.4GB, before overhead. Compute and score in shards, not one pandas frame. Eight-byte integer pair keys alone occupy about 0.55GB at 69.3M pairs; expanded string TSVs are larger. Measure disk and transfer requirements early.

Use country/source target partitions, bounded query batches, float32 matrices, integer keys and compressed intermediate Parquet. Process data/index versions sequentially where memory is tight. Start by benchmarking on an available 32-64GB CPU worker; this is a planning specification, not a provisioned instance. A smaller machine may work with target sharding and additional I/O. Avoid four full duplicate data/index builds across accounts.

Use deterministic query shards, e.g. 20k S1 per shard. Each shard manifest records exact IDs/row range, input/config/model hashes, code SHA, candidate and scored-pair hashes, row counts, duration and completion status. Write temporary outputs then atomically rename after validation. Resume only when all pins match; a file's existence alone is not completion. Final assembly rejects missing/overlapping shards.

For six hours of inference, the raw aggregate minimum is about 80.2 S1/s; at 40 candidates/query that is about 3,208 pairs/s. To reserve 50% runtime contingency within six hours, require approximately **120.3 S1/s** measured aggregate throughput, with indexing and upload budgeted separately. Extrapolate each stage and include long-tail batches. If the system misses this, launch the reliable full run earlier, add authorized shards, or cut the least valuable cost measured by oracle/score loss.

Three accounts have approximately $100 each; the fourth contributes zero until verified. Replace person-based allocations with **job-based caps**: provisional $45 retrieval/indexing, $60 training/experiments, $90 full inference/rehearsal and $15 storage/transfer, plus $90 reserve. Total planned active spend is $210, reserve $90; account balances and eligible service credits still need verification. These are caps, not current price quotations. Record account alias, unique job ID, maximum runtime, estimated cost and actual spend before launching. Never exceed an account's usable balance or existing authorization. Coding-agent costs are separate.

Share immutable private artifacts with checksums. Git holds code and small aggregate metadata only. The currently configured repository was observed public; do not commit raw records, candidate/output TSVs, models, private logs, credentials or presigned URLs. S3 sharing transfers artifacts, not AWS credit balances. Batch jobs suffice; no serving endpoint is required.

## 12. Schedule from now, with explicit stop rules

Use elapsed time for execution, but preserve the absolute submission buffer. At a 26 September 20:00 IST start:

| Elapsed / indicative IST | Required result |
| --- | --- |
| 0-1.5h / by 21:30 | Package scaffold, immutable split, exact scorer and tiny end-to-end output. Other agents can prepare disjoint tasks immediately. |
| 1.5-4h / by 27 Sep 00:00 | Full-pool retrieval measurements, initial classifier, thresholded development score and full-run ETA. |
| 4-8h / by 04:00 | Address/weak-name repairs and one hard-negative iteration; earliest complete incumbent inference running. |
| 8-12h / by 08:00 | Complete valid incumbent if runtime permits; residual experiment selected by measured lost score; country stress/reproduction checks. |
| By 09:30-10:00 | Champion chosen, locked audit once, model/features/candidate policy frozen. Advance this if measured runtime requires it. |
| 10:00-16:00 | Final inference under measured budget; completion checks per shard. Keep the incumbent intact. |
| 16:00-19:00 | Strict validation, exact ZIP, full methodology and 1-2-page summary; independent review of release hashes. |
| 19:00-21:30 | Authorized submission, SCORED status and package acknowledgment saved; resolve format/portal problems. |
| 21:30-23:59 | Contingency for necessary repair only. No speculative model changes. |

If starting later, compress research/optional experiments first. The critical path is P01->P03->P04->P05->P10->P11, with P02 and P09 alongside it where capacity exists. Do not sacrifice a complete submission to reach an arbitrary experiment count.

After each experiment, publish a compact report: baseline/new macro score; paired delta and interval; oracle U; country/singleton/weak-name slices; runtime/RAM; exact candidate/model/split hashes; and whether the result was promoted. A practical promotion bar is +0.001 absolute development macro F0.5 with positive paired evidence and no material transfer/runtime regression. Smaller gains may be accepted near the target when evidence is strong and cost negligible. This bar is a team choice, not statistical proof against distribution shift.

Stop rules: no score improvement from two targeted iterations -> change the diagnosed failure mode; retrieval U below target -> stop classifier architecture search; neural pilot fails elapsed/spend gate -> stop it; final-run ETA exceeds safe window -> freeze and launch earlier; no validated improvement -> ship incumbent. If 0.98 remains unreachable in measured validation, state the gap and submit the strongest compliant result rather than fabricate a target-achieving score.

## 13. Release rules that must survive the rush

Both files must include every test S1 exactly once, including all France records and empty predictions. Headers are exactly:

```text
matching_results.tsv: source1_entity_id<TAB>matched_entity_ids
candidate_pairs.tsv: source1_entity_id<TAB>candidate_entity_ids
```

Write UTF-8 TSV, comma-separated IDs with no list quoting, deterministic order, no duplicate IDs/rows, and existing **test** S2/S3 IDs only. Final matches are a subset of candidates. Export candidates from the **final set passed to the final matching system**, after all retrieval expansion/filtering. Every exported candidate must have been scored; every scored final pair must be represented. For a composite routed matcher, document the cascade and record its complete scored input union, including tree-only decisions. If a later standalone model replaces it and sees only a subset, export that actual final-model set instead. Avoid ambiguity by retaining one declared final composite scorer with one input manifest. [Problem statement pp3-5]

Run the supplied validator with `--check-ids` and a separate strict validator. The supplied tool permits warnings for some candidate problems and defaults to skipping full ID existence; a PASS alone is insufficient. Require two-file existence, ID equality, valid target membership, duplicate checks, subset checks, scored-candidate hash agreement and full shard coverage. Validate the files extracted from the final archive, not only working copies.

```text
<team_name>_submission.zip
  output/matching_results.tsv
  output/candidate_pairs.tsv
  code/business_entity_resolution/src/...
  code/business_entity_resolution/README.md
  code/business_entity_resolution/requirements.txt
  Documentation_template.md
  approach_summary.pdf or approach_summary.md
```

The runnable code folder must contain everything needed to recreate both outputs from supplied data: scripts/configs, seeds, normalization, model/threshold provenance and exact commands. Package necessary trained artifacts when allowed and practical, or include deterministic retraining with documented runtime and pins; do not depend on a teammate's temporary private URL. Include required license notices and parameter counts. Optional external pretrained weights need a reproducible, rule-compliant availability story.

The guideline requests 1-2 pages while the problem requests the full template without a limit; supply a short summary plus the completed template. Keep submission versions and file hashes. Maximum five submissions per team per day; confirm remaining attempts and portal reset/selection behavior. Use submissions for a working baseline and independently justified improvements, preserving attempts for repair rather than sweeping thresholds on the public leaderboard. One designated active portal operator handles upload; no simultaneous participant logins or extra challenge identities. [Guidelines pp1-2]

Method research and coding help are allowed activities for this plan; the submitted entity-resolution pipeline must use the provided data. No business lookup, registry scraping, geocoding API, external labels, hosted resolver or fabricated score. The PDFs' private-versus-both-leaderboards wording differs; keep both results and optimize robust held-out performance. Organizer clarifications, if obtained, must be logged with their exact scope.

## 14. Sol's first response should be evidence, not another plan

Read this document and the current shared instructions. Inspect the repository for work completed elsewhere before duplicating it. Then create the package and tiny fixture, implement the exact scorer and strict writer, make the split immutable, and measure independent name/address retrieval on the full target catalog. Produce the first score/ceiling/runtime report and continue along the critical path. Ask only for an actual missing credential, unresolved spending authorization or necessary external action; do not ask the user to choose A/B/C/D or wait for equal contributions.

Use `coordination/prompts/SOL_START.md` as the handoff prompt. This planning revision has not launched Sol, published remote branches, trained a model, spent AWS credits or submitted files to the challenge portal.
