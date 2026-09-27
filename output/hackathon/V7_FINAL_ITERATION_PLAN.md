# V7 final implementation and training plan — reviewed revision 2

**Status: specification for the execution teammate; V7 has not been implemented or trained by this review.**
Reviewed 27 September 2026. Time check at 17:03 IST: **4 h 27 min to internal acceptance (21:30), 6 h 56 min to official close (23:59)**. Recalculate at execution start. The original draft's future-dated 17:15 timestamp is not a start-time constraint.

This is the current technical plan. [The Sol plan](SOL_IMPLEMENTATION_PLAN.md) remains background for metric, retrieval and competition rules. [The branch map](../../coordination/BRANCH_CONSOLIDATION.md) identifies the consolidated sources. The [original V7 draft](../../coordination/evidence/v7-plan/V7_ORIGINAL_PLAN.md) is preserved as historical evidence.

The user authorized this session to **plan and consolidate existing branches only**. No job, training run, inference experiment, dependency installation, credential change or portal action is authorized by this document. The execution teammate claims the implementation paths and obtains any still-required compute authorization separately. An existing claim is not transferred by this plan.

## 0. The decision

1. Keep the packaged **V6 three-model ensemble (ens3)** as the incumbent: reported audit macro F0.5 **0.97089**, candidate oracle **0.99408**. Preserve its files and models byte for byte.
2. Keep exactly the V6 candidate pairs: v5.1 probability top 12 per S1, ties by target row, then p1 >= 0.01. **No new retrieval or candidate pruning in V7.**
3. Learn a small correction using V6 scores, sibling evidence, and **whole-population rival evidence**. Compare coherent rivals, not a fictitious rival assembled from different records' best attributes. Decoy-cluster features are a separately measured addition.
4. Select architecture and policy on one development subset; check one frozen, deployable model on a reserved policy subset. Reuse the previously opened audit only as a disclosed, pre-registered final comparison. No claim that repeated tuning has become an untouched test.
5. Use one canonical code SHA, one feature specification and one global collector. Existing stage-two code is a scaffold with known gaps, not an approved V7 launcher. Launch full scoring only if the measured end-to-end finish time fits the release buffer.

The target requires +0.00911 over V6, or about **39.3% of the remaining matching/selection gap** (0.99408 - 0.97089 = 0.02319). This is substantial. Existing sibling and expected-F gains were measured on a weaker base and **cannot be added** to V6's score. There is no defensible forecast of a V7 audit range or proof that V6 exhausted pair learning.

## 1. What is established, and what still needs evidence

All numbers here are **published teammate evidence**, not results reproduced by this planning review on competition data.

| Source pinned in main | Supported finding | Use |
| --- | --- | --- |
| V6, 60ec04d | Tune 0.97061, audit 0.97089 on 220,682 S1; tune U 0.99407, audit U 0.99408; packaged outputs and official validation recorded | Frozen baseline and p6 inputs |
| Stage two, db4056b; reproduced at a35aead | Earlier top-10 v5.1: 0.96780 to 0.96994 OOF; expected-F 0.97052; one France feature shard completed | F1 feature implementation and measured feasibility, not an incremental V6 gain |
| Release scaffold, b607e93 | Synthetic gate/predict/collect tests reported; real ens3 tune gate not run | Reuse its IO and release structure after section 3 gaps are addressed |
| v3-results, 44cb55c | Reported 10k check score 0.9737, U 0.9904 and 37 candidates/S1 in that lineage | F2 source reference; different candidates and cohort |
| v5.1 local experiment, ca522df | Reported reserved 15k second-stage development score 0.97647081 vs same-query 0.96776538; broader top-16, 0.0001 survivors | F3 source reference and hypothesis, not a V6 comparison or fresh audit |
| Strict validation, 220ff35 | Streaming validator, invented fixtures and prior comparison evidence | Release verification |

Apply the same evidence standard to every contributor. Source inspection, a published run report, independent reproduction and organizer scores are distinct evidence levels.

**A0, optional, at most 15 minutes:** reproduce fixed V6 and v5.1 means on the historical check IDs to contextualize the claims. Recover IDs from exact manifests; never assume two row numbers mean the same S1. For v3 the specified check derives from seed-42 rows 200000:220000 then seed-7 second half. For ca522df it derives from seed 27092026, tune plan order, positions 85000 onward. Assert membership and ID-map identity. A claimant's mean alone permits a mean comparison, **not a paired uncertainty estimate or error-complementarity analysis**. Those require both per-query vectors. A0 does not delay V7 or alter the pre-registered architecture list.

## 2. Inputs and immutable boundaries

### 2.1 Known pins

| Artifact | SHA-256 |
| --- | --- |
| V6 ZIP, AML_submission_v6.zip | 7d66db4163f24c674282e1b94aa866c190f94b6532f7d1caec55ea88e98f0778 |
| V6 matching_results.tsv | 3f69fe5b983dc1407acfa3a45fb55dc59fce8a3d334f153961d0868521715f09 |
| V6 candidate_pairs.tsv | 9c438703e605ac9d5fc4c0931b1c2272743054e75aeeb019b95891a2c83a9872 |
| V6 ensemble identity | d0fa7781e62a1a4226d7b9e40fe2ad65529db2af5503766b2691ace0a186ff63 |
| v5.1 base model | eec93e350d320b46453d69f47010f07883c0b3e5eb6dd47d2dda0153f46cd982 |
| V6 member model | b7e7db652d4ab6519a243fb590b9d02ddb20336c38df6cf2b5a3f6194a11523d |
| V6 seed-7 member | ad57c4f219a7e470801e8ae822e6b4395848dc3d7f2eeb3921daa129057ef255 |
| V6 127-leaf member | c93f3b87a34e2f2e0594520becebb28374b4653b787bfdcc0ec694ca91b38271 |

The full v5.1 hash above is also recorded in the published release report and experiment preparation metadata. Resolve exact data and row-map hashes, member tune-score hashes, and every score-shard hash from the private manifests before execution; verify all listed model pins against the actual objects. Abbreviations are not integrity checks. Preserve private storage and scoped access; Git receives aggregate metadata only.

Required private inputs:

- Canonical v5.1 plan and model manifest; all three member manifests and tune scores.
- Ens3 plan/manifest and its **88** test score shards, plus authoritative query coverage per shard.
- Supplied train/test S1 and S2+S3 row maps and data checksums.
- Existing v5.1 audit feature shards and audit-row manifest, read only for the frozen final comparison.
- The packaged V6 candidate file for byte-identical reuse after pair-set verification.
- Per-member test probabilities only if the chosen F0 configuration uses member spread. If obtaining them would delay the run, omit spread from **every** split before model selection; never substitute zeros on test.

Test invariants: **1,732,544 S1**, including **259,452 France S1**; **9,266,800 pairs**; reported **23,644 zero-survivor S1**. Nonempty score rows cover only 1,708,900 S1. Missing query IDs in Parquet do not establish a coverage failure when the shard/query manifest proves an intentionally empty survivor set.

Require uniqueness of (s1, t), one-to-one t-to-target_id mapping, source2-then-source3 ordering, finite probabilities in [0,1], and aligned p1/target IDs across members. Compare member keys before averaging; never align by physical row order alone. The train and test row maps are separate namespaces.

### 2.2 Candidate accounting

The frozen candidate set C(q) is established with **p1**, before V6 or V7 context ranking. Reranking by p6 or p7 must not remove candidates or change C(q). Score each pair in C(q) once with the declared final matcher.

Whole-population key hits are label-free context comparisons, not new output candidates; do not silently apply the final classifier to extra (S1,target) pairs. Record broad stage-one pair counts, survivor counts and reverse-context comparisons separately. Describe the entire cascade honestly; a small final candidate file does not mean only 9.27M comparisons occurred. Preserve the organizer clarification provenance concerning model filtering; do not invent a scoring formula for the candidate-size criterion.

**Copy V6's candidate TSV unchanged only after** comparing its canonical pair-set digest with V7's scored-input set. Re-sorting candidate IDs by p7 can change the file hash despite equal sets; preserve original bytes and ordering. Matching TSV order is deterministic and separate.

## 3. Mandatory changes for the execution teammate, discovered by static review

These are specifications, **not fixes made in this planning session**. Existing source is preserved on main.

| Location | Current behavior / risk | Required implementation acceptance |
| --- | --- | --- |
| src/v6_stage2.py | a35aead fixes one/two-survivor indexing; original sibling features use the supplied rank | Retain the fix; after choosing C with p1, recompute rank from (p6 descending, t ascending) before F1. Changing only the p column leaves stale sibling ordering |
| tools/v6_analysis/expected_f.py | Still asserts every tune S1 has survivors; K is a global; all-empty input is unsupported | Full S1 ledger, typed empty outputs, local K, deterministic ties; otherwise omit expected-F entirely |
| tools/v6_analysis/stage2_release.py at b607e93 | Gate tunes and judges policies on the same OOF cohort; accepts any gain > 2 SE | Implement sections 5-6, fixed V6 threshold 0.72 as the primary comparator, practical margins and slice checks |
| Same, cross_fit/write_release | Gate uses one held-out fold model per query, while deployment averages both models | Do not treat these as the same predictor. Validate the **actual frozen deployment model and policy** on the policy-check subset; keep that exact model afterward |
| Same, predict/resume | Presence of a JSON file causes an unconditional skip | Resume only after checking input, output, code, model, feature, assignment and logical pair hashes; write atomically |
| Same, collector | Verifies row counts and common Git HEAD, but not complete original pair identity or all executable hashes | Reject same-count altered pairs, dirty/unknown or mismatched code, wrong target IDs, nonfinite scores, swapped shards and wrong workers |
| Same, collector | Missing official validator can be accepted via validator_exit=None | Release requires observed exit 0 with --check-ids, plus strict checks; unavailable validation is incomplete |
| v3 features3.py | Calls v3 base.build; assumes translit and v3 normalized columns exist | Port only specified feature calculations through an explicit v5 adapter. No v3 base call, no fabricated translit flag or overwritten canonical normalizer |
| ca522df reverse.py | Uses version-specific Polars hashes, capped keys and sampled rank-union counts | Stable structured keys, explicit censored/missing flags, correctly defined counts and coherent competitor features |
| Runtime/release | Original proposal assumes Python 3.12, byte-identical Parquet and an unmeasured <$10 bill | Pin the actual compatible environment; distinguish semantic digests from container bytes; measure full cold/warm cost and authorized capacity |

In the existing release scaffold, reranking survivors by ens3 probability already exists. Reuse it while adding features; do not regress to the draft's ambiguous "replace p only" instruction. The broader expected-F helper remains unfixed; the newly published release scaffold currently uses thresholds.

## 4. Features: a bounded correction of the incumbent

All features use permitted supplied records and fixed base scores. Raw IDs, row indices, fold IDs, label counts, owner labels and country-specific memorized lookup tables are forbidden model inputs. IDs may index, split and deterministically break ties. Context is generated identically for training and inference; feature computation has no truth dependency.

### F0 — scores and uncertainty

p6 = mean of the three V6 probabilities; retain p1 separately. Include clipped logits (epsilon 1e-6), p6 rank/gaps, best/second/third/sum, survivor count and counts above 0.5/0.9. Optional member standard deviation and range require complete aligned member scores for every split. Prefix feature names so F1's historical name "p1" cannot overwrite real v5.1 p1: in F1 it now means the p6 input.

**Incumbent-preserving correction:** evaluate one global mixture
s = sigmoid((1-alpha) * logit(p6) + alpha * logit(p7)),
with alpha in {0.5, 1.0}. Pick alpha and the decision policy only on development predictions, then freeze. This is an unproven engineering safeguard, not a guaranteed improvement or a calibrated posterior. No automatic copying of V6 high-confidence decisions: some errors are confidently wrong. Every final candidate is still scored.

### F1 — sibling evidence, with a fixed context set

Reuse the tested 61-feature family on C(q), reranked by p6. Whole query groups stay in one feature batch. Compare at most the five highest-p6 **other** survivors; never compare a target with itself. Keep missing text as missing evidence, not perfect agreement. Include one-/two-survivor sentinels and a typed zero-row schema.

A high-scoring decoy can make its siblings appear consistent. Therefore sibling agreement alone is not a hard acceptance rule; the F2/F3 evidence below lets the model distinguish agreement around the wrong business.

### F3 — population rivals, highest-priority new evidence

**Population:** all 2,206,821 train S1 records for train-side context, all 1,732,544 test S1 for test context. Audit and policy-check S1 may appear as **unlabelled records** in the train index; their labels and truth-derived aliases may not enter it. This is a declared transductive feature policy. No within-100k probability-competition feature is used.

**Index specification:**

1. Use the pinned text_norm_v5 views: core, sorted core tokens, concat, skeleton, addr_norm and nonempty house+addr_tok. Define a structured (country, view_id, text) key; empty components are not a shared key. Minimum key length 5 is a fixed initial policy.
2. Use canonical UTF-8 length-prefixed components or a stable digest of them, retaining collision verification. Sort by key and source row. Do not use Python hash() or Polars hash() as a cross-machine contract.
3. Count postings over the full relevant S1 population. For keys with >20 S1, suppress expansion but retain key-hit and overflow indicators. Dropped common keys mean **censored search**, not "no rival exists".
4. Deduplicate the union of hits per target. Compute cheap comparisons once per distinct target and cache them; at six capped views there are at most 120 pre-dedup hits per target. Keep exact and overflow counts as diagnostics.
5. Exclude the current q before reducing competitors. A per-target top-two cache per criterion is enough to recover its best other q after self exclusion; retaining four is acceptable, but "count" must not then mean a count of the truncated cache.

**Coherent rival features — the main creative change:**

For each actual rival r, calculate name ratio, token-set ratio, address ratio, address token-set ratio and joint sqrt(name_ratio * address_ratio), with common [0,1] units and explicit validity masks. Select one best **joint** rival (ties by stable S1 row) and retain its entire name/address vector, key-hit types and q-minus-r gaps. Also retain the strongest name rival and strongest address rival as **separate witnesses**, each with its own vector. If different rivals win name and address, expose that fact.

Do not combine rival A's name maximum with rival B's address maximum into one implied perfect competitor. Independent maxima can remain supplemental features, clearly distinguished from same-record joint evidence.

For missing addresses, use the name-rival path and address-availability flags; a zero joint score is not evidence that competition is absent. Exclude empty/empty comparisons from similarity maxima. Gaps compare q and r with exactly the same scorer, text view, scaling and missingness rule.

**Default model inputs omit raw competitor population counts.** Keep rival presence, observed strengths, gaps, key evidence and overflow/missing flags. Population density and key suppression affect maxima too; omitting counts does not make the feature shift-free. Report zero/present/overflow rates and conditional p50/p90/p99 by country, missingness and key view. Do not compare a mostly-zero overall median to a 1.3x rule. France has no labelled analogue; US/India distributions cannot establish French accuracy.

### F2 — decoy clusters and token differences, second priority

Source: experiments/reference_v3_results/src/features3.py at 44cb55c. Reuse the calculations behind NEW_FEATURES, not its v3 pipeline wrapper.

- Directional IDF-weighted token coverage, extra/missing token mass, and differences that remain after matching plausible typo variants.
- Number overlap and house prefix/edit/digit structure as learned evidence; no universal number veto.
- Leave-one-out counts of other survivors sharing name core, house, address or a genuinely different token, with own-query agreement distinguished from disagreement.
- Record when apparent support is repeated copies of the same normalized signature. Do not describe many identical decoy records as independent confirmations.

Define all field adapters explicitly: v5 does not supply v3's translit column; omit that flag rather than inventing an equivalent. Retain the original fuzzy-token cutoff (75 on a 0-100 scale) for this bounded experiment. Group self contributions must be excluded consistently, including extra-token support.

IDF is label-free on the **full target catalog of the relevant split**, with one pinned country policy for both train and test. Cache once per split; do not rebuild per shard or estimate it from a survivor sample. Until benchmarked, F2 is optional: its token cross-products can dominate runtime.

### Ordered experiment list

B0 = frozen V6 at threshold 0.72. B1 = F0+F1. B2 = F0+F1+F3. B3 = F0+F1+F2.
Run B1, then B2; B3 only if F2 is ready and time remains. Test B4 = all families **only if both B2 and B3 beat B1 by >=0.0003 on development**. This prevents the old cumulative ladder from hiding a useful F2 behind harmful F3. An F0-only diagnostic is optional and never delays B1. No broad hyperparameter search.

These are hypotheses. A gain in an earlier lineage is not an additive gain here. Record each marginal score change, pair count, feature time and lost-score slice; skip optional rungs explicitly.


## 5. Training, splitting and selection protocol

### 5.1 Separate development choices from the deployment check

Freeze manifests before examining any new V7 outcome.

1. Reconstruct the canonical 100k tune population by ID. Base models were not fitted on these queries, but their variants, survivor cut and thresholds **were selected on this population**. V7 cross-fitting cannot erase that history.
2. Form business groups from shared labelled targets and repeated exact identity fingerprints. For the head's splitting, also keep queries sharing a survivor target in one component when feasible, so rival edges do not cross folds. Include zero-survivor queries. Report largest component and country/singleton balance. If this creates a giant component, document it and use the established business groups with held-out-owned targets excluded from head-fit negatives; do not split a known identity for convenience.
3. Deterministically allocate about **80k development D / 20k policy check C** by stable group hash and country/match-count strata. Commit group policy, seed, ID-list hashes and actual counts. C is withheld from **new V7 choices**, not advertised as historically untouched.
4. Use **3-fold grouped cross-fitting within D**, with an inner 10% of each training fold for early stopping. Every OOF score must come from a model that saw neither that query's labels nor its business group. Label-free F3 population/index and target IDF may be shared.
5. Choose feature family, alpha and decision policy using D OOF only. Training objective is unweighted binary loss; selection objective is exact macro per-S1 F0.5. Report calibration and scores by query length/missingness. Pair loss alone is not the gate.
6. Fit **one final head on D**, with rounds equal to the rounded median of inner-fold best iterations, at least 1. No arbitrary 1.1 multiplier. Freeze the actual model, all transforms, alpha and policy; verify reload predictions.
7. Evaluate that exact deployment head once on C. Do **not** refit on all 100k afterward. This deliberately trades 20k fitting queries for validation of the real release predictor and avoids the fold-average calibration mismatch in b607e93.
8. If C fails, stop V7. Do not try the runner-up or change thresholds after viewing C. Passing D/C remains development evidence because of historical upstream selection.

Default LightGBM recipe: binary; learning_rate=0.05; num_leaves=63; min_data_in_leaf=40; feature_fraction=0.8; bagging_fraction=0.8; bagging_freq=1; lambda_l2=1; at most 4000 rounds; early-stopping patience 100; fixed recorded seeds. Use the team's pinned CPU version, deterministic=true and force_col_wise=true. Record actual thread count. These settings do not guarantee identical results across different library builds or operating systems.

**Unexposed extra training rows are not automatically a new clean holdout.** They may have contributed targets as fit negatives or appeared in prior experiments. Creating a new confirmation partition requires an exposure audit and fresh full-catalog retrieval; it is outside today's critical path.

### 5.2 Decision policies

First optimize a **single global threshold** on D OOF correction scores: coarse step 0.02, then step 0.0025 within the best coarse neighborhood; pin endpoints and tie rule. If scores tie in objective, prefer the simpler policy, then higher threshold. Keep frozen V6 at 0.72 as the comparator; a swept V6 threshold may be reported as a diagnostic.

Optional expected-F prefix policy is allowed only after the helper's section 3 defects are fixed:

- Use q = sigmoid(logit(s) + b), b in {-0.3, 0, 0.3}; no temperature grid today. Calibrate/choose only on D.
- For each candidate prefix k, compute the expectation of 5*TP/(4*k+G) under the independent Bernoulli distribution, by the existing dynamic-programming approach. The empty choice has utility product(1-q). Do not replace the expected ratio with a ratio of expected counts.
- Include k=0, every allowed prefix, variable-length groups and all-empty batches. Tie by smaller k, then stable target order. Evaluation always uses **complete truth**, including unretrieved positives.
- Independence and ignoring positives outside survivors are approximations. They are particularly risky for duplicate variants and ownership losers. Label the utility accordingly; no claim of Bayes optimality for this dataset.
- Adopt only if the chosen alpha/prefix policy beats that alpha's threshold policy by >=0.0003 on D, and does not create a material singleton or one-true-match regression. This is a pre-check, not permission to try alternatives on C.

Apply the same deterministic ownership order to each comparator: global highest final score per target, ties to lower S1 row, then the frozen decision policy. Losing edges remain in the **candidate file**. No forced prediction for any S1, no S1 cardinality cap based on the maximum observed training count.

### 5.3 Ownership limitations and uncertainty

F3 evaluates raw-record competition against the full S1 population. This addresses the severe feature-density mismatch in within-tune rev_* features. It does **not** reproduce the effect of global probability ownership across all 2.2M training S1 when only 100k have final scores.

Report each evaluation **before and after** ownership and the number of conflicting targets, winners changed and per-country effects. Keep ownership identical to V6; do not select a new ownership algorithm on a 4.5%-population cohort. Full-population test ownership remains a disclosed transfer risk.

For paired intervals, save per-query baseline/new F and resample whole business/collision groups, computing a query-weighted mean per replicate (1000 fixed-seed replicates). A single-query SE is only a diagnostic when dependencies remain. Report both raw point gains and group-bootstrap lower bounds. These intervals address sampling variability, not adaptive selection history or France shift.

## 6. Promotion gates, frozen before implementation experiments

| Gate | Required evidence / pass rule | Failure action |
| --- | --- | --- |
| G0 input and artifact identity | Full hashes resolved; exact row maps; 88 expected shard/query manifests; 9,266,800 pairs; 1,732,544 total S1 including explicit empty coverage; baseline tune reproduced at fixed 0.72 within 0.0001 | Resolve inputs; no V7 training |
| G1 code/feature contract | Section 9 fixtures pass; compatible normalization and versioned feature schema; deterministic context; no label path into features; input/output pair identity | Fix before experiments; no changed feature schema mid-run |
| G2 development | Chosen D OOF recipe beats fixed V6 on the **same D IDs** by >=0.0015; positive paired group lower bound; India and US point deltas nonnegative | No C/audit/full inference; V6 remains incumbent |
| G2b deployment policy check, once | Actual frozen D-trained release head on C: gain >=0.0010, paired one-sided 95% group lower bound >0; no material slice failure below | V6; no alternative selection on C |
| G3 previously opened audit, once | Same frozen model/policy, all 220,682 audit S1, same C(q) policy/full target catalog: gain >=0.0010 and paired group lower bound >0; baseline reproduces 0.97089 within 0.0001; no material slice failure | V6; no retuning |
| G4 compute fit | Cold start, input transfers, index build, audit, slow-country shard throughput, collection/validation/packaging all fit section 8 with margin on authorized resources | Drop optional work before freeze or retain V6 |
| G5 release | Every required check passes; exact candidate-file SHA matches V6; actual code/model/feature/IDF/index/policy pins match across shards; final archive extracted and verified | Repair mechanical defects only; otherwise V6 |
| G6 France operational guard | Complete France coverage; mean predictions in [3.1,3.7], empty rate [3%,9%], 8+ predictions <=2%; report score/gap/overflow shifts and ownership conflict concentration | Treat as a conservative rollback guard; V6, without retuning France thresholds |

Slice rule for G2b/G3: report India, US, singleton, one-true-match, missing address, weak name and common-key overflow. For the first four, reject a mean regression below -0.001 **or** a paired upper 95% bound below zero. Report sparse-slice uncertainty rather than hiding it. These are conservative team decisions; they are not organizer rules.

G6 is inherited as an operational envelope around the known incumbent. **France's truth count is unknown.** The 3.45 labelled-country mean is not France ground truth, and passing the envelope does not establish accuracy. Do not force French output proportions to mimic training.

The audit has already been opened for V6. A V7-vs-V6 go/no-go decision **is selection on the audit** even if pre-registered; the original phrase "nothing may be selected on the audit" was inconsistent. Allow exactly this disclosed second comparison, with no subsequent model/policy choice. Record both systems and the decision regardless of outcome. Do not present its nominal interval as an untouched confirmatory guarantee.

**Promotion and target attainment are different.** A valid gain can promote V7 below 0.98. Claim 0.98 only for a named measured partition that actually reaches it; a stronger evidence statement additionally requires its one-sided lower bound >=0.98. Neither establishes a private leaderboard or France score.

## 7. Implementation handoff and task boundaries

Use the consolidated main SHA, then one isolated task branch per writer. Fetch control and respect the live S2-MODEL and V6-TRAIN claims. Existing owner claude-abhigyan-20260927-1600 has the stage-two release paths at the review snapshot; this plan does not seize them. Coordinate path transfers in Git.

| Task | Scope and proposed paths | Concrete deliverable |
| --- | --- | --- |
| R0 inputs/protocol, first | Private manifests; coordination/evidence/v7-stack/ | Full input pins, D/C/fold hashes, pre-registration, authorized job/cap plan and current claims |
| R1 release hardening | Existing owner: tools/v6_analysis/stage2_release.py, expected_f.py if used; tests/test_v6_stage2.py | G1 fixes, exact pair identity, resume safety, actual-deployment policy check and public CLI help |
| R2 core feature adapter | Proposed src/v7_stack.py, src/v7_features.py and their tests; claim before creation | F0/F1 schema, canonical p1/p6 separation, fixed C(q), exact empty behavior |
| R3 rival features | Proposed src/v7_population.py and tests | Full-population stable index; coherent-rival F3; masks/censoring; bounded cache and measured cost |
| R4 optional decoys | Proposed src/v7_decoy_features.py and tests | F2 adapter, pinned catalog IDF and independent B3 measurement |
| R5 model/policy | One training owner; private model and metrics | B1/B2/(B3/B4), D OOF choice, frozen D-trained head, one C check; no all-tune refit |
| R6 audit | One authorized evaluator, no feature/model edits | Stream existing audit features, fixed p1 cut, V6/V7 paired per-query vectors and G3 result |
| R7 assigned inference | Frozen feature/model code only | Exactly assigned shards; benchmark and output manifest per shard |
| R8 collect/release | One collector under short integration/submission claims | Global ownership, final selection, reused candidate bytes, both validators, archive, documentation and human handoff |

R0/R1/R2 are critical. R3 is the first new evidence bet. R4 is optional and should be dropped first when the schedule tightens. One capable execution session can own the critical path; three accounts or four agents are not prerequisites.

The existing stage2_release.py commands are gate / assign / predict / collect, but its current gate is **not** the revised V7 protocol. Before any real run, the teammate records the actual tested commands (including manifest paths and flags) in a runbook. Proposed v7_* files and revised CLI contracts above do not yet exist.

Private release manifest minimum:
run_id; code SHA and executable file hashes; parent V6/v5.1 identities; input row-map/data/score hashes; candidate-policy and pair-set digest; D/C/fold/group pins; ordered feature names/dtypes/missingness; model hash/rounds/parameters; alpha and decision/ownership policies; index/IDF semantic and object hashes; exact runtime versions; license inventory; gate results; required shard ledger; budget/shutdown owner.

## 8. Runtime and deadline plan

### Build once, score in persistent workers

Prefer **one published, country-partitioned population index and IDF artifact per split**, with checksums and private scoped reads. Replicate to assigned workers. Independent builds are acceptable only with identical specification and **logical** digests plus a golden-feature check; Parquet compression/metadata bytes are not a semantic equivalence test.

Normalize/index all S1 for F3, but compute target reverse context once per distinct needed target. Cache by (dataset hash, normalizer hash, index hash, target ID). Process complete S1 groups in bounded batches; do not normalize ten million targets afresh for every shard. Retain a worker process/index across its assigned shards. Record raw/normalized text, joins, IDF, model and cache memory, not only the dense feature matrix.

Audit: existing v5.1 feature shards already exist. Stream p1 over them **once**, retain top12/floor0.01 with correct query boundaries, apply V6 members to survivors and compute V7 only there. Persist the lean audit survivor scores privately so the frozen paired comparison is reproducible. No full audit retrieval rebuild and no rescore for each feature rung. The former 20-30 minute estimate is a planning assumption until measured.

Before assignment, benchmark cold setup and at least a representative US, India and heavy France shard. Count IO, context joins, all features and all model scoring. Use measured seconds per pair and p95 shard time, with at least **30% contingency**:

remaining_wall >= setup + max_worker(assigned_shard_times) * 1.3 + collection_validation_packaging + portal_buffer.

At 2 minutes/shard, 88 shards on three equally fast free workers already require **58.7 minutes before setup, audit contention and the 30% margin**. The draft's 50-minute allocation was not safe at that rate. Do not assume three machines each have 64 GB or are available at the same time. A worker doing audit is unavailable for test work until its audit slot finishes. A 32 GB worker may be useful after measurement; no unmeasured 4 GiB claim.

Use actual authorized capacity and balances. The previous V6 job's $15 authorization does not authorize V7. Record a new unique job ID, account alias, max duration, cost cap, current instance state and shutdown procedure before launch. No current-price or "under $10" claim is made here.

### Absolute cutoffs, IST on 27 September

| Latest checkpoint | Required state |
| --- | --- |
| 17:45 | Execution owner has input pins/access, pre-registration and a measured resource plan; otherwise shorten optional work |
| 18:30 | F0/F1 and release-contract fixtures pass; F3 benchmark determines whether it fits |
| 19:00 | Feature set ready. Drop F2 if incomplete; drop F3 if its complete feature/throughput path is not ready |
| 19:25 | D choices frozen, D-trained model serialized; one C policy check completed |
| 19:45 | Frozen-model audit comparison completed or definitely on a measured path within release budget; no passing D/C model by this time means V6 |
| Measured latest start, never later than 20:00 without demonstrated faster capacity | Test scoring can finish by 20:45 with 30% margin. Start earlier when the forecast requires it |
| 20:45 | All 88 shards accepted; otherwise V6 remains the 21:30 incumbent |
| 21:15 | Both files, extracted archive, strict/official validation, documentation and hashes complete |
| 21:30 | Internal acceptance and human portal handoff complete |
| 21:30-23:00 | Mechanical repair/upload contingency for an already frozen, validated system; no new models or thresholds |
| 23:59 | Official close |

Model-independent test index preparation may overlap model development if separately authorized and useful within budget. Starting full test scoring while G3 is pending is allowed only after D/C freeze, with explicit disposable-work budget and no submission before G3 passes. It must not starve audit/release work.

Reassignment is explicit and versioned. Record the prior worker's completion/stop acknowledgment or fence its assignment before retrying a shard; do not infer a dead worker from elapsed time. The collector accepts one complete output per shard and rejects conflicting copies.

## 9. Tests and release contract for the teammate

Existing consolidation checks run in this review are recorded separately; the following additional checks are required **during implementation**.

- 0/1/2/3 survivors and an entirely empty shard: stable schema, finite values or specified sentinels, no self-sibling, no self-rival; every expected S1 retained in evaluation/output.
- Permuted pair order and tied p1/p6/p7: same frozen candidates, correct p6 ranks, same features and deterministic ownership/prefix decisions.
- Rival witnesses: one name-only rival and a different address-only rival do not become a single perfect joint rival; missing address is not equality; common-key overflow is not absence; duplicate key hits do not inflate rival counts.
- Label isolation: feature outputs unchanged when truth files are unavailable or their contents are replaced in an invented fixture. Review the full call path, not merely the feature function's signature.
- Split isolation: no business group overlap, no label-derived fit statistics from C/audit, no in-sample score used as an OOF label predictor; indexes remain label-free.
- Feature parity: fixed golden records produce identical feature names/order/missing masks and numerically equal values on every worker; no fallback to zeros for a missing feature family.
- Expected-F, if used: compare dynamic programming to exhaustive Bernoulli enumeration for small K, plus all-empty/missing-query cases. Test the metric independently: both empty=1, missed nonempty=0, singleton false positive=0, PDF case=10/14.
- Collector corruption cases: same row count but altered target, mismatched target_id, duplicate pair, wrong shard, missing empty-query coverage, NaN p7, dirty executable, stale assignment, mismatched input/index/IDF/model, duplicated worker output.
- Determinism: logical sorted pair digest exact; prediction tolerance pinned (initial maximum absolute difference 1e-6). Raw Parquet byte equality is required only when its writer/compression/metadata settings are pinned for that purpose.
- Reload: the saved model reproduces in-memory predictions on 1000 held-out rows within 1e-6; alpha/calibration/selection are serialized and exercised too.

Per-shard result: s1, t, target_id, p1, p6, p7 (and final correction score if distinct), sorted by (s1,t), plus sidecar manifest with expected query-range/ID hash, actual nonempty/empty coverage, source-object and logical pair hashes, output object hash, model/feature/policy/code/index/IDF pins, assignment version, worker, runtime, peak RSS and environment. Write temp files then atomically publish completion.

Canonical pair digest: sort numeric (s1,t); serialize a fixed-width little-endian uint32 pair stream under a header containing dataset/row-map identities; SHA-256 it. Pin ID-to-row maps separately and verify target_id for every pair. Include **empty query coverage** in a separate query-ledger digest.

Final assembly:
1. Verify exactly the 88 expected shard IDs and disjoint complete query assignments, not merely 88 arbitrary filenames.
2. Verify the union of scored pair keys equals the frozen V6 pair set. Equality of counts or max-12 checks is insufficient.
3. Apply ownership once across the complete test population, then the frozen policy. No shard-local ownership.
4. Copy the verified V6 candidate file unchanged; write matching results with every S1 once, including France and empty lists. Existing test S2/S3 IDs only, no duplicates, matches subset candidates.
5. Run official validator with --check-ids and require success. Run src/strict_validate.py; separately verify the scored-pair manifest, ownership and shard contract, which TSV validation alone cannot establish.
6. Package and extract to a fresh directory; validate **extracted** files and compare expected SHA-256. Include reproducible code, frozen model/configuration or fully specified permitted retraining, notices, full template and concise summary. No teammate-only temporary URL dependency.
7. Preserve V6 and every submitted version. Human portal operator checks remaining attempts (maximum five/team/day), uploads under participant rules and records actual SCORED/selection receipt. No receipt is inferred from upload completion.

## 10. Rule and license checks

Only supplied business records and labels are used. Web research informs methods only. No identity lookups, external corpora, hosted resolver, geocoding or generated labels enter features.

LightGBM is the existing model family; inventory every base/head model, its actual tree/leaf parameter count and license, and verify the combined submitted model against the <=8B and MIT/Apache-2.0 requirements. Do not substitute "no new model family" for an inventory.

The existing requirements include **Unidecode 1.4.0**, whose published package license is GPL-2.0-or-later. Record its role and notices and the challenge's distinction between model license and preprocessing dependencies. Do not label all dependencies MIT/Apache. If an applicable organizer interpretation disallows it, the release owner must resolve that known dependency before claiming compliance; silently replacing normalization now would change existing model inputs. No license interpretation was granted by this planning review.

Candidate-size accounting must document broad stage-one scoring, final survivor matching and reverse context. Keep sensitive artifacts and raw/error records private even though origin is public. Aggregate evidence, source code and checksums are the Git deliverables.

## 11. Review answers and research provenance

- **Second audit look:** usable as one disclosed operational comparison; it is neither an untouched audit nor a way to select several alternatives. Freeze first and keep V6 on failure.
- **Population F3 leakage:** permissible under this plan's declared label-free transductive policy; audit/test labels and truth-based rival lookup are forbidden. Mask self matches and censored retrieval. Whole-population raw context does not resolve probability-ownership evaluation mismatch.
- **Margins:** retain the practical +0.0015 development / +0.0010 confirmation bars, with an actual-deployment check and grouped uncertainty. Do not treat these chosen margins as a theorem.
- **Index distribution:** publish one canonical artifact if possible; independently built indexes need semantic equivalence. Stable key hashing alone does not force identical Parquet bytes.
- **Schedule:** persistent workers, cached rival comparisons and cold/warm measurement matter more than nominal account count. Drop F2, expected-F and extra rungs before risking the incumbent.

Method references, checked during this review (no external record data used):

- [scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html): stacked predictions must distinguish prefit/in-sample inputs from held-out predictions. Our D/C protocol is an engineering response to this and to the observed selection history.
- [Dembczynski et al., ICML 2013](https://proceedings.mlr.press/v28/dembczynski13.html): motivates set-level F-measure decisions; our independent-Bernoulli prefix policy is a restricted approximation, not a reproduction of the paper's optimal rule.
- [Polars hash documentation](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.hash.html): hashes are not guaranteed stable across library versions.
- [LightGBM parameter documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html): deterministic CPU mode and explicit histogram orientation reduce variation under a pinned environment; different versions/builds can differ.
- [Unidecode 1.4.0 package metadata](https://pypi.org/project/Unidecode/1.4.0/): license and package provenance.

No new V7 score, throughput, spend, training artifact or submission is claimed by this document.
