# Stage-two release and four-person coordination implementation plan

> For agentic workers: implement task-by-task with superpowers:executing-plans when implementation is authorized. This plan does not authorize Git writes, dependency installs, cloud spending or teammates' actions. No automatic subagent dispatch.

**Goal:** Produce a reproducible improvement over the validated 0.9605 / 30-candidate fallback, with truthful candidate accounting, complete outputs and bounded compute.

**Architecture:** Retain v5.1 retrieval and existing score artifacts. Compare the second-stage sibling matcher on exactly specified survivor policies; make empty and small candidate sets safe. Reuse independent accounts for distinct tasks, not a newly constructed cross-account cluster.

**Tech stack:** Existing Python, Polars, NumPy, LightGBM, RapidFuzz, private S3 and SageMaker notebook workers. No additional dependencies proposed.

**Spec:** coordination/evidence/stage2/claude-20260927-1130.md at db4056b on codex/stage2-claude-20260927-1130; previous handoff at fdcfa30; organizer clarification sheet rows 133, 672, 811; user constraints in AD_Known.md.

## Global constraints

- Deadline 27 September 2026 23:59 IST; internal acceptance target 21:30 IST. Recalculate remaining time at execution; do not wait for every account.
- Preserve AML_submission.zip fallback, tune F0.5 0.9605, 30 candidates/S1. New branch is research, not a completed release.
- Source data only; no external identity data or hosted matching API. Keep raw records/models/private logs outside Git.
- Fixed 100k tune cohort; audit remains locked until champion selection. Stage-two cross-fitting now uses tune labels for training: call its estimate OOF, not untouched holdout.
- Every test S1 appears once in both output files; predictions subset candidates; validate existing IDs and duplicates.
- User controls Git state changes and dependency installation. A plan is not permission to push. Changes spanning more than 2-3 implementation files need user confirmation before edits.
- Quota requests do not launch compute. Paid compute needs exact machine/count, task, expected duration/cost and shutdown approval.

## Evidence and release gates

| Configuration | Evidence | What may be claimed |
|---|---|---|
| Existing cosine top30 | Released, recorded validator PASS | 0.9605 tune, 30 candidates |
| Stage2 top10, no floor | Two-fold OOF | 0.96994 threshold / 0.97052 expected-F |
| Top12 with p>=0.001 | Stage1/oracle analysis | 6.67 mean candidates and U 0.99481; stage2 score unknown |
| Reverse competition features | Partial execution only | No demonstrated gain |

Do not combine the 0.97052 result with the 6.67 candidate count until the same run measures both.

## Review focus

1. Zero survivors: include the query in metric denominators and output with empty lists.
2. One/two survivors: probability summary and sibling features must not index beyond the group.
3. Ties and target competition: deterministic candidate order and consistent final ownership handling.
4. Missing shards / stale outputs: fail a manifest check rather than silently evaluate a partial population.
5. Candidate-count interpretation: disclose broad first-stage scoring as well as final survivors; keep fallback if clarification remains unresolved.

## Ownership: proposed, confirm actual work before claiming

| Teammate | Bounded ownership | Deliverable |
|---|---|---|
| Aditya | Stage2 training and exact-policy comparison after core fixes | Frozen model/config + OOF report |
| Akash | Small/empty-set implementation fixes and final survivor policy integration | Passing edge cases + deterministic policy |
| Aadish | Independent metric/reproduction review, France and ownership checks | Findings and validation report |
| Abhigyan | Account/artifact inventory, transfer coordination, release integration | Verified inputs + final manifest/package |

Existing active tasks take precedence; obtain explicit transfers before touching another owner's files. Aadish may author independent tests, but nominate one writer to integrate each shared module. Four contributors does not require four active paid machines.

## Task 1: freeze inputs and coordinate artifacts (Abhigyan)

**Files:** coordination state/evidence through the team's authorized publisher; private run manifest in S3. No source edits needed.
**Consumes:** current account/bucket/role identifiers, db4056b and v5.1 artifact manifests.
**Produces:** one agreed artifact inventory and owner per task.

- [ ] Collect from Aditya and Akash: account ID, Mumbai bucket, role/principal for access, current code SHA, exact running task, instance types/count/status and output prefix. Collect Aadish's environment and required inputs; do not demand a full account setup for read-only analysis.
- [ ] Preserve existing plan.json, model/model_manifest.json, model/tune_scores.parquet and all 88 full-union score/ test shards under shared/er-v51-20260927/. Retain feat/ only where the assigned experiment requires it and retain the cascade fallback artifacts separately.
- [ ] Correct the OLD handoff copy recipe: its score/* exclusion is wrong for these v6 analysis scripts. Arrange scoped access or copy the required score shards, not a needless feature rebuild.
- [ ] Record source keys, sizes, available checksums, dataset identity and code SHA. Confirm 88 distinct expected test shard keys, disjoint intended S1 ranges and aggregate 1,732,544 test queries. The reported full-union test pair count is 443,792,782; investigate mismatch instead of silently accepting partial data.
- [ ] Give each experiment a unique output prefix and immutable configuration. Do not reuse worker control queues by copying Aditya's startup defaults unchanged.

**Acceptance:** required inputs readable and inventoried; no paid machine needed to establish bucket permissions. Abhigyan's currently allowed 4 GiB notebooks are not approved for the full job; use teammates' suitable machines or verified local capacity while quota is pending.

## Task 2: make the proposed shortlist executable (Akash, or current code owner)

**Modify:** code/business_entity_resolution/src/v6_stage2.py; tools/v6_analysis/expected_f.py.
**Test:** code/business_entity_resolution/tests/test_v6_stage2.py.
**Interface:** survivors(scored, k, floor) keeps deterministic order; features(...) retains identity columns and feature schema; expected-F evaluation scores ALL query IDs from tune_per.parquet.

- [ ] Add tests covering 0, 1, 2 and 3 candidates after a probability floor. Reuse records/text_views fixtures. For a one-survivor group assert p_2=p_3=0; for two assert p_2 equals the second probability and p_3=0. Verify no self-sibling and finite/sentinel features.
- [ ] In features, replace unsafe probability indexing with supported null-on-out-of-bounds access before filling nulls:

```python
pl.col("p").sort(descending=True).get(1, null_on_oob=True).over("s1").fill_null(0).alias("p_2")
pl.col("p").sort(descending=True).get(2, null_on_oob=True).over("s1").fill_null(0).alias("p_3")
```

Confirm compatibility with the installed pinned Polars version; if unsupported, implement guarded group extraction without changing dependencies. For wholly empty input, return a typed zero-row feature frame consistent with nonempty feature schema.

- [ ] Remove expected_f.py's requirement that every query have survivors. Evaluate present groups, assign zero predictions to absent query IDs and left-join back to the complete tune query population. Use the full truth counts; do not discard unretrieved positives.
- [ ] Pin the empty-query metric behavior with a pure metric assertion:

```python
# Empty truth + no prediction scores 1; nonempty truth + no prediction scores 0.
np.testing.assert_array_equal(per_query_f([0, 0], [0, 0], [0, 2]), [1.0, 0.0])
```

- [ ] For an all-empty batch, bypass probability optimization and emit empty selections with query coverage intact. Add equal-probability candidates in permuted input orders to verify target-row tie-breaking remains deterministic.
- [ ] Run existing and added tests in the already configured team environment:

```text
python -m pytest tests/test_v6_stage2.py -q
```

**Acceptance:** 0/1/2-survivor cases pass, no missing query rows, no dependency upgrades. These fixes were identified by static review here, not reproduced with installed runtime dependencies.

## Task 3: compare exact policies and produce a model (Aditya)

**Files:** tools/v6_analysis/stage2_features.py, stage2_cv.py, expected_f.py; private outputs and evidence report.
**Consumes:** validated Task 2 code and complete frozen score artifacts.
**Produces:** exact config, OOF predictions/metrics and, only after selection, serialized final stage2 model.

- [ ] Reproduce top10/no-floor as the reference using existing scripts, then run the proposed top12/0.001 configuration:

```text
python tools/v6_analysis/stage2_features.py tune 10 0
python tools/v6_analysis/stage2_cv.py K10_f0
python tools/v6_analysis/expected_f.py s2_oof_K10_f0.parquet p2
python tools/v6_analysis/stage2_features.py tune 12 0.001
python tools/v6_analysis/stage2_cv.py K12_f0.001
python tools/v6_analysis/expected_f.py s2_oof_K12_f0.001.parquet p2
```

- [ ] For BOTH policies record F0.5, oracle, total/mean/p95 candidates, singleton and truth-count-1 performance, country breakdown, runtime and peak RAM. Apply identical target-ownership logic to both for a fair comparison; report before/after ownership when material.
- [ ] Compare per-query paired deltas and confidence intervals across S1, not candidate rows. Treat repeated tune selection as potentially optimistic; no guaranteed leaderboard gain.
- [ ] Select a policy on measured score/size tradeoff, without inventing the organizer's ranking formula. Freeze its K, floor, normalization, feature order, split hashes, selection method and tie-breaking.
- [ ] Implement/verify the final fit-and-predict entry point: train on the allowed stage2 training cohort, save the LightGBM model, exact feature list and configuration, reload it and match predictions to an in-memory model on a small batch. This release entry point is NOT supplied by stage2_cv.py, which only writes OOF artifacts. One owner must implement it and record its actual command before launch.

**Acceptance:** no claim of seven-candidate stage2 quality without direct measurement; reproducible saved model and reload check; audit unopened during selection.

## Task 4: independent review and rule accounting (Aadish)

**Files:** tools/v6_analysis/_common.py, loss.py, test_checks.py and residuals.py (review existing logic first); separate evidence document. Coordinate any shared test changes with Task 2 owner.
**Consumes:** exact-policy OOF predictions, full query/truth counts, stage1/stage2 scores and chosen configuration.
**Produces:** independent metric agreement, coverage/ownership checks and risk report.

- [ ] Independently recompute macro F0.5 from predicted ID sets against complete truth. Include singleton and zero-survivor queries; compare with script output within floating-point tolerance.
- [ ] Check target-ownership ties: loss.py's equality-to-maximum flag can retain multiple owners. If final prediction uses one owner, ensure deterministic tie-breaking and align reported evaluation with that final policy.
- [ ] Run France distribution checks on actual chosen stage2 outputs, not merely earlier stage1 scores. Report empty rates, predicted counts, very large match sets and ownership collisions by country. France has no labels: distributions are diagnostics, not proof of accuracy or justification for arbitrary country thresholds.
- [ ] Resolve candidate reporting against organizer responses. Row 133 says first scoring-model input; row 672 permits a later pruned stage after a model filter; row 811 reviews code/computation. Record both broad stage1 pairs and final stage2 survivors. Seek a precise organizer clarification if needed; preserve the validated fallback rather than falsely asserting only seven model evaluations occur.
- [ ] A pure S2-CUT release that only filters already-scored predictions requires particular scrutiny: do not merely relabel existing broad-model outputs as a new tiny input set. Document actual inference execution and candidate accounting.

**Acceptance:** discrepancies resolved or disclosed before promotion; no audit-based model shopping.

## Task 5: integrate, audit once, validate and hand off (Abhigyan + nominated release writer)

**Files:** existing final assembly/validator/package tooling, release manifest and Documentation_template.md. Determine actual integration files after Task 3's inference entry point exists; do not assume the old v5 worker automatically executes v6.
**Consumes:** frozen tested model/config, approved candidate interpretation and complete input manifest.
**Produces:** one reproducible ZIP and report, retaining previous fallback.

- [ ] Smoke-test a bounded test batch containing zero/small survivor groups before full inference; measure memory before selecting worker count.
- [ ] Reserve and record the exact approved job/account/owner/cost estimate/shutdown behavior. Run on an already suitable teammate account if Abhigyan's increase is still pending. Avoid a three-account setup dependency.
- [ ] Evaluate the selected champion on the locked audit once with the full intended candidate-generation, stage2 and ownership pipeline. No tuning on this result. If validation fails, retain fallback and report failure.
- [ ] Score all test queries with the frozen configuration; assemble outputs by left-joining the complete test S1 list so absent survivor groups become explicit empty lists. Reject missing/duplicate shard inputs.
- [ ] Run the official validator with --check-ids and separately enforce matches subset candidates, exact ID coverage, duplicate-free lists and deterministic ownership if adopted. Hash artifacts and record exact reproduction commands/code version.
- [ ] Package both TSVs, executable source, pinned requirements and filled methodology. Include stage1 broad cost, survivor policy, OOF versus audit metrics and France limitations honestly.
- [ ] Stop idle compute, retain required artifacts and hand the package to the human portal operator. No agent portal upload or Git push implied by this plan.

## Optional work, only after the release path is working

Reverse-retrieval competition features (src/v6_reverse.py) remain unproven and relatively expensive. Assign one suitable worker and a bounded tune experiment only if the team has time/budget; require paired gain on the same survivor policy before integration. Do not schedule a full normalization/feature rebuild merely because extra accounts exist.

## Checks performed while preparing this plan

All 13 added files at db4056b reviewed; 12 Python files syntax-parsed. Existing local runtime lacks Polars/RapidFuzz/LightGBM/pytest, so no unit tests or metric reproduction were run here and no packages were installed. Organizer clarification rows were reread. No cloud actions taken. Implementation remains pending; this document is a proposed revision for team review, not an implementation task claim.