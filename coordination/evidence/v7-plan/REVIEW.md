# V7 planning review and branch consolidation

Session: astra-v7plan-20260927-1127. Date: 27 September 2026.
User scope: improve V7 for the execution teammate; consolidate branches to main; **no new implementation or training**.

## Deliverables

- Revised output/hackathon/V7_FINAL_ITERATION_PLAN.md, revision 2.
- Consolidated existing published V6 lineage, smoke fixes, stage-two release scaffold, validator, review and experimental evidence.
- Incompatible v3 pipelines preserved as exact reference snapshots; no replacement of canonical normalizer/blocker/feature APIs.
- Updated current-plan entry points and coordination/BRANCH_CONSOLIDATION.md.
- Original untracked draft preserved as V7_ORIGINAL_PLAN.md, SHA-256 9308e2bd2d7c9e3e58ce063a6a7fe703f8e21ed3bab143894e7d073ad92a59a9.

The integration claim was published at e40e11d71bfa2ee1705bfcea379df347e29ce447 after fetching control at 9714e61. No existing writer/job claim was transferred or released by this review. Live branches/worktrees were preserved.

## Principal planning changes

1. Coherent population-rival witnesses, separate name/address rivals, explicit self exclusion, missingness and capped-key censoring.
2. An incumbent-anchored logit mixture, bounded feature ablations and paired measurements rather than adding gains from incomparable experiments.
3. Development/policy-check separation; validate the exact saved deployment predictor, retain it after the check and disclose prior tune selection.
4. Correct treatment of the second audit look as a disclosed operational selection, not an untouched test.
5. Existing-code gap list: expected-F zero-survivor failure, OOF-versus-deployment averaging mismatch, stale resume, insufficient pair/source identity checks and missing-validator acceptance.
6. One published index/IDF specification, per-target context cache, persistent workers and a deadline calculation that includes setup, audit contention and contingency.
7. Exact candidate-set accounting and byte-identical reuse of the V6 candidate TSV only after scored-set equivalence.
8. Full model/dependency inventory, including the existing Unidecode license distinction; no unsupported compliance assertion.

Source review included SOL_IMPLEMENTATION_PLAN.md, V7 draft, startup/protocol/contracts, live control, published branch diffs, V6/stage-two/smoke evidence, v3 features, local ensemble/reverse code and the latest b607e93 release scaffold. Primary method/documentation links are in V7 section 11. No external business records were obtained.

## Checks actually run

From code/business_entity_resolution/, using the existing C:/Ml_challenge/.venv/Scripts/python.exe (Python 3.13.6):

~~~text
python -m pytest -q -p no:cacheprovider
  tests/test_metric.py
  tests/test_strict_validate.py
  tests/test_analyze_tune.py
  tests/test_splits.py
  tests/test_v6_stage2.py::test_survivors_top_k_and_floor
  tests/test_v6_stage2.py::test_features_use_confident_siblings
  tests/test_v6_train.py::test_cut_matches_stage2_survivors
~~~

PYTHONDONTWRITEBYTECODE=1. Result: **31 passed, 87 subtests passed in 15.09 seconds**.
These are existing non-training tests; no model fit or competition-data evaluation was run. Training/end-to-end fixtures were deliberately excluded to honor the user's restriction.

Additional checks:
- Syntax-parsed all 128 tracked Python files without importing/executing them.
- Verified all 57 v3-results and 14 Akash reference snapshot files against original Git bytes and their SHA-256 manifests: zero mismatches.
- Verified 54 inherited canonical package files against V6 or the explicitly integrated smoke/release contributor commits: zero unexplained changes.
- All 12 published code branch tips at the recorded fetch were ancestors of the consolidated head; live control deliberately remains separate.
- Scanned source/documentation/metadata for standard AWS access-key, private-key and presigned-signature patterns; no matching file reported. This is a bounded pattern check, not a claim to have audited existing public history.

## Scope and limits

No new model/retrieval/training source or test was authored. Existing source was merged or copied unchanged; new work consists of planning, documentation, provenance and Git integration. No dependency install, cloud launch, AWS mutation, real-data training/inference, new performance result or portal action occurred.

Prior teammate scores and runtime reports were reviewed, not independently rerun here. No claim that V7 reaches 0.98, that the reference experiments are release-ready, or that the current stage-two scaffold satisfies the revised gates.

The final accepted main SHA, completed integration state and handoff are recorded on codex/control after successful publication. See the branch map for exact source SHAs and disposition. Publication does not authorize another session to launch compute.
