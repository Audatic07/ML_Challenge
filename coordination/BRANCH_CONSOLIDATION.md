# Branch consolidation into main — 27 September 2026

The user requested V7 planning review and consolidation of existing branches onto main, explicitly prohibiting new implementation or training. Integration used isolated managed worktrees, with the short V7-PLAN / INTEGRATE claim published on codex/control at e40e11d.

**Canonical pipeline:** code/business_entity_resolution/ follows V6. No new model algorithm or training code was authored for this consolidation. Existing incompatible APIs are preserved as reference snapshots instead of replacing the V6 normalizer, blocker or feature builder.

| Published branch | Exact source commit | Disposition on main |
| --- | --- | --- |
| aadish_v1 | a1d42bd6f884db922bf2b2d347f6006b1ed71b8d | Ancestor of canonical V6 |
| codex/retrieval-v4-astra-20260927 | 0bf0bbce3023a23fe953b58dd6546906d6894b86 | Ancestor of canonical V6 |
| codex/v5-dist-claude-20260927 | fdcfa30cc8040cd1caab29d1bbaf2be8e5c135d3 | Ancestor of canonical V6; operational source/evidence retained |
| codex/stage2-claude-20260927-1130 | db4056b04647948c307a93ca988534a21d06f5ee | Ancestor of canonical V6; sibling/reverse helpers and evidence retained |
| codex/v6-train-claude-20260927-1420 | 60ec04d4b6dcd34bfd0ae24b55df8768048abb86 | Canonical V6 pipeline/evidence |
| codex/s2-model-claude-newacct-20260927-1155 | a35aead2a3e421cf702ea9b5a4fb39d17d5aa83d | Merged existing small-survivor fix and smoke evidence |
| codex/s2-model-claude-abhigyan-20260927-1600 | b607e93e4c763debe1ee7f811a5d2ba59f0312a2 | Merged existing release scaffold/tests; known gaps documented, not release approved |
| codex/stage2-plan-review-20260927 | 1e75a4c339c77451490acfadbed5f2ab2259d7b7 | Historical review retained under docs/superpowers/plans/ |
| codex/submission-support-astra-20260927-0450 | 220ff352451b68f734fb0f3c2aa46b4d0f908415 | Strict validator, analysis helpers and documentation drafts merged |
| codex/v51-final-results-20260927 | ca522df85e3a1c2ddf1bc92c405babbe60041dc0 | Original experiments/v51_local_20260927/ tree merged |
| results/v3-v6-v7-claude | 44cb55cd631c2a1e6f74dcf1616696a2b95148b5 | Source and aggregate results preserved in experiments/reference_v3_results/ |
| akash/v3-pipeline | d5f3775e0164825839c5e97c71b7dd60a00867fc | Source and handoff preserved in experiments/reference_akash_v3/ |
| codex/control | Live coordination branch; fetched separately | Intentionally separate; never merge live claims into implementation history |

The two incompatible lineages use explicit ancestry merges that preserve the canonical tree, accompanied by immutable source snapshots and SOURCE_MANIFEST.json files. These are **reference integrations**, not claims that their alternative pipeline behavior has been adopted. Each snapshot file has an original Git source path and SHA-256. The original branches/history remain available.

Historical chat and diagnostic/train log files were left off the consolidated working tree. Source branches already containing them were not rewritten. No raw datasets, new weights, private scores, presigned URLs or credentials were added. Existing public Git history and LFS data were not rewritten by this task.

Live worktrees and source branches were not deleted or archived: some have active claims and unpublished work. New implementation starts from main and no longer requires hunting across those branches. Control retains live claims; local control helper branches are not alternative implementations.

Current entry points:
- [V7 plan](../output/hackathon/V7_FINAL_ITERATION_PLAN.md)
- [Planning/integration evidence](evidence/v7-plan/REVIEW.md)
- [Canonical package](../code/business_entity_resolution/)
- [Strict validator](../code/business_entity_resolution/src/strict_validate.py)

Publication uses a normal fast-forward main update after checks. The accepted main SHA and final claim release are recorded on the live control branch; no force push is used.
