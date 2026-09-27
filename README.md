# Amazon ML Challenge — consolidated working baseline

Start with [V7 final implementation and training plan](output/hackathon/V7_FINAL_ITERATION_PLAN.md), then fetch and read the live [control-branch state](https://github.com/Audatic07/ML_Challenge/blob/codex/control/coordination/state.json).

- [Canonical implementation](code/business_entity_resolution/) contains the published V6 lineage, stage-two fixes/scaffold and strict validator. V6 remains the measured incumbent.
- [Branch consolidation map](coordination/BRANCH_CONSOLIDATION.md) records every source branch, exact commit and disposition.
- [V7 planning review](coordination/evidence/v7-plan/REVIEW.md) records the changes, static findings and non-training integration checks.
- [v5.1 local experiment](experiments/v51_local_20260927/), [v3-results reference](experiments/reference_v3_results/) and [Akash v3 reference](experiments/reference_akash_v3/) preserve alternative work without overwriting canonical APIs.

V7 is a specification, not an implemented or trained model. The existing stage-two release scaffold has documented gaps; inclusion on main does not approve its current gate for a V7 production run. Claims and spending authorization remain separate. Read the current user instruction before executing anything.

The Sol bootstrap plan and earlier handoffs are historical. Do not restart P00 or rebuild the incumbent because an old document says no code exists. Data, weights, predictions, private logs and access details stay outside public Git.
