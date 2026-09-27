# Start from consolidated main and the current V7 plan

1. Read [V7 final implementation and training plan](../output/hackathon/V7_FINAL_ITERATION_PLAN.md). The [Sol plan](../output/hackathon/SOL_IMPLEMENTATION_PLAN.md) supplies background rules; its initial build schedule and no-code status are historical.
2. Fetch origin, then read coordination/state.json from origin/codex/control. The copy on main is a historical seed, not the live board.
3. Read [PROTOCOL.md](PROTOCOL.md), [CONTRACTS.md](CONTRACTS.md), the [branch map](BRANCH_CONSOLIDATION.md) and evidence for your exact inputs.
4. Use one isolated writing checkout and a unique session/task branch. Claim bounded paths and any integration/job work through a normal serialized control-branch push. Preserve active claims; a planning revision does not transfer another writer's files.
5. Work only within the current user's authorization. This review was planning/consolidation only. Implementation, training, cloud resources and portal interaction belong to the separately authorized execution workflow.

Published V6 is the fallback (reported audit 0.97089). Canonical code is under code/business_entity_resolution/. Published experiments are preserved under experiments/. Do not compare scores from different query cohorts or candidates as if they were the same evaluation.

The V7 plan defines R0-R8 and its acceptance gates. Earlier P00-P11 entries can be stale; inspect actual evidence before claiming work. No fixed A/B/C/D roles, equal-contribution requirement or need to wait for every account.

Use the [current handoff prompt](prompts/SOL_START.md) when handing work to the execution teammate. The public repository carries source and aggregate metadata only; private data/models/scores remain in authorized private storage. Files do not start another session or authorize spending.
