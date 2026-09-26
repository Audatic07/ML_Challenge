# Current task queue: any agent may claim any work

Read [Sol's implementation plan](SOL_IMPLEMENTATION_PLAN.md), especially sections 10-12.
The canonical live queue will be `coordination/state.json` on remote `codex/control`;
the local file is an unpublished seed. No jobs or agent sessions have been launched.

| Order | Task | What completion means |
| --- | --- | --- |
| P00 | Scaffold and brief coordination setup | Runnable package/fixture, unique sessions and task claims |
| P01 | Audit, group split, exact scorer | Immutable manifests, tested metric and integrity report |
| P02 | Writer and strict validator | Both schemas and all failure cases checked |
| P03 | Independent name/address retrieval | Full-catalog recall, oracle and runtime measured |
| P04 | Features and supervised tree | Real held-out predictions and error slices |
| P05 | Selection and full incumbent | Exact macro score, full-run ETA, complete valid outputs |
| P06 | Highest-loss repair | Address/weak-name or hard-negative improvement with evidence |
| P07 | Optional set selection | OOF gate/utility policy beats incumbent |
| P08 | Optional rescue/neural pilot | Residual gain passes runtime/license/budget gates |
| P09 | Transfer and reproduction | Country stress tests, clean run and resource checks |
| P10 | Audit, freeze, full inference | Champion pinned, final outputs strictly validated |
| P11 | Package and portal | Exact ZIP, documentation and actual submission receipt |

P06-P08 are conditional and may be skipped. No equal-contribution rule applies. One fast
agent may complete the full critical path. Other agents should claim bounded, disjoint
work and publish exact-SHA evidence through [the protocol](../../coordination/PROTOCOL.md).

Official close: 27 September 2026 23:59 IST. Internal acceptance target: 21:30 IST.
0.98 remains an unverified objective; no trained model or leaderboard result exists here.
