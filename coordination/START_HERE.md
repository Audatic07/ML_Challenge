# Start immediately; contribute wherever useful

The current plan is [Sol implementation plan](../output/hackathon/SOL_IMPLEMENTATION_PLAN.md).
Give any available coding agent [the same startup prompt](prompts/SOL_START.md).
The fastest session may implement the entire critical path. Other sessions take ready,
nonoverlapping tasks; there is no fixed division by person and no equal-work requirement.

1. The first session inspects the workspace, preserves existing work and begins P00/P01.
   Do not wait for the other three operators to arrive.
2. Each additional session uses its own clone/worktree and unique session ID. It reads
   shared state and claims an unclaimed task or a concrete subtask from the current worker.
3. Publish code and small metadata to the agreed Git destination. Until publishing/access
   is available, one local writer may progress; other sessions can perform independent
   read-only reviews or prepare work in isolated directories without overlapping claims.
4. Use the normal-Git claim protocol in `PROTOCOL.md`. Code lives on task branches;
   canonical task state lives on `codex/control`. No additional agent platform is required.

```text
Sol / Claude / any available session
             |
        claim ready work <----> shared queue on codex/control
             |
      implement + measure ----> exact-SHA evidence
             |
      brief integration claim -> tested main -> next highest-value task
```

The current `state.json` is a seed, with no actual claims, remote publication, models or
cloud jobs. The claim protocol is documented; no new automation daemon/helper is claimed
to be implemented. Agents can execute it with ordinary Git. Timebox coordination setup
to 20 minutes and keep local model work moving.

The repository configured as origin was observed public. Use authorized private storage
for data, prediction TSVs and trained artifacts. Do not change visibility or publish to a
different destination without an agreed choice. Keep ordinary tool permissions intact.

All old role prompts and fixed-role helper/tests are retained in `legacy_fixed_roles/` as
history. The older PDF is also historical; the current executable specification is the
Markdown plan above. The retired helper will refuse execution rather than revive roles.
