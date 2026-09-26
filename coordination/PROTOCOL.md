# Opportunistic work with short, exclusive claims

No permanent roles. Any session can perform any task. Sol may own the critical path while
other sessions add independent improvements. Synchronization protects work; it must not
become a requirement to distribute effort equally or wait for every agent.

## Authoritative records

| Record | Location |
| --- | --- |
| Accepted implementation | Remote `main` |
| Claims, accepted tasks, integration pin, job caps and submission ledger | `coordination/state.json` on remote `codex/control` |
| Proposed work | `codex/<task>-<session>` branch, exact code SHA |
| Results and handoffs | `coordination/evidence/<task>/<session>-<id>.md`, committed with code/evidence |
| Large data/model/output artifacts | Authorized private storage, immutable manifest and SHA-256 |

After fetching, read `git show origin/codex/control:coordination/state.json`. A copy on
your own task branch can be stale. Never claim a local edit was delivered until its push
succeeds. The seed `state.json` is not a live remote board.

## Bootstrap and task claims

The first active session starts P00 and local fixtures immediately. For collaboration,
publish the seed control branch to the agreed remote once, inspect/reuse any existing
branch, and give each session its own clean control checkout in addition to its work
checkout. Keep coordination files small; no raw records or secrets in commits.

To change state, including claiming a task:

1. Fetch and start from the current remote control commit in the clean control checkout.
   Read all active claims and dependencies, and preserve any unpublished work separately.
2. Check that the task is ready and its exact paths do not overlap an active claim.
   Add a claim with task ID, unique session ID/UUID, branch, paths, base main SHA,
   UTC timestamp, next checkpoint and expected deliverable. Increment `epoch`.
3. Commit this state update with **the fetched remote control commit as its parent**.
   Push normally: `git push origin HEAD:refs/heads/codex/control`.
4. A successful push publishes the claim. A rejected/non-fast-forward push means another
   update won: preserve the unpublished commit, fetch, reread every precondition and
   make a new valid update from current state. Do not merge/cherry-pick the stale claim
   automatically, force-push, or continue as though you own the task.

This serializes small state updates through Git's normal branch update rules, provided
every writer follows the protocol. It is not a security boundary or a lease service.
Each update uses a unique session/claim ID so concurrent claims are distinct commits.
An existing control checkout on an old head must be refreshed safely before the next
update. Never rebase a stale claim and publish without checking task/path availability.

Tasks are 30-120 minute units where practical. A large task can have nonoverlapping
subtasks added to state. Claiming work grants temporary path ownership, not permanent
ownership of a model or subject area. Anyone can claim the next ready task.

## Active loop

Fetch/read at task boundaries, before cloud launches/integration, and at least every ten
active minutes. Implement and run meaningful checks. Commit code plus concise evidence;
publish the task branch and record its exact SHA in control state. Hand off a concrete
next action to a named session/task, and record acknowledgment when consumed. Do not
make humans copy chat history between coding agents.

Evidence includes hypothesis, commands, real results or 'not run', per-query macro metric,
candidate oracle, slices, runtime/cost, exact split/model/candidate pins and artifact
checksums. A changed code SHA invalidates review of the previous SHA.

Use `planned -> ready -> claimed -> review -> accepted`, or explicitly record
`blocked` / `skipped` with reason. Accepted dependencies release the next tasks.
An optional experiment is not a release dependency. Preserve a usable incumbent.

## Integration and release

Any session may acquire the short `INTEGRATE` claim through the same state procedure.
Check reviewed/tested SHAs, integrate into current main in an isolated checkout, run
combined checks and push normally. If main advanced, redo integration/checks on its new
head. Record the accepted code SHA and updated main pin, then release the claim.
Do not hold integration ownership for an entire experiment.

Independent review is most valuable for scorer/splits, selection changes and final
output validation. Other available agents review exact SHAs while work continues.
Routine modules do not wait for an all-agent meeting. The integrating agent still checks
the combined pipeline. Freeze candidate/model/split manifests before final inference.

Use a separate short `SUBMIT` claim and the ledger for each portal attempt. One active
portal operator uploads the exact hashed file; record SCORED status, score, selected
version and acknowledgment when actually observed. Never infer a receipt from elapsed
time or from a completed file upload alone.

## Recovery and jobs

- No expiry-based takeover. A stale checkpoint prompts inspection; release/stop the old
  writer or have the operator explicitly confirm it has ended before taking its paths.
- If communication fails, preserve local commits and continue independent local work.
  Defer integration and new shared/paid jobs until unique ownership is established.
- Each paid job has a unique ID, owner session, account alias, maximum duration and cap
  in state. Verify authorization and existing job state before launching/retrying.
- A resumed session reads state and immutable manifests. Chat memory is not required.
- Phase `freeze` permits only recorded release/repair work. Phase `complete` ends work.
- Sessions are not automatically launched or awakened by these files. A closed or
  quota-limited agent needs its operator or a separately authorized runner to resume it.
