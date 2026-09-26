# Shared instructions for Codex and Claude Code

## Current objective and execution model

The user replaced the fixed A/B/C/D roles on 26 September 2026. Any available agent may
do any useful work, including the entire critical path. Maximize measured progress toward
at least 0.98 macro per-S1 F0.5 and a valid final submission; equal contribution is not a
requirement. Do not ask an operator to choose an old role or wait for all four sessions.

Read on startup and after context loss:

1. `output/hackathon/SOL_IMPLEMENTATION_PLAN.md` (current technical plan).
2. `coordination/START_HERE.md`, `coordination/PROTOCOL.md` and `coordination/CONTRACTS.md`.
3. The canonical task state on the agreed remote `codex/control` branch, if published.
   The local `coordination/state.json` is only a seed until then. Fetch before reading it.
4. Current task evidence, code/model/data/split manifests and the user's latest directions.

The older `TEAM_PLAN.md`, its PDF and `coordination/legacy_fixed_roles/` are historical
reference material. They do not assign current roles or override this plan. The former
`team_sync.py sync --agent A` workflow is retired. Do not spend the contest rebuilding it.

## Work and synchronization

- Use one unique session ID and an isolated clone/worktree per writing agent. Work on
  `codex/<task>-<session>`. Never run two writing sessions in the same checkout.
- Claim the highest-value ready task and its actual file paths before shared work.
  Claims use a serialized normal Git push to `codex/control`, as described in PROTOCOL.
  A claim exists only after a successful push. On conflict, reread state and choose/retry;
  never automatically merge a stale claim. Sol can start local scaffolding immediately
  when it is the only writer and the remote is not ready.
- Claim a bounded task, not a permanent subject area. Other agents may take any unclaimed
  task. Do not modify a peer's claimed files without a recorded transfer or subtask.
- Fetch at task boundaries, before paid jobs/integration, and at least every ten active
  minutes. Publish exact code SHAs, real commands/results, immutable manifests and the
  next action at milestones or before stopping/context loss. Agents exchange evidence
  through the repository; humans do not relay conversations.
- Keep integration and portal submission serialized using short explicit claims, not a
  permanent coordinator role. Integrate tested code against current main; a rejected
  push requires fresh integration/tests. No force pushes or destructive resets.
- A stale heartbeat is a reason to inspect, not permission to take over live work.
  Stop or explicitly release the old writer/job before transferring its claim.
- Local subagents may help with read-only review or explicitly nonoverlapping work.
  The primary session owns its claims, evidence and publication.
- If communication fails, preserve local work and continue independent preparation;
  do not launch duplicate cloud jobs or assume a message/claim was delivered.

## Technical priorities

- Build scorer/splits and a complete incumbent first. Address retrieval must be independent
  of name filtering. Optimize the largest measured score loss, not model novelty.
- Evaluate against full target catalogs, by business groups, using all S1 queries including
  singletons and no-candidate cases. Never inject true matches into evaluation candidates.
- Track exact candidate oracle U and matching gap U-S. A candidate ceiling below 0.98
  cannot support the target with that candidate set. Aim for U >= 0.995 as working margin.
- Keep the audit partition locked until champion selection. No fabricated scores, tests,
  licenses, artifacts, job completions or submission receipts.
- Preserve deterministic manifests, restartable shards and one tested incumbent. Final
  output must be reproducible from the submitted code and permitted supplied data.

## Competition and release constraints

- Use provided business data only. Internet research may inform methods; it must not add
  identity lookups, registries, geocoding, external labels or hosted resolution calls.
- Final model must meet MIT/Apache-2.0 and <=8B requirements. Verify every component;
  optional pretrained weights require a documented permissible interpretation.
- Every test S1, including France, appears once in both exact TSV schemas. Predict
  zero/one/many S2/S3 IDs. Require existing test target IDs, no duplicates and matches
  subset candidates. Candidates must equal the set scored by the final matching system.
- At most five submissions per team per day. One portal operator at a time under the
  participant login rules. Preserve version history, a full template and a short summary.
- Official close: 2026-09-27 23:59 IST; internal acceptance deadline: 21:30 IST.
  Recalculate remaining time; obsolete September 25 schedules must not delay execution.

## Budget, artifacts and human boundaries

- Plan around three approximately $100 AWS accounts; fourth balance unknown. Allocate
  by jobs, not people: provisional $210 active spend and $90 reserve, within actual balances.
- Record unique job ID, account alias, owner session, duration and cost cap before launch;
  check whether the job already exists before retrying. Normal authorization still applies.
- Coding-agent subscriptions/API costs are separate from AWS credits.
- Git carries code and aggregate metadata. Keep records, prediction TSVs, models, private
  logs, secrets and presigned URLs in authorized private storage with checksums.
  The configured remote was observed public; do not publish sensitive artifacts there.
- Agents implement, measure, review, document and coordinate. Humans provide account
  access, necessary approvals and portal interaction where required. Files do not launch
  another person's session, wake a stopped agent or grant infrastructure permissions.
