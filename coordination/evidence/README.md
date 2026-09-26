# Evidence, keyed by task rather than permanent role

Write `coordination/evidence/<task>/<session>-<unique-id>.md` and commit it with the
code/evidence SHA being handed off. Include:

- Task, hypothesis, session, code/reviewed SHA and exact commands.
- Dataset/split/candidate/feature/model/contract pins.
- Actual results or explicitly 'not run'; macro F0.5, oracle ceiling, slices and runtime.
- Private artifact logical references, SHA-256, row counts and reproduction command.
- Limitations, blockers, next action and review result when relevant.

No raw business records, secrets, credentials or presigned URLs. Record delivery and
acknowledgment in canonical control state. A local file is not a delivered handoff.
