# Planning revision validation - 26 September 2026

Scope: research, supplied-data diagnostics and a Sol-ready implementation plan.
No model was trained, no cloud job launched, no shared remote changed, and no portal
submission made. The target score remains unverified.

## Observed checks

- `python tools/profile_labelled_noise.py` completed against the supplied ZIP in 58.73s.
  Deterministic sample: 22,034 S1, 75,776 true links. Results are in
  `output/hackathon/labelled_noise_profile.json`; no raw records are exported.
- A temporary three-query ZIP fixture was processed with `--modulus 1`: two exact
  matches for one query, an unrecovered true match for a second, and an empty truth
  for a third. Both diagnostic oracle means equalled 2/3 within 1e-12. Passed.
- Two independent temporary Git clones committed different claims for P01 from the
  same control head. The first normal push succeeded; the stale second push was
  rejected; the remote retained only the first claim. Passed. This tests the documented
  protocol's conflict mechanism, not an automatically deployed distributed service.
- Current team/state JSON parsed; all 12 task IDs are unique, dependencies resolve
  without cycles, no fixed task owners or live claims are seeded, and measured score
  remains null. Relative document links checked; Python syntax parsed. Passed.
- `python tools/team_sync.py sync --agent A` returned exit 2 with a clear retired-workflow
  message. The old helper/tests were preserved as `.py.txt` in the historical kit.
- Formula arithmetic checked: France share 0.1497520409; required illustrative France
  scores 0.95161147 if the rest scores 0.985, and 0.92322295 if the rest scores 0.99.
  Six-hour raw throughput is 80.21037 queries/s; with 50% runtime contingency 120.31556.

The new operational protocol and CLI/module layout are specifications for implementing
agents. A task-claim automation helper, ML package, production tests and validated model
artifacts are not claimed to exist. The current implementation plan supersedes the old
fixed-role documents and their historical PDF export.
