"""Retired fixed-role helper; see the task-claim protocol instead."""
from __future__ import annotations

import sys


if __name__ == "__main__":
    print(
        "The A/B/C/D coordination workflow is retired. Read "
        "coordination/PROTOCOL.md and output/hackathon/SOL_IMPLEMENTATION_PLAN.md. "
        "Use task claims on codex/control; no fixed role is required.",
        file=sys.stderr,
    )
    raise SystemExit(2)
