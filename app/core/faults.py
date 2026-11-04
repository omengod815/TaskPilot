"""Deliberate fault-injection seams (disabled unless explicitly enabled).

Tests need to prove recovery from a *process death* between "the side effect
was applied" and "the checkpoint was written".  Monkeypatching cannot simulate
that window reliably, so the graph exposes a named seam guarded by
``TASKPILOT_FAULT_INJECTION=1``.  In normal operation every call is a no-op.
"""

from __future__ import annotations

import os

_TRUE = {"1", "true", "yes", "on"}


def enabled() -> bool:
    return os.environ.get("TASKPILOT_FAULT_INJECTION", "").strip().lower() in _TRUE


def crash_at(point: str, exit_code: int = 42) -> None:
    """Hard-kill the process at ``point`` (simulates SIGKILL/power loss)."""

    if not enabled():
        return
    if os.environ.get("TASKPILOT_CRASH_AT", "").strip() == point:
        os._exit(exit_code)


def fail_at(point: str, exc: BaseException) -> None:
    if not enabled():
        return
    if os.environ.get("TASKPILOT_FAIL_AT", "").strip() == point:
        raise exc