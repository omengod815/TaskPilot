"""LangGraph checkpoint persistence.

``MemorySaver`` loses everything on restart, which is exactly the failure this
project must survive.  ``SqliteSaver`` stores every super-step checkpoint in
SQLite (WAL), keyed by ``thread_id = task_id``; a restarted process can call
``graph.invoke(None, config)`` and continue from the pending node.

Operational requirement: the checkpoint database must live on a persistent
volume in production (not an ephemeral container layer), and one task must be
executed by one worker at a time.  The lease/heartbeat in the task repository
enforces the latter for this single-host demo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver


def thread_config(task_id: str) -> dict[str, Any]:
    return {"configurable": {"thread_id": task_id}, "recursion_limit": 50}


class CheckpointManager:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._context = SqliteSaver.from_conn_string(str(self.db_path))
        self.saver: SqliteSaver = self._context.__enter__()

    def close(self) -> None:
        try:
            self._context.__exit__(None, None, None)
        except Exception:  # pragma: no cover - best effort on shutdown
            pass

    def __enter__(self) -> "CheckpointManager":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def checkpoint_state(graph: Any, config: dict[str, Any]) -> dict[str, Any] | None:
    """Return the persisted channel values, or ``None`` when no checkpoint exists."""

    snapshot = graph.get_state(config)
    values = getattr(snapshot, "values", None)
    return values or None


def pending_nodes(graph: Any, config: dict[str, Any]) -> tuple[str, ...]:
    snapshot = graph.get_state(config)
    return tuple(getattr(snapshot, "next", ()) or ())