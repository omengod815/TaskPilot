"""SQLite task repository (task state, trace, heartbeat, request de-duplication).

The repository is *not* a substitute for a checkpoint store: it stores the
human-facing task record, while LangGraph's ``SqliteSaver`` stores the graph
state needed to resume execution.  Keeping both is deliberate — the API must
answer "what happened" even when a checkpoint cannot be read.
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class TaskRepository:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    task_type TEXT,
                    status TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS task_requests (
                    user_id TEXT NOT NULL,
                    request_key TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (user_id, request_key)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_user_session ON tasks(user_id, session_id, created_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_traces_task ON traces(task_id, id)")
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(tasks)").fetchall()}
        if "heartbeat_at" not in columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN heartbeat_at TEXT")
        if "error_code" not in columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN error_code TEXT")
        if "attempts" not in columns:
            conn.execute("ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 1")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status, updated_at)")
        self._migrate_task_requests(conn)

    def _migrate_task_requests(self, conn: sqlite3.Connection) -> None:
        """Upgrade the old global ``request_key`` table to per-user scope.

        v2 keyed idempotent submissions only by ``request_key``.  That let a
        second user reuse another user's key and receive the first user's task.
        The v3 schema scopes the key by ``(user_id, request_key)``.  Existing
        rows are preserved under ``local-demo`` instead of being dropped.
        """

        info = conn.execute("PRAGMA table_info(task_requests)").fetchall()
        names = {row["name"] for row in info}
        pk_columns = [row["name"] for row in info if int(row["pk"] or 0) > 0]
        if "user_id" in names and len(pk_columns) >= 2:
            return
        conn.execute("ALTER TABLE task_requests RENAME TO task_requests_legacy")
        conn.execute(
            """
            CREATE TABLE task_requests (
                user_id TEXT NOT NULL,
                request_key TEXT NOT NULL,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (user_id, request_key)
            )
            """
        )
        if "user_id" in names:
            conn.execute(
                """
                INSERT OR IGNORE INTO task_requests(user_id, request_key, task_id, created_at)
                SELECT COALESCE(NULLIF(user_id, ''), 'local-demo'), request_key, task_id, created_at
                  FROM task_requests_legacy
                """
            )
        else:
            conn.execute(
                """
                INSERT OR IGNORE INTO task_requests(user_id, request_key, task_id, created_at)
                SELECT 'local-demo', request_key, task_id, created_at
                  FROM task_requests_legacy
                """
            )
        conn.execute("DROP TABLE task_requests_legacy")

    # ------------------------------------------------------------------
    def save_state(self, state: dict[str, Any]) -> None:
        now = utc_now()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks(task_id, user_id, session_id, goal, task_type, status, state_json, created_at, updated_at, heartbeat_at, error_code, attempts)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    task_type=excluded.task_type,
                    status=excluded.status,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at,
                    heartbeat_at=excluded.heartbeat_at,
                    error_code=excluded.error_code
                """,
                (
                    state["task_id"],
                    state.get("user_id", "local-demo"),
                    state.get("session_id", "default"),
                    state.get("goal", ""),
                    state.get("task_type"),
                    state.get("status", "running"),
                    _dump(state),
                    now,
                    now,
                    now,
                    state.get("error_code"),
                    int(state.get("attempts") or 1),
                ),
            )

    def heartbeat(self, task_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE tasks SET heartbeat_at=? WHERE task_id=?", (utc_now(), task_id))

    def get_state(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT state_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def get_row(self, task_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        return dict(row) if row else None

    def update_status(self, task_id: str, status: str, *, error_code: str | None = None) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE tasks SET status=?, error_code=?, updated_at=?, heartbeat_at=? WHERE task_id=?",
                (status, error_code, utc_now(), utc_now(), task_id),
            )

    def transition_status(
        self,
        task_id: str,
        expected: str,
        new_status: str,
        *,
        patch: dict[str, Any] | None = None,
    ) -> bool:
        """Compare-and-set a state transition; returns False if someone won first.

        This closes the approval TOCTOU window: two concurrent ``approve``
        requests both read ``pending_approval``, but only the one whose
        conditional UPDATE matches the expected status may proceed.
        """

        state = self.get_state(task_id)
        if state is None:
            return False
        merged = {**state, **(patch or {}), "status": new_status}
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE tasks SET status=?, state_json=?, updated_at=?, heartbeat_at=?
                 WHERE task_id=? AND status=?
                """,
                (new_status, _dump(merged), now, now, task_id, expected),
            )
        return cursor.rowcount == 1

    # ------------------------------------------------------------------
    def append_trace(self, task_id: str, node: str, event: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO traces(task_id, node, event_json, created_at) VALUES(?, ?, ?, ?)",
                (task_id, node, _dump(event), utc_now()),
            )

    def get_trace(self, task_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT node, event_json, created_at FROM traces WHERE task_id=? ORDER BY id", (task_id,)).fetchall()
        return [{"node": r["node"], "event": json.loads(r["event_json"]), "created_at": r["created_at"]} for r in rows]

    def trace_count(self, task_id: str) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM traces WHERE task_id=?", (task_id,)).fetchone()
        return int(row["n"]) if row else 0

    # ------------------------------------------------------------------
    def recent_states(self, user_id: str, session_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT state_json FROM tasks WHERE user_id=? AND session_id=? ORDER BY created_at DESC LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        return [json.loads(r["state_json"]) for r in rows]

    def list_tasks(self, limit: int = 50, *, user_id: str | None = None) -> list[dict[str, Any]]:
        """List task summaries.

        ``user_id=None`` is reserved for administrative/debug callers; the HTTP
        API always passes the caller's user id so one user cannot enumerate
        another user's goals (P0-04 in the v3 audit).
        """

        columns = "task_id, user_id, session_id, goal, task_type, status, error_code, created_at, updated_at, heartbeat_at"
        if user_id is None:
            sql = f"SELECT {columns} FROM tasks ORDER BY created_at DESC LIMIT ?"
            params: tuple[Any, ...] = (limit,)
        else:
            sql = f"SELECT {columns} FROM tasks WHERE user_id=? ORDER BY created_at DESC LIMIT ?"
            params = (user_id, limit)
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM tasks GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}

    # ------------------------------------------------------------------
    def mark_stale_running(self, stale_seconds: float) -> list[dict[str, Any]]:
        """Flag tasks whose owner died (no heartbeat) as ``interrupted``."""

        cutoff = time.time() - stale_seconds
        stale: list[dict[str, Any]] = []
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT task_id, state_json, updated_at FROM tasks WHERE status IN ('running', 'resuming')"
            ).fetchall()
            for row in rows:
                try:
                    updated = datetime.fromisoformat(row["updated_at"]).timestamp()
                except (TypeError, ValueError):
                    updated = 0.0
                if updated >= cutoff:
                    continue
                state = json.loads(row["state_json"])
                state["status"] = "interrupted"
                state.setdefault("degraded_reasons", []).append("服务重启或心跳丢失，任务已标记为 interrupted，可调用 /resume 恢复")
                conn.execute(
                    "UPDATE tasks SET status=?, state_json=?, updated_at=?, heartbeat_at=? WHERE task_id=?",
                    ("interrupted", _dump(state), utc_now(), utc_now(), row["task_id"]),
                )
                stale.append(state)
        return stale

    # ------------------------------------------------------------------
    def claim_for_resume(
        self,
        task_id: str,
        allowed_statuses: tuple[str, ...] | set[str],
        *,
        patch: dict[str, Any] | None = None,
    ) -> bool:
        """Atomically move a recoverable task to ``running``.

        Two concurrent ``POST /resume`` calls can both read ``interrupted``; only
        the conditional UPDATE whose ``rowcount`` is 1 may invoke the graph.  The
        loser gets ``False`` and the API returns 409 instead of running the task
        twice (P0-02 in the v3 audit).
        """

        statuses = tuple(allowed_statuses)
        if not statuses:
            return False
        state = self.get_state(task_id)
        if state is None or state.get("status") not in statuses:
            return False
        merged = {**state, **(patch or {}), "status": "running", "error_code": None}
        placeholders = ",".join("?" for _ in statuses)
        now = utc_now()
        with self._connect() as conn:
            cursor = conn.execute(
                f"""
                UPDATE tasks SET status='running', state_json=?, updated_at=?, heartbeat_at=?
                 WHERE task_id=? AND status IN ({placeholders})
                """,
                (_dump(merged), now, now, task_id, *statuses),
            )
            won = cursor.rowcount == 1
        return won

    # ------------------------------------------------------------------
    def find_by_request_key(self, request_key: str, *, user_id: str | None = None) -> str | None:
        """Look up a submission key.

        ``user_id`` should always be passed by the HTTP layer; ``None`` exists
        only for administrative/debug queries and preserves the old behavior.
        """

        with self._connect() as conn:
            if user_id is None:
                row = conn.execute("SELECT task_id FROM task_requests WHERE request_key=?", (request_key,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT task_id FROM task_requests WHERE user_id=? AND request_key=?",
                    (user_id, request_key),
                ).fetchone()
        return row["task_id"] if row else None

    def register_request_key(self, request_key: str, task_id: str, *, user_id: str = "local-demo") -> bool:
        """Atomically bind ``(user_id, Idempotency-Key)`` to a task id.

        Returns ``False`` when another request already owns the key for this
        user.  Different users may reuse the same key without seeing each
        other's task.
        """

        try:
            with self._connect() as conn:
                conn.execute(
                    "INSERT INTO task_requests(user_id, request_key, task_id, created_at) VALUES(?, ?, ?, ?)",
                    (user_id, request_key, task_id, utc_now()),
                )
            return True
        except sqlite3.IntegrityError:
            return False