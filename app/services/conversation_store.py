"""SQLite persistence for free-form multi-agent conversations.

The single-graph task state lives in ``TaskRepository``.  A conversation has a
different shape (messages, handoffs, per-agent turns), so it gets its own table
instead of being squeezed into ``tasks.state_json``.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class ConversationStore:
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
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    goal TEXT NOT NULL,
                    status TEXT NOT NULL,
                    turn INTEGER NOT NULL DEFAULT 0,
                    state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, created_at DESC)"
            )

    def save(self, state: dict[str, Any]) -> None:
        now = state.get("updated_at") or state.get("created_at")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations(
                    conversation_id, user_id, session_id, goal, status, turn,
                    state_json, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    status=excluded.status,
                    turn=excluded.turn,
                    state_json=excluded.state_json,
                    updated_at=excluded.updated_at
                """,
                (
                    state["conversation_id"],
                    state.get("user_id", "local-demo"),
                    state.get("session_id", "default"),
                    state.get("goal", ""),
                    state.get("status", "running"),
                    int(state.get("turn", 0)),
                    _dump(state),
                    state.get("created_at", now),
                    now,
                ),
            )

    def get(self, conversation_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT state_json FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        return json.loads(row["state_json"]) if row else None

    def list_conversations(self, *, user_id: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id, user_id, session_id, goal, status, turn,
                       created_at, updated_at
                  FROM conversations
                 WHERE user_id=?
                 ORDER BY created_at DESC
                 LIMIT ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_by_status(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM conversations GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}