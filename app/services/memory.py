"""Separated memory: short-term session notes vs long-term user preferences.

The original project mixed "history retrieval" with "task state".  That becomes
a correctness problem as soon as an agent resumes: a ``pending_approval`` task
must never be recalled as a finished conversation memory, and a stale chat
summary must not be confused with the checkpointed execution state.

This module therefore stores only two kinds of *memory*:

* ``short_term`` – a compact summary of a finished task in a session, TTL days;
* ``long_term``  – an explicit user preference, no TTL (or a long one).

Execution state lives in the task repository / LangGraph checkpoint, never here.
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

MEMORY_KINDS = ("short_term", "long_term")


def _tokens(text: str) -> set[str]:
    latin = set(re.findall(r"[a-zA-Z0-9_+-]{2,}", text.lower()))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    cjk: set[str] = set()
    for run in cjk_runs:
        cjk.add(run)
        cjk.update(run[i:i + 2] for i in range(len(run) - 1))
    return latin | cjk


class MemoryStore:
    def __init__(self, db_path: Path | str, *, max_items: int = 200, ttl_days: int = 14) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_items = max_items
        self.ttl_days = ttl_days
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
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tags_json TEXT NOT NULL DEFAULT '[]',
                    dedupe_hash TEXT NOT NULL,
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    last_used_at REAL,
                    UNIQUE(user_id, kind, dedupe_hash)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_kind ON memories(user_id, kind, created_at DESC)")

    # ------------------------------------------------------------------
    def remember(
        self,
        *,
        user_id: str,
        session_id: str,
        content: str,
        kind: str = "short_term",
        tags: list[str] | None = None,
        importance: float = 0.5,
        ttl_days: int | None = None,
    ) -> str | None:
        content = " ".join((content or "").split())
        if len(content) < 4 or kind not in MEMORY_KINDS:
            return None
        # Dedupe before writing: summaries repeat across re-runs.
        digest = hashlib.sha256(f"{kind}:{content[:500]}".encode("utf-8")).hexdigest()
        now = time.time()
        days = self.ttl_days if ttl_days is None else ttl_days
        expires = now + days * 86400 if kind == "short_term" and days > 0 else None
        memory_id = uuid.uuid4().hex
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO memories(id, user_id, session_id, kind, content, tags_json, dedupe_hash, importance, created_at, expires_at, last_used_at)
                    VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, kind, dedupe_hash) DO UPDATE SET last_used_at=excluded.last_used_at
                    """,
                    (
                        memory_id, user_id, session_id, kind, content[:4000],
                        __import__("json").dumps(tags or [], ensure_ascii=False),
                        digest, float(importance), now, expires, now,
                    ),
                )
                conn.execute(
                    """
                    DELETE FROM memories WHERE user_id=? AND id NOT IN (
                        SELECT id FROM memories WHERE user_id=? ORDER BY importance DESC, created_at DESC LIMIT ?
                    )
                    """,
                    (user_id, user_id, self.max_items),
                )
            return memory_id
        except sqlite3.Error:
            return None

    def recall(
        self,
        *,
        user_id: str,
        query: str,
        kinds: tuple[str, ...] = MEMORY_KINDS,
        limit: int = 5,
        max_chars: int = 1200,
    ) -> list[dict[str, Any]]:
        now = time.time()
        placeholders = ",".join("?" for _ in kinds)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM memories
                 WHERE user_id=? AND kind IN ({placeholders})
                   AND (expires_at IS NULL OR expires_at > ?)
                 ORDER BY created_at DESC LIMIT 300
                """,
                (user_id, *kinds, now),
            ).fetchall()
        query_tokens = _tokens(query)
        scored: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            score = len(query_tokens & _tokens(row["content"])) / max(1, len(query_tokens))
            if score <= 0 and row["kind"] == "long_term":
                score = 0.05  # long-term preferences are always somewhat relevant
            if score <= 0:
                continue
            age_days = max(0.0, (now - float(row["created_at"])) / 86400)
            score += 0.1 * float(row["importance"]) + 0.05 / (1 + age_days)
            scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)

        results: list[dict[str, Any]] = []
        used = 0
        touched: list[str] = []
        for score, row in scored:
            content = row["content"]
            if used + len(content) > max_chars:
                content = content[: max(0, max_chars - used)]
            if not content:
                break
            results.append(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "content": content,
                    "score": round(score, 4),
                    "tags": __import__("json").loads(row["tags_json"] or "[]"),
                }
            )
            touched.append(row["id"])
            used += len(content)
            if len(results) >= limit or used >= max_chars:
                break
        if touched:
            with self._connect() as conn:
                conn.executemany("UPDATE memories SET last_used_at=? WHERE id=?", [(now, i) for i in touched])
        return results

    def forget(self, *, user_id: str, kind: str | None = None) -> int:
        with self._connect() as conn:
            if kind:
                cursor = conn.execute("DELETE FROM memories WHERE user_id=? AND kind=?", (user_id, kind))
            else:
                cursor = conn.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
        return cursor.rowcount

    def purge_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?", (now,))
        return cursor.rowcount

    def stats(self) -> dict[str, Any]:
        with self._connect() as conn:
            rows = conn.execute("SELECT kind, COUNT(*) AS n FROM memories GROUP BY kind").fetchall()
        return {"by_kind": {row["kind"]: int(row["n"]) for row in rows}, "max_items_per_user": self.max_items}