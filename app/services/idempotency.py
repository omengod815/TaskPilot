"""Persistent tool idempotency (at-least-once delivery + one visible effect).

The dangerous failure this module exists for:

    tool wrote the side effect -> response was lost -> agent retried
    -> the customer was charged twice

An in-memory set of "already called" keys cannot survive a restart, cannot
coordinate two workers, and cannot tell "the request never arrived" apart from
"the request arrived and the reply was lost".  A client-generated UUID alone
does not help either: it only proves the *client* retried, not that the
*server* de-duplicated.

This store therefore keeps, in SQLite and under a unique primary key:

* the idempotency key (stable across retries of the same logical action),
* a canonical hash of the arguments (same key with different args is a bug),
* a status: ``IN_PROGRESS`` / ``SUCCEEDED`` / ``FAILED`` / ``UNKNOWN``,
* the stored result for replay,
* a lease so a crashed owner does not block the key forever.

SQLite gives us the atomic claim for free: ``BEGIN IMMEDIATE`` + ``INSERT``
on a unique key means exactly one caller wins, even across processes on the
same host.  The demo intentionally does not add Redis/Kafka: correctness here
comes from the database constraint, not from more middleware.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from app.core.errors import IdempotencyConflict, IdempotencyInProgress


class IdempotencyStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


class ClaimOutcome(str, Enum):
    CLAIMED = "claimed"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    UNKNOWN = "unknown"


@dataclass
class IdempotencyRecord:
    key: str
    tool: str
    params_hash: str
    task_id: str
    status: IdempotencyStatus
    result: Any | None
    error_code: str | None
    error_message: str | None
    attempts: int
    owner: str
    lease_expires_at: float
    created_at: float
    updated_at: float
    expires_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "tool": self.tool,
            "params_hash": self.params_hash,
            "task_id": self.task_id,
            "status": self.status.value,
            "result": self.result,
            "error_code": self.error_code,
            "error_message": (self.error_message or "")[:300],
            "attempts": self.attempts,
            "owner": self.owner,
            "lease_expires_at": self.lease_expires_at,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_at": self.expires_at,
        }


@dataclass
class Claim:
    outcome: ClaimOutcome
    record: IdempotencyRecord
    owner: str

    @property
    def result(self) -> Any | None:
        return self.record.result


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def params_hash(args: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(args).encode("utf-8")).hexdigest()


def build_idempotency_key(task_id: str, tool: str, args: dict[str, Any]) -> str:
    """Stable across retries/restarts because it derives from the *action*.

    ``task_id`` scopes the action to one user request; ``tool`` + canonical
    args identify the logical business action.  Re-running the same task after
    a crash therefore produces the same key and hits the stored result.
    """

    digest = hashlib.sha256(canonical_json({"task_id": task_id, "tool": tool, "args": args}).encode("utf-8")).hexdigest()
    return f"{tool}:{digest[:40]}"


class IdempotencyStore:
    def __init__(self, db_path: Path | str, *, ttl_hours: int = 24) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(0, ttl_hours) * 3600
        self._init_db()

    # ------------------------------------------------------------------
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
                CREATE TABLE IF NOT EXISTS idempotency_records (
                    key TEXT PRIMARY KEY,
                    tool TEXT NOT NULL,
                    params_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error_code TEXT,
                    error_message TEXT,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    owner TEXT NOT NULL DEFAULT '',
                    lease_expires_at REAL NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_idempotency_expires ON idempotency_records(expires_at)")

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> IdempotencyRecord:
        return IdempotencyRecord(
            key=row["key"],
            tool=row["tool"],
            params_hash=row["params_hash"],
            task_id=row["task_id"],
            status=IdempotencyStatus(row["status"]),
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error_code=row["error_code"],
            error_message=row["error_message"],
            attempts=int(row["attempts"]),
            owner=row["owner"],
            lease_expires_at=float(row["lease_expires_at"]),
            created_at=float(row["created_at"]),
            updated_at=float(row["updated_at"]),
            expires_at=float(row["expires_at"]),
        )

    # ------------------------------------------------------------------
    def get(self, key: str) -> IdempotencyRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM idempotency_records WHERE key=?", (key,)).fetchone()
        return self._row_to_record(row) if row else None

    def claim(
        self,
        key: str,
        *,
        tool: str,
        args_hash: str,
        task_id: str,
        owner: str | None = None,
        lease_seconds: float = 30.0,
    ) -> Claim:
        owner = owner or uuid.uuid4().hex
        now = time.time()
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM idempotency_records WHERE key=?", (key,)).fetchone()
            if row is None:
                expires = now + self.ttl_seconds if self.ttl_seconds else now + 86400
                conn.execute(
                    """
                    INSERT INTO idempotency_records(
                        key, tool, params_hash, task_id, status, result_json, error_code, error_message,
                        attempts, owner, lease_expires_at, created_at, updated_at, expires_at
                    ) VALUES(?, ?, ?, ?, ?, NULL, NULL, NULL, 1, ?, ?, ?, ?, ?)
                    """,
                    (key, tool, args_hash, task_id, IdempotencyStatus.IN_PROGRESS.value, owner, now + lease_seconds, now, now, expires),
                )
                conn.commit()
                record = self.get(key)
                assert record is not None
                return Claim(ClaimOutcome.CLAIMED, record, owner)

            record = self._row_to_record(row)
            if record.params_hash != args_hash or record.tool != tool:
                conn.rollback()
                raise IdempotencyConflict(
                    "同一幂等键被不同工具或不同参数复用，已拒绝执行",
                    details={"key": key, "stored_tool": record.tool, "incoming_tool": tool},
                )

            if record.status is IdempotencyStatus.SUCCEEDED:
                conn.rollback()
                return Claim(ClaimOutcome.SUCCEEDED, record, owner)
            if record.status is IdempotencyStatus.FAILED:
                conn.rollback()
                return Claim(ClaimOutcome.FAILED, record, owner)
            if record.status is IdempotencyStatus.IN_PROGRESS and record.lease_expires_at > now:
                conn.rollback()
                return Claim(ClaimOutcome.IN_PROGRESS, record, owner)

            # IN_PROGRESS with an expired lease, or UNKNOWN: the previous owner
            # may have died mid-flight.  We never replay blindly; the caller
            # must reconcile against the external system first.
            status = IdempotencyStatus.UNKNOWN
            conn.execute(
                """
                UPDATE idempotency_records
                   SET status=?, owner=?, lease_expires_at=?, attempts=attempts+1, updated_at=?
                 WHERE key=?
                """,
                (status.value, owner, now + lease_seconds, now, key),
            )
            conn.commit()
            updated = self.get(key)
            assert updated is not None
            return Claim(ClaimOutcome.UNKNOWN, updated, owner)
        except BaseException:
            try:
                conn.rollback()
            except Exception:
                pass
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------
    def _update(self, key: str, owner: str, **fields: Any) -> None:
        fields["updated_at"] = time.time()
        assignments = ", ".join(f"{name}=?" for name in fields)
        values = [fields[name] for name in fields]
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE idempotency_records SET {assignments} WHERE key=? AND owner=?",
                (*values, key, owner),
            )
            if cursor.rowcount == 0:
                raise IdempotencyConflict(
                    "幂等记录的所有者已变更（可能是租约到期后被其他执行者接管）",
                    details={"key": key, "owner": owner},
                )

    def mark_succeeded(self, key: str, owner: str, result: Any) -> None:
        now = time.time()
        self._update(
            key,
            owner,
            status=IdempotencyStatus.SUCCEEDED.value,
            result_json=json.dumps(result, ensure_ascii=False, default=str),
            error_code=None,
            error_message=None,
            lease_expires_at=0.0,
            expires_at=(now + self.ttl_seconds) if self.ttl_seconds else now + 86400,
        )

    def mark_failed(self, key: str, owner: str, code: str, message: str) -> None:
        self._update(
            key,
            owner,
            status=IdempotencyStatus.FAILED.value,
            error_code=code,
            error_message=message[:500],
            lease_expires_at=0.0,
        )

    def mark_unknown(self, key: str, owner: str, reason: str) -> None:
        self._update(
            key,
            owner,
            status=IdempotencyStatus.UNKNOWN.value,
            error_message=reason[:500],
            lease_expires_at=time.time() + 5.0,
        )

    def reset_for_retry(self, key: str, owner: str) -> None:
        """Only for actions proven *not* to have happened externally."""

        self._update(
            key,
            owner,
            status=IdempotencyStatus.IN_PROGRESS.value,
            result_json=None,
            lease_expires_at=time.time() + 30.0,
        )

    def purge_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM idempotency_records WHERE expires_at < ? AND status != ?", (now, IdempotencyStatus.IN_PROGRESS.value))
        return cursor.rowcount

    def stats(self) -> dict[str, int]:
        with self._connect() as conn:
            rows = conn.execute("SELECT status, COUNT(*) AS n FROM idempotency_records GROUP BY status").fetchall()
        return {row["status"]: int(row["n"]) for row in rows}


class InMemoryIdempotencyStore:
    """Test double with the same contract semantics (single process)."""

    def __init__(self, ttl_hours: int = 24) -> None:
        self.ttl_seconds = max(0, ttl_hours) * 3600
        self._records: dict[str, IdempotencyRecord] = {}
        self._lock = __import__("threading").RLock()

    def get(self, key: str) -> IdempotencyRecord | None:
        with self._lock:
            return self._records.get(key)

    def claim(self, key: str, *, tool: str, args_hash: str, task_id: str, owner: str | None = None, lease_seconds: float = 30.0) -> Claim:
        owner = owner or uuid.uuid4().hex
        now = time.time()
        with self._lock:
            record = self._records.get(key)
            if record is None:
                record = IdempotencyRecord(
                    key=key, tool=tool, params_hash=args_hash, task_id=task_id,
                    status=IdempotencyStatus.IN_PROGRESS, result=None, error_code=None,
                    error_message=None, attempts=1, owner=owner,
                    lease_expires_at=now + lease_seconds, created_at=now, updated_at=now,
                    expires_at=now + (self.ttl_seconds or 86400),
                )
                self._records[key] = record
                return Claim(ClaimOutcome.CLAIMED, record, owner)
            if record.params_hash != args_hash or record.tool != tool:
                raise IdempotencyConflict("同一幂等键被不同参数复用", details={"key": key})
            if record.status is IdempotencyStatus.SUCCEEDED:
                return Claim(ClaimOutcome.SUCCEEDED, record, owner)
            if record.status is IdempotencyStatus.FAILED:
                return Claim(ClaimOutcome.FAILED, record, owner)
            if record.status is IdempotencyStatus.IN_PROGRESS and record.lease_expires_at > now:
                return Claim(ClaimOutcome.IN_PROGRESS, record, owner)
            record.status = IdempotencyStatus.UNKNOWN
            record.owner = owner
            record.lease_expires_at = now + lease_seconds
            record.attempts += 1
            record.updated_at = now
            return Claim(ClaimOutcome.UNKNOWN, record, owner)

    def _owned(self, key: str, owner: str) -> IdempotencyRecord:
        record = self._records.get(key)
        if record is None or record.owner != owner:
            raise IdempotencyConflict("幂等记录不存在或所有者已变更", details={"key": key})
        return record

    def mark_succeeded(self, key: str, owner: str, result: Any) -> None:
        with self._lock:
            record = self._owned(key, owner)
            record.status = IdempotencyStatus.SUCCEEDED
            record.result = result
            record.lease_expires_at = 0.0
            record.updated_at = time.time()
            record.expires_at = time.time() + (self.ttl_seconds or 86400)

    def mark_failed(self, key: str, owner: str, code: str, message: str) -> None:
        with self._lock:
            record = self._owned(key, owner)
            record.status = IdempotencyStatus.FAILED
            record.error_code = code
            record.error_message = message
            record.lease_expires_at = 0.0
            record.updated_at = time.time()

    def mark_unknown(self, key: str, owner: str, reason: str) -> None:
        with self._lock:
            record = self._owned(key, owner)
            record.status = IdempotencyStatus.UNKNOWN
            record.error_message = reason
            record.updated_at = time.time()

    def reset_for_retry(self, key: str, owner: str) -> None:
        with self._lock:
            record = self._owned(key, owner)
            record.status = IdempotencyStatus.IN_PROGRESS
            record.result = None
            record.lease_expires_at = time.time() + 30.0

    def purge_expired(self, *, now: float | None = None) -> int:
        now = time.time() if now is None else now
        with self._lock:
            expired = [k for k, r in self._records.items() if r.expires_at < now and r.status is not IdempotencyStatus.IN_PROGRESS]
            for key in expired:
                del self._records[key]
        return len(expired)

    def stats(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self._records.values():
            counts[record.status.value] = counts.get(record.status.value, 0) + 1
        return counts


def raise_in_progress(record: IdempotencyRecord) -> None:
    raise IdempotencyInProgress(
        "相同业务动作正在执行中，请勿重复提交",
        details={"key": record.key, "lease_expires_at": record.lease_expires_at},
    )