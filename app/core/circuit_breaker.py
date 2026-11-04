"""Three-state circuit breaker with a single half-open probe.

CLOSED  -> normal traffic.
OPEN    -> every call fails fast for ``cooldown_seconds``.
HALF_OPEN -> exactly ``half_open_max_calls`` trial calls are allowed; a success
             closes the breaker, a failure re-opens it for another cooldown.

Two deployment modes are supported:

* in-process (default): a ``threading.RLock`` serialises transitions, so
  concurrent requests cannot corrupt the state;
* shared (optional): :class:`SqliteCircuitStore` keeps the state in the same
  SQLite database used by tasks, and each transition is a ``BEGIN IMMEDIATE``
  read-modify-write, so two uvicorn workers on one host agree on the state.
  Cross-host sharing would need Redis; that trade-off is documented in the
  README instead of being silently assumed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from app.core.errors import CircuitOpenError

CONFIG_KEYS = (
    "failure_threshold",
    "cooldown_seconds",
    "half_open_max_calls",
    "half_open_success_threshold",
)


class CircuitState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitConfig:
    failure_threshold: int = 5
    cooldown_seconds: float = 30.0
    half_open_max_calls: int = 1
    half_open_success_threshold: int = 1


@dataclass
class CircuitSnapshot:
    name: str
    state: str
    failures: int
    opened_at: float
    open_until: float
    half_open_calls: int
    half_open_successes: int
    total_failures: int = 0
    total_successes: int = 0
    last_error: str = ""
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state,
            "failures": self.failures,
            "open_until": round(self.open_until, 3),
            "half_open_calls": self.half_open_calls,
            "total_failures": self.total_failures,
            "total_successes": self.total_successes,
            "last_error": self.last_error,
            "transitions": self.transitions[-5:],
            "retry_after_seconds": round(max(0.0, self.open_until - time.time()), 3)
            if self.state == CircuitState.OPEN.value
            else 0.0,
        }


def _new_state(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "state": CircuitState.CLOSED.value,
        "failures": 0,
        "opened_at": 0.0,
        "open_until": 0.0,
        "half_open_calls": 0,
        "half_open_successes": 0,
        "total_failures": 0,
        "total_successes": 0,
        "last_error": "",
        "transitions": [],
        "version": 0,
    }


def _transition(state: dict[str, Any], to: str, now: float, reason: str) -> None:
    state["state"] = to
    state["version"] = int(state.get("version", 0)) + 1
    entry = {"at": round(now, 3), "to": to, "reason": reason}
    state.setdefault("transitions", []).append(entry)
    state["transitions"] = state["transitions"][-20:]


def _normalise(state: dict[str, Any], now: float) -> None:
    if state["state"] == CircuitState.OPEN.value and now >= float(state["open_until"]):
        _transition(state, CircuitState.HALF_OPEN.value, now, "cooldown_elapsed")
        state["half_open_calls"] = 0
        state["half_open_successes"] = 0


def _allow(state: dict[str, Any], now: float, cfg: CircuitConfig) -> tuple[bool, float]:
    """Return ``(allowed, retry_after)`` and reserve a half-open slot."""

    _normalise(state, now)
    current = state["state"]
    if current == CircuitState.OPEN.value:
        return False, max(0.0, float(state["open_until"]) - now)
    if current == CircuitState.HALF_OPEN.value:
        if int(state.get("half_open_calls", 0)) >= cfg.half_open_max_calls:
            return False, max(0.0, float(state["open_until"]) - now)
        state["half_open_calls"] = int(state.get("half_open_calls", 0)) + 1
        state["version"] = int(state.get("version", 0)) + 1
    return True, 0.0


def _on_success(state: dict[str, Any], now: float, cfg: CircuitConfig) -> None:
    state["total_successes"] = int(state.get("total_successes", 0)) + 1
    if state["state"] == CircuitState.HALF_OPEN.value:
        state["half_open_successes"] = int(state.get("half_open_successes", 0)) + 1
        if state["half_open_successes"] >= cfg.half_open_success_threshold:
            _transition(state, CircuitState.CLOSED.value, now, "half_open_success")
            state["failures"] = 0
            state["open_until"] = 0.0
            state["half_open_calls"] = 0
            state["half_open_successes"] = 0
    else:
        state["failures"] = 0


def _on_failure(state: dict[str, Any], now: float, cfg: CircuitConfig, error: str) -> None:
    state["total_failures"] = int(state.get("total_failures", 0)) + 1
    state["last_error"] = error[:200]
    if state["state"] == CircuitState.HALF_OPEN.value:
        _transition(state, CircuitState.OPEN.value, now, "half_open_failure")
        state["opened_at"] = now
        state["open_until"] = now + cfg.cooldown_seconds
        state["half_open_calls"] = 0
        state["half_open_successes"] = 0
        return
    state["failures"] = int(state.get("failures", 0)) + 1
    if state["failures"] >= cfg.failure_threshold and state["state"] != CircuitState.OPEN.value:
        _transition(state, CircuitState.OPEN.value, now, "failure_threshold")
        state["opened_at"] = now
        state["open_until"] = now + cfg.cooldown_seconds
        state["half_open_calls"] = 0
        state["half_open_successes"] = 0


class CircuitStore(Protocol):
    def update(self, name: str, mutate: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        ...  # pragma: no cover - protocol

    def read(self, name: str) -> dict[str, Any] | None:
        ...  # pragma: no cover - protocol


class SqliteCircuitStore:
    """Shared breaker state backed by SQLite (single-host multi-process)."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS circuit_state (
                    name TEXT PRIMARY KEY,
                    state_json TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )

    def read(self, name: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT state_json FROM circuit_state WHERE name=?", (name,)).fetchone()
        return json.loads(row["state_json"]) if row else None

    def update(self, name: str, mutate: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT state_json FROM circuit_state WHERE name=?", (name,)).fetchone()
            state = json.loads(row["state_json"]) if row else _new_state(name)
            mutate(state)
            conn.execute(
                """
                INSERT INTO circuit_state(name, state_json, updated_at) VALUES(?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET state_json=excluded.state_json, updated_at=excluded.updated_at
                """,
                (name, json.dumps(state, ensure_ascii=False), time.time()),
            )
            conn.commit()
            return state
        except BaseException:
            conn.rollback()
            raise
        finally:
            conn.close()


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        config: CircuitConfig | None = None,
        *,
        store: CircuitStore | None = None,
        clock: Callable[[], float] = time.time,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.name = name
        self.config = config or CircuitConfig()
        self.store = store
        self._clock = clock
        self._lock = threading.RLock()
        self._state = _new_state(name)
        self._on_event = on_event

    # ------------------------------------------------------------------
    def _emit(self, event: str, snapshot: dict[str, Any]) -> None:
        if self._on_event is not None:
            try:
                self._on_event(event, snapshot)
            except Exception:  # pragma: no cover - observers must never break calls
                pass

    def _apply(self, mutate: Callable[[dict[str, Any]], Any]) -> dict[str, Any]:
        if self.store is not None:
            return self.store.update(self.name, mutate)
        with self._lock:
            mutate(self._state)
            return dict(self._state)

    def snapshot(self) -> CircuitSnapshot:
        raw = self.store.read(self.name) if self.store is not None else dict(self._state)
        raw = raw or _new_state(self.name)
        return CircuitSnapshot(
            name=self.name,
            state=raw["state"],
            failures=int(raw.get("failures", 0)),
            opened_at=float(raw.get("opened_at", 0.0)),
            open_until=float(raw.get("open_until", 0.0)),
            half_open_calls=int(raw.get("half_open_calls", 0)),
            half_open_successes=int(raw.get("half_open_successes", 0)),
            total_failures=int(raw.get("total_failures", 0)),
            total_successes=int(raw.get("total_successes", 0)),
            last_error=str(raw.get("last_error", "")),
            transitions=list(raw.get("transitions", [])),
        )

    # ------------------------------------------------------------------
    def acquire(self) -> None:
        """Reserve a call slot or raise :class:`CircuitOpenError`."""

        now = self._clock()
        result: dict[str, Any] = {}

        def mutate(state: dict[str, Any]) -> None:
            before = state["state"]
            allowed, retry_after = _allow(state, now, self.config)
            result.update(
                allowed=allowed,
                retry_after=retry_after,
                before=before,
                after=state["state"],
                state=state,
            )

        state = self._apply(mutate)
        if not result.get("allowed"):
            snapshot = CircuitSnapshot(
                name=self.name,
                state=state["state"],
                failures=int(state.get("failures", 0)),
                opened_at=float(state.get("opened_at", 0.0)),
                open_until=float(state.get("open_until", 0.0)),
                half_open_calls=int(state.get("half_open_calls", 0)),
                half_open_successes=int(state.get("half_open_successes", 0)),
            )
            self._emit("circuit_rejected", snapshot.to_dict())
            raise CircuitOpenError(
                f"依赖 {self.name} 处于熔断状态，冷却剩余 {result.get('retry_after', 0):.1f} 秒",
                details={"circuit": snapshot.to_dict()},
                retry_after=float(result.get("retry_after") or 0.0),
            )
        if result.get("before") != result.get("after"):
            self._emit("circuit_half_open", {"name": self.name})

    def record_success(self) -> None:
        now = self._clock()
        self._apply(lambda state: _on_success(state, now, self.config))
        self._emit("circuit_success", {"name": self.name, "state": self.snapshot().state})

    def record_neutral(self) -> None:
        """Release a HALF_OPEN probe slot without judging the dependency.

        Used when a call was admitted but never actually hit the dependency
        (for example an idempotent replay served from the database).  Without
        this, one replay could occupy the only half-open slot forever.
        """

        def mutate(state: dict[str, Any]) -> None:
            if state["state"] == CircuitState.HALF_OPEN.value and int(state.get("half_open_calls", 0)) > 0:
                state["half_open_calls"] = int(state["half_open_calls"]) - 1
                state["version"] = int(state.get("version", 0)) + 1

        self._apply(mutate)

    def record_failure(self, error: BaseException | str) -> None:
        now = self._clock()
        message = error if isinstance(error, str) else f"{type(error).__name__}: {error}"
        self._apply(lambda state: _on_failure(state, now, self.config, message))
        snap = self.snapshot()
        self._emit("circuit_failure", {"name": self.name, "state": snap.state, "failures": snap.failures})
        if snap.state == CircuitState.OPEN.value and snap.opened_at >= now - 0.001:
            self._emit("circuit_open", snap.to_dict())

    def reset(self) -> None:
        now = self._clock()

        def mutate(state: dict[str, Any]) -> None:
            state.update(_new_state(self.name))
            state["version"] = int(state.get("version", 0)) + 1

        self._apply(mutate)
        self._emit("circuit_reset", {"name": self.name, "at": now})

    @contextmanager
    def guard(self) -> Iterator[None]:
        """Context manager that records success/failure automatically."""

        self.acquire()
        try:
            yield
        except BaseException as exc:
            self.record_failure(exc)
            raise
        else:
            self.record_success()


def circuit_config_from_settings(settings: Any) -> CircuitConfig:
    return CircuitConfig(
        failure_threshold=max(1, int(getattr(settings, "circuit_failure_threshold", 5))),
        cooldown_seconds=max(0.1, float(getattr(settings, "circuit_cooldown_seconds", 30.0))),
        half_open_max_calls=max(1, int(getattr(settings, "circuit_half_open_max_calls", 1))),
        half_open_success_threshold=max(1, int(getattr(settings, "circuit_half_open_success_threshold", 1))),
    )