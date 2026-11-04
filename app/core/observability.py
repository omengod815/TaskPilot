"""Observability helpers: structured events, timers, redaction."""

from __future__ import annotations

import logging
import re
import time
from contextlib import contextmanager
from typing import Any, Iterator

from app.core.context import current
from app.core.metrics import METRICS

_SECRET_PATTERNS = [
    re.compile(r"(sk-[A-Za-z0-9_\-]{8,})"),
    re.compile(r"(?i)(api[_-]?key|token|password|secret)([\"'=:\s]+)([^\s\"',}]+)"),
]


def redact(value: Any) -> Any:
    """Best-effort redaction for log payloads (never a security boundary)."""

    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if any(secret in str(key).lower() for secret in ("key", "token", "password", "secret", "authorization")):
                result[key] = "***"
            else:
                result[key] = redact(item)
        return result
    if isinstance(value, list):
        return [redact(item) for item in value]
    if isinstance(value, str):
        text = value
        for pattern in _SECRET_PATTERNS:
            if pattern.groups == 1:
                text = pattern.sub("***", text)
            else:
                text = pattern.sub(lambda m: f"{m.group(1)}{m.group(2)}***", text)
        return text
    return value


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    logger.log(level, event, extra={"fields": redact(fields)})


def log_exception(logger: logging.Logger, event: str, exc: BaseException, **fields: Any) -> None:
    logger.error(event, exc_info=exc, extra={"fields": redact(fields)})


class Stopwatch:
    def __init__(self) -> None:
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0.0

    def stop(self) -> float:
        self.elapsed_ms = round((time.perf_counter() - self._start) * 1000, 2)
        return self.elapsed_ms

    def __enter__(self) -> "Stopwatch":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()


@contextmanager
def timed(metric: str, **labels: Any) -> Iterator[Stopwatch]:
    watch = Stopwatch()
    try:
        yield watch
    finally:
        watch.stop()
        METRICS.observe(metric, watch.elapsed_ms, **labels)


def summarize_observation(value: Any, limit: int = 400) -> str:
    text = str(value)
    return text if len(text) <= limit else text[:limit] + "…"


def trace_event(node: str, status: str, **fields: Any) -> dict[str, Any]:
    from datetime import datetime, timezone

    return {
        "at": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "node": node,
        "status": status,
        **redact(fields),
    }


def current_ids() -> dict[str, str]:
    return current()