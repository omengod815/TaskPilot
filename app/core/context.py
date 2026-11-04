"""Request/task context propagated through logs, traces and metrics."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_task_id: ContextVar[str | None] = ContextVar("task_id", default=None)
_thread_id: ContextVar[str | None] = ContextVar("thread_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)

_VARS = {
    "request_id": _request_id,
    "task_id": _task_id,
    "thread_id": _thread_id,
    "trace_id": _trace_id,
}


def bind(**values: str | None) -> list[Token]:
    tokens: list[Token] = []
    for key, value in values.items():
        var = _VARS.get(key)
        if var is not None and value is not None:
            tokens.append(var.set(value))
    return tokens


def reset(tokens: list[Token]) -> None:
    for token in reversed(tokens):
        token.var.reset(token)  # type: ignore[attr-defined]


def current() -> dict[str, str]:
    return {key: value for key, var in _VARS.items() if (value := var.get()) is not None}


def current_request_id() -> str | None:
    return _request_id.get()


@contextmanager
def scope(**values: Any) -> Iterator[None]:
    tokens = bind(**{k: (str(v) if v is not None else None) for k, v in values.items()})
    try:
        yield
    finally:
        reset(tokens)