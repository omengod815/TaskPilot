"""Cooperative timeouts for blocking tool calls.

Python cannot kill a thread, so this is an *abandonment* timeout: the caller
stops waiting, the tool reports ``ToolTimeoutError``, and the underlying HTTP
client's own timeout (configured on every real client) releases the socket.
This limitation is documented instead of being hidden behind an ``async``
wrapper that would not actually cancel a blocked C call either.

The worker pool is created lazily and can be recreated after shutdown; a global
pool that stays dead after the first ``app.close()`` would break every later
TestClient in the same process (a real bug caught by the API test suite).
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

from app.core.errors import ToolTimeoutError

logger = logging.getLogger(__name__)
T = TypeVar("T")

_POOL: ThreadPoolExecutor | None = None
_POOL_LOCK = threading.Lock()


def _get_pool() -> ThreadPoolExecutor:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="tool-exec")
        return _POOL


def run_with_timeout(
    operation: Callable[[], T],
    timeout_seconds: float,
    *,
    label: str,
    pool: ThreadPoolExecutor | None = None,
) -> T:
    executor = pool or _get_pool()
    future: Future[T] = executor.submit(operation)
    try:
        return future.result(timeout=max(0.05, timeout_seconds))
    except FuturesTimeout as exc:
        future.cancel()
        raise ToolTimeoutError(
            f"{label} 执行超时（{timeout_seconds:.1f}s）；调用线程已放弃等待",
            details={"label": label, "timeout_seconds": timeout_seconds},
        ) from exc


def shutdown_pool(wait: bool = False) -> None:
    global _POOL
    with _POOL_LOCK:
        pool, _POOL = _POOL, None
    if pool is not None:
        pool.shutdown(wait=wait, cancel_futures=True)