"""Per-identity sliding-window rate limiting (single-process demo).

Multi-replica deployments must move this counter to Redis; the interface is
deliberately small so that swap is a drop-in.  The limiter rejects with HTTP 429
and a ``Retry-After`` header, which is also what a well-behaved client expects.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Deque


class SlidingWindowLimiter:
    def __init__(self, limit_per_minute: int, window_seconds: float = 60.0) -> None:
        self.limit = max(0, limit_per_minute)
        self.window = window_seconds
        self._lock = threading.RLock()
        self._hits: dict[str, Deque[float]] = defaultdict(deque)
        self._last_cleanup = time.time()

    def allow(self, identity: str, *, now: float | None = None) -> tuple[bool, float]:
        if self.limit <= 0:
            return True, 0.0
        now = time.time() if now is None else now
        with self._lock:
            self._maybe_cleanup(now)
            bucket = self._hits[identity]
            cutoff = now - self.window
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self.limit:
                retry_after = max(0.0, self.window - (now - bucket[0]))
                return False, retry_after
            bucket.append(now)
            return True, 0.0

    def _maybe_cleanup(self, now: float) -> None:
        if now - self._last_cleanup < self.window:
            return
        self._last_cleanup = now
        cutoff = now - self.window
        stale = [key for key, bucket in self._hits.items() if not bucket or bucket[-1] < cutoff]
        for key in stale:
            del self._hits[key]

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {key: len(bucket) for key, bucket in self._hits.items()}