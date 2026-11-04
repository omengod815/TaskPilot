import hashlib
import json
import logging
import threading
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class ResponseCache:
    """Small response cache. Redis is optional; local TTLCache is the fallback."""

    def __init__(self, ttl_seconds: int = 600, maxsize: int = 1000, redis_url: str = "") -> None:
        self.ttl_seconds = max(0, ttl_seconds)
        self.local = TTLCache(maxsize=maxsize, ttl=max(1, self.ttl_seconds or 1))
        self.lock = threading.RLock()
        self.redis = None
        self.hits = 0
        self.misses = 0
        if redis_url:
            try:
                import redis
                self.redis = redis.Redis.from_url(redis_url, decode_responses=True, socket_timeout=1.5)
                self.redis.ping()
                logger.info("ResponseCache: Redis backend enabled")
            except Exception as exc:
                self.redis = None
                logger.warning("Redis unavailable; using process-local cache: %s", exc)

    @staticmethod
    def make_key(payload: dict[str, Any]) -> str:
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def get(self, key: str) -> str | None:
        if self.ttl_seconds <= 0:
            self.misses += 1
            return None
        if self.redis is not None:
            try:
                value = self.redis.get(key)
                if value is not None:
                    self.hits += 1
                    return value
            except Exception as exc:
                logger.warning("Redis cache read failed; falling back locally: %s", exc)
        with self.lock:
            value = self.local.get(key)
        if value is None:
            self.misses += 1
        else:
            self.hits += 1
        return value

    def set(self, key: str, value: str) -> None:
        if self.ttl_seconds <= 0:
            return
        with self.lock:
            self.local[key] = value
        if self.redis is not None:
            try:
                self.redis.setex(key, self.ttl_seconds, value)
            except Exception as exc:
                logger.warning("Redis cache write failed; local cache remains available: %s", exc)

    def clear(self) -> None:
        with self.lock:
            self.local.clear()
        if self.redis is not None:
            try:
                # Only clear this application's namespaced keys, never FLUSHDB.
                for key in self.redis.scan_iter(match="taskpilot:llm:*"):
                    self.redis.delete(key)
            except Exception as exc:
                logger.warning("Redis cache clear failed: %s", exc)

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
            "local_items": len(self.local),
            "ttl_seconds": self.ttl_seconds,
        }
