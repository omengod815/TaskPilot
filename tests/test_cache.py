from app.core.cache import ResponseCache


def test_cache_hit_and_stats():
    cache = ResponseCache(ttl_seconds=60, redis_url="")
    key = "test-key"
    assert cache.get(key) is None
    cache.set(key, "hello")
    assert cache.get(key) == "hello"
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
