"""In-process metrics registry (counter + histogram).

Deliberately dependency-free: the project is a single-process SQLite demo, and
Prometheus + Grafana would add operational surface without adding evidence.
``/metrics`` exposes the snapshot as JSON so a fault-injection test can assert
on it directly.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Any

DEFAULT_BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10_000)


class MetricsRegistry:
    def __init__(self, buckets: tuple[int, ...] = DEFAULT_BUCKETS) -> None:
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], dict[str, Any]] = {}
        self.buckets = buckets

    @staticmethod
    def _key(metric: str, labels: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return metric, tuple(sorted((str(k), str(v)) for k, v in labels.items()))

    def inc(self, metric: str, value: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._counters[self._key(metric, labels)] += value

    def observe(self, metric: str, value: float, **labels: Any) -> None:
        key = self._key(metric, labels)
        with self._lock:
            entry = self._histograms.get(key)
            if entry is None:
                entry = {"count": 0, "sum": 0.0, "min": math.inf, "max": -math.inf, "buckets": [0] * len(self.buckets)}
                self._histograms[key] = entry
            entry["count"] += 1
            entry["sum"] += value
            entry["min"] = min(entry["min"], value)
            entry["max"] = max(entry["max"], value)
            for index, bound in enumerate(self.buckets):
                if value <= bound:
                    entry["buckets"][index] += 1

    def counter(self, name: str, **labels: Any) -> float:
        with self._lock:
            return self._counters.get(self._key(name, labels), 0.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = [
                {"name": name, "labels": dict(labels), "value": round(value, 6)}
                for (name, labels), value in sorted(self._counters.items())
            ]
            histograms = []
            for (name, labels), entry in sorted(self._histograms.items()):
                count = entry["count"]
                histograms.append(
                    {
                        "name": name,
                        "labels": dict(labels),
                        "count": count,
                        "sum": round(entry["sum"], 6),
                        "avg": round(entry["sum"] / count, 6) if count else 0.0,
                        "min": round(entry["min"], 6) if count else 0.0,
                        "max": round(entry["max"], 6) if count else 0.0,
                        "p50": self._percentile(entry["buckets"], count, 0.50),
                        "p95": self._percentile(entry["buckets"], count, 0.95),
                        "p99": self._percentile(entry["buckets"], count, 0.99),
                    }
                )
        return {"counters": counters, "histograms": histograms}

    def _percentile(self, bucket_counts: list[int], total: int, quantile: float) -> float:
        if total <= 0:
            return 0.0
        target = total * quantile
        cumulative = 0
        for index, count in enumerate(bucket_counts):
            cumulative += count
            if cumulative >= target:
                return float(self.buckets[index])
        return float(self.buckets[-1])

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


METRICS = MetricsRegistry()