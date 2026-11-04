"""Core reliability primitives: budget, retry classification, breaker, idempotency."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from app.core.budget import BudgetLimits, ExecutionBudget
from app.core.circuit_breaker import (
    CircuitBreaker,
    CircuitConfig,
    CircuitOpenError,
    CircuitState,
    SqliteCircuitStore,
)
from app.core.errors import BudgetExceeded, IdempotencyConflict, TaskTimeout
from app.core.retry import RetryPolicy, compute_delay, run_with_retry
from app.services.idempotency import ClaimOutcome, IdempotencyStore, build_idempotency_key


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
def test_step_budget_stops_the_loop():
    budget = ExecutionBudget.create(BudgetLimits(max_steps=2))
    assert budget.begin_step() == 1
    assert budget.begin_step() == 2
    with pytest.raises(BudgetExceeded) as exc:
        budget.begin_step()
    assert exc.value.reason == "max_react_steps_exceeded"
    assert budget.snapshot()["steps"] == 2


def test_tool_call_budget():
    budget = ExecutionBudget.create(BudgetLimits(max_tool_calls=1))
    budget.begin_tool_call("safe_calculate")
    with pytest.raises(BudgetExceeded) as exc:
        budget.begin_tool_call("search_web")
    assert exc.value.reason == "max_tool_calls_exceeded"


def test_token_budget_preflight():
    budget = ExecutionBudget.create(BudgetLimits(max_total_tokens=600))
    with pytest.raises(BudgetExceeded) as exc:
        budget.begin_model_call("x" * 4000, reserved_tokens=512)
    assert exc.value.reason == "token_budget_exceeded"


def test_deadline_is_absolute_and_survives_serialization():
    budget = ExecutionBudget.create(BudgetLimits(total_timeout_seconds=10), now=1000.0)
    restored = ExecutionBudget.from_dict(budget.to_dict())
    assert restored.deadline_at == 1010.0
    # Force an expired deadline without sleeping.
    restored.deadline_at = 0.0
    with pytest.raises(TaskTimeout):
        restored.begin_model_call("prompt")


# ---------------------------------------------------------------------------
# retry
# ---------------------------------------------------------------------------
class HttpFailure(Exception):
    def __init__(self, status: int, retry_after: float | None = None) -> None:
        super().__init__(f"http {status}")
        self.status_code = status
        if retry_after is not None:
            self.response = SimpleNamespace(headers={"Retry-After": str(retry_after)})


def test_500_is_retried_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    def operation():
        calls["n"] += 1
        if calls["n"] < 3:
            raise HttpFailure(503)
        return "ok"

    result = run_with_retry(operation, policy=RetryPolicy(max_attempts=3, base_delay=0.01), sleep=sleeps.append)
    assert result == "ok"
    assert calls["n"] == 3
    assert len(sleeps) == 2


def test_401_is_not_retried():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise HttpFailure(401)

    with pytest.raises(HttpFailure):
        run_with_retry(operation, policy=RetryPolicy(max_attempts=3), sleep=lambda _: None)
    assert calls["n"] == 1


def test_429_honours_retry_after_header():
    sleeps: list[float] = []
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] == 1:
            raise HttpFailure(429, retry_after=2.5)
        return "ok"

    run_with_retry(operation, policy=RetryPolicy(max_attempts=2, base_delay=0.01), sleep=sleeps.append)
    assert sleeps == [2.5]


def test_timeout_is_not_retried_for_non_idempotent_operation():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise TimeoutError("read timeout")

    with pytest.raises(TimeoutError):
        run_with_retry(operation, policy=RetryPolicy(max_attempts=3), is_idempotent=False, sleep=lambda _: None)
    assert calls["n"] == 1


def test_timeout_is_retried_for_idempotent_operation():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        if calls["n"] < 2:
            raise TimeoutError("read timeout")
        return "ok"

    assert run_with_retry(operation, policy=RetryPolicy(max_attempts=3), is_idempotent=True, sleep=lambda _: None) == "ok"
    assert calls["n"] == 2


def test_retry_does_not_overrun_deadline():
    calls = {"n": 0}

    def operation():
        calls["n"] += 1
        raise HttpFailure(500)

    with pytest.raises(HttpFailure):
        run_with_retry(operation, policy=RetryPolicy(max_attempts=5, base_delay=10), deadline=1.0, sleep=lambda _: None)
    assert calls["n"] == 1


def test_backoff_is_capped_and_jittered():
    import random

    policy = RetryPolicy(base_delay=1.0, max_delay=4.0, jitter_ratio=0.25)
    delays = [compute_delay(attempt, policy, rng=random.Random(seed)) for attempt, seed in [(10, 1), (10, 2), (10, 3)]]
    assert all(0 <= d <= 4.0 for d in delays)
    assert len(set(delays)) > 1


# ---------------------------------------------------------------------------
# circuit breaker
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_breaker_open_half_open_close():
    clock = FakeClock()
    breaker = CircuitBreaker("dep", CircuitConfig(failure_threshold=2, cooldown_seconds=10), clock=clock)
    for _ in range(2):
        breaker.acquire()
        breaker.record_failure(RuntimeError("boom"))
    assert breaker.snapshot().state == CircuitState.OPEN.value

    with pytest.raises(CircuitOpenError) as exc:
        breaker.acquire()
    assert exc.value.retry_after and exc.value.retry_after > 0

    clock.now += 11
    breaker.acquire()  # probe reserved
    assert breaker.snapshot().state == CircuitState.HALF_OPEN.value
    with pytest.raises(CircuitOpenError):
        breaker.acquire()  # only one probe at a time
    breaker.record_success()
    assert breaker.snapshot().state == CircuitState.CLOSED.value


def test_half_open_failure_reopens():
    clock = FakeClock()
    breaker = CircuitBreaker("dep2", CircuitConfig(failure_threshold=1, cooldown_seconds=5), clock=clock)
    breaker.acquire()
    breaker.record_failure(RuntimeError("boom"))
    clock.now += 6
    breaker.acquire()
    breaker.record_failure(RuntimeError("probe failed"))
    assert breaker.snapshot().state == CircuitState.OPEN.value


def test_half_open_allows_exactly_one_concurrent_probe():
    clock = FakeClock()
    breaker = CircuitBreaker("dep3", CircuitConfig(failure_threshold=1, cooldown_seconds=5, half_open_max_calls=1), clock=clock)
    breaker.acquire()
    breaker.record_failure(RuntimeError("x"))
    clock.now += 6

    barrier = threading.Barrier(8)
    admitted: list[int] = []
    rejected: list[int] = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            breaker.acquire()
            admitted.append(index)
        except CircuitOpenError:
            rejected.append(index)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(admitted) == 1
    assert len(rejected) == 7


def test_shared_sqlite_store_is_visible_to_another_instance(tmp_path):
    store = SqliteCircuitStore(tmp_path / "circuit.sqlite3")
    first = CircuitBreaker("shared", CircuitConfig(failure_threshold=1, cooldown_seconds=30), store=store)
    second = CircuitBreaker("shared", CircuitConfig(failure_threshold=1, cooldown_seconds=30), store=store)
    first.acquire()
    first.record_failure(RuntimeError("down"))
    with pytest.raises(CircuitOpenError):
        second.acquire()


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------
def test_concurrent_claims_produce_one_owner(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.sqlite3")
    key = build_idempotency_key("task-1", "execute_demo_action", {"goal": "send"})
    barrier = threading.Barrier(8)
    outcomes: list[ClaimOutcome] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        claim = store.claim(key, tool="execute_demo_action", args_hash="hash", task_id="task-1", lease_seconds=30)
        with lock:
            outcomes.append(claim.outcome)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert outcomes.count(ClaimOutcome.CLAIMED) == 1
    assert outcomes.count(ClaimOutcome.IN_PROGRESS) == 7


def test_same_key_with_different_arguments_is_rejected(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.sqlite3")
    key = "tool:abc"
    store.claim(key, tool="execute_demo_action", args_hash="hash-a", task_id="t")
    with pytest.raises(IdempotencyConflict):
        store.claim(key, tool="execute_demo_action", args_hash="hash-b", task_id="t")


def test_succeeded_result_is_replayed(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.sqlite3")
    key = "tool:replay"
    claim = store.claim(key, tool="execute_demo_action", args_hash="h", task_id="t")
    store.mark_succeeded(key, claim.owner, {"ok": True, "effect": 1})
    replay = store.claim(key, tool="execute_demo_action", args_hash="h", task_id="t")
    assert replay.outcome is ClaimOutcome.SUCCEEDED
    assert replay.result == {"ok": True, "effect": 1}


def test_expired_lease_becomes_unknown_not_replayed(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.sqlite3")
    key = "tool:lease"
    first = store.claim(key, tool="execute_demo_action", args_hash="h", task_id="t", lease_seconds=0.01)
    import time

    time.sleep(0.05)
    second = store.claim(key, tool="execute_demo_action", args_hash="h", task_id="t", lease_seconds=30)
    assert second.outcome is ClaimOutcome.UNKNOWN
    assert second.owner != first.owner
    store.mark_succeeded(key, second.owner, {"reconciled": True})
    assert store.get(key).status.value == "SUCCEEDED"


def test_purge_expired_keeps_in_progress(tmp_path):
    store = IdempotencyStore(tmp_path / "idem.sqlite3", ttl_hours=0)
    store.claim("k1", tool="t", args_hash="h", task_id="t")
    assert store.purge_expired(now=time_far_future()) >= 0
    assert store.get("k1") is not None


def time_far_future() -> float:
    import time

    return time.time() + 10 * 86400