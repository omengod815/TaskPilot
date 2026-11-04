"""Classified retry with exponential backoff and jitter.

What this module refuses to do is as important as what it does:

* 401/403 -> never retried (retrying cannot fix a credential problem and may
  lock the account or burn quota);
* validation/4xx business errors -> never retried;
* timeouts for **non-idempotent** operations -> never retried, because the
  remote side may already have applied the side effect;
* 429 -> ``Retry-After`` is honoured when the server sends one;
* a retry that cannot finish before the task deadline is abandoned instead of
  silently overrunning the budget.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from app.core.errors import (
    ErrorCategory,
    RateLimitedError,
    TaskPilotError,
    classify_exception,
    retry_after_seconds,
)

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.4
    max_delay: float = 8.0
    jitter_ratio: float = 0.25
    respect_retry_after: bool = True
    max_retry_after: float = 30.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts 必须 >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delay 不能为负数")

    @classmethod
    def from_settings(cls, settings: Any) -> "RetryPolicy":
        return cls(
            max_attempts=max(1, int(getattr(settings, "retry_max_attempts", 3))),
            base_delay=float(getattr(settings, "retry_base_delay_seconds", 0.4)),
            max_delay=float(getattr(settings, "retry_max_delay_seconds", 8.0)),
            jitter_ratio=float(getattr(settings, "retry_jitter_ratio", 0.25)),
        )


@dataclass(frozen=True)
class Attempt:
    number: int
    delay: float | None
    category: ErrorCategory
    error: str
    will_retry: bool


def compute_delay(
    attempt: int,
    policy: RetryPolicy,
    *,
    retry_after: float | None = None,
    rng: random.Random | None = None,
) -> float:
    """Exponential backoff with full-width jitter, capped by ``max_delay``.

    ``attempt`` is 1-based.  The returned value is always in
    ``[0, max_delay]`` and never negative.
    """

    rng = rng or random
    if retry_after is not None and policy.respect_retry_after:
        return min(max(0.0, retry_after), policy.max_retry_after)
    raw = policy.base_delay * (2 ** max(0, attempt - 1))
    raw = min(raw, policy.max_delay)
    jitter = raw * policy.jitter_ratio
    # Cap the *final* value too: with symmetric jitter the result could
    # otherwise exceed ``max_delay`` by up to the jitter ratio.
    return min(policy.max_delay, max(0.0, raw - jitter + rng.uniform(0.0, 2 * jitter)))


def run_with_retry(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    is_idempotent: bool = True,
    deadline: float | None = None,
    on_attempt: Callable[[Attempt], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.time,
    rng: random.Random | None = None,
) -> T:
    """Execute ``operation`` under a classified retry policy."""

    policy = policy or RetryPolicy()
    last_exc: BaseException | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except BaseException as exc:  # noqa: BLE001 - re-raised below
            category = classify_exception(exc)
            last_exc = exc
            # Authentication / permission / validation / business failures are
            # permanent: retrying them only wastes quota.
            permanent = category in {
                ErrorCategory.AUTH,
                ErrorCategory.PERMISSION,
                ErrorCategory.VALIDATION,
                ErrorCategory.BUSINESS,
                ErrorCategory.CONFLICT,
                ErrorCategory.BUDGET,
            }
            # A timeout means "we do not know whether it happened".  Only a
            # known-idempotent operation may be retried.
            timeout_unknown = category is ErrorCategory.TIMEOUT and not is_idempotent
            retryable = (
                not permanent
                and not timeout_unknown
                and category
                in {
                    ErrorCategory.NETWORK,
                    ErrorCategory.TIMEOUT,
                    ErrorCategory.RATE_LIMIT,
                    ErrorCategory.SERVER,
                }
            )
            will_retry = retryable and attempt < policy.max_attempts

            delay: float | None = None
            if will_retry:
                retry_after = retry_after_seconds(exc) if policy.respect_retry_after else None
                delay = compute_delay(attempt, policy, retry_after=retry_after, rng=rng)
                now = clock()
                if deadline is not None and now + delay >= deadline:
                    # A retry would overrun the task budget; stop and surface
                    # the original error instead of silently exceeding it.
                    will_retry = False
                    delay = None

            if on_attempt is not None:
                on_attempt(
                    Attempt(
                        number=attempt,
                        delay=delay,
                        category=category,
                        error=f"{type(exc).__name__}: {exc}",
                        will_retry=will_retry,
                    )
                )

            if not will_retry:
                raise
            sleep(delay or 0.0)

    assert last_exc is not None
    raise last_exc


def raise_rate_limited(exc: BaseException) -> None:
    """Helper used by callers that need a typed 429 error with Retry-After."""

    raise RateLimitedError(
        f"上游限流：{type(exc).__name__}",
        retry_after=retry_after_seconds(exc),
    ) from exc


def budget_remaining(deadline: float | None, *, clock: Callable[[], float] = time.time) -> float | None:
    return None if deadline is None else max(0.0, deadline - clock())


def error_summary(exc: BaseException) -> dict[str, Any]:
    """Small, log-safe summary of an exception (never includes prompts/keys)."""

    return {
        "error_type": type(exc).__name__,
        "category": classify_exception(exc).value,
        "message": str(exc)[:300],
        "retryable": getattr(exc, "retryable", False) if isinstance(exc, TaskPilotError) else None,
    }