"""Unified error taxonomy for the TaskPilot agent platform.

The taxonomy exists so that retry, circuit-breaking and API error mapping all
speak the same language.  A failure is not just "an exception": the caller
needs to know whether it is *transient* (retry), *permanent* (do not retry) or
*unknown* (a side effect may or may not have happened).
"""

from __future__ import annotations

import asyncio
import socket
from enum import Enum
from typing import Any


class ErrorCategory(str, Enum):
    NETWORK = "network"          # DNS / connection reset / refused
    TIMEOUT = "timeout"          # read / connect / total timeout
    RATE_LIMIT = "rate_limit"    # HTTP 429
    SERVER = "server"            # HTTP 5xx
    AUTH = "auth"                # HTTP 401
    PERMISSION = "permission"    # HTTP 403
    VALIDATION = "validation"    # bad request / schema mismatch
    BUSINESS = "business"        # business rule rejected the operation
    CIRCUIT_OPEN = "circuit_open"
    BUDGET = "budget"
    CONFLICT = "conflict"        # idempotency / state conflict
    UNAVAILABLE = "unavailable"  # dependency explicitly unavailable
    UNKNOWN = "unknown"


class ErrorCode(str, Enum):
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    RATE_LIMITED = "RATE_LIMITED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    TASK_TIMEOUT = "TASK_TIMEOUT"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    TOOL_VALIDATION_ERROR = "TOOL_VALIDATION_ERROR"
    TOOL_NOT_ALLOWED = "TOOL_NOT_ALLOWED"
    UPSTREAM_UNAVAILABLE = "UPSTREAM_UNAVAILABLE"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    IDEMPOTENCY_CONFLICT = "IDEMPOTENCY_CONFLICT"
    IDEMPOTENCY_IN_PROGRESS = "IDEMPOTENCY_IN_PROGRESS"
    IDEMPOTENCY_RECORD_FAILED = "IDEMPOTENCY_RECORD_FAILED"
    APPROVAL_REQUIRED = "APPROVAL_REQUIRED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# HTTP status used when an error escapes to the API layer.
_STATUS_BY_CODE: dict[ErrorCode, int] = {
    ErrorCode.VALIDATION_ERROR: 422,
    ErrorCode.UNAUTHORIZED: 401,
    ErrorCode.FORBIDDEN: 403,
    ErrorCode.NOT_FOUND: 404,
    ErrorCode.CONFLICT: 409,
    ErrorCode.RATE_LIMITED: 429,
    ErrorCode.BUDGET_EXCEEDED: 422,
    ErrorCode.TASK_TIMEOUT: 504,
    ErrorCode.TOOL_TIMEOUT: 504,
    ErrorCode.TOOL_VALIDATION_ERROR: 422,
    ErrorCode.TOOL_NOT_ALLOWED: 403,
    ErrorCode.UPSTREAM_UNAVAILABLE: 503,
    ErrorCode.CIRCUIT_OPEN: 503,
    ErrorCode.IDEMPOTENCY_CONFLICT: 409,
    ErrorCode.IDEMPOTENCY_IN_PROGRESS: 409,
    ErrorCode.IDEMPOTENCY_RECORD_FAILED: 500,
    ErrorCode.APPROVAL_REQUIRED: 409,
    ErrorCode.INTERNAL_ERROR: 500,
}

_RETRYABLE_CODES = {
    ErrorCode.RATE_LIMITED,
    ErrorCode.TOOL_TIMEOUT,
    ErrorCode.UPSTREAM_UNAVAILABLE,
    ErrorCode.CIRCUIT_OPEN,
}


class TaskPilotError(Exception):
    """Base class carrying a stable machine-readable error contract."""

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    category: ErrorCategory = ErrorCategory.UNKNOWN
    retryable: bool = False
    http_status: int | None = None

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | None = None,
        category: ErrorCategory | None = None,
        retryable: bool | None = None,
        details: dict[str, Any] | None = None,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        if code is not None:
            self.code = code
        if category is not None:
            self.category = category
        if retryable is not None:
            self.retryable = retryable
        else:
            self.retryable = self.code in _RETRYABLE_CODES
        self.message = message
        self.details = details or {}
        self.retry_after = retry_after

    @property
    def status_code(self) -> int:
        return self.http_status or _STATUS_BY_CODE.get(self.code, 500)

    def to_dict(self, request_id: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "retryable": self.retryable,
            "category": self.category.value,
        }
        if request_id:
            payload["request_id"] = request_id
        if self.details:
            payload["details"] = self.details
        if self.retry_after is not None:
            payload["retry_after_seconds"] = round(self.retry_after, 3)
        return payload


class ValidationError(TaskPilotError):
    code = ErrorCode.VALIDATION_ERROR
    category = ErrorCategory.VALIDATION


class AuthenticationError(TaskPilotError):
    code = ErrorCode.UNAUTHORIZED
    category = ErrorCategory.AUTH


class PermissionDenied(TaskPilotError):
    code = ErrorCode.FORBIDDEN
    category = ErrorCategory.PERMISSION


class NotFoundError(TaskPilotError):
    code = ErrorCode.NOT_FOUND
    category = ErrorCategory.UNKNOWN


class ConflictError(TaskPilotError):
    code = ErrorCode.CONFLICT
    category = ErrorCategory.CONFLICT


class RateLimitedError(TaskPilotError):
    code = ErrorCode.RATE_LIMITED
    category = ErrorCategory.RATE_LIMIT
    retryable = True

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message, retry_after=retry_after)


class BudgetExceeded(TaskPilotError):
    """Raised when a hard execution budget is exhausted (never retryable)."""

    code = ErrorCode.BUDGET_EXCEEDED
    category = ErrorCategory.BUDGET

    def __init__(self, reason: str, snapshot: dict[str, Any] | None = None) -> None:
        self.reason = reason
        self.snapshot = snapshot or {}
        super().__init__(f"执行预算已耗尽：{reason}", details={"reason": reason, "budget": self.snapshot})


class TaskTimeout(BudgetExceeded):
    code = ErrorCode.TASK_TIMEOUT
    category = ErrorCategory.TIMEOUT


class ToolError(TaskPilotError):
    code = ErrorCode.INTERNAL_ERROR
    category = ErrorCategory.UNKNOWN


class ToolValidationError(ToolError):
    code = ErrorCode.TOOL_VALIDATION_ERROR
    category = ErrorCategory.VALIDATION


class ToolNotAllowed(ToolError):
    code = ErrorCode.TOOL_NOT_ALLOWED
    category = ErrorCategory.PERMISSION


class ToolTimeoutError(ToolError):
    code = ErrorCode.TOOL_TIMEOUT
    category = ErrorCategory.TIMEOUT
    retryable = True


class ToolUnavailableError(ToolError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    category = ErrorCategory.UNAVAILABLE
    retryable = True


class CircuitOpenError(TaskPilotError):
    code = ErrorCode.CIRCUIT_OPEN
    category = ErrorCategory.CIRCUIT_OPEN
    retryable = True


class IdempotencyConflict(TaskPilotError):
    code = ErrorCode.IDEMPOTENCY_CONFLICT
    category = ErrorCategory.CONFLICT


class IdempotencyInProgress(TaskPilotError):
    code = ErrorCode.IDEMPOTENCY_IN_PROGRESS
    category = ErrorCategory.CONFLICT
    retryable = True


class ApprovalRequired(TaskPilotError):
    code = ErrorCode.APPROVAL_REQUIRED
    category = ErrorCategory.BUSINESS


class UpstreamUnavailable(TaskPilotError):
    code = ErrorCode.UPSTREAM_UNAVAILABLE
    category = ErrorCategory.UNAVAILABLE
    retryable = True


def _status_from_exception(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def classify_exception(exc: BaseException) -> ErrorCategory:
    """Map an arbitrary exception onto the retry taxonomy.

    Ordering matters: explicit TaskPilot errors win, then HTTP status, then
    socket/timeout types, then the SDK exception names (so the classifier also
    works when ``openai`` is not installed).
    """

    if isinstance(exc, TaskPilotError):
        return exc.category
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return ErrorCategory.TIMEOUT
    if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, BrokenPipeError, ConnectionRefusedError)):
        return ErrorCategory.NETWORK
    if isinstance(exc, socket.gaierror):
        return ErrorCategory.NETWORK

    status = _status_from_exception(exc)
    if status is not None:
        if status == 401:
            return ErrorCategory.AUTH
        if status == 403:
            return ErrorCategory.PERMISSION
        if status == 408:
            return ErrorCategory.TIMEOUT
        if status == 409:
            return ErrorCategory.CONFLICT
        if status == 422:
            return ErrorCategory.VALIDATION
        if status == 429:
            return ErrorCategory.RATE_LIMIT
        if 500 <= status <= 599:
            return ErrorCategory.SERVER
        if 400 <= status <= 499:
            return ErrorCategory.VALIDATION

    name = type(exc).__name__.lower()
    if "timeout" in name:
        return ErrorCategory.TIMEOUT
    if "ratelimit" in name:
        return ErrorCategory.RATE_LIMIT
    if "authentication" in name:
        return ErrorCategory.AUTH
    if "permission" in name:
        return ErrorCategory.PERMISSION
    if "badrequest" in name or "validation" in name:
        return ErrorCategory.VALIDATION
    if "connection" in name:
        return ErrorCategory.NETWORK
    if "internalserver" in name:
        return ErrorCategory.SERVER
    return ErrorCategory.UNKNOWN


def is_retryable_exception(exc: BaseException) -> bool:
    return classify_exception(exc) in {
        ErrorCategory.NETWORK,
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.SERVER,
    }


def retry_after_seconds(exc: BaseException) -> float | None:
    """Extract ``Retry-After`` from an SDK/HTTP exception, if present."""

    for attr in ("retry_after",):
        value = getattr(exc, attr, None)
        if isinstance(value, (int, float)) and value >= 0:
            return float(value)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = None
    try:
        raw = headers.get("retry-after") or headers.get("Retry-After")
    except Exception:  # pragma: no cover - exotic header objects
        return None
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return None