"""Tool registry: validation, allow-lists, permissions, timeouts, retries,
circuit breaking, idempotency and standardised results.

The model never calls Python directly.  It emits a *proposal*
(``{"tool": "...", "arguments": {...}}``); this registry decides whether that
proposal is allowed, validates the arguments with a Pydantic model, executes
inside the task budget, and returns a single :class:`ToolResult` shape that the
agent can reason about.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field, ValidationError as PydanticValidationError

from app.core.budget import ExecutionBudget
from app.core.circuit_breaker import CircuitBreaker, CircuitConfig, CircuitStore, SqliteCircuitStore
from app.core.config import Settings
from app.core.errors import (
    ApprovalRequired,
    ErrorCategory,
    ErrorCode,
    IdempotencyConflict,
    IdempotencyInProgress,
    ToolNotAllowed,
    ToolUnavailableError,
    ToolValidationError,
    classify_exception,
)
from app.core.metrics import METRICS, MetricsRegistry
from app.core.retry import RetryPolicy, run_with_retry
from app.core.timeouts import run_with_timeout
from app.services.idempotency import (
    ClaimOutcome,
    IdempotencyStore,
    IdempotencyStatus,
    build_idempotency_key,
    params_hash,
)
from app.tools import toolkit

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class CalculateArgs(BaseModel):
    expression: str = Field(min_length=1, max_length=120, description="仅包含数字、括号和 + - * / ** % 的表达式")


class AnalyzeCsvArgs(BaseModel):
    filename: str = Field(min_length=1, max_length=200, description="data/public 或上传目录中的 CSV 文件名")


class SearchWebArgs(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    top_k: int = Field(default=5, ge=1, le=10)


class DemoActionArgs(BaseModel):
    goal: str = Field(min_length=1, max_length=400)


# ---------------------------------------------------------------------------
# Specs and results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[[BaseModel, "ToolContext"], dict[str, Any]]
    allowed_task_types: frozenset[str]
    side_effect: bool = False
    requires_approval: bool = False
    idempotent: bool = True
    risk: str = "low"
    timeout_seconds: float | None = None
    breaker_name: str | None = None
    fallback: Callable[[BaseModel, "ToolContext"], dict[str, Any]] | None = None
    reconcile: Callable[[str], dict[str, Any] | None] | None = None

    def schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_model.model_json_schema(),
            "side_effect": self.side_effect,
            "requires_approval": self.requires_approval,
        }


@dataclass
class ToolContext:
    task_id: str
    user_id: str
    settings: Settings
    budget: ExecutionBudget
    owner: str
    idempotency_key: str | None = None
    approved: bool = False


@dataclass
class ToolResult:
    ok: bool
    tool: str
    data: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    retryable: bool = False
    category: str = "unknown"
    attempts: int = 0
    duration_ms: float = 0.0
    idempotency_key: str | None = None
    replayed: bool = False
    degraded: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "tool": self.tool,
            "data": self.data,
            "error_code": self.error_code,
            "error_message": (self.error_message or "")[:500],
            "retryable": self.retryable,
            "category": self.category,
            "attempts": self.attempts,
            "duration_ms": self.duration_ms,
            "idempotency_key": self.idempotency_key,
            "replayed": self.replayed,
            "degraded": self.degraded,
        }


class CircuitBreakerRegistry:
    def __init__(self, config: CircuitConfig, store: CircuitStore | None = None) -> None:
        self.config = config
        self.store = store
        self._lock = threading.RLock()
        self._breakers: dict[str, CircuitBreaker] = {}

    def get(self, name: str) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(name, self.config, store=self.store, on_event=self._on_event)
                self._breakers[name] = breaker
            return breaker

    @staticmethod
    def _on_event(event: str, payload: dict[str, Any]) -> None:
        if event in {"circuit_open", "circuit_half_open", "circuit_rejected"}:
            METRICS.inc("circuit_events_total", event=event, name=str(payload.get("name", "")))

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {name: breaker.snapshot().to_dict() for name, breaker in self._breakers.items()}


def _retry_policy(settings: Settings) -> RetryPolicy:
    return RetryPolicy.from_settings(settings)


class ToolRegistry:
    def __init__(
        self,
        settings: Settings,
        idempotency: IdempotencyStore,
        breakers: CircuitBreakerRegistry,
        *,
        metrics: MetricsRegistry = METRICS,
        concurrency: int | None = None,
    ) -> None:
        self.settings = settings
        self.idempotency = idempotency
        self.breakers = breakers
        self.metrics = metrics
        self._specs: dict[str, ToolSpec] = {}
        self._lock = threading.RLock()
        self._slots = threading.BoundedSemaphore(concurrency or settings.tool_concurrency)

    # ------------------------------------------------------------------
    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            if spec.name in self._specs:
                raise ValueError(f"tool already registered: {spec.name}")
            self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def names(self, task_type: str | None = None) -> list[str]:
        return sorted(
            name for name, spec in self._specs.items() if task_type is None or task_type in spec.allowed_task_types
        )

    def catalog(self, task_type: str) -> list[dict[str, Any]]:
        return [self._specs[name].schema() for name in self.names(task_type)]

    # ------------------------------------------------------------------
    def execute(
        self,
        name: str,
        raw_args: dict[str, Any] | None,
        *,
        task_type: str,
        ctx: ToolContext,
    ) -> ToolResult:
        started = time.perf_counter()
        spec = self._specs.get(name)
        if spec is None:
            return ToolResult(ok=False, tool=name, error_code=ErrorCode.TOOL_NOT_ALLOWED.value, error_message=f"工具 {name} 不在白名单中", category=ErrorCategory.VALIDATION.value)
        if task_type not in spec.allowed_task_types:
            return ToolResult(ok=False, tool=name, error_code=ErrorCode.TOOL_NOT_ALLOWED.value, error_message=f"任务类型 {task_type} 不允许调用 {name}", category=ErrorCategory.PERMISSION.value)
        if spec.requires_approval and not ctx.approved:
            raise ApprovalRequired(f"工具 {name} 需要人工审批后才能执行")

        try:
            args = spec.args_model.model_validate(raw_args or {})
        except PydanticValidationError as exc:
            self.metrics.inc("tool_calls_total", tool=name, outcome="validation_error")
            return ToolResult(
                ok=False, tool=name, error_code=ErrorCode.TOOL_VALIDATION_ERROR.value,
                error_message=f"参数校验失败：{exc.errors()[:2]}", category=ErrorCategory.VALIDATION.value,
                duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        # Concurrency guard: one slow dependency must not consume every worker.
        if not self._slots.acquire(timeout=max(0.1, ctx.budget.remaining_seconds())):
            return ToolResult(ok=False, tool=name, error_code=ErrorCode.UPSTREAM_UNAVAILABLE.value, error_message="工具并发已满，请稍后重试", retryable=True, category=ErrorCategory.UNAVAILABLE.value)

        try:
            return self._execute_validated(spec, args, ctx, started)
        finally:
            self._slots.release()

    # ------------------------------------------------------------------
    def _execute_validated(self, spec: ToolSpec, args: BaseModel, ctx: ToolContext, started: float) -> ToolResult:
        breaker = self.breakers.get(spec.breaker_name) if spec.breaker_name else None
        if breaker is not None:
            try:
                breaker.acquire()
            except Exception as exc:  # CircuitOpenError
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="circuit_open")
                if spec.fallback is not None:
                    return self._run_fallback(spec, args, ctx, started, reason=str(exc))
                return ToolResult(
                    ok=False, tool=spec.name, error_code=ErrorCode.CIRCUIT_OPEN.value,
                    error_message=str(exc), retryable=True, category=ErrorCategory.CIRCUIT_OPEN.value,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                )

        idem_key: str | None = None
        idem_owner = ctx.owner
        if spec.side_effect:
            idem_key = ctx.idempotency_key or build_idempotency_key(ctx.task_id, spec.name, args.model_dump())
            try:
                claim = self.idempotency.claim(
                    idem_key,
                    tool=spec.name,
                    args_hash=params_hash(args.model_dump()),
                    task_id=ctx.task_id,
                    owner=ctx.owner,
                    lease_seconds=self.settings.idempotency_lease_seconds,
                )
            except IdempotencyConflict as exc:
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="idempotency_conflict")
                return ToolResult(ok=False, tool=spec.name, error_code=ErrorCode.IDEMPOTENCY_CONFLICT.value, error_message=str(exc), category=ErrorCategory.CONFLICT.value)
            if claim.outcome is ClaimOutcome.SUCCEEDED:
                self.metrics.inc("idempotency_replays_total", tool=spec.name)
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="replayed")
                if breaker is not None:
                    breaker.record_success()
                return ToolResult(
                    ok=True, tool=spec.name, data=claim.record.result or {}, idempotency_key=idem_key,
                    replayed=True, attempts=claim.record.attempts,
                    duration_ms=round((time.perf_counter() - started) * 1000, 2),
                    degraded=bool((claim.record.result or {}).get("degraded")),
                )
            if claim.outcome is ClaimOutcome.IN_PROGRESS:
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="in_progress")
                if breaker is not None:
                    breaker.record_neutral()
                return ToolResult(ok=False, tool=spec.name, error_code=ErrorCode.IDEMPOTENCY_IN_PROGRESS.value, error_message="相同业务动作正在执行中", retryable=True, category=ErrorCategory.CONFLICT.value, idempotency_key=idem_key)
            if claim.outcome is ClaimOutcome.FAILED:
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="failed_replay")
                if breaker is not None:
                    breaker.record_neutral()
                record = claim.record
                return ToolResult(ok=False, tool=spec.name, error_code=record.error_code or ErrorCode.INTERNAL_ERROR.value, error_message=record.error_message or "上次执行失败", idempotency_key=idem_key, attempts=record.attempts, category=ErrorCategory.UNKNOWN.value)
            if claim.outcome is ClaimOutcome.UNKNOWN:
                reconciled = self._reconcile(spec, idem_key, claim.owner)
                if reconciled is not None:
                    self.idempotency.mark_succeeded(idem_key, claim.owner, reconciled)
                    self.metrics.inc("idempotency_reconciled_total", tool=spec.name)
                    if breaker is not None:
                        breaker.record_success()
                    return ToolResult(ok=True, tool=spec.name, data=reconciled, idempotency_key=idem_key, replayed=True, attempts=claim.record.attempts, duration_ms=round((time.perf_counter() - started) * 1000, 2))
                # Proven not applied externally -> safe to try again under the same key.
                self.idempotency.reset_for_retry(idem_key, claim.owner)
            # Make the stable key visible to the handler (file name, external
            # API request id, reconciliation lookup).
            ctx.idempotency_key = idem_key

        attempts = 0
        def on_attempt(attempt: Any) -> None:
            nonlocal attempts
            attempts = attempt.number
            if attempt.will_retry:
                self.metrics.inc("tool_retries_total", tool=spec.name)
                logger.warning(
                    "tool_retry tool=%s attempt=%s delay=%.3f category=%s",
                    spec.name, attempt.number, attempt.delay or 0.0, attempt.category.value,
                )

        timeout = spec.timeout_seconds or self.settings.tool_timeout_seconds
        # A single dependency call must never outlive the whole-task deadline.
        timeout = max(0.2, min(timeout, ctx.budget.remaining_seconds()))
        ctx.budget.begin_tool_call(spec.name)

        def call_once() -> dict[str, Any]:
            return run_with_timeout(lambda: spec.handler(args, ctx), timeout, label=f"tool:{spec.name}")

        try:
            data = run_with_retry(
                call_once,
                policy=_retry_policy(self.settings),
                is_idempotent=spec.idempotent and not spec.side_effect,
                deadline=ctx.budget.deadline_at,
                on_attempt=on_attempt,
            )
        except BaseException as exc:
            category = classify_exception(exc)
            message = str(exc)
            self.metrics.inc("tool_calls_total", tool=spec.name, outcome="error")
            self.metrics.observe("tool_latency_ms", (time.perf_counter() - started) * 1000, tool=spec.name)
            logger.warning("tool_failed tool=%s category=%s error=%s", spec.name, category.value, type(exc).__name__)
            if breaker is not None and category in {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.SERVER, ErrorCategory.UNAVAILABLE}:
                breaker.record_failure(exc)
            if spec.side_effect and idem_key is not None:
                try:
                    if category in {ErrorCategory.TIMEOUT, ErrorCategory.NETWORK, ErrorCategory.SERVER, ErrorCategory.UNKNOWN}:
                        self.idempotency.mark_unknown(idem_key, idem_owner, f"{type(exc).__name__}: {message}")
                    else:
                        self.idempotency.mark_failed(idem_key, idem_owner, getattr(exc, "code", ErrorCode.INTERNAL_ERROR).value if hasattr(getattr(exc, "code", None), "value") else ErrorCode.INTERNAL_ERROR.value, message)
                except Exception:
                    logger.exception("idempotency_update_failed key=%s", idem_key)
            if spec.fallback is not None and category in {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.SERVER, ErrorCategory.UNAVAILABLE}:
                return self._run_fallback(spec, args, ctx, started, reason=f"{type(exc).__name__}: {message}")
            return ToolResult(
                ok=False, tool=spec.name, error_code=getattr(exc, "code", ErrorCode.INTERNAL_ERROR).value if hasattr(getattr(exc, "code", None), "value") else ErrorCode.INTERNAL_ERROR.value,
                error_message=message, retryable=bool(getattr(exc, "retryable", False)), category=category.value,
                attempts=attempts, idempotency_key=idem_key, duration_ms=round((time.perf_counter() - started) * 1000, 2),
            )

        if breaker is not None:
            breaker.record_success()
        if spec.side_effect and idem_key is not None:
            try:
                self.idempotency.mark_succeeded(idem_key, idem_owner, data)
            except Exception as exc:
                # The external effect happened, but we failed to persist the
                # replay record.  Reporting success here would make the next
                # recovery replay an unrecorded action (v3 audit P0-03).
                logger.exception("idempotency_mark_succeeded_failed key=%s", idem_key)
                if breaker is not None:
                    breaker.record_neutral()
                duration = round((time.perf_counter() - started) * 1000, 2)
                self.metrics.inc("tool_calls_total", tool=spec.name, outcome="idempotency_record_failed")
                self.metrics.observe("tool_latency_ms", duration, tool=spec.name)
                payload = dict(data) if isinstance(data, dict) else {"result": data}
                payload["external_effect_applied"] = True
                return ToolResult(
                    ok=False,
                    tool=spec.name,
                    data=payload,
                    error_code=ErrorCode.IDEMPOTENCY_RECORD_FAILED.value,
                    error_message=f"工具已执行，但幂等结果写入失败：{type(exc).__name__}",
                    retryable=False,
                    category=ErrorCategory.UNKNOWN.value,
                    attempts=attempts or 1,
                    idempotency_key=idem_key,
                    duration_ms=duration,
                )
        self.metrics.inc("tool_calls_total", tool=spec.name, outcome="success")
        self.metrics.observe("tool_latency_ms", (time.perf_counter() - started) * 1000, tool=spec.name)
        logger.info("tool_succeeded tool=%s attempts=%s replayed=false", spec.name, attempts or 1)
        return ToolResult(
            ok=True, tool=spec.name, data=data, attempts=attempts or 1, idempotency_key=idem_key,
            degraded=bool(data.get("degraded")),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
        )

    # ------------------------------------------------------------------
    def _reconcile(self, spec: ToolSpec, key: str, owner: str) -> dict[str, Any] | None:
        if spec.reconcile is None:
            return None
        try:
            return spec.reconcile(key)
        except Exception:
            logger.exception("tool_reconcile_failed tool=%s key=%s", spec.name, key)
            return None

    def _run_fallback(self, spec: ToolSpec, args: BaseModel, ctx: ToolContext, started: float, *, reason: str) -> ToolResult:
        assert spec.fallback is not None
        try:
            data = spec.fallback(args, ctx)
        except Exception as exc:
            logger.exception("tool_fallback_failed tool=%s", spec.name)
            return ToolResult(ok=False, tool=spec.name, error_code=ErrorCode.UPSTREAM_UNAVAILABLE.value, error_message=f"降级路径也失败：{type(exc).__name__}", retryable=True, category=ErrorCategory.UNAVAILABLE.value, duration_ms=round((time.perf_counter() - started) * 1000, 2))
        data = {**data, "degraded": True, "degrade_reason": reason}
        self.metrics.inc("degraded_total", component=f"tool:{spec.name}", reason="dependency_unavailable")
        self.metrics.inc("tool_calls_total", tool=spec.name, outcome="degraded")
        logger.warning("tool_degraded tool=%s reason=%s", spec.name, reason)
        return ToolResult(ok=True, tool=spec.name, data=data, degraded=True, attempts=1, duration_ms=round((time.perf_counter() - started) * 1000, 2))


# ---------------------------------------------------------------------------
# Default registry
# ---------------------------------------------------------------------------
def build_default_registry(settings: Settings, idempotency: IdempotencyStore) -> tuple[ToolRegistry, CircuitBreakerRegistry]:
    store: CircuitStore | None = None
    try:
        store = SqliteCircuitStore(settings.taskpilot_db_path)
    except Exception:  # pragma: no cover - fall back to in-process state
        logger.warning("shared circuit store unavailable; using in-process states", exc_info=True)
    breakers = CircuitBreakerRegistry(
        CircuitConfig(
            failure_threshold=settings.circuit_failure_threshold,
            cooldown_seconds=settings.circuit_cooldown_seconds,
            half_open_max_calls=settings.circuit_half_open_max_calls,
            half_open_success_threshold=settings.circuit_half_open_success_threshold,
        ),
        store=store,
    )
    registry = ToolRegistry(settings, idempotency, breakers, concurrency=settings.tool_concurrency)

    registry.register(
        ToolSpec(
            name="safe_calculate",
            description="安全计算数学表达式，仅支持数字与 + - * / ** % 和括号。",
            args_model=CalculateArgs,
            handler=lambda args, ctx: toolkit.safe_calculate(args.expression),
            allowed_task_types=frozenset({"general", "data_analysis"}),
        )
    )
    registry.register(
        ToolSpec(
            name="analyze_csv",
            description="读取受控目录中的 CSV，返回行列、缺失值、数值统计和预测指标。",
            args_model=AnalyzeCsvArgs,
            handler=lambda args, ctx: toolkit.analyze_csv(args.filename),
            allowed_task_types=frozenset({"data_analysis"}),
        )
    )
    registry.register(
        ToolSpec(
            name="search_web",
            description="检索资料；未配置在线检索或依赖故障时降级为本地知识库关键词检索。",
            args_model=SearchWebArgs,
            handler=lambda args, ctx: toolkit.search_web(args.query, top_k=args.top_k),
            allowed_task_types=frozenset({"research"}),
            idempotent=True,
            breaker_name="tool:search_web",
            fallback=lambda args, ctx: toolkit.search_local_knowledge(args.query, top_k=args.top_k),
        )
    )
    registry.register(
        ToolSpec(
            name="execute_demo_action",
            description="生成外部操作计划文件（演示用，不会真实发送/支付/删除），必须人工审批。",
            args_model=DemoActionArgs,
            handler=lambda args, ctx: toolkit.execute_demo_action(
                args.goal, idempotency_key=ctx.idempotency_key, output_dir=ctx.settings.output_dir
            ),
            allowed_task_types=frozenset({"external_action"}),
            side_effect=True,
            requires_approval=True,
            idempotent=False,
            risk="high",
            reconcile=lambda key: toolkit.lookup_demo_action(key, output_dir=settings.output_dir),
        )
    )
    return registry, breakers