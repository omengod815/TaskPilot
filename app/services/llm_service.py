"""LLM adapter with classified retry, circuit breaking and fallback.

Coordinated retry: the OpenAI SDK's own retry loop is disabled
(``max_retries=0``).  Having two independent retry layers multiplies attempts
(3 SDK x 3 app = 9 calls) and makes the budget meaningless.  One layer here
owns the policy, the metrics and the deadline.

Fallback is deliberately *not* used for authentication/validation errors: a
second model behind the same bad key will fail identically, and pretending
otherwise hides the real problem.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.core.budget import ExecutionBudget
from app.core.cache import ResponseCache
from app.core.circuit_breaker import CircuitBreaker, CircuitConfig, CircuitStore, SqliteCircuitStore
from app.core.config import Settings, get_settings
from app.core.errors import BudgetExceeded, ErrorCategory, ErrorCode, classify_exception, retry_after_seconds
from app.core.metrics import METRICS
from app.core.retry import Attempt, RetryPolicy, run_with_retry

logger = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class LLMUnavailableError(RuntimeError):
    def __init__(self, message: str, *, code: ErrorCode = ErrorCode.UPSTREAM_UNAVAILABLE, category: ErrorCategory = ErrorCategory.UNAVAILABLE, retryable: bool = True, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.category = category
        self.retryable = retryable
        self.retry_after = retry_after


class StructuredOutputError(RuntimeError):
    pass


@dataclass
class LLMCallMeta:
    model: str = ""
    attempts: int = 0
    fallback_used: bool = False
    degraded: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_ms: float = 0.0
    category: str = "unknown"
    cache_hit: bool = False
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "attempts": self.attempts,
            "fallback_used": self.fallback_used,
            "degraded": self.degraded,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "duration_ms": self.duration_ms,
            "category": self.category,
            "cache_hit": self.cache_hit,
            "errors": self.errors[-3:],
        }


class AdaptiveLLMService:
    """Bounded retry + circuit breaker + optional fallback + schema validation."""

    def __init__(
        self,
        settings: Settings | None = None,
        cache: ResponseCache | None = None,
        *,
        breakers: dict[str, CircuitBreaker] | None = None,
        circuit_store: CircuitStore | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache = cache or ResponseCache(
            ttl_seconds=self.settings.llm_cache_ttl_seconds,
            redis_url=self.settings.redis_url,
        )
        self._client: OpenAI | None = None
        self._client_lock = threading.Lock()
        self.retry_count = 0
        self.fallback_count = 0
        self.failure_count = 0
        self.circuit_open_count = 0
        self._breakers: dict[str, CircuitBreaker] = breakers or {}
        self._breaker_lock = threading.RLock()
        self._retry_policy = RetryPolicy.from_settings(self.settings)
        self._circuit_store = circuit_store
        self._local = threading.local()

    # ------------------------------------------------------------------
    def _breaker(self, model: str) -> CircuitBreaker:
        with self._breaker_lock:
            breaker = self._breakers.get(model)
            if breaker is None:
                store = self._circuit_store
                if store is None:
                    try:
                        store = SqliteCircuitStore(self.settings.taskpilot_db_path)
                        self._circuit_store = store
                    except Exception:  # pragma: no cover
                        store = None
                breaker = CircuitBreaker(
                    f"llm:{model}",
                    CircuitConfig(
                        failure_threshold=self.settings.llm_circuit_failure_threshold,
                        cooldown_seconds=float(self.settings.llm_circuit_cooldown_seconds),
                        half_open_max_calls=1,
                    ),
                    store=store,
                    on_event=self._on_circuit_event,
                )
                self._breakers[model] = breaker
            return breaker

    def _on_circuit_event(self, event: str, payload: dict[str, Any]) -> None:
        METRICS.inc("circuit_events_total", event=event, name=str(payload.get("name", "")))
        if event in {"circuit_open", "circuit_rejected"}:
            self.circuit_open_count += 1

    def _get_client(self) -> OpenAI:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if not self.settings.qwen_api_key:
                        raise LLMUnavailableError(
                            "未配置 QWEN_API_KEY。请检查 .env；测试时可将 MOCK_LLM=1。",
                            category=ErrorCategory.AUTH,
                            retryable=False,
                        )
                    self._client = OpenAI(
                        api_key=self.settings.qwen_api_key,
                        base_url=self.settings.qwen_base_url,
                        timeout=self.settings.llm_timeout_seconds,
                        max_retries=0,  # single, app-level retry layer
                    )
        return self._client

    # ------------------------------------------------------------------
    def _call_model(
        self,
        messages: list[dict[str, str]],
        model: str,
        *,
        user_id: str,
        budget: ExecutionBudget | None,
        cacheable: bool = True,
    ) -> tuple[str, LLMCallMeta]:
        meta = LLMCallMeta(model=model)
        prompt_text = json.dumps(messages, ensure_ascii=False)
        cache_key = "taskpilot:llm:" + self.cache.make_key(
            {"user_id": user_id, "model": model, "messages": messages, "json_mode": self.settings.llm_json_mode}
        )
        if cacheable and self.settings.llm_cache_enabled:
            cached = self.cache.get(cache_key)
            if cached is not None:
                meta.cache_hit = True
                meta.attempts = 0
                METRICS.inc("llm_calls_total", model=model, outcome="cache_hit")
                logger.info("llm_cache_hit model=%s", model)
                return cached, meta

        breaker = self._breaker(model)
        try:
            breaker.acquire()
        except Exception as exc:
            raise LLMUnavailableError(
                f"模型 {model} 熔断中：{exc}",
                category=ErrorCategory.CIRCUIT_OPEN,
                retry_after=getattr(exc, "retry_after", None),
            ) from exc

        started = time.perf_counter()
        attempt_errors: list[str] = []

        def on_attempt(attempt: Attempt) -> None:
            if attempt.will_retry:
                self.retry_count += 1
                METRICS.inc("llm_retries_total", model=model, category=attempt.category.value)
                logger.warning(
                    "llm_retry model=%s attempt=%s delay=%.3f category=%s",
                    model, attempt.number, attempt.delay or 0.0, attempt.category.value,
                )

        def once() -> str:
            meta.attempts += 1
            if budget is not None:
                budget.begin_model_call(prompt_text)
            kwargs: dict[str, Any] = {"model": model, "messages": messages, "temperature": 0}
            if self.settings.llm_json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            if budget is not None:
                kwargs["timeout"] = max(0.5, min(self.settings.llm_timeout_seconds, budget.remaining_seconds()))
            try:
                response = self._get_client().chat.completions.create(**kwargs)
            except BaseException as exc:
                attempt_errors.append(f"{type(exc).__name__}: {str(exc)[:160]}")
                raise
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            meta.prompt_tokens += prompt_tokens
            meta.completion_tokens += completion_tokens
            if budget is not None:
                budget.record_tokens(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            METRICS.inc("llm_tokens_total", model=model, kind="prompt", value=prompt_tokens)
            METRICS.inc("llm_tokens_total", model=model, kind="completion", value=completion_tokens)
            content = response.choices[0].message.content or ""
            if not content.strip():
                raise StructuredOutputError("模型返回了空内容。")
            return content

        try:
            content = run_with_retry(
                once,
                policy=self._retry_policy,
                is_idempotent=True,  # chat completion is read-only
                deadline=budget.deadline_at if budget is not None else None,
                on_attempt=on_attempt,
            )
        except BaseException as exc:
            category = classify_exception(exc)
            meta.category = category.value
            meta.errors = attempt_errors
            meta.duration_ms = round((time.perf_counter() - started) * 1000, 2)
            self.failure_count += 1
            METRICS.inc("llm_calls_total", model=model, outcome="error")
            METRICS.observe("llm_latency_ms", meta.duration_ms, model=model)
            if category in {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.SERVER, ErrorCategory.UNAVAILABLE, ErrorCategory.RATE_LIMIT}:
                breaker.record_failure(exc)
            else:
                breaker.record_neutral()
            if isinstance(exc, BudgetExceeded):
                raise
            raise LLMUnavailableError(
                f"模型 {model} 调用失败：{type(exc).__name__}: {str(exc)[:200]}",
                code=ErrorCode.UPSTREAM_UNAVAILABLE,
                category=category,
                retryable=category in {ErrorCategory.NETWORK, ErrorCategory.TIMEOUT, ErrorCategory.RATE_LIMIT, ErrorCategory.SERVER, ErrorCategory.CIRCUIT_OPEN},
                retry_after=retry_after_seconds(exc),
            ) from exc

        breaker.record_success()
        meta.duration_ms = round((time.perf_counter() - started) * 1000, 2)
        meta.category = "ok"
        if meta.attempts == 0:
            meta.attempts = 1
        METRICS.inc("llm_calls_total", model=model, outcome="success")
        METRICS.observe("llm_latency_ms", meta.duration_ms, model=model)
        if cacheable and self.settings.llm_cache_enabled:
            self.cache.set(cache_key, content)
        return content, meta

    # ------------------------------------------------------------------
    def generate_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        user_id: str = "local-demo",
        *,
        budget: ExecutionBudget | None = None,
    ) -> T:
        return self.generate_structured_with_meta(messages, schema, user_id=user_id, budget=budget).value

    def generate_structured_with_meta(
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        user_id: str = "local-demo",
        budget: ExecutionBudget | None = None,
    ) -> "StructuredCall[T]":
        if self.settings.mock_llm:
            raise LLMUnavailableError(
                "MOCK_LLM=1 时应由 MockDecider / mock 分支提供确定性输出；此方法不会伪造真实模型调用。",
                category=ErrorCategory.VALIDATION,
                retryable=False,
            )
        models_to_try = [self.settings.qwen_chat_model]
        if self.settings.qwen_fallback_model and self.settings.qwen_fallback_model != self.settings.qwen_chat_model:
            models_to_try.append(self.settings.qwen_fallback_model)

        last_error: BaseException | None = None
        for index, model in enumerate(models_to_try):
            try:
                raw, meta = self._call_model(messages, model, user_id=user_id, budget=budget)
                parsed = self._validate_with_repair(raw, messages, schema, model, user_id, budget, meta)
                meta.fallback_used = index > 0
                if meta.fallback_used:
                    self.fallback_count += 1
                    METRICS.inc("llm_fallback_total", model=model)
                return StructuredCall(value=parsed, meta=meta)
            except (LLMUnavailableError, StructuredOutputError) as exc:
                last_error = exc
                category = getattr(exc, "category", classify_exception(exc))
                # Authentication / validation failures will repeat on the
                # fallback model (same key, same request); do not mask them.
                if category in {ErrorCategory.AUTH, ErrorCategory.PERMISSION, ErrorCategory.VALIDATION}:
                    raise
                if index + 1 < len(models_to_try):
                    logger.warning("primary_model_failed; trying fallback model=%s error=%s", models_to_try[index + 1], type(exc).__name__)
                    continue
                break
        if isinstance(last_error, StructuredOutputError):
            # Keep schema failures distinguishable from transport failures:
            # retrying the same transport layer cannot fix a malformed answer.
            raise last_error
        raise LLMUnavailableError(
            f"主模型/备用模型均不可用：{type(last_error).__name__ if last_error else 'unknown'}",
            category=getattr(last_error, "category", ErrorCategory.UNAVAILABLE),
            retryable=getattr(last_error, "retryable", True),
        ) from last_error

    def _validate_with_repair(
        self,
        raw: str,
        messages: list[dict[str, str]],
        schema: type[T],
        model: str,
        user_id: str,
        budget: ExecutionBudget | None,
        meta: LLMCallMeta,
    ) -> T:
        last_exc: BaseException | None = None
        for attempt in range(2):
            try:
                parsed = self._parse_json(raw)
                return schema.model_validate(parsed)
            except (ValueError, ValidationError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt == 1:
                    break
                repair_messages = messages + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            "上一个输出不符合要求。请修复为严格 JSON，必须符合以下 Pydantic JSON Schema，"
                            "不要输出 Markdown 代码围栏或解释。\n"
                            + json.dumps(schema.model_json_schema(), ensure_ascii=False)
                            + "\n校验错误：" + str(exc)
                        ),
                    },
                ]
                raw, repair_meta = self._call_model(repair_messages, model, user_id=user_id, budget=budget, cacheable=False)
                meta.attempts += repair_meta.attempts
                meta.prompt_tokens += repair_meta.prompt_tokens
                meta.completion_tokens += repair_meta.completion_tokens
                meta.errors.extend(repair_meta.errors)
        raise StructuredOutputError(f"模型输出未通过 {schema.__name__} 校验：{last_exc}") from last_exc

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        text = raw.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("没有找到 JSON 对象。")
        parsed = json.loads(text[start:end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("顶层 JSON 必须是对象。")
        return parsed

    # ------------------------------------------------------------------
    def metrics(self) -> dict[str, Any]:
        return {
            "retry_count": self.retry_count,
            "fallback_count": self.fallback_count,
            "failure_count": self.failure_count,
            "circuit_open_count": self.circuit_open_count,
            "circuits": {name: breaker.snapshot().to_dict() for name, breaker in self._breakers.items()},
            "cache": self.cache.stats(),
            "breaker_backend": "sqlite" if self._circuit_store is not None else "in-process",
        }


@dataclass
class StructuredCall:
    value: Any
    meta: LLMCallMeta

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value.model_dump(mode="json") if hasattr(self.value, "model_dump") else self.value, "meta": self.meta.to_dict()}