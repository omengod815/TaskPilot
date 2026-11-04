"""Fault injection at the LLM and tool layers (no network, no real model)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

from app.core.errors import BudgetExceeded, ErrorCategory, ToolUnavailableError
from app.services.idempotency import IdempotencyStore
from app.services.llm_service import AdaptiveLLMService, LLMUnavailableError, StructuredOutputError
from app.tools import toolkit
from app.tools.registry import ToolContext, build_default_registry


class HttpFailure(Exception):
    def __init__(self, status: int, retry_after: str | None = None) -> None:
        super().__init__(f"http {status}")
        self.status_code = status
        if retry_after is not None:
            self.response = SimpleNamespace(headers={"Retry-After": retry_after})


def completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=12, completion_tokens=6),
    )


class FakeClient:
    def __init__(self, behavior):
        self.behavior = behavior
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.behavior(kwargs)


def llm_settings(settings, **overrides):
    base = {
        "mock_llm": False,
        "qwen_api_key": "test-key",
        "qwen_chat_model": "primary-model",
        "qwen_fallback_model": "",
        "llm_max_retries": 2,
        "retry_max_attempts": 3,
        "retry_base_delay_seconds": 0.001,
        "retry_max_delay_seconds": 0.01,
        "llm_cache_enabled": False,
        "llm_circuit_failure_threshold": 5,
        "llm_circuit_cooldown_seconds": 60,
    }
    base.update(overrides)
    return settings.model_copy(update=base)


def test_429_retry_after_is_honoured(settings):
    state = {"n": 0}

    def behavior(kwargs):
        state["n"] += 1
        if state["n"] == 1:
            raise HttpFailure(429, retry_after="0.01")
        return completion('{"title":"ok","answer":"done","assumptions":[],"next_steps":[],"limitations":[]}')

    llm = AdaptiveLLMService(llm_settings(settings))
    llm._client = FakeClient(behavior)
    result = llm.generate_structured_with_meta(
        [{"role": "user", "content": "hi"}],
        __import__("app.schemas.tasks", fromlist=["GeneralOutput"]).GeneralOutput,
    )
    assert state["n"] == 2
    assert result.meta.attempts == 2
    assert result.value.answer == "done"


def test_401_is_not_retried_and_does_not_fall_back(settings):
    primary = FakeClient(lambda kwargs: (_ for _ in ()).throw(HttpFailure(401)))
    fallback = FakeClient(lambda kwargs: completion('{"title":"f","answer":"fallback"}'))
    client = FakeClient(lambda kwargs: primary._create(**kwargs) if kwargs["model"] == "primary-model" else fallback._create(**kwargs))

    llm = AdaptiveLLMService(llm_settings(settings, qwen_fallback_model="fallback-model"))
    llm._client = client
    with pytest.raises(LLMUnavailableError) as exc:
        llm.generate_structured([{"role": "user", "content": "hi"}], __import__("app.schemas.tasks", fromlist=["GeneralOutput"]).GeneralOutput)
    assert getattr(exc.value, "category", None) is ErrorCategory.AUTH
    assert len(primary.calls) == 1
    assert fallback.calls == []


def test_circuit_opens_and_fallback_model_is_used(settings):
    primary = FakeClient(lambda kwargs: (_ for _ in ()).throw(HttpFailure(500)))
    fallback = FakeClient(lambda kwargs: completion('{"title":"fb","answer":"from fallback","assumptions":[],"next_steps":[],"limitations":[]}'))
    client = FakeClient(lambda kwargs: primary._create(**kwargs) if kwargs["model"] == "primary-model" else fallback._create(**kwargs))

    llm = AdaptiveLLMService(
        llm_settings(settings, qwen_fallback_model="fallback-model", llm_circuit_failure_threshold=1)
    )
    llm._client = client
    result = llm.generate_structured_with_meta(
        [{"role": "user", "content": "hi"}],
        __import__("app.schemas.tasks", fromlist=["GeneralOutput"]).GeneralOutput,
    )
    assert result.meta.fallback_used is True
    assert result.value.answer == "from fallback"
    assert llm.metrics()["circuit_open_count"] >= 1
    assert len(fallback.calls) == 1


def test_budget_is_enforced_inside_llm_calls(settings):
    def behavior(kwargs):
        return completion('{"title":"x","answer":"y","assumptions":[],"next_steps":[],"limitations":[]}')

    llm = AdaptiveLLMService(llm_settings(settings))
    llm._client = FakeClient(behavior)
    from app.core.budget import BudgetLimits, ExecutionBudget

    budget = ExecutionBudget.create(BudgetLimits(max_model_calls=0))
    with pytest.raises(BudgetExceeded):
        llm.generate_structured([{"role": "user", "content": "hi"}], __import__("app.schemas.tasks", fromlist=["GeneralOutput"]).GeneralOutput, budget=budget)


def test_search_tool_degrades_to_local_knowledge(settings, monkeypatch):
    idem = IdempotencyStore(settings.taskpilot_db_path)
    registry, _ = build_default_registry(settings, idem)

    def boom(*args, **kwargs):
        raise ToolUnavailableError("tavily down")

    monkeypatch.setattr(toolkit, "search_web", boom)
    ctx = ToolContext(
        task_id="tool-1",
        user_id="u",
        settings=settings,
        budget=__import__("app.core.budget", fromlist=["ExecutionBudget"]).ExecutionBudget.create(settings.budget_limits()),
        owner="owner",
    )
    result = registry.execute("search_web", {"query": "agent 工程", "top_k": 3}, task_type="research", ctx=ctx)
    assert result.ok is True
    assert result.degraded is True
    assert result.data["mode"] in {"local_keyword_search", "local_fallback_no_tavily_key"}
    assert result.data["degrade_reason"]


def test_tool_breaker_opens_and_stops_calling_dependency(settings, monkeypatch):
    cfg = settings.model_copy(
        update={
            "circuit_failure_threshold": 1,
            "circuit_cooldown_seconds": 60,
            "retry_max_attempts": 1,
            "tool_timeout_seconds": 1,
        }
    )
    idem = IdempotencyStore(cfg.taskpilot_db_path)
    registry, breakers = build_default_registry(cfg, idem)

    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise ToolUnavailableError("provider down")

    monkeypatch.setattr(toolkit, "search_web", boom)

    def context():
        return ToolContext(
            task_id="tool-2",
            user_id="u",
            settings=cfg,
            budget=__import__("app.core.budget", fromlist=["ExecutionBudget"]).ExecutionBudget.create(cfg.budget_limits()),
            owner="owner",
        )

    first = registry.execute("search_web", {"query": "x"}, task_type="research", ctx=context())
    second = registry.execute("search_web", {"query": "x"}, task_type="research", ctx=context())

    assert first.degraded is True
    assert second.degraded is True
    assert calls["n"] == 1, "the second call must fail fast at the breaker"
    assert breakers.snapshot()["tool:search_web"]["state"] == "OPEN"


def test_tool_validation_error_is_not_retried(settings):
    idem = IdempotencyStore(settings.taskpilot_db_path)
    registry, _ = build_default_registry(settings, idem)
    ctx = ToolContext(
        task_id="tool-3",
        user_id="u",
        settings=settings,
        budget=__import__("app.core.budget", fromlist=["ExecutionBudget"]).ExecutionBudget.create(settings.budget_limits()),
        owner="owner",
    )
    result = registry.execute("safe_calculate", {"expression": "x" * 500}, task_type="general", ctx=ctx)
    assert result.ok is False
    assert result.error_code == "TOOL_VALIDATION_ERROR"
    assert result.attempts == 0


def test_structured_output_repair_is_attempted_then_fails(settings):
    responses = iter(['{"unexpected": true}', '{"still": "wrong"}'])
    llm = AdaptiveLLMService(llm_settings(settings))
    llm._client = FakeClient(lambda kwargs: completion(next(responses)))
    with pytest.raises(StructuredOutputError):
        llm.generate_structured(
            [{"role": "user", "content": "json please"}],
            __import__("app.schemas.tasks", fromlist=["GeneralOutput"]).GeneralOutput,
        )