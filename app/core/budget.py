"""Hard execution budgets for the ReAct loop.

Budgets are enforced in code, at every node that spends a scarce resource
(one reasoning step, one tool call, one model call, one token).  A prompt that
politely asks the model to "stop after 5 steps" is a suggestion; a budget is a
constraint.  Once a limit is hit we raise :class:`BudgetExceeded` and the graph
routes to a explicit ``partial`` terminal state.

The budget is stored inside ``AgentState`` as a plain dict, so it survives a
LangGraph checkpoint round-trip and keeps counting after a restart.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.errors import BudgetExceeded, TaskTimeout

# Rough, provider-independent estimate used only for *pre-flight* checks.
# Actual usage reported by the model response is always preferred.
CHARS_PER_TOKEN = 4
RESERVED_COMPLETION_TOKENS = 512


@dataclass(frozen=True)
class BudgetLimits:
    max_steps: int = 6
    max_tool_calls: int = 6
    max_model_calls: int = 8
    max_total_tokens: int = 24_000
    total_timeout_seconds: float = 120.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExecutionBudget:
    limits: BudgetLimits
    started_at: float
    deadline_at: float
    steps: int = 0
    tool_calls: int = 0
    model_calls: int = 0
    tokens_used: int = 0
    stop_reason: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------
    # construction / serialization
    # ------------------------------------------------------------------
    @classmethod
    def create(
        cls,
        limits: BudgetLimits | None = None,
        *,
        now: float | None = None,
        timeout_seconds: float | None = None,
    ) -> "ExecutionBudget":
        limits = limits or BudgetLimits()
        started = time.time() if now is None else now
        timeout = limits.total_timeout_seconds if timeout_seconds is None else timeout_seconds
        return cls(limits=limits, started_at=started, deadline_at=started + max(1.0, timeout))

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "ExecutionBudget":
        if not data:
            return cls.create()
        limits = BudgetLimits(**{k: v for k, v in (data.get("limits") or {}).items() if k in BudgetLimits.__annotations__})
        return cls(
            limits=limits,
            started_at=float(data.get("started_at") or time.time()),
            deadline_at=float(data.get("deadline_at") or time.time() + limits.total_timeout_seconds),
            steps=int(data.get("steps") or 0),
            tool_calls=int(data.get("tool_calls") or 0),
            model_calls=int(data.get("model_calls") or 0),
            tokens_used=int(data.get("tokens_used") or 0),
            stop_reason=data.get("stop_reason"),
            history=list(data.get("history") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["remaining_seconds"] = round(self.remaining_seconds(), 3)
        return payload

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def remaining_seconds(self, now: float | None = None) -> float:
        return max(0.0, self.deadline_at - (time.time() if now is None else now))

    def elapsed_seconds(self, now: float | None = None) -> float:
        return max(0.0, (time.time() if now is None else now) - self.started_at)

    def snapshot(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "max_steps": self.limits.max_steps,
            "tool_calls": self.tool_calls,
            "max_tool_calls": self.limits.max_tool_calls,
            "model_calls": self.model_calls,
            "max_model_calls": self.limits.max_model_calls,
            "tokens_used": self.tokens_used,
            "max_total_tokens": self.limits.max_total_tokens,
            "remaining_seconds": round(self.remaining_seconds(), 3),
            "stop_reason": self.stop_reason,
        }

    def _fail(self, reason: str) -> None:
        self.stop_reason = reason
        raise BudgetExceeded(reason, self.snapshot())

    def _check_deadline(self) -> None:
        if self.remaining_seconds() <= 0:
            self.stop_reason = "task_deadline_exceeded"
            raise TaskTimeout("任务总超时（deadline 已到）", self.snapshot())

    # ------------------------------------------------------------------
    # consumption points
    # ------------------------------------------------------------------
    def begin_step(self) -> int:
        """Called once per ReAct reasoning round, before the model decides."""

        self._check_deadline()
        if self.steps >= self.limits.max_steps:
            self._fail("max_react_steps_exceeded")
        self.steps += 1
        return self.steps

    def begin_tool_call(self, tool: str = "") -> int:
        self._check_deadline()
        if self.tool_calls >= self.limits.max_tool_calls:
            self._fail("max_tool_calls_exceeded")
        self.tool_calls += 1
        self.history.append({"kind": "tool", "tool": tool, "at": round(time.time(), 3)})
        return self.tool_calls

    def begin_model_call(self, prompt_text: str = "", *, reserved_tokens: int = RESERVED_COMPLETION_TOKENS) -> None:
        self._check_deadline()
        if self.model_calls >= self.limits.max_model_calls:
            self._fail("max_model_calls_exceeded")
        estimate = len(prompt_text) // CHARS_PER_TOKEN + reserved_tokens
        if self.tokens_used + estimate > self.limits.max_total_tokens:
            self._fail("token_budget_exceeded")
        self.model_calls += 1
        self.history.append({"kind": "model", "at": round(time.time(), 3), "prompt_tokens_estimate": estimate})

    def record_tokens(self, *, prompt_tokens: int = 0, completion_tokens: int = 0, total_tokens: int | None = None) -> None:
        used = total_tokens if total_tokens is not None else prompt_tokens + completion_tokens
        self.tokens_used += max(0, int(used))

    def require_time(self, *, label: str) -> None:
        if self.remaining_seconds() <= 0:
            self.stop_reason = f"timeout_before_{label}"
            raise TaskTimeout(f"任务总超时（剩余 0 秒），无法执行 {label}", self.snapshot())


def budget_from_state(state: dict[str, Any]) -> ExecutionBudget:
    raw = state.get("budget")
    if isinstance(raw, ExecutionBudget):
        return raw
    return ExecutionBudget.from_dict(raw if isinstance(raw, dict) else None)