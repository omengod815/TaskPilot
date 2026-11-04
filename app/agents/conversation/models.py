"""Typed models for the free-form multi-agent conversation runtime.

This is deliberately separate from the single-graph ``AgentState``: the
conversation runtime is a message bus plus a supervisor, not a fixed sequence
of LangGraph nodes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

AgentName = Literal["research", "data", "writing", "general"]
SupervisorTarget = Literal["research", "data", "writing", "general", "finish"]

AGENT_NAMES: tuple[str, ...] = ("research", "data", "writing", "general")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class AgentAction(BaseModel):
    """One worker-agent turn: call a tool or produce a reply."""

    action: Literal["tool", "final"] = "final"
    tool: str | None = Field(default=None, max_length=64)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reply: str = Field(default="", max_length=8000)
    reason: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def validate_shape(self) -> "AgentAction":
        if self.action == "tool" and not self.tool:
            raise ValueError("action=tool 时必须提供 tool")
        if self.action == "final" and self.tool:
            self.tool = None
        return self


class SupervisorDecision(BaseModel):
    """The supervisor chooses the next speaker from the transcript."""

    next_agent: SupervisorTarget
    instruction: str = Field(default="", max_length=1200)
    reason: str = Field(default="", max_length=600)

    @model_validator(mode="after")
    def finish_has_no_instruction(self) -> "SupervisorDecision":
        if self.next_agent == "finish":
            self.instruction = ""
        return self


@dataclass
class ConversationState:
    conversation_id: str
    user_id: str
    session_id: str
    goal: str
    max_turns: int
    status: str = "running"
    turn: int = 0
    messages: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    handoffs: list[dict[str, Any]] = field(default_factory=list)
    trace: list[dict[str, Any]] = field(default_factory=list)
    budget: dict[str, Any] = field(default_factory=dict)
    final_answer: str | None = None
    stop_reason: str | None = None
    degraded_reasons: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    @classmethod
    def create(
        cls,
        *,
        goal: str,
        user_id: str,
        session_id: str,
        max_turns: int,
        budget: dict[str, Any],
        conversation_id: str | None = None,
    ) -> "ConversationState":
        return cls(
            conversation_id=conversation_id or uuid.uuid4().hex,
            user_id=user_id,
            session_id=session_id,
            goal=goal,
            max_turns=max_turns,
            budget=dict(budget),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "goal": self.goal,
            "max_turns": self.max_turns,
            "status": self.status,
            "turn": self.turn,
            "messages": list(self.messages),
            "observations": list(self.observations),
            "handoffs": list(self.handoffs),
            "trace": list(self.trace),
            "budget": dict(self.budget),
            "final_answer": self.final_answer,
            "stop_reason": self.stop_reason,
            "degraded_reasons": list(self.degraded_reasons),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationState":
        return cls(
            conversation_id=data["conversation_id"],
            user_id=data.get("user_id", "local-demo"),
            session_id=data.get("session_id", "default"),
            goal=data.get("goal", ""),
            max_turns=int(data.get("max_turns", 6)),
            status=data.get("status", "running"),
            turn=int(data.get("turn", 0)),
            messages=list(data.get("messages") or []),
            observations=list(data.get("observations") or []),
            handoffs=list(data.get("handoffs") or []),
            trace=list(data.get("trace") or []),
            budget=dict(data.get("budget") or {}),
            final_answer=data.get("final_answer"),
            stop_reason=data.get("stop_reason"),
            degraded_reasons=list(data.get("degraded_reasons") or []),
            created_at=data.get("created_at", utc_now()),
            updated_at=data.get("updated_at", utc_now()),
        )


def add_message(
    state: ConversationState,
    *,
    role: Literal["user", "supervisor", "agent", "tool"],
    content: str,
    agent: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    message = {
        "role": role,
        "agent": agent,
        "content": content,
        "turn": state.turn,
        "at": utc_now(),
        **extra,
    }
    state.messages.append(message)
    return message