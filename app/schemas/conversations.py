from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class ConversationRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=4000)
    user_id: str = Field(default="local-demo", min_length=1, max_length=100)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    max_turns: int = Field(default=6, ge=2, le=12, description="最多 worker 发言轮数")

    @field_validator("goal")
    @classmethod
    def strip_goal(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("goal 至少需要 3 个字符")
        return value


class ConversationResponse(BaseModel):
    conversation_id: str
    status: str
    goal: str
    user_id: str
    session_id: str
    max_turns: int
    turn: int
    messages: list[dict[str, Any]] = Field(default_factory=list)
    observations: list[dict[str, Any]] = Field(default_factory=list)
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    trace: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = None
    stop_reason: str | None = None
    degraded_reasons: list[str] = Field(default_factory=list)
    agents: list[dict[str, Any]] = Field(default_factory=list)


class ConversationSummary(BaseModel):
    conversation_id: str
    user_id: str
    session_id: str
    goal: str
    status: str
    turn: int
    created_at: str
    updated_at: str