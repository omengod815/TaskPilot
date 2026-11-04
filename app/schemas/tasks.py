from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

TaskType = Literal["research", "data_analysis", "writing", "general", "external_action"]
TaskStatus = Literal[
    "running",
    "completed",
    "partial",
    "pending_approval",
    "interrupted",
    "cancelled",
    "failed",
]


class TaskRequest(BaseModel):
    goal: str = Field(min_length=3, max_length=4000, description="用户希望系统完成的目标")
    user_id: str = Field(default="local-demo", min_length=1, max_length=100)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    task_type: TaskType | None = None
    input_file: str | None = Field(default=None, max_length=200)
    approved: bool = False

    @field_validator("input_file")
    @classmethod
    def strip_input_file(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("goal")
    @classmethod
    def strip_goal(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 3:
            raise ValueError("goal 至少需要 3 个字符")
        return value


class ApprovalRequest(BaseModel):
    user_id: str = Field(default="local-demo", min_length=1, max_length=100)
    approved: bool
    reason: str = Field(default="", max_length=1000)


class TaskPlan(BaseModel):
    task_type: TaskType
    steps: list[str] = Field(min_length=1, max_length=8)
    tools: list[str] = Field(default_factory=list, max_length=8)
    requires_approval: bool = False
    context_budget_chars: int = Field(default=6000, ge=1000, le=12000)
    rationale: str = Field(default="", max_length=1000)


class SourceItem(BaseModel):
    title: str
    source: str
    excerpt: str = ""
    url: str = ""


class ResearchOutput(BaseModel):
    title: str
    summary: str
    findings: list[str] = Field(default_factory=list, max_length=12)
    sources: list[SourceItem] = Field(default_factory=list, max_length=12)
    limitations: list[str] = Field(default_factory=list, max_length=8)
    next_steps: list[str] = Field(default_factory=list, max_length=8)


class DataAnalysisOutput(BaseModel):
    title: str
    summary: str
    row_count: int = Field(default=0, ge=0)
    columns: list[str] = Field(default_factory=list)
    metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    findings: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class WritingOutput(BaseModel):
    title: str
    audience: str = "通用读者"
    tone: str = "清晰、专业"
    final_text: str
    assumptions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)


class GeneralOutput(BaseModel):
    title: str
    answer: str
    assumptions: list[str] = Field(default_factory=list)
    next_steps: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class TaskResponse(BaseModel):
    task_id: str
    status: TaskStatus
    task_type: TaskType | None = None
    goal: str
    plan: dict[str, Any] | None = None
    answer: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    error_code: str | None = None
    stop_reason: str | None = None
    observations: list[dict[str, Any]] = Field(default_factory=list)
    budget: dict[str, Any] = Field(default_factory=dict)


class MemoryWriteRequest(BaseModel):
    user_id: str = Field(default="local-demo", min_length=1, max_length=100)
    session_id: str = Field(default="default", min_length=1, max_length=100)
    content: str = Field(min_length=4, max_length=2000)
    kind: Literal["short_term", "long_term"] = "long_term"
    tags: list[str] = Field(default_factory=list, max_length=10)
    importance: float = Field(default=0.7, ge=0.0, le=1.0)