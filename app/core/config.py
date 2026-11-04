"""Typed configuration (12-factor, environment driven).

Every reliability knob is a setting instead of a module constant so that a
fault-injection test, a staging profile and production can differ without
editing code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.budget import BudgetLimits


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "TaskPilot Agent Platform"
    environment: str = "local"
    api_access_token: str = Field(default="", validation_alias="TASKPILOT_API_TOKEN")
    log_level: str = Field(default="INFO", validation_alias="TASKPILOT_LOG_LEVEL")
    log_json: bool = Field(default=True, validation_alias="TASKPILOT_LOG_JSON")

    # ------------------------------------------------------------------ llm
    qwen_api_key: str = Field(default="", validation_alias="QWEN_API_KEY")
    qwen_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        validation_alias=AliasChoices("QWEN_BASE_URL", "LLM_BASE_URL"),
    )
    qwen_chat_model: str = Field(default="qwen3.8-flash", validation_alias="QWEN_CHAT_MODEL")
    qwen_fallback_model: str = Field(default="", validation_alias="QWEN_FALLBACK_MODEL")
    mock_llm: bool = Field(default=True, validation_alias="MOCK_LLM")
    llm_timeout_seconds: float = Field(default=45.0, ge=1.0, validation_alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, ge=0, le=5, validation_alias="LLM_MAX_RETRIES")
    llm_json_mode: bool = Field(default=False, validation_alias="LLM_JSON_MODE")
    llm_cache_enabled: bool = Field(default=True, validation_alias="LLM_CACHE_ENABLED")
    llm_cache_ttl_seconds: int = Field(default=600, ge=0, validation_alias="LLM_CACHE_TTL_SECONDS")
    redis_url: str = Field(default="", validation_alias="REDIS_URL")

    # --------------------------------------------------------------- budgets
    react_max_steps: int = Field(default=6, ge=1, le=20, validation_alias="REACT_MAX_STEPS")
    react_max_tool_calls: int = Field(default=6, ge=1, le=20, validation_alias="REACT_MAX_TOOL_CALLS")
    react_max_model_calls: int = Field(default=8, ge=1, le=30, validation_alias="REACT_MAX_MODEL_CALLS")
    task_timeout_seconds: float = Field(default=120.0, ge=5.0, le=1800.0, validation_alias="TASK_TIMEOUT_SECONDS")
    task_max_tokens: int = Field(default=24_000, ge=1000, validation_alias="TASK_MAX_TOKENS")
    tool_timeout_seconds: float = Field(default=20.0, ge=1.0, le=300.0, validation_alias="TOOL_TIMEOUT_SECONDS")

    # ----------------------------------------------------------------- retry
    retry_max_attempts: int = Field(default=3, ge=1, le=8, validation_alias="RETRY_MAX_ATTEMPTS")
    retry_base_delay_seconds: float = Field(default=0.4, ge=0.0, le=10.0, validation_alias="RETRY_BASE_DELAY_SECONDS")
    retry_max_delay_seconds: float = Field(default=8.0, ge=0.0, le=120.0, validation_alias="RETRY_MAX_DELAY_SECONDS")
    retry_jitter_ratio: float = Field(default=0.25, ge=0.0, le=1.0, validation_alias="RETRY_JITTER_RATIO")

    # ------------------------------------------------------------- circuits
    llm_circuit_failure_threshold: int = Field(default=3, ge=1, le=20, validation_alias="LLM_CIRCUIT_FAILURE_THRESHOLD")
    llm_circuit_cooldown_seconds: int = Field(default=30, ge=1, le=600, validation_alias="LLM_CIRCUIT_COOLDOWN_SECONDS")
    circuit_failure_threshold: int = Field(default=4, ge=1, le=50, validation_alias="CIRCUIT_FAILURE_THRESHOLD")
    circuit_cooldown_seconds: float = Field(default=20.0, ge=0.1, le=600.0, validation_alias="CIRCUIT_COOLDOWN_SECONDS")
    circuit_half_open_max_calls: int = Field(default=1, ge=1, le=5, validation_alias="CIRCUIT_HALF_OPEN_MAX_CALLS")
    circuit_half_open_success_threshold: int = Field(default=1, ge=1, le=5, validation_alias="CIRCUIT_HALF_OPEN_SUCCESS_THRESHOLD")

    # --------------------------------------------------------- idempotency
    idempotency_ttl_hours: int = Field(default=24, ge=0, le=24 * 30, validation_alias="IDEMPOTENCY_TTL_HOURS")
    idempotency_lease_seconds: float = Field(default=30.0, ge=1.0, le=600.0, validation_alias="IDEMPOTENCY_LEASE_SECONDS")

    # ---------------------------------------------------------- persistence
    taskpilot_db_path: Path = Field(default=Path("data/taskpilot.sqlite3"), validation_alias="TASKPILOT_DB_PATH")
    checkpoint_db_path: Path = Field(default=Path("data/checkpoints.sqlite3"), validation_alias="TASKPILOT_CHECKPOINT_PATH")
    upload_dir: Path = Field(default=Path("data/uploads"), validation_alias="TASKPILOT_UPLOAD_DIR")
    output_dir: Path = Field(default=Path("outputs"), validation_alias="TASKPILOT_OUTPUT_DIR")
    context_max_chars: int = Field(default=8000, ge=1000, le=30000, validation_alias="CONTEXT_MAX_CHARS")
    memory_ttl_days: int = Field(default=14, ge=1, le=365, validation_alias="MEMORY_TTL_DAYS")
    memory_max_items: int = Field(default=200, ge=10, le=5000, validation_alias="MEMORY_MAX_ITEMS")
    recover_on_startup: bool = Field(default=True, validation_alias="RECOVER_ON_STARTUP")
    stale_task_seconds: float = Field(default=90.0, ge=10.0, le=3600.0, validation_alias="STALE_TASK_SECONDS")

    # --------------------------------------------------------------- safety
    tool_concurrency: int = Field(default=4, ge=1, le=16, validation_alias="TOOL_CONCURRENCY")
    max_goal_chars: int = Field(default=4000, ge=10, validation_alias="MAX_GOAL_CHARS")
    rate_limit_per_minute: int = Field(default=60, ge=0, le=10000, validation_alias="RATE_LIMIT_PER_MINUTE")
    shutdown_grace_seconds: float = Field(default=10.0, ge=0.0, le=120.0, validation_alias="SHUTDOWN_GRACE_SECONDS")
    tavily_api_key: str = Field(default="", validation_alias="TAVILY_API_KEY")

    # ------------------------------------------------------- test seams only
    fault_injection: bool = Field(default=False, validation_alias="TASKPILOT_FAULT_INJECTION")

    # ------------------------------------------------------------------
    def budget_limits(self) -> BudgetLimits:
        return BudgetLimits(
            max_steps=self.react_max_steps,
            max_tool_calls=self.react_max_tool_calls,
            max_model_calls=self.react_max_model_calls,
            max_total_tokens=self.task_max_tokens,
            total_timeout_seconds=self.task_timeout_seconds,
        )

    def ensure_directories(self) -> None:
        for path in (
            self.taskpilot_db_path.parent,
            self.checkpoint_db_path.parent,
            self.upload_dir,
            self.output_dir,
            Path("data/knowledge_base"),
            Path("data/public"),
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that need a fresh Settings instance."""

    get_settings.cache_clear()