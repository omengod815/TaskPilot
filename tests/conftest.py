import os
import sys
from pathlib import Path

# Unit tests are deterministic and must never call a paid external model API.
os.environ.setdefault("MOCK_LLM", "1")
os.environ.setdefault("TASKPILOT_LOG_LEVEL", "WARNING")
os.environ.setdefault("TASKPILOT_LOG_JSON", "0")
os.environ.setdefault("TASKPILOT_FAULT_INJECTION", "0")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402

from app.core.budget import ExecutionBudget  # noqa: E402
from app.core.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path):
    return Settings(
        _env_file=None,
        mock_llm=True,
        taskpilot_db_path=tmp_path / "tasks.sqlite3",
        checkpoint_db_path=tmp_path / "checkpoints.sqlite3",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        rate_limit_per_minute=0,
        log_json=False,
        task_timeout_seconds=60,
        react_max_steps=5,
        react_max_tool_calls=5,
    )


@pytest.fixture
def components(settings):
    """Build the real object graph against a temporary SQLite database."""

    from app.services.checkpoint import CheckpointManager
    from app.services.idempotency import IdempotencyStore
    from app.services.llm_service import AdaptiveLLMService
    from app.services.memory import MemoryStore
    from app.services.repository import TaskRepository
    from app.tools.registry import build_default_registry

    settings.ensure_directories()
    repository = TaskRepository(settings.taskpilot_db_path)
    idempotency = IdempotencyStore(settings.taskpilot_db_path, ttl_hours=settings.idempotency_ttl_hours)
    memory = MemoryStore(settings.taskpilot_db_path, max_items=settings.memory_max_items, ttl_days=settings.memory_ttl_days)
    registry, breakers = build_default_registry(settings, idempotency)
    llm = AdaptiveLLMService(settings=settings)
    checkpoints = CheckpointManager(settings.checkpoint_db_path)

    class Bundle:
        pass

    bundle = Bundle()
    bundle.settings = settings
    bundle.repository = repository
    bundle.idempotency = idempotency
    bundle.memory = memory
    bundle.registry = registry
    bundle.breakers = breakers
    bundle.llm = llm
    bundle.checkpoints = checkpoints
    yield bundle
    checkpoints.close()


def make_graph(components, decider=None, checkpointer=None):
    from app.agents.agent_graph import build_agent_graph

    return build_agent_graph(
        repository=components.repository,
        llm=components.llm,
        settings=components.settings,
        registry=components.registry,
        memory=components.memory,
        decider=decider,
        checkpointer=checkpointer if checkpointer is not None else components.checkpoints.saver,
    )


def initial_state(settings, task_id, *, goal="计算 (2 + 3) * 4", task_type="general", **extra):
    from app.core.budget import ExecutionBudget

    state = {
        "task_id": task_id,
        "user_id": "test-user",
        "session_id": "test-session",
        "goal": goal,
        "task_type": task_type,
        "status": "running",
        "observations": [],
        "react_step": 0,
        "budget": ExecutionBudget.create(settings.budget_limits()).to_dict(),
        "degraded_reasons": [],
        "trace": [],
        "metrics": {},
    }
    state.update(extra)
    return state


@pytest.fixture
def state_factory(settings):
    def factory(task_id, **kwargs):
        return initial_state(settings, task_id, **kwargs)

    return factory


@pytest.fixture
def graph_factory(components):
    def factory(decider=None, checkpointer=None):
        return make_graph(components, decider=decider, checkpointer=checkpointer)

    return factory


__all__ = ["ExecutionBudget", "Settings", "make_graph", "initial_state"]