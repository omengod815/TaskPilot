"""Subprocess used by the crash-recovery test.

It starts a real graph run in a separate process and hard-exits at the
``after_tool_success`` seam, leaving a checkpoint that is one super-step behind
the applied side effect.  The parent process then resumes from that checkpoint.

Environment (set by the test):
    TASKPILOT_TASK_ID, TASKPILOT_DB_PATH, TASKPILOT_CHECKPOINT_PATH,
    TASKPILOT_OUTPUT_DIR, TASKPILOT_UPLOAD_DIR, MOCK_LLM=1,
    TASKPILOT_FAULT_INJECTION=1, TASKPILOT_CRASH_AT=after_tool_success
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.agent_graph import build_agent_graph  # noqa: E402
from app.core.budget import ExecutionBudget  # noqa: E402
from app.core.config import Settings  # noqa: E402
from app.services.checkpoint import CheckpointManager, thread_config  # noqa: E402
from app.services.idempotency import IdempotencyStore  # noqa: E402
from app.services.llm_service import AdaptiveLLMService  # noqa: E402
from app.services.memory import MemoryStore  # noqa: E402
from app.services.repository import TaskRepository  # noqa: E402
from app.tools.registry import build_default_registry  # noqa: E402


def main() -> None:
    settings = Settings()
    settings.ensure_directories()
    task_id = os.environ["TASKPILOT_TASK_ID"]
    repository = TaskRepository(settings.taskpilot_db_path)
    idempotency = IdempotencyStore(settings.taskpilot_db_path)
    memory = MemoryStore(settings.taskpilot_db_path)
    registry, _ = build_default_registry(settings, idempotency)
    checkpoints = CheckpointManager(settings.checkpoint_db_path)
    graph = build_agent_graph(
        repository=repository,
        llm=AdaptiveLLMService(settings=settings),
        settings=settings,
        registry=registry,
        memory=memory,
        checkpointer=checkpoints.saver,
    )
    state = {
        "task_id": task_id,
        "user_id": "crash-user",
        "session_id": "crash-session",
        "goal": os.environ.get("TASKPILOT_GOAL", "请发送邮件给客户，说明项目延期"),
        "task_type": "external_action",
        "approved": True,  # approval already granted, so the tool runs and then we crash
        "status": "running",
        "observations": [],
        "react_step": 0,
        "budget": ExecutionBudget.create(settings.budget_limits()).to_dict(),
        "degraded_reasons": [],
        "trace": [],
        "metrics": {},
    }
    repository.save_state(state)
    graph.invoke(state, thread_config(task_id))
    # If the crash seam did not fire, exit non-zero so the test notices.
    sys.exit(3)


if __name__ == "__main__":
    main()