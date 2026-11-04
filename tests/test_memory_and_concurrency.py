"""Memory separation and concurrent idempotent tool execution."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from app.core.budget import ExecutionBudget
from app.services.context_service import ContextService
from app.services.idempotency import build_idempotency_key, params_hash
from app.services.memory import MemoryStore
from app.tools import toolkit
from app.tools.registry import ToolContext


def _ctx(settings, task_id, **extra):
    return ToolContext(
        task_id=task_id,
        user_id="u",
        settings=settings,
        budget=ExecutionBudget.create(settings.budget_limits()),
        owner=task_id,
        **extra,
    )


def test_long_term_and_short_term_are_separated(settings):
    memory = MemoryStore(settings.taskpilot_db_path)
    memory.remember(user_id="u1", session_id="s1", content="用户偏好简洁的中文报告", kind="long_term", importance=0.9)
    memory.remember(user_id="u1", session_id="s1", content="上次分析了 CSV 的 MAE", kind="short_term", importance=0.5)

    long_term = memory.recall(user_id="u1", query="报告风格", kinds=("long_term",), limit=5)
    short_term = memory.recall(user_id="u1", query="CSV MAE", kinds=("short_term",), limit=5)
    assert all(item["kind"] == "long_term" for item in long_term)
    assert any("报告" in item["content"] for item in long_term)
    assert all(item["kind"] == "short_term" for item in short_term)
    assert any("MAE" in item["content"] for item in short_term)


def test_memory_write_is_deduplicated(settings):
    memory = MemoryStore(settings.taskpilot_db_path)
    memory.remember(user_id="u2", session_id="s", content="同一条记忆内容", kind="short_term")
    memory.remember(user_id="u2", session_id="s", content="同一条记忆内容", kind="short_term")
    assert memory.stats()["by_kind"]["short_term"] == 1


def test_expired_memory_is_not_recalled(settings):
    memory = MemoryStore(settings.taskpilot_db_path)
    memory.remember(user_id="u3", session_id="s", content="过期记忆 CSV", kind="short_term")
    with sqlite3.connect(settings.taskpilot_db_path) as conn:
        conn.execute("UPDATE memories SET expires_at=? WHERE user_id='u3'", (time.time() - 10,))
    assert memory.recall(user_id="u3", query="CSV") == []


def test_memory_is_scoped_per_user(settings):
    memory = MemoryStore(settings.taskpilot_db_path)
    memory.remember(user_id="alice", session_id="s", content="alice 的私有偏好", kind="long_term")
    memory.remember(user_id="bob", session_id="s", content="bob 的私有偏好", kind="long_term")
    items = memory.recall(user_id="alice", query="偏好", limit=5)
    assert items and all("alice" in item["content"] for item in items)


def test_context_does_not_treat_pending_task_as_memory(settings):
    from app.services.repository import TaskRepository

    repo = TaskRepository(settings.taskpilot_db_path)
    repo.save_state({
        "task_id": "pending-1", "user_id": "ctx-user", "session_id": "s", "goal": "CSV 分析 MAE",
        "task_type": "data_analysis", "status": "pending_approval",
        "answer": {"summary": "等待审批的特殊内容"},
    })
    context = ContextService(repo, memory=MemoryStore(settings.taskpilot_db_path)).build({
        "task_id": "ctx-2", "user_id": "ctx-user", "session_id": "s", "goal": "CSV 分析 MAE", "task_type": "data_analysis",
    })
    assert "等待审批的特殊内容" not in context["context_text"]


def test_concurrent_tool_calls_create_exactly_one_side_effect(settings, components):
    task_id = "concurrent-1"
    goal = "给客户发送延期邮件"
    barrier = threading.Barrier(6)
    results = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        result = components.registry.execute(
            "execute_demo_action",
            {"goal": goal},
            task_type="external_action",
            ctx=_ctx(settings, task_id, approved=True),
        )
        with lock:
            results.append(result)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    files = list(Path(settings.output_dir).glob("action_plan_*.json"))
    assert len(files) == 1, "more than one effect was produced"
    executed = [r for r in results if r.ok and not r.replayed]
    replayed = [r for r in results if r.ok and r.replayed]
    assert len(executed) == 1, "exactly one caller should execute the effect"
    assert len(executed) + len(replayed) + len([r for r in results if not r.ok]) == len(results)
    others = [r for r in results if not r.ok]
    assert all(r.error_code in {"IDEMPOTENCY_IN_PROGRESS", "IDEMPOTENCY_CONFLICT"} for r in others)


def test_unknown_state_is_reconciled_before_replay(settings, components):
    task_id = "reconcile-1"
    goal = "给客户发送延期邮件"
    key = build_idempotency_key(task_id, "execute_demo_action", {"goal": goal})

    # The effect exists because a previous attempt applied it, but the record
    # is stuck IN_PROGRESS with an expired lease (owner crashed).
    toolkit.execute_demo_action(goal, idempotency_key=key, output_dir=settings.output_dir)
    first = components.idempotency.claim(
        key,
        tool="execute_demo_action",
        args_hash=params_hash({"goal": goal}),
        task_id=task_id,
        lease_seconds=0.01,
    )
    time.sleep(0.05)

    result = components.registry.execute(
        "execute_demo_action",
        {"goal": goal},
        task_type="external_action",
        ctx=_ctx(settings, task_id, approved=True, idempotency_key=key),
    )

    assert result.ok is True
    assert result.replayed is True
    assert len(list(Path(settings.output_dir).glob("action_plan_*.json"))) == 1
    record = components.idempotency.get(key)
    assert record.status.value == "SUCCEEDED"
    assert record.owner != first.owner or record.attempts >= 1