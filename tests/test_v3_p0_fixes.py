"""Regression tests for the v3 P0 audit fixes.

Covers:
* P0-01 per-user Idempotency-Key scope;
* P0-02 atomic resume claim;
* P0-03 mark_succeeded failure must not report success;
* P0-04 GET /tasks user scoping;
* legacy task_requests migration.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.budget import ExecutionBudget
from app.main import create_app
from app.services.repository import TaskRepository
from app.tools.registry import ToolContext


@pytest.fixture
def api(settings):
    cfg = settings.model_copy(
        update={
            "api_access_token": "secret-token",
            "rate_limit_per_minute": 0,
            "mock_llm": True,
        }
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, app, cfg


def _headers(key: str | None = None) -> dict[str, str]:
    headers = {"X-API-Key": "secret-token"}
    if key:
        headers["Idempotency-Key"] = key
    return headers


def test_same_idempotency_key_is_scoped_per_user(api):
    client, _, _ = api
    alice = client.post(
        "/tasks",
        json={"goal": "计算 1 + 1", "user_id": "alice"},
        headers=_headers("shared-key"),
    )
    assert alice.status_code == 200, alice.text

    bob = client.post(
        "/tasks",
        json={"goal": "解释什么是幂等性", "user_id": "bob"},
        headers=_headers("shared-key"),
    )
    assert bob.status_code == 200, bob.text
    assert alice.json()["task_id"] != bob.json()["task_id"]
    assert bob.json()["goal"] == "解释什么是幂等性"

    # Same user + same key still replays the original task.
    alice_again = client.post(
        "/tasks",
        json={"goal": "计算 1 + 1", "user_id": "alice"},
        headers=_headers("shared-key"),
    )
    assert alice_again.json()["task_id"] == alice.json()["task_id"]


def test_list_tasks_is_scoped_per_user(api):
    client, _, _ = api
    alice = client.post(
        "/tasks",
        json={"goal": "计算 1 + 1", "user_id": "alice"},
        headers=_headers("alice-list-1"),
    )
    bob = client.post(
        "/tasks",
        json={"goal": "计算 2 + 2", "user_id": "bob"},
        headers=_headers("bob-list-1"),
    )
    assert alice.status_code == 200 and bob.status_code == 200

    bob_items = client.get("/tasks", params={"user_id": "bob"}, headers=_headers()).json()["items"]
    assert bob_items
    assert {row["user_id"] for row in bob_items} == {"bob"}
    assert {row["task_id"] for row in bob_items} == {bob.json()["task_id"]}

    alice_items = client.get("/tasks", params={"user_id": "alice"}, headers=_headers()).json()["items"]
    assert {row["user_id"] for row in alice_items} == {"alice"}


def test_concurrent_resume_claim_has_one_winner(components, state_factory):
    task_id = "resume-race-1"
    state = state_factory(task_id, goal="计算 1 + 1", task_type="general")
    state["status"] = "interrupted"
    components.repository.save_state(state)

    barrier = threading.Barrier(2)
    outcomes: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        won = components.repository.claim_for_resume(task_id, {"interrupted", "failed"})
        with lock:
            outcomes.append(won)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == [False, True]
    stored = components.repository.get_state(task_id)
    assert stored["status"] == "running"


def test_claim_for_resume_rejects_completed(components, state_factory):
    task_id = "resume-completed-1"
    state = state_factory(task_id, goal="计算 1 + 1", task_type="general")
    state["status"] = "completed"
    components.repository.save_state(state)
    assert components.repository.claim_for_resume(task_id, {"interrupted", "running", "failed"}) is False


def test_mark_succeeded_failure_does_not_report_success(components, settings, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("sqlite write failed")

    monkeypatch.setattr(components.idempotency, "mark_succeeded", boom)

    ctx = ToolContext(
        task_id="idem-record-failure-1",
        user_id="u",
        settings=settings,
        budget=ExecutionBudget.create(settings.budget_limits()),
        owner="owner-1",
        approved=True,
    )
    result = components.registry.execute(
        "execute_demo_action",
        {"goal": "给客户发送延期邮件"},
        task_type="external_action",
        ctx=ctx,
    )

    assert result.ok is False
    assert result.error_code == "IDEMPOTENCY_RECORD_FAILED"
    assert result.data["external_effect_applied"] is True
    assert len(list(Path(settings.output_dir).glob("action_plan_*.json"))) == 1


def test_legacy_task_requests_table_is_migrated(tmp_path):
    db = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute(
            """
            CREATE TABLE task_requests (
                request_key TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT INTO task_requests(request_key, task_id, created_at) VALUES(?, ?, ?)",
            ("legacy-key", "legacy-task", "2026-01-01T00:00:00+00:00"),
        )

    repo = TaskRepository(db)
    assert repo.find_by_request_key("legacy-key", user_id="local-demo") == "legacy-task"
    assert repo.register_request_key("legacy-key", "bob-task", user_id="bob") is True
    assert repo.find_by_request_key("legacy-key", user_id="bob") == "bob-task"
    # The old global key no longer blocks a different user.
    assert repo.find_by_request_key("legacy-key", user_id="alice") is None