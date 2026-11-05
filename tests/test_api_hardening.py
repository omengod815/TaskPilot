"""API hardening: auth, validation, idempotent submission, rate limiting, resume."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


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


@pytest.fixture
def auth():
    return {"X-API-Key": "secret-token"}


def test_health_is_public_but_tasks_require_auth(api):
    client, _, _ = api
    assert client.get("/health").status_code == 200
    unauth = client.get("/tasks")
    assert unauth.status_code == 401
    body = unauth.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["retryable"] is False


def test_validation_error_contract(api, auth):
    client, _, _ = api
    response = client.post("/tasks", json={"goal": "x"}, headers=auth)
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "VALIDATION_ERROR"
    assert error["details"]["errors"]


def test_request_id_is_echoed(api, auth):
    client, _, _ = api
    response = client.get("/health", headers={"X-Request-ID": "req-abc-123"})
    assert response.headers["X-Request-ID"] == "req-abc-123"


def test_task_creation_is_idempotent_by_header(api, auth):
    client, app, _ = api
    payload = {"goal": "分析 sample_predictions.csv 的 MAE", "task_type": "data_analysis", "input_file": "sample_predictions.csv"}
    headers = {**auth, "Idempotency-Key": "client-key-1"}
    first = client.post("/tasks", json=payload, headers=headers)
    second = client.post("/tasks", json=payload, headers=headers)
    assert first.status_code == 200 and second.status_code == 200
    assert first.json()["task_id"] == second.json()["task_id"]

    ctx = app.state.get_ctx()
    assert len([t for t in ctx.repository.list_tasks(limit=50) if t["task_id"] == first.json()["task_id"]]) == 1


def test_approval_reject_then_conflict(api, auth):
    client, app, _ = api
    created = client.post(
        "/tasks",
        json={"goal": "请发送邮件给客户，说明项目延期", "user_id": "approval-user"},
        headers=auth,
    )
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    assert created.json()["status"] == "pending_approval"

    wrong_user = client.post(f"/tasks/{task_id}/approve", json={"user_id": "someone-else", "approved": True}, headers=auth)
    assert wrong_user.status_code == 404

    rejected = client.post(f"/tasks/{task_id}/approve", json={"user_id": "approval-user", "approved": False, "reason": "test"}, headers=auth)
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "cancelled"

    again = client.post(f"/tasks/{task_id}/approve", json={"user_id": "approval-user", "approved": True}, headers=auth)
    assert again.status_code == 409
    assert again.json()["error"]["code"] == "CONFLICT"


def test_approval_executes_external_action_once(api, auth):
    client, app, _ = api
    created = client.post(
        "/tasks",
        json={"goal": "请发送邮件给客户，说明项目延期", "user_id": "approval-user-2"},
        headers=auth,
    )
    task_id = created.json()["task_id"]
    approved = client.post(f"/tasks/{task_id}/approve", json={"user_id": "approval-user-2", "approved": True}, headers=auth)
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["status"] == "completed"
    assert body["observations"][0]["replayed"] is False

    # Re-sending the approval is rejected by the state machine, not re-executed.
    again = client.post(f"/tasks/{task_id}/approve", json={"user_id": "approval-user-2", "approved": True}, headers=auth)
    assert again.status_code == 409
    files = list((app.state.get_ctx().settings.output_dir).glob("action_plan_*.json"))
    assert len(files) == 1


def test_rate_limit_returns_429_with_retry_after(settings):
    cfg = settings.model_copy(update={"api_access_token": "secret-token", "rate_limit_per_minute": 2, "mock_llm": True})
    app = create_app(cfg)
    headers = {"X-API-Key": "secret-token"}
    with TestClient(app) as client:
        assert client.get("/tasks", headers=headers).status_code == 200
        assert client.get("/tasks", headers=headers).status_code == 200
        limited = client.get("/tasks", headers=headers)
        assert limited.status_code == 429
        assert limited.headers.get("Retry-After")
        assert limited.json()["error"]["code"] == "RATE_LIMITED"


def test_ready_and_metrics(api, auth):
    client, _, _ = api
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"]["tool_count"] >= 4
    metrics = client.get("/metrics", headers=auth)
    assert metrics.status_code == 200
    body = metrics.json()
    assert "llm" in body and "circuit_open_count" in body["llm"]
    assert body["service"]["version"].startswith("3.3")


def test_resume_interrupted_task(api, auth):
    client, app, _ = api
    created = client.post("/tasks", json={"goal": "计算 1 + 1", "user_id": "resume-user"}, headers=auth)
    task_id = created.json()["task_id"]
    assert created.json()["status"] == "completed"

    ctx = app.state.get_ctx()
    state = ctx.repository.get_state(task_id)
    state["status"] = "interrupted"
    ctx.repository.save_state(state)

    resumed = client.post(f"/tasks/{task_id}/resume", params={"user_id": "resume-user"}, headers=auth)
    assert resumed.status_code == 200, resumed.text
    assert resumed.json()["status"] == "completed"

    conflict = client.post(f"/tasks/{task_id}/resume", params={"user_id": "resume-user"}, headers=auth)
    assert conflict.status_code == 409


def test_transition_status_is_atomic(api, auth):
    """Two concurrent approvals: exactly one may win the state transition."""

    import threading

    client, app, _ = api
    created = client.post("/tasks", json={"goal": "请发送邮件给客户，说明项目延期", "user_id": "race-user"}, headers=auth)
    task_id = created.json()["task_id"]
    ctx = app.state.get_ctx()

    barrier = threading.Barrier(2)
    outcomes: list[bool] = []

    def worker():
        barrier.wait()
        outcomes.append(
            ctx.repository.transition_status(task_id, "pending_approval", "running", patch={"approved": True})
        )

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(outcomes) == [False, True]


def test_trace_contains_react_steps(api, auth):
    client, _, _ = api
    created = client.post("/tasks", json={"goal": "分析 sample_predictions.csv 的 RMSE", "task_type": "data_analysis"}, headers=auth)
    task_id = created.json()["task_id"]
    trace = client.get(f"/tasks/{task_id}/trace", headers=auth)
    assert trace.status_code == 200
    nodes = [item["node"] for item in trace.json()["trace"]]
    assert nodes.count("react_decide") >= 2
    assert "tool_execution" in nodes
    assert trace.json()["budget"]["limits"]["max_steps"] >= 1