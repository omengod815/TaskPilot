from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_data_analysis_task_runs_in_mock_mode():
    response = client.post("/tasks", json={
        "goal": "分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并说明结果局限。",
        "user_id": "api-test-user",
        "session_id": "api-test-session",
        "task_type": "data_analysis",
        "input_file": "sample_predictions.csv",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] in {"completed", "partial"}
    assert body["answer"] is not None
    assert any(item["node"] == "tool_execution" for item in body["trace"])


def test_external_action_requires_approval_and_can_be_rejected():
    response = client.post("/tasks", json={
        "goal": "请发送邮件给客户，说明项目延期。",
        "user_id": "approval-test-user",
        "session_id": "approval-test-session",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "pending_approval"
    assert body["answer"]["external_side_effect_performed"] is False

    rejected = client.post(f"/tasks/{body['task_id']}/approve", json={
        "user_id": "approval-test-user", "approved": False, "reason": "演示拒绝路径",
    })
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["status"] == "cancelled"
    assert rejected.json()["answer"]["external_side_effect_performed"] is False
