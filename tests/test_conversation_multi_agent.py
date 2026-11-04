"""Free-form multi-agent conversation tests (no real model)."""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.agents.conversation.agents import PROFILES, WorkerAgent
from app.agents.conversation.models import AgentAction, ConversationState
from app.agents.conversation.runtime import ConversationRuntime
from app.core.budget import ExecutionBudget
from app.main import create_app
from app.services.conversation_store import ConversationStore


def make_runtime(components, settings):
    store = ConversationStore(settings.taskpilot_db_path)
    return ConversationRuntime(settings, components.llm, components.registry, store), store


def make_state(settings, goal: str, *, max_turns: int = 6, user_id: str = "conv-user") -> ConversationState:
    return ConversationState.create(
        goal=goal,
        user_id=user_id,
        session_id="conv-session",
        max_turns=max_turns,
        budget=ExecutionBudget.create(settings.budget_limits()).to_dict(),
    )


def test_data_agent_then_writer_finishes(components, settings):
    runtime, _ = make_runtime(components, settings)
    result = runtime.run(make_state(settings, "分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并说明局限"))

    assert result.status == "completed"
    assert result.stop_reason == "finished"
    agents = [m["agent"] for m in result.messages if m.get("role") == "agent"]
    assert agents == ["data", "writing"]
    assert any(o.get("tool") == "analyze_csv" and o.get("ok") for o in result.observations)
    assert "MAE" in (result.final_answer or "")
    assert result.handoffs[0]["to"] == "data"
    assert any(h["to"] == "finish" for h in result.handoffs)


def test_research_then_writer_hands_off(components, settings):
    runtime, _ = make_runtime(components, settings)
    result = runtime.run(make_state(settings, "调研 Agent 记忆机制并写一段总结"))

    assert result.status == "completed"
    agents = [m["agent"] for m in result.messages if m.get("role") == "agent"]
    assert "research" in agents
    assert "writing" in agents
    assert any(o.get("tool") == "search_web" for o in result.observations)
    assert result.final_answer


def test_max_turns_stops_conversation(components, settings):
    runtime, _ = make_runtime(components, settings)
    result = runtime.run(make_state(settings, "分析 sample_predictions.csv 的 MAE", max_turns=1))

    assert result.status == "partial"
    assert result.stop_reason == "max_turns_exceeded"
    agents = [m["agent"] for m in result.messages if m.get("role") == "agent"]
    assert agents == ["data"]


class _FakeCall:
    def __init__(self, action: AgentAction) -> None:
        self.value = action
        self.meta = SimpleNamespace(to_dict=lambda: {"model": "fake"})


class _FakeLLM:
    def generate_structured_with_meta(self, *args, **kwargs):
        return _FakeCall(AgentAction(action="tool", tool="execute_demo_action", arguments={"goal": "x"}))


def test_worker_rejects_tool_outside_its_profile(components, settings):
    real_settings = settings.model_copy(update={"mock_llm": False})
    worker = WorkerAgent(PROFILES["data"], _FakeLLM(), components.registry, real_settings)
    state = make_state(settings, "分析 sample_predictions.csv")

    result = worker.run(state, "读取 CSV")

    assert "不在" in result["reply"]
    assert result["observations"][0]["error_code"] == "TOOL_NOT_ALLOWED"
    assert not any(o.get("tool") == "execute_demo_action" for o in state.observations)


def test_conversation_api_create_get_list(settings):
    cfg = settings.model_copy(update={"mock_llm": True, "rate_limit_per_minute": 0, "api_access_token": ""})
    app = create_app(cfg)
    with TestClient(app) as client:
        created = client.post(
            "/conversations",
            json={"goal": "分析 sample_predictions.csv 的 MAE", "user_id": "api-conv", "max_turns": 3},
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "completed"
        assert body["agents"]
        assert body["messages"]
        assert body["final_answer"]

        got = client.get(f"/conversations/{body['conversation_id']}", params={"user_id": "api-conv"})
        assert got.status_code == 200
        assert got.json()["conversation_id"] == body["conversation_id"]

        wrong_user = client.get(f"/conversations/{body['conversation_id']}", params={"user_id": "other"})
        assert wrong_user.status_code == 404

        listed = client.get("/conversations", params={"user_id": "api-conv"})
        assert listed.status_code == 200
        assert any(item["conversation_id"] == body["conversation_id"] for item in listed.json()["items"])