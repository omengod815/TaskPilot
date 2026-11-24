"""Resume / cooperative cancel for free-form conversations."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.agents.conversation.models import ConversationState
from app.agents.conversation.runtime import ConversationRuntime
from app.core.budget import ExecutionBudget
from app.main import create_app
from app.services.conversation_store import ConversationStore


def make_state(settings, goal: str, *, max_turns: int, user_id: str = "resume-user") -> ConversationState:
    return ConversationState.create(
        goal=goal,
        user_id=user_id,
        session_id="resume-session",
        max_turns=max_turns,
        budget=ExecutionBudget.create(settings.budget_limits()).to_dict(),
    )


def test_partial_conversation_can_resume(components, settings):
    store = ConversationStore(settings.taskpilot_db_path)
    runtime = ConversationRuntime(settings, components.llm, components.registry, store)
    state = make_state(settings, "调研 Agent 记忆机制并写一段总结", max_turns=2)
    first = runtime.run(state)
    assert first.status == "partial"

    claimed = store.claim_for_resume(
        first.conversation_id,
        first.user_id,
        extra_turns=2,
    )
    assert claimed is not None
    assert claimed["status"] == "running"
    assert claimed["cancel_requested"] is False
    assert claimed["max_turns"] >= first.turn + 2

    resumed = runtime.run(ConversationState.from_dict(claimed))
    assert resumed.status == "completed"
    assert resumed.final_answer


def test_cancel_flag_stops_runtime_before_next_turn(components, settings):
    store = ConversationStore(settings.taskpilot_db_path)
    runtime = ConversationRuntime(settings, components.llm, components.registry, store)
    state = make_state(settings, "调研 Agent 记忆机制并写一段总结", max_turns=4)
    store.save(state.to_dict())

    cancelled = store.request_cancel(state.conversation_id, state.user_id)
    assert cancelled is not None
    assert cancelled["cancel_requested"] is True

    result = runtime.run(state)
    assert result.status == "cancelled"
    assert result.stop_reason == "cancelled_by_user"
    assert not [m for m in result.messages if m.get("role") == "agent"]


def test_cancel_endpoint_marks_running_conversation(settings):
    cfg = settings.model_copy(update={"mock_llm": True, "rate_limit_per_minute": 0, "api_access_token": ""})
    store = ConversationStore(cfg.taskpilot_db_path)
    state = make_state(cfg, "调研 Agent 记忆机制并写一段总结", max_turns=4, user_id="cancel-user")
    store.save(state.to_dict())

    with TestClient(create_app(cfg)) as client:
        response = client.post(
            f"/conversations/{state.conversation_id}/cancel",
            params={"user_id": "cancel-user"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["cancel_requested"] is True

        got = client.get(
            f"/conversations/{state.conversation_id}",
            params={"user_id": "cancel-user"},
        )
        assert got.status_code == 200
        assert got.json()["cancel_requested"] is True

        wrong_user = client.post(
            f"/conversations/{state.conversation_id}/cancel",
            params={"user_id": "other"},
        )
        assert wrong_user.status_code == 404


def test_resume_api_and_completed_rejection(settings):
    cfg = settings.model_copy(update={"mock_llm": True, "rate_limit_per_minute": 0, "api_access_token": ""})
    app = create_app(cfg)
    with TestClient(app) as client:
        created = client.post(
            "/conversations",
            json={
                "goal": "调研 Agent 记忆机制并写一段总结",
                "user_id": "resume-api",
                "max_turns": 2,
            },
        )
        assert created.status_code == 200, created.text
        body = created.json()
        assert body["status"] == "partial"

        resumed = client.post(
            f"/conversations/{body['conversation_id']}/resume",
            json={"user_id": "resume-api", "extra_turns": 2},
        )
        assert resumed.status_code == 200, resumed.text
        assert resumed.json()["status"] == "completed"

        again = client.post(
            f"/conversations/{body['conversation_id']}/resume",
            json={"user_id": "resume-api", "extra_turns": 2},
        )
        assert again.status_code == 409

        cancel = client.post(
            f"/conversations/{body['conversation_id']}/cancel",
            params={"user_id": "resume-api"},
        )
        assert cancel.status_code == 409