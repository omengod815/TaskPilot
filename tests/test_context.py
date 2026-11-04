from app.services.context_service import ContextService
from app.services.repository import TaskRepository


def test_context_is_scoped_to_user_and_session(tmp_path):
    repo = TaskRepository(tmp_path / "tasks.sqlite3")
    repo.save_state({
        "task_id": "old-1", "user_id": "alice", "session_id": "s1", "goal": "CSV 数据分析", "task_type": "data_analysis",
        "status": "completed", "answer": {"summary": "MAE 为 0.2"},
    })
    repo.save_state({
        "task_id": "old-2", "user_id": "bob", "session_id": "s1", "goal": "CSV 数据分析", "task_type": "data_analysis",
        "status": "completed", "answer": {"summary": "不应泄露的内容"},
    })
    context = ContextService(repo).build({
        "task_id": "new-1", "user_id": "alice", "session_id": "s1", "goal": "CSV 数据分析 MAE", "task_type": "data_analysis",
    })
    assert "MAE 为 0.2" in context["context_text"]
    assert "不应泄露" not in context["context_text"]
