"""Real-model multi-turn conversation smoke test.

Creates a conversation, asks a follow-up through the same conversation, and
checks that the second answer is produced from the extended transcript.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402


def main() -> None:
    cfg = get_settings().model_copy(
        update={
            "mock_llm": False,
            "rate_limit_per_minute": 0,
            "api_access_token": "",
            "taskpilot_db_path": PROJECT_ROOT / "data" / "conversation_multiturn_smoke.sqlite3",
            "checkpoint_db_path": PROJECT_ROOT / "data" / "conversation_multiturn_smoke_checkpoints.sqlite3",
        }
    )
    if not cfg.qwen_api_key:
        raise SystemExit("QWEN_API_KEY 未配置，不能运行真实模型 smoke。")

    run_id = uuid.uuid4().hex[:10]
    user_id = f"multiturn-{run_id}"
    app = create_app(cfg)
    with TestClient(app) as client:
        created = client.post(
            "/conversations",
            json={
                "goal": "分析 sample_predictions.csv 的 MAE、RMSE 和 R2",
                "user_id": user_id,
                "session_id": f"multiturn-{run_id}",
                "max_turns": 3,
            },
        )
        print("CREATE_HTTP", created.status_code)
        first = created.json()
        if created.status_code != 200 or first.get("status") != "completed":
            print(json.dumps(first, ensure_ascii=False, indent=2)[:2000])
            raise SystemExit(2)

        follow = client.post(
            f"/conversations/{first['conversation_id']}/messages",
            json={
                "content": "请用一句话总结上面的指标，并说明样本量局限。",
                "user_id": user_id,
                "extra_turns": 2,
            },
        )
        print("FOLLOW_HTTP", follow.status_code)
        second = follow.json()
        if follow.status_code != 200 or second.get("status") != "completed":
            print(json.dumps(second, ensure_ascii=False, indent=2)[:2000])
            raise SystemExit(3)

        user_messages = [m for m in second.get("messages", []) if m.get("role") == "user"]
        agents = [m.get("agent") for m in second.get("messages", []) if m.get("role") == "agent"]
        final_answer = str(second.get("final_answer") or "")
        if len(user_messages) != 2:
            raise SystemExit(f"expected 2 user messages, got {len(user_messages)}")
        if "MAE" not in final_answer:
            raise SystemExit("follow-up final answer does not mention MAE")

        summary = {
            "run_id": run_id,
            "conversation_id": second["conversation_id"],
            "first_status": first.get("status"),
            "follow_status": second.get("status"),
            "first_final_answer": first.get("final_answer"),
            "follow_final_answer": final_answer,
            "agents": agents,
            "user_message_count": len(user_messages),
            "turn": second.get("turn"),
            "max_turns": second.get("max_turns"),
            "budget": second.get("budget"),
            "handoffs": second.get("handoffs"),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        outputs = PROJECT_ROOT / "outputs"
        outputs.mkdir(exist_ok=True)
        (outputs / "conversation_multiturn_real_smoke.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (outputs / "conversation_multiturn_real_smoke.txt").write_text(
            f"first_status={first.get('status')}\n"
            f"follow_status={second.get('status')}\n"
            f"user_messages={len(user_messages)}\n"
            f"agents={agents}\n"
            f"turn={second.get('turn')} max_turns={second.get('max_turns')}\n"
            f"model_calls={(second.get('budget') or {}).get('model_calls')}\n"
            f"tokens={(second.get('budget') or {}).get('tokens_used')}\n"
            f"first_answer={first.get('final_answer')}\n"
            f"follow_answer={final_answer}\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()