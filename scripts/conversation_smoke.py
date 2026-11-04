"""Real-model smoke test for the free-form multi-agent conversation runtime."""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from app.main import create_app


def main() -> None:
    app = create_app()
    run_id = uuid.uuid4().hex[:10]
    with TestClient(app) as client:
        response = client.post(
            "/conversations",
            json={
                "goal": "分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并写一段面向项目负责人的结论。",
                "user_id": f"conv-smoke-{run_id}",
                "session_id": f"conv-smoke-{run_id}",
                "max_turns": 4,
            },
        )
        print("HTTP", response.status_code)
        body = response.json()
        if response.status_code != 200:
            print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])
            raise SystemExit(1)

        agents = [m.get("agent") for m in body.get("messages", []) if m.get("role") == "agent"]
        summary = {
            "run_id": run_id,
            "conversation_id": body.get("conversation_id"),
            "status": body.get("status"),
            "stop_reason": body.get("stop_reason"),
            "agents": agents,
            "handoffs": body.get("handoffs", []),
            "observations": [
                {
                    "agent": o.get("agent"),
                    "tool": o.get("tool"),
                    "ok": o.get("ok"),
                    "error_code": o.get("error_code"),
                }
                for o in body.get("observations", [])
            ],
            "turn": body.get("turn"),
            "budget": body.get("budget", {}),
            "final_answer": body.get("final_answer"),
            "degraded_reasons": body.get("degraded_reasons", []),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/conversation_real_smoke.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if body.get("status") not in {"completed", "partial"}:
            raise SystemExit(2)


if __name__ == "__main__":
    main()