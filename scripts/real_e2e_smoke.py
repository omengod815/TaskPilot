"""One real-model end-to-end smoke test (uses .env, never prints the key)."""
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
            "/tasks",
            json={
                "goal": "分析 sample_predictions.csv 的 MAE、RMSE 和 R2，并说明局限。",
                # A fresh user/session avoids short-term memory from a previous
                # smoke answering the task without calling the tool.
                "user_id": f"real-smoke-{run_id}",
                "session_id": f"real-smoke-{run_id}",
                "task_type": "data_analysis",
                "input_file": "sample_predictions.csv",
            },
        )
        print("HTTP", response.status_code)
        body = response.json()
        if response.status_code != 200:
            print(json.dumps(body, ensure_ascii=False, indent=2)[:2000])
            raise SystemExit(1)
        trace = body.get("trace", [])
        trace_nodes = [item.get("node") for item in trace]
        summary = {
            "run_id": run_id,
            "task_id": body.get("task_id"),
            "status": body.get("status"),
            "task_type": body.get("task_type"),
            "stop_reason": body.get("stop_reason"),
            "trace_nodes": trace_nodes,
            "tool_called": "tool_execution" in trace_nodes,
            "actual_tools": [o.get("tool") for o in body.get("observations", []) if o.get("tool")],
            "observation_count": len(body.get("observations", [])),
            "budget": body.get("budget", {}),
            "answer_keys": sorted((body.get("answer") or {}).keys()),
            "answer_summary": (body.get("answer") or {}).get("summary"),
            "answer_metrics": (body.get("answer") or {}).get("metrics"),
            "limitations": (body.get("answer") or {}).get("limitations", []),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        Path("outputs").mkdir(exist_ok=True)
        Path("outputs/real_e2e_smoke.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not summary["tool_called"]:
            print("FAIL: the model completed without calling the expected data tool.")
            raise SystemExit(3)
        if summary["status"] not in {"completed", "partial"}:
            raise SystemExit(2)


if __name__ == "__main__":
    main()