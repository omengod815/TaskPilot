"""Evaluate task routing and completion against tests/eval_tasks.jsonl."""
import json
import os
from pathlib import Path

import httpx

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("TASKPILOT_API_TOKEN", "")
HEADERS = {"X-API-Key": API_TOKEN} if API_TOKEN else {}
DATASET = Path("tests/eval_tasks.jsonl")


def main() -> None:
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    route_ok = 0
    task_ok = 0
    results = []
    with httpx.Client(timeout=180.0, headers=HEADERS) as client:
        for index, row in enumerate(rows, start=1):
            response = client.post(f"{API_BASE}/tasks", json={
                "goal": row["goal"],
                "user_id": "eval-user",
                "session_id": f"eval-{index}",
                "input_file": "sample_predictions.csv" if row["expected_task_type"] == "data_analysis" else None,
            })
            response.raise_for_status()
            result = response.json()
            route_correct = result.get("task_type") == row["expected_task_type"]
            success = result.get("status") in {"completed", "partial", "pending_approval"}
            route_ok += int(route_correct)
            task_ok += int(success)
            results.append({
                "index": index,
                "expected": row["expected_task_type"],
                "actual": result.get("task_type"),
                "status": result.get("status"),
                "route_correct": route_correct,
                "task_success": success,
            })
    report = {
        "dataset_size": len(rows),
        "route_accuracy": round(route_ok / len(rows), 4) if rows else 0,
        "task_success_rate": round(task_ok / len(rows), 4) if rows else 0,
        "note": "MOCK_LLM=1 结果只验证路由和工程主链路；真实模型质量评估需在 MOCK_LLM=0 下另跑并人工抽查。",
        "results": results,
    }
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/eval_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, ensure_ascii=False, indent=2))
    print("完整结果：outputs/eval_report.json")


if __name__ == "__main__":
    main()
