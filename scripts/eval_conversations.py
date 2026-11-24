"""Run the fixed multi-agent conversation evaluation suite.

Default mode is deterministic (MOCK_LLM=1).  Pass ``--real`` to use the
configured Qwen model; the suite is intentionally small so a real run is
affordable, but it is not a statistically meaningful benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.main import create_app  # noqa: E402

SUITE_PATH = PROJECT_ROOT / "evals" / "conversation_suite.json"


def load_suite(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def run_suite(*, real: bool, limit: int | None = None, task_id: str | None = None) -> dict:
    suite = load_suite(SUITE_PATH)
    tasks = suite["tasks"][:limit] if limit else suite["tasks"]
    if task_id:
        tasks = [task for task in tasks if task.get("id") == task_id]
        if not tasks:
            raise SystemExit(f"unknown eval task: {task_id}")
    cfg = get_settings().model_copy(
        update={
            "mock_llm": not real,
            "rate_limit_per_minute": 0,
            "api_access_token": "",
            "react_max_model_calls": 12,
            "react_max_steps": 8,
            "react_max_tool_calls": 8,
            "task_timeout_seconds": 180.0,
            "taskpilot_db_path": PROJECT_ROOT / "data" / "conversation_eval.sqlite3",
            "checkpoint_db_path": PROJECT_ROOT / "data" / "conversation_eval_checkpoints.sqlite3",
        }
    )
    if real and not cfg.qwen_api_key:
        raise SystemExit("QWEN_API_KEY 未配置，不能运行 --real。")

    app = create_app(cfg)
    results: list[dict] = []
    started = time.perf_counter()
    with TestClient(app) as client:
        for task in tasks:
            one_started = time.perf_counter()
            response = client.post(
                "/conversations",
                json={
                    "goal": task["goal"],
                    "user_id": f"eval-{task['id']}",
                    "session_id": f"eval-{task['id']}",
                    "max_turns": int(task.get("max_turns", 4)),
                },
            )
            body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
            initial_status = body.get("status")
            if response.status_code == 200 and task.get("follow_up"):
                follow = client.post(
                    f"/conversations/{body['conversation_id']}/messages",
                    json={
                        "content": task["follow_up"],
                        "user_id": f"eval-{task['id']}",
                        "extra_turns": int(task.get("extra_turns", 2)),
                    },
                )
                body = follow.json() if follow.headers.get("content-type", "").startswith("application/json") else {}
                response = follow

            elapsed = round(time.perf_counter() - one_started, 2)
            messages = body.get("messages") or []
            agents = [m.get("agent") for m in messages if m.get("role") == "agent" and m.get("agent")]
            observations = body.get("observations") or []
            tools = [o.get("tool") for o in observations if o.get("ok")]
            final_answer = str(body.get("final_answer") or "")
            budget = body.get("budget") or {}
            model_calls = int(budget.get("model_calls", 0))
            tokens = int(budget.get("tokens_used", 0))

            expected_agents = list(task.get("expected_agents") or [])
            agent_hit = all(name in agents for name in expected_agents)
            tool_hit = all(name in tools for name in (task.get("required_tools") or []))
            answer_hit = all(word.lower() in final_answer.lower() for word in (task.get("answer_contains") or []))
            answer_any = task.get("answer_any") or []
            if answer_any:
                answer_hit = answer_hit and any(word.lower() in final_answer.lower() for word in answer_any)
            call_hit = model_calls <= int(task.get("max_model_calls", 9999))
            completed = body.get("status") == "completed"
            pass_all = bool(
                response.status_code == 200
                and completed
                and agent_hit
                and tool_hit
                and answer_hit
                and call_hit
            )
            results.append(
                {
                    "id": task["id"],
                    "status": body.get("status"),
                    "http_status": response.status_code,
                    "initial_status": initial_status,
                    "agents": agents,
                    "tools": tools,
                    "model_calls": model_calls,
                    "tokens": tokens,
                    "elapsed_seconds": elapsed,
                    "agent_hit": agent_hit,
                    "tool_hit": tool_hit,
                    "answer_hit": answer_hit,
                    "call_hit": call_hit,
                    "completed": completed,
                    "pass": pass_all,
                    "final_answer": final_answer[:800],
                }
            )

    total = len(results)
    passed = sum(1 for item in results if item["pass"])
    summary = {
        "suite_version": suite.get("version"),
        "mode": "real" if real else "mock",
        "total": total,
        "passed": passed,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "completed_rate": round(sum(1 for item in results if item["completed"]) / total, 4) if total else 0.0,
        "agent_hit_rate": round(sum(1 for item in results if item["agent_hit"]) / total, 4) if total else 0.0,
        "tool_hit_rate": round(sum(1 for item in results if item["tool_hit"]) / total, 4) if total else 0.0,
        "answer_hit_rate": round(sum(1 for item in results if item["answer_hit"]) / total, 4) if total else 0.0,
        "avg_model_calls": round(sum(item["model_calls"] for item in results) / total, 2) if total else 0.0,
        "avg_tokens": round(sum(item["tokens"] for item in results) / total, 1) if total else 0.0,
        "avg_elapsed_seconds": round(sum(item["elapsed_seconds"] for item in results) / total, 2) if total else 0.0,
        "wall_seconds": round(time.perf_counter() - started, 2),
    }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": summary,
        "results": results,
    }


def write_reports(report: dict, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "conversation_eval.json"
    md_path = out_dir / "conversation_eval.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Conversation multi-agent evaluation",
        "",
        f"- generated: `{report['generated_at']}`",
        f"- mode: **{report['summary']['mode']}**",
        f"- pass: **{report['summary']['passed']}/{report['summary']['total']}**",
        "",
        "| task | status | agents | tools | model calls | sec | pass |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for item in report["results"]:
        lines.append(
            f"| {item['id']} | {item['status']} | {', '.join(item['agents'])} | "
            f"{', '.join(item['tools']) or '-'} | {item['model_calls']} | "
            f"{item['elapsed_seconds']} | {'✅' if item['pass'] else '❌'} |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- completed rate: {report['summary']['completed_rate']:.2%}",
        f"- agent hit rate: {report['summary']['agent_hit_rate']:.2%}",
        f"- tool hit rate: {report['summary']['tool_hit_rate']:.2%}",
        f"- answer hit rate: {report['summary']['answer_hit_rate']:.2%}",
        f"- avg model calls: {report['summary']['avg_model_calls']}",
        f"- avg tokens: {report['summary']['avg_tokens']}",
        f"- avg elapsed: {report['summary']['avg_elapsed_seconds']}s",
        "",
        "> This is a fixed regression suite, not a statistical benchmark. Mock mode validates routing/plumbing; real mode is an optional smoke-level signal.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true", help="use the configured Qwen model")
    parser.add_argument("--limit", type=int, default=None, help="run only the first N tasks")
    parser.add_argument("--task", default=None, help="run one task by id")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "outputs")
    args = parser.parse_args()

    report = run_suite(real=args.real, limit=args.limit, task_id=args.task)
    json_path, md_path = write_reports(report, args.out)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    for item in report["results"]:
        print(
            f"{item['id']:<22} pass={item['pass']!s:<5} status={item['status']:<10} "
            f"agents={','.join(item['agents'])} calls={item['model_calls']} tokens={item['tokens']}"
        )
    print(f"JSON={json_path}")
    print(f"MARKDOWN={md_path}")
    if report["summary"]["passed"] != report["summary"]["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()