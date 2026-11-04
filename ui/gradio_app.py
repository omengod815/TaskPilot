import json
import os
from pathlib import Path

import gradio as gr
import httpx

API_BASE = os.getenv("API_BASE", "http://127.0.0.1:8000").rstrip("/")
API_TOKEN = os.getenv("TASKPILOT_API_TOKEN", "")
HEADERS = {"X-API-Key": API_TOKEN} if API_TOKEN else {}


def _answer_markdown(answer: dict | None) -> str:
    if not answer:
        return "暂无结果。"
    title = answer.get("title", "任务结果")
    text = answer.get("final_text") or answer.get("summary") or answer.get("answer") or ""
    pieces = [f"# {title}", str(text)]
    for key, label in [("findings", "发现"), ("sources", "来源"), ("metrics", "指标"), ("limitations", "局限与注意事项"), ("next_steps", "下一步")]:
        value = answer.get(key)
        if value:
            pieces.append(f"\n## {label}\n" + (json.dumps(value, ensure_ascii=False, indent=2) if isinstance(value, (dict, list)) else str(value)))
    return "\n\n".join(pieces)


def submit_task(goal: str, task_type: str, user_id: str, session_id: str, uploaded_file: str | None):
    if not goal or len(goal.strip()) < 3:
        return "请至少输入 3 个字符的任务目标。", "", "", ""
    try:
        with httpx.Client(timeout=180.0, headers=HEADERS) as client:
            input_file = None
            if uploaded_file:
                path = Path(uploaded_file)
                with path.open("rb") as file_handle:
                    uploaded = client.post(f"{API_BASE}/files/upload", files={"file": (path.name, file_handle, "text/csv")})
                uploaded.raise_for_status()
                input_file = uploaded.json()["filename"]
            payload = {
                "goal": goal.strip(),
                "user_id": user_id.strip() or "local-demo",
                "session_id": session_id.strip() or "default",
            }
            if task_type != "自动判断":
                payload["task_type"] = task_type
            if input_file:
                payload["input_file"] = input_file
            response = client.post(f"{API_BASE}/tasks", json=payload)
            response.raise_for_status()
            data = response.json()
        status = f"状态：{data.get('status')}\n任务 ID：{data.get('task_id')}\n类型：{data.get('task_type')}\n{data.get('message', '')}"
        return status, _answer_markdown(data.get("answer")), json.dumps(data.get("trace", []), ensure_ascii=False, indent=2), data.get("task_id", "")
    except Exception as exc:
        return f"请求失败：{type(exc).__name__}: {exc}\n请确认 FastAPI 已启动，地址为 {API_BASE}。", "", "", ""


def approve_task(task_id: str, user_id: str, approved: bool):
    if not task_id:
        return "没有可审批的任务。", "", ""
    try:
        with httpx.Client(timeout=180.0, headers=HEADERS) as client:
            response = client.post(
                f"{API_BASE}/tasks/{task_id}/approve",
                json={"user_id": user_id.strip() or "local-demo", "approved": approved, "reason": "通过 Gradio 界面提交的人工决定。"},
            )
            response.raise_for_status()
            data = response.json()
        status = f"状态：{data.get('status')}\n任务 ID：{data.get('task_id')}\n{data.get('message', '')}"
        return status, _answer_markdown(data.get("answer")), json.dumps(data.get("trace", []), ensure_ascii=False, indent=2)
    except Exception as exc:
        return f"审批请求失败：{type(exc).__name__}: {exc}", "", ""


with gr.Blocks(title="TaskPilot Agent Platform") as demo:
    gr.Markdown("# TaskPilot：通用多 Agent 任务规划与交付平台\n先用 `MOCK_LLM=1` 验证工作流；切换到真实模型后再评估生成质量。外部操作任务只会生成本地演示计划，不会发送邮件、付款、发布或删除真实数据。")
    with gr.Row():
        goal = gr.Textbox(label="任务目标", lines=4, placeholder="例如：分析 sample_predictions.csv 的 MAE、RMSE 和 R²，并解释结果局限。")
        task_type = gr.Dropdown(label="任务类型", choices=["自动判断", "research", "data_analysis", "writing", "general", "external_action"], value="自动判断")
    with gr.Row():
        user_id = gr.Textbox(label="用户 ID（演示隔离）", value="local-demo")
        session_id = gr.Textbox(label="会话 ID", value="session-001")
        uploaded_file = gr.File(label="可选 CSV 文件", file_types=[".csv"], type="filepath")
    run_button = gr.Button("执行任务", variant="primary")
    status_box = gr.Textbox(label="任务状态", lines=3)
    task_id_box = gr.Textbox(label="当前任务 ID（审批时自动使用）")
    answer_box = gr.Markdown(label="任务结果")
    trace_box = gr.Code(label="执行 Trace", language="json")
    gr.Markdown("## 人工审批（仅对被识别为外部副作用的任务显示需要审批的状态）")
    with gr.Row():
        approve_button = gr.Button("批准并继续")
        reject_button = gr.Button("拒绝执行")
    run_button.click(submit_task, inputs=[goal, task_type, user_id, session_id, uploaded_file], outputs=[status_box, answer_box, trace_box, task_id_box])
    approve_button.click(lambda tid, uid: approve_task(tid, uid, True), inputs=[task_id_box, user_id], outputs=[status_box, answer_box, trace_box])
    reject_button.click(lambda tid, uid: approve_task(tid, uid, False), inputs=[task_id_box, user_id], outputs=[status_box, answer_box, trace_box])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
