"""Intent-aware context assembly with an explicit memory hierarchy.

Four different things used to share one "context" bucket.  They are now kept
separate and budgeted separately:

1. **task state** – owned by the graph/checkpoint, never fed back as memory;
2. **short-term memory** – completed-task summaries for this user/session;
3. **long-term memory** – explicit preferences, retrieved by relevance;
4. **conversation history** – same user/session task records (completed only).

Retrieval is still a transparent lexical baseline (not a vector store).  That
is a documented limitation, not an accident: the upgrade's focus is reliability,
and swapping in BM25/embeddings is a one-function change.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from app.core.config import get_settings
from app.services.memory import MemoryStore
from app.services.repository import TaskRepository

settings = get_settings()

DEFAULT_TOOL_CATALOG = {
    "research": ["search_web(query, top_k): 检索资料；依赖故障时明确标记降级来源。"],
    "data_analysis": ["analyze_csv(filename): 在允许目录内读取 CSV 并返回统计摘要。"],
    "writing": ["writer: 按目标、受众、语气生成文稿；不调用外部副作用工具。"],
    "general": ["safe_calculate(expression): 只计算安全数学表达式；writer: 生成解释或计划。"],
    "external_action": ["execute_demo_action(goal): 仅生成本地操作计划文件；必须人工审批。"],
}


def _tokens(text: str) -> set[str]:
    latin = set(re.findall(r"[a-zA-Z0-9_+-]{2,}", text.lower()))
    cjk_runs = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    cjk: set[str] = set()
    for run in cjk_runs:
        cjk.add(run)
        cjk.update(run[i:i + 2] for i in range(len(run) - 1))
    return latin | cjk


def _compress_relevant(text: str, query: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    query_tokens = _tokens(query)
    parts = [p.strip() for p in re.split(r"(?<=[。！？.!?])\s+|\n+", text) if p.strip()]
    ranked = sorted(
        enumerate(parts),
        key=lambda pair: (len(query_tokens & _tokens(pair[1])), -pair[0]),
        reverse=True,
    )
    chosen: list[tuple[int, str]] = []
    used = 0
    for idx, part in ranked:
        if used + len(part) + 1 > max_chars:
            continue
        chosen.append((idx, part))
        used += len(part) + 1
    if not chosen:
        return text[:max_chars]
    chosen.sort(key=lambda item: item[0])
    return "\n".join(part for _, part in chosen)


class ContextService:
    def __init__(
        self,
        repository: TaskRepository,
        *,
        memory: MemoryStore | None = None,
        tool_catalog: Callable[[str], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.repository = repository
        self.memory = memory
        self.tool_catalog = tool_catalog

    def _tool_lines(self, task_type: str) -> list[str]:
        if self.tool_catalog is not None:
            lines = []
            for item in self.tool_catalog(task_type):
                params = ", ".join((item.get("parameters") or {}).get("properties", {}).keys())
                lines.append(f"{item['name']}({params}): {item.get('description', '')}")
            if lines:
                return lines
        return DEFAULT_TOOL_CATALOG.get(task_type, DEFAULT_TOOL_CATALOG["general"])

    def build(self, state: dict[str, Any]) -> dict[str, Any]:
        goal = state.get("goal", "")
        task_type = state.get("task_type", "general")
        user_id = state.get("user_id", "local-demo")
        session_id = state.get("session_id", "default")
        plan = state.get("plan") or {}
        budget = min(int(plan.get("context_budget_chars", settings.context_max_chars)), settings.context_max_chars)

        tool_lines = self._tool_lines(task_type)
        context_parts = [f"任务类型：{task_type}", "本任务可用能力：\n- " + "\n- ".join(tool_lines)]

        memory_chars = 0
        memory_selected = 0
        if self.memory is not None:
            memory_budget = max(300, int(budget * 0.35))
            memories = self.memory.recall(user_id=user_id, query=goal, limit=4, max_chars=memory_budget)
            if memories:
                grouped: dict[str, list[dict[str, Any]]] = {}
                for item in memories:
                    grouped.setdefault(item["kind"], []).append(item)
                labels = {"long_term": "用户长期偏好（跨会话）", "short_term": "本会话近期任务记忆"}
                blocks = []
                for kind, items in grouped.items():
                    lines = "\n".join(f"- {item['content'][:600]}" for item in items)
                    blocks.append(f"【{labels.get(kind, kind)}】\n{lines}")
                block = "\n".join(blocks)
                context_parts.append(block)
                memory_chars = len(block)
                memory_selected = len(memories)

        history = self.repository.recent_states(user_id, session_id, limit=20)
        goal_tokens = _tokens(goal)
        scored = []
        for item in history:
            if item.get("task_id") == state.get("task_id"):
                continue
            # Only finished tasks may become conversational memory.  A task
            # waiting for approval is state, not history.
            if item.get("status") not in {"completed", "partial"}:
                continue
            past_goal = item.get("goal", "")
            answer = item.get("answer") or {}
            answer_text = str(answer.get("summary") or answer.get("answer") or answer.get("final_text") or "")
            overlap = len(goal_tokens & _tokens(past_goal + " " + answer_text))
            if overlap:
                scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        history_lines = []
        for _, item in scored[:3]:
            answer = item.get("answer") or {}
            summary = answer.get("summary") or answer.get("answer") or answer.get("final_text") or ""
            history_lines.append(f"历史任务：{item.get('goal', '')}\n历史结果摘要：{str(summary)[:700]}")
        if history_lines:
            context_parts.append("与当前目标相关的同会话历史（只取前 3 条）：\n" + "\n---\n".join(history_lines))

        raw_context = "\n\n".join(context_parts)
        compressed = _compress_relevant(raw_context, goal, budget)
        return {
            "context_text": compressed,
            "context_chars": len(compressed),
            "context_budget_chars": budget,
            "history_candidates": len(scored),
            "history_selected": len(history_lines),
            "memory_selected": memory_selected,
            "memory_chars": memory_chars,
            "context_compressed": len(raw_context) > len(compressed),
            "retrieval_mode": "same_user_same_session_lexical_overlap+separated_memory",
        }