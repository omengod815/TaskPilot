"""Supervisor: choose the next speaker from the shared transcript.

The supervisor does not execute tools and does not own a fixed pipeline.  It
reads the conversation and returns one of the worker names or ``finish``.
"""

from __future__ import annotations

import json
from typing import Any

from app.agents.conversation.agents import agent_catalog
from app.agents.conversation.models import ConversationState, SupervisorDecision
from app.core.budget import ExecutionBudget
from app.core.config import Settings
from app.core.errors import BudgetExceeded
from app.services.llm_service import AdaptiveLLMService

_DATA_WORDS = ("csv", "数据分析", "统计", "表格", "mae", "rmse", "r2", "数据集", "缺失值")
_RESEARCH_WORDS = ("调研", "检索", "搜索", "资料", "来源", "研究", "search", "research")
_WRITING_WORDS = ("写", "报告", "总结", "方案", "草稿", "write", "draft", "summary")


class MockSupervisor:
    """Deterministic policy used by MOCK_LLM=1 and tests."""

    def decide(self, state: ConversationState) -> SupervisorDecision:
        # A follow-up user message starts a new mini-turn: only agents that
        # spoke after that message count as "already done" for this request.
        done = self._agents_since_last_user(state)
        lower = self._latest_user_text(state).lower()

        if not done:
            if any(word in lower for word in _DATA_WORDS):
                return SupervisorDecision(
                    next_agent="data",
                    instruction="读取目标中的 CSV，计算/引用真实指标，并说明样本量与局限。",
                    reason="mock: task mentions CSV/data analysis",
                )
            if any(word in lower for word in _RESEARCH_WORDS):
                return SupervisorDecision(
                    next_agent="research",
                    instruction="检索与目标相关的资料，保留来源，不要编造。",
                    reason="mock: task mentions research",
                )
            if any(word in lower for word in _WRITING_WORDS):
                return SupervisorDecision(
                    next_agent="writing",
                    instruction="根据目标生成结构化答复；没有证据时明确说明。",
                    reason="mock: task mentions writing",
                )
            return SupervisorDecision(
                next_agent="general",
                instruction="直接回答目标；需要算术时使用 safe_calculate。",
                reason="mock: general question",
            )

        wants_writing = any(word in lower for word in _WRITING_WORDS)
        if wants_writing and "writing" not in done and ("research" in done or "data" in done):
            return SupervisorDecision(
                next_agent="writing",
                instruction="综合已有工具观察与对话，生成最终答复；不得新增事实。",
                reason="mock: evidence exists, compose final answer",
            )
        return SupervisorDecision(next_agent="finish", reason="mock: enough evidence or done")

    @staticmethod
    def _latest_user_text(state: ConversationState) -> str:
        for message in reversed(state.messages):
            if message.get("role") == "user" and message.get("content"):
                return str(message["content"])
        return state.goal

    @staticmethod
    def _agents_since_last_user(state: ConversationState) -> set[str]:
        start = 0
        for index, message in enumerate(state.messages):
            if message.get("role") == "user":
                start = index + 1
        return {
            str(message.get("agent"))
            for message in state.messages[start:]
            if message.get("role") == "agent" and message.get("agent")
        }


class LLMSupervisor:
    """Real-model supervisor with a deterministic fallback."""

    def __init__(self, settings: Settings, llm: AdaptiveLLMService) -> None:
        self.settings = settings
        self.llm = llm
        self.fallback = MockSupervisor()

    def decide(self, state: ConversationState) -> SupervisorDecision:
        budget = ExecutionBudget.from_dict(state.budget)
        try:
            decision = self._decide_llm(state, budget)
        except BudgetExceeded:
            raise
        except Exception as exc:
            state.degraded_reasons.append(f"Supervisor 模型不可用，降级为规则调度：{type(exc).__name__}")
            decision = self.fallback.decide(state)
        finally:
            state.budget = budget.to_dict()
        done = MockSupervisor._agents_since_last_user(state)
        if decision.next_agent == "finish" and not done:
            # Never end an empty conversation: force one worker turn.
            return self.fallback.decide(state)
        return decision

    def _decide_llm(self, state: ConversationState, budget: ExecutionBudget) -> SupervisorDecision:
        system = (
            "你是 TaskPilot 的 Supervisor。你不直接回答用户，也不执行工具；"
            "你只根据共享对话决定下一个发言者。\n"
            "可选 next_agent：research（资料调研）、data（CSV/数值）、writing（成稿）、"
            "general（通用问答）、finish（信息足够，交给 Writing 生成最终答复）。\n"
            "规则：\n"
            "1. 需要外部资料时选 research；需要 CSV/指标时选 data；需要成稿时选 writing。\n"
            "2. 不要在没有证据时 finish；如果是简单通用问题，可以直接选 general。\n"
            "3. 同一个 Agent 不要连续调用超过 2 次；重复没有新信息。\n"
            "4. 只输出严格 JSON，字段符合 SupervisorDecision Schema。\n"
            "SupervisorDecision JSON Schema:\n"
            + json.dumps(SupervisorDecision.model_json_schema(), ensure_ascii=False)
        )
        payload = {
            "goal": state.goal,
            "turn": state.turn,
            "max_turns": state.max_turns,
            "remaining_budget": budget.snapshot(),
            "agents": agent_catalog(),
            "already_spoken": sorted(
                {m.get("agent") for m in state.messages if m.get("role") == "agent"}
            ),
            "transcript": [
                {
                    "role": m.get("role"),
                    "agent": m.get("agent"),
                    "content": str(m.get("content", ""))[:1500],
                }
                for m in state.messages[-12:]
            ],
        }
        call = self.llm.generate_structured_with_meta(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            SupervisorDecision,
            user_id=state.user_id,
            budget=budget,
        )
        return call.value


class Supervisor:
    def __init__(self, settings: Settings, llm: AdaptiveLLMService) -> None:
        self.settings = settings
        self.mock = MockSupervisor()
        self.llm_supervisor = None if settings.mock_llm else LLMSupervisor(settings, llm)

    def decide(self, state: ConversationState) -> SupervisorDecision:
        if self.llm_supervisor is None:
            return self.mock.decide(state)
        return self.llm_supervisor.decide(state)

    def catalog(self) -> list[dict[str, Any]]:
        return agent_catalog()