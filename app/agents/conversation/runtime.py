"""Free-form multi-agent conversation runtime.

Flow is a loop, not a fixed graph:

    user -> supervisor decides next speaker -> worker -> shared transcript
         -> supervisor decides again ... -> finish -> writing -> final

Workers only see the shared transcript and their own tool subset.  The
supervisor can choose any worker on any turn; there are no hard-coded
classify/planner/tool/writer edges for this path.
"""

from __future__ import annotations

import time
from typing import Any

from app.agents.conversation.agents import PROFILES, WorkerAgent
from app.agents.conversation.models import ConversationState, add_message, utc_now
from app.agents.conversation.supervisor import Supervisor
from app.core.budget import ExecutionBudget
from app.core.config import Settings
from app.core.errors import BudgetExceeded
from app.core.metrics import METRICS
from app.services.conversation_store import ConversationStore
from app.services.llm_service import AdaptiveLLMService
from app.tools.registry import ToolRegistry


class ConversationRuntime:
    def __init__(
        self,
        settings: Settings,
        llm: AdaptiveLLMService,
        registry: ToolRegistry,
        store: ConversationStore,
        *,
        supervisor: Supervisor | None = None,
        workers: dict[str, WorkerAgent] | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.registry = registry
        self.store = store
        self.supervisor = supervisor or Supervisor(settings, llm)
        self.workers = workers or {
            name: WorkerAgent(profile, llm, registry, settings) for name, profile in PROFILES.items()
        }

    # ------------------------------------------------------------------
    def run(self, state: ConversationState) -> ConversationState:
        if not any(m.get("role") == "user" for m in state.messages):
            add_message(state, role="user", content=state.goal)
        state.status = "running"
        state.stop_reason = None
        state.updated_at = utc_now()
        self.store.save(state.to_dict())

        try:
            while state.turn < state.max_turns:
                decision = self.supervisor.decide(state)
                now = utc_now()
                state.handoffs.append(
                    {
                        "turn": state.turn,
                        "to": decision.next_agent,
                        "reason": decision.reason,
                        "instruction": decision.instruction,
                        "at": now,
                    }
                )
                state.trace.append(
                    {
                        "at": now,
                        "node": "supervisor",
                        "turn": state.turn,
                        "next_agent": decision.next_agent,
                        "reason": decision.reason,
                        "status": "completed",
                    }
                )
                METRICS.inc("conversation_handoffs_total", to=decision.next_agent)
                add_message(
                    state,
                    role="supervisor",
                    agent="supervisor",
                    content=f"下一位：{decision.next_agent}；{decision.reason}",
                    next_agent=decision.next_agent,
                    instruction=decision.instruction,
                )

                if decision.next_agent == "finish":
                    state.final_answer = self._finish(state)
                    state.status = "completed"
                    state.stop_reason = "finished"
                    break

                worker = self.workers.get(decision.next_agent)
                if worker is None:
                    state.status = "failed"
                    state.stop_reason = "unknown_agent"
                    state.trace.append(
                        {
                            "at": utc_now(),
                            "node": "runtime",
                            "turn": state.turn,
                            "status": "failed",
                            "error": f"unknown agent: {decision.next_agent}",
                        }
                    )
                    break

                started = time.perf_counter()
                result = worker.run(state, decision.instruction)
                duration_ms = round((time.perf_counter() - started) * 1000, 2)
                state.observations.extend(result.get("observations") or [])
                state.turn += 1
                add_message(
                    state,
                    role="agent",
                    agent=worker.profile.name,
                    content=result.get("reply") or "",
                    model_calls=result.get("model_calls", 0),
                    observations=len(result.get("observations") or []),
                )
                state.trace.append(
                    {
                        "at": utc_now(),
                        "node": "agent_turn",
                        "turn": state.turn,
                        "agent": worker.profile.name,
                        "status": "completed",
                        "duration_ms": duration_ms,
                        "model_calls": result.get("model_calls", 0),
                        "observations": len(result.get("observations") or []),
                        "budget": ExecutionBudget.from_dict(state.budget).snapshot(),
                    }
                )
                METRICS.inc("conversation_agent_turns_total", agent=worker.profile.name, outcome="ok")
                state.updated_at = utc_now()
                self.store.save(state.to_dict())
            else:
                state.status = "partial"
                state.stop_reason = "max_turns_exceeded"
                state.final_answer = self._best_effort(state)
                state.trace.append(
                    {
                        "at": utc_now(),
                        "node": "runtime",
                        "turn": state.turn,
                        "status": "partial",
                        "error": "max_turns_exceeded",
                    }
                )
                METRICS.inc("conversation_agent_turns_total", agent="runtime", outcome="max_turns")
        except BudgetExceeded as exc:
            state.status = "partial"
            state.stop_reason = exc.reason
            state.final_answer = self._best_effort(state)
            state.trace.append(
                {
                    "at": utc_now(),
                    "node": "runtime",
                    "turn": state.turn,
                    "status": "budget_exceeded",
                    "error": exc.reason,
                }
            )
            METRICS.inc("conversation_agent_turns_total", agent="runtime", outcome="budget")
        except Exception as exc:
            state.status = "failed"
            state.stop_reason = "runtime_error"
            state.final_answer = self._best_effort(state)
            state.trace.append(
                {
                    "at": utc_now(),
                    "node": "runtime",
                    "turn": state.turn,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            METRICS.inc("conversation_agent_turns_total", agent="runtime", outcome="failed")
        finally:
            state.updated_at = utc_now()
            self.store.save(state.to_dict())
        return state

    # ------------------------------------------------------------------
    def _finish(self, state: ConversationState) -> str:
        last_agent = next(
            (m for m in reversed(state.messages) if m.get("role") == "agent"),
            None,
        )
        if last_agent and last_agent.get("agent") == "writing" and last_agent.get("content"):
            return str(last_agent["content"])

        writer = self.workers["writing"]
        result = writer.run(
            state,
            "根据共享对话与工具观察生成最终答复；只使用已出现的事实，保留局限说明。",
        )
        state.observations.extend(result.get("observations") or [])
        state.turn += 1
        add_message(
            state,
            role="agent",
            agent="writing",
            content=result.get("reply") or "",
            model_calls=result.get("model_calls", 0),
            observations=len(result.get("observations") or []),
        )
        state.trace.append(
            {
                "at": utc_now(),
                "node": "agent_turn",
                "turn": state.turn,
                "agent": "writing",
                "status": "completed",
                "model_calls": result.get("model_calls", 0),
                "observations": len(result.get("observations") or []),
                "finish": True,
            }
        )
        METRICS.inc("conversation_agent_turns_total", agent="writing", outcome="finish")
        return str(result.get("reply") or "")

    @staticmethod
    def _best_effort(state: ConversationState) -> str:
        last_agent = next(
            (m for m in reversed(state.messages) if m.get("role") == "agent"),
            None,
        )
        if last_agent and last_agent.get("content"):
            return str(last_agent["content"])
        return "对话未完成：没有可交付的 Agent 回复。"