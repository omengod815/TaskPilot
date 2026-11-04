"""ReAct loop behaviour and checkpointed recovery (including a real process kill)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.agents.react import ActionDecision, ScriptedDecider
from app.services.checkpoint import CheckpointManager, thread_config


def run(graph, state):
    return graph.invoke(state, thread_config(state["task_id"]))


def test_two_tool_calls_and_second_decision_sees_first_observation(graph_factory, state_factory):
    decider = ScriptedDecider(
        [
            ActionDecision(action="tool", tool="safe_calculate", arguments={"expression": "1 + 1"}),
            ActionDecision(action="tool", tool="safe_calculate", arguments={"expression": "2 + 2"}),
            ActionDecision(action="final", final_summary="完成"),
        ]
    )
    graph = graph_factory(decider=decider)
    state = state_factory("react-1", goal="先算 1+1 再算 2+2", task_type="general")
    result = run(graph, state)

    assert result["status"] == "completed"
    assert result["react_step"] == 3
    tools = [o for o in result["observations"] if o.get("tool")]
    assert [o["tool"] for o in tools] == ["safe_calculate", "safe_calculate"]
    assert tools[0]["data"]["result"] == 2.0
    # The decisive proof of a ReAct loop: decision #2 received observation #1.
    assert decider.inputs[1]["observations"][0]["data"]["result"] == 2.0
    assert len(decider.inputs) == 3


def test_unknown_tool_is_rejected_without_execution(graph_factory, state_factory):
    decider = ScriptedDecider(
        [
            ActionDecision(action="tool", tool="drop_database", arguments={}),
            ActionDecision(action="final", final_summary="结束"),
        ]
    )
    graph = graph_factory(decider=decider)
    state = state_factory("react-2", goal="执行危险操作", task_type="general")
    result = run(graph, state)

    rejected = result["observations"][0]
    assert rejected["ok"] is False
    assert rejected["error_code"] == "TOOL_NOT_ALLOWED"
    assert result["status"] in {"completed", "partial"}


def test_budget_exhaustion_stops_a_tool_hammering_agent(graph_factory, state_factory):
    script = [
        ActionDecision(action="tool", tool="safe_calculate", arguments={"expression": f"{i} + 1"}) for i in range(1, 8)
    ]
    decider = ScriptedDecider(script)
    graph = graph_factory(decider=decider)
    state = state_factory("react-3", goal="反复调用工具", task_type="general")
    state["budget"]["limits"]["max_steps"] = 2

    result = run(graph, state)

    assert result["status"] == "partial"
    assert result["stop_reason"] == "max_react_steps_exceeded"
    assert result["react_step"] == 2
    assert len([o for o in result["observations"] if o.get("tool")]) == 2
    assert result["answer"] is not None


def test_final_without_tool_skips_the_tool_node(graph_factory, state_factory):
    decider = ScriptedDecider([ActionDecision(action="final", final_summary="无需工具")])
    graph = graph_factory(decider=decider)
    state = state_factory("react-4", goal="讲个笑话", task_type="general")
    result = run(graph, state)

    assert result["status"] == "completed"
    assert result["react_step"] == 1
    assert result["observations"] == []


def test_external_action_requires_approval_and_executes_once_after_approval(graph_factory, state_factory, components):
    graph = graph_factory()
    state = state_factory("react-5", goal="请发送邮件给客户，说明项目延期", task_type="external_action")
    pending = run(graph, state)

    assert pending["status"] == "pending_approval"
    assert pending["answer"]["external_side_effect_performed"] is False
    files = list(Path(components.settings.output_dir).glob("action_plan_*.json"))
    assert files == []

    stored = components.repository.get_state("react-5")
    stored["approved"] = True
    completed = run(graph, stored)

    assert completed["status"] == "completed"
    files = list(Path(components.settings.output_dir).glob("action_plan_*.json"))
    assert len(files) == 1
    assert completed["observations"][0]["replayed"] is False


def test_approval_survives_process_restart(graph_factory, state_factory, components, settings):
    graph1 = graph_factory()
    state = state_factory("restart-1", goal="请发送邮件给客户，说明项目延期", task_type="external_action")
    pending = run(graph1, state)
    assert pending["status"] == "pending_approval"

    # Simulate a service restart: close the saver and open a new one on the
    # same file, then rebuild the graph.
    components.checkpoints.close()
    checkpoints2 = CheckpointManager(settings.checkpoint_db_path)
    graph2 = graph_factory(checkpointer=checkpoints2.saver)
    stored = components.repository.get_state("restart-1")
    stored["approved"] = True
    completed = run(graph2, stored)
    checkpoints2.close()

    assert completed["status"] == "completed"
    assert len(list(Path(settings.output_dir).glob("action_plan_*.json"))) == 1


@pytest.mark.skipif(os.name != "nt", reason="process-kill harness is exercised on Windows")
def test_process_crash_after_tool_success_does_not_duplicate_effect(components, settings, graph_factory):
    """Real subprocess dies between the side effect and its checkpoint.

    Recovery must not replay the effect: the idempotency record is the source
    of truth, so the second execution returns the stored result instead of
    writing a second action file.
    """

    task_id = "crash-recovery-1"
    env = dict(os.environ)
    project_root = Path(__file__).resolve().parents[1]
    env.update(
        {
            "PYTHONPATH": str(project_root),
            "MOCK_LLM": "1",
            "TASKPILOT_TASK_ID": task_id,
            "TASKPILOT_DB_PATH": str(settings.taskpilot_db_path),
            "TASKPILOT_CHECKPOINT_PATH": str(settings.checkpoint_db_path),
            "TASKPILOT_OUTPUT_DIR": str(settings.output_dir),
            "TASKPILOT_UPLOAD_DIR": str(settings.upload_dir),
            "TASKPILOT_FAULT_INJECTION": "1",
            "TASKPILOT_CRASH_AT": "after_tool_success",
            "TASKPILOT_LOG_JSON": "0",
        }
    )
    process = subprocess.run(
        [sys.executable, str(project_root / "scripts" / "crash_runner.py")],
        cwd=str(project_root),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert process.returncode == 42, f"expected hard crash, got {process.returncode}: {process.stderr[-2000:]}"

    files_before = list(Path(settings.output_dir).glob("action_plan_*.json"))
    assert len(files_before) == 1, "the side effect must have been applied before the crash"

    # Fresh process state: new saver, new graph, same databases.
    components.checkpoints.close()
    checkpoints2 = CheckpointManager(settings.checkpoint_db_path)
    graph2 = graph_factory(checkpointer=checkpoints2.saver)
    resumed = graph2.invoke(None, thread_config(task_id))
    checkpoints2.close()

    files_after = list(Path(settings.output_dir).glob("action_plan_*.json"))
    assert len(files_after) == 1, "recovery replayed the side effect"
    replays = [o for o in resumed.get("observations", []) if o.get("replayed")]
    assert replays, "recovery did not use the stored idempotent result"
    assert resumed["status"] == "completed"