"""Validation of the fixed multi-agent conversation evaluation suite."""

from __future__ import annotations

import json
from pathlib import Path


def test_conversation_eval_suite_is_well_formed():
    path = Path(__file__).resolve().parents[1] / "evals" / "conversation_suite.json"
    suite = json.loads(path.read_text(encoding="utf-8-sig"))

    tasks = suite["tasks"]
    assert suite["version"]
    assert len(tasks) >= 8

    ids = [task["id"] for task in tasks]
    assert len(ids) == len(set(ids))

    for task in tasks:
        assert task["goal"]
        assert int(task["max_turns"]) >= 1
        assert isinstance(task.get("expected_agents"), list)
        assert isinstance(task.get("required_tools"), list)
        assert task.get("answer_contains") or task.get("answer_any")
        assert int(task.get("max_model_calls", 0)) >= 1

    assert any(task.get("follow_up") for task in tasks)
    assert any(task["id"] == "mixed_research_data" for task in tasks)