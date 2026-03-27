from __future__ import annotations

from pathlib import Path

from agent_xray.grader import grade_task, load_rules
from agent_xray.root_cause import classify_task
from agent_xray.schema import AgentStep, AgentTask


def test_spin_root_cause() -> None:
    task = AgentTask(
        task_id="task-1",
        steps=[
            AgentStep("task-1", 1, "browser_snapshot", {}, tool_result="same"),
            AgentStep("task-1", 2, "browser_snapshot", {}, tool_result="same"),
            AgentStep("task-1", 3, "browser_snapshot", {}, tool_result="same"),
            AgentStep("task-1", 4, "browser_snapshot", {}, tool_result="same"),
            AgentStep("task-1", 5, "browser_snapshot", {}, tool_result="same"),
        ],
    )
    rules = load_rules(
        Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules" / "default.json"
    )
    grade = grade_task(task, rules)
    cause = classify_task(task, grade)
    assert cause.root_cause == "spin"
