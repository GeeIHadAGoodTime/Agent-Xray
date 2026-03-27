from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.surface import diff_tasks, reasoning_for_task, surface_for_task


def test_surface_contains_history() -> None:
    task = AgentTask(
        task_id="task-1",
        task_text="inspect page",
        steps=[
            AgentStep(
                "task-1",
                1,
                "browser_navigate",
                {"url": "https://example.test"},
                tool_result="loaded",
            ),
            AgentStep(
                "task-1",
                2,
                "browser_snapshot",
                {},
                tool_result="checkout page",
                llm_reasoning="I should inspect the page",
            ),
        ],
    )
    surface = surface_for_task(task)
    assert surface["steps"][0]["conversation_history"][0]["content"] == "inspect page"
    reasoning = reasoning_for_task(task)
    assert reasoning["reasoning_chain"][1]["reasoning"] == "I should inspect the page"


def test_diff_detects_divergence() -> None:
    left = AgentTask(task_id="left", steps=[AgentStep("left", 1, "a", {}, tool_result="ok")])
    right = AgentTask(task_id="right", steps=[AgentStep("right", 1, "b", {}, tool_result="ok")])
    diff = diff_tasks(left, right)
    assert diff["diverged_at_step"] == 1
