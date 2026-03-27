from __future__ import annotations

from pathlib import Path

from agent_xray.grader import grade_task, load_rules
from agent_xray.schema import AgentStep, AgentTask


def test_browser_flow_grade() -> None:
    task = AgentTask(
        task_id="task-1",
        task_text="buy something",
        task_category="commerce",
        steps=[
            AgentStep(
                "task-1",
                1,
                "browser_navigate",
                {"url": "https://shop.example.test"},
                tool_result="your cart",
            ),
            AgentStep(
                "task-1",
                2,
                "browser_fill_ref",
                {"ref": "e10", "text": "123 Main St"},
                tool_result="checkout",
                page_url="https://shop.example.test/checkout",
            ),
            AgentStep(
                "task-1",
                3,
                "browser_fill_ref",
                {"ref": "e20", "text": "4111 1111 1111 1111"},
                tool_result="card number cvv expir",
                page_url="https://shop.example.test/payment",
            ),
        ],
    )
    rules = load_rules(
        Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules" / "browser_flow.json"
    )
    result = grade_task(task, rules)
    assert result.grade == "GOLDEN"
    assert result.score >= 8
