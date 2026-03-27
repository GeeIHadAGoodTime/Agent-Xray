from __future__ import annotations

import pytest

from agent_xray.grader import RuleSet, grade_task, load_rules
from agent_xray.schema import AgentStep, AgentTask


def _rules(
    signals: list[dict[str, object]],
    *,
    thresholds: dict[str, int] | None = None,
) -> RuleSet:
    return RuleSet(
        name="edge",
        description="edge-case rules",
        signals=signals,
        grade_thresholds=thresholds or {"GOLDEN": 3, "GOOD": 2, "OK": 1, "WEAK": 0},
    )


def _task(*steps: AgentStep) -> AgentTask:
    return AgentTask(task_id="task-1", steps=list(steps))


def test_grade_empty_task_not_ok() -> None:
    result = grade_task(AgentTask(task_id="empty-task", steps=[]), load_rules())
    assert result.grade == "BROKEN"


def test_grade_missing_signal_field_ignored() -> None:
    task = _task(AgentStep("task-1", 1, "respond", {}, tool_result="ok"))
    result = grade_task(
        task,
        _rules(
            [
                {"field": "missing.metric", "op": "gte", "value": 1, "points": 5, "label": "missing"},
                {"field": "errors", "op": "equals", "value": 0, "points": 1, "label": "zero_errors"},
            ]
        ),
    )
    assert result.score == 1
    assert result.grade == "OK"
    assert result.signals[0].actual is None


def test_grade_deeply_nested_dotpath() -> None:
    task = _task(
        AgentStep(
            "task-1",
            1,
            "browser_fill_ref",
            {"ref": "payment", "text": "4111 1111 1111 1111"},
            tool_result="card number cvv expir",
            page_url="https://shop.example.test/payment",
        )
    )
    result = grade_task(
        task,
        _rules([{"field": "commerce.reached_payment", "op": "equals", "value": True, "points": 2}]),
    )
    assert result.score == 2
    assert result.grade == "GOOD"


def test_grade_conflicting_labels() -> None:
    task = _task(AgentStep("task-1", 1, "respond", {}, tool_result="ok"))
    result = grade_task(
        task,
        _rules(
            [
                {"field": "errors", "op": "equals", "value": 0, "points": 1, "label": "shared"},
                {"field": "step_count", "op": "equals", "value": 1, "points": 1, "label": "shared"},
            ]
        ),
    )
    assert result.score == 2
    assert [signal.name for signal in result.signals] == ["shared", "shared"]


@pytest.mark.parametrize(
    ("field", "op", "value"),
    [
        ("step_count", "gte", 1),
        ("step_count", "gt", 0),
        ("step_count", "lte", 1),
        ("step_count", "lt", 2),
        ("errors", "equals", 0),
        ("step_count", "in", [0, 1, 2]),
        ("site_name", "contains_any", ["unknown"]),
    ],
)
def test_grade_all_operators(field: str, op: str, value: object) -> None:
    task = _task(AgentStep("task-1", 1, "respond", {}, tool_result="ok"))
    result = grade_task(
        task,
        _rules([{"field": field, "op": op, "value": value, "points": 1}], thresholds={"GOLDEN": 2, "GOOD": 2, "OK": 1, "WEAK": 0}),
    )
    assert result.score == 1
    assert result.grade == "OK"
