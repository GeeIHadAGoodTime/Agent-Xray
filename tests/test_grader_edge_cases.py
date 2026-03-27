from __future__ import annotations

import pytest

from agent_xray.grader import RuleSet, _compare, grade_task, load_rules, validate_rules
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
                {
                    "field": "missing.metric",
                    "op": "gte",
                    "value": 1,
                    "points": 5,
                    "label": "missing",
                },
                {
                    "field": "errors",
                    "op": "equals",
                    "value": 0,
                    "points": 1,
                    "label": "zero_errors",
                },
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
    ("actual", "operator", "expected", "result"),
    [
        (2, "gte", 1, True),
        (2, "gt", 2, False),
        (2, "lte", 2, True),
        (2, "lt", 1, False),
        ("ok", "equals", "ok", True),
        ("ok", "in", ["ok", "other"], True),
        ("ok", "contains_any", ["ok", "other"], True),
        ("ok", "ne", "different", True),
        ("ok", "not_in", ["different"], True),
    ],
)
def test_compare_supports_op_and_shorthand_forms(
    actual: object, operator: str, expected: object, result: bool
) -> None:
    assert _compare(actual, {"op": operator, "value": expected}) is result
    assert _compare(actual, {operator: expected}) is result


def test_validate_rules_catches_bad_rules() -> None:
    rules = RuleSet(
        name="invalid",
        description="bad rules",
        signals=[
            {
                "field": "missing.metric",
                "op": "mystery",
                "value": 1,
                "points": 1,
                "label": "bad_op",
            },
            {"field": "step_count", "op": "gte", "points": 1, "label": "missing_value"},
            {"field": "errors", "op": "equals", "value": 0, "points": 1, "label": "dup_positive"},
            {"field": "errors", "op": "equals", "value": 0, "points": -1, "label": "dup_negative"},
        ],
        grade_thresholds={"GOLDEN": 1},
    )

    warnings = validate_rules(rules)

    assert any("unknown operator 'mystery'" in warning for warning in warnings)
    assert any("unknown field 'missing.metric'" in warning for warning in warnings)
    assert any("missing a comparison value" in warning for warning in warnings)
    assert any("contradictory scoring" in warning for warning in warnings)
    assert any("missing grade threshold 'GOOD'" in warning for warning in warnings)
    assert any("missing grade threshold 'OK'" in warning for warning in warnings)
    assert any("missing grade threshold 'WEAK'" in warning for warning in warnings)


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
        _rules(
            [{"field": field, "op": op, "value": value, "points": 1}],
            thresholds={"GOLDEN": 2, "GOOD": 2, "OK": 1, "WEAK": 0},
        ),
    )
    assert result.score == 1
    assert result.grade == "OK"
