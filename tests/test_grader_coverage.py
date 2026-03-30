from __future__ import annotations

import pytest

from agent_xray.analyzer import analyze_task
from agent_xray.grader import RuleSet, _compare, grade_task, validate_rules
from agent_xray.schema import AgentStep, AgentTask


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-1",
        steps=[AgentStep("task-1", 1, "respond", {}, tool_result="done")],
    )


def _rules(
    signals: list[dict[str, object]],
    *,
    thresholds: dict[str, int] | None = None,
    golden_requirements: list[object] | None = None,
) -> RuleSet:
    return RuleSet(
        name="coverage",
        description="coverage rules",
        signals=signals,
        grade_thresholds=thresholds or {"GOLDEN": 1, "GOOD": 0, "OK": 0, "WEAK": -1},
        golden_requirements=golden_requirements or [],
    )


def test_golden_threshold_is_reachable() -> None:
    result = grade_task(
        _task(),
        _rules([{"field": "step_count", "op": "gte", "value": 1, "points": 1, "label": "enough_steps"}]),
    )

    assert result.score == 1
    assert result.grade == "GOLDEN"


def test_validate_rules_catches_malformed_rules() -> None:
    warnings = validate_rules(
        RuleSet(
            name="invalid",
            description="invalid rules",
            signals=[
                {"field": "step_count", "op": "between", "value": 1, "points": 1, "label": "bad_between"},
                {"field": "missing.metric", "op": "eq", "points": 1, "label": "missing_value"},
                {"field": "errors", "op": "mystery", "value": 0, "points": 1, "label": "bad_op"},
            ],
            grade_thresholds={"GOLDEN": 1},
        )
    )

    assert any("unknown field 'missing.metric'" in warning for warning in warnings)
    assert any("missing a comparison value" in warning for warning in warnings)
    assert any("unknown operator 'mystery'" in warning for warning in warnings)


@pytest.mark.parametrize(
    ("actual", "rule", "expected"),
    [
        (1, {"op": "eq", "value": 1}, True),
        (1, {"op": "ne", "value": 2}, True),
        (2, {"op": "gt", "value": 1}, True),
        (2, {"op": "gte", "value": 2}, True),
        (1, {"op": "lt", "value": 2}, True),
        (1, {"op": "lte", "value": 1}, True),
        ("ok", {"op": "in", "value": ["ok", "other"]}, True),
        ("ok", {"op": "not_in", "value": ["other"]}, True),
        (3, {"op": "between", "value": [2, 4]}, True),
    ],
)
def test_compare_supports_requested_operators(
    actual: object, rule: dict[str, object], expected: bool
) -> None:
    assert _compare(actual, rule) is expected


def test_golden_requirements_gate_golden_grade() -> None:
    result = grade_task(
        _task(),
        _rules(
            [
                {"field": "step_count", "op": "gte", "value": 1, "points": 2, "label": "enough_steps"},
                {"field": "errors", "op": "equals", "value": 0, "points": 1, "label": "no_errors"},
            ],
            thresholds={"GOLDEN": 3, "GOOD": 1, "OK": 0, "WEAK": -1},
            golden_requirements=["missing_signal"],
        ),
    )

    assert result.score == 3
    assert result.grade == "GOOD"
    assert any("missing_signal requirement not met" in reason for reason in result.reasons)


def test_grounded_answer_signal_if_present() -> None:
    metrics = analyze_task(_task()).metrics()
    if "grounded_answer" not in metrics:
        pytest.skip("grounded_answer metric not present")

    result = grade_task(
        _task(),
        _rules([{"field": "grounded_answer", "op": "equals", "value": True, "points": 1, "label": "grounded"}]),
    )

    assert result.signals[0].name == "grounded"
