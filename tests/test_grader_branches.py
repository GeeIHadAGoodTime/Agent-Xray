from __future__ import annotations

import pytest

from agent_xray.grader import RuleSet, _compare, grade_task
from agent_xray.schema import AgentStep, AgentTask


def _task() -> AgentTask:
    return AgentTask(
        task_id="task-1",
        steps=[
            AgentStep(
                task_id="task-1",
                step=1,
                tool_name="browser_fill_ref",
                tool_input={"ref": "card-number", "text": "4111 1111 1111 1111"},
                tool_result="card number accepted",
                page_url="https://shop.example.test/payment",
            )
        ],
    )


def _rules(signal: dict[str, object]) -> RuleSet:
    return RuleSet(
        name="branch-rules",
        description="exercise grader comparison branches",
        signals=[signal],
        grade_thresholds={"GOLDEN": 2, "GOOD": 2, "OK": 1, "WEAK": 0},
    )


@pytest.mark.parametrize(
    ("actual", "rule"),
    [
        (5, {"op": "gte", "value": 5}),
        (5, {"op": "gt", "value": 4}),
        (5, {"op": "lte", "value": 5}),
        (5, {"op": "lt", "value": 6}),
        ("ok", {"op": "equals", "value": "ok"}),
        ("ok", {"op": "ne", "value": "bad"}),
        ("payment", {"op": "in", "value": ["cart", "payment"]}),
        ("payment", {"op": "not_in", "value": ["cart", "checkout"]}),
        (["card", "cvv"], {"op": "contains_any", "value": ["cvv", "zip"]}),
    ],
)
def test_compare_supports_every_explicit_operator(actual: object, rule: dict[str, object]) -> None:
    assert _compare(actual, rule) is True


@pytest.mark.parametrize(
    ("actual", "rule"),
    [
        (5, {"gte": 5}),
        (5, {"gt": 4}),
        (5, {"lte": 5}),
        (5, {"lt": 6}),
        ("ok", {"equals": "ok"}),
        ("ok", {"ne": "bad"}),
        ("payment", {"in": ["cart", "payment"]}),
        ("payment", {"not_in": ["cart", "checkout"]}),
        (["card", "cvv"], {"contains_any": ["cvv", "zip"]}),
    ],
)
def test_compare_supports_every_legacy_operator_key(
    actual: object, rule: dict[str, object]
) -> None:
    assert _compare(actual, rule) is True


@pytest.mark.parametrize(
    ("field", "op", "value"),
    [
        ("step_count", "gte", 1),
        ("step_count", "gt", 0),
        ("step_count", "lte", 1),
        ("step_count", "lt", 2),
        ("errors", "equals", 0),
        ("errors", "ne", 1),
        ("site_name", "in", ["shop", "unknown"]),
        ("site_name", "not_in", ["docs", "search"]),
        ("site_name", "contains_any", ["shop", "docs"]),
    ],
)
def test_grade_task_hits_all_compare_operators(field: str, op: str, value: object) -> None:
    result = grade_task(
        _task(),
        _rules({"field": field, "op": op, "value": value, "points": 1}),
    )
    assert result.score == 1
    assert result.grade == "OK"


def test_grade_task_missing_field_yields_none_actual_and_no_points() -> None:
    result = grade_task(
        _task(),
        _rules({"field": "missing.metric", "op": "gte", "value": 1, "points": 1}),
    )
    assert result.score == 0
    assert result.signals[0].actual is None
    assert result.signals[0].passed is False


@pytest.mark.parametrize(
    ("actual", "rule"),
    [
        (None, {"op": "gte", "value": 1}),
        (None, {"op": "gt", "value": 1}),
        (None, {"op": "lte", "value": 1}),
        (None, {"op": "lt", "value": 1}),
        ("five", {"op": "gte", "value": 1}),
        ("five", {"op": "lt", "value": 1}),
    ],
)
def test_compare_type_errors_warn_and_return_false(actual: object, rule: dict[str, object]) -> None:
    with pytest.warns(UserWarning, match="TypeError"):
        assert _compare(actual, rule) is False


@pytest.mark.parametrize(
    ("actual", "rule"),
    [
        ("value", {"op": "in", "value": "value"}),
        ("value", {"op": "not_in", "value": "value"}),
        ("value", {"op": "contains_any", "value": "value"}),
        ("value", {"in": "value"}),
        ("value", {"not_in": "value"}),
        ("value", {"contains_any": "value"}),
    ],
)
def test_compare_rejects_wrong_expected_container_types(
    actual: object, rule: dict[str, object]
) -> None:
    assert _compare(actual, rule) is False


def test_compare_raises_for_missing_comparator() -> None:
    with pytest.raises(ValueError, match="missing a comparator"):
        _compare(1, {"field": "step_count"})
