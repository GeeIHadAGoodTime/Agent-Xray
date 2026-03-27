from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.grader import RuleSet, grade_task, load_rules, normalize_score
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


def test_normalize_score_known_ruleset() -> None:
    rules = RuleSet(
        name="normalize",
        description="known score span",
        signals=[
            {"field": "errors", "op": "equals", "value": 0, "points": 2, "label": "zero_errors"},
            {
                "field": "step_count",
                "op": "gte",
                "value": 2,
                "points": 3,
                "else_points": -1,
                "label": "enough_steps",
            },
        ],
        grade_thresholds={"GOLDEN": 4, "GOOD": 2, "OK": 0, "WEAK": -1},
    )

    assert normalize_score(-1, rules) == pytest.approx(0.0)
    assert normalize_score(2, rules) == pytest.approx(0.5)
    assert normalize_score(5, rules) == pytest.approx(1.0)


def test_grade_task_includes_normalized_score() -> None:
    task = AgentTask(
        task_id="task-normalized",
        steps=[AgentStep("task-normalized", 1, "respond", {}, tool_result="ok")],
    )
    rules = RuleSet(
        name="normalize",
        description="task normalization",
        signals=[
            {"field": "errors", "op": "equals", "value": 0, "points": 2, "label": "zero_errors"},
            {
                "field": "step_count",
                "op": "gte",
                "value": 2,
                "points": 3,
                "else_points": -1,
                "label": "enough_steps",
            },
        ],
        grade_thresholds={"GOLDEN": 4, "GOOD": 2, "OK": 0, "WEAK": -1},
    )

    result = grade_task(task, rules)

    assert result.score == 1
    assert result.normalized_score == pytest.approx((1 - (-1)) / (5 - (-1)))


def test_load_rules_prints_validation_warnings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rules_path = tmp_path / "invalid_rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "name": "invalid",
                "signals": [
                    {
                        "field": "missing.metric",
                        "op": "mystery",
                        "value": 1,
                        "points": 1,
                        "label": "bad",
                    }
                ],
                "grade_thresholds": {"GOLDEN": 1},
            }
        ),
        encoding="utf-8",
    )

    load_rules(rules_path)
    err = capsys.readouterr().err

    assert "rules warning" in err
    assert "unknown operator 'mystery'" in err
    assert "unknown field 'missing.metric'" in err
