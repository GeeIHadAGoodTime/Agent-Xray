from __future__ import annotations

import pytest

from agent_xray.analyzer import analyze_tasks
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.reports import (
    report_actions,
    report_actions_data,
    report_broken,
    report_broken_data,
    report_compare_days,
    report_compare_days_data,
    report_compare_days_markdown,
    report_cost,
    report_cost_data,
    report_cost_markdown,
    report_fixes,
    report_fixes_data,
    report_fixes_markdown,
    report_flows,
    report_flows_data,
    report_flows_markdown,
    report_golden,
    report_golden_data,
    report_health,
    report_health_data,
    report_health_markdown,
    report_outcomes,
    report_outcomes_data,
    report_tools,
    report_tools_data,
)


def _prepare(tasks):
    rules = load_rules()
    grades = grade_tasks(tasks, rules)
    analyses = analyze_tasks(tasks)
    return grades, analyses


@pytest.fixture
def all_tasks(golden_task, broken_task, coding_task, research_task):
    return [golden_task, broken_task, coding_task, research_task]


# ── Health ───────────────────────────────────────────────────────────


def test_health_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_health(all_tasks, grades, analyses)
    assert "HEALTH DASHBOARD" in text
    assert "Total: 4 tasks" in text


def test_health_data_keys(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_health_data(all_tasks, grades, analyses)
    assert data["total"] == 4
    assert "distribution" in data
    assert set(data["distribution"]) >= {"GOLDEN", "BROKEN"}


def test_health_empty():
    grades, analyses = _prepare([])
    text = report_health([], grades, analyses)
    assert "Total: 0 tasks" in text
    data = report_health_data([], grades, analyses)
    assert data["total"] == 0


def test_health_markdown_contains_table(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_health_markdown(all_tasks, grades, analyses)
    assert "## Health Dashboard" in text
    assert "| Grade | Count | Pct |" in text


# ── Golden ───────────────────────────────────────────────────────────


def test_golden_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_golden(all_tasks, grades, analyses)
    assert "GOLDEN/GOOD RUNS" in text


def test_golden_data_has_tasks(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_golden_data(all_tasks, grades, analyses)
    assert "count" in data
    assert "tasks" in data
    assert data["count"] >= 1


def test_golden_empty():
    data = report_golden_data([], *_prepare([]))
    assert data["count"] == 0


# ── Broken ───────────────────────────────────────────────────────────


def test_broken_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_broken(all_tasks, grades, analyses)
    assert "BROKEN TASKS" in text


def test_broken_data_has_why(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_broken_data(all_tasks, grades, analyses)
    assert "count" in data
    assert "why" in data
    assert "worst_tasks" in data


def test_broken_empty():
    data = report_broken_data([], *_prepare([]))
    assert data["count"] == 0


# ── Tools ────────────────────────────────────────────────────────────


def test_tools_text_contains_header(all_tasks):
    _, analyses = _prepare(all_tasks)
    text = report_tools(all_tasks, analyses)
    assert "TOOL EFFECTIVENESS" in text


def test_tools_data_has_items(all_tasks):
    _, analyses = _prepare(all_tasks)
    data = report_tools_data(all_tasks, analyses)
    assert "tools" in data
    assert len(data["tools"]) > 0
    assert "tool" in data["tools"][0]
    assert "calls" in data["tools"][0]


def test_tools_empty():
    _, analyses = _prepare([])
    data = report_tools_data([], analyses)
    assert data["tools"] == []


# ── Flows ────────────────────────────────────────────────────────────


def test_flows_text_contains_header(all_tasks):
    _, analyses = _prepare(all_tasks)
    text = report_flows(all_tasks, analyses)
    assert "FLOW ANALYSIS" in text


def test_flows_data_has_sites(all_tasks):
    _, analyses = _prepare(all_tasks)
    data = report_flows_data(all_tasks, analyses)
    assert "sites" in data


def test_flows_no_browser_tasks(coding_task, research_task):
    tasks = [coding_task, research_task]
    _, analyses = _prepare(tasks)
    text = report_flows(tasks, analyses)
    assert "FLOW ANALYSIS" in text
    data = report_flows_data(tasks, analyses)
    assert "sites" in data
    assert any(group["domain"] == "coding" for group in data["groups"])


def test_flows_markdown_contains_stage_table(all_tasks):
    _, analyses = _prepare(all_tasks)
    text = report_flows_markdown(all_tasks, analyses)
    assert "## Flow Analysis" in text
    assert "| Stage | Reached | Reach % |" in text


# ── Outcomes ─────────────────────────────────────────────────────────


def test_outcomes_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_outcomes(all_tasks, grades, analyses)
    assert "OUTCOME DISTRIBUTION" in text


def test_outcomes_data_keys(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_outcomes_data(all_tasks, grades, analyses)
    assert "outcomes" in data
    assert "outcome_x_grade" in data


def test_outcomes_empty():
    data = report_outcomes_data([], *_prepare([]))
    assert data["outcomes"] == {}


# ── Actions ──────────────────────────────────────────────────────────


def test_actions_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_actions(all_tasks, grades, analyses)
    assert "PRIORITIZED ACTION ITEMS" in text


def test_actions_data_structure(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_actions_data(all_tasks, grades, analyses)
    assert "action_items" in data
    assert isinstance(data["action_items"], list)


def test_actions_empty():
    data = report_actions_data([], *_prepare([]))
    assert data["action_items"] == []


# â”€â”€ Cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_cost_text_contains_header(all_tasks):
    _, analyses = _prepare(all_tasks)
    text = report_cost(all_tasks, analyses)
    assert "COST ANALYSIS" in text
    assert "MOST EXPENSIVE TASKS" in text


def test_cost_data_has_groups(all_tasks):
    _, analyses = _prepare(all_tasks)
    data = report_cost_data(all_tasks, analyses)
    assert "summary" in data
    assert "by_model" in data
    assert "by_category" in data
    assert "by_day" in data


def test_cost_markdown_contains_table(all_tasks):
    _, analyses = _prepare(all_tasks)
    text = report_cost_markdown(all_tasks, analyses)
    assert "## Cost Analysis" in text
    assert "| Metric | Value |" in text


# â”€â”€ Fixes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def test_fixes_text_contains_header(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_fixes(all_tasks, grades, analyses)
    assert "FIX PLAN REPORT" in text


def test_fixes_data_has_entries(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_fixes_data(all_tasks, grades, analyses)
    assert "fixes" in data
    assert data["count"] >= 1
    assert "affected_tasks" in data["fixes"][0]


def test_fixes_markdown_contains_table(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_fixes_markdown(all_tasks, grades, analyses)
    assert "## Fix Plan Report" in text
    assert "| Priority | Root Cause | Affected Tasks | Impact Score | Investigate Task |" in text


# ── Compare Days ─────────────────────────────────────────────────────


def test_compare_days_text(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_compare_days(all_tasks, grades, analyses, "20260326", "20260327")
    assert "DAY COMPARISON" in text


def test_compare_days_data(all_tasks):
    grades, analyses = _prepare(all_tasks)
    data = report_compare_days_data(all_tasks, grades, analyses, "20260326", "20260327")
    assert "day1" in data
    assert "day2" in data


def test_compare_days_markdown(all_tasks):
    grades, analyses = _prepare(all_tasks)
    text = report_compare_days_markdown(all_tasks, grades, analyses, "20260326", "20260327")
    assert "## Day Comparison" in text
    assert "| Metric | 20260326 | 20260327 | Delta |" in text
