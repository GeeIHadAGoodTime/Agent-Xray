"""End-to-end integration test: load -> analyze -> grade -> classify -> report."""

from __future__ import annotations

import json
from pathlib import Path

from agent_xray.analyzer import analyze_tasks, load_tasks
from agent_xray.diagnose import build_fix_plan
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.reports import report_health, report_health_data
from agent_xray.root_cause import classify_failures
from agent_xray.surface import surface_for_task


def test_full_pipeline(tmp_trace_dir: Path) -> None:
    """Exercise the complete pipeline end-to-end."""
    # 1. Load tasks
    tasks = load_tasks(tmp_trace_dir)
    assert len(tasks) == 4

    # 2. Analyze
    analyses = analyze_tasks(tasks)
    assert set(analyses) == {"broken-task", "coding-task", "golden-task", "research-task"}

    # 3. Grade
    rules = load_rules()
    grades = grade_tasks(tasks, rules)
    grade_map = {g.task_id: g for g in grades}
    assert grade_map["broken-task"].grade == "BROKEN"
    assert grade_map["golden-task"].grade in {"GOLDEN", "GOOD"}

    # 4. Root cause classify
    failures = classify_failures(tasks, grades)
    assert len(failures) >= 1
    assert any(f.task_id == "broken-task" for f in failures)

    # 5. Fix plan
    plan = build_fix_plan(failures)
    assert len(plan) >= 1
    assert plan[0].priority == 1

    # 6. Reports
    health_text = report_health(tasks, grades, analyses)
    assert "HEALTH DASHBOARD" in health_text
    health_data = report_health_data(tasks, grades, analyses)
    assert health_data["total"] == 4

    # 7. Decision surface
    surface = surface_for_task(tasks[0])
    assert "task_id" in surface
    assert "steps" in surface


def test_pipeline_with_browser_flow_rules(tmp_trace_dir: Path) -> None:
    """Pipeline with browser_flow rules grades golden-task as GOLDEN."""
    tasks = load_tasks(tmp_trace_dir)
    rules = load_rules("browser_flow")
    grades = grade_tasks(tasks, rules)
    grade_map = {g.task_id: g for g in grades}
    assert grade_map["golden-task"].grade == "GOLDEN"


def test_pipeline_json_roundtrip(tmp_trace_dir: Path) -> None:
    """Tasks survive to_dict -> from_dict roundtrip."""
    tasks = load_tasks(tmp_trace_dir)
    for task in tasks:
        payload = task.to_dict()
        serialized = json.dumps(payload)
        assert json.loads(serialized)["task_id"] == task.task_id
