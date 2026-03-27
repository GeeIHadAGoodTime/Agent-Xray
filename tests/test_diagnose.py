from __future__ import annotations

from agent_xray.diagnose import build_fix_plan, format_fix_plan_text
from agent_xray.root_cause import RootCauseResult


def _result(task_id: str, root_cause: str, score: int, **kwargs) -> RootCauseResult:
    return RootCauseResult(
        task_id=task_id,
        root_cause=root_cause,
        grade="BROKEN",
        score=score,
        confidence="high",
        evidence=kwargs.get("evidence", [f"test evidence for {root_cause}"]),
        prompt_section=kwargs.get("prompt_section"),
        prompt_fix_hint=kwargs.get("prompt_fix_hint"),
    )


def test_build_fix_plan_groups_by_cause():
    results = [
        _result("t1", "spin", -3),
        _result("t2", "spin", -5),
        _result("t3", "routing_bug", -1),
    ]
    plan = build_fix_plan(results)
    assert len(plan) == 2
    causes = [e.root_cause for e in plan]
    assert "spin" in causes
    assert "routing_bug" in causes


def test_fix_plan_priority_by_impact():
    results = [
        _result("t1", "spin", -5),
        _result("t2", "spin", -5),
        _result("t3", "routing_bug", -1),
    ]
    plan = build_fix_plan(results)
    assert plan[0].priority == 1
    assert plan[0].root_cause == "spin"  # 2 tasks * 5 = 10 impact


def test_fix_plan_targets_exist():
    plan = build_fix_plan([_result("t1", "tool_bug", -3)])
    assert len(plan) == 1
    assert len(plan[0].targets) > 0
    assert "tool handler" in plan[0].targets[0].lower()


def test_fix_plan_prompt_bug_enrichment():
    result = _result("t1", "prompt_bug", -2, prompt_section="research")
    plan = build_fix_plan([result])
    assert any("research" in t.lower() for t in plan[0].targets)


def test_fix_plan_empty():
    assert build_fix_plan([]) == []


def test_format_fix_plan_text_no_results():
    text = format_fix_plan_text([])
    assert "Nothing to diagnose" in text


def test_format_fix_plan_text_has_entries():
    plan = build_fix_plan([_result("t1", "spin", -5)])
    text = format_fix_plan_text(plan)
    assert "FIX PLAN" in text
    assert "spin" in text
    assert "Priority #1" in text


def test_fix_plan_investigate_worst_task():
    results = [
        _result("t1", "spin", -2),
        _result("t2", "spin", -8),
    ]
    plan = build_fix_plan(results)
    spin_entry = next(e for e in plan if e.root_cause == "spin")
    assert spin_entry.investigate_task == "t2"  # worst score


def test_fix_plan_to_dict():
    plan = build_fix_plan([_result("t1", "early_abort", -1)])
    d = plan[0].to_dict()
    assert d["root_cause"] == "early_abort"
    assert "priority" in d
    assert "targets" in d
