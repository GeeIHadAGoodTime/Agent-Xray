from __future__ import annotations

from agent_xray.diagnose import (
    build_fix_plan,
    format_fix_plan_text,
    get_target_resolver,
    register_target_resolver,
)
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
    assert "tool handler" in plan[0].targets[0].lower() or "tool" in plan[0].targets[0].lower()


def test_custom_target_resolver_registration():
    class CustomResolver:
        def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
            return [f"{root_cause}:{len(evidence)}"]

    register_target_resolver("test-custom", CustomResolver())

    resolver = get_target_resolver("test-custom")
    assert resolver.resolve("spin", ["a", "b"]) == ["spin:2"]

    plan = build_fix_plan([_result("t1", "spin", -3)], target_resolver="test-custom")
    assert plan[0].targets == ["spin:1"]


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
    assert "severity=3/5" in text


def test_fix_plan_investigate_worst_task():
    results = [
        _result("t1", "spin", -2),
        _result("t2", "spin", -8),
    ]
    plan = build_fix_plan(results)
    spin_entry = next(e for e in plan if e.root_cause == "spin")
    assert spin_entry.investigate_task == "t2"  # worst score


def test_fix_plan_severity_scoring():
    results = [
        _result("t1", "approval_block", -1),
        _result("t2", "routing_bug", -1),
        _result("t3", "spin", -1),
        _result("t4", "prompt_bug", -1),
        _result("t5", "reasoning_bug", -1),
    ]
    plan = build_fix_plan(results)
    severity_by_cause = {entry.root_cause: entry.severity for entry in plan}
    assert severity_by_cause == {
        "approval_block": 5,
        "routing_bug": 4,
        "spin": 3,
        "prompt_bug": 2,
        "reasoning_bug": 1,
    }


def test_fix_plan_verify_command_generation():
    plan = build_fix_plan([_result("t1", "tool_bug", -3)])
    assert plan[0].verify_command == "agent-xray surface t1 | grep tools_available"


def test_fix_plan_to_dict():
    plan = build_fix_plan([_result("t1", "early_abort", -1)])
    d = plan[0].to_dict()
    assert d["root_cause"] == "early_abort"
    assert "priority" in d
    assert d["severity"] == 2
    assert "targets" in d
    assert "investigate" in d
    assert d["targets"] == d["investigate"]
    assert d["verify_command"] == "agent-xray reasoning t1 | grep -i success"
    assert "low_confidence" in d


def test_fix_plan_low_confidence_flag_default():
    """Root causes with fewer than 3 tasks (default min_sample_size) get low_confidence."""
    results = [_result("t1", "spin", -5)]
    plan = build_fix_plan(results)
    assert len(plan) == 1
    assert plan[0].low_confidence is True
    assert any("LOW_CONFIDENCE" in e for e in plan[0].evidence)


def test_fix_plan_low_confidence_above_threshold():
    """Root causes meeting min_sample_size should NOT be low_confidence."""
    results = [
        _result("t1", "spin", -5),
        _result("t2", "spin", -3),
        _result("t3", "spin", -2),
    ]
    plan = build_fix_plan(results)
    spin_entry = next(e for e in plan if e.root_cause == "spin")
    assert spin_entry.low_confidence is False
    assert not any("LOW_CONFIDENCE" in e for e in spin_entry.evidence)


def test_fix_plan_custom_min_sample_size():
    """Custom min_sample_size of 5 should flag groups with < 5 tasks."""
    results = [
        _result("t1", "spin", -5),
        _result("t2", "spin", -3),
        _result("t3", "spin", -2),
    ]
    plan = build_fix_plan(results, min_sample_size=5)
    spin_entry = next(e for e in plan if e.root_cause == "spin")
    assert spin_entry.low_confidence is True
    assert any("LOW_CONFIDENCE" in e for e in spin_entry.evidence)


def test_fix_plan_mixed_confidence():
    """Groups above and below threshold should be flagged independently."""
    results = [
        _result("t1", "spin", -5),
        _result("t2", "spin", -3),
        _result("t3", "spin", -2),
        _result("t4", "routing_bug", -1),
    ]
    plan = build_fix_plan(results)
    spin_entry = next(e for e in plan if e.root_cause == "spin")
    routing_entry = next(e for e in plan if e.root_cause == "routing_bug")
    assert spin_entry.low_confidence is False
    assert routing_entry.low_confidence is True


def test_fix_plan_min_sample_size_one():
    """min_sample_size=1 should never flag low_confidence."""
    results = [_result("t1", "spin", -5)]
    plan = build_fix_plan(results, min_sample_size=1)
    assert plan[0].low_confidence is False


def test_fix_plan_to_dict_includes_low_confidence():
    """to_dict should include the low_confidence field."""
    results = [_result("t1", "spin", -5)]
    plan = build_fix_plan(results)
    d = plan[0].to_dict()
    assert d["low_confidence"] is True


def test_format_fix_plan_text_shows_low_confidence():
    """Low-confidence entries should show [LOW CONFIDENCE] in text output."""
    plan = build_fix_plan([_result("t1", "spin", -5)])
    text = format_fix_plan_text(plan)
    assert "[LOW CONFIDENCE]" in text


def test_format_fix_plan_text_no_low_confidence_tag():
    """High-confidence entries should NOT show [LOW CONFIDENCE] in text output."""
    results = [
        _result("t1", "spin", -5),
        _result("t2", "spin", -3),
        _result("t3", "spin", -2),
    ]
    plan = build_fix_plan(results)
    text = format_fix_plan_text(plan)
    assert "[LOW CONFIDENCE]" not in text


def test_fix_plan_new_root_causes_have_targets():
    """valid_alternative_path and consultative_success should have targets."""
    for cause in ("valid_alternative_path", "consultative_success"):
        plan = build_fix_plan([_result("t1", cause, -1)])
        assert len(plan) == 1
        assert len(plan[0].targets) > 0
        assert plan[0].severity == 0


# ---------------------------------------------------------------------------
# Investigation hints philosophy tests
# ---------------------------------------------------------------------------


def test_default_resolver_returns_concepts_not_file_paths():
    """Default resolver should return conceptual search terms, never file paths."""
    from agent_xray.diagnose import INVESTIGATION_HINTS

    for cause, hints in INVESTIGATION_HINTS.items():
        for hint in hints:
            assert "/" not in hint, (
                f"INVESTIGATION_HINTS[{cause!r}] contains a slash: {hint!r} "
                "-- default hints should be concepts, not file paths"
            )
            assert not hint.endswith(".py"), (
                f"INVESTIGATION_HINTS[{cause!r}] ends with .py: {hint!r} "
                "-- default hints should be concepts, not file paths"
            )


def test_fix_plan_to_dict_has_investigate_alias():
    """to_dict() should emit both 'targets' and 'investigate' keys with the same value."""
    plan = build_fix_plan([_result("t1", "spin", -3)])
    d = plan[0].to_dict()
    assert "targets" in d
    assert "investigate" in d
    assert d["targets"] == d["investigate"]


def test_format_fix_plan_text_shows_search_your_codebase():
    """Output should say 'Search your codebase for:' instead of 'Targets:'."""
    plan = build_fix_plan([_result("t1", "spin", -5)])
    text = format_fix_plan_text(plan)
    assert "Search your codebase for:" in text
    assert "Targets:" not in text


def test_format_fix_plan_text_shows_what_happened():
    """Output should include a 'What happened:' line from evidence."""
    plan = build_fix_plan([_result("t1", "spin", -5, evidence=["browser_click_ref repeated 11 times"])])
    text = format_fix_plan_text(plan)
    assert "What happened:" in text
    assert "browser_click_ref repeated 11 times" in text


def test_format_fix_plan_text_shows_root_cause_description():
    """Output should include a 'Root cause:' line."""
    plan = build_fix_plan([_result("t1", "spin", -5)])
    text = format_fix_plan_text(plan)
    assert "Root cause:" in text


def test_novviola_resolver_returns_file_paths():
    """NovviolaTargetResolver should still return file paths (plugin behavior preserved)."""
    from agent_xray.contrib.novviola import NovviolaTargetResolver

    resolver = NovviolaTargetResolver()
    targets = resolver.resolve("routing_bug", ["3 step(s) exposed zero tools"])
    # At least one target should contain a slash or end with .py
    assert any("/" in t or t.endswith(".py") for t in targets), (
        f"NovviolaTargetResolver should return file paths, got: {targets}"
    )


def test_investigation_hints_backward_compat_alias():
    """FIX_TARGETS should be a backward-compatible alias for INVESTIGATION_HINTS."""
    from agent_xray.diagnose import FIX_TARGETS, INVESTIGATION_HINTS

    assert FIX_TARGETS is INVESTIGATION_HINTS
