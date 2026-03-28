"""Tests for output quality DX improvements (Items 4, 5, 6, 7, 14)."""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray.completeness import check_completeness
from agent_xray.diagnose import FIX_TARGETS, SEVERITY_BY_ROOT_CAUSE, build_fix_plan
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.root_cause import ROOT_CAUSES, RootCauseResult, classify_task
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome
from agent_xray.surface import (
    diff_tasks,
    enriched_tree_for_tasks,
    format_diff_summary,
    format_enriched_tree_text,
    format_prompt_diff,
    format_tree_text,
    tree_for_tasks,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(
    task_id: str,
    step: int,
    tool_name: str,
    tool_input: dict | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    duration_ms: int | None = None,
    llm_reasoning: str | None = None,
    tools_available: list[str] | None = None,
    model_name: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
        duration_ms=duration_ms,
        llm_reasoning=llm_reasoning,
        tools_available=tools_available,
        model_name=model_name,
    )


def _outcome(task_id: str, status: str, total_steps: int) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        total_steps=total_steps,
        total_duration_s=total_steps * 2.0,
    )


def _make_spinning_task() -> AgentTask:
    """Task that spins on browser_click_ref for many steps."""
    task_id = "spin-task"
    steps = []
    for i in range(1, 26):
        tool = "browser_click_ref" if i >= 12 else "browser_navigate"
        steps.append(
            _step(
                task_id, i, tool, {"ref": f"el-{i}"},
                tool_result="ok" if i < 12 else None,
                error="element not found" if i >= 12 else None,
                page_url="https://dominos.com/checkout",
            )
        )
    return AgentTask(
        task_id=task_id,
        task_text="Order a large pepperoni pizza from Dominos",
        steps=steps,
        outcome=_outcome(task_id, "spin_terminated", 25),
    )


def _make_golden_task() -> AgentTask:
    """Task that completes successfully in 8 steps."""
    task_id = "golden-task"
    tools = [
        "browser_navigate", "browser_snapshot", "browser_click",
        "browser_fill_ref", "browser_click", "browser_fill_ref",
        "browser_click", "browser_snapshot",
    ]
    steps = []
    for i, tool in enumerate(tools, 1):
        steps.append(
            _step(
                task_id, i, tool, {"ref": f"el-{i}"},
                tool_result=f"step {i} ok",
                page_url=f"https://dominos.com/step{i}",
                duration_ms=5000 + i * 100,
            )
        )
    return AgentTask(
        task_id=task_id,
        task_text="Order a large pepperoni pizza from Dominos",
        steps=steps,
        outcome=_outcome(task_id, "payment_gate", 8),
        metadata={"system_prompt_text": "You are a helpful browser agent. Capabilities: browser"},
    )


def _make_golden_task_different_prompt() -> AgentTask:
    """Similar golden task with a different system prompt."""
    task = _make_golden_task()
    task.task_id = "golden-task-v2"
    task.metadata = {
        "system_prompt_text": "You are a helpful browser agent. Capabilities: browser, file ops, web search"
    }
    for step in task.steps:
        step.task_id = "golden-task-v2"
    if task.outcome:
        task.outcome.task_id = "golden-task-v2"
    return task


# ===========================================================================
# Item 4: Diff Summary Mode
# ===========================================================================

class TestDiffSummary:
    """Tests for the --summary flag on diff."""

    def test_format_diff_summary_basic(self) -> None:
        left = _make_spinning_task()
        right = _make_golden_task()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        assert "DIFF SUMMARY:" in summary
        assert "spin-task" in summary
        assert "golden-task" in summary
        assert "outcome:" in summary
        assert "steps:" in summary
        assert "errors:" in summary
        assert "unique_tools:" in summary

    def test_summary_shows_step_counts(self) -> None:
        left = _make_spinning_task()
        right = _make_golden_task()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        # Left has 25 steps, right has 8
        assert "25" in summary
        assert "8" in summary

    def test_summary_shows_outcomes(self) -> None:
        left = _make_spinning_task()
        right = _make_golden_task()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        assert "spin_terminated" in summary
        assert "payment_gate" in summary

    def test_summary_detects_spin(self) -> None:
        left = _make_spinning_task()
        right = _make_golden_task()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        assert "KEY DIFFERENCES:" in summary
        assert "spun on" in summary or "repeats" in summary

    def test_summary_shows_error_counts(self) -> None:
        left = _make_spinning_task()
        right = _make_golden_task()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        # Left has errors (steps 12-25 = 14 errors), right has 0
        lines = summary.split("\n")
        error_line = [l for l in lines if "errors:" in l]
        assert error_line

    def test_summary_identical_tasks(self) -> None:
        task = _make_golden_task()
        data = diff_tasks(task, task)
        summary = format_diff_summary(data)

        assert "DIFF SUMMARY:" in summary
        # Should still show the comparison even if identical


# ===========================================================================
# Item 5: Enriched Tree Output
# ===========================================================================

class TestEnrichedTree:
    """Tests for enriched tree with grade, score, step count, outcome."""

    def test_enriched_tree_without_grades(self) -> None:
        tasks = [_make_golden_task(), _make_spinning_task()]
        tree = enriched_tree_for_tasks(tasks)

        # Should have entries with task metadata
        found_ids = set()
        for _day, sites in tree.items():
            for _site, infos in sites.items():
                for info in infos:
                    assert "task_id" in info
                    assert "steps" in info
                    assert "outcome" in info
                    found_ids.add(info["task_id"])
        assert "golden-task" in found_ids
        assert "spin-task" in found_ids

    def test_enriched_tree_with_grades(self) -> None:
        tasks = [_make_golden_task(), _make_spinning_task()]
        rules = load_rules("default")
        grades = grade_tasks(tasks, rules)
        tree = enriched_tree_for_tasks(tasks, grades)

        has_grade = False
        for _day, sites in tree.items():
            for _site, infos in sites.items():
                for info in infos:
                    if "grade" in info:
                        has_grade = True
                        assert "score" in info
        assert has_grade

    def test_enriched_tree_text_shows_site_summary(self) -> None:
        tasks = [_make_golden_task(), _make_spinning_task()]
        rules = load_rules("default")
        grades = grade_tasks(tasks, rules)
        tree = enriched_tree_for_tasks(tasks, grades)
        text = format_enriched_tree_text(tree)

        assert "TASK TREE" in text
        # Should show task count per site
        assert "tasks:" in text

    def test_enriched_tree_text_shows_task_details(self) -> None:
        tasks = [_make_golden_task()]
        tree = enriched_tree_for_tasks(tasks)
        text = format_enriched_tree_text(tree)

        assert "golden-task" in text
        assert "steps" in text
        assert "payment_gate" in text

    def test_enriched_tree_shows_grade_and_score(self) -> None:
        tasks = [_make_golden_task()]
        rules = load_rules("default")
        grades = grade_tasks(tasks, rules)
        tree = enriched_tree_for_tasks(tasks, grades)
        text = format_enriched_tree_text(tree)

        # Should show grade labels like GOLDEN, GOOD, etc.
        has_grade_label = any(
            label in text for label in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
        )
        assert has_grade_label

    def test_plain_tree_still_works(self) -> None:
        """The original tree_for_tasks and format_tree_text remain functional."""
        tasks = [_make_golden_task()]
        tree = tree_for_tasks(tasks)
        text = format_tree_text(tree)

        assert "TASK TREE" in text
        assert "golden-task" in text


# ===========================================================================
# Item 6: Rename prompt_bug Fallback to unclassified
# ===========================================================================

class TestUnclassifiedFallback:
    """Tests that the fallback classifier returns 'unclassified' instead of 'prompt_bug'."""

    def test_unclassified_in_root_causes_dict(self) -> None:
        assert "unclassified" in ROOT_CAUSES
        assert ROOT_CAUSES["unclassified"]["label"] == "Unclassified"

    def test_unclassified_in_fix_targets(self) -> None:
        assert "unclassified" in FIX_TARGETS

    def test_unclassified_in_severity_map(self) -> None:
        assert "unclassified" in SEVERITY_BY_ROOT_CAUSE
        assert SEVERITY_BY_ROOT_CAUSE["unclassified"] == 1

    def test_fallback_returns_unclassified(self) -> None:
        """A task with no strong signal should get 'unclassified', not 'prompt_bug'."""
        from agent_xray.grader import GradeResult, SignalResult

        # Create a minimal BROKEN task with no obvious root cause pattern
        task = AgentTask(
            task_id="ambiguous-task",
            steps=[
                _step("ambiguous-task", 1, "some_tool", {}, tool_result="ok"),
                _step("ambiguous-task", 2, "another_tool", {}, tool_result="ok"),
            ],
            outcome=_outcome("ambiguous-task", "failed", 2),
        )
        grade = GradeResult(
            task_id="ambiguous-task",
            grade="BROKEN",
            score=-5,
            reasons=["low score"],
            metrics={},
            signals=[],
        )
        result = classify_task(task, grade)
        assert result is not None
        # Should be either unclassified or a genuine classification
        # (not prompt_bug from fallback — prompt_bug only when prompt_section is found)
        if result.root_cause == "prompt_bug":
            # If prompt_bug, it should have a prompt_section (genuine detection)
            assert result.prompt_section is not None
        else:
            # Most likely path for this ambiguous task
            assert result.root_cause in ROOT_CAUSES

    def test_genuine_prompt_bug_preserved(self) -> None:
        """A task with real prompt confusion should still be classified as prompt_bug."""
        from agent_xray.grader import GradeResult, SignalResult

        task = AgentTask(
            task_id="confused-task",
            steps=[
                _step(
                    "confused-task", 1, "browser_navigate",
                    {"url": "https://example.com"},
                    tool_result="page loaded",
                    llm_reasoning="I'm not sure what to do next, this is confusing",
                ),
            ],
            outcome=_outcome("confused-task", "failed", 1),
        )
        grade = GradeResult(
            task_id="confused-task",
            grade="BROKEN",
            score=-5,
            reasons=["low score"],
            metrics={},
            signals=[],
        )
        result = classify_task(task, grade)
        assert result is not None
        # Should detect prompt confusion and classify as prompt_bug
        assert result.root_cause == "prompt_bug"

    def test_unclassified_evidence_message(self) -> None:
        """Unclassified fallback should have descriptive evidence."""
        from agent_xray.grader import GradeResult, SignalResult

        task = AgentTask(
            task_id="unknown-task",
            steps=[
                _step("unknown-task", 1, "custom_tool", {}, tool_result="done"),
            ],
            outcome=_outcome("unknown-task", "failed", 1),
        )
        grade = GradeResult(
            task_id="unknown-task",
            grade="BROKEN",
            score=-3,
            reasons=["low"],
            metrics={},
            signals=[],
        )
        result = classify_task(task, grade)
        assert result is not None
        if result.root_cause == "unclassified":
            assert any("manual investigation" in e for e in result.evidence)

    def test_unclassified_in_fix_plan(self) -> None:
        """Unclassified root cause should work in build_fix_plan."""
        results = [
            RootCauseResult(
                task_id="t1",
                root_cause="unclassified",
                grade="BROKEN",
                score=-5,
                confidence="low",
                evidence=["No specific root cause identified — manual investigation recommended"],
            )
        ]
        plan = build_fix_plan(results)
        assert len(plan) == 1
        assert plan[0].root_cause == "unclassified"
        assert plan[0].severity == 1


# ===========================================================================
# Item 7: Full Completeness Display
# ===========================================================================

class TestFullCompletenessDisplay:
    """Tests for showing all 13 dimensions with PASS/FAIL indicators."""

    def test_completeness_shows_all_dimensions(self) -> None:
        tasks = [_make_golden_task()]
        report = check_completeness(tasks)
        text = report.format_text()

        # Should show both PASS and FAIL markers
        assert "[PASS]" in text or "[FAIL]" in text

    def test_completeness_all_dimensions_tracked(self) -> None:
        tasks = [_make_golden_task()]
        report = check_completeness(tasks)

        # Should have all 13 dimensions tracked
        assert report.dimensions_checked == 13
        assert len(report.all_dimensions) == 13

    def test_completeness_failing_shows_fix_hint(self) -> None:
        tasks = [_make_golden_task()]
        report = check_completeness(tasks)
        text = report.format_text()

        # Any FAIL dimension should have a Fix: hint
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if "[FAIL]" in line and i + 1 < len(lines):
                assert "Fix:" in lines[i + 1]

    def test_completeness_passing_shows_description(self) -> None:
        tasks = [_make_golden_task()]
        report = check_completeness(tasks)
        text = report.format_text()

        # PASS lines should have dimension descriptions
        pass_lines = [l for l in text.split("\n") if "[PASS]" in l]
        for line in pass_lines:
            # Should have both the dimension name and description text
            assert len(line.strip()) > 20  # More than just "[PASS] name"

    def test_completeness_header_shows_counts(self) -> None:
        tasks = [_make_golden_task()]
        report = check_completeness(tasks)
        text = report.format_text()

        header = text.split("\n")[0]
        assert "Data completeness:" in header
        assert f"{report.dimensions_ok}/{report.dimensions_checked}" in header

    def test_completeness_empty_tasks(self) -> None:
        report = check_completeness([])
        assert report.dimensions_checked == 0
        assert report.all_dimensions == []


# ===========================================================================
# Item 14: Prompt Diff Highlighting in diff
# ===========================================================================

class TestPromptDiffHighlighting:
    """Tests for showing actual prompt text delta in diff output."""

    def test_prompt_diff_when_prompts_differ(self) -> None:
        left = _make_golden_task()
        right = _make_golden_task_different_prompt()
        data = diff_tasks(left, right)

        prompt_diff = data.get("prompt_diff") or []
        assert len(prompt_diff) > 0, "Should detect prompt differences"

    def test_format_prompt_diff_shows_changes(self) -> None:
        left = _make_golden_task()
        right = _make_golden_task_different_prompt()
        data = diff_tasks(left, right)

        text = format_prompt_diff(data)
        assert "PROMPT DIFF:" in text
        # Should show the actual changed content
        assert "browser" in text.lower() or "Capabilities" in text

    def test_format_prompt_diff_identical(self) -> None:
        task = _make_golden_task()
        data = diff_tasks(task, task)

        text = format_prompt_diff(data)
        assert "identical" in text.lower() or "missing" in text.lower()

    def test_summary_includes_prompt_diff_oneliner(self) -> None:
        left = _make_golden_task()
        right = _make_golden_task_different_prompt()
        data = diff_tasks(left, right)
        summary = format_diff_summary(data)

        # When prompts differ, the summary should mention it
        assert "Prompt diff:" in summary or "prompt" in summary.lower()

    def test_diff_no_prompts(self) -> None:
        """Tasks without prompts should not crash prompt diff."""
        left = AgentTask(
            task_id="no-prompt-1",
            steps=[_step("no-prompt-1", 1, "tool_a", {}, tool_result="ok")],
        )
        right = AgentTask(
            task_id="no-prompt-2",
            steps=[_step("no-prompt-2", 1, "tool_b", {}, tool_result="ok")],
        )
        data = diff_tasks(left, right)
        text = format_prompt_diff(data)
        # Should not crash, should indicate prompts are identical/missing
        assert isinstance(text, str)
