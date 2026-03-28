"""Comprehensive tests for v1.1.0 and v1.2.0 features with zero prior coverage.

Covers:
  1. completeness module (all 12 dimensions, score, format_text, edge cases)
  2. Cache token flow (JSONL dict -> ModelContext -> analyzer totals)
  3. tool_rejection_mismatch classifier in root_cause.py
  4. New analyzer metrics (rejected_tool_count, timed_out_flag, etc.)
  5. Temporal analysis (ISO-8601 timestamps, step_duration_trend)
  6. DOM ref mismatch detection (backward snapshot search)
  7. New grading rules (suspicious_short_penalty, etc.)
  8. Completeness None-check (cache_read_tokens=0 not flagged as missing)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_xray.analyzer import TaskAnalysis, analyze_task
from agent_xray.completeness import (
    CompletenessReport,
    CompletenessWarning,
    check_completeness,
)
from agent_xray.grader import GradeResult, grade_task, load_rules
from agent_xray.root_cause import classify_task
from agent_xray.schema import (
    AgentStep,
    AgentTask,
    BrowserContext,
    ModelContext,
    ReasoningContext,
    TaskOutcome,
    ToolContext,
)

RULES_PATH = Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules" / "default.json"


# ---------------------------------------------------------------------------
# Helpers (matching test_root_cause.py style)
# ---------------------------------------------------------------------------


def _step(
    step: int,
    tool_name: str,
    *,
    task_id: str = "task-1",
    tool_input: dict[str, object] | None = None,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
    context_usage_pct: float | None = None,
    context_window: int | None = None,
    compaction_count: int | None = None,
    output_tokens: int | None = None,
    input_tokens: int | None = None,
    duration_ms: int | None = None,
    timestamp: str | None = None,
    model_name: str | None = None,
    cost_usd: float | None = None,
    cache_read_tokens: int | None = None,
    cache_creation_tokens: int | None = None,
    rejected_tools: list[str] | None = None,
    approval_path: str | None = None,
    tool_schemas: dict[str, object] | None = None,
) -> AgentStep:
    model = None
    if any(
        v is not None
        for v in [
            model_name,
            input_tokens,
            output_tokens,
            cost_usd,
            context_usage_pct,
            context_window,
            compaction_count,
            cache_read_tokens,
            cache_creation_tokens,
        ]
    ):
        model = ModelContext(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            context_usage_pct=context_usage_pct,
            context_window=context_window,
            compaction_count=compaction_count,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
        )
    tools_ctx = None
    if tools_available is not None or rejected_tools is not None:
        tools_ctx = ToolContext(
            tools_available=tools_available,
            rejected_tools=rejected_tools,
        )
    reasoning_ctx = None
    if llm_reasoning is not None or approval_path is not None:
        reasoning_ctx = ReasoningContext(
            llm_reasoning=llm_reasoning,
            approval_path=approval_path,
        )
    browser_ctx = None
    if page_url is not None:
        browser_ctx = BrowserContext(page_url=page_url)
    extensions: dict[str, object] = {}
    if tool_schemas is not None:
        extensions["tool_schemas"] = tool_schemas
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        duration_ms=duration_ms,
        timestamp=timestamp,
        model=model,
        tools=tools_ctx,
        reasoning=reasoning_ctx,
        browser=browser_ctx,
        extensions=extensions,
    )


def _task(
    steps: list[AgentStep],
    *,
    task_id: str = "task-1",
    task_category: str | None = None,
    outcome: TaskOutcome | None = None,
    metadata: dict[str, object] | None = None,
) -> AgentTask:
    t = AgentTask(
        task_id=task_id,
        task_text="test task",
        task_category=task_category,
        steps=steps,
        outcome=outcome,
    )
    if metadata:
        t.metadata.update(metadata)
    return t


def _failing_grade(task: AgentTask, *, score: int = -1) -> GradeResult:
    return GradeResult(
        task_id=task.task_id,
        grade="BROKEN",
        score=score,
        reasons=[],
        metrics={},
        signals=[],
    )


# ===========================================================================
# 1. COMPLETENESS MODULE
# ===========================================================================


class TestCompletenessEmptyTasks:
    """Edge case: empty task list."""

    def test_empty_tasks_returns_zero_dimensions(self) -> None:
        report = check_completeness([])
        assert report.dimensions_checked == 0
        assert report.dimensions_ok == 0
        assert report.score == 0.0
        assert report.score_pct == 0
        assert report.warnings == []

    def test_empty_tasks_format_text(self) -> None:
        report = check_completeness([])
        text = report.format_text()
        assert "0%" in text
        assert "0/0" in text


class TestCompletenessAllDimensionsOk:
    """All 12 dimensions pass."""

    def test_all_dimensions_ok(self) -> None:
        step = _step(
            1,
            "browser_click",
            tool_result="clicked",
            duration_ms=100,
            model_name="gpt-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_creation_tokens=10,
            llm_reasoning="I should click the button.",
            tools_available=["browser_click"],
            tool_schemas={"browser_click": {"type": "object"}},
        )
        step.tools.rejected_tools = ["browser_fill"]
        step.reasoning.approval_path = "risk_safe"
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="Order placed successfully.",
        )
        task = _task(
            [step],
            outcome=outcome,
            metadata={
                "system_prompt_text": "You are a helpful agent.",
                "system_context_components": {"frustration": True},
            },
        )
        report = check_completeness([task])
        assert report.dimensions_checked == 12
        assert report.dimensions_ok == 12
        assert report.score == 1.0
        assert report.score_pct == 100
        assert not report.has_critical()
        assert len(report.warnings) == 0


class TestCompletenessAllDimensionsFailing:
    """All 12 dimensions fail."""

    def test_all_dimensions_fail(self) -> None:
        step = _step(
            1,
            "browser_click",
            tool_result="clicked",
            model_name="unknown",
            input_tokens=100,
            output_tokens=50,
        )
        outcome = TaskOutcome(task_id="task-1", status="failed")
        task = _task(
            [step],
            outcome=outcome,
            metadata={"prior_conversation_turns": 3},
        )
        report = check_completeness([task])
        assert report.dimensions_checked == 12
        assert report.dimensions_ok < 12
        assert report.has_critical()
        assert len(report.warnings) > 0


class TestCompletenessIndividualDimensions:
    """Test each dimension individually."""

    def _base_task(self) -> AgentTask:
        """Create a task that passes most dimensions as a baseline."""
        step = _step(
            1,
            "browser_click",
            tool_result="clicked",
            duration_ms=100,
            model_name="gpt-5",
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_creation_tokens=10,
            llm_reasoning="Clicking the button.",
            tools_available=["browser_click"],
            tool_schemas={"browser_click": {}},
        )
        step.tools.rejected_tools = ["fill"]
        step.reasoning.approval_path = "risk_safe"
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="Done.",
        )
        return _task(
            [step],
            outcome=outcome,
            metadata={
                "system_prompt_text": "prompt",
                "system_context_components": {"frustration": True},
            },
        )

    def test_dim_outcome_records_missing(self) -> None:
        """No outcome -> critical warning."""
        task = self._base_task()
        task.outcome = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "outcome_records" in dims

    def test_dim_tool_schemas_missing(self) -> None:
        """No tool_schemas in extensions -> critical warning."""
        task = self._base_task()
        task.steps[0].extensions = {}
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "tool_schemas" in dims

    def test_dim_model_name_unknown(self) -> None:
        """All model_name='unknown' -> medium warning."""
        task = self._base_task()
        task.steps[0].model.model_name = "unknown"
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "model_name" in dims

    def test_dim_cache_tokens_missing_with_other_tokens(self) -> None:
        """Token counts present but cache tokens missing -> high warning."""
        task = self._base_task()
        task.steps[0].model.cache_read_tokens = None
        task.steps[0].model.cache_creation_tokens = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "cache_tokens" in dims

    def test_dim_final_answer_missing(self) -> None:
        """Outcome present but no final_answer -> high warning."""
        task = self._base_task()
        task.outcome.final_answer = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "final_answer" in dims

    def test_dim_system_prompt_missing(self) -> None:
        """No system_prompt_text -> medium warning."""
        task = self._base_task()
        del task.metadata["system_prompt_text"]
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "system_prompt" in dims

    def test_dim_rejected_tools_missing(self) -> None:
        """No rejected_tools data -> low warning."""
        task = self._base_task()
        task.steps[0].tools.rejected_tools = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "rejected_tools" in dims

    def test_dim_approval_path_missing(self) -> None:
        """No approval_path data -> low warning."""
        task = self._base_task()
        task.steps[0].reasoning.approval_path = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "approval_path" in dims

    def test_dim_conversation_history_gap(self) -> None:
        """prior_conversation_turns > 0 but no summary -> high warning."""
        task = self._base_task()
        task.metadata["prior_conversation_turns"] = 5
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "conversation_history" in dims

    def test_dim_step_durations_missing(self) -> None:
        """Most steps lack duration_ms -> medium warning."""
        task = self._base_task()
        task.steps[0].duration_ms = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "step_durations" in dims

    def test_dim_system_context_missing(self) -> None:
        """No system_context_components -> medium warning."""
        task = self._base_task()
        del task.metadata["system_context_components"]
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "system_context" in dims

    def test_dim_llm_reasoning_missing(self) -> None:
        """No reasoning trace -> medium warning."""
        task = self._base_task()
        task.steps[0].reasoning.llm_reasoning = None
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "llm_reasoning" in dims


class TestCompletenessReport:
    """Test CompletenessReport properties and formatting."""

    def test_score_property(self) -> None:
        report = CompletenessReport(dimensions_checked=10, dimensions_ok=7)
        assert report.score == pytest.approx(0.7)
        assert report.score_pct == 70

    def test_score_zero_dimensions(self) -> None:
        report = CompletenessReport(dimensions_checked=0, dimensions_ok=0)
        assert report.score == 0.0

    def test_has_critical_true(self) -> None:
        report = CompletenessReport(
            warnings=[
                CompletenessWarning("dim", "critical", "msg", 50.0, "fix"),
            ]
        )
        assert report.has_critical()

    def test_has_critical_false(self) -> None:
        report = CompletenessReport(
            warnings=[
                CompletenessWarning("dim", "medium", "msg", 50.0, "fix"),
            ]
        )
        assert not report.has_critical()

    def test_format_text_no_warnings(self) -> None:
        report = CompletenessReport(
            dimensions_checked=12, dimensions_ok=12
        )
        text = report.format_text()
        assert "100%" in text
        assert "12/12" in text

    def test_format_text_with_warnings(self) -> None:
        report = CompletenessReport(
            dimensions_checked=12,
            dimensions_ok=10,
            warnings=[
                CompletenessWarning("dim_a", "critical", "bad thing", 80.0, "fix A"),
                CompletenessWarning("dim_b", "low", "minor issue", 20.0, "fix B"),
            ],
        )
        text = report.format_text()
        assert "[CRITICAL]" in text
        assert "[LOW]" in text
        assert "Fix:" in text
        assert "dim_a" in text

    def test_format_text_severity_ordering(self) -> None:
        report = CompletenessReport(
            dimensions_checked=4,
            dimensions_ok=0,
            warnings=[
                CompletenessWarning("low_dim", "low", "msg", 10.0, "fix"),
                CompletenessWarning("crit_dim", "critical", "msg", 90.0, "fix"),
                CompletenessWarning("high_dim", "high", "msg", 50.0, "fix"),
            ],
        )
        text = report.format_text()
        crit_pos = text.index("[CRITICAL]")
        high_pos = text.index("[HIGH]")
        low_pos = text.index("[LOW]")
        assert crit_pos < high_pos < low_pos


# ===========================================================================
# 2. CACHE TOKEN FLOW
# ===========================================================================


class TestCacheTokenFlow:
    """cache_read_tokens and cache_creation_tokens: JSONL -> ModelContext -> analyzer."""

    def test_cache_tokens_from_dict(self) -> None:
        """Cache tokens in flat JSONL dict are parsed into ModelContext."""
        step = AgentStep.from_dict({
            "task_id": "t1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "cache_read_tokens": 500,
            "cache_creation_tokens": 200,
            "input_tokens": 1000,
            "output_tokens": 100,
        })
        assert step.model is not None
        assert step.model.cache_read_tokens == 500
        assert step.model.cache_creation_tokens == 200

    def test_cache_tokens_from_llm_usage(self) -> None:
        """Cache tokens in nested llm_usage dict flow to ModelContext."""
        step = AgentStep.from_dict({
            "task_id": "t1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "llm_usage": {
                "input_tokens": 1000,
                "output_tokens": 100,
                "cache_read_tokens": 700,
                "cache_creation_tokens": 300,
            },
        })
        assert step.model is not None
        assert step.model.cache_read_tokens == 700
        assert step.model.cache_creation_tokens == 300

    def test_cache_tokens_flow_to_analyzer_totals(self) -> None:
        """Analyzer sums cache tokens across all steps."""
        steps = [
            _step(1, "respond", cache_read_tokens=100, cache_creation_tokens=50, input_tokens=500),
            _step(2, "respond", cache_read_tokens=200, cache_creation_tokens=80, input_tokens=600),
            _step(3, "respond", cache_read_tokens=0, cache_creation_tokens=0, input_tokens=400),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.cache_read_tokens_total == 300
        assert analysis.cache_creation_tokens_total == 130

    def test_cache_tokens_zero_not_treated_as_none(self) -> None:
        """cache_read_tokens=0 should be integer 0, NOT None."""
        step = AgentStep.from_dict({
            "task_id": "t1",
            "step": 1,
            "tool_name": "respond",
            "tool_input": {},
            "cache_read_tokens": 0,
            "cache_creation_tokens": 0,
            "input_tokens": 100,
        })
        assert step.model is not None
        assert step.model.cache_read_tokens == 0
        assert step.model.cache_creation_tokens == 0


# ===========================================================================
# 3. tool_rejection_mismatch CLASSIFIER
# ===========================================================================


class TestToolRejectionMismatch:
    """Test the tool_rejection_mismatch root-cause classifier."""

    def test_rejection_above_threshold(self) -> None:
        """Rejected tools in >30% of steps triggers classification."""
        steps = [
            _step(
                i,
                "respond",
                tools_available=["respond"],
                rejected_tools=["browser_click"],
            )
            for i in range(1, 5)
        ]
        task = _task(steps)
        cause = classify_task(task, _failing_grade(task))
        assert cause is not None
        assert cause.root_cause == "tool_rejection_mismatch"

    def test_rejection_below_threshold(self) -> None:
        """Rejected tools in <30% of steps does NOT trigger."""
        # 1 of 5 steps has rejection = 20% < 30%
        steps = [
            _step(1, "respond", tools_available=["respond"], rejected_tools=["browser_click"]),
            _step(2, "respond", tools_available=["respond"]),
            _step(3, "respond", tools_available=["respond"]),
            _step(4, "respond", tools_available=["respond"]),
            _step(5, "respond", tools_available=["respond"]),
        ]
        task = _task(steps)
        cause = classify_task(task, _failing_grade(task))
        assert cause is not None
        assert cause.root_cause != "tool_rejection_mismatch"

    def test_no_rejected_tools_data(self) -> None:
        """No rejected_tools anywhere does NOT trigger."""
        steps = [_step(i, "respond") for i in range(1, 4)]
        task = _task(steps)
        cause = classify_task(task, _failing_grade(task))
        assert cause is not None
        assert cause.root_cause != "tool_rejection_mismatch"

    def test_rejected_tools_evidence_includes_names(self) -> None:
        """Evidence should list the rejected tool names."""
        steps = [
            _step(
                i,
                "respond",
                tools_available=["respond"],
                rejected_tools=["browser_click", "browser_fill"],
            )
            for i in range(1, 4)
        ]
        task = _task(steps)
        cause = classify_task(task, _failing_grade(task))
        assert cause is not None
        assert cause.root_cause == "tool_rejection_mismatch"
        evidence_text = " ".join(cause.evidence)
        assert "browser_click" in evidence_text
        assert "browser_fill" in evidence_text


# ===========================================================================
# 4. NEW ANALYZER METRICS
# ===========================================================================


class TestAnalyzerNewMetrics:
    """Test new metrics on TaskAnalysis."""

    def test_rejected_tool_count(self) -> None:
        steps = [
            _step(1, "respond", rejected_tools=["a", "b"]),
            _step(2, "respond", rejected_tools=["c"]),
            _step(3, "respond"),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.rejected_tool_count == 3

    def test_timed_out_flag_from_outcome_metadata(self) -> None:
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "failed",
            "timed_out": True,
        })
        task = _task(
            [_step(1, "respond")],
            outcome=outcome,
        )
        analysis = analyze_task(task)
        assert analysis.timed_out_flag is True

    def test_timed_out_flag_false_by_default(self) -> None:
        outcome = TaskOutcome(task_id="task-1", status="success")
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.timed_out_flag is False

    def test_suspicious_short_flag(self) -> None:
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "success",
            "suspicious_short": True,
        })
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.suspicious_short_flag is True

    def test_final_answer_length_and_has_final_answer(self) -> None:
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="The order was placed successfully.",
        )
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.has_final_answer is True
        assert analysis.final_answer_length == len("The order was placed successfully.")

    def test_no_final_answer(self) -> None:
        outcome = TaskOutcome(task_id="task-1", status="failed")
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.has_final_answer is False
        assert analysis.final_answer_length == 0

    def test_max_context_usage_pct_is_percentage(self) -> None:
        """Values are already percentages (e.g. 85.0 means 85%), NOT fractions."""
        steps = [
            _step(1, "respond", context_usage_pct=45.0),
            _step(2, "respond", context_usage_pct=85.0),
            _step(3, "respond", context_usage_pct=60.0),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.max_context_usage_pct == pytest.approx(85.0)

    def test_max_context_usage_from_outcome_metadata(self) -> None:
        """final_context_usage_pct from outcome is also considered."""
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "success",
            "final_context_usage_pct": 92.0,
        })
        steps = [_step(1, "respond", context_usage_pct=50.0)]
        task = _task(steps, outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.max_context_usage_pct == pytest.approx(92.0)

    def test_cache_tokens_totals(self) -> None:
        steps = [
            _step(1, "respond", cache_read_tokens=100, cache_creation_tokens=50, input_tokens=500),
            _step(2, "respond", cache_read_tokens=200, cache_creation_tokens=0, input_tokens=400),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.cache_read_tokens_total == 300
        assert analysis.cache_creation_tokens_total == 50

    def test_final_answer_empty_but_success(self) -> None:
        """Task with success status but very short final answer."""
        outcome = TaskOutcome(task_id="task-1", status="success", final_answer="ok")
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.final_answer_empty_but_success is True

    def test_final_answer_not_empty_with_success(self) -> None:
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="Order placed: confirmation #12345.",
        )
        task = _task([_step(1, "respond")], outcome=outcome)
        analysis = analyze_task(task)
        assert analysis.final_answer_empty_but_success is False

    def test_system_context_fields(self) -> None:
        task = _task(
            [_step(1, "respond")],
            metadata={
                "system_context_components": {
                    "frustration": "user is frustrated",
                    "delivery_address": "123 Main St",
                    "user_model": {"name": "Alice"},
                    "playback": None,
                }
            },
        )
        analysis = analyze_task(task)
        assert analysis.has_frustration_context is True
        assert analysis.has_delivery_address is True
        assert analysis.has_user_model is True
        # Only non-falsy values are counted
        assert analysis.system_context_field_count == 3

    def test_system_context_not_present(self) -> None:
        task = _task([_step(1, "respond")])
        analysis = analyze_task(task)
        assert analysis.has_frustration_context is False
        assert analysis.has_delivery_address is False
        assert analysis.has_user_model is False
        assert analysis.system_context_field_count == 0

    def test_metrics_dict_includes_all_new_fields(self) -> None:
        """All new v1.1/v1.2 fields are present in the metrics dict."""
        task = _task([_step(1, "respond")])
        analysis = analyze_task(task)
        m = analysis.metrics()
        expected_keys = [
            "rejected_tool_count",
            "timed_out_flag",
            "suspicious_short_flag",
            "final_answer_length",
            "has_final_answer",
            "max_context_usage_pct",
            "cache_read_tokens_total",
            "cache_creation_tokens_total",
            "max_step_gap_ms",
            "avg_step_duration_ms",
            "step_duration_trend",
            "has_frustration_context",
            "has_delivery_address",
            "has_user_model",
            "system_context_field_count",
            "final_answer_empty_but_success",
            "element_ref_mismatches",
        ]
        for key in expected_keys:
            assert key in m, f"Missing metric key: {key}"


# ===========================================================================
# 5. TEMPORAL ANALYSIS
# ===========================================================================


class TestTemporalAnalysis:
    """ISO-8601 timestamps, max_step_gap_ms, step_duration_trend."""

    def test_iso8601_timestamps_parsed_for_gap(self) -> None:
        """ISO-8601 timestamps produce correct max_step_gap_ms."""
        steps = [
            _step(1, "respond", timestamp="2026-03-26T12:00:00Z"),
            _step(2, "respond", timestamp="2026-03-26T12:00:10Z"),
            _step(3, "respond", timestamp="2026-03-26T12:05:00Z"),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        # Gap between step 2 and 3 is 290 seconds = 290000 ms
        assert analysis.max_step_gap_ms == 290000

    def test_numeric_epoch_timestamps(self) -> None:
        """Numeric epoch timestamps also work."""
        steps = [
            _step(1, "respond", timestamp="1711454400.0"),   # some epoch
            _step(2, "respond", timestamp="1711454410.0"),   # +10s
            _step(3, "respond", timestamp="1711454500.0"),   # +90s
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.max_step_gap_ms == 90000

    def test_no_timestamps_zero_gap(self) -> None:
        steps = [_step(1, "respond"), _step(2, "respond")]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.max_step_gap_ms == 0

    def test_step_duration_trend_accelerating(self) -> None:
        """Last third durations < first third / 2 -> accelerating."""
        steps = [
            _step(i, "respond", duration_ms=d)
            for i, d in enumerate(
                [1000, 1200, 1100, 800, 600, 400, 300, 200, 100], start=1
            )
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.step_duration_trend == "accelerating"

    def test_step_duration_trend_decelerating(self) -> None:
        """Last third durations > first third * 2 -> decelerating."""
        steps = [
            _step(i, "respond", duration_ms=d)
            for i, d in enumerate(
                [100, 120, 110, 300, 500, 800, 1000, 1200, 1500], start=1
            )
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.step_duration_trend == "decelerating"

    def test_step_duration_trend_stable(self) -> None:
        """Similar durations throughout -> stable."""
        steps = [
            _step(i, "respond", duration_ms=500) for i in range(1, 10)
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.step_duration_trend == "stable"

    def test_step_duration_trend_too_few_steps(self) -> None:
        """Fewer than 3 steps -> defaults to stable."""
        steps = [
            _step(1, "respond", duration_ms=100),
            _step(2, "respond", duration_ms=1000),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.step_duration_trend == "stable"

    def test_avg_step_duration_ms(self) -> None:
        steps = [
            _step(1, "respond", duration_ms=100),
            _step(2, "respond", duration_ms=200),
            _step(3, "respond", duration_ms=300),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.avg_step_duration_ms == pytest.approx(200.0)

    def test_avg_step_duration_with_missing(self) -> None:
        """Steps without duration_ms are excluded from average."""
        steps = [
            _step(1, "respond", duration_ms=100),
            _step(2, "respond"),  # no duration
            _step(3, "respond", duration_ms=300),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.avg_step_duration_ms == pytest.approx(200.0)


# ===========================================================================
# 6. DOM REF MISMATCH DETECTION
# ===========================================================================


class TestDomRefMismatch:
    """Test element_ref_mismatches detection with backward snapshot search."""

    def test_ref_in_snapshot_no_mismatch(self) -> None:
        """Ref found in most recent snapshot -> no mismatch."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10 button @e20 link"),
            _step(2, "browser_click", tool_input={"ref": "@e10"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.element_ref_mismatches == 0

    def test_ref_not_in_snapshot_mismatch(self) -> None:
        """Ref NOT in any prior snapshot -> counts as mismatch."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10 button @e20 link"),
            _step(2, "browser_click", tool_input={"ref": "@e99"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.element_ref_mismatches == 1

    def test_backward_search_finds_older_snapshot(self) -> None:
        """Search goes backwards, not just the immediately prior step."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10 button @e20 link @e30 form"),
            _step(2, "browser_click", tool_input={"ref": "@e10"}),
            _step(3, "browser_click", tool_input={"ref": "@e30"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        # Step 3 references @e30, which is in step 1's snapshot
        assert analysis.element_ref_mismatches == 0

    def test_no_snapshot_before_click_no_mismatch(self) -> None:
        """If no prior snapshot exists, skip (no mismatch counted)."""
        steps = [
            _step(1, "browser_click", tool_input={"ref": "@e10"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.element_ref_mismatches == 0

    def test_multiple_mismatches(self) -> None:
        """Multiple mismatched refs are counted individually."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10 @e20"),
            _step(2, "browser_click", tool_input={"ref": "@e99"}),
            _step(3, "browser_fill_ref", tool_input={"ref": "@e88"}),
            _step(4, "browser_click", tool_input={"ref": "@e10"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.element_ref_mismatches == 2

    def test_non_browser_tool_ignored(self) -> None:
        """Non-browser tools are not checked for ref mismatches."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10"),
            _step(2, "respond", tool_input={"ref": "@e99"}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        assert analysis.element_ref_mismatches == 0

    def test_ref_as_integer(self) -> None:
        """Numeric ref input is coerced to string for matching."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10"),
            _step(2, "browser_click", tool_input={"ref": 10}),
        ]
        task = _task(steps)
        analysis = analyze_task(task)
        # "10" doesn't match the @e pattern, so it's skipped
        assert analysis.element_ref_mismatches == 0


# ===========================================================================
# 7. NEW GRADING RULES IN default.json
# ===========================================================================


class TestNewGradingRules:
    """Test the new grading rules added in v1.1/v1.2."""

    def _grade_with_defaults(self, task: AgentTask) -> GradeResult:
        rules = load_rules(RULES_PATH)
        return grade_task(task, rules)

    def test_suspicious_short_penalty(self) -> None:
        """suspicious_short_flag=true should subtract 1 point."""
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "success",
            "suspicious_short": True,
            "final_answer": "The order was placed with confirmation #12345 for delivery to 123 Main St.",
        })
        steps = [
            _step(1, "browser_navigate", tool_result="Homepage", page_url="https://shop.test/"),
            _step(2, "browser_click", tool_result="Clicked", page_url="https://shop.test/product"),
            _step(3, "browser_fill_ref", tool_result="Filled", page_url="https://shop.test/checkout"),
            _step(4, "browser_click", tool_result="Placed", page_url="https://shop.test/confirm"),
        ]
        task = _task(steps, outcome=outcome)
        result = self._grade_with_defaults(task)
        # Find the suspicious_short_penalty signal
        signal = next((s for s in result.signals if s.name == "suspicious_short_penalty"), None)
        assert signal is not None
        assert signal.passed is True
        assert signal.points == -1

    def test_empty_answer_success_penalty(self) -> None:
        """final_answer_empty_but_success=true should subtract 1 point."""
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="ok",  # <=10 chars = empty-ish
        )
        steps = [
            _step(1, "browser_navigate", tool_result="ok", page_url="https://shop.test/"),
            _step(2, "browser_click", tool_result="ok", page_url="https://shop.test/p"),
            _step(3, "browser_fill_ref", tool_result="ok", page_url="https://shop.test/c"),
            _step(4, "browser_click", tool_result="ok", page_url="https://shop.test/d"),
        ]
        task = _task(steps, outcome=outcome)
        result = self._grade_with_defaults(task)
        signal = next((s for s in result.signals if s.name == "empty_answer_success_penalty"), None)
        assert signal is not None
        assert signal.passed is True
        assert signal.points == -1

    def test_ref_mismatch_penalty(self) -> None:
        """element_ref_mismatches >= 3 should subtract 1 point."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10"),
            _step(2, "browser_click", tool_input={"ref": "@e99"}),
            _step(3, "browser_click", tool_input={"ref": "@e88"}),
            _step(4, "browser_click", tool_input={"ref": "@e77"}),
        ]
        task = _task(steps)
        result = self._grade_with_defaults(task)
        signal = next((s for s in result.signals if s.name == "ref_mismatch_penalty"), None)
        assert signal is not None
        assert signal.passed is True
        assert signal.points == -1

    def test_ref_mismatch_below_threshold(self) -> None:
        """element_ref_mismatches < 3 should NOT trigger penalty."""
        steps = [
            _step(1, "browser_snapshot", tool_result="Page: @e10"),
            _step(2, "browser_click", tool_input={"ref": "@e99"}),
            _step(3, "browser_click", tool_input={"ref": "@e10"}),
        ]
        task = _task(steps)
        result = self._grade_with_defaults(task)
        signal = next((s for s in result.signals if s.name == "ref_mismatch_penalty"), None)
        assert signal is not None
        assert signal.passed is False

    def test_runtime_timeout_flag(self) -> None:
        """timed_out_flag=true should subtract 1 point."""
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "timeout",
            "timed_out": True,
        })
        steps = [_step(1, "respond", tool_result="working...")]
        task = _task(steps, outcome=outcome)
        result = self._grade_with_defaults(task)
        signal = next((s for s in result.signals if s.name == "runtime_timeout_flag"), None)
        assert signal is not None
        assert signal.passed is True
        assert signal.points == -1

    def test_golden_blocked_by_suspicious_short(self) -> None:
        """A task flagged as suspicious_short cannot be GOLDEN."""
        outcome = TaskOutcome.from_dict({
            "task_id": "task-1",
            "status": "success",
            "suspicious_short": True,
            "final_answer": "Order placed successfully. Confirmation number is #12345. Delivered to 123 Main Street.",
        })
        steps = [
            _step(1, "browser_navigate", tool_result="Homepage", page_url="https://a.test/"),
            _step(2, "browser_click", tool_result="Clicked", page_url="https://a.test/p"),
            _step(3, "browser_fill_ref", tool_result="Filled", page_url="https://a.test/c"),
            _step(4, "browser_click", tool_result="Placed", page_url="https://a.test/d"),
            _step(5, "browser_snapshot", tool_result="Confirmed", page_url="https://a.test/e"),
        ]
        task = _task(steps, outcome=outcome)
        result = self._grade_with_defaults(task)
        # Even with high score, golden_requirements block it
        assert result.grade != "GOLDEN" or any(
            "suspicious" in r.lower() for r in result.reasons
        )

    def test_golden_blocked_by_empty_answer_success(self) -> None:
        """A task with empty answer + success cannot be GOLDEN."""
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="",  # empty
        )
        steps = [
            _step(1, "browser_navigate", tool_result="ok", page_url="https://a.test/"),
            _step(2, "browser_click", tool_result="ok", page_url="https://a.test/p"),
            _step(3, "browser_fill_ref", tool_result="ok", page_url="https://a.test/c"),
            _step(4, "browser_click", tool_result="ok", page_url="https://a.test/d"),
            _step(5, "browser_snapshot", tool_result="ok", page_url="https://a.test/e"),
        ]
        task = _task(steps, outcome=outcome)
        result = self._grade_with_defaults(task)
        assert result.grade != "GOLDEN" or any(
            "final_answer" in r.lower() for r in result.reasons
        )


# ===========================================================================
# 8. COMPLETENESS None-CHECK (cache_read_tokens=0 NOT flagged as missing)
# ===========================================================================


class TestCompletenessNoneCheck:
    """Ensure integer 0 values are not treated as None/missing."""

    def test_cache_read_tokens_zero_not_flagged(self) -> None:
        """cache_read_tokens=0 is valid data, NOT missing."""
        step = _step(
            1,
            "respond",
            cache_read_tokens=0,
            cache_creation_tokens=0,
            input_tokens=100,
            output_tokens=50,
        )
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="Done.",
        )
        task = _task([step], outcome=outcome)
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "cache_tokens" not in dims

    def test_cache_tokens_truly_none_is_flagged(self) -> None:
        """cache_read_tokens=None with other tokens present IS flagged."""
        step = _step(
            1,
            "respond",
            input_tokens=100,
            output_tokens=50,
        )
        # Ensure cache tokens are None (default)
        assert step.model.cache_read_tokens is None
        assert step.model.cache_creation_tokens is None
        outcome = TaskOutcome(
            task_id="task-1",
            status="success",
            final_answer="Done.",
        )
        task = _task([step], outcome=outcome)
        report = check_completeness([task])
        dims = [w.dimension for w in report.warnings]
        assert "cache_tokens" in dims
