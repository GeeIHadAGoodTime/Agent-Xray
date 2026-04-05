from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent_xray.analyzer as analyzer_mod
from agent_xray.analyzer import _step_cost, analyze_task, analyze_tasks
from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome, ToolContext


def _step(
    task_id: str,
    step: int,
    tool_name: str = "respond",
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: object | None = None,
    error: object | None = None,
    duration_ms: int | None = None,
    timestamp: str | None = None,
    page_url: str | None = None,
    model: ModelContext | None = None,
    tools: ToolContext | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        duration_ms=duration_ms,
        timestamp=timestamp,
        page_url=page_url,
        model=model,
        tools=tools,
    )


def _outcome(
    task_id: str,
    status: str,
    *,
    final_answer: str | None = None,
    total_duration_s: float | None = None,
    metadata: dict[str, object] | None = None,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        final_answer=final_answer,
        total_duration_s=total_duration_s,
        metadata=dict(metadata or {}),
    )


def test_step_cost_prefers_trace_cost_over_pricing_estimate() -> None:
    step = _step(
        "cost-priority",
        1,
        model=ModelContext(
            model_name="priced-model",
            input_tokens=2_000,
            output_tokens=1_500,
            cost_usd=0.75,
        ),
    )

    assert _step_cost(step, {"models": {"priced-model": {"input": 10.0, "output": 20.0}}}) == 0.75


def test_step_cost_estimates_from_pricing_data_with_cached_tokens() -> None:
    step = _step(
        "cost-estimate",
        1,
        model=ModelContext(
            model_name="priced-model",
            input_tokens=1_000,
            output_tokens=500,
            cache_read_tokens=200,
        ),
    )
    pricing_data = {
        "models": {
            "priced-model": {"input": 2.0, "output": 4.0, "cached_input": 1.0},
        },
        "aliases": {},
    }

    assert _step_cost(step, pricing_data) == pytest.approx(((800 * 2.0) + (200 * 1.0) + (500 * 4.0)) / 1_000_000)


def test_step_cost_returns_zero_for_unknown_model() -> None:
    step = _step(
        "cost-unknown",
        1,
        model=ModelContext(model_name="missing-model", input_tokens=500, output_tokens=500),
    )

    assert _step_cost(step, {"models": {}, "aliases": {}}) == 0.0


def test_analyze_task_counts_no_tools_and_unknown_tool_errors() -> None:
    task = AgentTask(
        task_id="routing-task",
        steps=[
            _step(
                "routing-task",
                1,
                "respond",
                error={"data": "unknown tool requested"},
                tools=ToolContext(tools_available=[]),
            ),
            _step(
                "routing-task",
                2,
                "respond",
                tools=ToolContext(tools_available=[]),
            ),
        ],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.errors == 1
    assert analysis.hallucinated_tools == 1
    assert analysis.no_tools_steps == 2
    assert analysis.error_kinds == {"unknown_tool": 1}


def test_analyze_task_treats_payment_gate_as_completed_without_success_empty_answer_flag() -> None:
    task = AgentTask(
        task_id="payment-gate-task",
        steps=[_step("payment-gate-task", 1, "browser_click", page_url="https://shop.example.test/pay")],
        outcome=_outcome("payment-gate-task", "payment_gate", final_answer="Paid"),
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.task_completed is True
    assert analysis.task_failed is False
    assert analysis.final_answer_empty_but_success is False


def test_analyze_task_uses_step_duration_fallback_when_outcome_duration_missing() -> None:
    task = AgentTask(
        task_id="duration-fallback",
        steps=[
            _step("duration-fallback", 1, duration_ms=120),
            _step("duration-fallback", 2, duration_ms=80),
        ],
        outcome=_outcome("duration-fallback", "completed", total_duration_s=None),
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.total_duration_ms == 200


def test_analyze_task_uses_outcome_duration_when_present() -> None:
    task = AgentTask(
        task_id="duration-outcome",
        steps=[
            _step("duration-outcome", 1, duration_ms=120),
            _step("duration-outcome", 2, duration_ms=80),
        ],
        outcome=_outcome("duration-outcome", "completed", total_duration_s=9.5),
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.total_duration_ms == 9500


def test_analyze_task_normalizes_malformed_tool_payloads_before_detection() -> None:
    step_one = _step("normalize-payloads", 1, "browser_click")
    step_two = _step("normalize-payloads", 2, "browser_click")
    step_one.tool_result = {"data": "Error: access denied"}
    step_two.error = {"data": "validation error: field required"}
    task = AgentTask(
        task_id="normalize-payloads",
        steps=[step_one, step_two],
    )

    analysis = analyze_task(task, detectors=[])

    assert task.steps[0].tool_result == "Error: access denied"
    assert task.steps[1].error == "validation error: field required"
    assert analysis.inline_tool_errors == 1
    assert analysis.errors == 1
    assert analysis.error_kinds == {"validation": 1}


def test_analyze_task_counts_soft_error_from_dict_data_payload() -> None:
    task = AgentTask(
        task_id="soft-dict",
        steps=[_step("soft-dict", 1, "browser_fill_ref", tool_result={"data": "Request timed out after 30s"})],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.soft_errors == 1
    assert analysis.soft_error_kinds == {"soft_timeout": 1}


def test_analyze_task_ignores_nonmatching_malformed_tool_result() -> None:
    task = AgentTask(
        task_id="soft-ignore",
        steps=[_step("soft-ignore", 1, "browser_click", tool_result=["all", "good"])],
    )

    analysis = analyze_task(task, detectors=[])

    assert task.steps[0].tool_result == "['all', 'good']"
    assert analysis.soft_errors == 0
    assert analysis.inline_tool_errors == 0


def test_analyze_task_metrics_flattens_detector_output(monkeypatch: pytest.MonkeyPatch) -> None:
    task = AgentTask(task_id="detector-task", steps=[_step("detector-task", 1)])

    def fake_run_detection(task_arg, detectors_arg):  # type: ignore[no-untyped-def]
        assert task_arg.task_id == "detector-task"
        assert detectors_arg == ["detector"]
        return {"custom_detector": {"custom_flag": True, "custom_count": 2}}

    monkeypatch.setattr(analyzer_mod, "run_detection", fake_run_detection)

    analysis = analyze_task(task, detectors=["detector"])
    metrics = analysis.metrics()

    assert analysis.signal_metrics == {"custom_detector": {"custom_flag": True, "custom_count": 2}}
    assert metrics["custom_detector"] == {"custom_flag": True, "custom_count": 2}
    assert metrics["custom_flag"] is True
    assert metrics["custom_count"] == 2


def test_analyze_tasks_returns_detector_enriched_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_detection(task_arg, detectors_arg):  # type: ignore[no-untyped-def]
        return {"task_detector": {"task_marker": task_arg.task_id, "detector_count": len(detectors_arg)}}

    monkeypatch.setattr(analyzer_mod, "run_detection", fake_run_detection)

    analyses = analyze_tasks(
        [
            AgentTask(task_id="task-a", steps=[_step("task-a", 1)]),
            AgentTask(task_id="task-b", steps=[_step("task-b", 1)]),
        ],
        detectors=[SimpleNamespace(name="x"), SimpleNamespace(name="y")],
    )

    assert set(analyses) == {"task-a", "task-b"}
    assert analyses["task-a"].metrics()["task_marker"] == "task-a"
    assert analyses["task-b"].metrics()["task_marker"] == "task-b"
    assert analyses["task-a"].metrics()["detector_count"] == 2


def test_analyze_task_tracks_numeric_timestamps_and_accelerating_trend() -> None:
    task = AgentTask(
        task_id="timing-task",
        steps=[
            _step("timing-task", 1, duration_ms=300, timestamp="1.0"),
            _step("timing-task", 2, duration_ms=250, timestamp="2.5"),
            _step("timing-task", 3, duration_ms=200, timestamp="5.0"),
            _step("timing-task", 4, duration_ms=90, timestamp="9.5"),
            _step("timing-task", 5, duration_ms=80, timestamp="10.5"),
            _step("timing-task", 6, duration_ms=70, timestamp="14.0"),
        ],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.max_step_gap_ms == 4500
    assert analysis.avg_step_duration_ms == pytest.approx((300 + 250 + 200 + 90 + 80 + 70) / 6)
    assert analysis.step_duration_trend == "accelerating"


def test_analyze_task_flags_empty_success_answer_only_for_success_statuses() -> None:
    completed = analyze_task(
        AgentTask(
            task_id="completed-empty",
            steps=[_step("completed-empty", 1)],
            outcome=_outcome("completed-empty", "completed", final_answer=""),
        ),
        detectors=[],
    )
    failed = analyze_task(
        AgentTask(
            task_id="failed-empty",
            steps=[_step("failed-empty", 1)],
            outcome=_outcome("failed-empty", "failed", final_answer=""),
        ),
        detectors=[],
    )

    assert completed.final_answer_empty_but_success is True
    assert failed.final_answer_empty_but_success is False


def test_analyze_task_empty_steps_defaults_are_stable() -> None:
    analysis = analyze_task(AgentTask(task_id="empty-task"), detectors=[])

    assert analysis.step_count == 0
    assert analysis.unique_urls == []
    assert analysis.unique_tools == []
    assert analysis.total_cost_usd == 0.0
    assert analysis.avg_cost_per_step == 0.0
    assert analysis.metrics()["step_count"] == 0


def test_analyze_task_single_step_preserves_basic_metrics() -> None:
    task = AgentTask(
        task_id="single-step",
        steps=[
            _step(
                "single-step",
                1,
                "browser_navigate",
                {"url": "https://docs.example.test/start"},
                tool_result="Opened docs home.",
                duration_ms=42,
                page_url="https://docs.example.test/start",
                model=ModelContext(input_tokens=11, output_tokens=7, cost_usd=0.02),
            )
        ],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.step_count == 1
    assert analysis.unique_urls == ["https://docs.example.test/start"]
    assert analysis.unique_tools == ["browser_navigate"]
    assert analysis.max_repeat_tool == "browser_navigate"
    assert analysis.max_repeat_count == 1
    assert analysis.tokens_in == 11
    assert analysis.tokens_out == 7
    assert analysis.total_cost_usd == pytest.approx(0.02)
