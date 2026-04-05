from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_xray.analyzer import (
    TaskAnalysis,
    analyze_task,
    analyze_tasks,
    build_task_tree,
    load_tasks,
    resolve_task,
    summarize_tool_result,
    tasks_from_steps,
)
from agent_xray.schema import AgentStep, AgentTask, ModelContext, TaskOutcome, ToolContext


def _step(
    task_id: str,
    step: int,
    tool_name: str,
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
        browser=None if page_url is None else None,
        page_url=page_url,
        model=model,
        tools=tools,
    )


def _outcome(
    task_id: str,
    status: str,
    *,
    final_answer: str | None = None,
    total_steps: int | None = None,
    total_duration_s: float | None = None,
    timestamp: str = "2026-04-05T12:00:00+00:00",
    metadata: dict[str, object] | None = None,
) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        final_answer=final_answer,
        total_steps=total_steps,
        total_duration_s=total_duration_s,
        timestamp=timestamp,
        metadata=dict(metadata or {}),
    )


def _write_jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def test_analyze_task_computes_primary_metrics_and_tokens() -> None:
    task = AgentTask(
        task_id="checkout-1",
        task_text="https://shop.example.test",
        steps=[
            _step(
                "checkout-1",
                1,
                "browser_navigate",
                {"url": "https://shop.example.test"},
                tool_result="Homepage",
                duration_ms=1200,
                timestamp="2026-04-05T12:00:00+00:00",
                page_url="https://shop.example.test/",
                model=ModelContext(input_tokens=120, output_tokens=30, cost_usd=0.01),
            ),
            _step(
                "checkout-1",
                2,
                "browser_click",
                {"ref": "@e1"},
                tool_result="Cart page",
                duration_ms=800,
                timestamp="2026-04-05T12:00:03+00:00",
                page_url="https://shop.example.test/cart",
                model=ModelContext(input_tokens=90, output_tokens=15, cost_usd=0.02),
            ),
            _step(
                "checkout-1",
                3,
                "browser_click",
                {"ref": "@e2"},
                error="Validation error: field required",
                duration_ms=600,
                timestamp="2026-04-05T12:00:08+00:00",
                page_url="https://shop.example.test/checkout",
            ),
        ],
        outcome=_outcome(
            "checkout-1",
            "success",
            final_answer="Order placed.",
            total_steps=3,
            total_duration_s=9.5,
        ),
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.step_count == 3
    assert analysis.unique_urls == [
        "https://shop.example.test/",
        "https://shop.example.test/cart",
        "https://shop.example.test/checkout",
    ]
    assert analysis.unique_url_paths == analysis.unique_urls
    assert analysis.unique_tools == ["browser_click", "browser_navigate"]
    assert analysis.tool_sequence == ["browser_navigate", "browser_click", "browser_click"]
    assert analysis.max_repeat_tool == "browser_click"
    assert analysis.max_repeat_count == 2
    assert analysis.errors == 1
    assert analysis.error_kinds == {"validation": 1}
    assert analysis.site_name  # site extraction is heuristic; verify non-empty
    assert analysis.final_url == "https://shop.example.test/checkout"
    assert analysis.total_duration_ms == 9500
    assert analysis.total_cost_usd == pytest.approx(0.03)
    assert analysis.avg_cost_per_step == pytest.approx(0.01)
    assert analysis.tokens_in == 210
    assert analysis.tokens_out == 45


def test_analyze_task_combines_trace_cost_and_pricing_estimation() -> None:
    pricing_data = {
        "models": {
            "x-model": {"input": 2.0, "output": 4.0, "cached_input": 1.0},
        },
        "aliases": {},
    }
    task = AgentTask(
        task_id="cost-task",
        steps=[
            _step(
                "cost-task",
                1,
                "respond",
                model=ModelContext(cost_usd=0.25),
            ),
            _step(
                "cost-task",
                2,
                "respond",
                model=ModelContext(
                    model_name="x-model",
                    input_tokens=1_000,
                    output_tokens=2_000,
                    cache_read_tokens=100,
                    cost_usd=None,
                ),
            ),
        ],
    )

    analysis = analyze_task(task, detectors=[], pricing_data=pricing_data)
    expected_step_two = ((900 * 2.0) + (100 * 1.0) + (2_000 * 4.0)) / 1_000_000

    assert analysis.total_cost_usd == pytest.approx(0.25 + expected_step_two)
    assert analysis.avg_cost_per_step == pytest.approx((0.25 + expected_step_two) / 2)


def test_analyze_task_collects_context_cache_and_outcome_flags() -> None:
    task = AgentTask(
        task_id="context-task",
        metadata={
            "system_context_components": {
                "frustration": "user is frustrated",
                "delivery_address": "123 Main St",
                "user_model": {"tier": "pro"},
                "empty_field": "",
            }
        },
        steps=[
            _step(
                "context-task",
                1,
                "respond",
                model=ModelContext(
                    context_usage_pct=0.45,
                    cache_read_tokens=50,
                    cache_creation_tokens=12,
                ),
                tools=ToolContext(rejected_tools=["web_search", "browser_click"]),
            ),
            _step(
                "context-task",
                2,
                "respond",
                model=ModelContext(
                    context_usage_pct=0.91,
                    cache_read_tokens=5,
                    cache_creation_tokens=3,
                ),
                tools=ToolContext(rejected_tools=["shell"]),
            ),
        ],
        outcome=_outcome(
            "context-task",
            "completed",
            final_answer="Done.",
            metadata={
                "timed_out": True,
                "suspicious_short": True,
                "final_context_usage_pct": 0.97,
            },
        ),
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.rejected_tool_count == 3
    assert analysis.timed_out_flag is True
    assert analysis.suspicious_short_flag is True
    assert analysis.max_context_usage_pct == pytest.approx(0.97)
    assert analysis.cache_read_tokens_total == 55
    assert analysis.cache_creation_tokens_total == 15
    assert analysis.timeout_like is True
    assert analysis.has_frustration_context is True
    assert analysis.has_delivery_address is True
    assert analysis.has_user_model is True
    assert analysis.system_context_field_count == 3


def test_analyze_task_computes_temporal_metrics_and_decelerating_trend() -> None:
    task = AgentTask(
        task_id="temporal-task",
        steps=[
            _step(
                "temporal-task",
                1,
                "respond",
                duration_ms=100,
                timestamp="2026-04-05T12:00:00+00:00",
            ),
            _step(
                "temporal-task",
                2,
                "respond",
                duration_ms=120,
                timestamp="2026-04-05T12:00:02+00:00",
            ),
            _step(
                "temporal-task",
                3,
                "respond",
                duration_ms=140,
                timestamp="2026-04-05T12:00:05+00:00",
            ),
            _step(
                "temporal-task",
                4,
                "respond",
                duration_ms=500,
                timestamp="2026-04-05T12:00:11+00:00",
            ),
            _step(
                "temporal-task",
                5,
                "respond",
                duration_ms=700,
                timestamp="2026-04-05T12:00:13+00:00",
            ),
            _step(
                "temporal-task",
                6,
                "respond",
                duration_ms=900,
                timestamp="2026-04-05T12:00:20+00:00",
            ),
        ],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.avg_step_duration_ms == pytest.approx((100 + 120 + 140 + 500 + 700 + 900) / 6)
    assert analysis.max_step_gap_ms == 7000
    assert analysis.step_duration_trend == "decelerating"


def test_analyze_task_flags_element_ref_mismatches() -> None:
    task = AgentTask(
        task_id="dom-task",
        steps=[
            _step("dom-task", 1, "browser_snapshot", tool_result="Button refs: @e1 @e3"),
            _step("dom-task", 2, "browser_click", {"ref": "@e2"}, tool_result="clicked"),
            _step("dom-task", 3, "browser_fill_ref", {"ref": "@e1"}, tool_result="filled"),
        ],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.element_ref_mismatches == 1


def test_analyze_task_normalizes_non_string_result_fields() -> None:
    task = AgentTask(
        task_id="normalize-task",
        steps=[
            _step(
                "normalize-task",
                1,
                "browser_click",
                error=None,
            ),
            _step(
                "normalize-task",
                2,
                "browser_click",
            ),
        ],
    )
    task.steps[0].tool_result = {"data": "Error: access denied"}
    task.steps[1].tool_result = {"state": "ok"}
    task.steps[1].error = ["timeout"]

    analysis = analyze_task(task, detectors=[])

    assert task.steps[0].tool_result == "Error: access denied"
    assert '"state": "ok"' in str(task.steps[1].tool_result)
    assert task.steps[1].error == "['timeout']"
    assert analysis.inline_tool_errors == 1
    assert analysis.errors == 1


def test_analyze_task_empty_trace_edge_case() -> None:
    analysis = analyze_task(AgentTask(task_id="empty-task"), detectors=[])

    assert analysis.step_count == 0
    assert analysis.error_rate == 0.0
    assert analysis.total_duration_ms == 0
    assert analysis.site_name == "unknown"
    assert analysis.final_url == ""
    assert analysis.avg_cost_per_step == 0.0


def test_analyze_task_single_step_edge_case() -> None:
    task = AgentTask(
        task_id="single-step",
        steps=[_step("single-step", 1, "respond", tool_result="ok", duration_ms=42)],
    )

    analysis = analyze_task(task, detectors=[])

    assert analysis.step_count == 1
    assert analysis.max_repeat_tool == "respond"
    assert analysis.max_repeat_count == 1
    assert analysis.avg_step_duration_ms == 42.0
    assert analysis.step_duration_trend == "stable"


def test_analyze_tasks_returns_mapping_by_task_id() -> None:
    task_one = AgentTask(task_id="a", steps=[_step("a", 1, "respond")])
    task_two = AgentTask(task_id="b", steps=[_step("b", 1, "browser_click")])

    analyses = analyze_tasks([task_one, task_two], detectors=[])

    assert set(analyses) == {"a", "b"}
    assert analyses["a"].task_id == "a"
    assert analyses["b"].task_id == "b"


def test_task_analysis_round_trip_with_include_task_preserves_core_fields() -> None:
    task = AgentTask(
        task_id="round-trip",
        steps=[_step("round-trip", 1, "respond", duration_ms=25)],
        outcome=_outcome("round-trip", "success", final_answer="ok"),
    )

    analysis = analyze_task(task, detectors=[])
    restored = TaskAnalysis.from_dict(analysis.to_dict(include_task=True))

    assert restored.task.task_id == "round-trip"
    assert restored.total_duration_ms == analysis.total_duration_ms
    assert restored.task_completed is True
    assert restored.signal_metrics == {}


def test_summarize_tool_result_prefers_error_and_truncates() -> None:
    step = _step(
        "summary-task",
        1,
        "respond",
        tool_result="x" * 100,
        error="problem\n" + ("y" * 40),
    )

    text = summarize_tool_result(step, limit=20)

    assert text.startswith("problem ")
    assert text.endswith("...")
    assert len(text) == 20


def test_tasks_from_steps_groups_steps_and_derives_missing_task_id_from_path() -> None:
    path = Path("trace_20260405.jsonl")
    steps = [
        _step("", 2, "respond", timestamp="2026-04-05T12:01:00+00:00"),
        _step("task-b", 1, "browser_click", timestamp="2026-04-05T12:02:00+00:00"),
        _step("", 1, "respond", timestamp="2026-04-05T12:00:00+00:00"),
    ]

    tasks = tasks_from_steps(steps, source_path=path)

    assert [task.task_id for task in tasks] == ["task-b", "trace_20260405"]
    assert tasks[1].day == "20260405"
    assert all(step.task_id for task in tasks for step in task.steps)


def test_load_tasks_parses_steps_outcomes_and_warns_on_malformed_rows(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_20260405.jsonl"
    _write_jsonl(
        trace_path,
        [
            "not-json",
            ["not-a-dict"],
            {"tool_name": "respond", "tool_input": {}},
            {
                "task_id": "task-1",
                "step": 1,
                "tool_name": "browser_click",
                "tool_input": {"ref": "go"},
                "user_text": "Check out",
                "task_category": "commerce",
                "timestamp": "2026-04-05T12:00:00+00:00",
            },
            {
                "event": "task_complete",
                "task_id": "task-1",
                "status": "success",
                "final_answer": "Done",
                "timestamp": "2026-04-05T12:00:05+00:00",
                "metadata": {"timed_out": False},
            },
            {
                "event": "task_complete",
                "task_id": "outcome-only",
                "status": "failed",
                "user_text": "Outcome only",
                "task_category": "support",
                "timestamp": "2026-04-05T13:00:00+00:00",
            },
        ],
    )

    with pytest.warns(UserWarning, match=r"skipped 3/6 lines"):
        tasks = load_tasks(tmp_path, pattern="*.jsonl", dedup=False)

    assert [task.task_id for task in tasks] == ["outcome-only", "task-1"]
    task = next(task for task in tasks if task.task_id == "task-1")
    assert task.task_text == "Check out"
    assert task.task_category == "commerce"
    assert task.outcome is not None
    assert task.outcome.status == "success"
    assert task.day == "20260405"
    outcome_only = next(task for task in tasks if task.task_id == "outcome-only")
    assert outcome_only.task_text == "Outcome only"
    assert outcome_only.task_category == "support"
    assert outcome_only.steps == []


def test_load_tasks_dedup_keeps_most_recent_task_by_task_text(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace_20260405.jsonl"
    _write_jsonl(
        trace_path,
        [
            {
                "event": "task_complete",
                "task_id": "older",
                "status": "success",
                "user_text": "Buy milk",
                "timestamp": "2026-04-05T10:00:00+00:00",
            },
            {
                "event": "task_complete",
                "task_id": "newer",
                "status": "success",
                "user_text": "Buy milk",
                "timestamp": "2026-04-05T11:00:00+00:00",
            },
        ],
    )

    deduped = load_tasks(tmp_path, pattern="*.jsonl")
    all_tasks = load_tasks(tmp_path, pattern="*.jsonl", dedup=False)

    assert [task.task_id for task in deduped] == ["newer"]
    assert [task.task_id for task in all_tasks] == ["newer", "older"]


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("alpha-123", "alpha-123"),
        ("alpha-", "alpha-123"),
        ("777", "beta-777"),
    ],
)
def test_resolve_task_supports_exact_prefix_and_substring(query: str, expected: str) -> None:
    tasks = [
        AgentTask(task_id="alpha-123"),
        AgentTask(task_id="beta-777"),
        AgentTask(task_id="gamma-999"),
    ]

    resolved = resolve_task(tasks, query)

    assert resolved.task_id == expected


def test_resolve_task_raises_with_available_hint_when_not_found() -> None:
    tasks = [AgentTask(task_id="alpha-123"), AgentTask(task_id="beta-777")]

    with pytest.raises(KeyError, match="Available: alpha-123, beta-777"):
        resolve_task(tasks, "missing")


def test_build_task_tree_groups_by_day_and_site() -> None:
    tasks = [
        AgentTask(
            task_id="task-a",
            day="20260405",
            steps=[_step("task-a", 1, "browser_navigate", page_url="https://shop.example.test/cart")],
        ),
        AgentTask(
            task_id="task-b",
            day="20260405",
            steps=[_step("task-b", 1, "browser_navigate", page_url="https://docs.example.test/api")],
        ),
        AgentTask(
            task_id="task-c",
            day="20260406",
            steps=[_step("task-c", 1, "respond")],
        ),
    ]

    tree = build_task_tree(tasks)

    assert tree == {
        "20260405": {"docs": ["task-b"], "shop": ["task-a"]},
        "20260406": {"unknown": ["task-c"]},
    }
