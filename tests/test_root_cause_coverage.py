from __future__ import annotations

from typing import Any

from agent_xray.analyzer import TaskAnalysis
from agent_xray.grader import GradeResult
from agent_xray.root_cause import classify_task
from agent_xray.schema import AgentStep, AgentTask


def _step(
    step: int,
    tool_name: str,
    *,
    error: str | None = None,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input={},
        error=error,
        page_url=page_url,
        tools_available=tools_available,
    )


def _task(*steps: AgentStep) -> AgentTask:
    return AgentTask(task_id="task-1", task_text="Investigate failure", steps=list(steps))


def _failing_grade(task: AgentTask) -> GradeResult:
    return GradeResult(
        task_id=task.task_id,
        grade="BROKEN",
        score=-3,
        reasons=[],
        metrics={},
        signals=[],
    )


def _analysis(task: AgentTask, **overrides: Any) -> TaskAnalysis:
    tool_sequence = [step.tool_name for step in task.steps]
    unique_tools = sorted(set(tool_sequence))
    unique_urls = [step.page_url for step in task.steps if step.page_url]
    base = {
        "task": task,
        "unique_urls": unique_urls,
        "unique_url_paths": list(unique_urls),
        "unique_tools": unique_tools,
        "tool_sequence": tool_sequence,
        "max_repeat_tool": tool_sequence[0] if tool_sequence else "",
        "max_repeat_count": max(
            (tool_sequence.count(tool) for tool in unique_tools),
            default=0,
        ),
        "errors": sum(1 for step in task.steps if step.error),
        "error_rate": (
            sum(1 for step in task.steps if step.error) / len(task.steps)
            if task.steps
            else 0.0
        ),
        "total_duration_ms": 0,
        "hallucinated_tools": 0,
        "no_tools_steps": 0,
        "site_name": "example",
        "final_url": unique_urls[-1] if unique_urls else "",
        "timeout_like": False,
        "task_completed": False,
        "error_kinds": {},
        "total_cost_usd": 0.0,
        "avg_cost_per_step": 0.0,
        "signal_metrics": {},
    }
    base.update(overrides)
    return TaskAnalysis(**base)


def test_classify_task_spin_with_repeated_tool_calls() -> None:
    task = _task(*[_step(index, "browser_snapshot", page_url="https://shop.example.test/cart") for index in range(1, 6)])

    result = classify_task(task, _failing_grade(task))

    assert result is not None
    assert result.root_cause == "spin"


def test_classify_task_early_abort_with_short_trace() -> None:
    task = _task(
        _step(1, "browser_navigate", page_url="https://shop.example.test"),
        _step(2, "respond", page_url="https://shop.example.test"),
    )

    result = classify_task(task, _failing_grade(task), analysis=_analysis(task))

    assert result is not None
    assert result.root_cause == "early_abort"


def test_classify_task_routing_bug_with_zero_tools_available() -> None:
    task = _task(
        _step(1, "respond", tools_available=[]),
        _step(2, "respond", tools_available=[]),
        _step(3, "respond", tools_available=[]),
    )

    result = classify_task(task, _failing_grade(task))

    assert result is not None
    assert result.root_cause == "routing_bug"


def test_classify_task_tool_bug_with_high_tool_error_rate() -> None:
    task = _task(
        _step(1, "run_tool", error="validation error: missing required field"),
        _step(2, "run_tool", error="unknown tool requested"),
        _step(3, "run_tool", error="validation error: malformed payload"),
    )

    result = classify_task(task, _failing_grade(task))

    assert result is not None
    assert result.root_cause == "tool_bug"


def test_classify_task_populates_also_matched_and_candidate_scores() -> None:
    task = _task(
        *[
            _step(index, "browser_snapshot", page_url="https://shop.example.test/cart")
            for index in range(1, 7)
        ]
    )

    result = classify_task(task, _failing_grade(task))

    assert result is not None
    assert result.root_cause == "spin"
    assert result.also_matched
    assert result.candidate_scores
    assert all(isinstance(candidate["confidence_score"], float) for candidate in result.candidate_scores)
    assert isinstance(result.confidence_score, float)
