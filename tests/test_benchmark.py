from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from agent_xray.analyzer import analyze_tasks, load_tasks
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome
from agent_xray.surface import surface_for_task

STEP_COUNT = 10_000
TIME_LIMIT_SECONDS = 5.0


def _benchmark_task() -> AgentTask:
    steps = [
        AgentStep(
            task_id="benchmark-task",
            step=index,
            tool_name="browser_click" if index % 2 else "browser_snapshot",
            tool_input={"ref": f"item-{index}"},
            tool_result=None,
            error=None,
            duration_ms=5,
            page_url=f"https://shop.example.test/flow/{index % 25}",
        )
        for index in range(1, STEP_COUNT + 1)
    ]
    return AgentTask(
        task_id="benchmark-task",
        task_category="commerce",
        steps=steps,
        outcome=TaskOutcome(
            task_id="benchmark-task",
            status="success",
            total_steps=STEP_COUNT,
            total_duration_s=STEP_COUNT / 100,
        ),
    )


def _write_trace(path: Path, task: AgentTask) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / "benchmark_trace_20260327.jsonl"
    payloads = [json.dumps(step.to_dict(), sort_keys=True) for step in task.steps]
    payloads.append(
        json.dumps(
            {
                "event": "task_complete",
                "task_id": task.task_id,
                "status": task.outcome.status if task.outcome else "success",
                "total_steps": task.outcome.total_steps if task.outcome else len(task.steps),
                "total_duration_s": task.outcome.total_duration_s if task.outcome else 0.0,
            },
            sort_keys=True,
        )
    )
    trace_path.write_text("\n".join(payloads), encoding="utf-8")
    return path


def _measure(fn) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


@pytest.mark.benchmark
def test_benchmark_load_tasks(tmp_path) -> None:
    trace_dir = _write_trace(tmp_path, _benchmark_task())
    tasks, elapsed = _measure(lambda: load_tasks(trace_dir))
    assert len(tasks) == 1
    assert len(tasks[0].steps) == STEP_COUNT
    assert elapsed < TIME_LIMIT_SECONDS


@pytest.mark.benchmark
def test_benchmark_analyze_tasks() -> None:
    tasks = [_benchmark_task()]
    analyses, elapsed = _measure(lambda: analyze_tasks(tasks))
    assert analyses["benchmark-task"].step_count == STEP_COUNT
    assert elapsed < TIME_LIMIT_SECONDS


@pytest.mark.benchmark
def test_benchmark_grade_tasks() -> None:
    tasks = [_benchmark_task()]
    rules = load_rules()
    grades, elapsed = _measure(lambda: grade_tasks(tasks, rules))
    assert grades[0].task_id == "benchmark-task"
    assert elapsed < TIME_LIMIT_SECONDS


@pytest.mark.benchmark
def test_benchmark_surface_for_task() -> None:
    surface, elapsed = _measure(lambda: surface_for_task(_benchmark_task()))
    assert surface["task_id"] == "benchmark-task"
    assert len(surface["steps"]) == STEP_COUNT
    assert elapsed < TIME_LIMIT_SECONDS
