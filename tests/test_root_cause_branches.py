from __future__ import annotations

from typing import Any

import pytest

from agent_xray.analyzer import TaskAnalysis
from agent_xray.grader import GradeResult
from agent_xray.root_cause import classify_task
from agent_xray.schema import AgentStep, AgentTask


def _step(
    task_id: str,
    step: int,
    tool_name: str,
    *,
    page_url: str | None = None,
    error: str | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input={},
        page_url=page_url,
        error=error,
        tools_available=tools_available,
        llm_reasoning=llm_reasoning,
    )


def _task(task_id: str, steps: list[AgentStep]) -> AgentTask:
    return AgentTask(task_id=task_id, task_text=f"investigate {task_id}", steps=steps)


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
        "max_repeat_count": 1 if tool_sequence else 0,
        "errors": sum(1 for step in task.steps if step.error),
        "error_rate": (sum(1 for step in task.steps if step.error) / len(task.steps))
        if task.steps
        else 0.0,
        "total_duration_ms": 0,
        "hallucinated_tools": 0,
        "no_tools_steps": 0,
        "site_name": "example",
        "final_url": unique_urls[-1] if unique_urls else "",
        "timeout_like": False,
        "error_kinds": {},
        "total_cost_usd": 0.0,
        "avg_cost_per_step": 0.0,
        "signal_metrics": {},
    }
    base.update(overrides)
    return TaskAnalysis(**base)


def _grade(task: AgentTask, grade: str = "BROKEN") -> GradeResult:
    return GradeResult(
        task_id=task.task_id,
        grade=grade,
        score=-1,
        reasons=[],
        metrics={},
        signals=[],
    )


@pytest.fixture(
    params=(
        "routing_bug",
        "approval_block",
        "spin",
        "environment_drift",
        "tool_bug",
        "tool_selection_bug_search_only",
        "prompt_bug",
        "model_limit",
        "stuck_loop",
        "early_abort",
        "reasoning_bug",
        "tool_selection_bug_low_diversity",
    )
)
def root_cause_case(request: pytest.FixtureRequest) -> tuple[str, AgentTask, TaskAnalysis]:
    name = str(request.param)

    if name == "routing_bug":
        task = _task("routing", [_step("routing", index, "respond") for index in range(1, 4)])
        return ("routing_bug", task, _analysis(task, no_tools_steps=3))

    if name == "approval_block":
        task = _task(
            "approval",
            [
                _step("approval", 1, "browser_click", error="approval denied"),
                _step("approval", 2, "browser_click", error="not approved"),
                _step("approval", 3, "respond"),
            ],
        )
        return (
            "approval_block",
            task,
            _analysis(task, errors=2, error_rate=2 / 3, error_kinds={"approval_block": 2}),
        )

    if name == "spin":
        task = _task("spin", [_step("spin", index, "browser_snapshot") for index in range(1, 7)])
        return ("spin", task, _analysis(task, max_repeat_tool="browser_snapshot", max_repeat_count=6))

    if name == "environment_drift":
        task = _task(
            "env",
            [
                _step("env", 1, "browser_click", error="timeout", page_url="https://shop.example/cart"),
                _step("env", 2, "browser_click", error="404 not found", page_url="https://shop.example/cart"),
                _step("env", 3, "browser_click", error="click failed", page_url="https://shop.example/cart"),
                _step("env", 4, "browser_snapshot", page_url="https://shop.example/cart"),
            ],
        )
        return (
            "environment_drift",
            task,
            _analysis(
                task,
                errors=3,
                error_rate=0.75,
                error_kinds={"timeout": 1, "not_found": 1, "click_fail": 1},
            ),
        )

    if name == "tool_bug":
        task = _task(
            "tool",
            [
                _step("tool", 1, "run_tool", error="validation error"),
                _step("tool", 2, "run_tool", error="unknown tool"),
                _step("tool", 3, "run_tool", error="validation error"),
                _step("tool", 4, "respond"),
            ],
        )
        return (
            "tool_bug",
            task,
            _analysis(
                task,
                errors=3,
                error_rate=0.75,
                error_kinds={"validation": 2, "unknown_tool": 1},
            ),
        )

    if name == "tool_selection_bug_search_only":
        task = _task(
            "tool-select-search",
            [
                _step(
                    "tool-select-search",
                    1,
                    "web_search",
                    tools_available=["web_search", "browser_click", "browser_navigate"],
                ),
                _step(
                    "tool-select-search",
                    2,
                    "read_url",
                    page_url="https://docs.example/guide",
                    tools_available=["web_search", "browser_click", "browser_navigate"],
                ),
                _step(
                    "tool-select-search",
                    3,
                    "fetch_page",
                    tools_available=["web_search", "browser_click", "browser_navigate"],
                ),
            ],
        )
        return ("tool_selection_bug", task, _analysis(task))

    if name == "prompt_bug":
        task = _task(
            "prompt",
            [
                _step("prompt", 1, "browser_snapshot", llm_reasoning="I am not sure what to do next."),
                _step("prompt", 2, "browser_click", llm_reasoning="The prompt is unclear."),
                _step("prompt", 3, "browser_snapshot", llm_reasoning="Still unsure which tool to use."),
                _step("prompt", 4, "respond"),
            ],
        )
        return ("prompt_bug", task, _analysis(task, max_repeat_tool="browser_snapshot", max_repeat_count=2))

    if name == "model_limit":
        task = _task(
            "model-limit",
            [
                _step("model-limit", index, "browser_click", page_url="https://shop.example/cart")
                for index in range(1, 53)
            ],
        )
        return (
            "model_limit",
            task,
            _analysis(task, unique_urls=["https://shop.example/cart"], max_repeat_count=2),
        )

    if name == "stuck_loop":
        task = _task(
            "stuck",
            [
                _step("stuck", 1, "browser_snapshot", page_url="https://shop.example/cart"),
                _step("stuck", 2, "browser_click", page_url="https://shop.example/cart"),
                _step("stuck", 3, "browser_wait", page_url="https://shop.example/cart"),
                _step("stuck", 4, "browser_snapshot", page_url="https://shop.example/cart"),
                _step("stuck", 5, "browser_click", page_url="https://shop.example/cart"),
            ],
        )
        return ("stuck_loop", task, _analysis(task, unique_urls=["https://shop.example/cart"]))

    if name == "early_abort":
        task = _task(
            "abort",
            [
                _step("abort", 1, "respond"),
                _step("abort", 2, "respond"),
            ],
        )
        return ("early_abort", task, _analysis(task))

    if name == "reasoning_bug":
        task = _task(
            "reasoning",
            [
                _step("reasoning", 1, "browser_navigate", page_url="https://shop.example/"),
                _step("reasoning", 2, "browser_click", page_url="https://shop.example/products"),
                _step("reasoning", 3, "browser_fill_ref", page_url="https://shop.example/cart"),
                _step("reasoning", 4, "browser_click", page_url="https://shop.example/checkout"),
            ],
        )
        return (
            "reasoning_bug",
            task,
            _analysis(
                task,
                unique_urls=[
                    "https://shop.example/",
                    "https://shop.example/products",
                    "https://shop.example/cart",
                    "https://shop.example/checkout",
                ],
            ),
        )

    if name == "tool_selection_bug_low_diversity":
        task = _task(
            "tool-select-low-diversity",
            [
                _step(
                    "tool-select-low-diversity",
                    1,
                    "browser_snapshot",
                    page_url="https://shop.example/",
                ),
                _step(
                    "tool-select-low-diversity",
                    2,
                    "browser_snapshot",
                    page_url="https://shop.example/products",
                ),
                _step(
                    "tool-select-low-diversity",
                    3,
                    "browser_snapshot",
                    page_url="https://shop.example/cart",
                ),
                _step(
                    "tool-select-low-diversity",
                    4,
                    "browser_snapshot",
                    page_url="https://shop.example/checkout",
                ),
            ],
        )
        return (
            "tool_selection_bug",
            task,
            _analysis(
                task,
                unique_urls=[
                    "https://shop.example/",
                    "https://shop.example/products",
                    "https://shop.example/cart",
                    "https://shop.example/checkout",
                ],
                unique_tools=["browser_snapshot"],
            ),
        )

    raise AssertionError(f"Unhandled root cause case: {name}")


def test_classify_each_root_cause_independently(
    root_cause_case: tuple[str, AgentTask, TaskAnalysis]
) -> None:
    expected, task, analysis = root_cause_case
    result = classify_task(task, _grade(task), analysis=analysis)
    assert result is not None
    assert result.root_cause == expected


def test_prompt_bug_fallback_enriches_section() -> None:
    task = _task(
        "prompt-fallback",
        [
            _step("prompt-fallback", 1, "browser_snapshot", page_url="https://shop.example/cart"),
            _step("prompt-fallback", 2, "browser_click", page_url="https://shop.example/checkout"),
            _step("prompt-fallback", 3, "web_search"),
            _step("prompt-fallback", 4, "browser_snapshot", page_url="https://shop.example/checkout"),
            _step("prompt-fallback", 5, "respond"),
        ],
    )
    analysis = _analysis(
        task,
        unique_urls=["https://shop.example/cart", "https://shop.example/checkout"],
        unique_tools=["browser_click", "browser_snapshot", "web_search", "respond"],
        errors=1,
        error_rate=0.2,
        hallucinated_tools=1,
    )
    result = classify_task(task, _grade(task), analysis=analysis)
    assert result is not None
    assert result.root_cause == "prompt_bug"
    assert result.prompt_section in {"research", "tools", "browser"}
    assert result.prompt_fix_hint is not None
