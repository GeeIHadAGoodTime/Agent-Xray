from __future__ import annotations

from pathlib import Path

from agent_xray.grader import GradeResult, grade_task, load_rules
from agent_xray.root_cause import classify_task
from agent_xray.schema import AgentStep, AgentTask

RULES_PATH = Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules" / "default.json"


def _step(
    step: int,
    tool_name: str,
    *,
    tool_input: dict[str, object] | None = None,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
        tools_available=tools_available,
        llm_reasoning=llm_reasoning,
    )


def _task(steps: list[AgentStep]) -> AgentTask:
    return AgentTask(task_id="task-1", task_text="investigate failure", steps=steps)


def _failing_grade(task: AgentTask, *, score: int = -1) -> GradeResult:
    return GradeResult(
        task_id=task.task_id,
        grade="BROKEN",
        score=score,
        reasons=[],
        metrics={},
        signals=[],
    )


def test_classify_spin() -> None:
    task = _task([_step(index, "browser_snapshot", tool_result="same") for index in range(1, 6)])
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "spin"


def test_classify_routing_bug() -> None:
    task = _task(
        [
            _step(1, "respond", tools_available=[]),
            _step(2, "respond", tools_available=[]),
            _step(3, "respond", tools_available=[]),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "routing_bug"


def test_classify_approval_block() -> None:
    task = _task(
        [
            _step(1, "browser_click", error="approval denied for browser_click"),
            _step(2, "browser_click", error="not approved to continue"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "approval_block"


def test_classify_tool_selection_bug() -> None:
    task = _task(
        [
            _step(
                1,
                "web_search",
                tool_input={"query": "checkout flow"},
                tools_available=["web_search", "browser_navigate", "browser_click"],
            ),
            _step(
                2,
                "read_url",
                tool_input={"url": "https://docs.example.test/guide"},
                tools_available=["web_search", "browser_navigate", "browser_click"],
            ),
            _step(
                3,
                "web_search",
                tool_input={"query": "payment page"},
                tools_available=["web_search", "browser_navigate", "browser_click"],
            ),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "tool_selection_bug"


def test_classify_early_abort() -> None:
    task = _task(
        [
            _step(1, "respond", tool_result="Starting."),
            _step(2, "respond", tool_result="Could not finish."),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "early_abort"


def test_classify_stuck_loop() -> None:
    task = _task(
        [
            _step(1, "browser_snapshot", page_url="https://shop.example.test/cart"),
            _step(2, "browser_click", page_url="https://shop.example.test/cart"),
            _step(3, "browser_snapshot", page_url="https://shop.example.test/cart"),
            _step(4, "browser_click", page_url="https://shop.example.test/cart"),
            _step(5, "browser_snapshot", page_url="https://shop.example.test/cart"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "stuck_loop"


def test_classify_reasoning_bug() -> None:
    task = _task(
        [
            _step(1, "browser_navigate", page_url="https://shop.example.test/"),
            _step(2, "browser_click", page_url="https://shop.example.test/products/widget"),
            _step(3, "browser_fill_ref", page_url="https://shop.example.test/cart"),
            _step(4, "browser_click", page_url="https://shop.example.test/checkout"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "reasoning_bug"


def test_classify_prompt_bug() -> None:
    task = _task(
        [
            _step(
                1,
                "browser_snapshot",
                page_url="https://shop.example.test/checkout",
                llm_reasoning="I am not sure which control is the real checkout button.",
            ),
            _step(
                2,
                "browser_snapshot",
                page_url="https://shop.example.test/checkout",
                llm_reasoning="The prompt is unclear and I cannot tell what to click.",
            ),
            _step(
                3,
                "browser_snapshot",
                page_url="https://shop.example.test/checkout",
                llm_reasoning="Still unsure how to proceed.",
            ),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "prompt_bug"


def test_classify_tool_bug() -> None:
    task = _task(
        [
            _step(1, "run_tool", error="validation error: field required"),
            _step(2, "run_tool", error="unknown tool requested"),
            _step(3, "run_tool", error="validation error: malformed payload"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "tool_bug"


def test_classify_environment_drift() -> None:
    task = _task(
        [
            _step(1, "browser_click", error="Timed out waiting for element"),
            _step(2, "browser_click", error="404 not found"),
            _step(3, "browser_click", error="click failed after timeout"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "environment_drift"


def test_classify_model_limit() -> None:
    tools = ["browser_click", "browser_snapshot", "browser_scroll", "browser_wait"]
    task = _task(
        [
            _step(index, tools[(index - 1) % len(tools)], page_url="https://shop.example.test/cart")
            for index in range(1, 52)
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "model_limit"


def test_classify_healthy_task_returns_none(golden_task: AgentTask) -> None:
    grade = grade_task(golden_task, load_rules(RULES_PATH))
    assert grade.grade == "GOOD"
    assert classify_task(golden_task, grade) is None
