from __future__ import annotations

import re
from pathlib import Path

from agent_xray.grader import GradeResult, grade_task, load_rules
from agent_xray.root_cause import (
    ClassificationConfig,
    PROMPT_BUG_PATTERNS,
    RootCauseResult,
    classify_failures,
    classify_task,
)
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome, ToolContext

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
    rejected_tools: list[str] | None = None,
    llm_reasoning: str | None = None,
    context_usage_pct: float | None = None,
    context_window: int | None = None,
    compaction_count: int | None = None,
    output_tokens: int | None = None,
) -> AgentStep:
    tools = None
    if rejected_tools is not None:
        tools = ToolContext(
            tools_available=tools_available,
            rejected_tools=rejected_tools,
        )
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
        tools=tools,
        tools_available=None if tools is not None else tools_available,
        llm_reasoning=llm_reasoning,
        context_usage_pct=context_usage_pct,
        context_window=context_window,
        compaction_count=compaction_count,
        output_tokens=output_tokens,
    )


def _task(steps: list[AgentStep], *, task_category: str | None = None) -> AgentTask:
    return AgentTask(
        task_id="task-1",
        task_text="investigate failure",
        task_category=task_category,
        steps=steps,
    )


def _outcome(
    status: str,
    *,
    total_steps: int | None = None,
    final_answer: str | None = None,
    metadata: dict[str, object] | None = None,
) -> TaskOutcome:
    return TaskOutcome(
        task_id="task-1",
        status=status,
        total_steps=total_steps,
        final_answer=final_answer,
        metadata=dict(metadata or {}),
    )


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
    assert cause.confidence_score == 0.9


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


def test_classify_delegation_failure() -> None:
    task = _task(
        [
            _step(1, "spawn_agent", tool_result="Worker launched."),
            _step(2, "wait_agent", error="timeout waiting for delegated worker result"),
            _step(3, "send_input", tool_result="failed to deliver follow-up to worker"),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "delegation_failure"
    assert cause.confidence == "high"
    assert cause.confidence_score == 1.0


def test_classify_test_failure_loop() -> None:
    task = _task(
        [
            _step(
                1, "read_file", tool_input={"path": "src/parser.py"}, tool_result="parser source"
            ),
            _step(
                2,
                "pytest",
                tool_input={"command": "python -m pytest tests/test_parser.py"},
                tool_result="2 failed, 5 passed: tests/test_parser.py::test_whitespace regression",
            ),
            _step(
                3,
                "pytest",
                tool_input={"command": "python -m pytest tests/test_parser.py"},
                tool_result="2 failed, 5 passed: tests/test_parser.py::test_whitespace regression",
            ),
        ],
        task_category="coding",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "test_failure_loop"
    assert cause.confidence == "high"
    assert cause.confidence_score == 1.0


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


def test_classify_insufficient_sources() -> None:
    task = _task(
        [
            _step(
                1,
                "web_search",
                tool_input={"query": "agent observability best practices"},
                tool_result="Top result: https://a.example.test/report",
            ),
            _step(
                2,
                "respond",
                tool_result="According to https://a.example.test/report, this should work.",
            ),
        ],
        task_category="research",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "insufficient_sources"
    assert cause.confidence == "high"
    assert cause.confidence_score == 1.0


def test_classify_early_abort() -> None:
    task = _task(
        [
            _step(1, "browser_navigate", tool_result="OK", page_url="https://shop.example.test"),
            _step(2, "browser_click", tool_result="Clicked.", page_url="https://shop.example.test"),
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


def test_classify_memory_overload() -> None:
    task = _task(
        [
            _step(
                1,
                "read_file",
                tool_input={"path": "trace.log"},
                tool_result="loaded long trace context",
                context_usage_pct=55.0,
                context_window=128000,
            ),
            _step(
                2,
                "shell",
                tool_input={"command": "analyze failures"},
                tool_result="Context window is full and I am losing track of earlier failures.",
                llm_reasoning="The context is full and I am losing track of earlier failures.",
                context_usage_pct=92.0,
                context_window=128000,
                compaction_count=2,
                output_tokens=24,
            ),
            _step(
                3,
                "respond",
                tool_result="Unsure.",
                llm_reasoning="I am not sure anymore because the context is too much.",
                context_usage_pct=95.0,
                context_window=128000,
                output_tokens=4,
            ),
        ],
        task_category="coding",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "memory_overload"
    assert cause.confidence == "medium"
    assert cause.confidence_score == 0.7


def test_classify_context_overflow() -> None:
    task = _task(
        [
            _step(
                1,
                "shell",
                llm_reasoning=(
                    "I have a clear plan with enough detail to trace the issue step by step before "
                    "changing anything."
                ),
            ),
            _step(
                2,
                "shell",
                llm_reasoning="The context window is full and I am losing track of earlier failures.",
            ),
            _step(
                3,
                "shell",
                error="validation error: malformed payload",
                llm_reasoning="Unsure.",
            ),
            _step(
                4,
                "respond",
                tool_result="Need retry.",
                llm_reasoning="Not sure.",
            ),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "context_overflow"
    assert cause.confidence == "medium"
    assert any("context pressure" in evidence for evidence in cause.evidence)


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


def test_classify_rate_limit_cascade() -> None:
    task = _task(
        [
            _step(1, "web_search", tool_result="429 Too Many Requests from upstream"),
            _step(2, "web_search", tool_result="rate limit exceeded by provider"),
            _step(3, "web_search", tool_result="too many requests, retry later"),
            _step(4, "respond", tool_result="Retrying failed."),
        ]
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "rate_limit_cascade"
    assert cause.confidence == "high"
    assert any("rate-limit" in evidence for evidence in cause.evidence)


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


def test_classify_timeout() -> None:
    tools = ["browser_navigate", "browser_click", "browser_snapshot"]
    steps = [
        _step(
            index,
            tools[(index - 1) % len(tools)],
            page_url=f"https://shop.example.test/step-{index}",
            tool_result="progress",
        )
        for index in range(1, 41)
    ]
    steps.extend(
        [
            _step(41, "browser_snapshot", page_url="https://shop.example.test/checkout"),
            _step(
                42,
                "browser_click",
                page_url="https://shop.example.test/checkout",
                tool_result="still blocked",
            ),
            _step(43, "browser_snapshot", page_url="https://shop.example.test/checkout"),
            _step(
                44,
                "browser_click",
                page_url="https://shop.example.test/checkout",
                tool_result="same modal remains",
            ),
            _step(
                45,
                "browser_click",
                page_url="https://shop.example.test/checkout",
                error="could not finish before stop condition",
            ),
        ]
    )
    task = _task(steps)
    task.outcome = _outcome("failed", total_steps=45)
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "timeout"
    assert cause.confidence == "medium"
    assert any("no new progress" in evidence for evidence in cause.evidence)


def test_numeric_confidence_compatibility() -> None:
    baseline = RootCauseResult(
        task_id="task-1",
        root_cause="spin",
        grade="BROKEN",
        score=-1,
        confidence="high",
    )
    numeric = RootCauseResult(
        task_id="task-2",
        root_cause="prompt_bug",
        grade="BROKEN",
        score=-2,
        confidence=0.72,
    )
    assert baseline.confidence == "high"
    assert baseline.confidence_score == 0.9
    assert numeric.confidence == "medium"
    assert numeric.confidence_score == 0.72


def test_classification_config_customization() -> None:
    task = _task(
        [
            _step(
                1,
                "web_search",
                tool_input={"query": "agent observability best practices"},
                tool_result="Top result: https://a.example.test/report",
            ),
            _step(
                2,
                "respond",
                tool_result="According to https://a.example.test/report, this should work.",
            ),
        ],
        task_category="research",
    )
    cause = classify_task(
        task,
        _failing_grade(task),
        config=ClassificationConfig(
            insufficient_sources_min_searches=1, low_source_diversity_threshold=1
        ),
    )
    assert cause is not None
    assert cause.root_cause == "early_abort"


def test_expected_rejections_excludes_policy_tools() -> None:
    task = _task(
        [
            _step(1, "respond", rejected_tools=["ask_user"]),
            _step(2, "respond", rejected_tools=["ask_user"]),
            _step(3, "respond", rejected_tools=["ask_user"]),
            _step(4, "respond", rejected_tools=["ask_user"]),
        ]
    )

    default_cause = classify_task(task, _failing_grade(task))
    assert default_cause is not None
    assert default_cause.root_cause == "tool_rejection_mismatch"

    allowlisted_cause = classify_task(
        task,
        _failing_grade(task),
        config=ClassificationConfig(expected_rejections=frozenset({"ask_user"})),
    )
    assert allowlisted_cause is not None
    assert allowlisted_cause.root_cause != "tool_rejection_mismatch"


def test_expected_rejections_partial() -> None:
    task = _task(
        [
            _step(1, "respond", rejected_tools=["ask_user", "browser_click"]),
            _step(2, "respond", rejected_tools=["ask_user", "browser_click"]),
            _step(3, "respond", rejected_tools=["ask_user", "browser_click"]),
            _step(4, "respond", rejected_tools=["ask_user", "browser_click"]),
        ]
    )

    cause = classify_task(
        task,
        _failing_grade(task),
        config=ClassificationConfig(expected_rejections=frozenset({"ask_user"})),
    )
    assert cause is not None
    assert cause.root_cause == "tool_rejection_mismatch"
    assert any("browser_click" in evidence for evidence in cause.evidence)
    assert not any(
        "rejected tools:" in evidence and "ask_user" in evidence
        for evidence in cause.evidence
    )


def test_expected_rejections_empty_default() -> None:
    task = _task(
        [
            _step(1, "respond", rejected_tools=["browser_click"]),
            _step(2, "respond", rejected_tools=["browser_click"]),
            _step(3, "respond", rejected_tools=["browser_click"]),
            _step(4, "respond", rejected_tools=["browser_click"]),
        ]
    )

    default_cause = classify_task(task, _failing_grade(task))
    explicit_default_cause = classify_task(
        task,
        _failing_grade(task),
        config=ClassificationConfig(),
    )
    assert default_cause is not None
    assert explicit_default_cause is not None
    assert default_cause.root_cause == "tool_rejection_mismatch"
    assert explicit_default_cause.root_cause == "tool_rejection_mismatch"


def test_classify_failures_passes_expected_rejections_config() -> None:
    task = _task(
        [
            _step(1, "respond", rejected_tools=["ask_user"]),
            _step(2, "respond", rejected_tools=["ask_user"]),
            _step(3, "respond", rejected_tools=["ask_user"]),
            _step(4, "respond", rejected_tools=["ask_user"]),
        ]
    )

    failures = classify_failures(
        [task],
        [_failing_grade(task)],
        config=ClassificationConfig(expected_rejections=frozenset({"ask_user"})),
    )
    assert len(failures) == 1
    assert failures[0].root_cause != "tool_rejection_mismatch"


def test_new_prompt_bug_patterns_match_expected_strings() -> None:
    cases = [
        (
            "login flow hit an authentication sign in fail on the account page",
            "auth",
            "login sequence guidance",
        ),
        (
            "captcha challenge blocked the browser from continuing",
            "browser",
            "captcha detection and fallback instructions",
        ),
        (
            "checkout popup modal overlay blocked the next click",
            "browser",
            "popup dismissal guidance",
        ),
        (
            "cookie consent banner required us to accept cookies before checkout",
            "browser",
            "cookie consent handling",
        ),
    ]

    for sample, expected_section, expected_hint in cases:
        matches = [
            (section, fix_hint)
            for pattern, section, fix_hint in PROMPT_BUG_PATTERNS
            if re.search(pattern, sample, re.IGNORECASE)
        ]
        assert matches
        assert any(
            section == expected_section and expected_hint in fix_hint
            for section, fix_hint in matches
        )


def test_classify_healthy_task_returns_none(golden_task: AgentTask) -> None:
    grade = grade_task(golden_task, load_rules(RULES_PATH))
    assert grade.grade == "GOLDEN"
    assert classify_task(golden_task, grade) is None


def test_classify_valid_alternative_path() -> None:
    """Task completed with diverse tools, no browser, enough steps -> valid_alternative_path."""
    from agent_xray.schema import TaskOutcome

    task = _task(
        [
            _step(1, "web_search", tool_input={"query": "LLC formation requirements"}),
            _step(2, "read_url", tool_input={"url": "https://law.example.test/llc"}),
            _step(3, "web_search", tool_input={"query": "registered agent services"}),
            _step(
                4,
                "respond",
                tool_result="Here is a comprehensive guide to forming an LLC...",
            ),
        ]
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="success",
        total_steps=4,
        total_duration_s=10.0,
        final_answer="Detailed LLC guide.",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "valid_alternative_path"
    assert cause.confidence == "high"
    assert any("no browser tools" in e for e in cause.evidence)


def test_valid_alternative_path_requires_completion() -> None:
    """Incomplete tasks should NOT be classified as valid_alternative_path."""
    task = _task(
        [
            _step(1, "web_search", tool_input={"query": "test"}),
            _step(2, "read_url", tool_input={"url": "https://a.example.test"}),
            _step(3, "web_search", tool_input={"query": "more"}),
            _step(4, "respond", tool_result="partial"),
        ]
    )
    # No outcome -> task_completed = False
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause != "valid_alternative_path"


def test_valid_alternative_path_not_when_browser_used() -> None:
    """Tasks using browser tools should not be valid_alternative_path."""
    from agent_xray.schema import TaskOutcome

    task = _task(
        [
            _step(1, "web_search", tool_input={"query": "test"}),
            _step(2, "browser_navigate", tool_input={"url": "https://a.example.test"}),
            _step(3, "read_url", tool_input={"url": "https://b.example.test"}),
            _step(4, "respond", tool_result="done"),
        ]
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="success",
        total_steps=4,
        total_duration_s=10.0,
        final_answer="Done.",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause != "valid_alternative_path"


def test_classify_consultative_success() -> None:
    """Task completed with a long final answer -> consultative_success."""
    from agent_xray.schema import TaskOutcome

    long_answer = "A" * 250
    # Use non-search tools to avoid triggering insufficient_sources
    task = _task(
        [
            _step(1, "think", tool_input={"thought": "Analyzing LLC requirements"}),
            _step(2, "respond", tool_result=long_answer),
        ]
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="success",
        total_steps=2,
        total_duration_s=5.0,
        final_answer=long_answer,
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause == "consultative_success"
    assert cause.confidence == "high"
    assert any("consultative" in e for e in cause.evidence)


def test_consultative_success_requires_long_answer() -> None:
    """Short answers should NOT be consultative_success."""
    from agent_xray.schema import TaskOutcome

    task = _task(
        [
            _step(1, "respond", tool_result="Yes."),
        ]
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="success",
        total_steps=1,
        total_duration_s=1.0,
        final_answer="Yes.",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    assert cause.root_cause != "consultative_success"


def test_valid_alternative_path_before_tool_selection_bug() -> None:
    """valid_alternative_path should win over tool_selection_bug when task succeeded."""
    from agent_xray.schema import TaskOutcome

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
            _step(
                4,
                "respond",
                tool_result="Here is a comprehensive analysis.",
                tools_available=["web_search", "browser_navigate", "browser_click"],
            ),
        ]
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="success",
        total_steps=4,
        total_duration_s=10.0,
        final_answer="Comprehensive analysis.",
    )
    cause = classify_task(task, _failing_grade(task))
    assert cause is not None
    # Should be valid_alternative_path, not tool_selection_bug
    assert cause.root_cause == "valid_alternative_path"
