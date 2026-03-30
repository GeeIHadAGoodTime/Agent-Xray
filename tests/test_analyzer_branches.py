from __future__ import annotations

import pytest

from agent_xray.analyzer import (
    _max_consecutive_repeat,
    _unique_url_paths,
    analyze_task,
    classify_soft_error,
    classify_error,
    extract_site_from_urlish,
    extract_site_name,
    final_answer_indicates_failure,
    has_inline_tool_error,
    site_from_host,
)
from agent_xray.grader import grade_task, load_rules
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome


def _task(*steps: AgentStep, task_text: str | None = None) -> AgentTask:
    return AgentTask(task_id="task-1", steps=list(steps), task_text=task_text)


def _step(
    step: int,
    tool_name: str,
    *,
    page_url: str | None = None,
    tool_input: dict[str, object] | None = None,
    tool_result: str | None = None,
    error: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-1",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        page_url=page_url,
        tool_result=tool_result,
        error=error,
    )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("approval denied for browser_click", "approval_block"),
        ("unknown tool requested", "unknown_tool"),
        ("not available in your capability tier", "tier_block"),
        ("request timed out after 30s", "timeout"),
        ("429 too many requests", "rate_limit"),
        ("no accessibility snapshot available", "snapshot_missing"),
        ("fill address failed: locator missing", "fill_fail"),
        ("form incomplete after submission", "fill_incomplete"),
        ("navigation failed because execution context was destroyed", "navigation_fail"),
        ("failed to focus window for checkout", "window_not_found"),
        ("validation error: field required", "validation"),
        ("404 not found", "not_found"),
        ("click button failed after timeout", "timeout"),
        ("invalid json in response", "parse_error"),
        ("connection refused by upstream", "connection_error"),
        ("totally different failure", "other"),
        (None, ""),
    ],
)
def test_classify_error_patterns(message: str | None, expected: str) -> None:
    assert classify_error(message) == expected


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("www.shop.example.com", "shop"),
        ("mobile.docs.example.io", "docs"),
        ("app.service.internal", "service"),
        ("localhost", "localhost"),
        ("127.0.0.1", "127-0-0-1"),
        ("api.example.co.uk", "api"),
        ("", None),
    ],
)
def test_site_from_host_handles_common_branches(host: str, expected: str | None) -> None:
    assert site_from_host(host) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://www.shop.example.test/cart", "shop"),
        ("shop.example.test/checkout", "shop"),
        ("http://user:pass@docs.example.test:8080/path", "docs"),
        ("about:blank", "no-navigation"),
        ("chrome://new-tab-page", "no-navigation"),
        ("", None),
        ("   ", None),
    ],
)
def test_extract_site_from_urlish_handles_urls_and_non_navigation_markers(
    raw: str, expected: str | None
) -> None:
    assert extract_site_from_urlish(raw) == expected


def test_extract_site_name_prefers_task_text_when_it_contains_url() -> None:
    task = _task(_step(1, "respond"), task_text="https://status.example.test/dashboard")
    assert extract_site_name(task) == "status"


def test_extract_site_name_falls_back_to_browser_no_navigation() -> None:
    task = _task(
        _step(1, "browser_snapshot"),
        _step(2, "browser_click"),
    )
    assert extract_site_name(task) == "no-navigation"


def test_extract_site_name_returns_unknown_without_any_url_signal() -> None:
    task = _task(_step(1, "respond"))
    assert extract_site_name(task) == "unknown"


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ([], ("", 0)),
        (["browser_click"], ("browser_click", 1)),
        (["a", "a", "b", "b"], ("a", 2)),
        (["a", "b", "b", "b", "a"], ("b", 3)),
        (["a", "b", "c", "c", "c", "c"], ("c", 4)),
    ],
)
def test_max_consecutive_repeat_edge_cases(sequence: list[str], expected: tuple[str, int]) -> None:
    assert _max_consecutive_repeat(sequence) == expected


def test_unique_url_paths_collapse_query_variants() -> None:
    urls = [
        "https://shop.example.test/search?q=headset",
        "https://shop.example.test/search?q=keyboard",
        "https://shop.example.test/cart",
    ]
    assert _unique_url_paths(urls) == [
        "https://shop.example.test/search?q=headset",
        "https://shop.example.test/cart",
    ]


def test_analyze_task_marks_timeout_like_for_large_step_count() -> None:
    task = AgentTask(
        task_id="large-timeout",
        steps=[AgentStep("large-timeout", index, "browser_wait", {}) for index in range(1, 76)],
    )
    analysis = analyze_task(task)
    assert analysis.timeout_like is True
    assert analysis.step_count == 75


def test_analyze_task_marks_timeout_like_for_timeout_outcome() -> None:
    task = AgentTask(
        task_id="timeout-task",
        steps=[AgentStep("timeout-task", 1, "browser_wait", {})],
        outcome=TaskOutcome(task_id="timeout-task", status="timeout"),
    )
    analysis = analyze_task(task)
    assert analysis.timeout_like is True


def test_analyze_task_spin_tiers_cover_mild_moderate_and_severe() -> None:
    mild = analyze_task(
        _task(
            _step(1, "browser_click"),
            _step(2, "browser_click"),
            _step(3, "browser_click"),
        )
    )
    moderate = analyze_task(_task(*[_step(index, "browser_snapshot") for index in range(1, 6)]))
    severe = analyze_task(_task(*[_step(index, "browser_wait") for index in range(1, 11)]))
    assert mild.spin_is_mild is True
    assert moderate.spin_is_moderate is True
    assert severe.spin_is_severe is True


def test_analyze_task_error_rate_tiers_are_exclusive() -> None:
    medium = analyze_task(
        _task(
            _step(1, "browser_click", error="timeout"),
            _step(2, "browser_click"),
            _step(3, "browser_click"),
            _step(4, "browser_click"),
        )
    )
    high = analyze_task(
        _task(
            _step(1, "browser_click", error="timeout"),
            _step(2, "browser_click", error="timeout"),
            _step(3, "browser_click"),
        )
    )
    assert medium.error_rate_is_medium is True
    assert medium.error_rate_is_high is False
    assert high.error_rate_is_high is True
    assert high.error_rate_is_medium is False


def test_soft_error_detected_not_on_page() -> None:
    assert classify_soft_error("NOT ON A PAYMENT PAGE") == "soft_not_on_page"


def test_soft_error_none_for_normal_result() -> None:
    assert classify_soft_error("Success") == ""


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Element not found for selector .checkout", "soft_element_missing"),
        ("Request timed out after 30s", "soft_timeout"),
        ("Access denied by upstream", "soft_access_denied"),
        ("429 too many requests", "soft_rate_limit"),
        ("Failed to submit form", "soft_failure"),
    ],
)
def test_classify_soft_error_patterns(message: str, expected: str) -> None:
    assert classify_soft_error(message) == expected


def test_soft_errors_counted_in_metrics() -> None:
    task = _task(
        _step(1, "browser_fill_ref", tool_result="Element not found"),
        _step(2, "browser_click", tool_result="429 too many requests"),
        _step(3, "browser_click", tool_result="Success"),
    )
    analysis = analyze_task(task)
    assert analysis.soft_errors == 2
    assert analysis.soft_error_kinds == {
        "soft_element_missing": 1,
        "soft_rate_limit": 1,
    }
    assert analysis.metrics()["soft_errors"] == 2
    assert analysis.metrics()["soft_error_kinds"] == {
        "soft_element_missing": 1,
        "soft_rate_limit": 1,
    }


def test_soft_errors_not_double_counted() -> None:
    task = _task(
        _step(
            1,
            "browser_fill_ref",
            tool_result="Element not found",
            error="timeout waiting for element",
        )
    )
    analysis = analyze_task(task)
    assert analysis.soft_errors == 0
    assert analysis.soft_error_kinds == {}


def test_has_inline_tool_error_matches_string_and_dict_payloads() -> None:
    assert has_inline_tool_error("Error: request failed") is True
    assert has_inline_tool_error({"data": "tool said error: upstream refused"}) is True
    assert has_inline_tool_error({"data": "all good"}) is False


def test_inline_tool_errors_counted_in_metrics_without_error_field() -> None:
    task = _task(
        _step(1, "browser_click", tool_result="Error: element no longer attached"),
        _step(2, "browser_click"),
        _step(3, "browser_click", tool_result="Error: should not count", error="timeout"),
    )
    task.steps[1].tool_result = {"data": "tool output says error: access denied"}
    analysis = analyze_task(task)
    assert analysis.inline_tool_errors == 2
    assert analysis.metrics()["inline_tool_errors"] == 2


def test_task_analysis_round_trip_preserves_inline_tool_errors() -> None:
    task = _task(_step(1, "browser_click", tool_result="Error: hidden failure"))
    analysis = analyze_task(task)
    restored = type(analysis).from_dict(analysis.to_dict())
    assert restored.inline_tool_errors == 1


def test_default_rules_apply_inline_error_penalty() -> None:
    task = _task(_step(1, "browser_click", tool_result="Error: hidden failure"))
    task.outcome = TaskOutcome(task_id=task.task_id, status="completed", final_answer="Done")
    result = grade_task(task, load_rules("default"))
    signal = next((s for s in result.signals if s.name == "inline_error_penalty"), None)
    assert signal is not None
    assert signal.passed is True
    assert signal.points == -2
    assert signal.actual == 1
    assert any("inline errors" in reason for reason in result.reasons)


def test_final_answer_failure_classifier_matches_explicit_failure_language() -> None:
    assert final_answer_indicates_failure(
        "The checkout page is currently showing an error, and I cannot proceed further."
    ) is True
    assert final_answer_indicates_failure("Checkout completed successfully.") is False


def test_analyze_task_records_final_answer_failure_signal() -> None:
    task = _task(_step(1, "browser_snapshot", page_url="https://shop.example.test/checkout"))
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="completed",
        final_answer="The checkout page is currently showing an error, and I cannot proceed further.",
    )
    analysis = analyze_task(task)
    assert analysis.final_answer_contains_failure_keywords is True
    assert analysis.metrics()["final_answer_contains_failure_keywords"] is True
    # Backward compat alias still present
    assert analysis.metrics()["final_answer_indicates_failure"] is True
    # Keywords matched are visible
    assert len(analysis.final_answer_failure_keywords_matched) > 0
    assert "error" in analysis.final_answer_failure_keywords_matched


def test_analyze_task_flags_ungrounded_answer_when_final_answer_introduces_new_number() -> None:
    task = _task(
        _step(1, "web_search", tool_result="Order confirmation number 12345 was captured.")
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="completed",
        final_answer="Order placed successfully. Confirmation #67890.",
    )

    analysis = analyze_task(task)

    assert analysis.ungrounded_answer is True
    assert analysis.metrics()["ungrounded_answer"] is True


def test_analyze_task_keeps_grounded_answer_when_specific_tokens_match_tool_results() -> None:
    task = _task(
        _step(
            1,
            "read_url",
            tool_result="Customer: Alice. Confirmation #12345. Receipt: https://shop.example.test/r/12345",
        )
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="completed",
        final_answer="Alice order confirmed with #12345. Receipt: https://shop.example.test/r/12345",
    )

    analysis = analyze_task(task)

    assert analysis.ungrounded_answer is False


def test_default_rules_apply_ungrounded_answer_penalty() -> None:
    task = _task(
        _step(1, "respond", tool_result="The only grounded data here is confirmation #12345.")
    )
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="completed",
        final_answer="Done. Confirmation #67890.",
    )

    result = grade_task(task, load_rules("default"))

    signal = next((s for s in result.signals if s.name == "ungrounded_answer_penalty"), None)
    assert signal is not None
    assert signal.passed is True
    assert signal.points == -1
    assert signal.actual is True


def test_task_analysis_round_trip_preserves_soft_error_fields() -> None:
    task = _task(_step(1, "browser_fill_ref", tool_result="Element not found"))
    analysis = analyze_task(task)
    restored = type(analysis).from_dict(analysis.to_dict())
    assert restored.soft_errors == 1
    assert restored.soft_error_kinds == {"soft_element_missing": 1}
    assert restored.final_answer_contains_failure_keywords is False
    assert restored.task.task_id == analysis.task.task_id
    assert restored.ungrounded_answer is False


def test_consultative_response_detected_for_short_think_only_task() -> None:
    """LLC-style task: 2 think steps, success, no browser tools -> consultative."""
    task = _task(_step(0, "think"), _step(1, "think"))
    task.outcome = TaskOutcome(
        task_id=task.task_id,
        status="completed",
        final_answer="What state do you want to register the LLC in?",
    )
    analysis = analyze_task(task)
    assert analysis.is_consultative_response is True
    assert analysis.metrics()["is_consultative_response"] is True


def test_consultative_response_false_when_browser_used() -> None:
    """Task with browser navigation is not consultative even if short."""
    task = _task(_step(0, "think"), _step(1, "browser_navigate"))
    task.outcome = TaskOutcome(task_id=task.task_id, status="completed", final_answer="Done")
    analysis = analyze_task(task)
    assert analysis.is_consultative_response is False


def test_consultative_response_false_when_many_steps() -> None:
    """Long task is not consultative even without browser tools."""
    steps = [_step(i, "think") for i in range(5)]
    task = _task(*steps)
    task.outcome = TaskOutcome(task_id=task.task_id, status="completed", final_answer="Done")
    analysis = analyze_task(task)
    assert analysis.is_consultative_response is False


def test_consultative_response_false_when_task_failed() -> None:
    """Failed short task is not consultative."""
    task = _task(_step(0, "think"))
    task.outcome = TaskOutcome(task_id=task.task_id, status="error", final_answer="Failed")
    analysis = analyze_task(task)
    assert analysis.is_consultative_response is False
