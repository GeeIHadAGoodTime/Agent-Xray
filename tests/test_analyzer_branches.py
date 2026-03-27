from __future__ import annotations

import pytest

from agent_xray.analyzer import (
    _max_consecutive_repeat,
    _unique_url_paths,
    analyze_task,
    classify_error,
    extract_site_from_urlish,
    extract_site_name,
    site_from_host,
)
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
def test_max_consecutive_repeat_edge_cases(
    sequence: list[str], expected: tuple[str, int]
) -> None:
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
    moderate = analyze_task(
        _task(*[_step(index, "browser_snapshot") for index in range(1, 6)])
    )
    severe = analyze_task(
        _task(*[_step(index, "browser_wait") for index in range(1, 11)])
    )
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
