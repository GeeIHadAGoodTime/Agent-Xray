from __future__ import annotations

from agent_xray.pytest_plugin import XrayFixture


def test_xray_fixture_analyze_dict_steps() -> None:
    report = XrayFixture().analyze(
        [
            {
                "task_id": "task-1",
                "step": 1,
                "tool_name": "web_search",
                "tool_input": {"q": "hotel"},
                "tool_result": "results",
            },
            {
                "task_id": "task-1",
                "step": 2,
                "tool_name": "browser_navigate",
                "tool_input": {"url": "https://example.test"},
                "tool_result": "loaded",
                "page_url": "https://example.test",
            },
        ]
    )
    assert report.grade in {"GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"}
    assert report.unique_tools == 2
    assert report.error_rate == 0
