from __future__ import annotations

import json

from agent_xray.capture import _sanitize_text, _sanitize_value, build_fixture, capture_task
from agent_xray.schema import AgentTask


def test_sanitize_removes_emails() -> None:
    sanitized = _sanitize_text("Reach me at alice@example.com for the receipt.")
    assert "alice@example.com" not in sanitized
    assert "*email*" in sanitized


def test_sanitize_removes_phone_numbers() -> None:
    sanitized = _sanitize_text("Call 312-555-0199 before delivery.")
    assert "312-555-0199" not in sanitized
    assert "*phone*" in sanitized


def test_sanitize_removes_credit_cards() -> None:
    sanitized = _sanitize_text("Use 4111 1111 1111 1111 for payment.")
    assert "4111 1111 1111 1111" not in sanitized
    assert "*card_number*" in sanitized


def test_sanitize_removes_urls_with_tokens() -> None:
    sanitized = _sanitize_value(
        {"url": "https://api.example.test/checkout?token=secret&session=abc123"}
    )
    assert sanitized["url"] == "https://shop.example.test/checkout"


def test_sanitize_preserves_tool_names(sample_step) -> None:
    task = AgentTask(task_id="sample-task", steps=[sample_step], task_text="Inspect checkout.")
    payload = build_fixture(task, sanitize=True)
    assert payload["step_sequence"][0]["tool_name"] == "browser_click"


def test_capture_creates_fixture_file(tmp_path, golden_task: AgentTask) -> None:
    path = capture_task([golden_task], golden_task.task_id, tmp_path / "fixture.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.exists()
    assert payload["task_id"] == golden_task.task_id
    assert payload["step_sequence"][0]["tool_name"] == "browser_navigate"
