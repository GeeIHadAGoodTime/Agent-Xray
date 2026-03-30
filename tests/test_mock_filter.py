"""Tests for mock task filtering in load_tasks()."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_xray.analyzer import load_tasks


def _write_jsonl(lines: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")


def test_mock_tasks_excluded_from_load():
    """Tasks with MagicMock model_name are filtered out."""
    real_tasks = [
        {"event": "task_complete", "task_id": "real-1", "outcome": "success",
         "model_name": "gpt-4.1-nano", "user_text": "Play jazz"},
        {"event": "task_complete", "task_id": "real-2", "outcome": "success",
         "model_name": "gpt-4.1-nano", "user_text": "What time is it"},
        {"event": "task_complete", "task_id": "real-3", "outcome": "failure",
         "model_name": "gpt-4.1-nano", "user_text": "Order pizza"},
    ]
    mock_tasks = [
        {"event": "task_complete", "task_id": "mock-1", "outcome": "success",
         "model_name": "<MagicMock name='mock._get_model()' id='123'>",
         "user_text": "Find an Italian restaurant"},
        {"event": "task_complete", "task_id": "mock-2", "outcome": "success",
         "model_name": "<MagicMock name='mock._get_model()' id='456'>",
         "user_text": "Search for shoes"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "agent-steps-20260329.jsonl"
        _write_jsonl(real_tasks + mock_tasks, log_file)
        tasks = load_tasks(tmpdir)

    assert len(tasks) == 3, f"Expected 3 real tasks, got {len(tasks)}"
    task_ids = {t.task_id for t in tasks}
    assert task_ids == {"real-1", "real-2", "real-3"}
    assert "mock-1" not in task_ids
    assert "mock-2" not in task_ids


def test_mock_with_steps_also_excluded():
    """Mock tasks that have steps before the task_complete are still excluded."""
    lines = [
        {"task_id": "mock-1", "tool_name": "browser_navigate",
         "tool_input": {"url": "https://example.com"}, "tool_result": "ok"},
        {"event": "task_complete", "task_id": "mock-1", "outcome": "success",
         "model_name": "<MagicMock name='mock._get_model()' id='789'>",
         "user_text": "Test task"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "agent-steps-20260329.jsonl"
        _write_jsonl(lines, log_file)
        tasks = load_tasks(tmpdir)

    assert len(tasks) == 0, f"Mock task should be excluded, got {len(tasks)}"


def test_real_tasks_unaffected():
    """Tasks with normal model names pass through."""
    lines = [
        {"event": "task_complete", "task_id": "t1", "outcome": "success",
         "model_name": "gpt-4.1-nano", "user_text": "Hello"},
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        log_file = Path(tmpdir) / "agent-steps-20260329.jsonl"
        _write_jsonl(lines, log_file)
        tasks = load_tasks(tmpdir)

    assert len(tasks) == 1
    assert tasks[0].task_id == "t1"
