"""Tests for the StepRecorder base class."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from agent_xray.analyzer import load_tasks
from agent_xray.instrument.base import StepRecorder
from agent_xray.schema import SCHEMA_VERSION


def test_record_step_creates_jsonl(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="test-task")
    recorder.record_step(
        tool_name="browser_navigate",
        tool_input={"url": "https://example.test"},
        tool_result="Page loaded.",
        duration_ms=500,
    )
    recorder.close()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["task_id"] == "test-task"
    assert payload["tool_name"] == "browser_navigate"
    assert payload["tool_input"]["url"] == "https://example.test"
    assert payload["tool_result"] == "Page loaded."
    assert payload["duration_ms"] == 500
    assert payload["schema_version"] == SCHEMA_VERSION
    assert "timestamp" in payload


def test_start_and_end_task(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="task-1")
    recorder.start_task("task-1", "Buy headphones", task_category="commerce")
    recorder.record_step(
        tool_name="search",
        tool_input={"q": "headphones"},
        tool_result="Found items.",
    )
    recorder.end_task("task-1", "success", final_answer="Done.")
    recorder.close()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    start_event = json.loads(lines[0])
    assert start_event["event"] == "task_start"
    assert start_event["task_id"] == "task-1"
    assert start_event["user_text"] == "Buy headphones"
    assert start_event["task_category"] == "commerce"

    end_event = json.loads(lines[2])
    assert end_event["event"] == "task_complete"
    assert end_event["outcome"] == "success"
    assert end_event["final_answer"] == "Done."
    assert end_event["total_steps"] == 1


def test_auto_increment_step(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="inc-task")
    recorder.record_step(tool_name="a", tool_input={})
    recorder.record_step(tool_name="b", tool_input={})
    recorder.record_step(tool_name="c", tool_input={})
    recorder.close()

    lines = _read_lines(tmp_path)
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == [1, 2, 3]


def test_explicit_step_number(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="explicit-task")
    recorder.record_step(step=10, tool_name="a", tool_input={})
    recorder.record_step(step=20, tool_name="b", tool_input={})
    recorder.close()

    lines = _read_lines(tmp_path)
    steps = [json.loads(line)["step"] for line in lines]
    assert steps == [10, 20]


def test_model_metadata(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="model-task")
    recorder.record_step(
        tool_name="tool",
        tool_input={},
        model_name="gpt-5-mini",
        input_tokens=100,
        output_tokens=50,
        cost_usd=0.005,
    )
    recorder.close()

    payload = json.loads(_read_lines(tmp_path)[0])
    assert payload["model_name"] == "gpt-5-mini"
    assert payload["input_tokens"] == 100
    assert payload["output_tokens"] == 50
    assert payload["cost_usd"] == 0.005


def test_tools_available(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="tools-task")
    recorder.record_step(
        tool_name="tool_a",
        tool_input={},
        tools_available=["tool_a", "tool_b", "tool_c"],
    )
    recorder.close()

    payload = json.loads(_read_lines(tmp_path)[0])
    assert payload["tools_available"] == ["tool_a", "tool_b", "tool_c"]


def test_error_step(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="error-task")
    recorder.record_step(
        tool_name="failing_tool",
        tool_input={"key": "value"},
        error="Timed out.",
        duration_ms=5000,
    )
    recorder.close()

    payload = json.loads(_read_lines(tmp_path)[0])
    assert payload["error"] == "Timed out."
    assert "tool_result" not in payload


def test_context_manager(tmp_path: Path) -> None:
    with StepRecorder(tmp_path, task_id="ctx-task") as recorder:
        recorder.record_step(tool_name="tool", tool_input={})
    # Should be closed
    assert len(_read_lines(tmp_path)) == 1


def test_load_tasks_reads_recorded_output(tmp_path: Path) -> None:
    """Verify that load_tasks can read the JSONL produced by StepRecorder."""
    recorder = StepRecorder(tmp_path, task_id="roundtrip-task")
    recorder.start_task("roundtrip-task", "Test roundtrip")
    recorder.record_step(
        tool_name="browser_navigate",
        tool_input={"url": "https://shop.test"},
        tool_result="Loaded.",
        duration_ms=300,
    )
    recorder.record_step(
        tool_name="browser_click",
        tool_input={"ref": "buy-btn"},
        tool_result="Clicked.",
        duration_ms=150,
    )
    recorder.end_task("roundtrip-task", "success", final_answer="All done.")
    recorder.close()

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "roundtrip-task"
    assert task.task_text == "Test roundtrip"
    assert len(task.steps) == 2
    assert task.steps[0].tool_name == "browser_navigate"
    assert task.steps[1].tool_name == "browser_click"
    assert task.outcome is not None
    assert task.outcome.status == "success"
    assert task.outcome.final_answer == "All done."


def test_thread_safety(tmp_path: Path) -> None:
    """Verify concurrent writes don't crash or corrupt output."""
    recorder = StepRecorder(tmp_path, task_id="threaded-task")
    errors: list[str] = []

    def write_steps(thread_id: int) -> None:
        try:
            for i in range(20):
                recorder.record_step(
                    tool_name=f"tool-{thread_id}",
                    tool_input={"i": i},
                    tool_result=f"result-{thread_id}-{i}",
                )
        except Exception as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=write_steps, args=(tid,)) for tid in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    recorder.close()

    assert not errors
    lines = _read_lines(tmp_path)
    assert len(lines) == 100  # 5 threads * 20 steps
    for line in lines:
        payload = json.loads(line)
        assert "tool_name" in payload
        assert "task_id" in payload


def test_kwargs_extension_fields(tmp_path: Path) -> None:
    recorder = StepRecorder(tmp_path, task_id="ext-task")
    recorder.record_step(
        tool_name="tool",
        tool_input={},
        page_url="https://example.test/page",
        custom_field="custom_value",
    )
    recorder.close()

    payload = json.loads(_read_lines(tmp_path)[0])
    assert payload["page_url"] == "https://example.test/page"
    assert payload["custom_field"] == "custom_value"


def _read_lines(tmp_path: Path) -> list[str]:
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    return files[0].read_text(encoding="utf-8").strip().splitlines()
