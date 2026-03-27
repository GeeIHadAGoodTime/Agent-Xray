"""Tests for the LangChain callback handler instrumentor."""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent_xray.analyzer import load_tasks
from agent_xray.instrument.langchain_cb import XRayCallbackHandler


def test_tool_start_end_records_step(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-task")

    handler.on_tool_start(
        {"name": "search"},
        "headphones",
        run_id="run-1",
    )
    # Small delay so duration_ms > 0
    time.sleep(0.01)
    handler.on_tool_end("Found 3 results.", run_id="run-1")

    handler.close()

    lines = _read_lines(tmp_path)
    # task_start + step + task_complete
    step_lines = [line for line in lines if '"tool_name"' in line]
    assert len(step_lines) == 1

    payload = json.loads(step_lines[0])
    assert payload["tool_name"] == "search"
    assert payload["tool_input"]["input"] == "headphones"
    assert payload["tool_result"] == "Found 3 results."
    assert payload["duration_ms"] >= 0


def test_tool_error_records_error(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-err")

    handler.on_tool_start(
        {"name": "failing_tool"},
        "bad input",
        run_id="run-2",
    )
    handler.on_tool_error(ValueError("boom"), run_id="run-2")

    handler.close()

    step_lines = [line for line in _read_lines(tmp_path) if '"tool_name"' in line]
    assert len(step_lines) == 1
    payload = json.loads(step_lines[0])
    assert payload["tool_name"] == "failing_tool"
    assert "boom" in payload["error"]


def test_llm_metadata_propagates(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-meta")

    handler.on_llm_start(
        {"kwargs": {"model_name": "gpt-5-mini"}},
        ["prompt"],
        invocation_params={"model_name": "gpt-5-mini"},
    )
    # Simulate LLM returning token usage
    from unittest.mock import MagicMock

    llm_result = MagicMock()
    llm_result.llm_output = {"token_usage": {"prompt_tokens": 120, "completion_tokens": 40}}
    handler.on_llm_end(llm_result)

    handler.on_tool_start({"name": "search"}, "query", run_id="run-3")
    handler.on_tool_end("results", run_id="run-3")

    handler.close()

    step_lines = [line for line in _read_lines(tmp_path) if '"tool_name"' in line]
    assert len(step_lines) == 1
    payload = json.loads(step_lines[0])
    assert payload["model_name"] == "gpt-5-mini"
    assert payload["input_tokens"] == 120
    assert payload["output_tokens"] == 40


def test_multiple_tools_in_sequence(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-multi")

    for i in range(3):
        handler.on_tool_start(
            {"name": f"tool_{i}"},
            f"input_{i}",
            run_id=f"run-{i}",
        )
        handler.on_tool_end(f"result_{i}", run_id=f"run-{i}")

    handler.close()

    step_lines = [line for line in _read_lines(tmp_path) if '"tool_name"' in line]
    assert len(step_lines) == 3
    names = [json.loads(line)["tool_name"] for line in step_lines]
    assert names == ["tool_0", "tool_1", "tool_2"]


def test_roundtrip_with_load_tasks(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-rt")

    handler.on_tool_start({"name": "navigate"}, "https://shop.test", run_id="r1")
    handler.on_tool_end("Page loaded.", run_id="r1")
    handler.on_tool_start({"name": "click"}, "buy-btn", run_id="r2")
    handler.on_tool_end("Clicked.", run_id="r2")

    handler.close()

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "lc-rt"
    assert len(task.steps) == 2
    assert task.steps[0].tool_name == "navigate"
    assert task.steps[1].tool_name == "click"
    assert task.outcome is not None
    assert task.outcome.status == "success"


def test_dict_tool_input(tmp_path: Path) -> None:
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-dict")

    # LangChain sometimes passes dicts as input_str
    handler.on_tool_start(
        {"name": "search"},
        {"query": "test", "limit": 5},  # type: ignore[arg-type]
        run_id="run-dict",
    )
    handler.on_tool_end("results", run_id="run-dict")
    handler.close()

    step_lines = [line for line in _read_lines(tmp_path) if '"tool_name"' in line]
    payload = json.loads(step_lines[0])
    assert payload["tool_input"]["query"] == "test"
    assert payload["tool_input"]["limit"] == 5


def test_unmatched_tool_end_uses_fifo(tmp_path: Path) -> None:
    """If run_id doesn't match, fall back to the first pending tool."""
    handler = XRayCallbackHandler(output_dir=str(tmp_path), task_id="lc-fifo")

    handler.on_tool_start({"name": "tool_a"}, "input_a", run_id="run-a")
    handler.on_tool_end("result_a", run_id="unknown-id")

    handler.close()

    step_lines = [line for line in _read_lines(tmp_path) if '"tool_name"' in line]
    assert len(step_lines) == 1
    payload = json.loads(step_lines[0])
    assert payload["tool_name"] == "tool_a"
    assert payload["tool_result"] == "result_a"


def _read_lines(tmp_path: Path) -> list[str]:
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    return files[0].read_text(encoding="utf-8").strip().splitlines()
