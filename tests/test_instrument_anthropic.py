"""Tests for the Anthropic SDK instrumentor using mocks."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_xray.analyzer import load_tasks


def _install_mock_anthropic() -> types.ModuleType:
    """Install a fake anthropic package into sys.modules."""
    anthropic_mod = types.ModuleType("anthropic")
    resources_mod = types.ModuleType("anthropic.resources")

    class _Messages:
        def create(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class _AsyncMessages:
        async def create(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    resources_mod.Messages = _Messages  # type: ignore[attr-defined]
    resources_mod.AsyncMessages = _AsyncMessages  # type: ignore[attr-defined]
    anthropic_mod.resources = resources_mod  # type: ignore[attr-defined]

    sys.modules["anthropic"] = anthropic_mod
    sys.modules["anthropic.resources"] = resources_mod
    return anthropic_mod


def _make_response(
    tool_uses: list[dict[str, Any]],
    *,
    model: str = "claude-sonnet-4-20250514",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> Any:
    """Build a mock Anthropic response with tool_use blocks."""
    blocks = []
    for tu in tool_uses:
        block = MagicMock()
        block.type = "tool_use"
        block.id = tu.get("id", "toolu_01")
        block.name = tu["name"]
        block.input = tu.get("input", {})
        blocks.append(block)

    response = MagicMock()
    response.content = blocks
    response.model = model
    response.usage = MagicMock()
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens
    return response


@pytest.fixture(autouse=True)
def _cleanup_anthropic_mock() -> Any:
    """Remove mock anthropic modules after each test."""
    yield
    for key in list(sys.modules):
        if key.startswith("anthropic"):
            del sys.modules[key]


def test_instrument_logs_tool_uses(tmp_path: Path) -> None:
    _install_mock_anthropic()

    from agent_xray.instrument.anthropic_sdk import AnthropicInstrumentor

    instrumentor = AnthropicInstrumentor(output_dir=str(tmp_path), task_id="anth-task")
    instrumentor.instrument()

    response = _make_response(
        [
            {"name": "browser_navigate", "input": {"url": "https://shop.test"}},
            {"name": "browser_click", "input": {"ref": "btn"}},
        ]
    )

    from agent_xray.instrument.anthropic_sdk import _log_response

    _log_response(
        instrumentor.recorder,
        response,
        elapsed_ms=200,
        kwargs={
            "tools": [
                {"name": "browser_navigate"},
                {"name": "browser_click"},
                {"name": "browser_fill"},
            ]
        },
    )

    instrumentor.uninstrument()

    files = list(tmp_path.glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    step1 = json.loads(lines[0])
    assert step1["tool_name"] == "browser_navigate"
    assert step1["tool_input"]["url"] == "https://shop.test"
    assert step1["model_name"] == "claude-sonnet-4-20250514"
    assert step1["input_tokens"] == 100
    assert step1["output_tokens"] == 50
    assert step1["duration_ms"] == 100  # 200 / 2 tools
    assert "browser_navigate" in step1["tools_available"]

    step2 = json.loads(lines[1])
    assert step2["tool_name"] == "browser_click"
    assert step2["tool_input"]["ref"] == "btn"


def test_no_tool_use_blocks_skipped(tmp_path: Path) -> None:
    _install_mock_anthropic()

    from agent_xray.instrument.anthropic_sdk import AnthropicInstrumentor, _log_response

    instrumentor = AnthropicInstrumentor(output_dir=str(tmp_path), task_id="no-tools")

    # Response with no tool_use blocks (text only)
    response = MagicMock()
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Hello!"
    response.content = [text_block]
    response.model = "claude-sonnet-4-20250514"
    response.usage = MagicMock(input_tokens=50, output_tokens=20)

    _log_response(instrumentor.recorder, response, elapsed_ms=100, kwargs={})
    instrumentor.recorder.close()

    files = list(tmp_path.glob("*.jsonl"))
    if files:
        content = files[0].read_text(encoding="utf-8").strip()
        assert content == ""


def test_roundtrip_with_load_tasks(tmp_path: Path) -> None:
    _install_mock_anthropic()

    from agent_xray.instrument.anthropic_sdk import AnthropicInstrumentor, _log_response

    instrumentor = AnthropicInstrumentor(output_dir=str(tmp_path), task_id="rt-task")
    recorder = instrumentor.recorder
    recorder.start_task("rt-task", "Test roundtrip")

    response = _make_response([{"name": "browser_navigate", "input": {"url": "https://shop.test"}}])
    _log_response(recorder, response, elapsed_ms=300, kwargs={})

    recorder.end_task("rt-task", "success", final_answer="Done.")
    instrumentor.uninstrument()

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "rt-task"
    assert len(tasks[0].steps) == 1
    assert tasks[0].steps[0].tool_name == "browser_navigate"
    assert tasks[0].outcome is not None
    assert tasks[0].outcome.status == "success"


def test_xray_trace_decorator(tmp_path: Path) -> None:
    _install_mock_anthropic()

    from agent_xray.instrument.anthropic_sdk import xray_trace

    @xray_trace(output_dir=str(tmp_path), task_id="deco-task")
    def my_agent(prompt: str) -> str:
        return "result"

    result = my_agent("test prompt")
    assert result == "result"

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "deco-task"
    assert tasks[0].outcome is not None
    assert tasks[0].outcome.status == "success"


def test_xray_trace_decorator_on_failure(tmp_path: Path) -> None:
    _install_mock_anthropic()

    from agent_xray.instrument.anthropic_sdk import xray_trace

    @xray_trace(output_dir=str(tmp_path), task_id="fail-task")
    def failing_agent(prompt: str) -> str:
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        failing_agent("test")

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].outcome is not None
    assert tasks[0].outcome.status == "failed"


def _traced_call(instrumentor: Any, response: Any, kwargs: dict[str, Any]) -> Any:
    return response
