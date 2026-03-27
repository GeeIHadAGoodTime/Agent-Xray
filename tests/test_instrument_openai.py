"""Tests for the OpenAI SDK instrumentor using mocks."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent_xray.analyzer import load_tasks


def _install_mock_openai() -> types.ModuleType:
    """Install a fake openai package into sys.modules."""
    openai_mod = types.ModuleType("openai")
    resources_mod = types.ModuleType("openai.resources")
    chat_mod = types.ModuleType("openai.resources.chat")

    class _Completions:
        def create(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    class _AsyncCompletions:
        async def create(self, *args: Any, **kwargs: Any) -> Any:
            raise NotImplementedError

    chat_mod.Completions = _Completions  # type: ignore[attr-defined]
    chat_mod.AsyncCompletions = _AsyncCompletions  # type: ignore[attr-defined]
    resources_mod.chat = chat_mod  # type: ignore[attr-defined]
    openai_mod.resources = resources_mod  # type: ignore[attr-defined]

    sys.modules["openai"] = openai_mod
    sys.modules["openai.resources"] = resources_mod
    sys.modules["openai.resources.chat"] = chat_mod
    return openai_mod


def _make_response(
    tool_calls: list[dict[str, Any]],
    *,
    model: str = "gpt-4o",
    prompt_tokens: int = 200,
    completion_tokens: int = 80,
) -> Any:
    """Build a mock OpenAI chat completion response."""
    mock_tool_calls = []
    for tc in tool_calls:
        fn = MagicMock()
        fn.name = tc["name"]
        fn.arguments = json.dumps(tc.get("arguments", {}))
        mock_tc = MagicMock()
        mock_tc.id = tc.get("id", "call_01")
        mock_tc.function = fn
        mock_tool_calls.append(mock_tc)

    message = MagicMock()
    message.tool_calls = mock_tool_calls

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


@pytest.fixture(autouse=True)
def _cleanup_openai_mock() -> Any:
    """Remove mock openai modules after each test."""
    yield
    for key in list(sys.modules):
        if key.startswith("openai"):
            del sys.modules[key]


def test_log_response_captures_tool_calls(tmp_path: Path) -> None:
    _install_mock_openai()

    from agent_xray.instrument.openai_sdk import OpenAIInstrumentor, _log_response

    instrumentor = OpenAIInstrumentor(output_dir=str(tmp_path), task_id="oai-task")
    response = _make_response(
        [
            {"name": "search", "arguments": {"query": "headphones"}},
            {"name": "click", "arguments": {"ref": "product-1"}},
        ]
    )

    _log_response(
        instrumentor.recorder,
        response,
        elapsed_ms=400,
        kwargs={
            "tools": [
                {"function": {"name": "search"}},
                {"function": {"name": "click"}},
                {"function": {"name": "fill"}},
            ]
        },
    )
    instrumentor.recorder.close()

    lines = _read_lines(tmp_path)
    assert len(lines) == 2

    step1 = json.loads(lines[0])
    assert step1["tool_name"] == "search"
    assert step1["tool_input"]["query"] == "headphones"
    assert step1["model_name"] == "gpt-4o"
    assert step1["input_tokens"] == 200
    assert step1["output_tokens"] == 80
    assert step1["duration_ms"] == 200  # 400 / 2
    assert "search" in step1["tools_available"]

    step2 = json.loads(lines[1])
    assert step2["tool_name"] == "click"
    assert step2["tool_input"]["ref"] == "product-1"


def test_no_tool_calls_skipped(tmp_path: Path) -> None:
    _install_mock_openai()

    from agent_xray.instrument.openai_sdk import OpenAIInstrumentor, _log_response

    instrumentor = OpenAIInstrumentor(output_dir=str(tmp_path), task_id="no-tc")

    message = MagicMock()
    message.tool_calls = None
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]

    _log_response(instrumentor.recorder, response, elapsed_ms=100, kwargs={})
    instrumentor.recorder.close()

    files = list(tmp_path.glob("*.jsonl"))
    if files:
        content = files[0].read_text(encoding="utf-8").strip()
        assert content == ""


def test_roundtrip_with_load_tasks(tmp_path: Path) -> None:
    _install_mock_openai()

    from agent_xray.instrument.openai_sdk import OpenAIInstrumentor, _log_response

    instrumentor = OpenAIInstrumentor(output_dir=str(tmp_path), task_id="rt-oai")
    recorder = instrumentor.recorder
    recorder.start_task("rt-oai", "OpenAI roundtrip test")

    response = _make_response([{"name": "web_search", "arguments": {"q": "test"}}])
    _log_response(recorder, response, elapsed_ms=150, kwargs={})

    recorder.end_task("rt-oai", "success", final_answer="Found it.")
    instrumentor.recorder.close()

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    assert tasks[0].task_id == "rt-oai"
    assert tasks[0].task_text == "OpenAI roundtrip test"
    assert len(tasks[0].steps) == 1
    assert tasks[0].steps[0].tool_name == "web_search"
    assert tasks[0].outcome is not None
    assert tasks[0].outcome.status == "success"


def test_instrument_and_uninstrument(tmp_path: Path) -> None:
    mock_mod = _install_mock_openai()

    from agent_xray.instrument.openai_sdk import OpenAIInstrumentor

    original_create = mock_mod.resources.chat.Completions.create

    instrumentor = OpenAIInstrumentor(output_dir=str(tmp_path))
    instrumentor.instrument()

    # After instrument, the method should be different
    assert mock_mod.resources.chat.Completions.create is not original_create

    instrumentor.uninstrument()

    # After uninstrument, the original should be restored
    assert mock_mod.resources.chat.Completions.create is original_create


def test_malformed_arguments_handled(tmp_path: Path) -> None:
    _install_mock_openai()

    from agent_xray.instrument.openai_sdk import OpenAIInstrumentor, _log_response

    instrumentor = OpenAIInstrumentor(output_dir=str(tmp_path), task_id="bad-args")

    fn = MagicMock()
    fn.name = "tool"
    fn.arguments = "not valid json{"
    tc = MagicMock()
    tc.id = "call_01"
    tc.function = fn
    message = MagicMock()
    message.tool_calls = [tc]
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    response.model = "gpt-4o"
    response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)

    _log_response(instrumentor.recorder, response, elapsed_ms=50, kwargs={})
    instrumentor.recorder.close()

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tool_name"] == "tool"
    assert "raw" in payload["tool_input"]


def _read_lines(tmp_path: Path) -> list[str]:
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    return files[0].read_text(encoding="utf-8").strip().splitlines()
