"""Tests for the MCP proxy instrumentor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent_xray.analyzer import load_tasks
from agent_xray.instrument.mcp_proxy import XRayMCPProxy, _extract_result_text


def test_extract_result_text_string() -> None:
    assert _extract_result_text("hello") == "hello"


def test_extract_result_text_none() -> None:
    assert _extract_result_text(None) is None


def test_extract_result_text_content_list() -> None:
    result = MagicMock()
    item = MagicMock()
    item.text = "Page loaded."
    result.content = [item]
    assert _extract_result_text(result) == "Page loaded."


def test_extract_result_text_text_attr() -> None:
    result = MagicMock(spec=["text"])
    result.text = "Direct text"
    # No content attribute
    del result.content
    assert _extract_result_text(result) == "Direct text"


def test_extract_result_text_dict_items() -> None:
    result = MagicMock()
    result.content = [{"text": "from dict"}]
    result.text = None
    assert _extract_result_text(result) == "from dict"


@pytest.mark.asyncio
async def test_call_tool_records_step(tmp_path: Path) -> None:
    mock_client = AsyncMock()
    mock_result = MagicMock()
    mock_result.content = [MagicMock(text="Clicked button.")]
    mock_client.call_tool.return_value = mock_result

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path), task_id="mcp-task")
    result = await proxy.call_tool("browser_click", {"ref": "submit-btn"})

    assert result is mock_result
    mock_client.call_tool.assert_called_once_with("browser_click", {"ref": "submit-btn"})

    proxy.close()

    lines = _read_step_lines(tmp_path)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tool_name"] == "browser_click"
    assert payload["tool_input"]["ref"] == "submit-btn"
    assert payload["tool_result"] == "Clicked button."
    assert payload["duration_ms"] >= 0
    assert payload["task_id"] == "mcp-task"


@pytest.mark.asyncio
async def test_call_tool_records_error(tmp_path: Path) -> None:
    mock_client = AsyncMock()
    mock_client.call_tool.side_effect = RuntimeError("Connection lost")

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path), task_id="mcp-err")

    with pytest.raises(RuntimeError, match="Connection lost"):
        await proxy.call_tool("browser_navigate", {"url": "https://shop.test"})

    proxy.close()

    lines = _read_step_lines(tmp_path)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tool_name"] == "browser_navigate"
    assert "Connection lost" in payload["error"]
    assert "tool_result" not in payload


@pytest.mark.asyncio
async def test_list_tools_caches_names(tmp_path: Path) -> None:
    mock_client = AsyncMock()
    tool_a = MagicMock()
    tool_a.name = "browser_click"
    tool_b = MagicMock()
    tool_b.name = "browser_navigate"
    mock_result = MagicMock()
    mock_result.tools = [tool_a, tool_b]
    mock_client.list_tools.return_value = mock_result

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path), task_id="mcp-list")
    await proxy.list_tools()

    # Now call a tool and check that tools_available is populated
    mock_client.call_tool.return_value = MagicMock(content=[MagicMock(text="ok")])
    await proxy.call_tool("browser_click", {"ref": "btn"})
    proxy.close()

    lines = _read_step_lines(tmp_path)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tools_available"] == ["browser_click", "browser_navigate"]


@pytest.mark.asyncio
async def test_roundtrip_with_load_tasks(tmp_path: Path) -> None:
    mock_client = AsyncMock()
    mock_client.call_tool.return_value = MagicMock(content=[MagicMock(text="Page loaded.")])

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path), task_id="mcp-rt")
    await proxy.call_tool("browser_navigate", {"url": "https://shop.test"})
    await proxy.call_tool("browser_click", {"ref": "product"})
    proxy.close()

    tasks = load_tasks(tmp_path)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "mcp-rt"
    assert len(task.steps) == 2
    assert task.steps[0].tool_name == "browser_navigate"
    assert task.steps[1].tool_name == "browser_click"
    assert task.outcome is not None
    assert task.outcome.status == "success"


def test_sync_call_tool(tmp_path: Path) -> None:
    mock_client = MagicMock()
    mock_client.call_tool.return_value = "Simple result"

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path), task_id="mcp-sync")
    result = proxy.call_tool_sync("my_tool", {"key": "val"})

    assert result == "Simple result"
    proxy.close()

    lines = _read_step_lines(tmp_path)
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["tool_name"] == "my_tool"
    assert payload["tool_result"] == "Simple result"


def test_getattr_proxy(tmp_path: Path) -> None:
    """Non-call_tool attributes should proxy to the wrapped client."""
    mock_client = MagicMock()
    mock_client.custom_attr = "custom_value"

    proxy = XRayMCPProxy(mock_client, output_dir=str(tmp_path))
    assert proxy.custom_attr == "custom_value"
    proxy.close()


def _read_step_lines(tmp_path: Path) -> list[str]:
    files = list(tmp_path.glob("*.jsonl"))
    assert files
    all_lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    return [line for line in all_lines if '"tool_name"' in line]
