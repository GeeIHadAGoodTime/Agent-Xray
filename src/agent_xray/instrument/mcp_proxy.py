"""MCP tool-call logging proxy for agent-xray.

Wraps an MCP client connection and logs every ``call_tool`` request and
response to JSONL.  This is the recommended integration path for Playwright
MCP or any other MCP tool server.

Usage::

    from agent_xray.instrument import XRayMCPProxy

    traced = XRayMCPProxy(mcp_client, output_dir="./traces")
    result = await traced.call_tool("browser_navigate", {"url": "https://example.test"})
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .base import StepRecorder

_DEFAULT_OUTPUT_DIR = "./traces"


class XRayMCPProxy:
    """Proxy that wraps an MCP client and logs tool calls as agent-xray steps.

    Args:
        client: The underlying MCP client instance.  Must expose a
            ``call_tool(name, arguments)`` coroutine (or sync method).
        output_dir: Trace output directory.
        task_id: Default task identifier.

    Example:
        >>> proxy = XRayMCPProxy(mcp_client, output_dir="./traces")
        >>> # result = await proxy.call_tool("browser_click", {"ref": "btn"})
        >>> proxy.close()
    """

    def __init__(
        self,
        client: Any,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        *,
        task_id: str | None = None,
    ) -> None:
        self._client = client
        self._task_id = task_id or f"mcp-{uuid.uuid4().hex[:8]}"
        self._recorder = StepRecorder(output_dir, task_id=self._task_id)
        self._recorder.start_task(self._task_id)
        self._tools_cache: list[str] | None = None

    @property
    def recorder(self) -> StepRecorder:
        """Access the underlying step recorder."""
        return self._recorder

    @property
    def client(self) -> Any:
        """Access the wrapped MCP client."""
        return self._client

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Call a tool on the wrapped client and record the step.

        Args:
            name: Tool name.
            arguments: Tool input arguments.

        Returns:
            The tool result from the underlying client.
        """
        tool_input = dict(arguments) if arguments else {}
        start = time.monotonic()
        error_msg: str | None = None
        result_text: str | None = None
        result = None
        try:
            result = await self._client.call_tool(name, arguments)
            result_text = _extract_result_text(result)
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._recorder.record_step(
                task_id=self._task_id,
                tool_name=name,
                tool_input=tool_input,
                tool_result=result_text,
                error=error_msg,
                duration_ms=elapsed_ms,
                tools_available=self._tools_cache,
            )
        return result

    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any] | None = None,
    ) -> Any:
        """Synchronous variant of :meth:`call_tool`.

        Uses the sync ``call_tool`` method on the client if available.

        Args:
            name: Tool name.
            arguments: Tool input arguments.

        Returns:
            The tool result from the underlying client.
        """
        tool_input = dict(arguments) if arguments else {}
        start = time.monotonic()
        error_msg: str | None = None
        result_text: str | None = None
        result = None
        try:
            result = self._client.call_tool(name, arguments)
            result_text = _extract_result_text(result)
        except Exception as exc:
            error_msg = str(exc)
            raise
        finally:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            self._recorder.record_step(
                task_id=self._task_id,
                tool_name=name,
                tool_input=tool_input,
                tool_result=result_text,
                error=error_msg,
                duration_ms=elapsed_ms,
                tools_available=self._tools_cache,
            )
        return result

    async def list_tools(self) -> Any:
        """List available tools and cache their names for step metadata.

        Returns:
            The tool list from the underlying client.
        """
        result = await self._client.list_tools()
        tools = getattr(result, "tools", result)
        if isinstance(tools, list):
            names = []
            for tool in tools:
                name = getattr(tool, "name", None)
                if name is None and isinstance(tool, dict):
                    name = tool.get("name")
                if name:
                    names.append(str(name))
            self._tools_cache = names
        return result

    def close(self) -> None:
        """End the task and close the recorder."""
        self._recorder.end_task(self._task_id, "success")
        self._recorder.close()

    def __getattr__(self, name: str) -> Any:
        """Proxy all other attribute access to the wrapped client."""
        return getattr(self._client, name)


def _extract_result_text(result: Any) -> str | None:
    """Best-effort extraction of text from an MCP tool result."""
    if result is None:
        return None
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(str(text))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        if parts:
            return "\n".join(parts)
    text = getattr(result, "text", None)
    if text is not None:
        return str(text)
    return str(result)


__all__ = [
    "XRayMCPProxy",
]
