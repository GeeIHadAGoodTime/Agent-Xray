"""Trace MCP tool calls (e.g. Playwright browser) for agent-xray."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)
trace_file = TRACE_DIR / "mcp_run.jsonl"


class TracedMCPClient:
    """Wraps any MCP client to log tool calls as agent-xray JSONL."""

    def __init__(self, inner_client: Any, task_id: str, trace_path: Path) -> None:
        self._inner = inner_client
        self._task_id = task_id
        self._trace = trace_path.open("a")
        self._step = 0

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self._step += 1
        start = time.monotonic()
        try:
            result = self._inner.call_tool(name, arguments)
            result_text = str(result.content[0].text) if result.content else ""
            error = None
        except Exception as exc:
            result_text, error = None, str(exc)
            raise
        finally:
            self._trace.write(
                json.dumps(
                    {
                        "task_id": self._task_id,
                        "step": self._step,
                        "tool_name": name,
                        "tool_input": arguments,
                        "tool_result": result_text,
                        "error": error,
                        "duration_ms": int((time.monotonic() - start) * 1000),
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                )
                + "\n"
            )
            self._trace.flush()
        return result

    def finish(self, status: str = "success") -> None:
        self._trace.write(
            json.dumps(
                {
                    "event": "task_complete",
                    "task_id": self._task_id,
                    "status": status,
                    "total_steps": self._step,
                }
            )
            + "\n"
        )
        self._trace.close()


# Usage with a real MCP client (e.g. mcp.ClientSession):
#
#   from mcp import ClientSession
#   session = ClientSession(server_read, server_write)
#   traced = TracedMCPClient(session, "browse-checkout", trace_file)
#   traced.call_tool("browser_navigate", {"url": "https://shop.example.test"})
#   traced.call_tool("browser_click", {"ref": "add-to-cart"})
#   traced.call_tool("browser_snapshot", {})
#   traced.finish()
#
#   # Then analyze:
#   #   agent-xray analyze ./traces
#   #   agent-xray surface browse-checkout --log-dir ./traces

print("TracedMCPClient defined. See docstring for usage.")
print(f"Traces will be written to {trace_file}")
