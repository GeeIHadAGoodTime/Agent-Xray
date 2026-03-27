"""Trace a LangChain agent with a callback that writes agent-xray JSONL."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

TRACE_DIR = Path("traces")
TRACE_DIR.mkdir(exist_ok=True)
trace_file = TRACE_DIR / "langchain_run.jsonl"


class AgentXrayCallback(BaseCallbackHandler):
    """LangChain callback that writes agent-xray JSONL on each tool call."""

    def __init__(self, task_id: str, trace_path: Path) -> None:
        self.task_id = task_id
        self.trace = trace_path.open("a")
        self.step = 0
        self._tool_starts: dict[str, float] = {}

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._tool_starts[str(run_id)] = time.monotonic()

    def on_tool_end(self, output: str, *, run_id: UUID, **kwargs: Any) -> None:
        self.step += 1
        start = self._tool_starts.pop(str(run_id), time.monotonic())
        name = kwargs.get("name", "unknown_tool")
        self.trace.write(
            json.dumps(
                {
                    "task_id": self.task_id,
                    "step": self.step,
                    "tool_name": name,
                    "tool_input": {},
                    "tool_result": output[:500],
                    "duration_ms": int((time.monotonic() - start) * 1000),
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            )
            + "\n"
        )
        self.trace.flush()

    def on_chain_end(self, outputs: dict[str, Any], **kwargs: Any) -> None:
        self.trace.write(
            json.dumps(
                {
                    "event": "task_complete",
                    "task_id": self.task_id,
                    "status": "success",
                    "total_steps": self.step,
                }
            )
            + "\n"
        )
        self.trace.close()


# Usage:
#
#   from langchain.agents import AgentExecutor
#   cb = AgentXrayCallback("lang-task-1", trace_file)
#   executor = AgentExecutor(agent=agent, tools=tools, callbacks=[cb])
#   executor.invoke({"input": "Search for the latest AI news"})
#
#   # Then analyze:
#   #   agent-xray analyze ./traces

print("AgentXrayCallback defined. See docstring for usage.")
print(f"Traces will be written to {trace_file}")
