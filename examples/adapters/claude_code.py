from __future__ import annotations

from typing import Iterable, Mapping

from agent_xray.schema import AgentStep


def adapt_claude_code(records: Iterable[Mapping[str, object]]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for index, record in enumerate(records, start=1):
        payload = {
            "task_id": record.get("task_id", "claude-code-task"),
            "step": record.get("step", index),
            "tool_name": record.get("tool_name"),
            "tool_input": record.get("tool_input", {}),
            "tool_result": record.get("tool_result"),
            "llm_reasoning": record.get("llm_reasoning"),
            "error": record.get("error"),
            "timestamp": record.get("timestamp"),
            "tools_available": record.get("tools_available") or record.get("tools_available_names"),
        }
        if payload["tool_name"]:
            steps.append(AgentStep.from_dict(payload))
    return steps
