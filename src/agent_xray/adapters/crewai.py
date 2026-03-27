"""Adapter for CrewAI task and tool execution traces.

This module handles CrewAI-flavored JSONL logs that expose agent role metadata
alongside tool execution inputs and outputs.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import AgentStep
from . import _coerce_text, _iter_json_objects, _normalize_tool_input


def load(path: Path) -> list[AgentStep]:
    """Load a CrewAI trace file.

    Args:
        path: JSONL file containing CrewAI task and tool records.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    steps: list[AgentStep] = []
    for _, payload in _iter_json_objects(path):
        tool_name = payload.get("tool") or payload.get("tool_name")
        if tool_name is None:
            continue
        task_id = str(
            payload.get("task_id") or payload.get("task") or payload.get("agent_role") or path.stem
        )
        tool_input = payload.get("tool_input")
        if isinstance(tool_input, dict):
            normalized_input = tool_input
        else:
            normalized_input = {
                "value": tool_input,
                "task": payload.get("task"),
                "agent_role": payload.get("agent_role"),
            }
        steps.append(
            AgentStep.from_dict(
                {
                    "task_id": task_id,
                    "step": len(steps) + 1,
                    "tool_name": str(tool_name),
                    "tool_input": _normalize_tool_input(normalized_input),
                    "tool_result": _coerce_text(payload.get("output")),
                    "timestamp": payload.get("timestamp") or payload.get("ts"),
                    "llm_reasoning": _coerce_text(payload.get("thought")),
                }
            )
        )
    return steps
