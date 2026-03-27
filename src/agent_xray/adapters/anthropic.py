"""Adapter for Anthropic message traces with tool_use and tool_result blocks.

This module handles Claude-style assistant content arrays that emit
``tool_use`` blocks and user messages that carry matching ``tool_result`` blocks.
"""

from __future__ import annotations

from pathlib import Path

from ..schema import AgentStep
from . import _coerce_text, _iter_json_objects, _normalize_tool_input, _usage_metadata


def load(path: Path) -> list[AgentStep]:
    """Load an Anthropic message trace file.

    Args:
        path: JSONL transcript containing Anthropic tool-use message blocks.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    steps: list[AgentStep] = []
    pending_by_id: dict[str, AgentStep] = {}
    task_id = path.stem

    for _, payload in _iter_json_objects(path):
        task_id = str(payload.get("task_id") or payload.get("conversation_id") or task_id)
        role = payload.get("role")
        content = payload.get("content")

        if role == "assistant" and isinstance(content, list):
            model_name = payload.get("model")
            input_tokens, output_tokens, _ = _usage_metadata(payload.get("usage"))
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_use_id = block.get("id")
                step = AgentStep.from_dict(
                    {
                        "task_id": task_id,
                        "step": len(steps) + 1,
                        "tool_name": block.get("name"),
                        "tool_input": _normalize_tool_input(block.get("input")),
                        "model_name": (str(model_name) if model_name is not None else None),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
                steps.append(step)
                if isinstance(tool_use_id, str):
                    pending_by_id[tool_use_id] = step
            continue

        if role == "user" and isinstance(content, list):
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                tool_use_id = block.get("tool_use_id")
                if isinstance(tool_use_id, str) and tool_use_id in pending_by_id:
                    pending_by_id[tool_use_id].tool_result = _coerce_text(block.get("content"))

    return steps
