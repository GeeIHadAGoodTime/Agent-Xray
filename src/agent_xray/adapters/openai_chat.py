"""Adapter for raw OpenAI Chat Completions message transcripts.

This module handles assistant messages with ``tool_calls`` entries paired with
``tool`` role messages keyed by ``tool_call_id``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import AgentStep
from . import _coerce_text, _iter_json_objects, _safe_json_dict, _usage_metadata


def _tool_result_from_call(tool_call: dict[str, Any]) -> str | None:
    for key in ("output", "result"):
        if tool_call.get(key) is not None:
            return _coerce_text(tool_call.get(key))
    function = tool_call.get("function")
    if isinstance(function, dict):
        for key in ("output", "result"):
            if function.get(key) is not None:
                return _coerce_text(function.get(key))
    return None


def load(path: Path) -> list[AgentStep]:
    """Load a raw OpenAI Chat Completions trace file.

    Args:
        path: JSONL transcript containing assistant ``tool_calls`` and ``tool`` replies.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    steps: list[AgentStep] = []
    pending_by_call_id: dict[str, AgentStep] = {}
    task_id = path.stem

    for _, payload in _iter_json_objects(path):
        task_id = str(payload.get("conversation_id") or payload.get("task_id") or task_id)

        if payload.get("role") == "assistant" and isinstance(payload.get("tool_calls"), list):
            model_name = payload.get("model")
            input_tokens, output_tokens, duration_ms = _usage_metadata(payload.get("usage"))
            for tool_call in payload.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict) or function.get("name") is None:
                    continue
                step = AgentStep.from_dict(
                    {
                        "task_id": task_id,
                        "step": len(steps) + 1,
                        "tool_name": function.get("name"),
                        "tool_input": _safe_json_dict(function.get("arguments")),
                        "tool_result": _tool_result_from_call(tool_call),
                        "llm_reasoning": _coerce_text(payload.get("content")),
                        "model_name": None if model_name is None else str(model_name),
                        "duration_ms": duration_ms,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                )
                steps.append(step)
                call_id = tool_call.get("id")
                if isinstance(call_id, str):
                    pending_by_call_id[call_id] = step
            continue

        if payload.get("role") == "tool":
            call_id = payload.get("tool_call_id")
            if isinstance(call_id, str) and call_id in pending_by_call_id:
                pending_by_call_id[call_id].tool_result = _coerce_text(payload.get("content"))

    return steps
