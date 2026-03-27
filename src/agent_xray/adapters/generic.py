"""Adapter for loosely structured generic JSONL step traces.

This module handles records that already expose ``tool_name`` and ``tool_input``
fields either at the top level or inside common nested payload keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import AgentStep
from . import _iter_json_objects, _normalize_tool_input

_NESTED_KEYS = ("step", "agent_step", "record", "payload", "data")


def _normalized_record(payload: dict[str, Any], path: Path) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = [payload]
    for key in _NESTED_KEYS:
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append({**payload, **value})
    for value in payload.values():
        if isinstance(value, dict):
            candidates.append({**payload, **value})
    for candidate in candidates:
        tool_name = candidate.get("tool_name")
        if tool_name is None:
            continue
        record = dict(candidate)
        record["task_id"] = str(record.get("task_id") or path.stem)
        record["tool_name"] = str(tool_name)
        record["tool_input"] = _normalize_tool_input(record.get("tool_input"))
        return record
    return None


def load(path: Path) -> list[AgentStep]:
    """Load a generic JSONL trace file.

    Args:
        path: JSONL file containing agent steps in a near-normalized structure.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    steps: list[AgentStep] = []
    for _, payload in _iter_json_objects(path):
        record = _normalized_record(payload, path)
        if record is None:
            continue
        record.setdefault("step", len(steps) + 1)
        steps.append(AgentStep.from_dict(record))
    return steps
