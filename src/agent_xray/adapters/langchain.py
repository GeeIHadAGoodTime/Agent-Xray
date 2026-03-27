"""Adapter for LangChain callback event traces.

This module handles JSONL traces containing ``agent_action`` and
``on_tool_start``/``on_tool_end`` callback events.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..schema import AgentStep
from . import _coerce_text, _duration_ms, _iter_json_objects, _normalize_tool_input


def _event_timestamp(payload: dict[str, Any]) -> Any:
    for key in ("timestamp", "ts", "time", "start_time", "end_time"):
        if payload.get(key) is not None:
            return payload.get(key)
    return None


def _tool_input(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("input", "inputs"):
            if key in data:
                return _normalize_tool_input(data.get(key))
    if payload.get("tool_input") is not None:
        return _normalize_tool_input(payload.get("tool_input"))
    return {}


def _tool_output(payload: dict[str, Any]) -> str | None:
    data = payload.get("data")
    if isinstance(data, dict):
        for key in ("output", "outputs", "result", "observation"):
            if data.get(key) is not None:
                return _coerce_text(data.get(key))
    for key in ("output", "observation", "result"):
        if payload.get(key) is not None:
            return _coerce_text(payload.get(key))
    return None


def load(path: Path) -> list[AgentStep]:
    """Load a LangChain callback trace file.

    Args:
        path: JSONL file containing LangChain callback events.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    steps: list[AgentStep] = []
    starts_by_run: dict[str, dict[str, Any]] = {}
    ends_by_run: dict[str, dict[str, Any]] = {}

    for _, payload in _iter_json_objects(path):
        event = str(payload.get("event") or "")
        run_id = str(payload.get("run_id") or "")

        if event == "agent_action":
            tool_name = payload.get("tool") or payload.get("name")
            if tool_name is None:
                continue
            task_id = str(
                payload.get("task_id")
                or payload.get("metadata", {}).get("task_id")
                or payload.get("parent_run_id")
                or run_id
                or path.stem
            )
            steps.append(
                AgentStep.from_dict(
                    {
                        "task_id": task_id,
                        "step": len(steps) + 1,
                        "tool_name": str(tool_name),
                        "tool_input": _normalize_tool_input(payload.get("tool_input")),
                        "tool_result": _coerce_text(payload.get("observation")),
                        "llm_reasoning": _coerce_text(payload.get("log")),
                        "error": _coerce_text(payload.get("error")),
                        "timestamp": _event_timestamp(payload),
                    }
                )
            )
            continue

        if event == "on_tool_start" and run_id:
            starts_by_run[run_id] = payload
            if run_id in ends_by_run:
                end_payload = ends_by_run.pop(run_id)
                start_payload = starts_by_run.pop(run_id)
                task_id = str(
                    start_payload.get("metadata", {}).get("task_id")
                    or start_payload.get("parent_run_id")
                    or run_id
                    or path.stem
                )
                steps.append(
                    AgentStep.from_dict(
                        {
                            "task_id": task_id,
                            "step": len(steps) + 1,
                            "tool_name": start_payload.get("name") or "",
                            "tool_input": _tool_input(start_payload),
                            "tool_result": _tool_output(end_payload),
                            "duration_ms": _duration_ms(
                                _event_timestamp(start_payload), _event_timestamp(end_payload)
                            ),
                            "timestamp": _event_timestamp(start_payload),
                        }
                    )
                )
            continue

        if event == "on_tool_end" and run_id:
            if run_id not in starts_by_run:
                ends_by_run[run_id] = payload
                continue
            start_payload = starts_by_run.pop(run_id)
            task_id = str(
                start_payload.get("metadata", {}).get("task_id")
                or start_payload.get("parent_run_id")
                or run_id
                or path.stem
            )
            steps.append(
                AgentStep.from_dict(
                    {
                        "task_id": task_id,
                        "step": len(steps) + 1,
                        "tool_name": start_payload.get("name") or "",
                        "tool_input": _tool_input(start_payload),
                        "tool_result": _tool_output(payload),
                        "duration_ms": _duration_ms(
                            _event_timestamp(start_payload), _event_timestamp(payload)
                        ),
                        "timestamp": _event_timestamp(start_payload),
                    }
                )
            )

    return steps
