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
    steps: list[AgentStep] = []
    pending_by_call_id: dict[str, AgentStep] = {}
    task_id = path.stem
    run_model: str | None = None
    run_input_tokens: int | None = None
    run_output_tokens: int | None = None
    run_duration_ms: int | None = None

    for _, payload in _iter_json_objects(path):
        if payload.get("object") == "run":
            task_id = str(payload.get("id") or task_id)
            run_model = None if payload.get("model") is None else str(payload.get("model"))
            run_input_tokens, run_output_tokens, run_duration_ms = _usage_metadata(
                payload.get("usage")
            )
            continue

        if payload.get("object") == "run_step" and payload.get("type") == "tool_calls":
            details = payload.get("step_details")
            tool_calls = details.get("tool_calls") if isinstance(details, dict) else None
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict) or function.get("name") is None:
                    continue
                input_tokens, output_tokens, duration_ms = _usage_metadata(payload.get("usage"))
                step = AgentStep.from_dict(
                    {
                        "task_id": task_id,
                        "step": len(steps) + 1,
                        "tool_name": function.get("name"),
                        "tool_input": _safe_json_dict(function.get("arguments")),
                        "tool_result": _tool_result_from_call(tool_call),
                        "model_name": run_model,
                        "duration_ms": duration_ms if duration_ms is not None else run_duration_ms,
                        "input_tokens": (
                            input_tokens if input_tokens is not None else run_input_tokens
                        ),
                        "output_tokens": (
                            output_tokens if output_tokens is not None else run_output_tokens
                        ),
                    }
                )
                steps.append(step)
            continue

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
                        "model_name": (str(model_name) if model_name is not None else run_model),
                        "duration_ms": duration_ms if duration_ms is not None else run_duration_ms,
                        "input_tokens": (
                            input_tokens if input_tokens is not None else run_input_tokens
                        ),
                        "output_tokens": (
                            output_tokens if output_tokens is not None else run_output_tokens
                        ),
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
