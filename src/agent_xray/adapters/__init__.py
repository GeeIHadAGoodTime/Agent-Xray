"""Framework-specific adapters for converting trace logs to ``AgentStep`` records.

This module contains shared parsing helpers plus adapter registration and
auto-detection logic for supported trace formats.
"""

from __future__ import annotations

import importlib
import json
import warnings
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from ..schema import AgentStep

FORMATS = {
    "generic": "agent_xray.adapters.generic",
    "openai": "agent_xray.adapters.openai_sdk",
    "openai_chat": "agent_xray.adapters.openai_chat",
    "langchain": "agent_xray.adapters.langchain",
    "anthropic": "agent_xray.adapters.anthropic",
    "crewai": "agent_xray.adapters.crewai",
    "otel": "agent_xray.adapters.otel",
    "auto": None,
}


def _warn(path: Path, message: str, *, line_number: int | None = None) -> None:
    location = f"{path}"
    if line_number is not None:
        location = f"{location}:{line_number}"
    warnings.warn(f"{location}: {message}", stacklevel=2)


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("#") or stripped.startswith("//")


def _iter_json_objects(
    path: Path, *, limit: int | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    seen = 0
    with path.open(encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or _is_comment_line(line):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                _warn(path, "skipping malformed JSON line", line_number=line_number)
                continue
            if not isinstance(payload, dict):
                _warn(path, "skipping non-object JSON line", line_number=line_number)
                continue
            yield line_number, {str(key): value for key, value in payload.items()}
            seen += 1
            if limit is not None and seen >= limit:
                break


def _normalize_tool_input(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if value is None:
        return {}
    return {"value": value}


def _safe_json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str):
        return _normalize_tool_input(value)
    text = value.strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"arguments": value}
    if isinstance(parsed, dict):
        return {str(key): item for key, item in parsed.items()}
    return {"arguments": parsed}


def _coerce_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text") is not None:
                    parts.append(str(item["text"]))
                elif item.get("content") is not None:
                    parts.append(str(item["content"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=True, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        if value.get("text") is not None:
            return str(value["text"])
        if value.get("content") is not None:
            return _coerce_text(value.get("content"))
        return json.dumps(value, ensure_ascii=True, sort_keys=True)
    return str(value)


def _usage_metadata(usage: Any) -> tuple[int | None, int | None, int | None]:
    if not isinstance(usage, dict):
        return (None, None, None)
    input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("output_tokens")
    if output_tokens is None:
        output_tokens = usage.get("completion_tokens")
    duration_ms = usage.get("duration_ms")
    if duration_ms is None:
        duration_ms = usage.get("total_duration_ms")
    if duration_ms is None:
        duration_ms = usage.get("latency_ms")
    return (
        None if input_tokens is None else int(input_tokens),
        None if output_tokens is None else int(output_tokens),
        None if duration_ms is None else int(duration_ms),
    )


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _duration_ms(start: Any, end: Any) -> int | None:
    start_time = _parse_timestamp(start)
    end_time = _parse_timestamp(end)
    if start_time is None or end_time is None:
        return None
    delta_ms = int((end_time - start_time).total_seconds() * 1000)
    return delta_ms if delta_ms >= 0 else None


def _contains_content_type(payload: dict[str, Any], content_type: str) -> bool:
    content = payload.get("content")
    if isinstance(content, list):
        return any(isinstance(item, dict) and item.get("type") == content_type for item in content)
    return payload.get("type") == content_type


def _has_nested_tool_name(payload: dict[str, Any]) -> bool:
    if payload.get("tool_name") is not None and payload.get("task_id") is not None:
        return True
    return any(
        isinstance(value, dict) and value.get("tool_name") is not None for value in payload.values()
    )


def _sample_payloads(path: Path, *, limit: int = 25) -> list[dict[str, Any]]:
    return [payload for _, payload in _iter_json_objects(path, limit=limit)]


def _otel_text_score(path: Path) -> int:
    try:
        with path.open(encoding="utf-8", errors="ignore") as handle:
            head = handle.read(8192)
    except OSError:
        return 0
    if '"resourceSpans"' in head or '"scopeSpans"' in head:
        return 30
    return 0


def _heuristic_scores(path: Path, payloads: list[dict[str, Any]]) -> dict[str, int]:
    scores = {name: 0 for name in FORMATS if name != "auto"}
    scores["otel"] += _otel_text_score(path)

    for payload in payloads:
        object_type = str(payload.get("object") or "")
        if object_type == "run":
            scores["openai"] += 10
        elif object_type == "run_step":
            scores["openai"] += 8
            if payload.get("type") == "tool_calls":
                scores["openai"] += 4

        if payload.get("role") == "assistant" and isinstance(payload.get("tool_calls"), list):
            scores["openai"] += 3
            scores["openai_chat"] += 10
            for tool_call in payload.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict) and function.get("name") is not None:
                    scores["openai_chat"] += 2

        if payload.get("role") == "tool" and payload.get("tool_call_id") is not None:
            scores["openai"] += 1
            scores["openai_chat"] += 7

        event = str(payload.get("event") or "")
        if event in {"agent_action", "on_tool_start", "on_tool_end", "on_llm_start", "on_llm_end"}:
            scores["langchain"] += 10

        if _contains_content_type(payload, "tool_use") or _contains_content_type(
            payload, "tool_result"
        ):
            scores["anthropic"] += 10

        if "agent_role" in payload or any(str(key).startswith("crew_") for key in payload):
            scores["crewai"] += 10

        if _has_nested_tool_name(payload):
            scores["generic"] += 6

    return scores


def _load_count(path: Path, format_name: str) -> int:
    module_name = FORMATS[format_name]
    if module_name is None:
        return 0
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            module = importlib.import_module(module_name)
            load = getattr(module, "load", None)
            if not callable(load):
                return 0
            return sum(1 for step in load(path) if isinstance(step, AgentStep))
    except Exception:
        return 0


def format_info(path: str | Path) -> tuple[str, float]:
    """Detect the best matching adapter and return its confidence score.

    Args:
        path: Trace file to inspect.

    Returns:
        A ``(format_name, confidence)`` tuple where confidence is a float in the
        inclusive range ``0.0`` to ``1.0``.
    """

    resolved_path = Path(path) if not isinstance(path, Path) else path
    payloads = _sample_payloads(resolved_path)
    heuristic_scores = _heuristic_scores(resolved_path, payloads)

    candidates = [name for name, score in heuristic_scores.items() if score > 0]
    if "generic" not in candidates:
        candidates.append("generic")

    combined_scores: dict[str, int] = {}
    for format_name in candidates:
        combined_scores[format_name] = heuristic_scores.get(format_name, 0) * 10 + (
            _load_count(resolved_path, format_name) * 5
        )

    ranked = sorted(
        combined_scores.items(),
        key=lambda item: (
            item[1],
            heuristic_scores.get(item[0], 0),
            -list(FORMATS).index(item[0]),
        ),
        reverse=True,
    )
    detected_format, top_score = ranked[0] if ranked else ("generic", 0)
    second_score = ranked[1][1] if len(ranked) > 1 else 0
    if top_score <= 0:
        confidence = 0.0
    elif second_score <= 0:
        confidence = 1.0
    else:
        confidence = round(top_score / (top_score + second_score), 2)
    return detected_format, confidence


def autodetect(path: str | Path) -> str:
    """Guess the most likely trace format for a file."""

    return format_info(path)[0]


def adapt(path: str | Path, format: str = "auto") -> list[AgentStep]:
    """Load a trace file and convert all entries to ``AgentStep`` records.

    Args:
        path: Trace file path.
        format: Adapter name, or ``"auto"`` to detect it from the file.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    path = Path(path) if not isinstance(path, Path) else path
    if format not in FORMATS:
        raise ValueError(f"unsupported trace format: {format}")
    if format == "auto":
        format = autodetect(path)
    module_name = FORMATS.get(format)
    if module_name is None:
        raise ValueError("auto format must be resolved before adapter import")
    module = importlib.import_module(module_name)
    load = getattr(module, "load", None)
    if not callable(load):
        raise ValueError(f"adapter module did not expose a callable load(): {module_name}")
    return [step for step in load(path) if isinstance(step, AgentStep)]


__all__ = [
    "FORMATS",
    "adapt",
    "autodetect",
    "format_info",
]
