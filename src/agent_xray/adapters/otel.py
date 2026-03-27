"""Adapter for OpenTelemetry GenAI semantic convention span exports.

This module converts OTel JSON exports into ``AgentStep`` records by extracting
LLM spans, child tool spans, and associated GenAI attributes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from ..schema import AgentStep, ModelContext, ReasoningContext

opentelemetry = find_spec("opentelemetry.trace") is not None


def _require_otel() -> None:
    if not opentelemetry:
        raise ImportError(
            "OpenTelemetry adapter requires opentelemetry-api and opentelemetry-sdk. "
            "Install with: pip install agent-xray[otel]"
        )


def _decode_any_value(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in value:
            return value[key]
    if "arrayValue" in value:
        values = (
            value["arrayValue"].get("values", []) if isinstance(value["arrayValue"], dict) else []
        )
        return [_decode_any_value(item) for item in values]
    if "kvlistValue" in value and isinstance(value["kvlistValue"], dict):
        pairs = value["kvlistValue"].get("values", [])
        return {
            str(item.get("key")): _decode_any_value(item.get("value"))
            for item in pairs
            if isinstance(item, dict)
        }
    if "bytesValue" in value:
        return value["bytesValue"]
    return value


def _decode_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        if all(not isinstance(item, dict) for item in value.values()):
            return {str(key): item for key, item in value.items()}
        if "attributes" in value:
            value = value["attributes"]
    if isinstance(value, list):
        decoded: dict[str, Any] = {}
        for item in value:
            if not isinstance(item, dict):
                continue
            key = item.get("key")
            if key is None:
                continue
            decoded[str(key)] = _decode_any_value(item.get("value"))
        return decoded
    return {}


def _serialize_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def _parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return {}
    if text[0] not in "[{":
        return {"value": text}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"value": text}


def _first_attr(attributes: dict[str, Any], *names: str) -> Any:
    for name in names:
        if attributes.get(name) is not None:
            return attributes[name]
    return None


def _duration_ms(span: dict[str, Any]) -> int | None:
    start = span.get("startTimeUnixNano")
    end = span.get("endTimeUnixNano")
    try:
        if start is None or end is None:
            return None
        return max(0, int((int(end) - int(start)) / 1_000_000))
    except (TypeError, ValueError):
        return None


def _timestamp(span: dict[str, Any]) -> str | None:
    start = span.get("startTimeUnixNano")
    return str(start) if start is not None else None


def _task_id(trace_id: str, attributes: dict[str, Any]) -> str:
    for key in (
        "gen_ai.conversation.id",
        "session.id",
        "thread.id",
        "task.id",
        "workflow.id",
    ):
        value = attributes.get(key)
        if value:
            return str(value)
    return trace_id or "otel-trace"


def _is_genai_span(attributes: dict[str, Any], name: str) -> bool:
    return any(key.startswith("gen_ai.") for key in attributes) or name.lower().startswith(
        ("chat", "llm", "gen_ai")
    )


def _is_tool_span(attributes: dict[str, Any], name: str) -> bool:
    operation = str(attributes.get("gen_ai.operation.name") or "").lower()
    if operation in {"execute_tool", "tool"}:
        return True
    return (
        any(key in attributes for key in ("gen_ai.tool.name", "tool.name"))
        or "tool" in name.lower()
    )


def _model_context(attributes: dict[str, Any]) -> ModelContext | None:
    model_name = _first_attr(attributes, "gen_ai.request.model", "gen_ai.response.model")
    temperature = _first_attr(attributes, "gen_ai.request.temperature")
    tool_choice = _first_attr(attributes, "gen_ai.request.tool_choice")
    input_tokens = _first_attr(attributes, "gen_ai.usage.input_tokens", "llm.usage.prompt_tokens")
    output_tokens = _first_attr(
        attributes,
        "gen_ai.usage.output_tokens",
        "llm.usage.completion_tokens",
    )
    cost_usd = _first_attr(
        attributes,
        "gen_ai.usage.cost",
        "gen_ai.response.cost",
        "llm.cost.usd",
    )
    if all(
        value is None
        for value in (model_name, temperature, tool_choice, input_tokens, output_tokens, cost_usd)
    ):
        return None
    return ModelContext(
        model_name=None if model_name is None else str(model_name),
        temperature=None if temperature is None else float(temperature),
        tool_choice=None if tool_choice is None else str(tool_choice),
        input_tokens=None if input_tokens is None else int(input_tokens),
        output_tokens=None if output_tokens is None else int(output_tokens),
        cost_usd=None if cost_usd is None else float(cost_usd),
    )


def _tool_name(span: dict[str, Any], attributes: dict[str, Any]) -> str:
    return str(
        _first_attr(attributes, "gen_ai.tool.name", "tool.name", "tool")
        or attributes.get("gen_ai.operation.name")
        or span.get("name")
        or "respond"
    )


def _tool_input(attributes: dict[str, Any]) -> dict[str, Any]:
    arguments = _first_attr(
        attributes,
        "gen_ai.tool.call.arguments",
        "tool.arguments",
        "input",
        "gen_ai.input.messages",
    )
    if arguments is None:
        return {}
    parsed = _parse_jsonish(arguments)
    if isinstance(parsed, dict):
        return parsed
    if isinstance(parsed, list):
        return {"items": parsed}
    return {"value": parsed}


def _tool_result(attributes: dict[str, Any]) -> str | None:
    return _serialize_value(
        _first_attr(
            attributes,
            "gen_ai.tool.call.result",
            "output",
            "gen_ai.output.messages",
            "gen_ai.response.output_text",
        )
    )


def _reasoning(attributes: dict[str, Any]) -> ReasoningContext | None:
    text = _first_attr(
        attributes,
        "gen_ai.system_instructions",
        "gen_ai.input.messages",
    )
    serialized = _serialize_value(text)
    if serialized is None:
        return None
    return ReasoningContext(llm_reasoning=serialized)


def _error(span: dict[str, Any], attributes: dict[str, Any]) -> str | None:
    status = span.get("status")
    if isinstance(status, dict):
        message = status.get("message")
        if message:
            return str(message)
        code = status.get("code")
        if code not in (None, "STATUS_CODE_UNSET", 0):
            return str(code)
    error_type = attributes.get("error.type")
    return None if error_type is None else str(error_type)


def _iter_spans(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    spans: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for resource_span in payload.get("resourceSpans", []):
        resource_attributes = _decode_attributes(resource_span.get("resource", {}))
        for scope_span in resource_span.get("scopeSpans", []):
            for span in scope_span.get("spans", []):
                attributes = resource_attributes | _decode_attributes(span.get("attributes"))
                spans.append((str(span.get("traceId") or ""), span, attributes))
    return spans


def load(path: Path) -> list[AgentStep]:
    """Load an OpenTelemetry JSON export.

    Args:
        path: JSON file containing OTel span export data.

    Returns:
        A list of parsed ``AgentStep`` records.
    """

    _require_otel()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    spans_by_trace: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for trace_id, span, attributes in _iter_spans(payload):
        spans_by_trace[trace_id].append((span, attributes))

    steps: list[AgentStep] = []
    for trace_id, entries in spans_by_trace.items():
        entries.sort(key=lambda item: int(item[0].get("startTimeUnixNano") or 0))
        by_parent: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
        for span, attributes in entries:
            parent_id = str(span.get("parentSpanId") or "")
            by_parent[parent_id].append((span, attributes))

        step_number = 1
        llm_entries = [
            (span, attributes)
            for span, attributes in entries
            if _is_genai_span(attributes, str(span.get("name") or ""))
            and not _is_tool_span(attributes, str(span.get("name") or ""))
        ]
        if not llm_entries:
            llm_entries = entries

        trace_task_id = _task_id(trace_id, llm_entries[0][1] if llm_entries else {})
        for span, attributes in llm_entries:
            span_id = str(span.get("spanId") or "")
            model = _model_context(attributes)
            reasoning = _reasoning(attributes)
            child_tools = [
                (child_span, child_attributes)
                for child_span, child_attributes in by_parent.get(span_id, [])
                if _is_tool_span(child_attributes, str(child_span.get("name") or ""))
            ]
            if child_tools:
                for child_span, child_attributes in child_tools:
                    child_model = _model_context(child_attributes) or model
                    tool_input = _tool_input(child_attributes)
                    page_url = (
                        tool_input.get("url") if isinstance(tool_input.get("url"), str) else None
                    )
                    steps.append(
                        AgentStep(
                            task_id=trace_task_id,
                            step=step_number,
                            tool_name=_tool_name(child_span, child_attributes),
                            tool_input=tool_input,
                            tool_result=_tool_result(child_attributes),
                            error=_error(child_span, child_attributes),
                            duration_ms=_duration_ms(child_span),
                            timestamp=_timestamp(child_span),
                            model=child_model,
                            reasoning=reasoning,
                            page_url=page_url,
                        )
                    )
                    step_number += 1
                continue

            tool_input = _tool_input(attributes)
            steps.append(
                AgentStep(
                    task_id=trace_task_id,
                    step=step_number,
                    tool_name=_tool_name(span, attributes),
                    tool_input=tool_input,
                    tool_result=_tool_result(attributes),
                    error=_error(span, attributes),
                    duration_ms=_duration_ms(span),
                    timestamp=_timestamp(span),
                    model=model,
                    reasoning=reasoning,
                )
            )
            step_number += 1

    return sorted(steps, key=lambda item: (item.task_id, item.step))
