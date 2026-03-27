from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _coerce_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _coerce_list_of_str(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]


@dataclass(slots=True)
class AgentStep:
    task_id: str
    step: int
    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str | None = None
    llm_reasoning: str | None = None
    error: str | None = None
    duration_ms: int | None = None
    timestamp: str | None = None
    model_name: str | None = None
    temperature: float | None = None
    tool_choice: str | None = None
    message_count: int | None = None
    tools_available: list[str] | None = None
    page_url: str | None = None
    system_prompt_hash: str | None = None
    context_usage_pct: float | None = None
    context_window: int | None = None
    compaction_count: int | None = None
    snapshot_compressed: bool | None = None
    had_screenshot: bool | None = None
    correction_messages: list[str] | None = None
    spin_intervention: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> AgentStep:
        return cls(
            task_id=str(payload.get("task_id", "")),
            step=int(payload.get("step", 0) or 0),
            tool_name=str(payload.get("tool_name", "")),
            tool_input=_coerce_dict(payload.get("tool_input")),
            tool_result=(
                None if payload.get("tool_result") is None else str(payload.get("tool_result"))
            ),
            llm_reasoning=(
                None if payload.get("llm_reasoning") is None else str(payload.get("llm_reasoning"))
            ),
            error=None if payload.get("error") is None else str(payload.get("error")),
            duration_ms=(
                None if payload.get("duration_ms") is None else int(payload.get("duration_ms"))
            ),
            timestamp=payload.get("timestamp") or payload.get("ts"),
            model_name=(
                None if payload.get("model_name") is None else str(payload.get("model_name"))
            ),
            temperature=(
                None if payload.get("temperature") is None else float(payload.get("temperature"))
            ),
            tool_choice=(
                None if payload.get("tool_choice") is None else str(payload.get("tool_choice"))
            ),
            message_count=(
                None if payload.get("message_count") is None else int(payload.get("message_count"))
            ),
            tools_available=_coerce_list_of_str(
                payload.get("tools_available") or payload.get("tools_available_names")
            ),
            page_url=None if payload.get("page_url") is None else str(payload.get("page_url")),
            system_prompt_hash=(
                None
                if payload.get("system_prompt_hash") is None
                else str(payload.get("system_prompt_hash"))
            ),
            context_usage_pct=(
                None
                if payload.get("context_usage_pct") is None
                else float(payload.get("context_usage_pct"))
            ),
            context_window=(
                None
                if payload.get("context_window") is None
                else int(payload.get("context_window"))
            ),
            compaction_count=(
                None
                if payload.get("compaction_count") is None
                else int(payload.get("compaction_count"))
            ),
            snapshot_compressed=payload.get("snapshot_compressed"),
            had_screenshot=(
                payload.get("had_screenshot")
                if "had_screenshot" in payload
                else payload.get("had_screenshot_image")
            ),
            correction_messages=_coerce_list_of_str(payload.get("correction_messages")),
            spin_intervention=(
                None
                if payload.get("spin_intervention") is None
                else str(payload.get("spin_intervention"))
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TaskOutcome:
    task_id: str
    status: str
    final_answer: str | None = None
    total_steps: int | None = None
    total_duration_s: float | None = None
    timestamp: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TaskOutcome:
        metadata = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "event",
                "task_id",
                "outcome",
                "status",
                "final_answer",
                "total_steps",
                "total_duration_s",
                "timestamp",
                "ts",
            }
        }
        return cls(
            task_id=str(payload.get("task_id", "")),
            status=str(payload.get("outcome") or payload.get("status") or ""),
            final_answer=(
                None if payload.get("final_answer") is None else str(payload.get("final_answer"))
            ),
            total_steps=(
                None if payload.get("total_steps") is None else int(payload.get("total_steps"))
            ),
            total_duration_s=(
                None
                if payload.get("total_duration_s") is None
                else float(payload.get("total_duration_s"))
            ),
            timestamp=payload.get("timestamp") or payload.get("ts"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AgentTask:
    task_id: str
    steps: list[AgentStep] = field(default_factory=list)
    task_text: str | None = None
    task_category: str | None = None
    day: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    outcome: TaskOutcome | None = None

    @property
    def sorted_steps(self) -> list[AgentStep]:
        return sorted(self.steps, key=lambda step: step.step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_text": self.task_text,
            "task_category": self.task_category,
            "day": self.day,
            "metadata": self.metadata,
            "outcome": self.outcome.to_dict() if self.outcome else None,
            "steps": [step.to_dict() for step in self.sorted_steps],
        }


AGENT_STEP_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentStep",
    "type": "object",
    "required": ["task_id", "step", "tool_name", "tool_input"],
    "properties": {
        "task_id": {"type": "string"},
        "step": {"type": "integer", "minimum": 0},
        "tool_name": {"type": "string"},
        "tool_input": {"type": "object"},
        "tool_result": {"type": ["string", "null"]},
        "llm_reasoning": {"type": ["string", "null"]},
        "error": {"type": ["string", "null"]},
        "duration_ms": {"type": ["integer", "null"], "minimum": 0},
        "timestamp": {"type": ["string", "null"]},
        "model_name": {"type": ["string", "null"]},
        "temperature": {"type": ["number", "null"]},
        "tool_choice": {"type": ["string", "null"]},
        "message_count": {"type": ["integer", "null"], "minimum": 0},
        "tools_available": {"type": ["array", "null"], "items": {"type": "string"}},
        "page_url": {"type": ["string", "null"]},
        "system_prompt_hash": {"type": ["string", "null"]},
        "context_usage_pct": {"type": ["number", "null"]},
        "context_window": {"type": ["integer", "null"], "minimum": 0},
        "compaction_count": {"type": ["integer", "null"], "minimum": 0},
        "snapshot_compressed": {"type": ["boolean", "null"]},
        "had_screenshot": {"type": ["boolean", "null"]},
        "correction_messages": {"type": ["array", "null"], "items": {"type": "string"}},
        "spin_intervention": {"type": ["string", "null"]},
    },
    "additionalProperties": True,
}

TASK_OUTCOME_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "TaskOutcome",
    "type": "object",
    "required": ["task_id", "status"],
    "properties": {
        "task_id": {"type": "string"},
        "status": {"type": "string"},
        "final_answer": {"type": ["string", "null"]},
        "total_steps": {"type": ["integer", "null"], "minimum": 0},
        "total_duration_s": {"type": ["number", "null"], "minimum": 0},
        "timestamp": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    },
    "additionalProperties": True,
}

AGENT_TASK_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "AgentTask",
    "type": "object",
    "required": ["task_id", "steps"],
    "properties": {
        "task_id": {"type": "string"},
        "task_text": {"type": ["string", "null"]},
        "task_category": {"type": ["string", "null"]},
        "day": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
        "outcome": {"anyOf": [{"type": "null"}, TASK_OUTCOME_JSON_SCHEMA]},
        "steps": {"type": "array", "items": AGENT_STEP_JSON_SCHEMA},
    },
    "additionalProperties": True,
}
