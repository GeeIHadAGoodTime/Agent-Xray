from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import AgentStep, AgentTask


class ToolRegistry(Protocol):
    def tool_names(
        self, *, task: AgentTask | None = None, step: AgentStep | None = None
    ) -> list[str]: ...

    def describe(self, tool_name: str) -> str | None: ...


class PromptBuilder(Protocol):
    def build_prompt(self, task: AgentTask) -> str | None: ...


class StepAdapter(Protocol):
    def adapt_record(self, record: Mapping[str, Any]) -> AgentStep | None: ...


@dataclass(slots=True)
class StaticToolRegistry:
    descriptions: dict[str, str] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)

    def tool_names(
        self, *, task: AgentTask | None = None, step: AgentStep | None = None
    ) -> list[str]:
        if step and step.tools_available:
            return list(step.tools_available)
        if self.names:
            return list(self.names)
        return sorted(self.descriptions)

    def describe(self, tool_name: str) -> str | None:
        return self.descriptions.get(tool_name)


@dataclass(slots=True)
class StaticPromptBuilder:
    prompt: str

    def build_prompt(self, task: AgentTask) -> str | None:
        return self.prompt


def coerce_step(record: Mapping[str, Any]) -> AgentStep | None:
    if not record:
        return None
    tool_name = record.get("tool_name")
    task_id = record.get("task_id")
    if tool_name is None or task_id is None:
        return None
    return AgentStep.from_dict(dict(record))


def coerce_steps(records: Iterable[Mapping[str, Any]]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for record in records:
        step = coerce_step(record)
        if step is not None:
            steps.append(step)
    return steps
