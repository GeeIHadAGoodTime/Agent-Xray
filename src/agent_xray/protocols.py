from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .schema import AgentStep, AgentTask


class ToolRegistry(Protocol):
    """Protocol for tool-surface lookup and tool descriptions."""

    def tool_names(
        self, *, task: AgentTask | None = None, step: AgentStep | None = None
    ) -> list[str]:
        """Return tool names visible for a task or specific step."""
        ...

    def describe(self, tool_name: str) -> str | None:
        """Return a human-readable description for a tool, if known."""
        ...


class PromptBuilder(Protocol):
    """Protocol for reconstructing a prompt for a task."""

    def build_prompt(self, task: AgentTask) -> str | None:
        """Return a prompt string for the task, if one can be reconstructed."""
        ...


class StepAdapter(Protocol):
    """Protocol for converting raw records into :class:`AgentStep` objects."""

    def adapt_record(self, record: Mapping[str, Any]) -> AgentStep | None:
        """Convert one raw record into an ``AgentStep`` or return ``None``."""
        ...


@dataclass(slots=True)
class StaticToolRegistry:
    """In-memory :class:`ToolRegistry` implementation for tests and scripts.

    Attributes:
        descriptions: Mapping of tool name to description text.
        names: Optional fixed list of tool names to return when step metadata is
            missing.
    """

    descriptions: dict[str, str] = field(default_factory=dict)
    names: list[str] = field(default_factory=list)

    def tool_names(
        self, *, task: AgentTask | None = None, step: AgentStep | None = None
    ) -> list[str]:
        """Return step-specific tools, configured names, or described names."""
        if step and step.tools and step.tools.tools_available is not None:
            return list(step.tools.tools_available)
        if self.names:
            return list(self.names)
        return sorted(self.descriptions)

    def describe(self, tool_name: str) -> str | None:
        """Look up a configured description for ``tool_name``."""
        return self.descriptions.get(tool_name)


@dataclass(slots=True)
class StaticPromptBuilder:
    """Prompt builder that always returns one static prompt string."""

    prompt: str

    def build_prompt(self, task: AgentTask) -> str | None:
        """Return the configured prompt unchanged for every task."""
        return self.prompt


def coerce_step(record: Mapping[str, Any]) -> AgentStep | None:
    if not record:
        return None
    tool_name = record.get("tool_name")
    task_id = record.get("task_id")
    if tool_name is None or task_id is None:
        return None
    normalized = {str(key): value for key, value in record.items()}
    return AgentStep.from_dict(normalized)


def coerce_steps(records: Iterable[Mapping[str, Any]]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for record in records:
        step = coerce_step(record)
        if step is not None:
            steps.append(step)
    return steps
