from __future__ import annotations

from typing import Iterable, Mapping

from agent_xray.schema import AgentStep


def adapt_records(records: Iterable[Mapping[str, object]]) -> list[AgentStep]:
    steps: list[AgentStep] = []
    for record in records:
        if "tool_name" not in record or "task_id" not in record:
            continue
        steps.append(AgentStep.from_dict(dict(record)))
    return steps
