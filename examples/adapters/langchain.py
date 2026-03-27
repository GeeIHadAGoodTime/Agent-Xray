from __future__ import annotations

from typing import Iterable, Mapping

from agent_xray.schema import AgentStep


def adapt_langchain(events: Iterable[Mapping[str, object]], task_id: str = "langchain-task") -> list[AgentStep]:
    steps: list[AgentStep] = []
    step_num = 0
    for event in events:
        if event.get("type") != "agent_action":
            continue
        step_num += 1
        payload = {
            "task_id": task_id,
            "step": step_num,
            "tool_name": event.get("tool"),
            "tool_input": {"input": event.get("tool_input")},
            "tool_result": event.get("observation"),
            "llm_reasoning": event.get("log"),
            "error": event.get("error"),
        }
        if payload["tool_name"]:
            steps.append(AgentStep.from_dict(payload))
    return steps
