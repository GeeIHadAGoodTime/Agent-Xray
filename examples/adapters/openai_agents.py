from __future__ import annotations

from typing import Iterable, Mapping

from agent_xray.schema import AgentStep


def adapt_openai_agents(messages: Iterable[Mapping[str, object]], task_id: str = "openai-agents-task") -> list[AgentStep]:
    steps: list[AgentStep] = []
    step_num = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls", []) or []:
            step_num += 1
            function = tool_call.get("function", {})
            payload = {
                "task_id": task_id,
                "step": step_num,
                "tool_name": function.get("name"),
                "tool_input": {"arguments": function.get("arguments")},
                "llm_reasoning": message.get("content"),
                "model_name": message.get("model"),
            }
            if payload["tool_name"]:
                steps.append(AgentStep.from_dict(payload))
    return steps
