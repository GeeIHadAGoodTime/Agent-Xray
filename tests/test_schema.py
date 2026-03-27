from __future__ import annotations

from agent_xray.schema import AGENT_STEP_JSON_SCHEMA, AgentStep, AgentTask


def test_agent_step_round_trip() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 1,
            "tool_name": "browser_navigate",
            "tool_input": {"url": "https://example.test"},
            "tool_result": "ok",
            "tools_available": ["browser_navigate"],
        }
    )
    data = step.to_dict()
    assert data["task_id"] == "task-1"
    assert data["tool_input"]["url"] == "https://example.test"
    assert step.tools_available == ["browser_navigate"]


def test_agent_step_schema_required_fields() -> None:
    assert set(AGENT_STEP_JSON_SCHEMA["required"]) == {"task_id", "step", "tool_name", "tool_input"}


def test_agent_step_model_cost_round_trip() -> None:
    step = AgentStep.from_dict(
        {
            "task_id": "task-1",
            "step": 2,
            "tool_name": "respond",
            "tool_input": {},
            "model": {
                "model_name": "gpt-4o",
                "input_tokens": 100,
                "output_tokens": 50,
                "cost_usd": 0.0125,
            },
        }
    )
    assert step.model is not None
    assert step.model.model_name == "gpt-4o"
    assert step.cost_usd == 0.0125


def test_agent_task_from_steps_uses_step_task_id() -> None:
    task = AgentTask.from_steps([AgentStep("task-42", 1, "search", {"q": "hello"})])
    assert task.task_id == "task-42"
    assert len(task.steps) == 1
