from __future__ import annotations

from agent_xray.schema import AGENT_STEP_JSON_SCHEMA, AgentStep


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
