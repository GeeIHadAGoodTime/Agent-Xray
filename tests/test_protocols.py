from __future__ import annotations

from agent_xray.protocols import (
    StaticPromptBuilder,
    StaticToolRegistry,
    coerce_step,
    coerce_steps,
)
from agent_xray.schema import AgentStep, AgentTask


def test_static_tool_registry_from_descriptions():
    reg = StaticToolRegistry(descriptions={"click": "Click element", "fill": "Fill form"})
    assert sorted(reg.tool_names()) == ["click", "fill"]
    assert reg.describe("click") == "Click element"
    assert reg.describe("missing") is None


def test_static_tool_registry_from_names():
    reg = StaticToolRegistry(names=["a", "b"])
    assert reg.tool_names() == ["a", "b"]


def test_static_tool_registry_uses_step_tools(sample_step: AgentStep):
    reg = StaticToolRegistry(names=["fallback"])
    names = reg.tool_names(step=sample_step)
    assert "browser_click" in names


def test_static_prompt_builder():
    builder = StaticPromptBuilder(prompt="You are an agent.")
    task = AgentTask(task_id="t1")
    assert builder.build_prompt(task) == "You are an agent."


def test_coerce_step_valid():
    step = coerce_step({"task_id": "t1", "step": 1, "tool_name": "click", "tool_input": {}})
    assert step is not None
    assert step.tool_name == "click"


def test_coerce_step_missing_tool_name():
    assert coerce_step({"task_id": "t1", "step": 1}) is None


def test_coerce_step_empty():
    assert coerce_step({}) is None


def test_coerce_steps_filters_invalid():
    records = [
        {"task_id": "t1", "step": 1, "tool_name": "a", "tool_input": {}},
        {"task_id": "t1", "step": 2},
        {"task_id": "t1", "step": 3, "tool_name": "b", "tool_input": {}},
    ]
    steps = coerce_steps(records)
    assert len(steps) == 2
    assert steps[0].tool_name == "a"
    assert steps[1].tool_name == "b"
