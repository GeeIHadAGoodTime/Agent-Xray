from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.signals.multi_agent import MultiAgentDetector


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    extensions: dict[str, object] | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-multi",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        extensions=extensions or {},
    )


def test_multi_agent_detector_detects_sub_agent_spawn() -> None:
    detector = MultiAgentDetector()
    signals = detector.detect_step(
        _step(1, "spawn_agent", {"agent_name": "Researcher", "depth": 1})
    )
    assert signals["is_delegation"] is True
    assert signals["is_sub_agent_call"] is True
    assert signals["delegation_target"] == "Researcher"


def test_multi_agent_detector_detects_handoff_pattern() -> None:
    detector = MultiAgentDetector()
    signals = detector.detect_step(
        _step(1, "respond", {"message": "Handoff to analyst for synthesis"})
    )
    assert signals["is_handoff"] is True
    assert signals["is_delegation"] is True
    assert signals["delegation_target"] == "analyst"


def test_multi_agent_detector_summarizes_unique_agents() -> None:
    detector = MultiAgentDetector()
    steps = [
        _step(1, "spawn_agent", {"agent_name": "Researcher"}),
        _step(2, "web_search", {"query": "x", "agent_role": "Researcher"}),
        _step(3, "route_task", {"target_agent": "Analyst"}),
        _step(4, "summarize", {"agent_role": "Analyst"}),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-multi", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["delegation_count"] == 2
    assert summary["unique_agents"] == 2


def test_multi_agent_detector_delegation_success_rate_tracks_followthrough() -> None:
    detector = MultiAgentDetector()
    steps = [
        _step(1, "spawn_agent", {"agent_name": "Researcher"}, tool_result="Assigned successfully."),
        _step(2, "web_search", {"query": "x", "agent_role": "Researcher"}),
        _step(3, "delegate_task", {"delegate_to": "Reviewer"}, error="Reviewer unavailable"),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-multi", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["delegation_success_rate"] == 0.5


def test_multi_agent_detector_tracks_max_delegation_depth() -> None:
    detector = MultiAgentDetector()
    steps = [
        _step(1, "spawn_agent", {"agent_name": "Planner", "depth": 1}),
        _step(2, "spawn_agent", {"agent_name": "Worker", "depth": 2}),
        _step(3, "send_input", {"agent_id": "Worker", "depth": 3}),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-multi", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["max_delegation_depth"] == 3


def test_multi_agent_detector_ignores_single_agent_steps() -> None:
    detector = MultiAgentDetector()
    step = _step(1, "web_search", {"query": "plain task"})
    signals = detector.detect_step(step)
    summary = detector.summarize(AgentTask(task_id="task-multi", steps=[step]), [signals])
    assert signals["is_delegation"] is False
    assert summary["delegation_count"] == 0
