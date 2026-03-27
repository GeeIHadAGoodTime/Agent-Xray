from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.signals.planning import PlanningDetector


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    llm_reasoning: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-plan",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        llm_reasoning=llm_reasoning,
    )


def test_planning_detector_detects_plan_creation() -> None:
    detector = PlanningDetector()
    signals = detector.detect_step(
        _step(
            1,
            "create_plan",
            {"plan_id": "plan-alpha", "steps": ["search", "write", "verify"]},
        )
    )
    assert signals["is_plan_creation"] is True
    assert signals["plan_id"] == "plan-alpha"


def test_planning_detector_detects_plan_step_execution() -> None:
    detector = PlanningDetector()
    signals = detector.detect_step(
        _step(
            1,
            "execute_plan_step",
            {"plan_id": "plan-alpha", "step": 1},
            tool_result="Completed step 1",
        )
    )
    assert signals["is_plan_step"] is True
    assert signals["is_plan_revision"] is False


def test_planning_detector_detects_plan_revision() -> None:
    detector = PlanningDetector()
    signals = detector.detect_step(
        _step(1, "replan", {"plan_id": "plan-alpha"}, tool_result="Updated plan after failure")
    )
    assert signals["is_plan_revision"] is True


def test_planning_detector_summarizes_plan_metrics() -> None:
    detector = PlanningDetector()
    steps = [
        _step(1, "create_plan", {"plan_id": "plan-alpha", "steps": ["search", "write", "verify"]}),
        _step(2, "execute_plan_step", {"plan_id": "plan-alpha", "step": 1, "status": "completed"}),
        _step(
            3, "execute_plan_step", {"plan_id": "plan-alpha", "step": 2}, tool_result="completed"
        ),
        _step(4, "replan", {"plan_id": "plan-alpha"}, tool_result="Updated plan"),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-plan", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["plans_created"] == 1
    assert summary["plan_steps_executed"] == 2
    assert summary["plan_revisions"] == 1


def test_planning_detector_completion_rate_uses_expected_steps() -> None:
    detector = PlanningDetector()
    steps = [
        _step(1, "create_plan", {"plan_id": "plan-alpha", "steps": ["search", "write", "verify"]}),
        _step(2, "execute_plan_step", {"plan_id": "plan-alpha", "step": 1, "status": "completed"}),
        _step(3, "execute_plan_step", {"plan_id": "plan-alpha", "step": 2}, tool_result="done"),
    ]
    summary = detector.summarize(
        AgentTask(task_id="task-plan", steps=steps),
        [detector.detect_step(step) for step in steps],
    )
    assert summary["plan_completion_rate"] == 2 / 3


def test_planning_detector_goal_tracking_is_detected_without_false_creation() -> None:
    detector = PlanningDetector()
    signals = detector.detect_step(
        _step(1, "respond", {}, llm_reasoning="Goal: finish verification after search and write.")
    )
    assert signals["is_goal_tracking"] is True
    assert signals["is_plan_creation"] is False
