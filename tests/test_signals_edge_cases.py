from __future__ import annotations

from agent_xray.schema import AgentStep, AgentTask
from agent_xray.signals.coding import CodingDetector
from agent_xray.signals.commerce import CommerceDetector
from agent_xray.signals.research import ResearchDetector


def test_commerce_detector_no_false_positive_on_coding_task(coding_task: AgentTask) -> None:
    detector = CommerceDetector()
    summary = detector.summarize(
        coding_task, [detector.detect_step(step) for step in coding_task.sorted_steps]
    )
    assert summary["reached_cart"] is False
    assert summary["reached_checkout"] is False
    assert summary["reached_payment"] is False


def test_coding_detector_no_false_positive_on_commerce_task(golden_task: AgentTask) -> None:
    detector = CodingDetector()
    summary = detector.summarize(
        golden_task, [detector.detect_step(step) for step in golden_task.sorted_steps]
    )
    assert summary["file_operations"] == 0
    assert summary["unique_files_touched"] == 0


def test_coding_detector_url_not_counted_as_file_path() -> None:
    detector = CodingDetector()
    step = AgentStep(
        task_id="task-1",
        step=1,
        tool_name="browser_open",
        tool_input={"url": "https://example.test/docs", "note": "See example.test for docs"},
    )
    summary = detector.summarize(AgentTask(task_id="task-1", steps=[step]), [detector.detect_step(step)])
    assert detector.detect_step(step)["has_file_path"] is False
    assert summary["unique_files_touched"] == 0


def test_research_detector_no_false_positive_on_coding_task(coding_task: AgentTask) -> None:
    detector = ResearchDetector()
    summary = detector.summarize(
        coding_task, [detector.detect_step(step) for step in coding_task.sorted_steps]
    )
    assert summary["search_count"] == 0
    assert summary["read_count"] == 0
    assert summary["source_diversity"] == 0


def test_all_detectors_stateless(golden_task: AgentTask) -> None:
    for detector in (CommerceDetector(), CodingDetector(), ResearchDetector()):
        first = detector.summarize(
            golden_task, [detector.detect_step(step) for step in golden_task.sorted_steps]
        )
        second = detector.summarize(
            golden_task, [detector.detect_step(step) for step in golden_task.sorted_steps]
        )
        assert first == second
