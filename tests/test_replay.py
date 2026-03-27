from __future__ import annotations

import json

from agent_xray.capture import build_fixture
from agent_xray.replay import replay_fixture
from agent_xray.schema import AgentTask


def test_replay_stable_result(tmp_path, golden_task: AgentTask) -> None:
    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    result = replay_fixture(fixture_path, [golden_task])

    assert result["verdict"] == "STABLE"
    assert result["current_task_id"] == golden_task.task_id


def test_replay_regression_detected(
    tmp_path,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    result = replay_fixture(fixture_path, [clone_task(broken_task, golden_task.task_id)])

    assert result["verdict"] == "REGRESSION"


def test_replay_improved_result(
    tmp_path,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    fixture_path = tmp_path / "broken.json"
    fixture_path.write_text(json.dumps(build_fixture(broken_task)), encoding="utf-8")

    result = replay_fixture(fixture_path, [clone_task(golden_task, broken_task.task_id)])

    assert result["verdict"] == "IMPROVED"


def test_replay_empty_fixture(tmp_path, golden_task: AgentTask) -> None:
    fixture_path = tmp_path / "empty.json"
    fixture_path.write_text("{}", encoding="utf-8")

    result = replay_fixture(fixture_path, [golden_task])

    assert result["verdict"] == "UNMATCHED"


def test_replay_unmatched_task(tmp_path, golden_task: AgentTask, research_task: AgentTask) -> None:
    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    result = replay_fixture(fixture_path, [research_task])

    assert result["verdict"] == "UNMATCHED"
