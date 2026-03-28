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


def test_replay_evaluation_drift(tmp_path, golden_task: AgentTask) -> None:
    """Replay should return EVALUATION_DRIFT when integrity hashes show change."""
    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    # Provide a hash that won't match the actual module source
    bad_hashes = {"agent_xray.grader": "deliberately_wrong_hash"}

    result = replay_fixture(fixture_path, [golden_task], integrity_hashes=bad_hashes)

    assert result["verdict"] == "EVALUATION_DRIFT"
    assert "agent_xray.grader" in result["detail"]


def test_replay_no_drift_with_valid_hashes(tmp_path, golden_task: AgentTask) -> None:
    """Replay should return normal verdict when integrity hashes match."""
    from agent_xray.flywheel import _sha256_of_source
    from agent_xray import grader as grader_module

    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    valid_hashes = {"agent_xray.grader": _sha256_of_source(grader_module)}

    result = replay_fixture(fixture_path, [golden_task], integrity_hashes=valid_hashes)

    assert result["verdict"] in {"STABLE", "IMPROVED", "REGRESSION"}
    assert result["verdict"] != "EVALUATION_DRIFT"


def test_replay_no_drift_without_hashes(tmp_path, golden_task: AgentTask) -> None:
    """Replay without integrity_hashes should behave normally (no drift check)."""
    fixture_path = tmp_path / "golden.json"
    fixture_path.write_text(json.dumps(build_fixture(golden_task)), encoding="utf-8")

    result = replay_fixture(fixture_path, [golden_task])

    assert result["verdict"] in {"STABLE", "IMPROVED", "REGRESSION", "UNMATCHED"}
    assert result["verdict"] != "EVALUATION_DRIFT"
