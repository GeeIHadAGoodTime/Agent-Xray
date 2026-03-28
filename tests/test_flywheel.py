from __future__ import annotations

import json

import pytest

from agent_xray.flywheel import IntegrityLock, check_integrity, run_flywheel


def test_flywheel_baseline_comparison(tmp_path) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    trace = log_dir / "trace.jsonl"
    trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "task_id": "task-1",
                        "step": 1,
                        "tool_name": "web_search",
                        "tool_input": {"q": "laptop"},
                        "tool_result": "results",
                    }
                ),
                json.dumps(
                    {
                        "task_id": "task-1",
                        "step": 2,
                        "tool_name": "browser_navigate",
                        "tool_input": {"url": "https://shop.example.test"},
                        "tool_result": "checkout",
                        "page_url": "https://shop.example.test/checkout",
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "grade_distribution": {"GOOD": 1},
                "task_grades": {"task-1": "GOOD"},
            }
        ),
        encoding="utf-8",
    )

    result = run_flywheel(log_dir, baseline_path=baseline_path)

    assert result.total_tasks == 1
    assert result.baseline_grade_distribution == {"GOOD": 1}
    assert result.grade_deltas is not None
    assert result.trend in {"stable", "improving", "degrading"}


def test_integrity_lock_no_drift(tmp_path) -> None:
    """IntegrityLock with matching hashes should pass check_integrity."""
    test_file = tmp_path / "test_rules.json"
    test_file.write_text('{"name": "test"}', encoding="utf-8")

    import hashlib

    expected = hashlib.sha256(test_file.read_bytes()).hexdigest()
    locks = [IntegrityLock(file_path=str(test_file), expected_hash=expected, actual_hash=expected)]

    violated = check_integrity(locks)
    assert violated == []


def test_integrity_lock_detects_drift(tmp_path) -> None:
    """IntegrityLock should detect when a file is modified."""
    test_file = tmp_path / "test_rules.json"
    test_file.write_text('{"name": "test"}', encoding="utf-8")

    locks = [IntegrityLock(file_path=str(test_file), expected_hash="fake_hash_abc", actual_hash="fake_hash_abc")]

    violated = check_integrity(locks)
    assert len(violated) == 1
    assert violated[0].file_path == str(test_file)
    assert violated[0].actual_hash != "fake_hash_abc"


def test_integrity_lock_detects_missing_file(tmp_path) -> None:
    """IntegrityLock should flag a file that was deleted."""
    locks = [IntegrityLock(
        file_path=str(tmp_path / "nonexistent.json"),
        expected_hash="some_hash",
        actual_hash="some_hash",
    )]

    violated = check_integrity(locks)
    assert len(violated) == 1
    assert violated[0].actual_hash == ""


def test_integrity_lock_module_hashing() -> None:
    """IntegrityLock should work for module source hashing."""
    from agent_xray.flywheel import _sha256_of_source
    from agent_xray import grader as grader_module

    h = _sha256_of_source(grader_module)
    locks = [IntegrityLock(file_path="agent_xray.grader", expected_hash=h, actual_hash=h)]

    violated = check_integrity(locks)
    assert violated == []


def test_flywheel_raises_on_drift(tmp_path) -> None:
    """Flywheel should raise RuntimeError if a tracked file changes mid-run."""
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    trace = log_dir / "trace.jsonl"
    trace.write_text(
        json.dumps(
            {
                "task_id": "task-1",
                "step": 1,
                "tool_name": "respond",
                "tool_input": {},
                "tool_result": "ok",
            }
        ),
        encoding="utf-8",
    )
    # Create a task bank file that we'll track
    bank = tmp_path / "tasks.json"
    bank.write_text('{"tasks": []}', encoding="utf-8")

    # Monkey-patch check_integrity to simulate drift
    import agent_xray.flywheel as fw_module

    original_check = fw_module.check_integrity

    def _fake_drift(locks):
        # Simulate drift by returning all locks as violated
        for lock in locks:
            lock.actual_hash = "CHANGED"
        return locks

    fw_module.check_integrity = _fake_drift
    try:
        with pytest.raises(RuntimeError, match="EVALUATION_DRIFT"):
            run_flywheel(log_dir, task_bank_paths=[bank])
    finally:
        fw_module.check_integrity = original_check
