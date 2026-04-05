from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_xray.grader import GradeResult
from agent_xray.watch import watch_file
import agent_xray.watch as watch_mod


def _step_row(task_id: str, step: int = 1, *, user_text: str = "Do the task") -> dict[str, object]:
    return {
        "task_id": task_id,
        "step": step,
        "tool_name": "browser_click",
        "tool_input": {"ref": "go"},
        "tool_result": "Clicked.",
        "timestamp": "2026-04-05T12:00:00Z",
        "user_text": user_text,
        "task_category": "commerce",
    }


def _outcome_row(task_id: str, *, status: str = "success") -> dict[str, object]:
    return {
        "event": "task_complete",
        "task_id": task_id,
        "status": status,
        "total_steps": 1,
        "timestamp": "2026-04-05T12:00:05Z",
    }


def _append_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _grade(task_id: str, grade: str = "GOOD", score: int = 2) -> GradeResult:
    return GradeResult(
        task_id=task_id,
        grade=grade,
        score=score,
        reasons=[],
        metrics={},
        signals=[],
    )


def test_watch_file_reports_missing_file(capsys: pytest.CaptureFixture[str]) -> None:
    watch_file("J:/PROJECTS/agent-xray/does-not-exist.jsonl", color=False)

    assert "File not found" in capsys.readouterr().err


def test_watch_file_prints_initialization_and_final_tally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="stub-rules"))

    def fake_sleep(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert f"Watching {trace_path} (rules: stub-rules, poll: 0.0s)" in out
    assert "Press Ctrl+C to stop." in out
    assert "Final tally:" in out
    assert "[Total: 0 | GOLDEN: 0 | GOOD: 0 | OK: 0 | WEAK: 0 | BROKEN: 0]" in out


def test_watch_file_detects_appended_lines_and_grades_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="stub-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        calls.append(task.task_id)
        assert rules.name == "stub-rules"
        assert len(task.steps) == 1
        assert task.outcome is not None
        return _grade(task.task_id, "GOOD", 2)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)

    state = {"count": 0}

    def fake_sleep(_: float) -> None:
        state["count"] += 1
        if state["count"] == 1:
            _append_rows(trace_path, [_step_row("task-1", user_text="Watch append")])
            return
        if state["count"] == 2:
            _append_rows(trace_path, [_outcome_row("task-1")])
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert calls == ["task-1"]
    assert "GOOD" in out
    assert '"Watch append"' in out
    assert "[Total: 1 | GOLDEN: 0 | GOOD: 1 | OK: 0 | WEAK: 0 | BROKEN: 0]" in out


def test_watch_file_emits_json_for_completed_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    _append_rows(trace_path, [_step_row("task-json"), _outcome_row("task-json")])

    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="json-rules"))
    monkeypatch.setattr(watch_mod, "grade_task", lambda task, rules: _grade(task.task_id, "GOLDEN", 4))

    def fake_sleep(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_file(trace_path, poll_interval=0.0, json_output=True, color=False)
    out = capsys.readouterr().out.strip()
    payload = json.loads(out)

    assert payload == {
        "task_id": "task-json",
        "grade": "GOLDEN",
        "score": 4,
        "step_count": 1,
        "outcome": "success",
        "task_text": "Do the task",
        "timestamp": "12:00:05",
    }


def test_watch_file_skips_outcome_only_tasks_without_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    _append_rows(trace_path, [_outcome_row("outcome-only")])

    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="stub-rules"))

    def fail_grade_task(task, rules):  # type: ignore[no-untyped-def]
        raise AssertionError("grade_task should not be called for tasks without steps")

    monkeypatch.setattr(watch_mod, "grade_task", fail_grade_task)

    def fake_sleep(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert "outcome-only" not in out
    assert "[Total: 0 | GOLDEN: 0 | GOOD: 0 | OK: 0 | WEAK: 0 | BROKEN: 0]" in out


def test_watch_file_does_not_regrade_same_task_on_duplicate_outcomes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    _append_rows(trace_path, [_step_row("dup-task"), _outcome_row("dup-task")])
    calls: list[str] = []

    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="stub-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        calls.append(task.task_id)
        return _grade(task.task_id, "OK", 0)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)

    state = {"count": 0}

    def fake_sleep(_: float) -> None:
        state["count"] += 1
        if state["count"] == 1:
            _append_rows(trace_path, [_outcome_row("dup-task")])
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert calls == ["dup-task"]
    assert "[Total: 1 | GOLDEN: 0 | GOOD: 0 | OK: 1 | WEAK: 0 | BROKEN: 0]" in out
