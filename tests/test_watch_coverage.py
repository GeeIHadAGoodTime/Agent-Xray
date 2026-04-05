from __future__ import annotations

import builtins
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_xray.watch as watch_mod
from agent_xray.grader import GradeResult


def _step_row(task_id: str, *, step: int = 1, user_text: str = "Mock watched task") -> str:
    return json.dumps(
        {
            "task_id": task_id,
            "step": step,
            "tool_name": "browser_click",
            "tool_input": {"ref": "go"},
            "tool_result": "Clicked.",
            "timestamp": "2026-04-05T12:00:00Z",
            "user_text": user_text,
        }
    ) + "\n"


def _outcome_row(task_id: str, *, status: str = "success", use_event: bool = True) -> str:
    payload: dict[str, object] = {
        "task_id": task_id,
        "status": status,
        "total_steps": 1,
        "timestamp": "2026-04-05T12:00:05Z",
    }
    if use_event:
        payload["event"] = "task_complete"
    else:
        payload["outcome"] = status
        payload["tool_name"] = ""
    return json.dumps(payload) + "\n"


def _grade(task_id: str, grade: str, score: int) -> GradeResult:
    return GradeResult(
        task_id=task_id,
        grade=grade,
        score=score,
        reasons=[],
        metrics={},
        signals=[],
    )


class _FakeTailFile:
    def __init__(self, state: dict[str, object], behavior: BaseException | None = None) -> None:
        self._state = state
        self._behavior = behavior
        self._pos = 0

    def __enter__(self) -> _FakeTailFile:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # type: ignore[no-untyped-def]
        return False

    def seek(self, pos: int) -> None:
        self._pos = pos

    def readlines(self) -> list[str]:
        if self._behavior is not None:
            behavior = self._behavior
            self._behavior = None
            raise behavior
        content = str(self._state["content"])
        chunk = content[self._pos :]
        self._pos = len(content)
        return chunk.splitlines(keepends=True)

    def tell(self) -> int:
        return self._pos


def _fake_open_factory(state: dict[str, object]):
    def fake_open(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        behaviors = state.setdefault("behaviors", [])
        behavior = behaviors.pop(0) if behaviors else None
        state["opened"] = state.get("opened", 0) + 1
        return _FakeTailFile(state, behavior=behavior)

    return fake_open


def test_watch_file_initializes_with_mocked_open_and_sleep(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {"content": ""}

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="mock-rules"))
    monkeypatch.setattr(watch_mod.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert "Watching" in out
    assert "mock-rules" in out
    assert state["opened"] == 1


def test_watch_file_retries_after_read_error_and_then_grades_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {
        "content": "",
        "behaviors": [OSError("boom")],
    }
    graded: list[str] = []

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="mock-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        graded.append(task.task_id)
        return _grade(task.task_id, "GOOD", 2)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)

    sleep_calls = {"count": 0}

    def fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            state["content"] = _step_row("retry-task") + _outcome_row("retry-task")
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    captured = capsys.readouterr()

    assert "Read error: boom" in captured.err
    assert graded == ["retry-task"]
    assert "GOOD" in captured.out


def test_watch_file_processes_new_data_on_later_poll_with_mocked_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {"content": _step_row("tail-task")}
    graded: list[str] = []

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="tail-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        graded.append(task.task_id)
        return _grade(task.task_id, "OK", 0)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)

    sleep_calls = {"count": 0}

    def fake_sleep(_: float) -> None:
        sleep_calls["count"] += 1
        if sleep_calls["count"] == 1:
            state["content"] = str(state["content"]) + _outcome_row("tail-task")
            return
        raise KeyboardInterrupt

    monkeypatch.setattr(watch_mod.time, "sleep", fake_sleep)

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert graded == ["tail-task"]
    assert '"Mock watched task"' in out
    assert "[Total: 1 | GOLDEN: 0 | GOOD: 0 | OK: 1 | WEAK: 0 | BROKEN: 0]" in out


def test_watch_file_ignores_invalid_lines_before_valid_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {
        "content": (
            "not-json\n"
            + json.dumps(["not-a-dict"])
            + "\n"
            + json.dumps({"step": 1, "tool_name": "browser_click", "tool_input": {}})
            + "\n"
            + _step_row("valid-task")
            + _outcome_row("valid-task")
        )
    }
    graded: list[str] = []

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="mock-rules"))
    monkeypatch.setattr(
        watch_mod,
        "grade_task",
        lambda task, rules: graded.append(task.task_id) or _grade(task.task_id, "GOOD", 1),
    )
    monkeypatch.setattr(watch_mod.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert graded == ["valid-task"]
    assert "valid-task" in out


def test_watch_file_recognizes_outcome_rows_without_event_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {"content": _step_row("alt-outcome") + _outcome_row("alt-outcome", use_event=False)}
    graded: list[str] = []

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="mock-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        graded.append(task.outcome.status if task.outcome else "missing")
        return _grade(task.task_id, "GOLDEN", 4)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)
    monkeypatch.setattr(watch_mod.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert graded == ["success"]
    assert "GOLDEN" in out


def test_watch_file_updates_tally_for_multiple_completed_tasks_in_one_poll(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "watch.jsonl"
    trace_path.write_text("", encoding="utf-8")
    state: dict[str, object] = {
        "content": (
            _step_row("task-a", user_text="Alpha task")
            + _outcome_row("task-a")
            + _step_row("task-b", user_text="Beta task")
            + _outcome_row("task-b")
        )
    }

    monkeypatch.setattr(builtins, "open", _fake_open_factory(state))
    monkeypatch.setattr(watch_mod, "load_rules", lambda path: SimpleNamespace(name="mock-rules"))

    def fake_grade_task(task, rules):  # type: ignore[no-untyped-def]
        if task.task_id == "task-a":
            return _grade("task-a", "GOOD", 2)
        return _grade("task-b", "BROKEN", -3)

    monkeypatch.setattr(watch_mod, "grade_task", fake_grade_task)
    monkeypatch.setattr(watch_mod.time, "sleep", lambda _: (_ for _ in ()).throw(KeyboardInterrupt()))

    watch_mod.watch_file(trace_path, poll_interval=0.0, color=False)
    out = capsys.readouterr().out

    assert "Alpha task" in out
    assert "Beta task" in out
    assert "[Total: 2 | GOLDEN: 0 | GOOD: 1 | OK: 0 | WEAK: 0 | BROKEN: 1]" in out
