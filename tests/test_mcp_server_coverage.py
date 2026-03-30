from __future__ import annotations

import importlib
import json
import types
from pathlib import Path

import pytest


def _module():
    return importlib.import_module("agent_xray.mcp_server")


def _task(task_id: str, task_text: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(task_id=task_id, task_text=task_text, steps=[])


def test_triage_includes_grade_note_suggested_next_and_distribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = [
        _task("broken-1", "Broken checkout flow"),
        _task("golden-1", "Successful checkout flow"),
    ]
    grades = [
        types.SimpleNamespace(task_id="broken-1", grade="BROKEN", score=-4),
        types.SimpleNamespace(task_id="golden-1", grade="GOLDEN", score=10),
    ]
    fix_plan = [
        types.SimpleNamespace(
            root_cause="spin",
            priority="P0",
            targets=["prompt/tool-choice"],
            fix_hint="Reduce repeated snapshots",
            investigate_task="broken-1",
        )
    ]
    module = _module()

    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: tasks)
    monkeypatch.setattr("agent_xray.grader.load_rules", lambda: types.SimpleNamespace(name="default"))
    monkeypatch.setattr("agent_xray.grader.grade_tasks", lambda tasks_arg, rules_arg: grades)
    monkeypatch.setattr(
        "agent_xray.root_cause.classify_failures",
        lambda tasks_arg, grades_arg: [types.SimpleNamespace(task_id="broken-1")],
    )
    monkeypatch.setattr("agent_xray.diagnose.build_fix_plan", lambda results: fix_plan)
    monkeypatch.setattr(
        "agent_xray.surface.surface_for_task",
        lambda task: {
            "steps": [
                {
                    "step": 1,
                    "tool_name": "browser_snapshot",
                    "error": "timeout waiting for checkout",
                    "tool_result": "spinner still visible",
                }
            ]
        },
    )

    payload = json.loads(module.triage("logs"))

    assert "do NOT verify output correctness" in payload["grade_note"]
    assert payload["summary"]["grade_distribution"] == {"BROKEN": 1, "GOLDEN": 1}
    assert payload["suggested_next"] == payload["suggested_next_tools"][0]
    assert payload["worst_task"]["task_id"] == "broken-1"
    assert payload["fix_plan"][0]["root_cause"] == "spin"


def test_grade_honors_dedupe_parameter_and_emits_structural_qualifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}
    task = _task("task-1", "Investigate issue")
    grade_result = types.SimpleNamespace(task_id="task-1", grade="OK", score=2, reasons=["clean"])

    def fake_load_tasks(log_dir: str, format_name: str, **kwargs: object) -> list[object]:
        calls["kwargs"] = kwargs
        return [task]

    module = _module()
    monkeypatch.setattr(module, "_load_tasks", fake_load_tasks)
    monkeypatch.setattr(
        "agent_xray.grader.load_rules",
        lambda rules="default": types.SimpleNamespace(name=str(rules)),
    )
    monkeypatch.setattr("agent_xray.grader.grade_tasks", lambda tasks_arg, rules_arg: [grade_result])

    payload = json.loads(module.grade("logs", dedupe=False))

    assert calls["kwargs"] == {"days": None, "site": None, "outcome": None, "dedupe": False}
    assert payload["structural_qualifier"] == "summary_only"
    assert "execution structure" in payload["grade_note"]
    assert payload["summary"]["distribution"] == {"OK": 1}


def test_golden_capture_includes_correctness_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [_task("task-1")])
    monkeypatch.setattr("agent_xray.grader.load_rules", lambda: types.SimpleNamespace(name="default"))
    monkeypatch.setattr(
        "agent_xray.analyzer.analyze_task",
        lambda task: types.SimpleNamespace(site_name="shop.example.test"),
    )
    monkeypatch.setattr("agent_xray.golden.capture_exemplar", lambda *args, **kwargs: "golden.json")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, encoding="utf-8": json.dumps({"task_id": "task-1", "site": "shop.example.test"}),
    )

    payload = json.loads(module.golden_capture("logs", "task-1"))

    assert payload["exemplar"]["task_id"] == "task-1"
    assert "Output correctness has NOT been verified" in payload["correctness_warning"]


def test_surface_task_returns_step_by_step_data(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [_task("task-1")])
    monkeypatch.setattr(
        "agent_xray.surface.surface_for_task",
        lambda task: {
            "metadata": {"task_id": task.task_id},
            "steps": [
                {
                    "step": 1,
                    "step_number": 1,
                    "tool_name": "browser_click",
                    "tool_input": {"ref": "checkout"},
                    "tool_result": "opened checkout",
                },
                {
                    "step": 2,
                    "step_number": 2,
                    "tool_name": "respond",
                    "tool_input": {},
                    "tool_result": "done",
                },
            ],
        },
    )

    payload = json.loads(module.surface_task("logs", "task-1"))

    assert [step["tool_name"] for step in payload["steps"]] == ["browser_click", "respond"]
    assert payload["steps"][0]["step"] == 1


def test_triage_returns_error_for_empty_log_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [])

    payload = json.loads(module.triage("empty"))

    assert payload["error"] == "No tasks found"


def test_surface_task_returns_error_for_invalid_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [_task("task-1")])

    payload = json.loads(module.surface_task("logs", "missing-task"))

    assert "not found" in payload["error"]


def test_golden_capture_returns_error_for_invalid_task_id(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(module, "_load_tasks", lambda *args, **kwargs: [_task("task-1")])

    payload = json.loads(module.golden_capture("logs", "missing-task"))

    assert "not found" in payload["error"]


def test_grade_wraps_malformed_parameter_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _module()
    monkeypatch.setattr(
        module,
        "_load_tasks",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("malformed parameters")),
    )

    payload = json.loads(module.grade("logs"))

    assert payload["error"] == "malformed parameters"


@pytest.mark.skipif(not hasattr(_module(), "enforce_quick"), reason="enforce_quick not present")
def test_enforce_quick_combines_init_and_check(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, object, object]] = []
    module = _module()

    monkeypatch.setattr(
        "agent_xray.enforce.enforce_quick",
        lambda **kwargs: calls.append(
            ("enforce_quick", kwargs.get("hypothesis"), kwargs.get("test_command"))
        ) or {"status": "ok"},
    )

    payload = json.loads(module.enforce_quick(test_command="python -m pytest"))

    assert calls == [("enforce_quick", "", "python -m pytest")]
    assert payload["status"] == "ok"
