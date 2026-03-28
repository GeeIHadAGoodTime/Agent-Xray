from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray import __version__
from agent_xray.cli import (
    build_parser,
    cmd_analyze,
    cmd_compare,
    cmd_diff,
    cmd_flywheel,
    cmd_grade,
    cmd_quickstart,
    cmd_surface,
    cmd_tree,
)
from agent_xray.schema import AgentTask

RULES_DIR = Path(__file__).resolve().parents[1] / "src" / "agent_xray" / "rules"


def test_cmd_analyze_returns_output(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_analyze(
        Namespace(log_dir=tmp_trace_dir, days=None, rules=None, format="auto", json=False)
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Analyzed 4 task(s)" in captured.out


def test_cmd_analyze_json_flag(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_analyze(
        Namespace(log_dir=tmp_trace_dir, days=None, rules=None, format="auto", json=True)
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["summary"]["tasks"] == 4
    assert {item["task_id"] for item in payload["tasks"]} == {
        "broken-task",
        "coding-task",
        "golden-task",
        "research-task",
    }


def test_cmd_grade_default_rules(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_grade(
        Namespace(log_dir=tmp_trace_dir, days=None, rules=None, format="auto", json=True)
    )
    payload = json.loads(capsys.readouterr().out)
    grades = {item["task_id"]: item["grade"] for item in payload["tasks"]}
    assert result == 0
    assert payload["summary"]["rules"] == "default"
    assert grades["broken-task"] == "BROKEN"
    assert grades["golden-task"] == "GOLDEN"


def test_cmd_grade_browser_flow_rules(
    tmp_trace_dir,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cmd_grade(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=RULES_DIR / "browser_flow.json",
            format="auto",
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    grades = {item["task_id"]: item["grade"] for item in payload["tasks"]}
    assert result == 0
    assert payload["summary"]["rules"] == "browser_flow"
    assert grades["golden-task"] == "GOLDEN"


def test_cmd_grade_coding_rules(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_grade(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=RULES_DIR / "coding_agent.json",
            format="auto",
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    grades = {item["task_id"]: item["grade"] for item in payload["tasks"]}
    assert result == 0
    assert payload["summary"]["rules"] == "coding_agent"
    assert grades["coding-task"] == "GOOD"


def test_cmd_surface_task(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_surface(
        Namespace(
            task_id="golden-task",
            log_dir_opt=tmp_trace_dir,
            days=None,
            format="auto",
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["task_id"] == "golden-task"
    assert payload["steps"][0]["tool_name"] == "browser_navigate"


def test_cmd_tree_output(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_tree(Namespace(log_dir_opt=tmp_trace_dir, days=None, format="auto", json=True))
    payload = json.loads(capsys.readouterr().out)
    task_ids = {task_id for task_ids in payload["20260326"].values() for task_id in task_ids}
    assert result == 0
    assert "20260326" in payload
    assert "golden-task" in task_ids


def test_cmd_flywheel_basic(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_flywheel(
        Namespace(
            log_dir=tmp_trace_dir,
            rules=None,
            fixture_dir=None,
            baseline=None,
            out=None,
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["total_tasks"] == 4
    assert payload["rules_name"] == "default"


def test_cmd_flywheel_with_baseline(
    tmp_path,
    tmp_trace_dir,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(
        json.dumps(
            {
                "grade_distribution": {"BROKEN": 2, "GOOD": 2},
                "task_grades": {
                    "broken-task": "BROKEN",
                    "coding-task": "OK",
                    "golden-task": "OK",
                    "research-task": "OK",
                },
            }
        ),
        encoding="utf-8",
    )
    result = cmd_flywheel(
        Namespace(
            log_dir=tmp_trace_dir,
            rules=None,
            fixture_dir=None,
            baseline=baseline_path,
            out=None,
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["baseline_grade_distribution"] == {"BROKEN": 2, "GOOD": 2}
    assert payload["trend"] == "improving"


def test_cmd_diff_two_dirs(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_diff(
        Namespace(
            task_id_1="golden-task",
            task_id_2="broken-task",
            log_dir_opt=tmp_trace_dir,
            days=None,
            format="auto",
            json=True,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["diverged_at_step"] == 1


def test_cmd_compare_two_dirs(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
    capsys: pytest.CaptureFixture[str],
) -> None:
    left_dir = write_trace_dir("compare-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("compare-right", [clone_task(golden_task, "checkout-task")])
    result = cmd_compare(
        Namespace(left_log_dir=left_dir, right_log_dir=right_dir, rules=None, json=True)
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["matched_tasks"] == 1
    assert payload["grade_deltas"]["GOLDEN"] == 1
    assert payload["grade_deltas"]["BROKEN"] == -1


def test_cli_help_exits_zero() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_cli_version_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--version"])
    captured = capsys.readouterr()
    assert excinfo.value.code == 0
    assert captured.out.strip() == f"agent-xray {__version__}"


def test_cli_parser_accepts_top_level_flags() -> None:
    parser = build_parser()
    args = parser.parse_args(["--verbose", "--no-color", "quickstart"])
    assert args.command == "quickstart"
    assert args.verbose is True
    assert args.no_color is True


def test_analyze_help_includes_example(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze", "--help"])
    assert (
        "Example: agent-xray analyze ./traces --rules browser_flow --json"
        in capsys.readouterr().out
    )


def test_cmd_analyze_missing_dir_shows_quickstart_hint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cmd_analyze(
        Namespace(
            log_dir=tmp_path / "missing-traces",
            days=None,
            rules=None,
            format="auto",
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 1
    assert "Directory not found:" in captured.out
    assert "Run agent-xray quickstart for a demo." in captured.out


def test_cmd_quickstart_runs_and_creates_demo(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    result = cmd_quickstart(Namespace(verbose=False, quiet=False, no_color=True))
    captured = capsys.readouterr()
    assert result == 0
    assert "QUICKSTART" in captured.out
    assert "Workflow: grade -> surface broken-task -> report health" in captured.out
    match = re.search(r"Quickstart traces: (.+)", captured.out)
    assert match is not None
    demo_dir = Path(match.group(1).strip())
    assert demo_dir.exists()
    assert any(demo_dir.glob("*.jsonl"))
