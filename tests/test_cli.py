from __future__ import annotations

import json
import re
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray import __version__
from agent_xray.enforce import ChangeRecord, TestResult
from agent_xray.cli import (
    build_parser,
    cmd_analyze,
    cmd_compare,
    cmd_enforce,
    cmd_enforce_quick,
    cmd_diff,
    cmd_flywheel,
    cmd_grade,
    cmd_quickstart,
    cmd_surface,
    cmd_tree,
    main,
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


def test_cmd_analyze_shows_task_bank_hint_when_not_provided(
    tmp_trace_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    result = cmd_analyze(
        Namespace(log_dir=tmp_trace_dir, days=None, rules=None, format="auto", json=False)
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Tip: provide a task bank with --task-bank" in captured.out


def test_cmd_analyze_no_task_bank_hint_when_provided(
    tmp_trace_dir, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bank = tmp_path / "task_bank.json"
    bank.write_text(json.dumps([{"task_id": "golden-task", "criteria": []}]), encoding="utf-8")
    result = cmd_analyze(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=None,
            format="auto",
            json=False,
            task_bank=str(bank),
            verbose=False,
            quiet=False,
            no_color=True,
            pattern=None,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "--task-bank" not in captured.out


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


def test_cmd_grade_text_includes_structural_note(
    tmp_trace_dir, capsys: pytest.CaptureFixture[str]
) -> None:
    result = cmd_grade(
        Namespace(
            log_dir=tmp_trace_dir,
            days=None,
            rules=None,
            format="auto",
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
            pattern=None,
            grade_filter=None,
            site_filter=None,
            outcome_filter=None,
            since_filter=None,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Note: Grades measure execution structure, not output correctness." in captured.out


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


def test_cmd_surface_defaults_to_text(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    """BUG #7: Surface should default to text, not JSON."""
    result = cmd_surface(
        Namespace(
            task_id="golden-task",
            log_dir_opt=tmp_trace_dir,
            days=None,
            format="auto",
            json=False,
            output_format=None,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    # Text output should contain the human-readable header
    assert "AGENT XRAY SURFACE:" in captured.out
    assert "golden-task" in captured.out
    # Should NOT be valid JSON
    try:
        json.loads(captured.out)
        assert False, "Default output should be text, not JSON"
    except json.JSONDecodeError:
        pass  # Expected: text output is not JSON


def test_cmd_surface_output_format_json(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    """BUG #7: --output-format json should produce JSON output."""
    result = cmd_surface(
        Namespace(
            task_id="golden-task",
            log_dir_opt=tmp_trace_dir,
            days=None,
            format="auto",
            json=False,
            output_format="json",
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["task_id"] == "golden-task"


def test_cmd_surface_json_flag_still_works(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    """BUG #7: Legacy --json flag should still produce JSON output."""
    result = cmd_surface(
        Namespace(
            task_id="golden-task",
            log_dir_opt=tmp_trace_dir,
            days=None,
            format="auto",
            json=True,
            output_format=None,
        )
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["task_id"] == "golden-task"


def test_cmd_tree_output(tmp_trace_dir, capsys: pytest.CaptureFixture[str]) -> None:
    result = cmd_tree(Namespace(log_dir_opt=tmp_trace_dir, days=None, format="auto", json=True))
    payload = json.loads(capsys.readouterr().out)
    # Enriched tree returns list of dicts per site; extract task_ids from them
    task_ids = {
        entry["task_id"] if isinstance(entry, dict) else entry
        for entries in payload["20260326"].values()
        for entry in entries
    }
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
    output = captured.out + captured.err
    assert result == 1
    assert "Path not found:" in output
    assert "AGENT_XRAY_LOG_DIR" in output


def test_cmd_analyze_empty_dir_reports_no_traces_found(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = cmd_analyze(
        Namespace(
            log_dir=tmp_path,
            days=None,
            rules=None,
            format="auto",
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
            pattern=None,
        )
    )
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert result == 1
    assert (
        output.strip()
        == f"No agent traces found in {tmp_path}. Expected JSONL files with agent step data."
    )


def test_cmd_analyze_malformed_trace_file_reports_friendly_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = tmp_path / "broken.jsonl"
    trace_path.write_text("{not json}\n", encoding="utf-8")
    result = cmd_analyze(
        Namespace(
            log_dir=trace_path,
            days=None,
            rules=None,
            format="auto",
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
            pattern=None,
        )
    )
    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert result == 1
    assert output.strip() == f"Malformed trace file: {trace_path}"


def test_main_catches_file_not_found_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = Path("C:/tmp/missing-traces")

    class _Parser:
        def parse_args(self) -> Namespace:
            def _handler(_: Namespace) -> int:
                raise FileNotFoundError(f"log path does not exist: {missing}")

            return Namespace(func=_handler)

    monkeypatch.setattr("agent_xray.cli.build_parser", lambda: _Parser())
    assert main() == 1
    captured = capsys.readouterr()
    assert (captured.out + captured.err).strip() == f"No traces found at {missing}"


def test_main_catches_malformed_trace_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    trace_path = Path("C:/tmp/bad-trace.jsonl")

    class _Parser:
        def parse_args(self) -> Namespace:
            def _handler(_: Namespace) -> int:
                raise json.JSONDecodeError(str(trace_path), "{bad json}", 1)

            return Namespace(func=_handler)

    monkeypatch.setattr("agent_xray.cli.build_parser", lambda: _Parser())
    assert main() == 1
    captured = capsys.readouterr()
    assert (captured.out + captured.err).strip() == f"Malformed trace file: {trace_path}"


def test_main_catches_import_error_with_json_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Parser:
        def parse_args(self) -> Namespace:
            def _handler(_: Namespace) -> int:
                raise ImportError("missing optional dependency")

            return Namespace(func=_handler, json=True)

    monkeypatch.setattr("agent_xray.cli.build_parser", lambda: _Parser())

    assert main() == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"error": {"message": "Error: missing optional dependency"}}


def test_main_catches_keyboard_interrupt_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Parser:
        def parse_args(self) -> Namespace:
            def _handler(_: Namespace) -> int:
                raise KeyboardInterrupt

            return Namespace(func=_handler)

    monkeypatch.setattr("agent_xray.cli.build_parser", lambda: _Parser())

    assert main() == 1

    captured = capsys.readouterr()
    assert (captured.out + captured.err).strip() == "Interrupted."


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


def _enforce_result(
    *,
    decision: str,
    audit_verdict: str = "VALID",
    audit_reasons: list[str] | None = None,
    net_improvement: int = 0,
) -> ChangeRecord:
    before = TestResult(
        exit_code=1,
        passed=4,
        failed=1,
        errors=0,
        skipped=0,
        total=5,
        duration_seconds=1.0,
        output="before",
    )
    after = TestResult(
        exit_code=0 if decision == "COMMITTED" else 1,
        passed=5 if decision == "COMMITTED" else 4,
        failed=0 if decision == "COMMITTED" else 1,
        errors=0,
        skipped=0,
        total=5,
        duration_seconds=1.0,
        output="after",
    )
    return ChangeRecord(
        iteration=1,
        before=before,
        after=after,
        decision=decision,
        audit_verdict=audit_verdict,
        audit_reasons=audit_reasons or [],
        net_improvement=net_improvement,
    )


def test_cmd_enforce_check_shows_rejected_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis, project_root=None: _enforce_result(
            decision="REJECTED",
            audit_reasons=[
                "No gaming signals detected",
                "Change too large: 37 files exceeds limit of 5 -- break into smaller iterations",
                "Guidance: Split this change into smaller iterations touching fewer files.",
            ],
        ),
    )
    result = cmd_enforce(
        Namespace(
            enforce_command="check",
            json=False,
            project_root=".",
            hypothesis="",
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Iteration 1: REJECTED (change too large) -" in captured.out
    assert "37 files exceeds limit of 5" in captured.out
    assert "Split this change into smaller iterations touching fewer files." in captured.out


def test_cmd_enforce_check_shows_gaming_reverted_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis, project_root=None: _enforce_result(
            decision="REVERTED",
            audit_verdict="GAMING",
            audit_reasons=["Gaming detected -- auto-reverting"],
        ),
    )
    result = cmd_enforce(
        Namespace(
            enforce_command="check",
            json=False,
            project_root=".",
            hypothesis="",
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Iteration 1: GAMING -> REVERTED - Gaming detected" in captured.out


def test_cmd_enforce_check_shows_regression_reverted_reason(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis, project_root=None: _enforce_result(
            decision="REVERTED",
            audit_verdict="VALID",
            audit_reasons=["Regressions detected: tests/test_api.py::test_alpha"],
        ),
    )
    result = cmd_enforce(
        Namespace(
            enforce_command="check",
            json=False,
            project_root=".",
            hypothesis="",
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Iteration 1: REGRESSION -> REVERTED - Regressions detected" in captured.out


def test_cmd_enforce_check_shows_committed_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_check",
        lambda hypothesis, project_root=None: _enforce_result(
            decision="COMMITTED",
            audit_verdict="VALID",
            audit_reasons=["No gaming signals detected"],
            net_improvement=3,
        ),
    )
    result = cmd_enforce(
        Namespace(
            enforce_command="check",
            json=False,
            project_root=".",
            hypothesis="",
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Iteration 1: COMMITTED - VALID (+3 tests)" in captured.out


def test_cmd_enforce_quick_shows_session_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_quick",
        lambda **kwargs: {
            **_enforce_result(
                decision="RECOMMEND_COMMIT",
                audit_verdict="VALID",
                audit_reasons=["No gaming signals detected"],
                net_improvement=2,
            ).to_dict(),
            "session_initialized": True,
            "session_reused": False,
            "session_dir": ".agent-xray-enforce",
            "baseline": _enforce_result(decision="COMMITTED").before.to_dict(),
        },
    )
    result = cmd_enforce_quick(
        Namespace(
            test_command="pytest tests/",
            log_dir="logs",
            project_root=".",
            hypothesis="",
            max_files=10,
            max_diff_lines=500,
            json=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )
    captured = capsys.readouterr()
    assert result == 0
    assert "Session: initialized (.agent-xray-enforce)" in captured.out
    assert "Iteration 1: RECOMMEND_COMMIT - VALID (+2 tests)" in captured.out


def test_parser_accepts_enforce_quick_top_level_command() -> None:
    parser = build_parser()
    args = parser.parse_args(["enforce-quick", "pytest tests/", "--log-dir", "logs/"])
    assert args.test_command == "pytest tests/"
    assert args.log_dir == "logs/"
    assert args.func == cmd_enforce_quick


def test_parser_accepts_enforce_nested_quick_command() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "enforce",
            "quick",
            "--hypothesis",
            "fix parser",
            "--test-command",
            "pytest tests/",
        ]
    )
    assert args.enforce_command == "quick"
    assert args.hypothesis == "fix parser"
    assert args.test_command == "pytest tests/"
    assert args.func == cmd_enforce


def test_top_level_help_has_getting_started(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    help_text = capsys.readouterr().out
    assert "Getting Started" in help_text
    assert "triage -> surface_task/surface -> grade -> root_cause/root-cause -> inspect_task/inspect" in help_text


def test_enforce_auto_help_mentions_template_variables(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["enforce", "auto", "--help"])
    help_text = capsys.readouterr().out
    assert "{failing_tests}" in help_text
    assert "{last_error}" in help_text


def test_cmd_enforce_plan_accepts_space_separated_expected_tests(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_args: dict[str, object] = {}

    def mock_enforce_plan(hypothesis: str, expected_tests: list[str], project_root: str = ".") -> dict[str, object]:
        captured_args["hypothesis"] = hypothesis
        captured_args["expected_tests"] = expected_tests
        captured_args["project_root"] = project_root
        return {
            "hypothesis": hypothesis,
            "expected_tests": expected_tests,
            "status": "plan_registered",
        }

    monkeypatch.setattr("agent_xray.enforce.enforce_plan", mock_enforce_plan)

    result = cmd_enforce(
        Namespace(
            enforce_command="plan",
            json=False,
            project_root=".",
            hypothesis="fix bug",
            expected_tests=["test_foo", "test_bar,baz"],
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured_args["expected_tests"] == ["test_foo", "test_bar", "baz"]
    assert "Expected tests: test_foo, test_bar, baz" in captured.out


def test_enforce_plan_parser_accepts_repeated_expected_tests() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["enforce", "plan", "--hypothesis", "fix bug", "--expected-tests", "test_foo", "test_bar"]
    )
    assert args.expected_tests == ["test_foo", "test_bar"]


def test_cmd_enforce_diff_outputs_preview(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "agent_xray.enforce.enforce_diff",
        lambda project_root=".", full=False: {
            "files": ["src/foo.py"],
            "file_count": 1,
            "diff_lines": ["+new line"],
            "diff_line_count": 1,
            "would_reject": False,
            "reject_reason": "",
            "truncated": False,
        },
    )

    result = cmd_enforce(
        Namespace(
            enforce_command="diff",
            json=False,
            project_root=".",
            full=False,
            verbose=False,
            quiet=False,
            no_color=True,
        )
    )

    captured = capsys.readouterr()
    assert result == 0
    assert "Files: 1" in captured.out
    assert "Would reject: no" in captured.out
    assert "+new line" in captured.out
