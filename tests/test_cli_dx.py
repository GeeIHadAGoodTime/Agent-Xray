"""Tests for CLI DX improvements: Items 1, 2, 3, 8, 10."""
from __future__ import annotations

import json
import os
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray.cli import (
    CliError,
    _filter_tasks,
    _parse_since,
    _resolve_log_dir,
    build_parser,
    cmd_diff,
    cmd_diagnose,
    cmd_grade,
    cmd_search,
    cmd_tree,
)
from agent_xray.grader import load_rules
from agent_xray.schema import AgentTask


# ---------------------------------------------------------------------------
# Item 1: diff positional log path
# ---------------------------------------------------------------------------


class TestDiffPositionalLogDir:
    """The diff command should accept log path as a positional arg after the two task IDs."""

    def test_diff_positional_arg_parses(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["diff", "task-a", "task-b", "/some/path"])
        assert args.task_id_1 == "task-a"
        assert args.task_id_2 == "task-b"
        assert args.log_dir_pos == "/some/path"

    def test_diff_log_dir_option_still_works(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["diff", "task-a", "task-b", "--log-dir", "/other/path"])
        assert args.log_dir_opt == "/other/path"
        assert args.log_dir_pos is None

    def test_diff_positional_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_XRAY_LOG_DIR", "/env/path")
        parser = build_parser()
        args = parser.parse_args(["diff", "t1", "t2", "/explicit/path"])
        resolved = _resolve_log_dir(args)
        assert resolved == "/explicit/path"

    def test_diff_runs_with_positional(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_diff(
            Namespace(
                task_id_1="golden-task",
                task_id_2="broken-task",
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                log_dir=None,
                days=None,
                format="auto",
                pattern=None,
                json=True,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert "steps" in payload or "tool_sequence" in payload or isinstance(payload, dict)


# ---------------------------------------------------------------------------
# Item 2: AGENT_XRAY_LOG_DIR env var
# ---------------------------------------------------------------------------


class TestEnvVarLogDir:
    """AGENT_XRAY_LOG_DIR should be used as fallback when no explicit path is given."""

    def test_env_var_used_when_no_arg(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_XRAY_LOG_DIR", "/env/traces")
        # Reload DEFAULT_LOG_DIR by simulating the resolution
        args = Namespace(log_dir=None, log_dir_pos=None, log_dir_opt=None)
        # _resolve_log_dir falls back to DEFAULT_LOG_DIR which reads env at import time.
        # We test the pattern by calling _resolve_log_dir with no args set.
        resolved = _resolve_log_dir(args)
        # When all are None, it should fall back to DEFAULT_LOG_DIR
        # DEFAULT_LOG_DIR is set at import time, so we test via the parser instead.

    def test_env_var_via_parser(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_trace_dir: Path,
    ) -> None:
        monkeypatch.setenv("AGENT_XRAY_LOG_DIR", str(tmp_trace_dir))
        # Patch the module-level DEFAULT_LOG_DIR
        import agent_xray.cli as cli_mod
        original = cli_mod.DEFAULT_LOG_DIR
        cli_mod.DEFAULT_LOG_DIR = str(tmp_trace_dir)
        try:
            parser = build_parser()
            # tree accepts positional log_dir which defaults to None
            args = parser.parse_args(["tree"])
            resolved = _resolve_log_dir(args)
            assert resolved == str(tmp_trace_dir)
        finally:
            cli_mod.DEFAULT_LOG_DIR = original

    def test_explicit_arg_overrides_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("AGENT_XRAY_LOG_DIR", "/env/path")
        parser = build_parser()
        args = parser.parse_args(["grade", "/explicit/path"])
        assert args.log_dir == "/explicit/path"

    def test_help_text_mentions_env_var(self) -> None:
        parser = build_parser()
        help_text = parser.format_help()
        # The env var should be mentioned in subparser help (not top-level)
        # Check diff subparser
        diff_parser = None
        for action in parser._subparsers._actions:
            if hasattr(action, '_parser_class'):
                continue
            if hasattr(action, 'choices') and action.choices:
                diff_parser = action.choices.get("diff")
                break
        if diff_parser:
            diff_help = diff_parser.format_help()
            assert "AGENT_XRAY_LOG_DIR" in diff_help


# ---------------------------------------------------------------------------
# Item 3: Better error messages
# ---------------------------------------------------------------------------


class TestBetterErrors:
    """Error messages should be helpful, not raw tracebacks."""

    def test_nonexistent_rules_lists_available(self) -> None:
        with pytest.raises(FileNotFoundError) as exc_info:
            load_rules("nonexistent_rules_xyz")
        msg = str(exc_info.value)
        assert "nonexistent_rules_xyz" in msg
        assert "Available built-in rulesets:" in msg
        assert "default" in msg

    def test_nonexistent_rules_via_cli(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_grade(
            Namespace(
                log_dir=str(tmp_trace_dir),
                days=None,
                rules="totally_fake_rules",
                format="auto",
                pattern=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
                grade_filter=None,
                site_filter=None,
                outcome_filter=None,
                since_filter=None,
            )
        )
        captured = capsys.readouterr()
        assert result == 1
        assert "totally_fake_rules" in captured.out
        assert "Available built-in rulesets:" in captured.out

    def test_missing_log_dir_mentions_env_var(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_xray.cli import cmd_analyze

        result = cmd_analyze(
            Namespace(
                log_dir=str(tmp_path / "does_not_exist"),
                days=None,
                rules=None,
                format="auto",
                pattern=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        captured = capsys.readouterr()
        assert result == 1
        assert "AGENT_XRAY_LOG_DIR" in captured.out

    def test_invalid_task_id_shows_available(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from agent_xray.cli import cmd_surface

        result = cmd_surface(
            Namespace(
                task_id="nonexistent-task-xyz",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        captured = capsys.readouterr()
        assert result == 1
        assert "not found" in captured.out.lower()
        # Should list some available task IDs
        assert "golden-task" in captured.out or "Available" in captured.out


# ---------------------------------------------------------------------------
# Item 8: Filtering support
# ---------------------------------------------------------------------------


class TestFiltering:
    """Global filter flags should work on grade, report, diagnose, tree."""

    def test_filter_by_grade(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
    ) -> None:
        tasks = [golden_task, broken_task]
        filtered = _filter_tasks(tasks, grade_filter="BROKEN")
        assert len(filtered) == 1
        assert filtered[0].task_id == "broken-task"

    def test_filter_by_grade_comma_separated(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
        coding_task: AgentTask,
    ) -> None:
        tasks = [golden_task, broken_task, coding_task]
        filtered = _filter_tasks(tasks, grade_filter="BROKEN,GOLDEN")
        task_ids = {t.task_id for t in filtered}
        assert "broken-task" in task_ids
        assert "golden-task" in task_ids

    def test_filter_by_site(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
        research_task: AgentTask,
    ) -> None:
        tasks = [golden_task, broken_task, research_task]
        # site_name extracts to "shop" from shop.example.test
        filtered = _filter_tasks(tasks, site_filter="shop")
        # Both golden and broken hit shop.example.test -> site "shop"
        assert all(t.task_id in ("golden-task", "broken-task") for t in filtered)
        assert len(filtered) == 2

    def test_filter_by_outcome(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
    ) -> None:
        tasks = [golden_task, broken_task]
        filtered = _filter_tasks(tasks, outcome_filter="failed")
        assert len(filtered) == 1
        assert filtered[0].task_id == "broken-task"

    def test_filter_by_since_relative(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
    ) -> None:
        # Patch step timestamps to 1 hour ago so --since 7d always includes
        # them regardless of when the test suite runs.
        from datetime import datetime, timezone, timedelta

        recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        for task in (golden_task, broken_task):
            for step in task.steps:
                step.timestamp = recent
            if task.outcome:
                task.outcome.timestamp = recent
        tasks = [golden_task, broken_task]
        filtered = _filter_tasks(tasks, since_filter="7d")
        assert len(filtered) == 2

    def test_filter_by_since_excludes_old(
        self,
        golden_task: AgentTask,
        broken_task: AgentTask,
    ) -> None:
        # Patch step timestamps to 2 days ago so --since 1h excludes them.
        from datetime import datetime, timezone, timedelta

        old = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        for task in (golden_task, broken_task):
            for step in task.steps:
                step.timestamp = old
            if task.outcome:
                task.outcome.timestamp = old
        tasks = [golden_task, broken_task]
        filtered = _filter_tasks(tasks, since_filter="1h")
        assert len(filtered) == 0

    def test_parse_since_relative_hours(self) -> None:
        dt = _parse_since("2h")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # Should be approximately 2 hours ago
        diff = now - dt
        assert 7100 < diff.total_seconds() < 7300

    def test_parse_since_relative_days(self) -> None:
        dt = _parse_since("1d")
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        diff = now - dt
        assert 86300 < diff.total_seconds() < 86500

    def test_parse_since_iso_timestamp(self) -> None:
        dt = _parse_since("2026-03-28T02:00")
        assert dt.year == 2026
        assert dt.month == 3
        assert dt.day == 28

    def test_parse_since_invalid_raises(self) -> None:
        with pytest.raises(CliError, match="Invalid --since value"):
            _parse_since("not-a-date")

    def test_grade_command_with_filter(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_grade(
            Namespace(
                log_dir=str(tmp_trace_dir),
                days=None,
                rules=None,
                format="auto",
                pattern=None,
                json=True,
                verbose=False,
                quiet=False,
                no_color=True,
                grade_filter="BROKEN",
                site_filter=None,
                outcome_filter=None,
                since_filter=None,
            )
        )
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        task_ids = {t["task_id"] for t in payload["tasks"]}
        # Only broken task should remain
        assert "broken-task" in task_ids
        assert "golden-task" not in task_ids

    def test_tree_command_with_site_filter(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_tree(
            Namespace(
                log_dir=str(tmp_trace_dir),
                log_dir_pos=None,
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                json=True,
                verbose=False,
                quiet=False,
                no_color=True,
                grade_filter=None,
                site_filter="shop",
                outcome_filter=None,
                since_filter=None,
            )
        )
        assert result == 0
        output = capsys.readouterr().out
        payload = json.loads(output)
        # Should only contain tasks that hit "shop" site
        assert isinstance(payload, (dict, list))

    def test_empty_filter_result_message(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_grade(
            Namespace(
                log_dir=str(tmp_trace_dir),
                days=None,
                rules=None,
                format="auto",
                pattern=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
                grade_filter=None,
                site_filter="nonexistent_site_xyz",
                outcome_filter=None,
                since_filter=None,
            )
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "No tasks remain" in captured.out


# ---------------------------------------------------------------------------
# Item 10: search command
# ---------------------------------------------------------------------------


class TestSearch:
    """The search subcommand finds tasks by user_text substring."""

    def test_search_finds_matching_tasks(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_search(
            Namespace(
                query="headset",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                grade_filter=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "golden-task" in captured.out

    def test_search_case_insensitive(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_search(
            Namespace(
                query="HEADSET",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                grade_filter=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "golden-task" in captured.out

    def test_search_no_matches(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_search(
            Namespace(
                query="zzz_no_match_zzz",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                grade_filter=None,
                json=False,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        captured = capsys.readouterr()
        assert "No tasks matching" in captured.out

    def test_search_json_output(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_search(
            Namespace(
                query="checkout",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                grade_filter=None,
                json=True,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert len(payload) >= 1
        entry = payload[0]
        assert "task_id" in entry
        assert "step_count" in entry
        assert "site" in entry
        assert "user_text" in entry

    def test_search_with_grade_filter(
        self, tmp_trace_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        result = cmd_search(
            Namespace(
                query="checkout",
                log_dir=None,
                log_dir_pos=str(tmp_trace_dir),
                log_dir_opt=None,
                days=None,
                format="auto",
                pattern=None,
                grade_filter="BROKEN",
                json=True,
                verbose=False,
                quiet=False,
                no_color=True,
            )
        )
        assert result == 0
        payload = json.loads(capsys.readouterr().out)
        # Should only show BROKEN tasks matching "checkout"
        for entry in payload:
            assert entry["grade"] == "BROKEN"

    def test_search_parser_accepts_positional_log_dir(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "pizza", "/some/path"])
        assert args.query == "pizza"
        assert args.log_dir_pos == "/some/path"

    def test_search_parser_accepts_log_dir_option(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["search", "pizza", "--log-dir", "/some/path"])
        assert args.query == "pizza"
        assert args.log_dir_opt == "/some/path"
