"""Tests for watch command, timeline report, and spins report (Items 9, 11, 12)."""

from __future__ import annotations

import json
import textwrap
from argparse import Namespace
from pathlib import Path

import pytest

from agent_xray.analyzer import analyze_tasks
from agent_xray.cli import cmd_report, cmd_watch, _parse_bucket_arg
from agent_xray.grader import grade_tasks, load_rules
from agent_xray.reports import (
    report_spins,
    report_spins_data,
    report_spins_markdown,
    report_timeline,
    report_timeline_data,
    report_timeline_markdown,
)
from agent_xray.schema import AgentStep, AgentTask, TaskOutcome
from agent_xray.watch import (
    _build_task_from_accumulated,
    _extract_timestamp_time,
    _format_line,
    _format_tally,
    _truncate,
)


# ── Helpers ────────────────────────────────────────────────────────────


def _step(
    task_id: str,
    step: int,
    tool_name: str,
    tool_input: dict | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    duration_ms: int | None = None,
    timestamp: str | None = None,
    page_url: str | None = None,
) -> AgentStep:
    return AgentStep(
        task_id=task_id,
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        duration_ms=duration_ms,
        timestamp=timestamp,
        page_url=page_url,
    )


def _outcome(task_id: str, status: str, total_steps: int) -> TaskOutcome:
    return TaskOutcome(
        task_id=task_id,
        status=status,
        total_steps=total_steps,
        total_duration_s=total_steps * 0.5,
        timestamp="2026-03-26T12:30:00Z",
    )


def _make_spin_task(
    task_id: str = "spin-task",
    tool: str = "browser_click_ref",
    repeats: int = 5,
    *,
    page_url: str = "https://dominos.com/menu",
    error: str | None = None,
) -> AgentTask:
    """Build a task with a spin sequence."""
    steps = []
    for i in range(1, repeats + 1):
        steps.append(
            _step(
                task_id,
                i,
                tool,
                {"ref": f"@e{i}"},
                tool_result=f"Clicked ref @e{i}.",
                error=error,
                duration_ms=200,
                timestamp=f"2026-03-26T10:{i:02d}:00Z",
                page_url=page_url,
            )
        )
    return AgentTask(
        task_id=task_id,
        task_text="Order a large pepperoni pizza from Dominos.",
        task_category="commerce",
        steps=steps,
        outcome=_outcome(task_id, "failed", len(steps)),
    )


def _make_timestamped_tasks() -> list[AgentTask]:
    """Build several tasks across different hours for timeline testing."""
    tasks = []
    for hour in (10, 10, 11, 11, 11, 13):
        tid = f"task-h{hour}-{len(tasks)}"
        steps = [
            _step(
                tid,
                1,
                "browser_navigate",
                {"url": "https://shop.example.test"},
                tool_result="Homepage loaded.",
                duration_ms=500,
                timestamp=f"2026-03-26T{hour:02d}:05:00Z",
                page_url="https://shop.example.test/",
            ),
            _step(
                tid,
                2,
                "browser_click",
                {"ref": "buy-button"},
                tool_result="Clicked buy.",
                duration_ms=300,
                timestamp=f"2026-03-26T{hour:02d}:06:00Z",
                page_url="https://shop.example.test/checkout",
            ),
        ]
        tasks.append(
            AgentTask(
                task_id=tid,
                task_text="Buy a product.",
                task_category="commerce",
                steps=steps,
                outcome=TaskOutcome(
                    task_id=tid,
                    status="success",
                    total_steps=2,
                    total_duration_s=0.8,
                    timestamp=f"2026-03-26T{hour:02d}:06:30Z",
                ),
            )
        )
    return tasks


def _prepare(tasks):
    rules = load_rules()
    grades = grade_tasks(tasks, rules)
    analyses = analyze_tasks(tasks)
    return grades, analyses


def _report_ns(
    report_type: str,
    log_dir,
    *,
    use_json: bool = False,
    markdown: bool = False,
    bucket: str = "1h",
):
    return Namespace(
        log_dir=log_dir,
        days=None,
        rules=None,
        format="auto",
        pattern=None,
        report_type=report_type,
        json=use_json,
        markdown=markdown,
        day1=None,
        day2=None,
        bucket=bucket,
        verbose=False,
        quiet=False,
        no_color=True,
        grade_filter=None,
        site_filter=None,
        outcome_filter=None,
        since_filter=None,
    )


def _write_tasks_to_dir(path: Path, tasks: list[AgentTask]) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    trace_path = path / "trace_20260326.jsonl"
    lines: list[str] = []
    for task in tasks:
        for index, step in enumerate(task.sorted_steps):
            payload = step.to_dict()
            if index == 0:
                payload["user_text"] = task.task_text
                payload["task_category"] = task.task_category
            lines.append(json.dumps(payload, sort_keys=True))
        if task.outcome is not None:
            lines.append(
                json.dumps(
                    {
                        "event": "task_complete",
                        "task_id": task.task_id,
                        "status": task.outcome.status,
                        "final_answer": task.outcome.final_answer,
                        "total_steps": task.outcome.total_steps,
                        "total_duration_s": task.outcome.total_duration_s,
                        "timestamp": task.outcome.timestamp,
                    },
                    sort_keys=True,
                )
            )
    trace_path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ── Item 9: Watch module unit tests ───────────────────────────────────


class TestWatchHelpers:
    def test_truncate_short(self):
        assert _truncate("hello", 10) == "hello"

    def test_truncate_long(self):
        result = _truncate("a" * 60, 50)
        assert len(result) == 50
        assert result.endswith("...")

    def test_extract_timestamp_time_from_outcome(self):
        task = AgentTask(
            task_id="t1",
            steps=[],
            outcome=TaskOutcome(
                task_id="t1",
                status="success",
                timestamp="2026-03-26T10:15:40Z",
            ),
        )
        assert _extract_timestamp_time(task) == "10:15:40"

    def test_extract_timestamp_time_from_step(self):
        task = AgentTask(
            task_id="t1",
            steps=[
                _step("t1", 1, "browser_click", timestamp="2026-03-26T14:22:33Z"),
            ],
        )
        assert _extract_timestamp_time(task) == "14:22:33"

    def test_extract_timestamp_time_no_data(self):
        task = AgentTask(task_id="t1", steps=[])
        assert _extract_timestamp_time(task) == "??:??:??"

    def test_format_tally(self):
        counts = {"GOLDEN": 2, "GOOD": 3, "OK": 5, "WEAK": 1, "BROKEN": 0}
        result = _format_tally(counts, color=False)
        assert "Total: 11" in result
        assert "GOLDEN: 2" in result
        assert "BROKEN: 0" in result

    def test_format_line(self):
        task = AgentTask(
            task_id="abc123def456",
            task_text="Order pizza from Dominos.",
            steps=[_step("abc123def456", 1, "browser_click", timestamp="2026-03-26T10:00:00Z")],
            outcome=TaskOutcome(
                task_id="abc123def456",
                status="success",
                timestamp="2026-03-26T10:15:40Z",
            ),
        )
        from agent_xray.grader import GradeResult
        grade = GradeResult(
            task_id="abc123def456",
            grade="GOLDEN",
            score=4,
            reasons=[],
            metrics={},
            signals=[],
        )
        line = _format_line(task, grade, color=False)
        assert "abc123def456" in line
        assert "GOLDEN" in line
        assert "1 steps" in line
        assert "Order pizza" in line

    def test_build_task_from_accumulated(self):
        steps = [
            {
                "task_id": "t1",
                "step": 1,
                "tool_name": "browser_click",
                "tool_input": {"ref": "btn"},
                "user_text": "Do something.",
                "task_category": "commerce",
            },
            {
                "task_id": "t1",
                "step": 2,
                "tool_name": "browser_snapshot",
                "tool_input": {},
            },
        ]
        outcome = {
            "task_id": "t1",
            "event": "task_complete",
            "status": "success",
            "total_steps": 2,
            "timestamp": "2026-03-26T12:00:00Z",
        }
        task = _build_task_from_accumulated("t1", steps, outcome)
        assert task.task_id == "t1"
        assert len(task.steps) == 2
        assert task.task_text == "Do something."
        assert task.task_category == "commerce"
        assert task.outcome is not None
        assert task.outcome.status == "success"


class TestWatchFileMode:
    def test_watch_writes_jsonl_and_reads(self, tmp_path):
        """Test that watch_file can read a pre-written JSONL file."""
        # We test the building blocks since watch_file is an infinite loop.
        trace = tmp_path / "test.jsonl"
        lines = [
            json.dumps({
                "task_id": "t1",
                "step": 1,
                "tool_name": "browser_click",
                "tool_input": {"ref": "x"},
                "tool_result": "Clicked.",
                "timestamp": "2026-03-26T10:00:00Z",
            }),
            json.dumps({
                "event": "task_complete",
                "task_id": "t1",
                "status": "success",
                "total_steps": 1,
                "timestamp": "2026-03-26T10:00:05Z",
            }),
        ]
        trace.write_text("\n".join(lines) + "\n", encoding="utf-8")

        # Verify we can parse the file contents
        task = _build_task_from_accumulated(
            "t1",
            [json.loads(lines[0])],
            json.loads(lines[1]),
        )
        assert task.task_id == "t1"
        assert len(task.steps) == 1
        assert task.outcome is not None


# ── Item 11: Timeline Report tests ────────────────────────────────────


class TestTimelineReport:
    def test_timeline_text_has_header(self):
        tasks = _make_timestamped_tasks()
        grades, analyses = _prepare(tasks)
        text = report_timeline(tasks, grades, analyses)
        assert "TIMELINE REPORT" in text

    def test_timeline_text_has_buckets(self):
        tasks = _make_timestamped_tasks()
        grades, analyses = _prepare(tasks)
        text = report_timeline(tasks, grades, analyses)
        # Should have buckets for hours 10, 11, 13
        assert "10:00" in text or "2026-03-26 10" in text
        assert "Tasks" in text

    def test_timeline_data_structure(self):
        tasks = _make_timestamped_tasks()
        grades, analyses = _prepare(tasks)
        data = report_timeline_data(tasks, grades, analyses)
        assert "buckets" in data
        assert "bucket_minutes" in data
        assert data["bucket_minutes"] == 60
        assert len(data["buckets"]) >= 3  # at least 3 different hours

    def test_timeline_data_bucket_fields(self):
        tasks = _make_timestamped_tasks()
        grades, analyses = _prepare(tasks)
        data = report_timeline_data(tasks, grades, analyses)
        for bucket in data["buckets"]:
            assert "bucket" in bucket
            assert "tasks" in bucket
            assert "GOLDEN" in bucket
            assert "BROKEN" in bucket
            assert "avg_duration_s" in bucket
            assert "error_rate_pct" in bucket

    def test_timeline_markdown_has_table(self):
        tasks = _make_timestamped_tasks()
        grades, analyses = _prepare(tasks)
        text = report_timeline_markdown(tasks, grades, analyses)
        assert "## Timeline Report" in text
        assert "| Hour |" in text

    def test_timeline_empty(self):
        grades, analyses = _prepare([])
        text = report_timeline([], grades, analyses)
        assert "TIMELINE REPORT" in text
        assert "No tasks" in text

    def test_parse_bucket_arg_defaults(self):
        assert _parse_bucket_arg(None) == 60
        assert _parse_bucket_arg("") == 60
        assert _parse_bucket_arg("1h") == 60
        assert _parse_bucket_arg("2h") == 120
        assert _parse_bucket_arg("15m") == 15
        assert _parse_bucket_arg("30m") == 30
        assert _parse_bucket_arg("garbage") == 60

    def test_timeline_cli_text(self, tmp_path, capsys):
        tasks = _make_timestamped_tasks()
        trace_dir = _write_tasks_to_dir(tmp_path / "timeline-traces", tasks)
        assert cmd_report(_report_ns("timeline", trace_dir)) == 0
        out = capsys.readouterr().out
        assert "TIMELINE REPORT" in out

    def test_timeline_cli_json(self, tmp_path, capsys):
        tasks = _make_timestamped_tasks()
        trace_dir = _write_tasks_to_dir(tmp_path / "timeline-traces-json", tasks)
        assert cmd_report(_report_ns("timeline", trace_dir, use_json=True)) == 0
        data = json.loads(capsys.readouterr().out)
        assert "buckets" in data

    def test_timeline_cli_markdown(self, tmp_path, capsys):
        tasks = _make_timestamped_tasks()
        trace_dir = _write_tasks_to_dir(tmp_path / "timeline-traces-md", tasks)
        assert cmd_report(_report_ns("timeline", trace_dir, markdown=True)) == 0
        out = capsys.readouterr().out
        assert "## Timeline Report" in out


# ── Item 12: Spins Report tests ───────────────────────────────────────


class TestSpinsReport:
    def test_spins_text_has_header(self):
        tasks = [_make_spin_task()]
        grades, analyses = _prepare(tasks)
        text = report_spins(tasks, analyses)
        assert "SPIN ANALYSIS" in text

    def test_spins_text_shows_tool(self):
        tasks = [_make_spin_task(tool="browser_click_ref", repeats=5)]
        grades, analyses = _prepare(tasks)
        text = report_spins(tasks, analyses)
        assert "browser_click_ref" in text
        assert "BY TOOL:" in text

    def test_spins_text_shows_site(self):
        tasks = [_make_spin_task(page_url="https://dominos.com/menu")]
        grades, analyses = _prepare(tasks)
        text = report_spins(tasks, analyses)
        assert "BY SITE:" in text
        assert "dominos.com" in text

    def test_spins_text_shows_patterns(self):
        tasks = [_make_spin_task()]
        grades, analyses = _prepare(tasks)
        text = report_spins(tasks, analyses)
        assert "SPIN PATTERNS:" in text

    def test_spins_text_shows_worst(self):
        tasks = [_make_spin_task(repeats=10)]
        grades, analyses = _prepare(tasks)
        text = report_spins(tasks, analyses)
        assert "WORST SPIN SEQUENCES:" in text
        assert "10 repeats" in text

    def test_spins_data_structure(self):
        tasks = [_make_spin_task()]
        grades, analyses = _prepare(tasks)
        data = report_spins_data(tasks, analyses)
        assert data["tasks_with_spins"] == 1
        assert data["total_sequences"] >= 1
        assert "by_tool" in data
        assert "by_site" in data
        assert "by_pattern" in data
        assert "worst_sequences" in data

    def test_spins_data_by_tool(self):
        tasks = [_make_spin_task(tool="browser_click_ref")]
        grades, analyses = _prepare(tasks)
        data = report_spins_data(tasks, analyses)
        tools = [e["tool"] for e in data["by_tool"]]
        assert "browser_click_ref" in tools

    def test_spins_data_worst_sequences(self):
        tasks = [
            _make_spin_task("s1", repeats=10),
            _make_spin_task("s2", repeats=5),
        ]
        grades, analyses = _prepare(tasks)
        data = report_spins_data(tasks, analyses)
        # Worst should be sorted by repeats descending
        worst = data["worst_sequences"]
        assert len(worst) >= 2
        assert worst[0]["repeats"] >= worst[1]["repeats"]

    def test_spins_ref_not_found_pattern(self):
        task = _make_spin_task(
            task_id="ref-404",
            error="Ref @e5 not found on page.",
            repeats=4,
        )
        grades, analyses = _prepare([task])
        data = report_spins_data([task], analyses)
        patterns = [p["pattern"] for p in data["by_pattern"]]
        assert "ref_not_found_loop" in patterns

    def test_spins_search_retry_pattern(self):
        task_id = "search-retry"
        steps = [
            _step(task_id, i, "web_search", {"query": "pizza near me"},
                  tool_result="No results.", timestamp=f"2026-03-26T10:{i:02d}:00Z")
            for i in range(1, 5)
        ]
        task = AgentTask(
            task_id=task_id,
            task_text="Search for pizza.",
            steps=steps,
            outcome=_outcome(task_id, "failed", 4),
        )
        grades, analyses = _prepare([task])
        data = report_spins_data([task], analyses)
        patterns = [p["pattern"] for p in data["by_pattern"]]
        assert "search_retry" in patterns

    def test_spins_markdown_has_table(self):
        tasks = [_make_spin_task()]
        grades, analyses = _prepare(tasks)
        text = report_spins_markdown(tasks, analyses)
        assert "## Spin Analysis" in text
        assert "| Tool |" in text

    def test_spins_empty(self):
        # Task with no spins (only 2 consecutive)
        task_id = "no-spin"
        steps = [
            _step(task_id, 1, "browser_navigate", timestamp="2026-03-26T10:00:00Z"),
            _step(task_id, 2, "browser_click", timestamp="2026-03-26T10:01:00Z"),
            _step(task_id, 3, "browser_snapshot", timestamp="2026-03-26T10:02:00Z"),
        ]
        task = AgentTask(
            task_id=task_id,
            task_text="Do something.",
            steps=steps,
            outcome=_outcome(task_id, "success", 3),
        )
        grades, analyses = _prepare([task])
        text = report_spins([task], analyses)
        assert "No spin sequences detected" in text

    def test_spins_cli_text(self, tmp_path, capsys):
        tasks = [_make_spin_task()]
        trace_dir = _write_tasks_to_dir(tmp_path / "spin-traces", tasks)
        assert cmd_report(_report_ns("spins", trace_dir)) == 0
        out = capsys.readouterr().out
        assert "SPIN ANALYSIS" in out

    def test_spins_cli_json(self, tmp_path, capsys):
        tasks = [_make_spin_task()]
        trace_dir = _write_tasks_to_dir(tmp_path / "spin-traces-json", tasks)
        assert cmd_report(_report_ns("spins", trace_dir, use_json=True)) == 0
        data = json.loads(capsys.readouterr().out)
        assert "tasks_with_spins" in data

    def test_spins_cli_markdown(self, tmp_path, capsys):
        tasks = [_make_spin_task()]
        trace_dir = _write_tasks_to_dir(tmp_path / "spin-traces-md", tasks)
        assert cmd_report(_report_ns("spins", trace_dir, markdown=True)) == 0
        out = capsys.readouterr().out
        assert "## Spin Analysis" in out


# ── Watch CLI subcommand parser test ──────────────────────────────────


class TestWatchCLI:
    def test_watch_parser_exists(self):
        from agent_xray.cli import build_parser
        parser = build_parser()
        # Verify watch subcommand is registered by parsing args
        args = parser.parse_args(["watch", "test.jsonl"])
        assert args.file == "test.jsonl"
        assert args.func == cmd_watch

    def test_watch_parser_with_rules(self):
        from agent_xray.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "test.jsonl", "--rules", "browser_flow"])
        assert args.rules == "browser_flow"

    def test_watch_parser_with_json(self):
        from agent_xray.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "test.jsonl", "--json"])
        assert args.json is True

    def test_watch_parser_with_poll(self):
        from agent_xray.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(["watch", "test.jsonl", "--poll", "5.0"])
        assert args.poll == 5.0
