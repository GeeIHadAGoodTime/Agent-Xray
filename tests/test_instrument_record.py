"""Tests for the agent-xray record CLI command."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from agent_xray.analyzer import load_tasks
from agent_xray.cli import build_parser, cmd_record


def _make_args(
    output_dir: str,
    command: list[str],
    *,
    task_id: str | None = None,
) -> Any:
    """Build a namespace matching what argparse would produce."""
    import argparse

    ns = argparse.Namespace()
    ns.output_dir = output_dir
    ns.task_id = task_id
    ns.command = command
    ns.verbose = False
    ns.quiet = False
    ns.no_color = True
    return ns


def test_record_no_command(tmp_path: Path) -> None:
    args = _make_args(str(tmp_path), [])
    result = cmd_record(args)
    assert result == 1


def test_record_captures_json_steps(tmp_path: Path) -> None:
    """Test recording from a subprocess that prints JSON tool calls to stdout."""
    script = tmp_path / "agent.py"
    lines = [
        json.dumps({"tool_name": "search", "tool_input": {"q": "test"}, "tool_result": "found"}),
        json.dumps({"tool_name": "click", "tool_input": {"ref": "btn"}, "duration_ms": 100}),
        "This is just regular output that should be ignored",
        json.dumps({"tool_name": "fill", "tool_input": {"ref": "form"}, "error": "Timed out"}),
    ]
    script.write_text(
        "import sys\n" + "\n".join(f"print({line!r})" for line in lines) + "\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "traces"
    args = _make_args(str(output_dir), ["--", sys.executable, str(script)], task_id="rec-task")
    result = cmd_record(args)
    assert result == 0

    tasks = load_tasks(output_dir)
    assert len(tasks) == 1
    task = tasks[0]
    assert task.task_id == "rec-task"
    assert len(task.steps) == 3
    assert task.steps[0].tool_name == "search"
    assert task.steps[1].tool_name == "click"
    assert task.steps[2].tool_name == "fill"
    assert task.steps[2].error == "Timed out"
    assert task.outcome is not None
    assert task.outcome.status == "success"


def test_record_failing_subprocess(tmp_path: Path) -> None:
    script = tmp_path / "fail_agent.py"
    script.write_text(
        "import sys, json\n"
        'print(json.dumps({"tool_name": "step1", "tool_input": {}}))\n'
        "sys.exit(1)\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "traces"
    args = _make_args(str(output_dir), ["--", sys.executable, str(script)], task_id="fail-rec")
    result = cmd_record(args)
    assert result == 1

    tasks = load_tasks(output_dir)
    assert len(tasks) == 1
    assert tasks[0].outcome is not None
    assert tasks[0].outcome.status == "failed"


def test_record_nonexistent_command(tmp_path: Path) -> None:
    output_dir = tmp_path / "traces"
    args = _make_args(str(output_dir), ["--", "nonexistent_binary_xyz123"])
    result = cmd_record(args)
    assert result == 1


def test_record_parser_registered() -> None:
    """Verify the record subcommand is registered in the CLI parser."""
    parser = build_parser()
    # This should not raise
    args = parser.parse_args(["record", "--output-dir", "./traces", "--", "echo", "hi"])
    assert args.command == ["--", "echo", "hi"]
    assert args.output_dir == "./traces"
