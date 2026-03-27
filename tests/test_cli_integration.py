from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from agent_xray.schema import AgentTask

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"


def _run_cli(*args: str, extra_pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    pythonpath_entries = [str(SRC_DIR)]
    if extra_pythonpath is not None:
        pythonpath_entries.insert(0, str(extra_pythonpath))
    if env.get("PYTHONPATH"):
        pythonpath_entries.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    return subprocess.run(
        [sys.executable, "-m", "agent_xray.cli", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _json_output(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.stdout.strip(), f"expected JSON output, stderr was: {result.stderr}"
    return json.loads(result.stdout)


def test_cli_analyze_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("analyze", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["summary"]["tasks"] == 4


def test_cli_grade_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("grade", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["summary"]["tasks"] == 4
    assert any(item["task_id"] == "broken-task" for item in payload["tasks"])


def test_cli_surface_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("surface", "golden-task", "--log-dir", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["task_id"] == "golden-task"
    assert payload["steps"][0]["tool_name"] == "browser_navigate"


def test_cli_reasoning_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("reasoning", "golden-task", "--log-dir", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["task_id"] == "golden-task"
    assert payload["reasoning_chain"][0]["decision"]["tool_name"] == "browser_navigate"


def test_cli_diff_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli(
        "diff",
        "golden-task",
        "broken-task",
        "--log-dir",
        str(tmp_trace_dir),
        "--json",
    )
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["diverged_at_step"] == 1


def test_cli_tree_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("tree", "--log-dir", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert "20260326" in payload


def test_cli_capture_and_replay_subprocess(tmp_path: Path, tmp_trace_dir: Path) -> None:
    fixture_path = tmp_path / "captured-task.json"
    capture = _run_cli(
        "capture",
        "golden-task",
        "--log-dir",
        str(tmp_trace_dir),
        "--out",
        str(fixture_path),
        "--json",
    )
    capture_payload = _json_output(capture)
    assert capture.returncode == 0
    assert Path(str(capture_payload["fixture"])).exists()

    replay = _run_cli(
        "replay",
        str(fixture_path),
        "--log-dir",
        str(tmp_trace_dir),
        "--json",
    )
    replay_payload = _json_output(replay)
    assert replay.returncode == 0
    assert replay_payload["fixture_task_id"] == "golden-task"


def test_cli_flywheel_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("flywheel", str(tmp_trace_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["total_tasks"] == 4


def test_cli_compare_subprocess(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    left_dir = write_trace_dir("compare-left-cli", [clone_task(broken_task, "shared-task")])
    right_dir = write_trace_dir("compare-right-cli", [clone_task(golden_task, "shared-task")])
    result = _run_cli("compare", str(left_dir), str(right_dir), "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["matched_tasks"] == 1
    assert payload["grade_deltas"]["GOOD"] == 1


def test_cli_report_subprocess(tmp_trace_dir: Path) -> None:
    result = _run_cli("report", str(tmp_trace_dir), "health", "--json")
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["total"] == 4


def test_cli_tui_subprocess_reports_missing_textual(tmp_path: Path, tmp_trace_dir: Path) -> None:
    shadow_dir = tmp_path / "shadow"
    textual_dir = shadow_dir / "textual"
    textual_dir.mkdir(parents=True)
    (textual_dir / "__init__.py").write_text(
        "raise ImportError('shadow textual')\n", encoding="utf-8"
    )

    result = _run_cli("tui", str(tmp_trace_dir), extra_pythonpath=shadow_dir)
    assert result.returncode == 1
    assert "TUI requires textual" in result.stdout


def test_cli_invalid_task_reports_error_message(tmp_trace_dir: Path) -> None:
    result = _run_cli("surface", "missing-task", "--log-dir", str(tmp_trace_dir), "--json")
    assert result.returncode != 0
    assert "task id 'missing-task' not found" in result.stdout


def test_cli_report_compare_requires_days(tmp_trace_dir: Path) -> None:
    result = _run_cli("report", str(tmp_trace_dir), "compare")
    assert result.returncode == 1
    assert "--day1 and --day2 are required" in result.stdout


def test_cli_invalid_format_exits_with_argparse_error(tmp_trace_dir: Path) -> None:
    result = _run_cli("analyze", str(tmp_trace_dir), "--format", "invalid-format")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_cli_analyze_supports_each_adapter_format(
    adapter_format: str, adapter_trace_paths: dict[str, Path]
) -> None:
    result = _run_cli(
        "analyze",
        str(adapter_trace_paths[adapter_format]),
        "--format",
        adapter_format,
        "--json",
    )
    payload = _json_output(result)
    assert result.returncode == 0
    assert payload["summary"]["tasks"] >= 1
