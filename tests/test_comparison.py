from __future__ import annotations

import json
from pathlib import Path

from agent_xray.comparison import (
    _detect_comparison_type,
    compare_model_runs,
    format_model_comparison,
)
from agent_xray.schema import AgentTask


def _divergent_variant(
    task: AgentTask,
    task_id: str,
    clone_task,
) -> AgentTask:
    cloned = clone_task(task, task_id)
    cloned.steps[1].tool_name = "browser_snapshot"
    cloned.steps[1].tool_input = {"focus": "cart-status"}
    return cloned


def test_compare_identical_dirs(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    tasks = [clone_task(golden_task, "task-1"), clone_task(broken_task, "task-2")]
    left_dir = write_trace_dir("identical-left", tasks)
    right_dir = write_trace_dir("identical-right", tasks)

    result = compare_model_runs(left_dir, right_dir)

    assert result.matched_tasks == 2
    assert result.divergences == []
    assert all(delta == 0 for delta in result.grade_deltas.values())


def test_compare_detects_grade_improvement(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    left_dir = write_trace_dir("improve-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("improve-right", [clone_task(golden_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)

    assert result.grade_deltas["GOLDEN"] == 1
    assert result.grade_deltas["BROKEN"] == -1


def test_compare_detects_grade_regression(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    left_dir = write_trace_dir("regress-left", [clone_task(golden_task, "checkout-task")])
    right_dir = write_trace_dir("regress-right", [clone_task(broken_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)

    assert result.grade_deltas["GOLDEN"] == -1
    assert result.grade_deltas["BROKEN"] == 1


def test_compare_finds_divergence_point(
    write_trace_dir,
    golden_task: AgentTask,
    clone_task,
) -> None:
    left_dir = write_trace_dir("diverge-left", [clone_task(golden_task, "checkout-task")])
    right_dir = write_trace_dir(
        "diverge-right", [_divergent_variant(golden_task, "checkout-task", clone_task)]
    )

    result = compare_model_runs(left_dir, right_dir)

    assert len(result.divergences) == 1
    assert result.divergences[0].task_id == "checkout-task"
    assert result.divergences[0].step == 2


def test_compare_cost_comparison(write_trace_dir, golden_task: AgentTask, clone_task) -> None:
    left_dir = write_trace_dir(
        "cost-left",
        [clone_task(golden_task, "checkout-task", model_name="model-left", cost_usd=0.05)],
    )
    right_dir = write_trace_dir(
        "cost-right",
        [clone_task(golden_task, "checkout-task", model_name="model-right", cost_usd=0.02)],
    )

    result = compare_model_runs(left_dir, right_dir)

    assert result.left_label == "model-left"
    assert result.right_label == "model-right"
    assert result.left_cost.total_cost > result.right_cost.total_cost
    assert result.left_cost.avg_cost_per_task > result.right_cost.avg_cost_per_task


def test_compare_empty_dirs(write_trace_dir) -> None:
    left_dir = write_trace_dir("empty-left", [])
    right_dir = write_trace_dir("empty-right", [])

    result = compare_model_runs(left_dir, right_dir)

    assert result.matched_tasks == 0
    assert result.divergences == []
    assert result.left_grade_distribution == {}
    assert result.right_grade_distribution == {}


def test_compare_mismatched_tasks(
    write_trace_dir,
    golden_task: AgentTask,
    research_task: AgentTask,
    clone_task,
) -> None:
    left_dir = write_trace_dir("mismatch-left", [clone_task(golden_task, "left-task")])
    right_dir = write_trace_dir("mismatch-right", [clone_task(research_task, "right-task")])

    result = compare_model_runs(left_dir, right_dir)

    assert result.matched_tasks == 0
    assert result.divergences == []


def test_compare_json_output(write_trace_dir, golden_task: AgentTask, clone_task) -> None:
    left_dir = write_trace_dir("json-left", [clone_task(golden_task, "task-json")])
    right_dir = write_trace_dir("json-right", [clone_task(golden_task, "task-json")])

    result = compare_model_runs(left_dir, right_dir)
    payload = result.to_dict()

    assert json.loads(json.dumps(payload))["matched_tasks"] == 1
    assert payload["rules_name"] == "default"


# --- BUG #5: Divergence detection ---


def test_divergence_summary_flags_grade_shifts(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    """When grades differ between left and right, divergence_summary must report it."""
    left_dir = write_trace_dir("div-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("div-right", [clone_task(golden_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)

    ds = result.divergence_summary
    assert ds["has_grade_divergence"] is True
    assert "GOLDEN" in ds["grade_shifts"]
    assert ds["grade_shifts"]["GOLDEN"] == 1
    assert "BROKEN" in ds["grade_shifts"]
    assert ds["grade_shifts"]["BROKEN"] == -1


def test_divergence_summary_no_shifts_when_identical(
    write_trace_dir,
    golden_task: AgentTask,
    clone_task,
) -> None:
    """When runs are identical, divergence_summary should report no grade divergence."""
    tasks = [clone_task(golden_task, "task-1")]
    left_dir = write_trace_dir("nodiv-left", tasks)
    right_dir = write_trace_dir("nodiv-right", tasks)

    result = compare_model_runs(left_dir, right_dir)

    ds = result.divergence_summary
    assert ds["has_grade_divergence"] is False
    assert ds["grade_shifts"] == {}


def test_divergence_summary_includes_success_pct(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    """Divergence summary should include success rate percentage delta."""
    left_dir = write_trace_dir("pct-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("pct-right", [clone_task(golden_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)

    ds = result.divergence_summary
    assert "left_success_pct" in ds
    assert "right_success_pct" in ds
    assert "success_pct_delta" in ds
    assert ds["right_success_pct"] > ds["left_success_pct"]


def test_root_cause_distribution_populated(
    write_trace_dir,
    broken_task: AgentTask,
    clone_task,
) -> None:
    """Root cause distribution should be populated for runs with BROKEN tasks."""
    left_dir = write_trace_dir("rc-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("rc-right", [clone_task(broken_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)

    # broken_task should classify to some root cause
    assert result.left_root_cause_distribution
    assert result.right_root_cause_distribution


def test_format_model_comparison_shows_divergence_summary(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    """format_model_comparison should include a Divergence Summary section."""
    left_dir = write_trace_dir("fmt-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("fmt-right", [clone_task(golden_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)
    text = format_model_comparison(result)

    assert "Divergence Summary:" in text
    assert "Grade shifts:" in text
    assert "GOLDEN" in text


# --- BUG #9: Auto-detect comparison type ---


def test_detect_day_comparison_from_dir_names() -> None:
    """Directories with date stamps should produce a day comparison header."""
    ctype, header = _detect_comparison_type(
        Path("logs/2026-03-22"), Path("logs/2026-03-23"), "gpt-5", "gpt-5"
    )
    assert ctype == "day"
    assert "2026-03-22" in header
    assert "2026-03-23" in header


def test_detect_day_comparison_underscore_dates() -> None:
    """Date-stamped directories with underscores should also be detected."""
    ctype, header = _detect_comparison_type(
        Path("traces_20260322"), Path("traces_20260323"), "left", "right"
    )
    assert ctype == "day"
    assert "2026" in header


def test_detect_model_comparison_from_dir_names() -> None:
    """Directories with model names should produce a model comparison header."""
    ctype, header = _detect_comparison_type(
        Path("runs/gpt-4-run"), Path("runs/claude-3-run"), "gpt-4", "claude-3"
    )
    assert ctype == "model"
    assert "Model Comparison" in header


def test_detect_model_comparison_from_labels() -> None:
    """When dir names are generic but labels contain model names, detect model comparison."""
    ctype, header = _detect_comparison_type(
        Path("run-a"), Path("run-b"), "gpt-5-mini", "claude-3-sonnet"
    )
    assert ctype == "model"
    assert "Model Comparison" in header


def test_detect_generic_run_comparison() -> None:
    """Directories without dates or model names should produce a run comparison header."""
    ctype, header = _detect_comparison_type(
        Path("experiment-a"), Path("experiment-b"), "left", "right"
    )
    assert ctype == "run"
    assert "Run Comparison" in header


def test_compare_header_uses_detected_type(
    write_trace_dir,
    golden_task: AgentTask,
    clone_task,
    tmp_path,
) -> None:
    """The comparison_header on the result should reflect auto-detected type."""
    tasks = [clone_task(golden_task, "task-1")]
    # Use date-stamped directory names
    left_dir = tmp_path / "traces_20260322"
    right_dir = tmp_path / "traces_20260323"
    from tests.conftest import _write_tasks
    _write_tasks(left_dir, tasks)
    _write_tasks(right_dir, tasks)

    result = compare_model_runs(left_dir, right_dir)

    assert result.comparison_type == "day"
    assert "2026" in result.comparison_header
    text = format_model_comparison(result)
    assert "Day Comparison" in text


def test_json_output_includes_new_fields(
    write_trace_dir,
    golden_task: AgentTask,
    broken_task: AgentTask,
    clone_task,
) -> None:
    """to_dict() should include the new comparison_type and divergence_summary fields."""
    left_dir = write_trace_dir("jsonv2-left", [clone_task(broken_task, "checkout-task")])
    right_dir = write_trace_dir("jsonv2-right", [clone_task(golden_task, "checkout-task")])

    result = compare_model_runs(left_dir, right_dir)
    payload = result.to_dict()
    roundtripped = json.loads(json.dumps(payload))

    assert "comparison_type" in roundtripped
    assert "comparison_header" in roundtripped
    assert "divergence_summary" in roundtripped
    assert "left_root_cause_distribution" in roundtripped
    assert "right_root_cause_distribution" in roundtripped
    assert "root_cause_deltas" in roundtripped
