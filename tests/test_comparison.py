from __future__ import annotations

import json

from agent_xray.comparison import compare_model_runs
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
