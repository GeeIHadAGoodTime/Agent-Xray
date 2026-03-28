"""MCP server exposing agent-xray enforce and analysis helpers as tools."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

server = FastMCP("agent-xray")


def _json_response(payload: Any) -> str:
    return json.dumps(payload, indent=2)


def _serialize(value: Any) -> Any:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


def _load_tasks(log_dir: str, format_name: str) -> list[Any]:
    from agent_xray.analyzer import load_adapted_tasks, load_tasks

    if format_name != "auto":
        return load_adapted_tasks(log_dir, format=format_name)

    tasks = load_tasks(log_dir)
    if tasks:
        return tasks
    return load_adapted_tasks(log_dir, format="auto")


@server.tool()
def enforce_init(
    test_command: str,
    project_root: str = ".",
    max_iterations: int = 50,
    stash_first: bool = False,
    max_files_per_change: int = 5,
    max_diff_lines: int = 200,
) -> str:
    """Start the enforce workflow by capturing a baseline run and creating session state.

    Use this first, before any code change, when you want repeatable before/after evidence.
    Prerequisites: a git repo, a deterministic test command, and the correct project root.
    Common mistakes: using ad-hoc manual checks, pointing at flaky tests, or initializing after edits already exist.
    Next step: call `enforce_plan` to register your hypothesis, then make one small edit and call `enforce_check`.
    """
    try:
        from agent_xray.enforce import EnforceConfig, enforce_init as run_enforce_init

        config = EnforceConfig(
            test_command=test_command,
            project_root=project_root,
            max_iterations=max_iterations,
            stash_first=stash_first,
            max_files_per_change=max_files_per_change,
            max_diff_lines=max_diff_lines,
        )
        baseline, session_dir = run_enforce_init(config)
        return _json_response(
            {
                "baseline": _serialize(baseline),
                "session_dir": str(session_dir),
            }
        )
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_check(hypothesis: str = "", project_root: str = ".") -> str:
    """Evaluate one proposed change against the active enforce baseline and return the full change record.

    Use this after exactly one small edit, ideally after `enforce_plan`, to measure real before/after movement.
    Prerequisites: an active session from `enforce_init` and the same deterministic test setup used for baseline.
    Common mistakes: batching unrelated edits, changing tests to make progress look better, or skipping hypothesis tracking.
    Next step: if COMMITTED, call `enforce_plan` for the next hypothesis. If REVERTED, revise your approach and call `enforce_plan` again. After several iterations, call `enforce_challenge` to audit for gaming.
    """
    try:
        from agent_xray.enforce import enforce_check as run_enforce_check

        record = run_enforce_check(hypothesis=hypothesis, project_root=project_root)
        return _json_response(_serialize(record))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_diff(project_root: str = ".") -> str:
    """Preview the current diff and whether enforce would reject it for scope before you spend a test run.

    Use this between `enforce_plan` and `enforce_check` when you want to confirm the change still fits one-change-at-a-time discipline.
    Prerequisites: an enforce session is recommended, and `project_root` must point at the repo you are editing.
    Common mistakes: assuming a large multi-file diff is still a clean experiment or using this instead of `enforce_check`.
    Next step: if the diff looks clean, call `enforce_check` to measure the change. If too large, split your edit first.
    """
    try:
        from agent_xray.enforce import enforce_diff as run_enforce_diff

        return _json_response(run_enforce_diff(project_root=project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_plan(
    hypothesis: str,
    expected_tests: list[str] | None = None,
    project_root: str = ".",
) -> str:
    """Register the next hypothesis and expected test movement before editing code.

    Use this immediately after `enforce_init` or after a previous iteration is resolved, before touching files.
    Prerequisites: an active enforce session and a concrete, falsifiable hypothesis for one change.
    Common mistakes: vague hypotheses, predicting unrelated tests, or treating plan as optional when you want disciplined iteration history.
    Next step: make exactly one small edit, optionally call `enforce_diff` to preview scope, then call `enforce_check` to measure the result.
    """
    try:
        from agent_xray.enforce import enforce_plan as run_enforce_plan

        return _json_response(
            run_enforce_plan(
                hypothesis=hypothesis,
                expected_tests=expected_tests,
                project_root=project_root,
            )
        )
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_guard(project_root: str = ".") -> str:
    """Check whether there are unreviewed working-tree changes outside the enforce loop.

    Use this when an agent resumes work or before `enforce_check` if you suspect stray edits.
    Prerequisites: the target repo must be reachable via `project_root`.
    Common mistakes: assuming the working tree is clean because tests pass or ignoring edits created outside the tracked hypothesis.
    Next step: if stray edits are found, revert or commit them before calling `enforce_plan` for your next hypothesis.
    """
    try:
        from agent_xray.enforce import enforce_guard as run_enforce_guard

        return _json_response(run_enforce_guard(project_root=project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_status(project_root: str = ".") -> str:
    """Read the current enforce session state, including baseline, counts, and pending history.

    Use this to rehydrate context before the next iteration or to inspect where the session currently stands.
    Prerequisites: an existing session under the target project root.
    Common mistakes: treating status as evidence of improvement instead of running `enforce_check`.
    Next step: call `enforce_plan` to register your next hypothesis, or call `enforce_challenge` to audit cumulative progress.
    """
    try:
        from agent_xray.enforce import enforce_status as run_enforce_status

        return _json_response(run_enforce_status(project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_challenge(project_root: str = ".") -> str:
    """Run the adversarial cross-iteration audit over the current enforce session.

    Use this after one or more checks to catch gaming, scope creep, repeated hot-file churn, and other cumulative failure modes.
    Prerequisites: an active session with recorded iterations.
    Common mistakes: relying on per-iteration checks alone or using challenge before any enforce history exists.
    Next step: if clean, call `enforce_report` for a shareable summary. If gaming detected, revert suspect iterations and re-plan.
    """
    try:
        from agent_xray.enforce import enforce_challenge as run_enforce_challenge

        result = run_enforce_challenge(project_root=project_root)
        return _json_response(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_reset(project_root: str = ".") -> str:
    """Abandon the current enforce session and remove its persisted session state.

    Use this only when you intentionally want to discard the current experiment and start a new baseline.
    Prerequisites: an existing session under the target project root.
    Common mistakes: resetting when you really wanted `challenge`, `status`, or a new `check` on the same experiment.
    Next step: call `enforce_init` to start a fresh session with a new baseline.
    """
    try:
        from agent_xray.enforce import enforce_reset as run_enforce_reset

        return _json_response({"success": run_enforce_reset(project_root)})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_report(project_root: str = ".", format: str = "json") -> str:
    """Generate the final enforce report in text, JSON, or Markdown.

    Use this after a session has real history and you want a shareable summary of baseline, iterations, gaming signals, and outcomes.
    Prerequisites: an existing enforce session and a valid format of `text`, `json`, or `markdown`.
    Common mistakes: calling it before initializing a session or treating the report as a substitute for per-iteration checks.
    Next step: this is typically the last enforce action. Call `enforce_reset` if you want to start a new experiment.
    """
    try:
        from agent_xray.enforce import build_enforce_report
        from agent_xray.enforce_report import (
            format_enforce_json,
            format_enforce_markdown,
            format_enforce_text,
        )

        report = build_enforce_report(project_root)
        if format == "text":
            rendered: Any = format_enforce_text(report, color=False)
        elif format == "markdown":
            rendered = format_enforce_markdown(report)
        elif format == "json":
            rendered = json.loads(format_enforce_json(report))
        else:
            raise ValueError("format must be one of: text, json, markdown")

        return _json_response({"format": format, "report": rendered})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def analyze(log_dir: str, rules: str | None = None, format: str = "auto", task_bank: str | None = None) -> str:
    """Start here. Analyze agent traces to get grade distribution, root causes, and a fix plan.

    Use this near the start of trace triage when you want the broad picture before drilling into individual tasks.
    Prerequisites: a readable trace directory or JSONL file and, optionally, a valid ruleset name or path.
    High-value path: provide task_bank (path to task_bank.json) for expectation-aware grading that checks each task against its defined success criteria.
    Common mistakes: using analysis output as a substitute for deterministic task evaluation or forgetting to choose the correct trace format.
    Next step: call `grade` (with task_bank if available) for scored per-task details, then call `root_cause` to classify failures. Use `completeness` to verify trace quality if results look suspicious.
    """
    try:
        from agent_xray.analyzer import analyze_tasks
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        grades = grade_tasks(tasks, rule_set)
        grade_by_task = {grade.task_id: grade for grade in grades}
        analyses = analyze_tasks(tasks)
        distribution: dict[str, int] = {}
        for grade in grades:
            distribution[grade.grade] = distribution.get(grade.grade, 0) + 1

        payload = {
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "grade_distribution": distribution,
            },
            "tasks": [
                {
                    "task_id": task.task_id,
                    "analysis": _serialize(analyses[task.task_id]),
                    "grade": _serialize(grade_by_task[task.task_id]),
                }
                for task in tasks
            ],
        }
        return _json_response(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def grade(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None) -> str:
    """Grade traces against a ruleset and return scored per-task details.

    Use this after `analyze` when you need a scored view of which tasks are golden, weak, or broken.
    High-value path: provide task_bank (path to task_bank.json) for expectation-aware grading. This matches each task to its bank entry and evaluates success_criteria (must_reach_url, must_answer_contains, payment_fields_visible, etc.). Without task_bank, grading uses generic signals only.
    Prerequisites: a readable trace selection and a ruleset that matches the task domain.
    Common mistakes: grading without a task bank (misses expectation failures), mixing incomparable runs, or assuming grades explain root cause by themselves.
    Next step: call `root_cause` on the same traces to classify why tasks are failing. For individual task deep-dives, inspect specific task_ids from the results.
    """
    try:
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)

        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)

        distribution: dict[str, int] = {}
        for result in grades:
            distribution[result.grade] = distribution.get(result.grade, 0) + 1

        payload: dict[str, Any] = {
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "task_bank": task_bank or "none (generic grading only)",
                "distribution": distribution,
            },
            "tasks": [_serialize(result) for result in grades],
        }
        return _json_response(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def root_cause(log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Classify weak or broken tasks into likely root causes and return grouped plus per-task results.

    Use this after `grade` when you want an evidence-backed shortlist of likely failure modes to investigate next.
    Prerequisites: trace data that grades poorly enough to classify and a ruleset appropriate for the task domain.
    Common mistakes: skipping surface inspection on critical tasks or treating the classifier as ground truth instead of a ranked heuristic.
    Next step: use the classified root causes to prioritize fixes. Call `completeness` if you suspect missing trace data is skewing results.
    """
    try:
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures, summarize_root_causes

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)
        grades = grade_tasks(tasks, rule_set)
        failures = classify_failures(tasks, grades)
        return _json_response(
            {
                "summary": {
                    "tasks": len(tasks),
                    "rules": rule_set.name,
                    "classified_failures": len(failures),
                },
                "distribution": summarize_root_causes(failures),
                "tasks": [_serialize(result) for result in failures],
            }
        )
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def completeness(log_dir: str, format: str = "auto") -> str:
    """Measure how complete the trace instrumentation is across key observability dimensions.

    Use this before trusting grades, root-cause labels, or decision-surface output from a new integration.
    Prerequisites: a readable trace directory or JSONL file.
    Common mistakes: debugging agent behavior before confirming that prompt, tool, reasoning, and outcome data were actually captured.
    Next step: if completeness is low, fix instrumentation gaps before re-running `analyze`. If completeness is high, proceed to `grade` and `root_cause` with confidence.
    """
    try:
        from agent_xray.completeness import check_completeness

        tasks = _load_tasks(log_dir, format)
        report = check_completeness(tasks)
        return _json_response(
            {
                "score": report.score,
                "score_pct": report.score_pct,
                "dimensions_checked": report.dimensions_checked,
                "dimensions_ok": report.dimensions_ok,
                "all_dimensions": report.all_dimensions,
                "warnings": [_serialize(warning) for warning in report.warnings],
            }
        )
    except Exception as e:
        return _json_response({"error": str(e)})


def main() -> None:
    """Run the agent-xray MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
