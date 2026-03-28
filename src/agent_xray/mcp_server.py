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
    max_files_per_change: int = 5,
    max_diff_lines: int = 200,
) -> str:
    """Initialize an enforcement session by capturing a baseline test run and session directory."""
    try:
        from agent_xray.enforce import EnforceConfig, enforce_init as run_enforce_init

        config = EnforceConfig(
            test_command=test_command,
            project_root=project_root,
            max_iterations=max_iterations,
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
    """Evaluate the current working tree against the active enforcement session and return the full change record."""
    try:
        from agent_xray.enforce import enforce_check as run_enforce_check

        record = run_enforce_check(hypothesis=hypothesis, project_root=project_root)
        return _json_response(_serialize(record))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_plan(
    hypothesis: str,
    expected_tests: list[str] | None = None,
    project_root: str = ".",
) -> str:
    """Register a hypothesis and expected tests before making changes in an enforcement session."""
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
    """Detect uncommitted changes that have not been processed through the enforce pipeline."""
    try:
        from agent_xray.enforce import enforce_guard as run_enforce_guard

        return _json_response(run_enforce_guard(project_root=project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_status(project_root: str = ".") -> str:
    """Return the current enforcement session state, including baseline and iteration counts."""
    try:
        from agent_xray.enforce import enforce_status as run_enforce_status

        return _json_response(run_enforce_status(project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_challenge(project_root: str = ".") -> str:
    """Run an adversarial audit over unreviewed enforcement iterations and return the findings."""
    try:
        from agent_xray.enforce import enforce_challenge as run_enforce_challenge

        result = run_enforce_challenge(project_root=project_root)
        return _json_response(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_reset(project_root: str = ".") -> str:
    """Abandon the active enforcement session and delete its persisted session directory."""
    try:
        from agent_xray.enforce import enforce_reset as run_enforce_reset

        return _json_response({"success": run_enforce_reset(project_root)})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_report(project_root: str = ".", format: str = "json") -> str:
    """Generate a full enforcement report and return it in JSON, text, or markdown form."""
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
def analyze(log_dir: str, rules: str | None = None, format: str = "auto") -> str:
    """Load tasks from a trace directory, analyze them, and return per-task analysis with grade summary."""
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
def grade(log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Grade tasks from a trace directory and return the distribution plus per-task grading details."""
    try:
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)
        grades = grade_tasks(tasks, rule_set)
        distribution: dict[str, int] = {}
        for result in grades:
            distribution[result.grade] = distribution.get(result.grade, 0) + 1

        return _json_response(
            {
                "summary": {
                    "tasks": len(tasks),
                    "rules": rule_set.name,
                    "distribution": distribution,
                },
                "tasks": [_serialize(result) for result in grades],
            }
        )
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def root_cause(log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Classify likely root causes for weak or broken tasks and return the grouped distribution and per-task results."""
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
    """Check trace completeness across observability dimensions and return warnings plus coverage scores."""
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
