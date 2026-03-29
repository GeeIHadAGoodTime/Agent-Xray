"""MCP server exposing agent-xray enforce and analysis helpers as tools."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any

from mcp.server.fastmcp import FastMCP

server = FastMCP("agent-xray")


_MCP_MAX_CHARS = 30_000  # MCP responses must fit in agent context windows


def _json_response(payload: Any) -> str:
    return json.dumps(payload, indent=2)


def _compact_json(payload: Any) -> str:
    """JSON response capped for MCP context windows."""
    result = json.dumps(payload, indent=2)
    if len(result) <= _MCP_MAX_CHARS:
        return result
    # Re-serialize without indent to save space
    result = json.dumps(payload, separators=(",", ":"))
    if len(result) <= _MCP_MAX_CHARS:
        return result
    # Truncate with warning
    return result[:_MCP_MAX_CHARS] + '\n\n... TRUNCATED (use CLI for full output) ...'


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
    """Start a systematic A/B testing flywheel: plan → edit → check → auto-commit or auto-revert → repeat.

    This captures a baseline test run, then each cycle you register a hypothesis (`enforce_plan`), make one small edit, and call `enforce_check`. The engine auto-commits passing changes and auto-reverts failures — you do NOT run git commands yourself. Each cycle ends in exactly one commit or one revert before the next cycle starts.
    Prerequisites: a git repo, a deterministic test command, and the correct project root.
    Common mistakes: batching multiple changes in one cycle, doing git commits yourself (the engine handles them), or initializing after edits already exist.
    Next step: call `enforce_plan` to register your first hypothesis, make one small edit, then call `enforce_check`. Repeat this cycle for each change.
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
    """Run the A/B test for one edit: compare before/after, then auto-commit or auto-revert.

    This is the core of the flywheel. After your edit, this tool runs the test command, compares results against the previous state, audits for gaming, and makes a decision. If COMMITTED, the engine does `git commit` for you. If REVERTED, it does `git reset --hard` for you. You never touch git yourself.
    Prerequisites: an active session from `enforce_init` and exactly one small edit since the last cycle.
    Common mistakes: batching unrelated edits in one cycle, doing git commits yourself, changing tests to fake progress, or skipping `enforce_plan`.
    Next step: if COMMITTED, call `enforce_plan` for your next hypothesis (new cycle). If REVERTED, revise your approach and call `enforce_plan` again. After several cycles, call `enforce_challenge` to audit for gaming.
    """
    try:
        from agent_xray.enforce import enforce_check as run_enforce_check

        record = run_enforce_check(hypothesis=hypothesis, project_root=project_root)
        return _compact_json(_serialize(record))
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

        return _compact_json(run_enforce_diff(project_root=project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_plan(
    hypothesis: str,
    expected_tests: list[str] | None = None,
    project_root: str = ".",
) -> str:
    """Start a new cycle: register what you intend to fix and which tests should move.

    This is the first step of each flywheel cycle. Declare your hypothesis BEFORE editing any code. If resuming after a break, call `enforce_guard` first to detect unreviewed changes.
    Prerequisites: an active enforce session and a concrete, falsifiable hypothesis for one change.
    Common mistakes: vague hypotheses, editing code before calling plan, predicting unrelated tests, or skipping plan (the engine tracks prediction accuracy).
    Next step: make exactly one small edit, optionally call `enforce_diff` to preview scope, then call `enforce_check` to run the A/B test. The engine will auto-commit or auto-revert.
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

        return _compact_json(run_enforce_guard(project_root=project_root))
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

        return _compact_json(run_enforce_status(project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_challenge(project_root: str = ".") -> str:
    """Run the adversarial cross-iteration audit over the flywheel history.

    Use this after several commit/revert cycles to catch gaming, scope creep, repeated hot-file churn, and other cumulative failure modes across the full iteration sequence.
    Prerequisites: an active session with recorded iterations (at least 2-3 cycles).
    Common mistakes: relying on per-cycle checks alone or using challenge before any enforce history exists.
    Next step: if clean, call `enforce_report` for a shareable summary. If gaming detected, revert suspect iterations and re-plan.
    """
    try:
        from agent_xray.enforce import enforce_challenge as run_enforce_challenge

        result = run_enforce_challenge(project_root=project_root)
        return _compact_json(_serialize(result))
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
    """Generate the final flywheel report: all cycles, commits, reverts, gaming signals, and net progress.

    This summarizes the enforce session history, not agent traces — for trace analysis reports, use `report`.
    Use this after the flywheel has run several cycles and you want a shareable summary of what was committed, what was reverted, and why.
    Prerequisites: an existing enforce session and a valid format of `text`, `json`, or `markdown`.
    Common mistakes: calling it before initializing a session or treating the report as a substitute for per-cycle `enforce_check`.
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

        return _compact_json({"format": format, "report": rendered})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def analyze(log_dir: str, rules: str | None = None, format: str = "auto", task_bank: str | None = None) -> str:
    """Start here. Analyze agent traces to get grade distribution, root causes, and a fix plan.

    For a site-level overview, try `tree` first. For a single-call alternative that combines grade + root_cause + baseline, use `flywheel`.
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

        # MCP: return summary + worst 10 tasks (not all — full dump kills agent context)
        sorted_grades = sorted(grades, key=lambda g: g.score)
        worst_10 = sorted_grades[:10]
        payload = {
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "grade_distribution": distribution,
            },
            "worst_tasks": [
                {
                    "task_id": g.task_id,
                    "grade": g.grade,
                    "score": g.score,
                    "reasons": [r for r in (g.reasons if hasattr(g, "reasons") else [])],
                    "site": analyses[g.task_id].site_name if g.task_id in analyses else "unknown",
                    "user_text": (next((t.task_text for t in tasks if t.task_id == g.task_id), "") or "")[:80],
                }
                for g in worst_10
            ],
            "note": "Showing 10 worst tasks. Use CLI `agent-xray analyze` for full per-task output.",
        }
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def grade(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None) -> str:
    """Grade traces against a ruleset and return scored per-task details.

    Use this after `analyze` when you need a scored view of which tasks are golden, weak, or broken.
    High-value path: provide task_bank (path to task_bank.json) for expectation-aware grading. This matches each task to its bank entry and evaluates success_criteria (must_reach_url, must_answer_contains, payment_fields_visible, etc.). Without task_bank, grading uses generic signals only.
    Prerequisites: a readable trace selection and a ruleset that matches the task domain.
    Validate your task_bank first with `task_bank_validate`.
    Common mistakes: grading without a task bank (misses expectation failures), mixing incomparable runs, or assuming grades explain root cause by themselves.
    Next step: call `root_cause` on the same traces to classify why tasks are failing. Use `search_tasks` to find task_ids by keyword, then `surface_task` to inspect them.
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

        # MCP: return distribution + worst 15 tasks (not all 187)
        sorted_grades = sorted(grades, key=lambda g: g.score)
        worst = sorted_grades[:15]
        payload: dict[str, Any] = {
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "task_bank": task_bank or "none (generic grading only)",
                "distribution": distribution,
            },
            "worst_tasks": [
                {
                    "task_id": r.task_id,
                    "grade": r.grade,
                    "score": r.score,
                    "reasons": r.reasons[:5] if hasattr(r, "reasons") else [],
                }
                for r in worst
            ],
            "note": "Showing 15 worst tasks. Use CLI `agent-xray grade` for full per-task output.",
        }
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def root_cause(log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Classify weak or broken tasks into likely root causes and return grouped plus per-task results.

    Use this after `grade` when you want an evidence-backed shortlist of likely failure modes to investigate next.
    Prerequisites: trace data that grades poorly enough to classify and a ruleset appropriate for the task domain.
    Common mistakes: skipping surface inspection on critical tasks or treating the classifier as ground truth instead of a ranked heuristic.
    This classifies failure modes analytically. For a prioritized fix plan with verify commands, call `diagnose` instead.
    Next step: call `diagnose` on the same log_dir for a prioritized fix plan with investigation targets, or call `completeness` to verify data quality.
    """
    try:
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures, summarize_root_causes

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)
        grades = grade_tasks(tasks, rule_set)
        failures = classify_failures(tasks, grades)

        # MCP: cap to 20 worst failures sorted by score ascending
        sorted_failures = sorted(failures, key=lambda f: f.score if hasattr(f, "score") else 0)
        shown = sorted_failures[:20]
        payload = {
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "classified_failures": len(failures),
            },
            "distribution": summarize_root_causes(failures),
            "tasks": [_serialize(result) for result in shown],
        }
        if len(failures) > 20:
            payload["note"] = f"Showing 20 worst of {len(failures)} failures. Use CLI `agent-xray root-cause` for full output."
        return _compact_json(payload)
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


@server.tool()
def surface_task(log_dir: str, task_id: str, format: str = "auto", task_bank: str | None = None) -> str:
    """Inspect the full decision surface for a single task, showing per-step tool choices, reasoning, and context.

    Use this when you need to understand exactly what happened inside one task — what tools were available, what the model chose, and why.
    High-value path: provide task_bank (path to task_bank.json) to include matched expectations and success criteria alongside the surface.
    For reasoning chain details, call `reasoning` on the same task. To compare two specific tasks side-by-side, use `diff_tasks`.
    Next step: if the surface reveals a tool selection problem, call `root_cause` to classify it. If the surface looks correct but the grade is wrong, call `grade` with task_bank to check expectation alignment.
    """
    try:
        from agent_xray.surface import surface_for_task as run_surface

        tasks = _load_tasks(log_dir, format)
        task = None
        for t in tasks:
            if t.task_id == task_id:
                task = t
                break
        if task is None:
            return _json_response({"error": f"Task {task_id!r} not found in {log_dir}"})

        surface = run_surface(task)

        if task_bank:
            from agent_xray.contrib.task_bank import load_task_bank, match_task_to_bank

            bank = load_task_bank(task_bank)
            matched = match_task_to_bank(task, bank)
            if matched:
                surface["task_bank_entry"] = _serialize(matched)

        # MCP: strip duplicate prompt fields and quadratic conversation_history
        metadata = surface.get("metadata", {})
        metadata.pop("system_prompt_text", None)
        metadata.pop("system_context_components", None)
        for step in surface.get("steps", []):
            step.pop("conversation_history", None)
            for key in ("tool_result", "result_summary"):
                val = step.get(key)
                if isinstance(val, str) and len(val) > 500:
                    step[key] = val[:500] + "..."

        return _compact_json(surface)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def search_tasks(log_dir: str, query: str, format: str = "auto") -> str:
    """Search tasks by user_text substring to find specific task IDs for further inspection.

    Use this when you know what the user asked but not the task_id, or when filtering traces by keyword before drilling in.
    Next step: call `surface_task` on a matched task_id to inspect its full decision surface, or call `grade` to see how matched tasks scored.
    """
    try:
        from agent_xray.analyzer import analyze_task

        tasks = _load_tasks(log_dir, format)
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        for task in tasks:
            text = task.task_text or ""
            if query_lower not in text.lower():
                continue
            analysis = analyze_task(task)
            matches.append({
                "task_id": task.task_id,
                "outcome": task.outcome.status if task.outcome else "",
                "step_count": len(task.steps),
                "site": analysis.site_name,
                "user_text": text[:80],
            })

        # MCP: cap to 25 matches
        total_matches = len(matches)
        shown = matches[:25]
        payload: dict[str, Any] = {
            "query": query,
            "total_matches": total_matches,
            "shown": len(shown),
            "matches": shown,
        }
        if total_matches > 25:
            payload["note"] = "Showing first 25. Use CLI for full results."
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def diagnose(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None) -> str:
    """Classify failures and build a prioritized fix plan with investigation targets and verify commands.

    This builds on root_cause classification to produce actionable fixes. If you haven't classified failures yet, call `root_cause` first.
    Validate your task_bank first with `task_bank_validate`.
    Use this after `grade` and `root_cause` when you want an actionable ranked list of what to fix first.
    High-value path: provide task_bank (path to task_bank.json) for expectation-aware failure classification.
    Next step: for each fix-plan entry, call `surface_task` on the investigate_task to understand the failure, then apply the fix and re-run `grade` to verify improvement.
    """
    try:
        from agent_xray.diagnose import build_fix_plan
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)

        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)

        failures = classify_failures(tasks, grades)
        plan = build_fix_plan(failures, log_dir=log_dir)

        return _json_response({
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "task_bank": task_bank or "none",
                "failures_classified": len(failures),
                "fix_plan_entries": len(plan),
            },
            "fix_plan": [_serialize(entry) for entry in plan],
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def compare_runs(left_log_dir: str, right_log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Compare two trace sets side by side to find grade shifts, cost deltas, and decision divergences.

    Use this when you have a before/after pair of runs (different models, different days, or different prompt versions) and want to quantify what changed.
    To compare two specific tasks side-by-side, use `diff_tasks`.
    Next step: for each divergence point, call `surface_task` on the task_id to inspect what each run did differently. Call `diagnose` on the worse run to build a fix plan.
    """
    try:
        from agent_xray.comparison import compare_model_runs

        result = compare_model_runs(
            left_log_dir,
            right_log_dir,
            rules_path=rules if rules != "default" else None,
        )

        return _compact_json(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def report(
    log_dir: str,
    report_type: str,
    rules: str = "default",
    format: str = "auto",
    task_bank: str | None = None,
) -> str:
    """Generate a focused report by type: health, golden, broken, tools, flows, outcomes, actions, coding, research, cost, fixes, timeline, or spins.

    This analyzes agent traces, not enforce sessions — for enforce session reports, use `enforce_report`.
    Use this when you need a specific analytical view of the traces rather than the broad `analyze` overview.
    High-value path: provide task_bank (path to task_bank.json) for expectation-aware reports (especially fixes and broken). Use `pricing_show` for per-model cost lookup.
    Next step: after reviewing a report, call `surface_task` on specific task_ids that need investigation, or call `diagnose` for a prioritized fix plan.
    """
    try:
        from agent_xray.analyzer import analyze_task, analyze_tasks
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.reports import (
            report_actions_data,
            report_broken_data,
            report_coding_data,
            report_cost_data,
            report_fixes_data,
            report_flows_data,
            report_golden_data,
            report_health_data,
            report_outcomes_data,
            report_research_data,
            report_spins_data,
            report_timeline_data,
            report_tools_data,
        )

        valid_types = [
            "health", "golden", "broken", "tools", "flows", "outcomes",
            "actions", "coding", "research", "cost", "fixes", "timeline", "spins",
        ]
        if report_type not in valid_types:
            return _json_response({"error": f"Unknown report_type: {report_type!r}. Choose from: {', '.join(valid_types)}"})

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)

        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)

        analyses = analyze_tasks(tasks)

        data_funcs: dict[str, Any] = {
            "health": lambda: report_health_data(tasks, grades, analyses),
            "golden": lambda: report_golden_data(tasks, grades, analyses),
            "broken": lambda: report_broken_data(tasks, grades, analyses),
            "tools": lambda: report_tools_data(tasks, analyses),
            "flows": lambda: report_flows_data(tasks, analyses),
            "outcomes": lambda: report_outcomes_data(tasks, grades, analyses),
            "actions": lambda: report_actions_data(tasks, grades, analyses),
            "coding": lambda: report_coding_data(tasks, analyses),
            "research": lambda: report_research_data(tasks, analyses),
            "cost": lambda: report_cost_data(tasks, analyses),
            "fixes": lambda: report_fixes_data(tasks, grades, analyses),
            "timeline": lambda: report_timeline_data(tasks, grades, analyses),
            "spins": lambda: report_spins_data(tasks, analyses),
        }

        return _compact_json({
            "report_type": report_type,
            "tasks": len(tasks),
            "rules": rule_set.name,
            "data": data_funcs[report_type](),
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def diff_tasks(log_dir: str, task_id_1: str, task_id_2: str, format: str = "auto") -> str:
    """Compare two tasks side by side: tool sequences, timing, outcomes.

    Use this when two tasks attempted the same goal but diverged in behavior or grade.
    Next step: surface_task on the diverging task.
    """
    try:
        from agent_xray.surface import diff_tasks as run_diff_tasks

        tasks = _load_tasks(log_dir, format)
        left = right = None
        for t in tasks:
            if t.task_id == task_id_1:
                left = t
            if t.task_id == task_id_2:
                right = t
        if left is None:
            return _json_response({"error": f"Task {task_id_1!r} not found in {log_dir}"})
        if right is None:
            return _json_response({"error": f"Task {task_id_2!r} not found in {log_dir}"})

        result = _serialize(run_diff_tasks(left, right))

        # MCP: strip duplicate prompt fields, conversation_history, and dedup prompts across sides
        left_hash = result.get("left", {}).get("metadata", {}).get("system_prompt_hash")
        right_hash = result.get("right", {}).get("metadata", {}).get("system_prompt_hash")
        same_prompt = left_hash and right_hash and left_hash == right_hash

        for side in ("left", "right"):
            side_data = result.get(side, {})
            metadata = side_data.get("metadata", {})
            metadata.pop("system_prompt_text", None)
            metadata.pop("system_context_components", None)
            if same_prompt:
                side_data.pop("prompt_text", None)
            for step in side_data.get("steps", []):
                step.pop("conversation_history", None)
                for key in ("tool_result", "result_summary"):
                    val = step.get(key)
                    if isinstance(val, str) and len(val) > 500:
                        step[key] = val[:500] + "..."

        if same_prompt:
            result["prompt_note"] = f"Both tasks share the same prompt (hash: {left_hash}). Prompt text omitted."

        return _compact_json(result)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def reasoning(log_dir: str, task_id: str, format: str = "auto") -> str:
    """Extract the model's reasoning chain for a task, showing how it decided what to do at each step.

    Use this when you need to understand the model's internal decision-making, not just its actions.
    Next step: use findings to guide `diagnose` priorities, or call `surface_task` for the full decision surface if you started with reasoning.
    """
    try:
        from agent_xray.surface import reasoning_for_task

        tasks = _load_tasks(log_dir, format)
        task = None
        for t in tasks:
            if t.task_id == task_id:
                task = t
                break
        if task is None:
            return _json_response({"error": f"Task {task_id!r} not found in {log_dir}"})

        result = reasoning_for_task(task)
        return _compact_json(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def tree(log_dir: str, rules: str | None = None, format: str = "auto") -> str:
    """Bird's-eye view of trace organization as a day/site/task hierarchy.

    Use this to orient yourself before drilling into specific tasks or sites.
    Next step: surface_task on a specific task_id.
    """
    try:
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.surface import enriched_tree_for_tasks

        tasks = _load_tasks(log_dir, format)
        if not tasks:
            return _json_response({"tree": {}, "task_count": 0})

        grades = None
        if rules:
            rule_set = load_rules(rules)
            grades = grade_tasks(tasks, rule_set)

        enriched = enriched_tree_for_tasks(tasks, grades)

        # MCP: collapse per-task lists into site-level counts + outcome summary
        compact_tree: dict[str, dict[str, dict[str, Any]]] = {}
        for day, sites in enriched.items():
            compact_tree[day] = {}
            for site, task_list in sites.items():
                outcomes: dict[str, int] = {}
                for t_info in task_list:
                    outcome = t_info.get("outcome") or t_info.get("grade") or "unknown"
                    outcomes[outcome] = outcomes.get(outcome, 0) + 1
                compact_tree[day][site] = {
                    "count": len(task_list),
                    **outcomes,
                }
        return _compact_json({
            "task_count": len(tasks),
            "rules": rules or "none",
            "tree": compact_tree,
            "note": "Sites collapsed to counts. Use CLI `agent-xray tree` for per-task details.",
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_rank(
    log_dir: str,
    rules: str | None = None,
    optimize: str = "balanced",
    format: str = "auto",
) -> str:
    """Rank best runs by efficiency, grouping by site with configurable optimization profile.

    Use this to find exemplar runs that represent ideal agent behavior.
    Next step: golden_compare to check for regressions.
    """
    try:
        from agent_xray.golden import rank_golden_runs
        from agent_xray.grader import load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        rankings = rank_golden_runs(tasks, rules=rule_set, optimize=optimize)

        payload: dict[str, Any] = {
            "summary": {
                "tasks": len(tasks),
                "optimize": optimize,
                "sites_ranked": len(rankings),
            },
            "rankings": {
                site: [_serialize(r) for r in ranks]
                for site, ranks in rankings.items()
            },
        }
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_compare(
    log_dir: str,
    fixtures_dir: str,
    rules: str | None = None,
    format: str = "auto",
) -> str:
    """Regression detection against golden captures. Compares current runs to fixture baselines.

    Create fixtures using `capture_task`.
    Use this after golden_rank to verify that agent quality has not degraded.
    Next step: surface_task on any REGRESSION task.
    """
    try:
        from pathlib import Path

        from agent_xray.golden import rank_golden_runs
        from agent_xray.grader import load_rules
        from agent_xray.replay import load_fixture

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        rankings = rank_golden_runs(tasks, rules=rule_set, optimize="balanced")

        fixtures_path = Path(fixtures_dir)
        if not fixtures_path.exists():
            return _json_response({"error": f"Fixtures directory not found: {fixtures_dir}"})

        results: list[dict[str, Any]] = []
        for fixture_path in sorted(fixtures_path.glob("*.json")):
            try:
                fixture = load_fixture(fixture_path)
            except Exception:
                continue
            fixture_site = str(fixture.get("site", ""))
            fixture_steps = int(fixture.get("total_steps", 0) or 0)
            site_ranks = rankings.get(fixture_site, [])
            if site_ranks:
                best = site_ranks[0]
                step_delta = best.step_count - fixture_steps
                results.append({
                    "fixture": fixture_path.name,
                    "site": fixture_site,
                    "fixture_steps": fixture_steps,
                    "current_best_task": best.task_id,
                    "current_best_steps": best.step_count,
                    "current_best_efficiency": round(best.efficiency, 4),
                    "step_delta": step_delta,
                    "verdict": (
                        "IMPROVED" if step_delta < -2
                        else "REGRESSION" if step_delta > 5
                        else "STABLE"
                    ),
                })
            else:
                results.append({
                    "fixture": fixture_path.name,
                    "site": fixture_site,
                    "fixture_steps": fixture_steps,
                    "current_best_task": None,
                    "current_best_steps": None,
                    "current_best_efficiency": None,
                    "step_delta": None,
                    "verdict": "UNMATCHED",
                })

        return _compact_json({
            "summary": {
                "tasks": len(tasks),
                "fixtures_compared": len(results),
                "regressions": sum(1 for r in results if r["verdict"] == "REGRESSION"),
                "improvements": sum(1 for r in results if r["verdict"] == "IMPROVED"),
            },
            "comparisons": results,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def task_bank_validate(path: str) -> str:
    """Check task bank schema and criteria for correctness.

    Use this before grading with a task bank to catch schema errors early.
    Next step: call `grade` with the `task_bank` parameter set to this file's path.
    """
    try:
        from agent_xray.contrib.task_bank import validate_task_bank

        result = validate_task_bank(path)
        return _json_response({
            "errors": result.errors,
            "warnings": result.warnings,
            "valid": len(result.errors) == 0,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def task_bank_list(path: str) -> str:
    """List all entries in a task bank, showing task IDs, descriptions, and success criteria.

    Use this to inspect what a task bank contains before using it for grading.
    Next step: call `grade` with the `task_bank` parameter set to this file's path.
    """
    try:
        from agent_xray.contrib.task_bank import load_task_bank

        entries = load_task_bank(path)

        # MCP: cap to 30 entries, show category summary, truncate criteria to keys only
        categories: dict[str, int] = {}
        for entry in entries:
            cat = entry.get("category", "uncategorized") if isinstance(entry, dict) else "uncategorized"
            categories[cat] = categories.get(cat, 0) + 1

        shown_entries = []
        for entry in entries[:30]:
            if isinstance(entry, dict):
                compact = dict(entry)
                criteria = compact.get("success_criteria", {})
                if isinstance(criteria, dict):
                    compact["success_criteria"] = list(criteria.keys())
                shown_entries.append(compact)
            else:
                shown_entries.append(entry)

        payload: dict[str, Any] = {
            "total_entries": len(entries),
            "categories": categories,
            "shown": len(shown_entries),
            "entries": shown_entries,
        }
        if len(entries) > 30:
            payload["note"] = f"Showing 30 of {len(entries)} entries. Use CLI for full output."
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def flywheel(log_dir: str, rules: str | None = None, format: str = "auto") -> str:
    """Full quality loop in one call: grade + root cause + baseline comparison.

    Use this for an end-to-end quality assessment when you want everything at once.
    Next step: diagnose for fix plan.
    """
    try:
        from agent_xray.flywheel import run_flywheel

        result = run_flywheel(
            log_dir,
            rules_path=rules,
        )
        payload = _serialize(result.to_dict())
        # MCP: remove per-task maps (available via grade tool if needed)
        payload.pop("task_grades", None)
        payload.pop("task_scores", None)
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def capture_task(log_dir: str, task_id: str, format: str = "auto") -> str:
    """Save a task as a sanitized fixture for replay and regression testing.

    Use this to capture a golden or interesting task for future comparison.
    Next step: pass the captured fixture directory to `golden_compare` as `fixtures_dir` to detect regressions against future runs.
    """
    try:
        from pathlib import Path

        from agent_xray.capture import capture_task as run_capture

        tasks = _load_tasks(log_dir, format)
        output_path = Path.cwd() / "captured" / f"{task_id}.json"
        path = run_capture(tasks, task_id, output_path)
        return _json_response({
            "fixture": str(path),
            "task_id": task_id,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def pricing_show(model_name: str) -> str:
    """Look up per-token pricing for a model, showing input/output/cached costs.

    Use this to understand cost implications before analyzing spend.
    Next step: report cost for full cost analysis.
    """
    try:
        from agent_xray.pricing import format_model_pricing, load_pricing

        pricing_data = load_pricing()
        formatted = format_model_pricing(model_name, pricing_data)
        return _json_response({
            "model": model_name,
            "pricing": formatted,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


def main() -> None:
    """Run the agent-xray MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
