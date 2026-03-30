"""MCP server exposing agent-xray enforce and analysis helpers as tools."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

server = FastMCP("agent-xray")


_MCP_MAX_CHARS = 20_000  # MCP responses must fit in agent context windows
_MCP_SEARCH_MATCH_LIMIT = 25
_MCP_TEST_OUTPUT_MAX_CHARS = 500
_GRADE_DEPENDENT_REPORT_TYPES = {
    "health",
    "golden",
    "broken",
    "outcomes",
    "actions",
    "fixes",
    "timeline",
}
_REPORT_TYPES = [
    "health",
    "golden",
    "broken",
    "tools",
    "flows",
    "outcomes",
    "actions",
    "coding",
    "research",
    "cost",
    "fixes",
    "timeline",
    "spins",
]


def _json_response(payload: Any) -> str:
    return json.dumps(payload, separators=(",", ":"))


def _compact_json(payload: Any) -> str:
    """JSON response capped for MCP context windows.

    Always compact (no indent). On overflow, returns a valid JSON envelope
    with a truncation notice instead of slicing mid-object.
    """
    result = json.dumps(payload, separators=(",", ":"))
    if len(result) <= _MCP_MAX_CHARS:
        return result
    # Return valid JSON with truncation metadata — never slice mid-object
    return json.dumps({
        "truncated": True,
        "original_chars": len(result),
        "note": "Response too large for MCP. Use CLI for full output.",
    }, separators=(",", ":"))


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


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _truncate_test_result_outputs(value: Any) -> Any:
    if isinstance(value, dict):
        serialized = {str(key): _truncate_test_result_outputs(item) for key, item in value.items()}
        if (
            "output" in serialized
            and {"passed", "failed", "errors", "total"}.issubset(serialized)
            and isinstance(serialized["output"], str)
        ):
            serialized["output"] = _truncate_text(
                serialized["output"], _MCP_TEST_OUTPUT_MAX_CHARS,
            )
        return serialized
    if isinstance(value, list):
        return [_truncate_test_result_outputs(item) for item in value]
    return value


def _extract_site(task: Any) -> str:
    for step in reversed(getattr(task, "steps", [])):
        page_url = getattr(step, "page_url", None)
        if not page_url:
            browser = getattr(step, "browser", None)
            if isinstance(browser, dict):
                page_url = browser.get("page_url")
            else:
                page_url = getattr(browser, "page_url", None)
        if not page_url:
            continue
        host = urlparse(str(page_url)).netloc
        if host:
            return host
    return "unknown"


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
    scope: list[str] | None = None,
    test_timeout: int = 300,
) -> str:
    """Start a systematic A/B testing flywheel: plan -> edit -> check -> review evidence -> commit or revert -> repeat."""
    try:
        from agent_xray.enforce import EnforceConfig, enforce_init as run_enforce_init

        config = EnforceConfig(
            test_command=test_command,
            project_root=project_root,
            max_iterations=max_iterations,
            stash_first=stash_first,
            max_files_per_change=max_files_per_change,
            max_diff_lines=max_diff_lines,
            scope=scope,
            test_timeout=test_timeout,
        )
        baseline, session_dir = run_enforce_init(config)
        return _compact_json(
            {
                "baseline": _serialize(baseline),
                "session_dir": str(session_dir),
            }
        )
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_check(hypothesis: str = "", project_root: str = ".") -> str:
    """Run the A/B test for one edit: compare before/after tests, audit for gaming, and surface the full evidence with a recommendation."""
    try:
        from agent_xray.enforce import enforce_check as run_enforce_check

        record = run_enforce_check(hypothesis=hypothesis, project_root=project_root)
        return _compact_json(_truncate_test_result_outputs(_serialize(record)))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_diff(project_root: str = ".") -> str:
    """Preview the current diff and whether enforce would reject it for scope before you spend a test run."""
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
    """Start a new cycle: register what you intend to fix and which tests should move."""
    try:
        from agent_xray.enforce import enforce_plan as run_enforce_plan

        return _compact_json(
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
    """Check whether there are unreviewed working-tree changes outside the enforce loop."""
    try:
        from agent_xray.enforce import enforce_guard as run_enforce_guard

        return _compact_json(run_enforce_guard(project_root=project_root))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_status(project_root: str = ".") -> str:
    """Read the current enforce session state, including baseline, counts, and pending history."""
    try:
        from agent_xray.enforce import enforce_status as run_enforce_status

        return _compact_json(_truncate_test_result_outputs(run_enforce_status(project_root)))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_challenge(project_root: str = ".") -> str:
    """Run the adversarial cross-iteration audit over the flywheel history and surface cumulative patterns for review."""
    try:
        from agent_xray.enforce import enforce_challenge as run_enforce_challenge

        result = run_enforce_challenge(project_root=project_root)
        return _compact_json(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_reset(project_root: str = ".") -> str:
    """Abandon the current enforce session and remove its persisted session state."""
    try:
        from agent_xray.enforce import enforce_reset as run_enforce_reset

        return _json_response({"success": run_enforce_reset(project_root)})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce_report(project_root: str = ".", format: str = "json") -> str:
    """Generate the final flywheel report: all cycles, evidence, recommendations, gaming signals, and net progress."""
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
    """Analyze agent traces to get grade distribution, root causes, and a fix plan."""
    try:
        from agent_xray.analyzer import analyze_task
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)
        distribution: dict[str, int] = {}
        for grade in grades:
            distribution[grade.grade] = distribution.get(grade.grade, 0) + 1

        # MCP: return summary + worst 10 tasks (not all — full dump kills agent context)
        sorted_grades = sorted(grades, key=lambda g: g.score)
        worst_10 = sorted_grades[:10]

        # Lazy-analyze only worst 10 tasks (not all 187) — saves O(N_total) work
        worst_ids = {g.task_id for g in worst_10}
        task_map = {t.task_id: t for t in tasks}
        analyses = {tid: analyze_task(task_map[tid]) for tid in worst_ids if tid in task_map}

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
                    "user_text": (task_map.get(g.task_id).task_text or "")[:80] if g.task_id in task_map else "",
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
    """Grade traces against a ruleset and return scored per-task details."""
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
    """Classify weak or broken tasks into root cause categories with per-task evidence. Use to understand WHY tasks fail before hypothesizing a fix."""
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
    """Measure how complete the trace instrumentation is across key observability dimensions."""
    try:
        from agent_xray.completeness import check_completeness

        tasks = _load_tasks(log_dir, format)
        report = check_completeness(tasks)
        return _compact_json(
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
    """Replay a task step-by-step: exact tools available, inputs, results, and model reasoning at each step. Use BEFORE fixing — see exactly what the agent saw when it failed."""
    try:
        from agent_xray.surface import surface_for_task as run_surface

        tasks = _load_tasks(log_dir, format)
        task_map = {t.task_id: t for t in tasks}
        task = task_map.get(task_id)
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
    """Search tasks by user_text substring to find specific task IDs for further inspection."""
    try:
        tasks = _load_tasks(log_dir, format)
        query_lower = query.lower()
        matches: list[dict[str, Any]] = []
        truncated = False
        for task in tasks:
            text = task.task_text or ""
            if query_lower not in text.lower():
                continue
            matches.append({
                "task_id": task.task_id,
                "outcome": task.outcome.status if task.outcome else "",
                "step_count": len(task.steps),
                "site": _extract_site(task),
                "user_text": text[:80],
            })
            if len(matches) >= _MCP_SEARCH_MATCH_LIMIT:
                truncated = True
                break

        # MCP: stop after 25 matches to avoid scanning/analyzing large match sets
        total_matches = len(matches)
        payload: dict[str, Any] = {
            "query": query,
            "total_matches": total_matches,
            "shown": total_matches,
            "matches": matches,
        }
        if truncated:
            payload["note"] = (
                "Stopped after 25 matches for MCP efficiency. "
                "Use CLI for full results."
            )
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def diagnose(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None) -> str:
    """Classify failures and build a prioritized fix plan with investigation targets. Use to decide WHAT to fix before starting an enforce cycle."""
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

        return _compact_json({
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
    """Compare two trace sets side by side to find grade shifts, cost deltas, and decision divergences."""
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
    """Generate a focused report by type: health, golden, broken, tools, flows, outcomes, actions, coding, research, cost, fixes, timeline, or spins."""
    try:
        from agent_xray.analyzer import analyze_tasks
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

        if report_type not in _REPORT_TYPES:
            return _json_response({"error": f"Unknown report_type: {report_type!r}. Choose from: {', '.join(_REPORT_TYPES)}"})

        tasks = _load_tasks(log_dir, format)
        grades: list[Any] = []
        rule_name = rules
        if report_type in _GRADE_DEPENDENT_REPORT_TYPES:
            rule_set = load_rules(rules)
            rule_name = rule_set.name
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
            "rules": rule_name,
            "data": data_funcs[report_type](),
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def diff_tasks(log_dir: str, task_id_1: str, task_id_2: str, format: str = "auto") -> str:
    """Compare two tasks side by side: tool sequences, timing, outcomes. Use to see what a successful task did differently from a failed one."""
    try:
        from agent_xray.surface import diff_tasks as run_diff_tasks

        tasks = _load_tasks(log_dir, format)
        task_map = {t.task_id: t for t in tasks}
        left = task_map.get(task_id_1)
        right = task_map.get(task_id_2)
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
    """Extract the model's reasoning chain for a task — what it thought, why it chose each tool. Lighter than surface_task; use when you only need reasoning, not full tool I/O."""
    try:
        from agent_xray.surface import reasoning_for_task

        tasks = _load_tasks(log_dir, format)
        task_map = {t.task_id: t for t in tasks}
        task = task_map.get(task_id)
        if task is None:
            return _json_response({"error": f"Task {task_id!r} not found in {log_dir}"})

        result = reasoning_for_task(task)
        return _compact_json(_serialize(result))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def tree(log_dir: str, rules: str | None = None, format: str = "auto") -> str:
    """Bird's-eye view: day/site/task hierarchy with pass/fail counts. Use first to see which sites are failing before drilling into specific tasks."""
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
                    "sample_task_ids": [
                        str(t_info.get("task_id", ""))
                        for t_info in task_list[:3]
                        if t_info.get("task_id")
                    ],
                    **outcomes,
                }
        return _compact_json({
            "task_count": len(tasks),
            "rules": rules or "none",
            "tree": compact_tree,
            "note": "Sites collapsed to counts with up to 3 sample_task_ids. Use CLI `agent-xray tree` for per-task details.",
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
    """Rank best runs by efficiency, grouping by site with configurable optimization profile."""
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
    """Regression detection: compare current runs against golden fixtures. Use to see exactly where a broken run diverged from the known-good path."""
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
        skipped: list[str] = []
        for fixture_path in sorted(fixtures_path.glob("*.json")):
            try:
                fixture = load_fixture(fixture_path)
            except Exception as exc:
                skipped.append(f"{fixture_path.name}: {exc}")
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

        summary: dict[str, Any] = {
            "tasks": len(tasks),
            "fixtures_compared": len(results),
            "regressions": sum(1 for r in results if r["verdict"] == "REGRESSION"),
            "improvements": sum(1 for r in results if r["verdict"] == "IMPROVED"),
        }
        if skipped:
            summary["fixtures_skipped"] = len(skipped)
            summary["skip_errors"] = skipped[:10]
        return _compact_json({
            "summary": summary,
            "comparisons": results,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def task_bank_validate(path: str) -> str:
    """Check task bank schema and criteria for correctness."""
    try:
        from agent_xray.contrib.task_bank import validate_task_bank

        result = validate_task_bank(path)
        return _compact_json({
            "errors": result.errors[:50],
            "warnings": result.warnings[:50],
            "valid": len(result.errors) == 0,
            "total_errors": len(result.errors),
            "total_warnings": len(result.warnings),
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def task_bank_list(path: str) -> str:
    """List all entries in a task bank, showing task IDs, descriptions, and success criteria."""
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
    """Full quality loop in one call: grade + root cause + baseline comparison."""
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
    """Save a task as a sanitized fixture for replay and regression testing."""
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
    """Look up per-token pricing for a model, showing input/output/cached costs."""
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
