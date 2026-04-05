"""MCP server exposing agent-xray enforce and analysis helpers as tools."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP

from .dedup import _dedupe_tasks

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
    "overhead",
    "prompt-impact",
    "compare",
]
_STRUCTURAL_GRADE_NOTE = (
    "Grades measure execution structure (tool diversity, loop resistance, error "
    "rate, completion). They do NOT verify output correctness — a GOLDEN task "
    "can still have the wrong answer. Default rules are generic starting points; "
    "create product-specific rulesets (see rules_init) before running optimization "
    "campaigns, otherwise you will optimize for structure instead of your product's "
    "actual quality definition. Use surface_task or inspect_task to verify behavior."
)


def _default_suggestion(message: str) -> str:
    lowered = message.lower()
    if "not found" in lowered:
        return "Check the identifier or path and retry."
    if "grade_filter" in lowered:
        return "Retry without grade_filter or use one of: BROKEN, WEAK, OK, GOOD, GOLDEN."
    if "rules" in lowered:
        return "Verify the rules path or choose a valid built-in ruleset."
    if "baseline" in lowered:
        return "Capture or point to a valid baseline directory, then retry."
    if "day1" in lowered or "day2" in lowered or "compare report" in lowered:
        return "Provide both day1 and day2 in YYYYMMDD format."
    return "Check the input parameters and retry."


def _normalize_error_payload(payload: Any) -> Any:
    if isinstance(payload, dict) and "error" in payload:
        error = str(payload.get("error", ""))
        suggestion = payload.get("suggestion")
        if suggestion is None:
            suggestion = payload.get("hint")
        suggestion_text = str(suggestion).strip() if suggestion is not None else ""
        return {
            "error": error,
            "suggestion": suggestion_text or _default_suggestion(error),
        }
    return payload


def _json_response(payload: Any) -> str:
    return json.dumps(_normalize_error_payload(payload), separators=(",", ":"))


def _compact_json(payload: Any) -> str:
    """JSON response capped for MCP context windows.

    Always compact (no indent). On overflow, returns a valid JSON envelope
    with a truncation notice instead of slicing mid-object.
    """
    payload = _normalize_error_payload(payload)
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


def _with_grade_note(payload: dict[str, Any]) -> dict[str, Any]:
    payload.setdefault("grade_note", _STRUCTURAL_GRADE_NOTE)
    return payload


def _truncate_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 3:
        return value[:max_chars]
    return value[: max_chars - 3] + "..."


def _truncate_test_result_outputs(value: Any) -> Any:
    if isinstance(value, dict):
        serialized = {str(key): _truncate_test_result_outputs(item) for key, item in value.items()}
        if "output" in serialized and isinstance(serialized["output"], str):
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


# ---------------------------------------------------------------------------
# Task loading with mtime-based cache (MCP server stays alive across calls)
# ---------------------------------------------------------------------------
_task_cache: dict[tuple[str, str, int | None], tuple[float, list[Any]]] = {}
_dir_mtime_cache: dict[str, tuple[float, float]] = {}  # log_dir -> (dir_mtime, max_file_mtime)
_CACHE_MAX_ENTRIES = 8


def _dir_mtime(log_dir: str) -> float:
    """Fast mtime check: max mtime of .jsonl and .json trace files, with dir-level short-circuit."""
    from pathlib import Path
    root = Path(log_dir)
    if root.is_file():
        return root.stat().st_mtime
    # Short-circuit: if the directory mtime hasn't changed, skip per-file glob
    try:
        dir_mt = root.stat().st_mtime
    except OSError:
        return 0.0
    prev = _dir_mtime_cache.get(log_dir)
    if prev and prev[0] == dir_mt:
        return prev[1]
    # Directory changed — rescan individual file mtimes (both .jsonl and .json)
    mtimes = [f.stat().st_mtime for f in root.glob("*.jsonl")]
    mtimes.extend(f.stat().st_mtime for f in root.glob("*.json"))
    max_mt = max(mtimes) if mtimes else 0.0
    _dir_mtime_cache[log_dir] = (dir_mt, max_mt)
    return max_mt


def _load_tasks(
    log_dir: str,
    format_name: str,
    *,
    days: int | None = None,
    site: str | None = None,
    outcome: str | None = None,
    dedupe: bool = True,
) -> list[Any]:
    from agent_xray.analyzer import load_adapted_tasks, load_tasks

    # Cache key: (log_dir, format, days) — site/outcome are post-filters
    cache_key = (log_dir, format_name, days)
    current_mtime = _dir_mtime(log_dir)

    cached = _task_cache.get(cache_key)
    if cached and cached[0] >= current_mtime:
        tasks = list(cached[1])  # shallow copy to avoid mutation
    else:
        if format_name != "auto":
            tasks = load_adapted_tasks(log_dir, format=format_name, days=days, dedup=False)
        else:
            tasks = load_tasks(log_dir, days=days, dedup=False)
            if not tasks:
                tasks = load_adapted_tasks(log_dir, format="auto", days=days, dedup=False)

        # Store in cache (evict oldest if full)
        if len(_task_cache) >= _CACHE_MAX_ENTRIES:
            oldest_key = next(iter(_task_cache))
            del _task_cache[oldest_key]
        _task_cache[cache_key] = (current_mtime, tasks)

    # Post-filters (not cached — applied fresh each call)
    if site:
        site_lower = site.lower()
        tasks = [t for t in tasks if site_lower in _extract_site(t).lower()]
    if outcome:
        outcome_lower = outcome.lower()
        tasks = [t for t in tasks if t.outcome is not None and outcome_lower in t.outcome.status.lower()]
    if dedupe:
        tasks = _dedupe_tasks(tasks)
    return tasks


def _filter_by_grade(tasks: list[Any], grades: list[Any], grade_filter: str | None) -> tuple[list[Any], list[Any]]:
    """Filter tasks and their corresponding grades by grade value.

    Applied AFTER the caller grades with its own rules/task_bank, so the filter
    matches the actual grades in the output (not a stale default-rules pre-filter).
    Returns (filtered_tasks, filtered_grades).
    """
    if not grade_filter:
        return tasks, grades
    grade_upper = grade_filter.strip().upper()
    _VALID_GRADES = {"BROKEN", "WEAK", "OK", "GOOD", "GOLDEN"}
    if grade_upper not in _VALID_GRADES:
        return [], []  # Callers handle empty with clear error messages
    grade_map = {g.task_id: g for g in grades}
    filtered_tasks = []
    filtered_grades = []
    for t in tasks:
        g = grade_map.get(t.task_id)
        if g and g.grade.upper() == grade_upper:
            filtered_tasks.append(t)
            filtered_grades.append(g)
    return filtered_tasks, filtered_grades


def _resolve_task(tasks: list[Any], task_id: str) -> Any:
    """Find a task by exact or prefix match, or raise."""
    task_map = {t.task_id: t for t in tasks}
    task = task_map.get(task_id)
    if task is None:
        # Prefix match
        prefix = [t for t in tasks if t.task_id.startswith(task_id)]
        if len(prefix) == 1:
            task = prefix[0]
    if task is None:
        raise ValueError(f"Task {task_id!r} not found")
    return task


@server.tool()
def triage(log_dir: str, format: str = "auto", days: int | None = None, site: str | None = None, outcome: str | None = None, grade_filter: str | None = None, dedupe: bool = True) -> str:
    """START HERE — one-call investigation: grades all tasks, surfaces the worst failure step-by-step, and returns a prioritized fix plan (filterable by days/site/outcome/grade_filter)."""
    try:
        from agent_xray.diagnose import build_fix_plan
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures
        from agent_xray.surface import surface_for_task

        tasks = _load_tasks(log_dir, format, days=days, site=site, outcome=outcome, dedupe=dedupe)
        if not tasks:
            return _json_response({"error": "No tasks found", "hint": "Check log_dir path and days filter. Note: outcome filters by task status (failed/success), grade_filter filters by xray grade (BROKEN/WEAK/OK/GOOD/GOLDEN)."})

        rules = load_rules()
        grades = grade_tasks(tasks, rules)
        tasks, grades = _filter_by_grade(tasks, grades, grade_filter)
        if not tasks:
            return _json_response({"error": "No tasks match grade_filter", "hint": f"No tasks graded as {grade_filter!r}. Try without grade_filter to see all grades."})

        # Grade distribution
        dist: dict[str, int] = {}
        for g in grades:
            dist[g.grade] = dist.get(g.grade, 0) + 1

        # Find the single worst task
        sorted_grades = sorted(grades, key=lambda g: g.score)
        worst = sorted_grades[0] if sorted_grades else None
        task_map = {t.task_id: t for t in tasks}

        # Root causes from failures
        failure_grades = [g for g in grades if g.grade in ("BROKEN", "WEAK")]
        rc_results = classify_failures(
            [task_map[g.task_id] for g in failure_grades if g.task_id in task_map],
            failure_grades,
        ) if failure_grades else []

        # Fix plan
        fix_plan = build_fix_plan(rc_results) if rc_results else []

        # Surface the worst task (compact)
        worst_surface = None
        if worst and worst.task_id in task_map:
            surface = surface_for_task(task_map[worst.task_id])
            steps = surface.get("steps", [])
            # Ultra-compact: just tool sequence + errors
            compact_steps = []
            for s in steps:
                entry: dict[str, Any] = {"tool": s.get("tool_name", ""), "step": s.get("step", 0)}
                if s.get("error"):
                    entry["error"] = str(s["error"])[:200]
                result = s.get("tool_result", "")
                if isinstance(result, str) and len(result) > 100:
                    entry["result_preview"] = result[:100] + "..."
                elif result:
                    entry["result_preview"] = str(result)[:100]
                compact_steps.append(entry)
            worst_surface = {
                "task_id": worst.task_id,
                "grade": worst.grade,
                "score": worst.score,
                "user_text": (task_map[worst.task_id].task_text or "")[:120],
                "steps": compact_steps,
            }

        # Context-sensitive tool suggestions based on what triage found
        suggestions: list[str] = []
        golden_count = dist.get("GOLDEN", 0)
        broken_count = dist.get("BROKEN", 0)
        golden_candidates = sorted(
            (g for g in grades if g.grade == "GOLDEN"),
            key=lambda g: g.score,
            reverse=True,
        )
        broken_candidates = sorted(
            (g for g in grades if g.grade == "BROKEN"),
            key=lambda g: g.score,
        )
        if worst:
            suggestions.append(f"inspect_task(log_dir='{log_dir}', task_id='{worst.task_id}') — full deep-dive on worst task")
        if golden_count > 0 and broken_count > 0 and worst:
            golden_ids = [g.task_id for g in grades if g.grade == "GOLDEN"]
            suggestions.append(f"diff_tasks(log_dir='{log_dir}', task_id_a='{golden_ids[0]}', task_id_b='{worst.task_id}') — compare GOLDEN vs BROKEN to learn what works")
        if golden_count >= 3:
            suggestions.append(f"golden_rank(log_dir='{log_dir}') — rank {golden_count} GOLDENs by efficiency to find best exemplar")
        if worst:
            suggestions.append(f"signal_detect(log_dir='{log_dir}', task_id='{worst.task_id}') — domain-specific signals (commerce, research, coding)")
        suggestions.append("compare_runs(left_log_dir='<before>', right_log_dir='<after>') — ALWAYS run after a fix to verify improvement across ALL tasks")
        if len(tasks) > 20:
            suggestions.append(f"tree(log_dir='{log_dir}') — hierarchical view of all {len(tasks)} tasks by day/site/grade")

        golden_candidates = sorted(
            (g for g in grades if g.grade == "GOLDEN"),
            key=lambda g: (-g.score, g.task_id),
        )
        broken_candidates = sorted(
            (g for g in grades if g.grade == "BROKEN"),
            key=lambda g: (g.score, g.task_id),
        )
        log_dir_literal = repr(str(log_dir))
        suggestions = []
        if worst:
            suggestions.append(
                "surface_task("
                f"log_dir={log_dir_literal}, "
                f"task_id={repr(worst.task_id)})"
                " - Full step-by-step replay showing what the agent saw at each step"
            )
            suggestions.append(
                "reasoning("
                f"log_dir={log_dir_literal}, "
                f"task_id={repr(worst.task_id)})"
                " - LLM reasoning chain: why the model made each decision"
            )
            suggestions.append(
                f"inspect_task(log_dir={log_dir_literal}, task_id={repr(worst.task_id)})"
            )
        if golden_candidates and broken_candidates:
            suggestions.append(
                "diff_tasks("
                f"log_dir={log_dir_literal}, "
                f"task_id_a={repr(golden_candidates[0].task_id)}, "
                f"task_id_b={repr(broken_candidates[0].task_id)})"
            )
        if golden_count >= 3:
            suggestions.append(f"golden_rank(log_dir={log_dir_literal})")
        if worst:
            suggestions.append(
                f"signal_detect(log_dir={log_dir_literal}, task_id={repr(worst.task_id)})"
            )
        suggestions.append(
            "compare_runs("
            f"left_log_dir={log_dir_literal}, "
            "right_log_dir='<second_log_dir>') - provide the post-fix trace directory to verify improvement"
        )
        if len(tasks) >= 20:
            suggestions.append(f"tree(log_dir={log_dir_literal})")

        payload: dict[str, Any] = {
            "summary": {
                "tasks": len(tasks),
                "grade_distribution": dist,
                "broken_count": broken_count,
                "golden_count": golden_count,
            },
            "worst_task": worst_surface,
            "fix_plan": [
                {"root_cause": fp.root_cause, "priority": fp.priority, "targets": fp.targets, "hint": fp.fix_hint, "task_id": fp.investigate_task}
                for fp in fix_plan[:5]
            ] if fix_plan else [],
            "suggested_next_tools": suggestions,
            "suggested_next": suggestions[0] if suggestions else None,
        }
        return _compact_json(_with_grade_note(payload))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def enforce(verb: str = "auto", hypothesis: str = "", project_root: str = ".", format: str = "json") -> str:
    """Run the enforce discipline workflow.

    Most users should just call enforce() with default verb='auto'.
    """
    try:
        import inspect

        import agent_xray.enforce as enforce_module
        from agent_xray.enforce_report import (
            format_enforce_json,
            format_enforce_markdown,
            format_enforce_text,
        )

        def _pretty_json(payload: Any) -> str:
            return json.dumps(_normalize_error_payload(payload), indent=2)

        def _invoke_simple(func: Any, **kwargs: Any) -> Any:
            signature = inspect.signature(func)
            parameters = signature.parameters
            accepts_varkw = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters.values()
            )
            call_kwargs = {
                key: value
                for key, value in kwargs.items()
                if accepts_varkw or key in parameters
            }
            missing_required = [
                name
                for name, parameter in parameters.items()
                if parameter.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
                and parameter.default is inspect.Parameter.empty
                and name not in call_kwargs
            ]
            if missing_required:
                raise ValueError(
                    f"{func.__name__} is not available through the simplified MCP enforce tool; "
                    f"underlying function requires: {', '.join(missing_required)}"
                )
            return func(**call_kwargs)

        verb_key = (verb or "auto").strip().lower()
        payload: Any

        if verb_key == "auto":
            payload = _serialize(
                _invoke_simple(
                    enforce_module.enforce_auto,
                    hypothesis=hypothesis,
                    project_root=project_root,
                )
            )
        elif verb_key == "init":
            payload = _serialize(
                _invoke_simple(
                    enforce_module.enforce_init,
                    hypothesis=hypothesis,
                    project_root=project_root,
                )
            )
        elif verb_key == "check":
            payload = _truncate_test_result_outputs(
                _serialize(
                    enforce_module.enforce_check(
                        hypothesis=hypothesis,
                        project_root=project_root,
                    )
                )
            )
        elif verb_key == "plan":
            payload = _serialize(
                enforce_module.enforce_plan(
                    hypothesis=hypothesis,
                    project_root=project_root,
                )
            )
        elif verb_key == "diff":
            payload = _serialize(enforce_module.enforce_diff(project_root=project_root))
        elif verb_key == "guard":
            payload = _serialize(enforce_module.enforce_guard(project_root=project_root))
        elif verb_key == "status":
            payload = _truncate_test_result_outputs(
                _serialize(enforce_module.enforce_status(project_root))
            )
        elif verb_key == "challenge":
            payload = _serialize(enforce_module.enforce_challenge(project_root=project_root))
        elif verb_key == "reset":
            payload = {"success": enforce_module.enforce_reset(project_root)}
        elif verb_key == "report":
            report = enforce_module.build_enforce_report(project_root)
            if format == "text":
                rendered: Any = format_enforce_text(report, color=False)
            elif format == "markdown":
                rendered = format_enforce_markdown(report)
            elif format == "json":
                rendered = json.loads(format_enforce_json(report))
            else:
                raise ValueError("format must be one of: text, json, markdown")
            payload = {"format": format, "report": rendered}
        else:
            raise ValueError(
                "verb must be one of: auto, init, check, plan, diff, guard, status, "
                "challenge, reset, report"
            )

        return _pretty_json(payload)
    except Exception as e:
        return json.dumps(_normalize_error_payload({"error": str(e)}), indent=2)


@server.tool()
def enforce_quick(
    test_command: str,
    hypothesis: str = "",
    log_dir: str | None = None,
    max_files: int = 10,
    max_diff_lines: int = 500,
) -> str:
    """Run enforce init+check in one call and return the combined review payload."""
    try:
        import agent_xray.enforce as enforce_module

        payload = _truncate_test_result_outputs(
            _serialize(
                enforce_module.enforce_quick(
                    test_command=test_command,
                    hypothesis=hypothesis,
                    log_dir=log_dir,
                    max_files=max_files,
                    max_diff_lines=max_diff_lines,
                    project_root=".",
                )
            )
        )
        return json.dumps(payload, indent=2)
    except Exception as e:
        return json.dumps(_normalize_error_payload({"error": str(e)}), indent=2)


@server.tool()
def preflight_diff(rules_file: str, project_root: str = ".") -> str:
    """Check the current git diff against project guardrails BEFORE spending an enforce iteration — catches forbidden patterns, banned imports, and custom regex rules."""
    try:
        import subprocess

        from agent_xray.enforce_report import (
            check_against_rules,
            load_project_rules,
        )

        rules = load_project_rules(rules_file)
        if not rules:
            return _json_response({"error": f"No rules found in {rules_file}"})

        result = subprocess.run(
            ["git", "diff", "HEAD"],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        diff = result.stdout
        if not diff:
            return _json_response({"violations": [], "status": "clean", "hint": "No staged/unstaged changes to check."})

        violations = check_against_rules(diff, rules)
        return _compact_json({
            "violations": violations,
            "count": len(violations),
            "status": "FAIL" if violations else "PASS",
            "hint": "Fix violations before running enforce_check." if violations else "Diff passes all project rules.",
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def analyze(log_dir: str, rules: str | None = None, format: str = "auto", task_bank: str | None = None, days: int | None = None, site: str | None = None) -> str:
    """Analyze agent traces to get grade distribution, root causes, and a fix plan (filterable by days/site)."""
    try:
        from agent_xray.analyzer import analyze_task
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format, days=days, site=site, dedupe=True)
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
        return _compact_json(_with_grade_note(payload))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def grade(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None, days: int | None = None, site: str | None = None, outcome: str | None = None, grade_filter: str | None = None, dedupe: bool = True) -> str:
    """Grade traces against a ruleset and return scored per-task details — outcome filters by task status (failed/success), grade_filter filters by xray grade (BROKEN/WEAK/OK/GOOD/GOLDEN)."""
    try:
        from agent_xray.grader import grade_tasks, load_rules

        tasks = _load_tasks(log_dir, format, days=days, site=site, outcome=outcome, dedupe=dedupe)
        rule_set = load_rules(rules)

        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)

        tasks, grades = _filter_by_grade(tasks, grades, grade_filter)
        if grade_filter and not tasks:
            return _json_response({"error": "No tasks match grade_filter", "hint": f"No tasks graded as {grade_filter!r}. Valid grades: BROKEN, WEAK, OK, GOOD, GOLDEN."})

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
            "structural_qualifier": "summary_only",
            "worst_tasks": [
                {
                    "task_id": r.task_id,
                    "grade": r.grade,
                    "score": r.score,
                    "reasons": r.reasons[:5] if hasattr(r, "reasons") else [],
                }
                for r in worst
            ],
            "next": f"inspect_task(log_dir=log_dir, task_id='{worst[0].task_id}') for full investigation, diagnose(log_dir=log_dir) for fix plan, compare_runs(left_log_dir='<before>', right_log_dir='<after>') after fixes" if worst else "No failures found.",
        }
        return _compact_json(_with_grade_note(payload))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def root_cause(log_dir: str, rules: str = "default", format: str = "auto", days: int | None = None, site: str | None = None, outcome: str | None = None, grade_filter: str | None = None, dedupe: bool = True) -> str:
    """Classify weak/broken tasks into root cause categories with evidence — understand WHY tasks fail before fixing (filterable by days/site/outcome/grade_filter)."""
    try:
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures, summarize_root_causes

        tasks = _load_tasks(log_dir, format, days=days, site=site, outcome=outcome, dedupe=dedupe)
        rule_set = load_rules(rules)
        grades = grade_tasks(tasks, rule_set)
        tasks, grades = _filter_by_grade(tasks, grades, grade_filter)
        if grade_filter and not tasks:
            return _json_response({"error": "No tasks match grade_filter", "hint": f"No tasks graded as {grade_filter!r}. Valid grades: BROKEN, WEAK, OK, GOOD, GOLDEN."})
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
            payload["note"] = f"Showing 20 worst of {len(failures)}. Next: diagnose() for prioritized fix plan, or surface_task(task_id)/reasoning(task_id) to inspect specific failures."
        else:
            payload["next"] = f"inspect_task(log_dir=log_dir, task_id='{shown[0].task_id}') for full investigation, diagnose(log_dir=log_dir) for fix plan, compare_runs(left_log_dir='<before>', right_log_dir='<after>') after fixes" if shown else "diagnose(log_dir=log_dir) for fix plan"
        return _compact_json(_with_grade_note(payload))
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
def surface_task(log_dir: str, task_id: str, format: str = "auto", task_bank: str | None = None, max_steps: int | None = None) -> str:
    """Replay a task step-by-step: tools, inputs, results, reasoning — use BEFORE fixing to see what the agent saw (pass max_steps to limit, or use reasoning() for a lighter view)."""
    try:
        from agent_xray.surface import surface_for_task as run_surface

        tasks = _load_tasks(log_dir, format)
        task_map = {t.task_id: t for t in tasks}
        task = task_map.get(task_id)
        if task is None:
            return _json_response({"error": f"Task {task_id!r} not found in {log_dir}"})

        surface = run_surface(task)
        surface.setdefault("grade_note", _STRUCTURAL_GRADE_NOTE)

        if task_bank:
            from agent_xray.analyzer import analyze_task
            from agent_xray.contrib.task_bank import (
                evaluate_task_criteria,
                load_task_bank,
                match_task_to_bank,
            )

            bank = load_task_bank(task_bank)
            analysis = analyze_task(task)
            matched = match_task_to_bank(task, bank, analysis=analysis)
            if matched:
                criteria = matched.get("success_criteria", {})
                criterion_lines = evaluate_task_criteria(task, analysis, criteria)
                surface["task_bank_match"] = {
                    "id": matched.get("id", "unknown"),
                    "category": matched.get("category", ""),
                    "expected_user_text": matched.get("user_text", ""),
                    "difficulty": matched.get("difficulty", ""),
                    "criteria_results": criterion_lines,
                }

        # MCP: strip duplicate prompt fields and quadratic conversation_history
        metadata = surface.get("metadata", {})
        metadata.pop("system_prompt_text", None)
        metadata.pop("system_context_components", None)

        steps = surface.get("steps", [])

        # If max_steps requested, keep only first + last N steps
        if max_steps and len(steps) > max_steps:
            kept = steps[:max_steps]
            surface["steps"] = kept
            surface["steps_note"] = f"Showing {max_steps} of {len(steps)} steps. Pass higher max_steps or use CLI for full output."
            steps = kept

        # Progressive truncation: try increasingly aggressive limits until it fits
        for result_limit in (500, 200, 80, 0):
            for step in steps:
                step.pop("conversation_history", None)
                for key in ("tool_result", "result_summary"):
                    val = step.get(key)
                    if isinstance(val, str):
                        if result_limit == 0:
                            step[key] = f"[{len(val)} chars — use reasoning() for details]"
                        elif len(val) > result_limit:
                            step[key] = val[:result_limit] + "..."
                # Strip tool_input values >200 chars in aggressive modes
                if result_limit <= 200:
                    tool_input = step.get("tool_input", {})
                    if isinstance(tool_input, dict):
                        for k, v in list(tool_input.items()):
                            if isinstance(v, str) and len(v) > 200:
                                tool_input[k] = v[:200] + "..."

            result = json.dumps(surface, separators=(",", ":"))
            if len(result) <= _MCP_MAX_CHARS:
                return result

        # Final fallback: keep only metadata + step summaries (tool_name, error, duration)
        compact_steps = []
        for step in surface.get("steps", []):
            compact_steps.append({
                "step": step.get("step_number", step.get("step", "?")),
                "tool": step.get("tool_name", "?"),
                "error": step.get("error"),
                "duration_ms": step.get("duration_ms"),
                "page_url": step.get("page_url", step.get("browser", {}).get("page_url") if isinstance(step.get("browser"), dict) else None),
            })
        surface["steps"] = compact_steps
        surface["truncation"] = "Progressive truncation applied. Use reasoning() for model thinking, or CLI for full surface."
        return _compact_json(surface)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def search_tasks(log_dir: str, query: str, format: str = "auto", days: int | None = None, site: str | None = None) -> str:
    """Search tasks by user_text substring to find specific task IDs for further inspection (filterable by days/site)."""
    try:
        tasks = _load_tasks(log_dir, format, days=days, site=site, dedupe=True)
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
def diagnose(log_dir: str, rules: str = "default", format: str = "auto", task_bank: str | None = None, days: int | None = None, site: str | None = None, outcome: str | None = None, grade_filter: str | None = None, dedupe: bool = True) -> str:
    """Build a prioritized fix plan from classified failures — decide WHAT to fix before starting an enforce cycle (filterable by days/site/outcome/grade_filter)."""
    try:
        from agent_xray.diagnose import build_fix_plan
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures

        tasks = _load_tasks(log_dir, format, days=days, site=site, outcome=outcome, dedupe=dedupe)
        rule_set = load_rules(rules)

        if task_bank:
            from agent_xray.contrib.task_bank import grade_with_task_bank
            grades = grade_with_task_bank(tasks, task_bank, rule_set)
        else:
            grades = grade_tasks(tasks, rule_set)

        tasks, grades = _filter_by_grade(tasks, grades, grade_filter)
        if grade_filter and not tasks:
            return _json_response({"error": "No tasks match grade_filter", "hint": f"No tasks graded as {grade_filter!r}. Valid grades: BROKEN, WEAK, OK, GOOD, GOLDEN."})
        failures = classify_failures(tasks, grades)
        plan = build_fix_plan(failures, log_dir=log_dir)

        return _compact_json(_with_grade_note({
            "summary": {
                "tasks": len(tasks),
                "rules": rule_set.name,
                "task_bank": task_bank or "none",
                "failures_classified": len(failures),
                "fix_plan_entries": len(plan),
            },
            "fix_plan": [_serialize(entry) for entry in plan],
            "next": f"inspect_task(log_dir=log_dir, task_id='{plan[0].investigate_task}') to replay top fix target, enforce(hypothesis='<why>') to start fixing, compare_runs(left_log_dir='<before>', right_log_dir='<after>') after" if plan and plan[0].investigate_task else "enforce(hypothesis='<why>') to start disciplined fixing, compare_runs(left_log_dir='<before>', right_log_dir='<after>') after",
        }))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def compare_runs(left_log_dir: str, right_log_dir: str, rules: str = "default", format: str = "auto") -> str:
    """Compare two trace sets side by side to find grade shifts, cost deltas, and decision divergences."""
    try:
        from agent_xray.baseline import suggest_baseline_capture
        from agent_xray.comparison import compare_model_runs
        from agent_xray.grader import grade_tasks, load_rules

        result = compare_model_runs(
            left_log_dir,
            right_log_dir,
            rules_path=rules if rules != "default" else None,
        )

        payload = _serialize(result)
        if isinstance(payload, dict):
            rule_set = load_rules(rules if rules != "default" else None)
            before_grades = {
                grade.task_id: grade.grade
                for grade in grade_tasks(_load_tasks(left_log_dir, format), rule_set)
            }
            after_grades = {
                grade.task_id: grade.grade
                for grade in grade_tasks(_load_tasks(right_log_dir, format), rule_set)
            }
            baseline_suggestions = suggest_baseline_capture(before_grades, after_grades)
            if baseline_suggestions:
                payload["baseline_suggestions"] = baseline_suggestions
            return _compact_json(_with_grade_note(payload))
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def report(
    log_dir: str,
    report_type: str,
    rules: str = "default",
    format: str = "auto",
    task_bank: str | None = None,
    baseline_dir: str | None = None,
    day1: str | None = None,
    day2: str | None = None,
    days: int | None = None,
    site: str | None = None,
    min_steps: int = 0,
) -> str:
    """Generate a focused report (16 types including overhead, prompt-impact, compare); overhead needs baseline_dir, compare needs day1+day2."""
    try:
        from agent_xray.analyzer import analyze_tasks
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.reports import (
            report_actions_data,
            report_broken_data,
            report_coding_data,
            report_compare_days_data,
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

        tasks = _load_tasks(log_dir, format, days=days, site=site, dedupe=True)
        grades: list[Any] = []
        rule_name = rules
        needs_grades = report_type in _GRADE_DEPENDENT_REPORT_TYPES or report_type in ("overhead", "prompt-impact", "compare")
        if needs_grades:
            rule_set = load_rules(rules)
            rule_name = rule_set.name
            if task_bank:
                from agent_xray.contrib.task_bank import grade_with_task_bank
                grades = grade_with_task_bank(tasks, task_bank, rule_set)
            else:
                grades = grade_tasks(tasks, rule_set)

        analyses = analyze_tasks(tasks)

        # Handle overhead report (needs baselines)
        if report_type == "overhead":
            from agent_xray.baseline import (
                group_by_prompt_hash,
                load_baselines,
                measure_all_overhead,
                overhead_report_data,
            )
            if not baseline_dir:
                return _json_response({"error": "overhead report requires baseline_dir parameter"})
            baselines = load_baselines(baseline_dir)
            if not baselines:
                return _json_response({"error": f"No baselines found in {baseline_dir}"})
            grade_map = {g.task_id: g.grade for g in grades} if grades else {}
            results = measure_all_overhead(tasks, grade_map, baselines)
            hash_groups = group_by_prompt_hash(tasks, analyses, grade_map, baselines)
            payload = {
                "report_type": "overhead",
                "tasks": len(tasks),
                "rules": rule_name,
                "data": overhead_report_data(results, hash_groups),
            }
            return _compact_json(_with_grade_note(payload))

        # Handle prompt-impact report
        if report_type == "prompt-impact":
            from agent_xray.baseline import (
                group_by_prompt_hash,
                prompt_impact_data,
            )
            grade_map = {g.task_id: g.grade for g in grades} if grades else {}
            hash_groups = group_by_prompt_hash(tasks, analyses, grade_map)
            payload = {
                "report_type": "prompt-impact",
                "tasks": len(tasks),
                "rules": rule_name,
                "data": prompt_impact_data(hash_groups),
            }
            return _compact_json(_with_grade_note(payload))

        # Handle compare (day-over-day) report
        if report_type == "compare":
            if not day1 or not day2:
                return _json_response({"error": "compare report requires day1 and day2 parameters (YYYYMMDD format)"})
            payload = {
                "report_type": "compare",
                "tasks": len(tasks),
                "rules": rule_name,
                "data": report_compare_days_data(tasks, grades, analyses, day1, day2),
            }
            return _compact_json(_with_grade_note(payload))

        data_funcs: dict[str, Any] = {
            "health": lambda: report_health_data(tasks, grades, analyses),
            "golden": lambda: report_golden_data(tasks, grades, analyses, min_steps=min_steps),
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

        payload = {
            "report_type": report_type,
            "tasks": len(tasks),
            "rules": rule_name,
            "data": data_funcs[report_type](),
        }
        if needs_grades:
            _with_grade_note(payload)
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def diff_tasks(log_dir: str, task_id_a: str, task_id_b: str, format: str = "auto") -> str:
    """Compare two tasks side by side: tool sequences, timing, outcomes. Use to see what a successful task did differently from a failed one."""
    try:
        from agent_xray.surface import diff_tasks as run_diff_tasks

        tasks = _load_tasks(log_dir, format)
        task_map = {t.task_id: t for t in tasks}
        left = task_map.get(task_id_a)
        right = task_map.get(task_id_b)
        if left is None:
            return _json_response({"error": f"Task {task_id_a!r} not found in {log_dir}"})
        if right is None:
            return _json_response({"error": f"Task {task_id_b!r} not found in {log_dir}"})

        result = _serialize(run_diff_tasks(left, right))

        # MCP: strip duplicate prompt fields, conversation_history, and dedup prompts across sides
        left_hash = result.get("left", {}).get("metadata", {}).get("system_prompt_hash")
        right_hash = result.get("right", {}).get("metadata", {}).get("system_prompt_hash")
        same_prompt = left_hash and right_hash and left_hash == right_hash

        # Progressive truncation for diff_tasks (same approach as surface_task)
        for result_limit in (500, 200, 80, 0):
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
                        if isinstance(val, str):
                            if result_limit == 0:
                                step[key] = f"[{len(val)} chars]"
                            elif len(val) > result_limit:
                                step[key] = val[:result_limit] + "..."
                    if result_limit <= 200:
                        tool_input = step.get("tool_input", {})
                        if isinstance(tool_input, dict):
                            for k, v in list(tool_input.items()):
                                if isinstance(v, str) and len(v) > 200:
                                    tool_input[k] = v[:200] + "..."

            if same_prompt:
                result["prompt_note"] = f"Both tasks share the same prompt (hash: {left_hash}). Prompt text omitted."

            rendered = json.dumps(result, separators=(",", ":"))
            if len(rendered) <= _MCP_MAX_CHARS:
                return rendered

        # Final fallback: compact step summaries only
        for side in ("left", "right"):
            side_data = result.get(side, {})
            compact = []
            for step in side_data.get("steps", []):
                compact.append({
                    "tool": step.get("tool_name", "?"),
                    "error": step.get("error"),
                    "duration_ms": step.get("duration_ms"),
                })
            side_data["steps"] = compact
        result["truncation"] = "Progressive truncation applied. Use CLI for full diff."
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
def tree(log_dir: str, rules: str | None = None, format: str = "auto", days: int | None = None, site: str | None = None) -> str:
    """Bird's-eye view: day/site/task hierarchy with pass/fail counts — see which sites are failing before drilling in (filterable by days/site)."""
    try:
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.surface import enriched_tree_for_tasks

        tasks = _load_tasks(log_dir, format, days=days, site=site, dedupe=True)
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
        payload = {
            "task_count": len(tasks),
            "rules": rules or "none",
            "tree": compact_tree,
            "note": "Sites collapsed to counts with up to 3 sample_task_ids. Use CLI `agent-xray tree` for per-task details.",
        }
        if grades is not None:
            _with_grade_note(payload)
        return _compact_json(payload)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_rank(
    log_dir: str,
    rules: str | None = None,
    optimize: str = "balanced",
    format: str = "auto",
    per_site: bool = True,
) -> str:
    """Rank best runs by efficiency, grouping by site with configurable optimization profile."""
    try:
        from agent_xray.golden import STRUCTURAL_RANKING_NOTE, rank_golden_runs
        from agent_xray.grader import load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        rankings = rank_golden_runs(tasks, rules=rule_set, optimize=optimize, per_site=per_site)

        payload: dict[str, Any] = {
            "summary": {
                "tasks": len(tasks),
                "optimize": optimize,
                "sites_ranked": len(rankings),
                "per_site": per_site,
            },
            "rankings": {
                site: [_serialize(r) for r in ranks]
                for site, ranks in rankings.items()
            },
            "note": STRUCTURAL_RANKING_NOTE,
        }
        return _compact_json(_with_grade_note(payload))
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
        payload = {
            "summary": summary,
            "comparisons": results,
        }
        return _compact_json(_with_grade_note(payload))
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


def flywheel(log_dir: str, rules: str | None = None, format: str = "auto") -> str:
    """Full quality loop in one call: grade + root cause + baseline comparison. Note: flywheel loads all tasks internally; use grade/diagnose with days/site for filtered analysis."""
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
        return _compact_json(_with_grade_note(payload))
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
    """Look up per-token pricing for a model, showing input/output/cached costs and alias resolution."""
    try:
        from agent_xray.pricing import format_model_pricing, load_pricing

        pricing_data = load_pricing()
        formatted = format_model_pricing(model_name, pricing_data)
        result: dict[str, Any] = {"model": model_name, "pricing": formatted}
        # Show alias resolution path
        models = pricing_data.get("models", {})
        aliases = pricing_data.get("aliases", {})
        if model_name in models:
            result["resolved_via"] = "exact"
        elif model_name in aliases:
            result["resolved_via"] = "alias"
            result["canonical_model"] = aliases[model_name]
        else:
            # Check prefix match
            for key in models:
                if model_name.startswith(key):
                    result["resolved_via"] = "prefix"
                    result["canonical_model"] = key
                    break
        return _json_response(result)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def replay(log_dir: str, fixture_path: str, format: str = "auto") -> str:
    """Compare a saved golden fixture against current traces. Returns IMPROVED, REGRESSION, STABLE, or UNMATCHED verdict with milestone and step count comparison."""
    try:
        from agent_xray.replay import replay_fixture as run_replay

        tasks = _load_tasks(log_dir, format)
        result = run_replay(fixture_path, tasks)
        return _compact_json(result)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def validate_targets(log_dir: str, project_root: str, rules: str = "default", format: str = "auto", resolver: str | None = None) -> str:
    """Check that fix-plan target paths actually exist on disk. Catches stale file references in diagnose output."""
    try:
        from agent_xray.diagnose import (
            build_fix_plan,
            validate_fix_targets,
        )
        from agent_xray.grader import grade_tasks, load_rules
        from agent_xray.root_cause import classify_failures

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules)
        grades = grade_tasks(tasks, rule_set)
        failures = classify_failures(tasks, grades)
        plan = build_fix_plan(failures, log_dir=log_dir)
        validated = validate_fix_targets(plan, project_root)

        stale = []
        valid = 0
        total = 0
        for entry in validated:
            for target in entry.targets:
                total += 1
                stale_markers = [e for e in entry.evidence if "STALE_TARGET" in e and target in e]
                if stale_markers:
                    stale.append({"root_cause": entry.root_cause, "target": target})
                else:
                    valid += 1

        return _compact_json({
            "project_root": project_root,
            "total_targets": total,
            "valid": valid,
            "stale": len(stale),
            "stale_targets": stale[:20],
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def rules_list() -> str:
    """List all available built-in rulesets with names and descriptions. Use to discover what rulesets exist before grading."""
    try:
        from importlib.resources import files as pkg_files
        from pathlib import Path

        rules_dir = Path(str(pkg_files("agent_xray.rules")))
        rulesets: list[dict[str, str]] = []
        for path in sorted(rules_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rulesets.append({
                    "name": data.get("name", path.stem),
                    "description": data.get("description", ""),
                    "file": path.name,
                })
            except Exception:
                continue

        return _compact_json({
            "count": len(rulesets),
            "rulesets": rulesets,
            "usage": "Pass the name to any tool's 'rules' parameter, e.g. grade(rules='browser_flow')",
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def rules_show(name: str) -> str:
    """Show a ruleset's full configuration: signals, grade thresholds, and golden requirements."""
    try:
        from agent_xray.grader import load_rules

        rules = load_rules(name)
        return _compact_json({
            "name": rules.name,
            "description": rules.description,
            "signals": rules.signals,
            "grade_thresholds": rules.grade_thresholds,
            "golden_requirements": rules.golden_requirements,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


def rules_init(base: str = "default") -> str:
    """Generate a scaffold for a custom ruleset based on an existing one. Returns JSON you can save and customize."""
    try:
        scaffold = {
            "name": "my_custom_rules",
            "description": "Custom grading rules. Edit signals, thresholds, and requirements to match your agent.",
            "extends": base,
            "signals": [
                {
                    "name": "example_custom_signal",
                    "metric": "unique_tools",
                    "gte": 5,
                    "points": 1,
                    "reason": "+1 used 5+ unique tools (customize this)",
                }
            ],
            "grade_thresholds": {},
            "golden_requirements": [],
        }
        return _compact_json(scaffold)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def baseline_capture(log_dir: str, task_id: str, output: str | None = None, format: str = "auto") -> str:
    """Capture a task's metrics as a baseline for overhead measurement. Use on golden/exemplar tasks."""
    try:
        from pathlib import Path

        from agent_xray.analyzer import analyze_task, resolve_task
        from agent_xray.baseline import build_baseline, save_baseline

        tasks = _load_tasks(log_dir, format)
        task = resolve_task(tasks, task_id)
        analysis = analyze_task(task)
        baseline = build_baseline(task, analysis)
        out = Path(output) if output else Path.cwd() / "baselines" / f"{analysis.site_name}.json"
        path = save_baseline(baseline, out)
        return _compact_json({
            "saved_to": str(path),
            "baseline": baseline.to_dict(),
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def baseline_list(baselines_dir: str) -> str:
    """List all saved baselines in a directory with their metrics."""
    try:
        from agent_xray.baseline import load_baselines

        baselines = load_baselines(baselines_dir)
        if not baselines:
            return _json_response({"baselines": [], "note": f"No baselines found in {baselines_dir}"})

        items = [
            {
                "site": name,
                "steps": bl.step_count,
                "duration_s": round(bl.duration_s, 1),
                "cost_usd": round(bl.cost_usd, 4),
                "error_count": bl.error_count,
                "milestones": bl.milestones,
                "task_id": bl.task_id,
            }
            for name, bl in sorted(baselines.items())
        ]
        return _compact_json({
            "count": len(items),
            "baselines": items,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_best(log_dir: str, rules: str | None = None, optimize: str = "balanced", format: str = "auto") -> str:
    """Find the single best exemplar per site — the most efficient golden run. Use to identify which tasks to capture as baselines or fixtures."""
    try:
        from agent_xray.golden import STRUCTURAL_RANKING_NOTE, find_exemplars
        from agent_xray.grader import load_rules

        tasks = _load_tasks(log_dir, format)
        rule_set = load_rules(rules) if rules else load_rules()
        exemplars = find_exemplars(tasks, rules=rule_set, optimize=optimize)

        return _compact_json(_with_grade_note({
            "summary": {
                "tasks": len(tasks),
                "optimize": optimize,
                "exemplars_found": len(exemplars),
            },
            "exemplars": [_serialize(e) for e in exemplars],
            "note": STRUCTURAL_RANKING_NOTE,
        }))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_profiles() -> str:
    """Show available optimization profiles with their weight distributions for golden ranking."""
    try:
        from agent_xray.golden import OPTIMIZATION_PROFILES, STRUCTURAL_RANKING_NOTE

        return _compact_json({
            "profiles": {
                name: weights
                for name, weights in sorted(OPTIMIZATION_PROFILES.items())
            },
            "usage": "Pass the profile name to golden_rank or golden_best's 'optimize' parameter",
            "note": STRUCTURAL_RANKING_NOTE,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


def pricing_list() -> str:
    """List all known models with input/output/cached token pricing per 1M tokens."""
    try:
        from agent_xray.pricing import load_pricing

        pricing_data = load_pricing()
        models = pricing_data.get("models", {})
        rows = []
        for model_name in sorted(models):
            entry = models[model_name]
            rows.append({
                "model": model_name,
                "input_per_1m": entry.get("input", 0.0),
                "output_per_1m": entry.get("output", 0.0),
                "cached_per_1m": entry.get("cached_input"),
            })
        aliases = pricing_data.get("aliases", {})
        return _compact_json({
            "models": rows,
            "aliases": aliases,
            "total": f"{len(models)} models, {len(aliases)} aliases",
        })
    except Exception as e:
        return _json_response({"error": str(e)})


def baseline_generate(log_dir: str, task_id: str, format: str = "auto") -> str:
    """Generate a naked prompt (system message only, no tools/history) for baseline comparison."""
    try:
        from agent_xray.baseline import generate_naked_prompt

        tasks = _load_tasks(log_dir, format)
        task = _resolve_task(tasks, task_id)
        prompt = generate_naked_prompt(task)
        return _compact_json({"task_id": task.task_id, "naked_prompt": prompt})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def task_bank_show(path: str, task_id: str) -> str:
    """Show a single task bank entry by ID or prefix match."""
    try:
        from agent_xray.contrib.task_bank import load_task_bank as load_entries

        entries = load_entries(path)
        matched = next((e for e in entries if str(e.get("id")) == task_id), None)
        if matched is None:
            prefix = [e for e in entries if str(e.get("id", "")).startswith(task_id)]
            if len(prefix) == 1:
                matched = prefix[0]
        if matched is None:
            return _json_response({"error": f"Task bank entry not found: {task_id}"})
        return _compact_json(matched)
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def format_detect(log_path: str) -> str:
    """Auto-detect the trace format of a log file or directory with confidence score."""
    try:
        from pathlib import Path

        from agent_xray.adapters import format_info

        p = Path(log_path)
        if p.is_dir():
            # format_info expects a file; pick the first trace file in the directory
            candidates = sorted(
                set(p.glob("*.jsonl")) | set(p.glob("*.json"))
            )
            if not candidates:
                return _json_response({"error": f"No .jsonl or .json files found in {log_path}"})
            fmt, confidence = format_info(candidates[0])
            return _compact_json({
                "format": fmt,
                "confidence": round(confidence, 3),
                "detected_from": candidates[0].name,
                "total_files": len(candidates),
            })

        fmt, confidence = format_info(log_path)
        return _compact_json({
            "format": fmt,
            "confidence": round(confidence, 3),
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def gaming_audit(diff: str, files_modified: list[str] | None = None, allow_test_modification: bool = False, project_allowlist: list[str] | None = None) -> str:
    """Run 8 gaming detectors on a diff to check for test-gaming, hardcoded values, mock injection, etc."""
    try:
        from agent_xray.enforce_audit import audit_change, classify_diff_quality

        verdict, reasons, signal_names = audit_change(
            diff,
            files_modified,
            allow_test_modification=allow_test_modification,
            project_allowlist=project_allowlist,
        )
        quality = classify_diff_quality(diff, files_modified or [], 0)
        return _compact_json({
            "verdict": verdict,
            "quality": quality,
            "reasons": reasons,
            "signals": signal_names,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


def pricing_update() -> str:
    """Fetch latest model pricing from GitHub and update the local cache."""
    try:
        from agent_xray.pricing import update_pricing_cache

        ok, msg = update_pricing_cache()
        return _json_response({"success": ok, "message": msg})
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def inspect_task(log_dir: str, task_id: str, format: str = "auto") -> str:
    """Comprehensive single-task report: grade + root cause + surface (step-by-step) + reasoning chain in one call."""
    try:
        from agent_xray.analyzer import analyze_task
        from agent_xray.grader import grade_task, load_rules
        from agent_xray.root_cause import classify_task as classify_rc
        from agent_xray.surface import reasoning_for_task, surface_for_task

        tasks = _load_tasks(log_dir, format)
        task = _resolve_task(tasks, task_id)
        rules = load_rules()
        analysis = analyze_task(task)
        grade = grade_task(task, rules, analysis=analysis)
        rc = classify_rc(task, grade)

        # Compact surface (tool sequence + errors only)
        surface = surface_for_task(task)
        steps = surface.get("steps", [])
        compact_steps = []
        for s in steps:
            entry: dict[str, Any] = {"step": s.get("step", 0), "tool": s.get("tool_name", "")}
            if s.get("error"):
                entry["error"] = str(s["error"])[:300]
            result = s.get("tool_result", "")
            if isinstance(result, str) and len(result) > 150:
                entry["result"] = result[:150] + "..."
            elif result:
                entry["result"] = str(result)[:150]
            compact_steps.append(entry)

        # Compact reasoning chain
        reasoning = reasoning_for_task(task)
        chain = []
        for r in reasoning.get("reasoning_chain", []):
            chain.append({
                "step": r.get("step"),
                "reasoning": (r.get("reasoning") or "")[:200],
                "tool": r.get("decision", {}).get("tool_name", ""),
            })

        payload: dict[str, Any] = {
            "task_id": task.task_id,
            "user_text": (task.task_text or "")[:150],
            "grade": grade.grade,
            "score": grade.score,
            "root_cause": rc.root_cause if rc else None,
            "confidence": rc.confidence if rc else None,
            "evidence": rc.evidence[:3] if rc and rc.evidence else [],
            "steps": compact_steps,
            "reasoning_chain": chain,
            "site": analysis.site_name,
            "metrics": {
                "errors": analysis.errors,
                "total_steps": len(task.steps),
                "total_cost_usd": analysis.total_cost_usd,
                "duration_ms": analysis.total_duration_ms,
            },
            "next": {
                "signals": f"signal_detect(log_dir=log_dir, task_id='{task.task_id}') for domain-specific signals (commerce, research, planning)",
                "compare": f"diff_tasks(log_dir=log_dir, task_id_a='<good_task_id>', task_id_b='{task.task_id}') to compare with a working task",
                "fix": "diagnose(log_dir=log_dir) for prioritized fix plan, then enforce(hypothesis='<why>') to start fixing",
            },
        }
        _with_grade_note(payload)

        result = json.dumps(payload, separators=(",", ":"))
        if len(result) > _MCP_MAX_CHARS:
            # Trim reasoning chain first, then step results
            for r in chain:
                r["reasoning"] = (r.get("reasoning") or "")[:80] + "..."
            for s in compact_steps:
                if "result" in s:
                    s["result"] = s["result"][:60] + "..."
            result = json.dumps(payload, separators=(",", ":"))

        return result
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def golden_capture(
    log_dir: str,
    task_id: str,
    output: str | None = None,
    optimize: str = "balanced",
    format: str = "auto",
) -> str:
    """Capture a golden exemplar task for future comparison and efficiency benchmarking."""
    try:
        from pathlib import Path

        from agent_xray.analyzer import analyze_task
        from agent_xray.golden import STRUCTURAL_EXEMPLAR_WARNING, capture_exemplar
        from agent_xray.grader import load_rules

        tasks = _load_tasks(log_dir, format)
        task = _resolve_task(tasks, task_id)
        rules = load_rules()
        site = analyze_task(task).site_name or None
        exemplar_path = capture_exemplar(
            tasks, rules=rules, site=site, optimize=optimize, output_path=output,
        )
        exemplar = json.loads(Path(exemplar_path).read_text(encoding="utf-8"))
        payload: dict[str, Any] = {
            "exemplar": exemplar,
            "correctness_warning": STRUCTURAL_EXEMPLAR_WARNING,
        }
        if output:
            payload["saved_to"] = str(exemplar_path)
        return _compact_json(_with_grade_note(payload))
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def signal_detect(log_dir: str, task_id: str, detector: str | None = None, format: str = "auto") -> str:
    """Run signal detectors on a single task, optionally filtering to one detector by name."""
    try:
        from agent_xray.signals import discover_detectors, run_detection

        tasks = _load_tasks(log_dir, format)
        task = _resolve_task(tasks, task_id)
        all_detectors = discover_detectors()

        if detector:
            matched = [d for d in all_detectors if d.name.lower() == detector.lower()]
            if not matched:
                available = [d.name for d in all_detectors]
                return _json_response({"error": f"Detector {detector!r} not found. Available: {available}"})
            results = run_detection(task, detectors=matched)
        else:
            results = run_detection(task, detectors=all_detectors)

        return _compact_json({
            "task_id": task.task_id,
            "detectors_run": list(results.keys()),
            "signals": results,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


@server.tool()
def match_task(log_dir: str, task_id: str, bank_path: str, format: str = "auto") -> str:
    """Fuzzy-match a task to the best task bank entry by user text, site, and category."""
    try:
        from agent_xray.analyzer import analyze_task
        from agent_xray.contrib.task_bank import load_task_bank, match_task_to_bank

        tasks = _load_tasks(log_dir, format)
        task = _resolve_task(tasks, task_id)
        bank = load_task_bank(bank_path)
        analysis = analyze_task(task)
        matched = match_task_to_bank(task, bank, analysis=analysis)

        if matched is None:
            return _json_response({"task_id": task.task_id, "match": None, "note": "No bank entry matched above threshold"})

        return _compact_json({
            "task_id": task.task_id,
            "match": matched,
        })
    except Exception as e:
        return _json_response({"error": str(e)})


_TOOL_DOC_OVERRIDES: dict[str, str] = {
    "triage": "Summarize a trace set into grade distribution, the single worst task, and a prioritized fix plan, returning JSON text or an error payload.",
    "analyze": (
        "Analyze agent traces to get grade distribution, root causes, and a fix plan "
        "(filterable by days/site)."
    ),
    "enforce_quick": "Run enforce init+check in one call and return the combined review payload.",
    "report": "Generate a focused report such as health, tools, flows, outcomes, cost, fixes, timeline, overhead, or compare, returning JSON text or an error payload.",
}

_TOOL_NAMES = (
    "triage",
    "enforce",
    "enforce_quick",
    "preflight_diff",
    "analyze",
    "grade",
    "root_cause",
    "completeness",
    "surface_task",
    "search_tasks",
    "diagnose",
    "compare_runs",
    "report",
    "diff_tasks",
    "reasoning",
    "tree",
    "golden_rank",
    "golden_compare",
    "task_bank_validate",
    "task_bank_list",
    "capture_task",
    "pricing_show",
    "replay",
    "validate_targets",
    "rules_list",
    "rules_show",
    "baseline_capture",
    "baseline_list",
    "golden_best",
    "golden_profiles",
    "task_bank_show",
    "format_detect",
    "gaming_audit",
    "inspect_task",
    "golden_capture",
    "signal_detect",
    "match_task",
)


def _apply_tool_docstrings() -> None:
    """Ensure every MCP tool docstring states what it does, when to use it, and what it returns."""
    for tool_name in _TOOL_NAMES:
        func = globals().get(tool_name)
        if not callable(func):
            continue
        override = _TOOL_DOC_OVERRIDES.get(tool_name)
        if override:
            func.__doc__ = override
            continue
        doc = (func.__doc__ or "").strip()
        if not doc:
            func.__doc__ = f"{tool_name} tool for the corresponding agent-xray analysis."


_apply_tool_docstrings()


def main() -> None:
    """Run the agent-xray MCP server over stdio transport."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
