"""Text, data, and Markdown report generation for agent-xray.`r`n`r`nEach report type exposes a text formatter, a structured ``*_data`` variant for`r`nJSON output, and a GitHub-flavored ``*_markdown`` variant for docs and issue`r`nthreads. Reports operate on already-loaded tasks, grades, and analyses so the`r`nCLI and tests can share the same computations.`r`n"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .analyzer import TaskAnalysis, classify_error
from .diagnose import build_fix_plan
from .grader import GradeResult
from .root_cause import ROOT_CAUSES, classify_failures
from .schema import AgentTask

GRADE_LABELS = ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]


def _grade_distribution(grades: list[GradeResult]) -> dict[str, int]:
    counts = Counter(g.grade for g in grades)
    return {label: counts.get(label, 0) for label in GRADE_LABELS}


def _bar(count: int, total: int, width: int = 30) -> str:
    if total == 0:
        return ""
    filled = round(count * width / total)
    return "#" * filled + "." * (width - filled)


def _pct(count: int, total: int) -> int:
    return round(count * 100 / max(1, total))


def _format_money(amount: float) -> str:
    return f"${amount:.4f}"


def _task_category(task: AgentTask) -> str:
    return task.task_category or "unknown"


def _primary_model(task: AgentTask) -> str:
    counts = Counter((step.model_name or "unknown") for step in task.sorted_steps)
    return counts.most_common(1)[0][0] if counts else "unknown"


def _task_models(task: AgentTask) -> list[str]:
    models = sorted({step.model_name for step in task.sorted_steps if step.model_name})
    return models or ["unknown"]


def _markdown_escape(value: Any) -> str:
    return str(value).replace("\r", " ").replace("\n", "<br>").replace("|", "\\|")


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["-", *(["-"] * (len(headers) - 1))]]
    header = "| " + " | ".join(_markdown_escape(item) for item in headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(_markdown_escape(item) for item in row) + " |" for row in rows]
    return "\n".join([header, divider, *body])


def _text_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    header = "  " + "  ".join(f"{str(cell):<{widths[index]}}" for index, cell in enumerate(headers))
    divider = "  " + "  ".join("-" * width for width in widths)
    body = [
        "  " + "  ".join(f"{str(cell):<{widths[index]}}" for index, cell in enumerate(row))
        for row in rows
    ]
    return [header, divider, *body]


def _has_browser_steps(task: AgentTask) -> bool:
    return any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.sorted_steps)


def _tool_sequence(task: AgentTask) -> list[str]:
    return [step.tool_name.lower() for step in task.sorted_steps]


def _task_has_tool_keyword(task: AgentTask, *keywords: str) -> bool:
    return any(any(keyword in tool for keyword in keywords) for tool in _tool_sequence(task))


def _task_has_result_keyword(task: AgentTask, *keywords: str) -> bool:
    for step in task.sorted_steps:
        text = f"{step.tool_result or ''} {step.error or ''}".lower()
        if any(keyword in text for keyword in keywords):
            return True
    return False


def _task_has_url_keyword(analysis: TaskAnalysis, *keywords: str) -> bool:
    urls = " ".join(url.lower() for url in analysis.unique_urls)
    return any(keyword in urls for keyword in keywords)


def _delta_arrow(v1: float | int, v2: float | int, higher_is_better: bool = True) -> str:
    diff = v2 - v1
    if diff == 0:
        return "="
    arrow = "^" if diff > 0 else "v"
    good = (diff > 0) == higher_is_better
    marker = "OK" if good else "!!"
    return f"{arrow}{abs(diff)} {marker}"


# â”€â”€ Health Dashboard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _health_summary(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    dist = _grade_distribution(grades)
    total = len(grades)
    error_tasks = sum(1 for analysis in analyses.values() if analysis.errors > 0)
    spin_tasks = sum(1 for analysis in analyses.values() if analysis.is_spin)
    timeout_tasks = sum(1 for analysis in analyses.values() if analysis.timeout_like)
    halluc_tasks = sum(1 for analysis in analyses.values() if analysis.hallucinated_tools > 0)
    approval_tasks = sum(
        1 for analysis in analyses.values() if analysis.error_kinds.get("approval_block", 0) > 0
    )
    by_day: dict[str, list[GradeResult]] = defaultdict(list)
    for task, grade in zip(tasks, grades, strict=False):
        by_day[task.day or "unknown"].append(grade)
    day_trends = {}
    for day in sorted(by_day):
        day_grades = by_day[day]
        day_total = len(day_grades)
        golden = sum(1 for grade in day_grades if grade.grade == "GOLDEN")
        good = sum(1 for grade in day_grades if grade.grade == "GOOD")
        broken = sum(1 for grade in day_grades if grade.grade == "BROKEN")
        day_trends[day] = {
            "tasks": day_total,
            "golden": golden,
            "good": good,
            "broken": broken,
            "pass_pct": _pct(golden + good, day_total),
        }
    return {
        "total": total,
        "distribution": dist,
        "error_tasks": error_tasks,
        "spin_tasks": spin_tasks,
        "timeout_tasks": timeout_tasks,
        "hallucination_tasks": halluc_tasks,
        "approval_blocked_tasks": approval_tasks,
        "day_trends": day_trends,
    }


def report_health(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render the health dashboard as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A human-readable health dashboard string.
    """
    data = _health_summary(tasks, grades, analyses)
    total = data["total"]
    lines = ["HEALTH DASHBOARD", "=" * 60, ""]

    for label in GRADE_LABELS:
        count = data["distribution"][label]
        lines.append(f"  {label:7s} {_bar(count, total)} {count:3d} ({_pct(count, total)}%)")
    lines.append(f"\n  Total: {total} tasks")

    lines.extend(
        [
            "",
            f"  Error tasks: {data['error_tasks']}/{total} ({_pct(data['error_tasks'], total)}%)",
            f"  Spin tasks:  {data['spin_tasks']}/{total} ({_pct(data['spin_tasks'], total)}%)",
            f"  Timeouts:    {data['timeout_tasks']}/{total}",
            f"  Hallucinations: {data['hallucination_tasks']}/{total}",
            f"  Approval blocks: {data['approval_blocked_tasks']}/{total}",
        ]
    )

    if len(data["day_trends"]) > 1:
        rows = [
            [
                day,
                stats["tasks"],
                stats["golden"],
                stats["good"],
                stats["broken"],
                f"{stats['pass_pct']}%",
            ]
            for day, stats in data["day_trends"].items()
        ]
        lines.extend(
            [
                "",
                "DAY TRENDS:",
                *_text_table(["Day", "Tasks", "GOLDEN", "GOOD", "BROKEN", "Pass%"], rows),
            ]
        )

    return "\n".join(lines)


def report_health_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured health dashboard data.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing grade distribution, incident counts, and day trends.
    """
    return _health_summary(tasks, grades, analyses)


def report_health_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render the health dashboard as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown health dashboard string.
    """
    data = _health_summary(tasks, grades, analyses)
    summary_rows = [
        [label, data["distribution"][label], f"{_pct(data['distribution'][label], data['total'])}%"]
        for label in GRADE_LABELS
    ]
    metric_rows = [
        ["Total tasks", data["total"]],
        ["Error tasks", f"{data['error_tasks']} ({_pct(data['error_tasks'], data['total'])}%)"],
        ["Spin tasks", f"{data['spin_tasks']} ({_pct(data['spin_tasks'], data['total'])}%)"],
        ["Timeout tasks", data["timeout_tasks"]],
        ["Hallucination tasks", data["hallucination_tasks"]],
        ["Approval-blocked tasks", data["approval_blocked_tasks"]],
    ]
    parts = [
        "## Health Dashboard",
        "",
        _markdown_table(["Grade", "Count", "Pct"], summary_rows),
        "",
        _markdown_table(["Metric", "Value"], metric_rows),
    ]
    if len(data["day_trends"]) > 1:
        day_rows = [
            [
                day,
                stats["tasks"],
                stats["golden"],
                stats["good"],
                stats["broken"],
                f"{stats['pass_pct']}%",
            ]
            for day, stats in data["day_trends"].items()
        ]
        parts.extend(
            [
                "",
                "## Day Trends",
                "",
                _markdown_table(["Day", "Tasks", "GOLDEN", "GOOD", "BROKEN", "Pass%"], day_rows),
            ]
        )
    return "\n".join(parts)


# â”€â”€ Golden Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _golden_items(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    *,
    min_steps: int = 0,
) -> list[dict[str, Any]]:
    grade_by_id = {grade.task_id: grade for grade in grades}
    items: list[dict[str, Any]] = []
    for task in tasks:
        grade = grade_by_id.get(task.task_id)
        analysis = analyses.get(task.task_id)
        if grade is None or analysis is None:
            continue
        if grade.grade not in {"GOLDEN", "GOOD"} or analysis.step_count < min_steps:
            continue
        commerce = analysis.signal_metrics.get("commerce", {})
        milestones = []
        if commerce.get("reached_cart"):
            milestones.append("cart")
        if commerce.get("reached_checkout"):
            milestones.append("checkout")
        if commerce.get("reached_payment"):
            milestones.append("payment")
        items.append(
            {
                "task_id": task.task_id,
                "grade": grade.grade,
                "score": grade.score,
                "steps": analysis.step_count,
                "urls": len(analysis.unique_urls),
                "fills": commerce.get("fill_count", 0),
                "site": analysis.site_name,
                "milestones": milestones,
            }
        )
    items.sort(key=lambda item: -item["score"])
    return items


def report_golden(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    *,
    min_steps: int = 0,
) -> str:
    """Render GOLDEN and GOOD runs as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.
        min_steps: Minimum step count required to include a task.

    Returns:
        A text report describing high-quality runs.
    """
    golden_good = _golden_items(tasks, grades, analyses, min_steps=min_steps)
    lines = [f"GOLDEN/GOOD RUNS ({len(golden_good)} tasks)", "=" * 60, ""]
    for item in golden_good:
        milestone_str = ",".join(item["milestones"]) if item["milestones"] else "-"
        lines.append(
            f"  [{item['grade']:6s} {item['score']:+3d}] {item['task_id'][:20]:20s} "
            f"steps={item['steps']:2d} urls={item['urls']:2d} "
            f"fills={item['fills']} site={item['site']} "
            f"flow={milestone_str}"
        )
    if golden_good:
        avg_steps = sum(item["steps"] for item in golden_good) / len(golden_good)
        avg_urls = sum(item["urls"] for item in golden_good) / len(golden_good)
        site_counts: Counter[str] = Counter(item["site"] for item in golden_good)
        lines.extend(
            [
                "",
                f"  Avg steps: {avg_steps:.1f}, Avg URLs: {avg_urls:.1f}",
                f"  Sites: {dict(site_counts.most_common(10))}",
            ]
        )
    return "\n".join(lines)


def report_golden_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured data for GOLDEN and GOOD runs.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing ranked high-quality task entries.
    """
    items = _golden_items(tasks, grades, analyses)
    return {"count": len(items), "tasks": items}


def report_golden_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    *,
    min_steps: int = 0,
) -> str:
    """Render GOLDEN and GOOD runs as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.
        min_steps: Minimum step count required to include a task.

    Returns:
        A Markdown report describing high-quality runs.
    """
    rows = [
        [
            item["task_id"],
            item["grade"],
            item["score"],
            item["steps"],
            item["urls"],
            item["fills"],
            item["site"],
            ", ".join(item["milestones"]) or "-",
        ]
        for item in _golden_items(tasks, grades, analyses, min_steps=min_steps)
    ]
    return "\n".join(
        [
            "## Golden / Good Runs",
            "",
            _markdown_table(
                ["Task", "Grade", "Score", "Steps", "URLs", "Fills", "Site", "Flow"], rows
            ),
        ]
    )


# â”€â”€ Broken Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_broken(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render BROKEN tasks as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report describing the worst tasks and likely failure modes.
    """
    grade_by_id = {g.task_id: g for g in grades}
    broken = [
        (t, grade_by_id[t.task_id], analyses[t.task_id])
        for t in tasks
        if grade_by_id[t.task_id].grade == "BROKEN"
    ]
    broken.sort(key=lambda x: x[1].score)
    lines = [f"BROKEN TASKS ({len(broken)} tasks)", "=" * 60, ""]

    # WHY breakdown
    why: Counter[str] = Counter()
    for _, _, analysis in broken:
        if analysis.is_spin:
            why["spin"] += 1
        if analysis.timeout_like:
            why["timeout"] += 1
        if analysis.error_rate > 0.5:
            why["high_errors"] += 1
        if analysis.hallucinated_tools >= 2:
            why["hallucinated"] += 1
        if len(analysis.unique_urls) <= 1 and analysis.step_count >= 5:
            why["stuck_page"] += 1
        if analysis.no_tools_steps:
            why["routing"] += 1
    if why:
        lines.append("WHY breakdown:")
        for reason, count in why.most_common():
            lines.append(f"  {reason}: {count}")
        lines.append("")

    lines.append("Worst tasks:")
    for task, grade, analysis in broken[:15]:
        spin_info = (
            f" spin={analysis.max_repeat_tool}x{analysis.max_repeat_count}"
            if analysis.is_spin
            else ""
        )
        lines.append(
            f"  [{grade.score:+3d}] {task.task_id[:24]:24s} "
            f"steps={analysis.step_count} errs={analysis.errors}{spin_info} "
            f"site={analysis.site_name}"
        )
    return "\n".join(lines)


def report_broken_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured BROKEN-task data.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing reason counts and worst-task details.
    """
    grade_by_id = {g.task_id: g for g in grades}
    broken = [
        (t, grade_by_id[t.task_id], analyses[t.task_id])
        for t in tasks
        if grade_by_id[t.task_id].grade == "BROKEN"
    ]
    broken.sort(key=lambda x: x[1].score)
    why: Counter[str] = Counter()
    for _, _, analysis in broken:
        if analysis.is_spin:
            why["spin"] += 1
        if analysis.timeout_like:
            why["timeout"] += 1
        if analysis.error_rate > 0.5:
            why["high_errors"] += 1
        if analysis.hallucinated_tools >= 2:
            why["hallucinated"] += 1
        if len(analysis.unique_urls) <= 1 and analysis.step_count >= 5:
            why["stuck_page"] += 1
        if analysis.no_tools_steps:
            why["routing"] += 1
    items = []
    for task, grade, analysis in broken[:15]:
        items.append(
            {
                "task_id": task.task_id,
                "score": grade.score,
                "steps": analysis.step_count,
                "errors": analysis.errors,
                "spin": analysis.is_spin,
                "site": analysis.site_name,
            }
        )
    return {"count": len(broken), "why": dict(why), "worst_tasks": items}


def report_broken_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render BROKEN tasks as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown report describing the worst tasks and likely failure modes.
    """
    data = report_broken_data(tasks, grades, analyses)
    why_rows = [[reason, count] for reason, count in Counter(data["why"]).most_common()]
    task_rows = [
        [item["task_id"], item["score"], item["steps"], item["errors"], item["spin"], item["site"]]
        for item in data["worst_tasks"]
    ]
    return "\n".join(
        [
            "## Broken Tasks",
            "",
            _markdown_table(["Reason", "Count"], why_rows),
            "",
            "## Worst Tasks",
            "",
            _markdown_table(["Task", "Score", "Steps", "Errors", "Spin", "Site"], task_rows),
        ]
    )


# â”€â”€ Tool Effectiveness â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_tools(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render per-tool usage, error-rate, and latency statistics.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report describing tool usage, latency, and error rates.
    """
    tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    tool_durations: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        for step in task.sorted_steps:
            tool_calls[step.tool_name] += 1
            if step.error:
                tool_errors[step.tool_name] += 1
            if step.duration_ms:
                tool_durations[step.tool_name].append(step.duration_ms)

    lines = ["TOOL EFFECTIVENESS", "=" * 60, ""]
    lines.append(
        f"  {'Tool':30s} {'Calls':>6s} {'Errors':>7s} {'Err%':>5s} {'Avg ms':>7s} {'Flag':>8s}"
    )
    lines.append("  " + "-" * 65)
    for tool, calls in tool_calls.most_common():
        errs = tool_errors.get(tool, 0)
        err_pct = round(errs * 100 / max(1, calls))
        durations = tool_durations.get(tool, [])
        avg_ms = round(sum(durations) / len(durations)) if durations else 0
        flag = ""
        if err_pct > 50 and calls >= 5:
            flag = "BROKEN"
        elif err_pct > 20 and calls >= 5:
            flag = "HIGH ERR"
        lines.append(f"  {tool:30s} {calls:6d} {errs:7d} {err_pct:4d}% {avg_ms:7d} {flag:>8s}")

    broken_tools = [
        (name, tool_calls[name], tool_errors[name])
        for name in tool_calls
        if tool_errors.get(name, 0) / max(1, tool_calls[name]) > 0.3 and tool_calls[name] >= 5
    ]
    if broken_tools:
        lines.extend(["", "BROKEN/HIGH-ERROR TOOLS (>30% error, 5+ calls):"])
        for name, calls, errs in sorted(broken_tools, key=lambda x: -x[2]):
            lines.append(f"  {name}: {errs}/{calls} errors ({round(errs * 100 / calls)}%)")

    return "\n".join(lines)


def report_tools_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured tool-effectiveness data.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing per-tool usage, errors, and latency.
    """
    tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    tool_durations: dict[str, list[int]] = defaultdict(list)
    for task in tasks:
        for step in task.sorted_steps:
            tool_calls[step.tool_name] += 1
            if step.error:
                tool_errors[step.tool_name] += 1
            if step.duration_ms:
                tool_durations[step.tool_name].append(step.duration_ms)
    items = []
    for tool, calls in tool_calls.most_common():
        errs = tool_errors.get(tool, 0)
        durations = tool_durations.get(tool, [])
        avg_ms = round(sum(durations) / len(durations)) if durations else 0
        items.append(
            {
                "tool": tool,
                "calls": calls,
                "errors": errs,
                "error_pct": round(errs * 100 / max(1, calls)),
                "avg_ms": avg_ms,
            }
        )
    return {"tools": items}


def report_tools_markdown(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render tool effectiveness as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown tool-effectiveness report.
    """
    data = report_tools_data(tasks, analyses)
    rows = [
        [item["tool"], item["calls"], item["errors"], f"{item['error_pct']}%", item["avg_ms"]]
        for item in data["tools"]
    ]
    return "\n".join(
        [
            "## Tool Effectiveness",
            "",
            _markdown_table(["Tool", "Calls", "Errors", "Err%", "Avg ms"], rows),
        ]
    )


# â”€â”€ Flow Funnel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def _coding_flow_flags(task: AgentTask, analysis: TaskAnalysis) -> dict[str, bool]:
    coding = analysis.signal_metrics.get("coding", {})
    tools = _tool_sequence(task)
    edit_indices = [
        index
        for index, tool in enumerate(tools)
        if any(
            keyword in tool for keyword in ("edit", "write", "patch", "create", "delete", "file")
        )
    ]
    test_indices = [
        index
        for index, tool in enumerate(tools)
        if any(keyword in tool for keyword in ("test", "lint", "build", "type"))
    ]
    last_edit = max(edit_indices) if edit_indices else -1
    return {
        "Started": bool(task.steps),
        "Edit": coding.get("file_operations", 0) > 0 or bool(edit_indices),
        "Test": (
            coding.get("test_runs", 0) > 0
            or coding.get("lint_runs", 0) > 0
            or coding.get("build_runs", 0) > 0
            or bool(test_indices)
        ),
        "Fix": bool(
            edit_indices
            and test_indices
            and any(index > min(test_indices) for index in edit_indices)
        ),
        "Verify": bool(
            test_indices and last_edit >= 0 and any(index > last_edit for index in test_indices)
        ),
    }


def _research_flow_flags(task: AgentTask, analysis: TaskAnalysis) -> dict[str, bool]:
    research = analysis.signal_metrics.get("research", {})
    return {
        "Started": bool(task.steps),
        "Search": research.get("search_count", 0) > 0 or _task_has_tool_keyword(task, "search"),
        "Read": research.get("read_count", 0) > 0
        or _task_has_tool_keyword(task, "read", "fetch", "scrape"),
        "Synthesize": research.get("has_synthesis_step", False)
        or _task_has_tool_keyword(task, "respond", "answer", "summarize"),
    }


def _browser_flow_flags(task: AgentTask, analysis: TaskAnalysis) -> dict[str, bool]:
    return {
        "Started": bool(task.steps),
        "Browse": _has_browser_steps(task)
        or _task_has_url_keyword(analysis, "search", "docs", "issues"),
        "Form": _task_has_tool_keyword(task, "fill", "type", "select")
        or _task_has_url_keyword(analysis, "form", "settings", "wizard"),
        "Complete": (
            bool(task.outcome and task.outcome.status == "success")
            or _task_has_url_keyword(analysis, "review", "confirm", "complete", "success", "done")
            or _task_has_result_keyword(task, "submitted", "completed", "confirmed", "success")
        ),
    }


def _commerce_flow_flags(task: AgentTask, analysis: TaskAnalysis) -> dict[str, bool]:
    commerce = analysis.signal_metrics.get("commerce", {})
    return {
        "Started": bool(task.steps),
        "Cart": bool(commerce.get("reached_cart")),
        "Checkout": bool(commerce.get("reached_checkout")),
        "Payment": bool(commerce.get("reached_payment")),
    }


def _detect_flow_group(analysis: TaskAnalysis) -> dict[str, Any] | None:
    task = analysis.task
    category = _task_category(task)
    commerce = analysis.signal_metrics.get("commerce", {})
    coding = analysis.signal_metrics.get("coding", {})
    research = analysis.signal_metrics.get("research", {})
    if (
        category == "commerce"
        or commerce.get("reached_cart")
        or commerce.get("reached_checkout")
        or commerce.get("reached_payment")
        or _task_has_url_keyword(analysis, "cart", "checkout", "payment")
    ):
        return {
            "domain": "commerce",
            "label": analysis.site_name or "commerce",
            "stage_order": ["Started", "Cart", "Checkout", "Payment"],
            "flags": _commerce_flow_flags(task, analysis),
            "final_url": analysis.final_url,
        }
    if (
        category == "coding"
        or coding.get("file_operations", 0) > 0
        or _task_has_tool_keyword(task, "edit", "patch", "write", "test", "lint", "build")
    ):
        return {
            "domain": "coding",
            "label": "coding",
            "stage_order": ["Started", "Edit", "Test", "Fix", "Verify"],
            "flags": _coding_flow_flags(task, analysis),
            "final_url": "",
        }
    if (
        category == "research"
        or research.get("search_count", 0) > 0
        or research.get("read_count", 0) > 0
        or _task_has_tool_keyword(task, "search", "read", "respond")
    ):
        return {
            "domain": "research",
            "label": "research",
            "stage_order": ["Started", "Search", "Read", "Synthesize"],
            "flags": _research_flow_flags(task, analysis),
            "final_url": analysis.final_url,
        }
    if _has_browser_steps(task):
        return {
            "domain": "browser",
            "label": analysis.site_name or "browser",
            "stage_order": ["Started", "Browse", "Form", "Complete"],
            "flags": _browser_flow_flags(task, analysis),
            "final_url": analysis.final_url,
        }
    return None


def _flow_summary(analyses: dict[str, TaskAnalysis]) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    for analysis in analyses.values():
        detected = _detect_flow_group(analysis)
        if detected is None:
            continue
        key = f"{detected['domain']}::{detected['label']}"
        group = groups.setdefault(
            key,
            {
                "label": detected["label"],
                "domain": detected["domain"],
                "tasks": 0,
                "stage_order": list(detected["stage_order"]),
                "stages": {stage: 0 for stage in detected["stage_order"]},
                "spins": 0,
                "avg_steps": 0.0,
                "final_urls": Counter(),
            },
        )
        group["tasks"] += 1
        group["avg_steps"] += analysis.step_count
        if analysis.is_spin:
            group["spins"] += 1
        for stage in detected["stage_order"]:
            if detected["flags"].get(stage):
                group["stages"][stage] += 1
        if detected["final_url"]:
            group["final_urls"][detected["final_url"]] += 1
    ordered = sorted(
        groups.values(), key=lambda item: (-item["tasks"], item["domain"], item["label"])
    )
    for group in ordered:
        if group["tasks"]:
            group["avg_steps"] = round(group["avg_steps"] / group["tasks"], 1)
        group["final_urls"] = dict(group["final_urls"].most_common(3))
    legacy_sites = {
        group["label"]: {
            "domain": group["domain"],
            "tasks": group["tasks"],
            "stages": dict(group["stages"]),
            "avg_steps": group["avg_steps"],
            "spins": group["spins"],
        }
        for group in ordered
    }
    return {"sites": legacy_sites, "groups": ordered}


def report_flows(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render detected flows as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report showing domain-specific flow progression.
    """
    _ = tasks
    data = _flow_summary(analyses)
    lines = ["FLOW ANALYSIS", "=" * 60]
    if not data["groups"]:
        lines.append("  No detectable flows found.")
        return "\n".join(lines)

    for group in data["groups"]:
        counts = "   ".join(f"{group['stages'][stage]:>8d}" for stage in group["stage_order"])
        rates = "   ".join(
            f"{_pct(group['stages'][stage], group['tasks']):>7d}%" for stage in group["stage_order"]
        )
        lines.extend(
            [
                "",
                f"  {group['label']} [{group['domain']}] ({group['tasks']} tasks)",
                f"    {' -> '.join(group['stage_order'])}",
                f"    {counts}",
                f"    {rates}",
                f"    spins={group['spins']}  avg_steps={group['avg_steps']}",
            ]
        )

        if group["final_urls"]:
            lines.append("    Common endpoints:")
            for url, count in group["final_urls"].items():
                lines.append(f"      {url} ({count})")

    return "\n".join(lines)


def report_flows_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured domain-neutral flow data.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing grouped domain-neutral flow summaries.
    """
    _ = tasks
    return _flow_summary(analyses)


def report_flows_markdown(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render detected flows as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown flow report showing domain-specific progression.
    """
    _ = tasks
    data = _flow_summary(analyses)
    if not data["groups"]:
        return "## Flow Analysis\n\nNo detectable flows found."
    summary_rows = [
        [group["label"], group["domain"], group["tasks"], group["spins"], group["avg_steps"]]
        for group in data["groups"]
    ]
    parts = [
        "## Flow Analysis",
        "",
        _markdown_table(["Group", "Domain", "Tasks", "Spins", "Avg Steps"], summary_rows),
    ]
    for group in data["groups"]:
        stage_rows = [
            [stage, group["stages"][stage], f"{_pct(group['stages'][stage], group['tasks'])}%"]
            for stage in group["stage_order"]
        ]
        parts.extend(
            [
                "",
                f"## {group['label']} [{group['domain']}]",
                "",
                _markdown_table(["Stage", "Reached", "Reach %"], stage_rows),
            ]
        )
    return "\n".join(parts)


# â”€â”€ Outcome Distribution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_outcomes(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render outcome distribution and outcome-versus-grade summaries.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report describing outcome distribution and grade cross-tabs.
    """
    grade_by_id = {g.task_id: g for g in grades}
    outcome_counts: Counter[str] = Counter()
    with_text = 0
    with_outcome = 0
    prompt_variants: Counter[str] = Counter()
    grade_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)

    for task in tasks:
        outcome = task.outcome.status if task.outcome else "unknown"
        outcome_counts[outcome] += 1
        if task.task_text:
            with_text += 1
        if task.outcome:
            with_outcome += 1
        variant = task.metadata.get("prompt_variant", "unknown")
        if isinstance(variant, str):
            prompt_variants[variant] += 1
        grade = grade_by_id[task.task_id].grade
        grade_by_outcome[outcome][grade] += 1

    total = len(tasks)
    lines = ["OUTCOME DISTRIBUTION", "=" * 60, ""]
    for outcome, count in outcome_counts.most_common():
        lines.append(f"  {outcome:20s} {count:4d} ({round(count * 100 / max(1, total))}%)")

    lines.extend(
        [
            "",
            f"  Data coverage: {with_text}/{total} have user_text, {with_outcome}/{total} have outcome",
        ]
    )

    if len(prompt_variants) > 1 or (len(prompt_variants) == 1 and "unknown" not in prompt_variants):
        lines.append(f"  Prompt variants: {dict(prompt_variants.most_common())}")

    # Cross-tab: Outcome x Grade
    lines.extend(
        ["", "OUTCOME x GRADE:", f"  {'Outcome':20s} " + " ".join(f"{g:>7s}" for g in GRADE_LABELS)]
    )
    for outcome in sorted(grade_by_outcome):
        row = grade_by_outcome[outcome]
        cells = " ".join(f"{row.get(g, 0):7d}" for g in GRADE_LABELS)
        lines.append(f"  {outcome:20s} {cells}")

    return "\n".join(lines)


def report_outcomes_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured outcome-distribution data.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary with outcome counts and grade cross-tabs.
    """
    grade_by_id = {g.task_id: g for g in grades}
    outcome_counts: Counter[str] = Counter()
    grade_by_outcome: dict[str, Counter[str]] = defaultdict(Counter)
    for task in tasks:
        outcome = task.outcome.status if task.outcome else "unknown"
        outcome_counts[outcome] += 1
        grade_by_outcome[outcome][grade_by_id[task.task_id].grade] += 1
    return {
        "outcomes": dict(outcome_counts.most_common()),
        "outcome_x_grade": {
            outcome: dict(grade_by_outcome[outcome]) for outcome in sorted(grade_by_outcome)
        },
    }


def report_outcomes_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render outcome distribution as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown outcome-distribution report.
    """
    data = report_outcomes_data(tasks, grades, analyses)
    total = len(tasks)
    outcome_rows = [
        [name, count, f"{_pct(count, total)}%"] for name, count in data["outcomes"].items()
    ]
    cross_rows = [
        [outcome, *[data["outcome_x_grade"][outcome].get(label, 0) for label in GRADE_LABELS]]
        for outcome in sorted(data["outcome_x_grade"])
    ]
    parts = [
        "## Outcome Distribution",
        "",
        _markdown_table(["Outcome", "Count", "Pct"], outcome_rows),
    ]
    if cross_rows:
        parts.extend(
            [
                "",
                "## Outcome x Grade",
                "",
                _markdown_table(["Outcome", *GRADE_LABELS], cross_rows),
            ]
        )
    return "\n".join(parts)


# â”€â”€ Action Items â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_actions(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render prioritized action items inferred from task and tool failures.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report listing operational priorities.
    """
    lines = ["PRIORITIZED ACTION ITEMS", "=" * 60, ""]
    priority = 0

    # 1. Broken tools (>50% error, 10+ calls)
    tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    for task in tasks:
        for step in task.sorted_steps:
            tool_calls[step.tool_name] += 1
            if step.error:
                tool_errors[step.tool_name] += 1
    broken_tools = [
        (name, tool_calls[name], tool_errors[name])
        for name in tool_calls
        if tool_errors.get(name, 0) / max(1, tool_calls[name]) > 0.5 and tool_calls[name] >= 10
    ]
    if broken_tools:
        priority += 1
        lines.append(f"  #{priority} FIX BROKEN TOOLS")
        for name, calls, errs in sorted(broken_tools, key=lambda x: -x[2]):
            lines.append(f"    {name}: {errs}/{calls} errors ({round(errs * 100 / calls)}%)")
        lines.append("")

    # 2. Spin epidemic (tool causing 10+ task spins)
    spin_tools: Counter[str] = Counter()
    for a in analyses.values():
        if a.max_repeat_count >= 5:
            spin_tools[a.max_repeat_tool] += 1
    spin_epidemics = [(tool, count) for tool, count in spin_tools.items() if count >= 3]
    if spin_epidemics:
        priority += 1
        lines.append(f"  #{priority} SPIN EPIDEMIC")
        for tool, count in sorted(spin_epidemics, key=lambda x: -x[1]):
            lines.append(f"    {tool}: causes spin in {count} tasks")
        lines.append("")

    # 3. Routing gap (steps with 0 tools available)
    routing_tasks = sum(1 for a in analyses.values() if a.no_tools_steps > 0)
    if routing_tasks >= 3:
        priority += 1
        lines.append(
            f"  #{priority} ROUTING GAP: {routing_tasks} tasks had steps with 0 tools available"
        )
        lines.append("")

    # 4. Hallucinated tools
    halluc_tools: Counter[str] = Counter()
    for task in tasks:
        for step in task.sorted_steps:
            if classify_error(step.error) == "unknown_tool":
                # Extract tool name from error if possible
                halluc_tools[step.tool_name] += 1
    frequent_halluc = [(tool, count) for tool, count in halluc_tools.items() if count >= 3]
    if frequent_halluc:
        priority += 1
        lines.append(f"  #{priority} HALLUCINATED TOOLS")
        for tool, count in sorted(frequent_halluc, key=lambda x: -x[1]):
            lines.append(f"    {tool}: called {count} times but doesn't exist")
        lines.append("")

    # 5. Low flow completion
    browser_tasks = [
        a
        for a in analyses.values()
        if any(s.tool_name.startswith(("browser_", "desktop_")) for s in a.task.steps)
    ]
    if browser_tasks:
        payment_reached = sum(
            1 for a in browser_tasks if a.signal_metrics.get("commerce", {}).get("reached_payment")
        )
        pct = round(payment_reached * 100 / max(1, len(browser_tasks)))
        if pct < 10:
            priority += 1
            lines.append(
                f"  #{priority} LOW FLOW COMPLETION: only {payment_reached}/{len(browser_tasks)} "
                f"browser tasks reached payment ({pct}%)"
            )
            lines.append("")

    # 6. Approval blocks
    approval_tasks = sum(1 for a in analyses.values() if a.error_kinds.get("approval_block", 0) > 0)
    if approval_tasks >= 3:
        priority += 1
        lines.append(
            f"  #{priority} APPROVAL BLOCKS: {approval_tasks} tasks blocked by approval policy"
        )
        lines.append("")

    if priority == 0:
        lines.append("  No critical action items found.")
    return "\n".join(lines)


def report_actions_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured action-item data.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing prioritized operational action items.
    """
    items: list[dict[str, Any]] = []
    tool_calls: Counter[str] = Counter()
    tool_errors: Counter[str] = Counter()
    for task in tasks:
        for step in task.sorted_steps:
            tool_calls[step.tool_name] += 1
            if step.error:
                tool_errors[step.tool_name] += 1
    broken_tools = [
        {"tool": n, "calls": tool_calls[n], "errors": tool_errors[n]}
        for n in tool_calls
        if tool_errors.get(n, 0) / max(1, tool_calls[n]) > 0.5 and tool_calls[n] >= 10
    ]
    if broken_tools:
        items.append({"type": "broken_tools", "details": broken_tools})
    spin_tools: Counter[str] = Counter()
    for a in analyses.values():
        if a.max_repeat_count >= 5:
            spin_tools[a.max_repeat_tool] += 1
    epidemics = [{"tool": t, "affected_tasks": c} for t, c in spin_tools.items() if c >= 3]
    if epidemics:
        items.append({"type": "spin_epidemic", "details": epidemics})
    routing_tasks = sum(1 for a in analyses.values() if a.no_tools_steps > 0)
    if routing_tasks >= 3:
        items.append({"type": "routing_gap", "affected_tasks": routing_tasks})
    return {"action_items": items}


def report_actions_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render action items as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown report listing operational priorities.
    """
    data = report_actions_data(tasks, grades, analyses)
    rows = [
        [index, item["type"], item.get("affected_tasks", "-"), item.get("details", "-")]
        for index, item in enumerate(data["action_items"], start=1)
    ]
    return "\n".join(
        [
            "## Prioritized Action Items",
            "",
            _markdown_table(["Priority", "Type", "Affected", "Details"], rows),
        ]
    )


# Ã¢â€â‚¬Ã¢â€â‚¬ Cost Analysis Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def _cost_summary(tasks: list[AgentTask], analyses: dict[str, TaskAnalysis]) -> dict[str, Any]:
    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    total_steps = 0
    task_rows: list[dict[str, Any]] = []
    by_model: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tasks": set(), "steps": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    )
    by_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tasks": 0, "steps": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    )
    by_day: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"tasks": 0, "steps": 0, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
    )

    for task in tasks:
        analysis = analyses[task.task_id]
        task_tokens_in = analysis.tokens_in
        task_tokens_out = analysis.tokens_out
        task_cost = analysis.total_cost_usd
        task_steps = analysis.step_count
        total_tokens_in += task_tokens_in
        total_tokens_out += task_tokens_out
        total_cost += task_cost
        total_steps += task_steps

        category = _task_category(task)
        day = task.day or "unknown"
        category_bucket = by_category[category]
        category_bucket["tasks"] += 1
        category_bucket["steps"] += task_steps
        category_bucket["tokens_in"] += task_tokens_in
        category_bucket["tokens_out"] += task_tokens_out
        category_bucket["cost_usd"] += task_cost

        day_bucket = by_day[day]
        day_bucket["tasks"] += 1
        day_bucket["steps"] += task_steps
        day_bucket["tokens_in"] += task_tokens_in
        day_bucket["tokens_out"] += task_tokens_out
        day_bucket["cost_usd"] += task_cost

        task_rows.append(
            {
                "task_id": task.task_id,
                "model": _primary_model(task),
                "models": _task_models(task),
                "category": category,
                "day": day,
                "steps": task_steps,
                "tokens_in": task_tokens_in,
                "tokens_out": task_tokens_out,
                "cost_usd": round(task_cost, 6),
                "cost_per_step": round(task_cost / max(1, task_steps), 6),
            }
        )

        for step in task.sorted_steps:
            model = step.model_name or "unknown"
            bucket = by_model[model]
            bucket["tasks"].add(task.task_id)
            bucket["steps"] += 1
            bucket["tokens_in"] += step.input_tokens or 0
            bucket["tokens_out"] += step.output_tokens or 0
            bucket["cost_usd"] += float(step.cost_usd or 0.0)

    task_rows.sort(key=lambda item: (-item["cost_usd"], item["task_id"]))

    def _finalize(name: str, bucket: dict[str, Any]) -> dict[str, Any]:
        task_count = (
            len(bucket["tasks"]) if isinstance(bucket["tasks"], set) else int(bucket["tasks"])
        )
        return {
            "tasks": task_count,
            "steps": bucket["steps"],
            "tokens_in": bucket["tokens_in"],
            "tokens_out": bucket["tokens_out"],
            "cost_usd": round(bucket["cost_usd"], 6),
            "avg_cost_per_task": round(bucket["cost_usd"] / max(1, task_count), 6),
            "avg_cost_per_step": round(bucket["cost_usd"] / max(1, bucket["steps"]), 6),
            "name": name,
        }

    return {
        "summary": {
            "tasks": len(tasks),
            "steps": total_steps,
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "cost_usd": round(total_cost, 6),
            "avg_cost_per_task": round(total_cost / max(1, len(tasks)), 6),
            "avg_cost_per_step": round(total_cost / max(1, total_steps), 6),
        },
        "by_model": [_finalize(name, bucket) for name, bucket in sorted(by_model.items())],
        "by_category": [_finalize(name, bucket) for name, bucket in sorted(by_category.items())],
        "by_day": [_finalize(name, bucket) for name, bucket in sorted(by_day.items())],
        "most_expensive_tasks": task_rows[:10],
    }


def report_cost(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render cost analysis as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report describing token and cost distribution.
    """
    data = _cost_summary(tasks, analyses)
    summary = data["summary"]
    lines = ["COST ANALYSIS", "=" * 60, ""]
    lines.extend(
        [
            f"  Total tokens in:  {summary['tokens_in']}",
            f"  Total tokens out: {summary['tokens_out']}",
            f"  Total cost:       {_format_money(summary['cost_usd'])}",
            f"  Cost per task:    {_format_money(summary['avg_cost_per_task'])}",
            f"  Cost per step:    {_format_money(summary['avg_cost_per_step'])}",
        ]
    )

    def _append_group(title: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        table = [
            [
                row["name"],
                row["tasks"],
                row["steps"],
                row["tokens_in"],
                row["tokens_out"],
                _format_money(row["cost_usd"]),
                _format_money(row["avg_cost_per_task"]),
                _format_money(row["avg_cost_per_step"]),
            ]
            for row in rows
        ]
        lines.extend(
            [
                "",
                f"{title}:",
                *_text_table(
                    ["Group", "Tasks", "Steps", "In", "Out", "Cost", "Cost/Task", "Cost/Step"],
                    table,
                ),
            ]
        )

    _append_group("BY MODEL", data["by_model"])
    _append_group("BY CATEGORY", data["by_category"])
    _append_group("BY DAY", data["by_day"])

    task_rows = [
        [
            item["task_id"],
            item["model"],
            item["category"],
            item["day"],
            item["steps"],
            item["tokens_in"],
            item["tokens_out"],
            _format_money(item["cost_usd"]),
            _format_money(item["cost_per_step"]),
        ]
        for item in data["most_expensive_tasks"]
    ]
    if task_rows:
        lines.extend(
            [
                "",
                "MOST EXPENSIVE TASKS:",
                *_text_table(
                    ["Task", "Model", "Category", "Day", "Steps", "In", "Out", "Cost", "Cost/Step"],
                    task_rows,
                ),
            ]
        )
    return "\n".join(lines)


def report_cost_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured cost-analysis data.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing task-level and grouped token/cost metrics.
    """
    return _cost_summary(tasks, analyses)


def report_cost_markdown(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render cost analysis as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown cost-analysis report.
    """
    data = _cost_summary(tasks, analyses)
    summary = data["summary"]
    parts = [
        "## Cost Analysis",
        "",
        _markdown_table(
            ["Metric", "Value"],
            [
                ["Total tokens in", summary["tokens_in"]],
                ["Total tokens out", summary["tokens_out"]],
                ["Total cost", _format_money(summary["cost_usd"])],
                ["Cost per task", _format_money(summary["avg_cost_per_task"])],
                ["Cost per step", _format_money(summary["avg_cost_per_step"])],
            ],
        ),
    ]
    for title, rows in (
        ("By Model", data["by_model"]),
        ("By Category", data["by_category"]),
        ("By Day", data["by_day"]),
    ):
        parts.extend(
            [
                "",
                f"## {title}",
                "",
                _markdown_table(
                    ["Group", "Tasks", "Steps", "In", "Out", "Cost", "Cost/Task", "Cost/Step"],
                    [
                        [
                            row["name"],
                            row["tasks"],
                            row["steps"],
                            row["tokens_in"],
                            row["tokens_out"],
                            _format_money(row["cost_usd"]),
                            _format_money(row["avg_cost_per_task"]),
                            _format_money(row["avg_cost_per_step"]),
                        ]
                        for row in rows
                    ],
                ),
            ]
        )
    parts.extend(
        [
            "",
            "## Most Expensive Tasks",
            "",
            _markdown_table(
                ["Task", "Model", "Category", "Day", "Steps", "In", "Out", "Cost", "Cost/Step"],
                [
                    [
                        item["task_id"],
                        item["model"],
                        item["category"],
                        item["day"],
                        item["steps"],
                        item["tokens_in"],
                        item["tokens_out"],
                        _format_money(item["cost_usd"]),
                        _format_money(item["cost_per_step"]),
                    ]
                    for item in data["most_expensive_tasks"]
                ],
            ),
        ]
    )
    return "\n".join(parts)


# Ã¢â€â‚¬Ã¢â€â‚¬ Fix Plan Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬Ã¢â€â‚¬


def _fix_plan_summary(tasks: list[AgentTask], grades: list[GradeResult]) -> dict[str, Any]:
    failures = classify_failures(tasks, grades)
    plan = build_fix_plan(failures)
    failures_by_cause: dict[str, list[Any]] = defaultdict(list)
    for failure in failures:
        failures_by_cause[failure.root_cause].append(failure)
    fixes = []
    for entry in plan:
        cause_meta = ROOT_CAUSES.get(entry.root_cause, {})
        evidence: list[str] = []
        for failure in failures_by_cause.get(entry.root_cause, []):
            for item in failure.evidence:
                if item not in evidence:
                    evidence.append(item)
        fixes.append(
            {
                "priority": entry.priority,
                "root_cause": entry.root_cause,
                "root_cause_label": cause_meta.get("label", entry.root_cause),
                "affected_tasks": sorted(
                    item.task_id for item in failures_by_cause.get(entry.root_cause, [])
                ),
                "affected_count": entry.count,
                "impact_score": entry.impact,
                "investigate_task": entry.investigate_task,
                "targets": list(entry.targets),
                "fix_hint": entry.fix_hint,
                "evidence": evidence[:5] or list(entry.evidence),
            }
        )
    return {"count": len(fixes), "failures_considered": len(failures), "fixes": fixes}


def report_fixes(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render the prioritized fix plan as terminal text.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report describing the prioritized fix plan.
    """
    _ = analyses
    data = _fix_plan_summary(tasks, grades)
    lines = ["FIX PLAN REPORT", "=" * 60, ""]
    if not data["fixes"]:
        lines.append("  No BROKEN or WEAK tasks found. Nothing to diagnose.")
        return "\n".join(lines)
    for item in data["fixes"]:
        lines.extend(
            [
                f"Priority #{item['priority']}: {item['root_cause']} ({item['affected_count']} task(s), impact={item['impact_score']})",
                f"  Targets: {', '.join(item['targets'])}",
                f"  Fix hint: {item['fix_hint']}",
                f"  Investigate task: {item['investigate_task']}",
                f"  Affected tasks: {', '.join(item['affected_tasks'])}",
                f"  Evidence: {'; '.join(item['evidence'])}"
                if item["evidence"]
                else "  Evidence: none",
                "",
            ]
        )
    return "\n".join(lines)


def report_fixes_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured fix-plan data.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing prioritized root-cause fixes.
    """
    _ = analyses
    return _fix_plan_summary(tasks, grades)


def report_fixes_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render the prioritized fix plan as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown report describing the prioritized fix plan.
    """
    _ = analyses
    data = _fix_plan_summary(tasks, grades)
    if not data["fixes"]:
        return "## Fix Plan Report\n\nNo BROKEN or WEAK tasks found. Nothing to diagnose."
    summary_rows = [
        [
            item["priority"],
            item["root_cause"],
            item["affected_count"],
            item["impact_score"],
            item["investigate_task"],
        ]
        for item in data["fixes"]
    ]
    parts = [
        "## Fix Plan Report",
        "",
        _markdown_table(
            ["Priority", "Root Cause", "Affected Tasks", "Impact Score", "Investigate Task"],
            summary_rows,
        ),
    ]
    for item in data["fixes"]:
        parts.extend(
            [
                "",
                f"## Priority {item['priority']}: {item['root_cause']}",
                "",
                _markdown_table(
                    ["Field", "Value"],
                    [
                        ["Root cause", item["root_cause_label"]],
                        ["Affected tasks", ", ".join(item["affected_tasks"]) or "-"],
                        ["Impact score", item["impact_score"]],
                        ["Targets", ", ".join(item["targets"])],
                        ["Fix hint", item["fix_hint"]],
                    ],
                ),
                "",
                "```text",
                "\n".join(item["evidence"]) if item["evidence"] else "none",
                "```",
            ]
        )
    return "\n".join(parts)


# â”€â”€ Day Comparison â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_compare_days(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    day1: str,
    day2: str,
) -> str:
    """Render a side-by-side comparison between two day buckets.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.
        day1: First day in ``YYYYMMDD`` form.
        day2: Second day in ``YYYYMMDD`` form.

    Returns:
        A text comparison report with deltas between the two days.
    """
    grade_by_id = {g.task_id: g for g in grades}

    def _day_stats(day: str) -> dict[str, Any]:
        day_tasks = [t for t in tasks if t.day == day]
        day_grades = [grade_by_id[t.task_id] for t in day_tasks if t.task_id in grade_by_id]
        day_analyses = [analyses[t.task_id] for t in day_tasks if t.task_id in analyses]
        dist = Counter(g.grade for g in day_grades)
        total = len(day_tasks)
        golden_good = dist.get("GOLDEN", 0) + dist.get("GOOD", 0)
        broken = dist.get("BROKEN", 0)
        spins = sum(1 for a in day_analyses if a.is_spin)
        payment = sum(
            1 for a in day_analyses if a.signal_metrics.get("commerce", {}).get("reached_payment")
        )
        avg_steps = sum(a.step_count for a in day_analyses) / max(1, total)
        return {
            "total": total,
            "golden_good_pct": round(golden_good * 100 / max(1, total)),
            "broken_pct": round(broken * 100 / max(1, total)),
            "spin_pct": round(spins * 100 / max(1, total)),
            "payment_pct": round(payment * 100 / max(1, total)),
            "avg_steps": round(avg_steps, 1),
        }

    s1 = _day_stats(day1)
    s2 = _day_stats(day2)

    def _delta_arrow(v1: float | int, v2: float | int, higher_is_better: bool = True) -> str:
        diff = v2 - v1
        if diff == 0:
            return "="
        arrow = "^" if diff > 0 else "v"
        good = (diff > 0) == higher_is_better
        marker = "OK" if good else "!!"
        return f"{arrow}{abs(diff)} {marker}"

    lines = [
        f"DAY COMPARISON: {day1} vs {day2}",
        "=" * 60,
        "",
        f"  {'Metric':20s} {day1:>12s} {day2:>12s} {'Delta':>12s}",
        "  " + "-" * 56,
        f"  {'Tasks':20s} {s1['total']:>12d} {s2['total']:>12d}",
        f"  {'Golden+Good%':20s} {s1['golden_good_pct']:>11d}% {s2['golden_good_pct']:>11d}% {_delta_arrow(s1['golden_good_pct'], s2['golden_good_pct']):>12s}",
        f"  {'Broken%':20s} {s1['broken_pct']:>11d}% {s2['broken_pct']:>11d}% {_delta_arrow(s1['broken_pct'], s2['broken_pct'], higher_is_better=False):>12s}",
        f"  {'Spin%':20s} {s1['spin_pct']:>11d}% {s2['spin_pct']:>11d}% {_delta_arrow(s1['spin_pct'], s2['spin_pct'], higher_is_better=False):>12s}",
        f"  {'Payment reach%':20s} {s1['payment_pct']:>11d}% {s2['payment_pct']:>11d}% {_delta_arrow(s1['payment_pct'], s2['payment_pct']):>12s}",
        f"  {'Avg steps':20s} {s1['avg_steps']:>12.1f} {s2['avg_steps']:>12.1f}",
    ]
    return "\n".join(lines)


def report_compare_days_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    day1: str,
    day2: str,
) -> dict[str, Any]:
    """Return structured comparison data for two day buckets.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.
        day1: First day in ``YYYYMMDD`` form.
        day2: Second day in ``YYYYMMDD`` form.

    Returns:
        A dictionary containing per-day metrics for comparison.
    """
    grade_by_id = {g.task_id: g for g in grades}

    def _day_stats(day: str) -> dict[str, Any]:
        day_tasks = [t for t in tasks if t.day == day]
        day_grades = [grade_by_id[t.task_id] for t in day_tasks if t.task_id in grade_by_id]
        day_analyses = [analyses[t.task_id] for t in day_tasks if t.task_id in analyses]
        dist = Counter(g.grade for g in day_grades)
        total = len(day_tasks)
        golden_good = dist.get("GOLDEN", 0) + dist.get("GOOD", 0)
        broken = dist.get("BROKEN", 0)
        return {
            "total": total,
            "golden_good_pct": round(golden_good * 100 / max(1, total)),
            "broken_pct": round(broken * 100 / max(1, total)),
            "spin_pct": round(sum(1 for a in day_analyses if a.is_spin) * 100 / max(1, total)),
            "avg_steps": round(sum(a.step_count for a in day_analyses) / max(1, total), 1),
        }

    return {"day1": {day1: _day_stats(day1)}, "day2": {day2: _day_stats(day2)}}


def report_compare_days_markdown(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    day1: str,
    day2: str,
) -> str:
    """Render a day-over-day comparison as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        grades: Grade results aligned to ``tasks``.
        analyses: Task analyses keyed by task id.
        day1: First day in ``YYYYMMDD`` form.
        day2: Second day in ``YYYYMMDD`` form.

    Returns:
        A Markdown comparison report with deltas between the two days.
    """
    data = report_compare_days_data(tasks, grades, analyses, day1, day2)
    s1 = data["day1"][day1]
    s2 = data["day2"][day2]
    rows = [
        ["Tasks", s1["total"], s2["total"], "-"],
        [
            "Golden+Good%",
            f"{s1['golden_good_pct']}%",
            f"{s2['golden_good_pct']}%",
            _delta_arrow(s1["golden_good_pct"], s2["golden_good_pct"]),
        ],
        [
            "Broken%",
            f"{s1['broken_pct']}%",
            f"{s2['broken_pct']}%",
            _delta_arrow(s1["broken_pct"], s2["broken_pct"], higher_is_better=False),
        ],
        [
            "Spin%",
            f"{s1['spin_pct']}%",
            f"{s2['spin_pct']}%",
            _delta_arrow(s1["spin_pct"], s2["spin_pct"], higher_is_better=False),
        ],
        ["Avg steps", s1["avg_steps"], s2["avg_steps"], "-"],
    ]
    return "\n".join(
        [
            f"## Day Comparison: {day1} vs {day2}",
            "",
            _markdown_table(["Metric", day1, day2, "Delta"], rows),
        ]
    )


# â”€â”€ Coding Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_coding(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render a report focused on coding-oriented task behavior.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report summarizing coding-task behavior.
    """
    coding_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("coding", {}).get("file_operations", 0) > 0
    ]
    lines = [f"CODING REPORT ({len(coding_tasks)} tasks)", "=" * 60, ""]
    if not coding_tasks:
        lines.append("  No coding tasks found.")
        return "\n".join(lines)

    total_files = 0
    total_tests = 0
    total_lints = 0
    total_git = 0
    verify_cycle = 0
    for _, analysis in coding_tasks:
        cm = analysis.signal_metrics.get("coding", {})
        total_files += cm.get("file_operations", 0)
        total_tests += cm.get("test_runs", 0)
        total_lints += cm.get("lint_runs", 0)
        total_git += cm.get("git_operations", 0)
        if cm.get("has_test_verify_cycle"):
            verify_cycle += 1

    n = len(coding_tasks)
    lines.extend(
        [
            f"  File operations: {total_files} (avg {total_files / n:.1f}/task)",
            f"  Test runs:       {total_tests} (avg {total_tests / n:.1f}/task)",
            f"  Lint runs:       {total_lints}",
            f"  Git operations:  {total_git}",
            f"  Edit+test cycle: {verify_cycle}/{n} tasks ({round(verify_cycle * 100 / n)}%)",
            "",
            "Tasks:",
        ]
    )
    for task, analysis in sorted(coding_tasks, key=lambda x: -x[1].step_count):
        cm = analysis.signal_metrics.get("coding", {})
        lines.append(
            f"  {task.task_id[:24]:24s} files={cm.get('unique_files_touched', 0)} "
            f"tests={cm.get('test_runs', 0)} errs={analysis.errors} "
            f"steps={analysis.step_count}"
        )
    return "\n".join(lines)


def report_coding_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured coding-task report data.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing per-task coding metrics.
    """
    coding_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("coding", {}).get("file_operations", 0) > 0
    ]
    items = []
    for task, analysis in coding_tasks:
        cm = analysis.signal_metrics.get("coding", {})
        items.append(
            {
                "task_id": task.task_id,
                "file_operations": cm.get("file_operations", 0),
                "test_runs": cm.get("test_runs", 0),
                "lint_runs": cm.get("lint_runs", 0),
                "git_operations": cm.get("git_operations", 0),
                "unique_files": cm.get("unique_files_touched", 0),
                "has_verify_cycle": cm.get("has_test_verify_cycle", False),
                "errors": analysis.errors,
                "steps": analysis.step_count,
            }
        )
    return {"count": len(items), "tasks": items}


def report_coding_markdown(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render coding-task behavior as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown coding-task report.
    """
    data = report_coding_data(tasks, analyses)
    rows = [
        [
            item["task_id"],
            item["file_operations"],
            item["test_runs"],
            item["lint_runs"],
            item["git_operations"],
            item["unique_files"],
            item["has_verify_cycle"],
            item["errors"],
            item["steps"],
        ]
        for item in data["tasks"]
    ]
    return "\n".join(
        [
            "## Coding Report",
            "",
            _markdown_table(
                [
                    "Task",
                    "File Ops",
                    "Tests",
                    "Lints",
                    "Git",
                    "Unique Files",
                    "Verify Cycle",
                    "Errors",
                    "Steps",
                ],
                rows,
            ),
        ]
    )


# â”€â”€ Research Report â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€


def report_research(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render a report focused on research-oriented task behavior.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A text report summarizing research-task behavior.
    """
    research_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("research", {}).get("search_count", 0) > 0
    ]
    lines = [f"RESEARCH REPORT ({len(research_tasks)} tasks)", "=" * 60, ""]
    if not research_tasks:
        lines.append("  No research tasks found.")
        return "\n".join(lines)

    total_searches = 0
    total_reads = 0
    total_citations = 0
    total_domains = 0
    has_synthesis = 0
    for _, analysis in research_tasks:
        rm = analysis.signal_metrics.get("research", {})
        total_searches += rm.get("search_count", 0)
        total_reads += rm.get("read_count", 0)
        total_citations += rm.get("citation_count", 0)
        total_domains += rm.get("source_diversity", 0)
        if rm.get("has_synthesis_step"):
            has_synthesis += 1

    n = len(research_tasks)
    lines.extend(
        [
            f"  Searches:         {total_searches} (avg {total_searches / n:.1f}/task)",
            f"  Reads:            {total_reads} (avg {total_reads / n:.1f}/task)",
            f"  Citations:        {total_citations}",
            f"  Unique domains:   {total_domains} (avg {total_domains / n:.1f}/task)",
            f"  Has synthesis:    {has_synthesis}/{n} ({round(has_synthesis * 100 / n)}%)",
            "",
            "Tasks:",
        ]
    )
    for task, analysis in sorted(research_tasks, key=lambda x: -x[1].step_count):
        rm = analysis.signal_metrics.get("research", {})
        lines.append(
            f"  {task.task_id[:24]:24s} searches={rm.get('search_count', 0)} "
            f"reads={rm.get('read_count', 0)} domains={rm.get('source_diversity', 0)} "
            f"citations={rm.get('citation_count', 0)}"
        )
    return "\n".join(lines)


def report_research_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured research-task report data.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A dictionary containing per-task research metrics.
    """
    research_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("research", {}).get("search_count", 0) > 0
    ]
    items = []
    for task, analysis in research_tasks:
        rm = analysis.signal_metrics.get("research", {})
        items.append(
            {
                "task_id": task.task_id,
                "search_count": rm.get("search_count", 0),
                "read_count": rm.get("read_count", 0),
                "source_diversity": rm.get("source_diversity", 0),
                "citation_count": rm.get("citation_count", 0),
                "has_synthesis": rm.get("has_synthesis_step", False),
            }
        )
    return {"count": len(items), "tasks": items}


def report_research_markdown(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render research-task behavior as GitHub-flavored Markdown.

    Args:
        tasks: Loaded tasks included in the report window.
        analyses: Task analyses keyed by task id.

    Returns:
        A Markdown research-task report.
    """
    data = report_research_data(tasks, analyses)
    rows = [
        [
            item["task_id"],
            item["search_count"],
            item["read_count"],
            item["source_diversity"],
            item["citation_count"],
            item["has_synthesis"],
        ]
        for item in data["tasks"]
    ]
    return "\n".join(
        [
            "## Research Report",
            "",
            _markdown_table(
                ["Task", "Searches", "Reads", "Domains", "Citations", "Synthesis"], rows
            ),
        ]
    )
