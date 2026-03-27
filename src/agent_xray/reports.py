"""Report generation module — ports all Viola step-log-analysis report types.

Report types:
  health    — Grade distribution + day-over-day trends
  golden    — GOLDEN/GOOD runs with aggregate stats
  broken    — BROKEN tasks with WHY breakdown
  tools     — Per-tool error rate and effectiveness
  flows     — Commerce flow funnel (Started → Cart → Checkout → Payment)
  outcomes  — Outcome distribution and cross-tab with grades
  actions   — Prioritized action items by impact
  compare   — Day-over-day comparison with deltas
"""

from __future__ import annotations

import re
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
    body = [
        "| " + " | ".join(_markdown_escape(item) for item in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _text_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    widths = [len(str(header)) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(str(cell)))
    header = "  " + "  ".join(
        f"{str(cell):<{widths[index]}}" for index, cell in enumerate(headers)
    )
    divider = "  " + "  ".join("-" * width for width in widths)
    body = [
        "  " + "  ".join(
            f"{str(cell):<{widths[index]}}" for index, cell in enumerate(row)
        )
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


# ── Health Dashboard ─────────────────────────────────────────────────


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

    lines.extend([
        "",
        f"  Error tasks: {data['error_tasks']}/{total} ({_pct(data['error_tasks'], total)}%)",
        f"  Spin tasks:  {data['spin_tasks']}/{total} ({_pct(data['spin_tasks'], total)}%)",
        f"  Timeouts:    {data['timeout_tasks']}/{total}",
        f"  Hallucinations: {data['hallucination_tasks']}/{total}",
        f"  Approval blocks: {data['approval_blocked_tasks']}/{total}",
    ])

    if len(data["day_trends"]) > 1:
        rows = [
            [day, stats["tasks"], stats["golden"], stats["good"], stats["broken"], f"{stats['pass_pct']}%"]
            for day, stats in data["day_trends"].items()
        ]
        lines.extend(["", "DAY TRENDS:", *_text_table(["Day", "Tasks", "GOLDEN", "GOOD", "BROKEN", "Pass%"], rows)])

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
            [day, stats["tasks"], stats["golden"], stats["good"], stats["broken"], f"{stats['pass_pct']}%"]
            for day, stats in data["day_trends"].items()
        ]
        parts.extend([
            "",
            "## Day Trends",
            "",
            _markdown_table(["Day", "Tasks", "GOLDEN", "GOOD", "BROKEN", "Pass%"], day_rows),
        ])
    return "\n".join(parts)


# ── Golden Report ────────────────────────────────────────────────────


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
        items.append({
            "task_id": task.task_id,
            "grade": grade.grade,
            "score": grade.score,
            "steps": analysis.step_count,
            "urls": len(analysis.unique_urls),
            "fills": commerce.get("fill_count", 0),
            "site": analysis.site_name,
            "milestones": milestones,
        })
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
        lines.extend([
            "",
            f"  Avg steps: {avg_steps:.1f}, Avg URLs: {avg_urls:.1f}",
            f"  Sites: {dict(site_counts.most_common(10))}",
        ])
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
    return "\n".join([
        "## Golden / Good Runs",
        "",
        _markdown_table(["Task", "Grade", "Score", "Steps", "URLs", "Fills", "Site", "Flow"], rows),
    ])


# ── Broken Report ────────────────────────────────────────────────────


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
        spin_info = f" spin={analysis.max_repeat_tool}x{analysis.max_repeat_count}" if analysis.is_spin else ""
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
        items.append({
            "task_id": task.task_id,
            "score": grade.score,
            "steps": analysis.step_count,
            "errors": analysis.errors,
            "spin": analysis.is_spin,
            "site": analysis.site_name,
        })
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
    return "\n".join([
        "## Broken Tasks",
        "",
        _markdown_table(["Reason", "Count"], why_rows),
        "",
        "## Worst Tasks",
        "",
        _markdown_table(["Task", "Score", "Steps", "Errors", "Spin", "Site"], task_rows),
    ])


# ── Tool Effectiveness ───────────────────────────────────────────────


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
    lines.append(f"  {'Tool':30s} {'Calls':>6s} {'Errors':>7s} {'Err%':>5s} {'Avg ms':>7s} {'Flag':>8s}")
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
        items.append({
            "tool": tool, "calls": calls, "errors": errs,
            "error_pct": round(errs * 100 / max(1, calls)),
            "avg_ms": avg_ms,
        })
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
    return "\n".join([
        "## Tool Effectiveness",
        "",
        _markdown_table(["Tool", "Calls", "Errors", "Err%", "Avg ms"], rows),
    ])


# ── Flow Funnel ──────────────────────────────────────────────────────


def report_flows(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render a commerce-style funnel report grouped by inferred site."""
    by_site: dict[str, list[TaskAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        if any(step.tool_name.startswith(("browser_", "desktop_")) for step in analysis.task.steps):
            by_site[analysis.site_name].append(analysis)

    lines = ["FLOW FUNNEL ANALYSIS", "=" * 60]
    if not by_site:
        lines.append("  No browser tasks found.")
        return "\n".join(lines)

    for site in sorted(by_site, key=lambda s: -len(by_site[s])):
        site_analyses = by_site[site]
        total = len(site_analyses)
        cart = sum(1 for a in site_analyses if a.signal_metrics.get("commerce", {}).get("reached_cart"))
        checkout = sum(1 for a in site_analyses if a.signal_metrics.get("commerce", {}).get("reached_checkout"))
        payment = sum(1 for a in site_analyses if a.signal_metrics.get("commerce", {}).get("reached_payment"))
        fills = sum(1 for a in site_analyses if a.signal_metrics.get("commerce", {}).get("real_fill_count", 0) > 0)
        spins = sum(1 for a in site_analyses if a.is_spin)
        avg_steps = sum(a.step_count for a in site_analyses) / max(1, total)

        site_total = total  # bind for closure
        pct = lambda n, t=site_total: f"{round(n * 100 / max(1, t))}%"  # noqa: E731
        lines.extend([
            "",
            f"  {site} ({total} tasks)",
            "    Started  -> Cart     -> Checkout -> Payment",
            f"    {total:7d}   {cart:8d}   {checkout:8d}   {payment:7d}",
            f"    {'100%':>7s}   {pct(cart):>8s}   {pct(checkout):>8s}   {pct(payment):>7s}",
            f"    fills={fills}  spins={spins}  avg_steps={avg_steps:.0f}",
        ])

        # Common endpoints (where tasks ended)
        final_urls: Counter[str] = Counter()
        for a in site_analyses:
            if a.final_url:
                final_urls[a.final_url] += 1
        if final_urls:
            lines.append("    Common endpoints:")
            for url, count in final_urls.most_common(3):
                lines.append(f"      {url} ({count})")

    return "\n".join(lines)


def report_flows_data(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured commerce funnel data grouped by inferred site."""
    by_site: dict[str, list[TaskAnalysis]] = defaultdict(list)
    for analysis in analyses.values():
        if any(step.tool_name.startswith(("browser_", "desktop_")) for step in analysis.task.steps):
            by_site[analysis.site_name].append(analysis)
    sites = {}
    for site in sorted(by_site, key=lambda s: -len(by_site[s])):
        sa = by_site[site]
        total = len(sa)
        sites[site] = {
            "tasks": total,
            "cart": sum(1 for a in sa if a.signal_metrics.get("commerce", {}).get("reached_cart")),
            "checkout": sum(1 for a in sa if a.signal_metrics.get("commerce", {}).get("reached_checkout")),
            "payment": sum(1 for a in sa if a.signal_metrics.get("commerce", {}).get("reached_payment")),
            "fills": sum(1 for a in sa if a.signal_metrics.get("commerce", {}).get("real_fill_count", 0) > 0),
            "spins": sum(1 for a in sa if a.is_spin),
            "avg_steps": round(sum(a.step_count for a in sa) / max(1, total), 1),
        }
    return {"sites": sites}


# ── Outcome Distribution ─────────────────────────────────────────────


def report_outcomes(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render outcome distribution and outcome-versus-grade summaries."""
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

    lines.extend([
        "",
        f"  Data coverage: {with_text}/{total} have user_text, {with_outcome}/{total} have outcome",
    ])

    if len(prompt_variants) > 1 or (len(prompt_variants) == 1 and "unknown" not in prompt_variants):
        lines.append(f"  Prompt variants: {dict(prompt_variants.most_common())}")

    # Cross-tab: Outcome x Grade
    lines.extend(["", "OUTCOME x GRADE:", f"  {'Outcome':20s} " + " ".join(f"{g:>7s}" for g in GRADE_LABELS)])
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
    """Return structured outcome distribution and cross-tab data."""
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


# ── Action Items ─────────────────────────────────────────────────────


def report_actions(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render prioritized action items inferred from task and tool failures."""
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
        lines.append(f"  #{priority} ROUTING GAP: {routing_tasks} tasks had steps with 0 tools available")
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
    browser_tasks = [a for a in analyses.values() if any(
        s.tool_name.startswith(("browser_", "desktop_")) for s in a.task.steps
    )]
    if browser_tasks:
        payment_reached = sum(
            1 for a in browser_tasks
            if a.signal_metrics.get("commerce", {}).get("reached_payment")
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
    approval_tasks = sum(
        1 for a in analyses.values()
        if a.error_kinds.get("approval_block", 0) > 0
    )
    if approval_tasks >= 3:
        priority += 1
        lines.append(f"  #{priority} APPROVAL BLOCKS: {approval_tasks} tasks blocked by approval policy")
        lines.append("")

    if priority == 0:
        lines.append("  No critical action items found.")
    return "\n".join(lines)


def report_actions_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    """Return structured prioritized action items inferred from failures."""
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


# ── Day Comparison ───────────────────────────────────────────────────


def report_compare_days(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    day1: str,
    day2: str,
) -> str:
    """Render a side-by-side comparison between two day buckets."""
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
            1 for a in day_analyses
            if a.signal_metrics.get("commerce", {}).get("reached_payment")
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
    """Return structured comparison data for two day buckets."""
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


# ── Coding Report ───────────────────────────────────────────────────


def report_coding(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render a report focused on coding-oriented task behavior."""
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
    lines.extend([
        f"  File operations: {total_files} (avg {total_files / n:.1f}/task)",
        f"  Test runs:       {total_tests} (avg {total_tests / n:.1f}/task)",
        f"  Lint runs:       {total_lints}",
        f"  Git operations:  {total_git}",
        f"  Edit+test cycle: {verify_cycle}/{n} tasks ({round(verify_cycle * 100 / n)}%)",
        "",
        "Tasks:",
    ])
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
    """Return structured data for coding-oriented task behavior."""
    coding_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("coding", {}).get("file_operations", 0) > 0
    ]
    items = []
    for task, analysis in coding_tasks:
        cm = analysis.signal_metrics.get("coding", {})
        items.append({
            "task_id": task.task_id,
            "file_operations": cm.get("file_operations", 0),
            "test_runs": cm.get("test_runs", 0),
            "lint_runs": cm.get("lint_runs", 0),
            "git_operations": cm.get("git_operations", 0),
            "unique_files": cm.get("unique_files_touched", 0),
            "has_verify_cycle": cm.get("has_test_verify_cycle", False),
            "errors": analysis.errors,
            "steps": analysis.step_count,
        })
    return {"count": len(items), "tasks": items}


# ── Research Report ─────────────────────────────────────────────────


def report_research(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
    """Render a report focused on research-oriented task behavior."""
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
    lines.extend([
        f"  Searches:         {total_searches} (avg {total_searches / n:.1f}/task)",
        f"  Reads:            {total_reads} (avg {total_reads / n:.1f}/task)",
        f"  Citations:        {total_citations}",
        f"  Unique domains:   {total_domains} (avg {total_domains / n:.1f}/task)",
        f"  Has synthesis:    {has_synthesis}/{n} ({round(has_synthesis * 100 / n)}%)",
        "",
        "Tasks:",
    ])
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
    """Return structured data for research-oriented task behavior."""
    research_tasks = [
        (t, analyses[t.task_id])
        for t in tasks
        if analyses[t.task_id].signal_metrics.get("research", {}).get("search_count", 0) > 0
    ]
    items = []
    for task, analysis in research_tasks:
        rm = analysis.signal_metrics.get("research", {})
        items.append({
            "task_id": task.task_id,
            "search_count": rm.get("search_count", 0),
            "read_count": rm.get("read_count", 0),
            "source_diversity": rm.get("source_diversity", 0),
            "citation_count": rm.get("citation_count", 0),
            "has_synthesis": rm.get("has_synthesis_step", False),
        })
    return {"count": len(items), "tasks": items}
