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

from collections import Counter, defaultdict
from typing import Any

from .analyzer import TaskAnalysis, classify_error
from .grader import GradeResult
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


# ── Health Dashboard ─────────────────────────────────────────────────


def report_health(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
    dist = _grade_distribution(grades)
    total = len(grades)
    lines = ["HEALTH DASHBOARD", "=" * 60, ""]

    # Grade distribution bar chart
    for label in GRADE_LABELS:
        count = dist[label]
        pct = round(count * 100 / max(1, total))
        lines.append(f"  {label:7s} {_bar(count, total)} {count:3d} ({pct}%)")
    lines.append(f"\n  Total: {total} tasks")

    # Key rates
    error_tasks = sum(1 for a in analyses.values() if a.errors > 0)
    spin_tasks = sum(1 for a in analyses.values() if a.is_spin)
    timeout_tasks = sum(1 for a in analyses.values() if a.timeout_like)
    halluc_tasks = sum(1 for a in analyses.values() if a.hallucinated_tools > 0)
    approval_tasks = sum(1 for a in analyses.values() if a.error_kinds.get("approval_block", 0) > 0)
    lines.extend([
        "",
        f"  Error tasks: {error_tasks}/{total} ({round(error_tasks * 100 / max(1, total))}%)",
        f"  Spin tasks:  {spin_tasks}/{total} ({round(spin_tasks * 100 / max(1, total))}%)",
        f"  Timeouts:    {timeout_tasks}/{total}",
        f"  Hallucinations: {halluc_tasks}/{total}",
        f"  Approval blocks: {approval_tasks}/{total}",
    ])

    # Day-over-day trends
    by_day: dict[str, list[GradeResult]] = defaultdict(list)
    for task, grade in zip(tasks, grades, strict=False):
        day = task.day or "unknown"
        by_day[day].append(grade)
    if len(by_day) > 1:
        lines.extend(["", "DAY TRENDS:", f"  {'Day':12s} {'Tasks':>5s} {'GOLDEN':>7s} {'GOOD':>5s} {'BROKEN':>7s} {'Pass%':>6s}"])
        for day in sorted(by_day):
            day_grades = by_day[day]
            day_total = len(day_grades)
            golden = sum(1 for g in day_grades if g.grade == "GOLDEN")
            good = sum(1 for g in day_grades if g.grade == "GOOD")
            broken = sum(1 for g in day_grades if g.grade == "BROKEN")
            pass_pct = round((golden + good) * 100 / max(1, day_total))
            lines.append(f"  {day:12s} {day_total:5d} {golden:7d} {good:5d} {broken:7d} {pass_pct:5d}%")

    return "\n".join(lines)


def report_health_data(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> dict[str, Any]:
    dist = _grade_distribution(grades)
    total = len(grades)
    error_tasks = sum(1 for a in analyses.values() if a.errors > 0)
    spin_tasks = sum(1 for a in analyses.values() if a.is_spin)
    timeout_tasks = sum(1 for a in analyses.values() if a.timeout_like)
    halluc_tasks = sum(1 for a in analyses.values() if a.hallucinated_tools > 0)
    approval_tasks = sum(1 for a in analyses.values() if a.error_kinds.get("approval_block", 0) > 0)
    by_day: dict[str, list[GradeResult]] = defaultdict(list)
    for task, grade in zip(tasks, grades, strict=False):
        by_day[task.day or "unknown"].append(grade)
    day_trends = {}
    for day in sorted(by_day):
        dg = by_day[day]
        dt = len(dg)
        golden = sum(1 for g in dg if g.grade == "GOLDEN")
        good = sum(1 for g in dg if g.grade == "GOOD")
        broken = sum(1 for g in dg if g.grade == "BROKEN")
        day_trends[day] = {
            "tasks": dt, "golden": golden, "good": good, "broken": broken,
            "pass_pct": round((golden + good) * 100 / max(1, dt)),
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


# ── Golden Report ────────────────────────────────────────────────────


def report_golden(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
    *,
    min_steps: int = 0,
) -> str:
    grade_by_id = {g.task_id: g for g in grades}
    golden_good = [
        (t, grade_by_id[t.task_id], analyses[t.task_id])
        for t in tasks
        if grade_by_id[t.task_id].grade in {"GOLDEN", "GOOD"}
        and analyses[t.task_id].step_count >= min_steps
    ]
    golden_good.sort(key=lambda x: -x[1].score)
    lines = [f"GOLDEN/GOOD RUNS ({len(golden_good)} tasks)", "=" * 60, ""]
    for task, grade, analysis in golden_good:
        commerce = analysis.signal_metrics.get("commerce", {})
        milestones = []
        if commerce.get("reached_payment"):
            milestones.append("PAY")
        if commerce.get("reached_checkout"):
            milestones.append("CKOUT")
        if commerce.get("reached_cart"):
            milestones.append("CART")
        milestone_str = ",".join(milestones) if milestones else "-"
        lines.append(
            f"  [{grade.grade:6s} {grade.score:+3d}] {task.task_id[:20]:20s} "
            f"steps={analysis.step_count:2d} urls={len(analysis.unique_urls):2d} "
            f"fills={commerce.get('fill_count', 0)} site={analysis.site_name} "
            f"flow={milestone_str}"
        )
    if golden_good:
        avg_steps = sum(a.step_count for _, _, a in golden_good) / len(golden_good)
        avg_urls = sum(len(a.unique_urls) for _, _, a in golden_good) / len(golden_good)
        site_counts: Counter[str] = Counter(a.site_name for _, _, a in golden_good)
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
    grade_by_id = {g.task_id: g for g in grades}
    golden_good = [
        (t, grade_by_id[t.task_id], analyses[t.task_id])
        for t in tasks
        if grade_by_id[t.task_id].grade in {"GOLDEN", "GOOD"}
    ]
    golden_good.sort(key=lambda x: -x[1].score)
    items = []
    for task, grade, analysis in golden_good:
        commerce = analysis.signal_metrics.get("commerce", {})
        items.append({
            "task_id": task.task_id,
            "grade": grade.grade,
            "score": grade.score,
            "steps": analysis.step_count,
            "urls": len(analysis.unique_urls),
            "fills": commerce.get("fill_count", 0),
            "site": analysis.site_name,
        })
    return {"count": len(golden_good), "tasks": items}


# ── Broken Report ────────────────────────────────────────────────────


def report_broken(
    tasks: list[AgentTask],
    grades: list[GradeResult],
    analyses: dict[str, TaskAnalysis],
) -> str:
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


# ── Tool Effectiveness ───────────────────────────────────────────────


def report_tools(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
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


# ── Flow Funnel ──────────────────────────────────────────────────────


def report_flows(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
) -> str:
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
