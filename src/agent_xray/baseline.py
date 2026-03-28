"""Baseline measurement system -- compare agent performance against minimal-prompt baselines."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analyzer import TaskAnalysis, analyze_task
from .schema import AgentTask


@dataclass(slots=True)
class Baseline:
    """A captured baseline representing minimal-prompt task performance.

    Attributes:
        task_id: Source task identifier used to create the baseline.
        site_name: Normalized site label for matching baselines to tasks.
        user_text: Original user instruction for the baseline task.
        step_count: Number of steps the baseline task took.
        duration_s: Total duration in seconds.
        total_tokens_in: Total input tokens consumed.
        total_tokens_out: Total output tokens consumed.
        cost_usd: Total cost in US dollars.
        error_count: Number of steps with errors.
        milestones: Ordered milestones reached during the baseline run.
        tool_sequence: Ordered list of tools used in the baseline run.
        naked_prompt: The minimal instruction used for the baseline.
    """

    task_id: str
    site_name: str
    user_text: str
    step_count: int
    duration_s: float
    total_tokens_in: int
    total_tokens_out: int
    cost_usd: float
    error_count: int
    milestones: list[str]
    tool_sequence: list[str]
    naked_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "site_name": self.site_name,
            "user_text": self.user_text,
            "step_count": self.step_count,
            "duration_s": self.duration_s,
            "total_tokens_in": self.total_tokens_in,
            "total_tokens_out": self.total_tokens_out,
            "cost_usd": self.cost_usd,
            "error_count": self.error_count,
            "milestones": self.milestones,
            "tool_sequence": self.tool_sequence,
            "naked_prompt": self.naked_prompt,
        }


@dataclass(slots=True)
class OverheadResult:
    """Overhead measurement for one task compared against its site baseline.

    Attributes:
        task_id: Identifier of the measured task.
        site_name: Normalized site label shared with the baseline.
        grade: Grade assigned to the task by the grading ruleset.
        baseline_steps: Step count from the baseline run.
        actual_steps: Step count from the measured task.
        step_overhead_pct: ``(actual - baseline) / baseline * 100``.
        baseline_duration_s: Baseline duration in seconds.
        actual_duration_s: Measured task duration in seconds.
        duration_overhead_pct: Duration overhead percentage.
        baseline_cost: Baseline cost in US dollars.
        actual_cost: Measured task cost in US dollars.
        cost_overhead_pct: Cost overhead percentage.
        baseline_errors: Error count from the baseline run.
        actual_errors: Error count from the measured task.
        success_delta: ``"better"``, ``"same"``, or ``"worse"`` milestone comparison.
        overhead_category: Human label for the overhead band.
        contributing_factors: Human-readable explanations of overhead sources.
    """

    task_id: str
    site_name: str
    grade: str
    baseline_steps: int
    actual_steps: int
    step_overhead_pct: float
    baseline_duration_s: float
    actual_duration_s: float
    duration_overhead_pct: float
    baseline_cost: float
    actual_cost: float
    cost_overhead_pct: float
    baseline_errors: int
    actual_errors: int
    success_delta: str
    overhead_category: str
    contributing_factors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "site_name": self.site_name,
            "grade": self.grade,
            "baseline_steps": self.baseline_steps,
            "actual_steps": self.actual_steps,
            "step_overhead_pct": round(self.step_overhead_pct, 1),
            "duration_overhead_pct": round(self.duration_overhead_pct, 1),
            "cost_overhead_pct": round(self.cost_overhead_pct, 1),
            "overhead_category": self.overhead_category,
            "success_delta": self.success_delta,
            "contributing_factors": self.contributing_factors,
        }


@dataclass(slots=True)
class PromptHashGroup:
    """Aggregated metrics for all tasks sharing the same ``system_prompt_hash``.

    Attributes:
        prompt_hash: The shared system prompt hash value.
        task_count: Number of tasks in this group.
        avg_steps: Average step count across tasks.
        avg_duration_s: Average duration in seconds.
        avg_cost: Average cost in US dollars.
        avg_errors: Average error count.
        golden_rate: Fraction of tasks graded GOLDEN or GOOD.
        broken_rate: Fraction of tasks graded BROKEN.
        avg_overhead_pct: Average step overhead vs baseline (NaN when no baselines match).
        sample_task_ids: Up to 3 representative task identifiers.
    """

    prompt_hash: str
    task_count: int
    avg_steps: float
    avg_duration_s: float
    avg_cost: float
    avg_errors: float
    golden_rate: float
    broken_rate: float
    avg_overhead_pct: float
    sample_task_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_hash": self.prompt_hash,
            "task_count": self.task_count,
            "avg_steps": round(self.avg_steps, 1),
            "avg_duration_s": round(self.avg_duration_s, 1),
            "avg_cost": round(self.avg_cost, 4),
            "avg_errors": round(self.avg_errors, 1),
            "golden_rate": round(self.golden_rate, 2),
            "broken_rate": round(self.broken_rate, 2),
            "avg_overhead_pct": round(self.avg_overhead_pct, 1),
            "sample_task_ids": self.sample_task_ids,
        }


# ---------------------------------------------------------------------------
# Naked prompt generation
# ---------------------------------------------------------------------------

_TOOL_VERB_MAP: dict[str, str] = {
    "browser_navigate": "Go to",
    "browser_click": "Click",
    "browser_fill_ref": "Fill",
    "browser_fill": "Fill",
    "browser_snapshot": "Take a snapshot of the page",
    "browser_wait": "Wait for the page",
    "browser_scroll": "Scroll",
    "browser_select": "Select",
    "browser_back": "Go back",
    "web_search": "Search the web for",
    "read_url": "Read",
    "read_file": "Read",
    "edit_file": "Edit",
    "run_tests": "Run",
    "git_commit": "Commit",
    "respond": "Respond with the result",
}


def _describe_step(step_tool: str, step_input: dict[str, Any]) -> str:
    """Convert a tool call into a minimal imperative sentence."""
    verb = _TOOL_VERB_MAP.get(step_tool, step_tool.replace("_", " ").capitalize())

    if step_tool == "browser_navigate":
        url = step_input.get("url", "")
        return f"{verb} {url}." if url else f"{verb} the target page."

    if step_tool in ("browser_click", "browser_select"):
        ref = step_input.get("ref", "")
        label = step_input.get("label", ref)
        return f"{verb} {label}." if label else f"{verb} the element."

    if step_tool in ("browser_fill_ref", "browser_fill"):
        ref = step_input.get("ref", "")
        text = step_input.get("text", "")
        fields = step_input.get("fields", [])
        parts = []
        if ref:
            parts.append(ref)
        if fields:
            if isinstance(fields, list):
                parts.append(", ".join(str(f) for f in fields))
            else:
                parts.append(str(fields))
        if text:
            parts.append(f"with '{text}'")
        detail = " ".join(parts) if parts else "the form"
        return f"{verb} {detail}."

    if step_tool == "web_search":
        query = step_input.get("query", "")
        return f"{verb} '{query}'." if query else f"{verb} the topic."

    if step_tool in ("read_url", "read_file"):
        target = step_input.get("url") or step_input.get("path") or ""
        return f"{verb} {target}." if target else f"{verb} the resource."

    if step_tool == "edit_file":
        target = step_input.get("path") or ""
        return f"{verb} {target}." if target else f"{verb} the file."

    if step_tool == "run_tests":
        command = step_input.get("command", "")
        return f"{verb} {command}." if command else f"{verb} the test suite."

    if step_tool == "git_commit":
        message = step_input.get("message", "")
        return f"{verb} with message '{message}'." if message else f"{verb} the changes."

    if step_tool == "browser_snapshot":
        return "Take a snapshot of the page."

    if step_tool == "respond":
        return "Respond with the result."

    # Generic fallback
    detail_parts = [f"{k}={v}" for k, v in step_input.items() if v]
    if detail_parts:
        return f"{verb} ({', '.join(detail_parts[:3])})."
    return f"{verb}."


def generate_naked_prompt(task: AgentTask) -> str:
    """Generate a minimal prompt describing the actions taken in a task.

    Iterates through the task's sorted steps, extracting tool names and key
    input fields, and joins them into imperative sentences.

    Args:
        task: A completed task (ideally a golden/exemplar run).

    Returns:
        A multi-sentence instruction string describing the task steps.
    """
    sentences: list[str] = []
    for step in task.sorted_steps:
        sentence = _describe_step(step.tool_name, step.tool_input)
        sentences.append(sentence)
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Baseline building and persistence
# ---------------------------------------------------------------------------


def build_baseline(task: AgentTask, analysis: TaskAnalysis) -> Baseline:
    """Capture a task's metrics as a baseline.

    Args:
        task: The source task (typically a golden exemplar).
        analysis: Pre-computed analysis for the task.

    Returns:
        Baseline: A snapshot of the task's performance metrics.
    """
    return Baseline(
        task_id=task.task_id,
        site_name=analysis.site_name,
        user_text=task.task_text or "",
        step_count=analysis.step_count,
        duration_s=analysis.total_duration_ms / 1000.0,
        total_tokens_in=analysis.tokens_in,
        total_tokens_out=analysis.tokens_out,
        cost_usd=analysis.total_cost_usd,
        error_count=analysis.errors,
        milestones=_extract_milestones(task),
        tool_sequence=analysis.tool_sequence,
        naked_prompt=generate_naked_prompt(task),
    )


def _extract_milestones(task: AgentTask) -> list[str]:
    """Extract milestone labels from a task using the commerce detector."""
    try:
        from .capture import detect_milestone
    except ImportError:
        return []

    milestones: list[str] = []
    seen: set[str] = set()
    for step in task.sorted_steps:
        milestone = detect_milestone(step)
        if milestone and milestone not in seen:
            seen.add(milestone)
            milestones.append(milestone)
    return milestones


def save_baseline(baseline: Baseline, path: str | Path) -> Path:
    """Persist a baseline to a JSON file.

    Args:
        baseline: Baseline to serialize.
        path: Output file path.

    Returns:
        Path: The written file path.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(baseline.to_dict(), indent=2), encoding="utf-8")
    return out


def load_baseline(path: str | Path) -> Baseline:
    """Load a single baseline from a JSON file.

    Args:
        path: Path to a baseline JSON file.

    Returns:
        Baseline: Parsed baseline data.

    Raises:
        FileNotFoundError: When the path does not exist.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return Baseline(
        task_id=str(data.get("task_id", "")),
        site_name=str(data.get("site_name", "")),
        user_text=str(data.get("user_text", "")),
        step_count=int(data.get("step_count", 0)),
        duration_s=float(data.get("duration_s", 0.0)),
        total_tokens_in=int(data.get("total_tokens_in", 0)),
        total_tokens_out=int(data.get("total_tokens_out", 0)),
        cost_usd=float(data.get("cost_usd", 0.0)),
        error_count=int(data.get("error_count", 0)),
        milestones=list(data.get("milestones", [])),
        tool_sequence=list(data.get("tool_sequence", [])),
        naked_prompt=str(data.get("naked_prompt", "")),
    )


def load_baselines(directory: str | Path) -> dict[str, Baseline]:
    """Load all ``.json`` baselines from a directory, keyed by ``site_name``.

    Args:
        directory: Directory containing baseline JSON files.

    Returns:
        dict[str, Baseline]: Baselines indexed by site name.
    """
    baselines: dict[str, Baseline] = {}
    root = Path(directory)
    if not root.is_dir():
        return baselines
    for json_path in sorted(root.glob("*.json")):
        try:
            baseline = load_baseline(json_path)
            baselines[baseline.site_name] = baseline
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return baselines


# ---------------------------------------------------------------------------
# Overhead measurement
# ---------------------------------------------------------------------------

_OVERHEAD_CATEGORIES = [
    (50.0, "efficient"),
    (150.0, "acceptable"),
    (300.0, "bloated"),
]


def _categorize_overhead(pct: float) -> str:
    for threshold, label in _OVERHEAD_CATEGORIES:
        if pct < threshold:
            return label
    return "pathological"


def _overhead_pct(actual: float, baseline: float) -> float:
    if baseline <= 0:
        return 0.0
    return (actual - baseline) / baseline * 100.0


def _compare_milestones(
    baseline_milestones: list[str], actual_milestones: list[str]
) -> str:
    """Compare milestone lists and return a delta label."""
    baseline_set = set(baseline_milestones)
    actual_set = set(actual_milestones)
    if actual_set > baseline_set:
        return "better"
    if actual_set == baseline_set:
        return "same"
    return "worse"


def _identify_contributing_factors(
    baseline: Baseline, analysis: TaskAnalysis
) -> list[str]:
    """Identify human-readable explanations for overhead."""
    factors: list[str] = []
    baseline_set = set(baseline.tool_sequence)
    actual_seq = analysis.tool_sequence

    # Extra tools not in baseline
    extra_tools = set(actual_seq) - baseline_set
    if extra_tools:
        factors.append(
            f"Extra tools not in baseline: {', '.join(sorted(extra_tools))}"
        )

    # Repeated consecutive tools
    if analysis.max_repeat_count >= 3:
        factors.append(
            f"Tool '{analysis.max_repeat_tool}' repeated {analysis.max_repeat_count}x consecutively"
        )

    # More errors than baseline
    error_delta = analysis.errors - baseline.error_count
    if error_delta > 0:
        factors.append(f"{error_delta} more error(s) than baseline")

    # Step count delta
    step_delta = analysis.step_count - baseline.step_count
    if step_delta > 0:
        factors.append(f"{step_delta} extra step(s) beyond baseline")

    # web_search before browser_navigate pattern
    for i in range(len(actual_seq) - 1):
        if actual_seq[i] == "web_search" and actual_seq[i + 1].startswith("browser_"):
            factors.append(
                "Redundant web_search before browser navigation"
            )
            break

    return factors


def _extract_actual_milestones(task: AgentTask) -> list[str]:
    """Extract milestones from a task, tolerating import failures."""
    return _extract_milestones(task)


def measure_overhead(
    task: AgentTask,
    analysis: TaskAnalysis,
    grade: str,
    baseline: Baseline,
) -> OverheadResult:
    """Compare one task against its site baseline.

    Args:
        task: The task being measured.
        analysis: Pre-computed analysis for the task.
        grade: Grade label assigned to the task.
        baseline: The site baseline to compare against.

    Returns:
        OverheadResult: Detailed overhead breakdown.
    """
    actual_duration_s = analysis.total_duration_ms / 1000.0
    step_pct = _overhead_pct(analysis.step_count, baseline.step_count)
    duration_pct = _overhead_pct(actual_duration_s, baseline.duration_s)
    cost_pct = _overhead_pct(analysis.total_cost_usd, baseline.cost_usd)
    actual_milestones = _extract_actual_milestones(task)

    return OverheadResult(
        task_id=task.task_id,
        site_name=analysis.site_name,
        grade=grade,
        baseline_steps=baseline.step_count,
        actual_steps=analysis.step_count,
        step_overhead_pct=step_pct,
        baseline_duration_s=baseline.duration_s,
        actual_duration_s=actual_duration_s,
        duration_overhead_pct=duration_pct,
        baseline_cost=baseline.cost_usd,
        actual_cost=analysis.total_cost_usd,
        cost_overhead_pct=cost_pct,
        baseline_errors=baseline.error_count,
        actual_errors=analysis.errors,
        success_delta=_compare_milestones(baseline.milestones, actual_milestones),
        overhead_category=_categorize_overhead(step_pct),
        contributing_factors=_identify_contributing_factors(baseline, analysis),
    )


def measure_all_overhead(
    tasks: list[AgentTask],
    grades: dict[str, str],
    baselines: dict[str, Baseline],
) -> list[OverheadResult]:
    """Measure overhead for all tasks that have a matching site baseline.

    Args:
        tasks: Tasks to measure.
        grades: Mapping of ``task_id`` to grade label.
        baselines: Baselines indexed by ``site_name``.

    Returns:
        list[OverheadResult]: Results for tasks with matching baselines.
    """
    results: list[OverheadResult] = []
    for task in tasks:
        analysis = analyze_task(task)
        baseline = baselines.get(analysis.site_name)
        if baseline is None:
            continue
        grade = grades.get(task.task_id, "BROKEN")
        results.append(measure_overhead(task, analysis, grade, baseline))
    return results


# ---------------------------------------------------------------------------
# Prompt hash grouping
# ---------------------------------------------------------------------------


def _prompt_hash_for_task(task: AgentTask) -> str:
    """Extract the most common system_prompt_hash across a task's steps."""
    hashes: dict[str, int] = {}
    for step in task.steps:
        h = step.system_prompt_hash
        if h:
            hashes[h] = hashes.get(h, 0) + 1
    if not hashes:
        return "unknown"
    return max(hashes, key=lambda k: hashes[k])


def group_by_prompt_hash(
    tasks: list[AgentTask],
    analyses: dict[str, TaskAnalysis],
    grades: dict[str, str],
    baselines: dict[str, Baseline] | None = None,
) -> list[PromptHashGroup]:
    """Group tasks by ``system_prompt_hash`` and compute per-group aggregates.

    Args:
        tasks: All tasks to group.
        analyses: Pre-computed analyses keyed by ``task_id``.
        grades: Grade labels keyed by ``task_id``.
        baselines: Optional baselines for overhead computation.

    Returns:
        list[PromptHashGroup]: One entry per distinct prompt hash, sorted by
        task count descending.
    """
    groups: dict[str, list[AgentTask]] = {}
    for task in tasks:
        h = _prompt_hash_for_task(task)
        groups.setdefault(h, []).append(task)

    result: list[PromptHashGroup] = []
    for prompt_hash, group_tasks in groups.items():
        step_counts: list[int] = []
        durations: list[float] = []
        costs: list[float] = []
        error_counts: list[int] = []
        golden_good = 0
        broken = 0
        overhead_pcts: list[float] = []

        for task in group_tasks:
            analysis = analyses.get(task.task_id)
            if analysis is None:
                analysis = analyze_task(task)
            step_counts.append(analysis.step_count)
            durations.append(analysis.total_duration_ms / 1000.0)
            costs.append(analysis.total_cost_usd)
            error_counts.append(analysis.errors)

            grade = grades.get(task.task_id, "BROKEN")
            if grade in ("GOLDEN", "GOOD"):
                golden_good += 1
            if grade == "BROKEN":
                broken += 1

            if baselines:
                bl = baselines.get(analysis.site_name)
                if bl and bl.step_count > 0:
                    overhead_pcts.append(
                        _overhead_pct(analysis.step_count, bl.step_count)
                    )

        n = len(group_tasks)
        result.append(
            PromptHashGroup(
                prompt_hash=prompt_hash,
                task_count=n,
                avg_steps=sum(step_counts) / n if n else 0.0,
                avg_duration_s=sum(durations) / n if n else 0.0,
                avg_cost=sum(costs) / n if n else 0.0,
                avg_errors=sum(error_counts) / n if n else 0.0,
                golden_rate=golden_good / n if n else 0.0,
                broken_rate=broken / n if n else 0.0,
                avg_overhead_pct=(
                    sum(overhead_pcts) / len(overhead_pcts) if overhead_pcts else 0.0
                ),
                sample_task_ids=[t.task_id for t in group_tasks[:3]],
            )
        )

    result.sort(key=lambda g: g.task_count, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def _text_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    header = "  " + "  ".join(f"{str(c):<{widths[i]}}" for i, c in enumerate(headers))
    divider = "  " + "  ".join("-" * w for w in widths)
    body = [
        "  " + "  ".join(f"{str(c):<{widths[i]}}" for i, c in enumerate(row))
        for row in rows
    ]
    return [header, divider, *body]


def format_overhead_report(
    results: list[OverheadResult],
    hash_groups: list[PromptHashGroup] | None = None,
) -> str:
    """Format a terminal-friendly overhead report.

    Args:
        results: Per-task overhead results.
        hash_groups: Optional prompt-hash aggregation data.

    Returns:
        str: Multi-line human-readable report.
    """
    lines: list[str] = ["PROMPT OVERHEAD ANALYSIS", "=" * 60]

    if not results:
        lines.append("No tasks matched any baseline.")
        return "\n".join(lines)

    # Per-site summary table
    site_data: dict[str, list[OverheadResult]] = {}
    for r in results:
        site_data.setdefault(r.site_name, []).append(r)

    rows: list[list[Any]] = []
    for site, site_results in sorted(site_data.items()):
        bl_steps = site_results[0].baseline_steps
        avg_actual = sum(r.actual_steps for r in site_results) / len(site_results)
        avg_pct = sum(r.step_overhead_pct for r in site_results) / len(site_results)
        rows.append([
            site,
            f"{bl_steps} steps",
            f"{avg_actual:.0f} steps",
            f"+{avg_pct:.0f}%",
        ])
    lines.append("")
    lines.extend(
        _text_table(["Site", "Baseline", "Avg Run", "Step Overhead"], rows)
    )

    # Overhead distribution
    categories = {"efficient": 0, "acceptable": 0, "bloated": 0, "pathological": 0}
    for r in results:
        categories[r.overhead_category] = categories.get(r.overhead_category, 0) + 1
    total = len(results)
    lines.append("")
    lines.append("OVERHEAD DISTRIBUTION:")
    for cat, label_desc in [
        ("efficient", "Efficient (<50%)"),
        ("acceptable", "Acceptable (50-150%)"),
        ("bloated", "Bloated (150-300%)"),
        ("pathological", "Pathological (>300%)"),
    ]:
        count = categories[cat]
        pct = round(count * 100 / max(1, total))
        lines.append(f"  {label_desc + ':':<30} {count:>3} tasks ({pct}%)")

    # Prompt hash correlation
    if hash_groups:
        lines.append("")
        lines.append("PROMPT HASH CORRELATION:")
        for g in hash_groups[:5]:
            golden_pct = round(g.golden_rate * 100)
            lines.append(
                f"  Hash {g.prompt_hash[:8]:<10} ({g.task_count:>3} tasks): "
                f"avg {g.avg_steps:.1f} steps, {golden_pct}% golden, "
                f"overhead +{g.avg_overhead_pct:.0f}%"
            )
        if len(hash_groups) >= 2:
            best = min(hash_groups[:5], key=lambda g: g.avg_steps)
            worst = max(hash_groups[:5], key=lambda g: g.avg_steps)
            if best.prompt_hash != worst.prompt_hash:
                delta = worst.avg_steps - best.avg_steps
                lines.append(
                    f"  -> Hash {best.prompt_hash[:8]} is {delta:.1f} steps more "
                    f"efficient (likely the minimal prompt variant)"
                )

    # Biggest overhead sources
    all_factors: dict[str, int] = {}
    for r in results:
        for f in r.contributing_factors:
            all_factors[f] = all_factors.get(f, 0) + 1
    if all_factors:
        lines.append("")
        lines.append("BIGGEST OVERHEAD SOURCES:")
        sorted_factors = sorted(all_factors.items(), key=lambda x: x[1], reverse=True)
        for i, (factor, count) in enumerate(sorted_factors[:5], 1):
            lines.append(f"  {i}. {factor} ({count} task(s))")

    return "\n".join(lines)


def format_prompt_impact_report(
    hash_groups: list[PromptHashGroup],
) -> str:
    """Format a prompt-impact report showing per-hash performance.

    Args:
        hash_groups: Prompt hash groups to display.

    Returns:
        str: Multi-line human-readable report.
    """
    lines: list[str] = ["PROMPT IMPACT ANALYSIS", "=" * 60]

    if not hash_groups:
        lines.append("No prompt hash data found.")
        return "\n".join(lines)

    rows: list[list[Any]] = []
    for g in hash_groups:
        golden_pct = round(g.golden_rate * 100)
        broken_pct = round(g.broken_rate * 100)
        rows.append([
            g.prompt_hash[:8],
            g.task_count,
            f"{g.avg_steps:.1f}",
            f"{g.avg_duration_s:.1f}s",
            f"${g.avg_cost:.4f}",
            f"{golden_pct}%",
            f"{broken_pct}%",
        ])
    lines.append("")
    lines.extend(
        _text_table(
            ["Hash", "Tasks", "Avg Steps", "Avg Duration", "Avg Cost", "Golden%", "Broken%"],
            rows,
        )
    )

    if len(hash_groups) >= 2:
        best = min(hash_groups, key=lambda g: g.avg_steps)
        worst = max(hash_groups, key=lambda g: g.avg_steps)
        if best.prompt_hash != worst.prompt_hash:
            delta = worst.avg_steps - best.avg_steps
            lines.append("")
            lines.append(
                f"Best prompt hash: {best.prompt_hash[:8]} "
                f"(avg {best.avg_steps:.1f} steps, "
                f"{round(best.golden_rate * 100)}% golden)"
            )
            lines.append(
                f"Worst prompt hash: {worst.prompt_hash[:8]} "
                f"(avg {worst.avg_steps:.1f} steps, "
                f"{round(worst.golden_rate * 100)}% golden)"
            )
            lines.append(f"Delta: {delta:.1f} steps per task")

    return "\n".join(lines)


def overhead_report_data(
    results: list[OverheadResult],
    hash_groups: list[PromptHashGroup] | None = None,
) -> dict[str, Any]:
    """Return structured data for the overhead report.

    Args:
        results: Per-task overhead results.
        hash_groups: Optional prompt-hash aggregation data.

    Returns:
        dict: JSON-serializable overhead report data.
    """
    categories = {"efficient": 0, "acceptable": 0, "bloated": 0, "pathological": 0}
    for r in results:
        categories[r.overhead_category] = categories.get(r.overhead_category, 0) + 1
    return {
        "total_measured": len(results),
        "distribution": categories,
        "tasks": [r.to_dict() for r in results],
        "hash_groups": [g.to_dict() for g in (hash_groups or [])],
    }


def prompt_impact_data(
    hash_groups: list[PromptHashGroup],
) -> dict[str, Any]:
    """Return structured data for the prompt-impact report.

    Args:
        hash_groups: Prompt hash groups.

    Returns:
        dict: JSON-serializable prompt impact data.
    """
    return {
        "total_groups": len(hash_groups),
        "groups": [g.to_dict() for g in hash_groups],
    }
