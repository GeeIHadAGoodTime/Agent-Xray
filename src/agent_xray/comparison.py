from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .analyzer import analyze_task, load_tasks
from .grader import grade_task, load_rules
from .root_cause import classify_task as classify_root_cause
from .schema import GRADE_ORDER, AgentTask
from .surface import surface_for_task

_DATE_PATTERN = re.compile(r"(20\d{2}[-_]?\d{2}[-_]?\d{2})")
_MODEL_PATTERN = re.compile(
    r"(gpt[-_]?\d|claude[-_]?\d|gemini|llama|mistral|o[134][-_]|sonnet|opus|haiku)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class DivergencePoint:
    task_id: str
    step: int
    left_tool: str
    right_tool: str
    decision_surface_identical: bool
    surface_note: str
    better_model: str | None
    better_grade: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelCostSummary:
    label: str
    avg_cost_per_task: float
    total_cost: float
    task_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelComparisonResult:
    left_label: str
    right_label: str
    left_grade_distribution: dict[str, int]
    right_grade_distribution: dict[str, int]
    grade_deltas: dict[str, int]
    divergences: list[DivergencePoint]
    left_cost: ModelCostSummary
    right_cost: ModelCostSummary
    matched_tasks: int
    rules_name: str
    comparison_type: str = "model"
    comparison_header: str = ""
    divergence_summary: dict[str, Any] = field(default_factory=dict)
    left_root_cause_distribution: dict[str, int] = field(default_factory=dict)
    right_root_cause_distribution: dict[str, int] = field(default_factory=dict)
    root_cause_deltas: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["divergences"] = [item.to_dict() for item in self.divergences]
        payload["left_cost"] = self.left_cost.to_dict()
        payload["right_cost"] = self.right_cost.to_dict()
        return payload


def _distribution(grades: list[str]) -> dict[str, int]:
    counts = Counter(grades)
    return {
        name: counts.get(name, 0)
        for name in ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
        if counts.get(name, 0) or name in counts
    }


def _task_cost(task: AgentTask) -> float:
    return float(
        sum(
            step.model.cost_usd
            for step in task.steps
            if step.model and step.model.cost_usd is not None
        )
    )


def _infer_label(tasks: list[AgentTask], fallback: str) -> str:
    names = [
        step.model.model_name
        for task in tasks
        for step in task.steps
        if step.model and step.model.model_name
    ]
    if not names:
        return fallback
    return Counter(str(name) for name in names).most_common(1)[0][0]


def _detect_comparison_type(
    left_path: str | Path,
    right_path: str | Path,
    left_label: str,
    right_label: str,
) -> tuple[str, str]:
    """Detect whether this is a day-vs-day or model-vs-model comparison.

    Returns:
        (comparison_type, header_string)
    """
    left_name = Path(left_path).name
    right_name = Path(right_path).name

    left_date = _DATE_PATTERN.search(left_name)
    right_date = _DATE_PATTERN.search(right_name)
    if left_date and right_date:
        left_d = left_date.group(1).replace("_", "-")
        right_d = right_date.group(1).replace("_", "-")
        return "day", f"Day Comparison: {left_d} vs {right_d}"

    left_model = _MODEL_PATTERN.search(left_name)
    right_model = _MODEL_PATTERN.search(right_name)
    if left_model and right_model:
        return "model", f"Model Comparison: {left_label} vs {right_label}"

    # Fall back: if the inferred labels look like model names, use model comparison
    left_label_model = _MODEL_PATTERN.search(left_label)
    right_label_model = _MODEL_PATTERN.search(right_label)
    if left_label_model or right_label_model:
        return "model", f"Model Comparison: {left_label} vs {right_label}"

    return "run", f"Run Comparison: {left_label} vs {right_label}"


def _build_divergence_summary(
    left_distribution: dict[str, int],
    right_distribution: dict[str, int],
    grade_deltas: dict[str, int],
    left_root_causes: dict[str, int],
    right_root_causes: dict[str, int],
    root_cause_deltas: dict[str, int],
) -> dict[str, Any]:
    """Build a summary of where the two runs diverge."""
    summary: dict[str, Any] = {}

    # Grade-level divergence
    grade_shifts = {
        grade: delta for grade, delta in grade_deltas.items() if delta != 0
    }
    has_grade_divergence = bool(grade_shifts)
    summary["has_grade_divergence"] = has_grade_divergence
    summary["grade_shifts"] = grade_shifts

    left_total = sum(left_distribution.values())
    right_total = sum(right_distribution.values())
    if left_total and right_total:
        left_golden_good = left_distribution.get("GOLDEN", 0) + left_distribution.get("GOOD", 0)
        right_golden_good = right_distribution.get("GOLDEN", 0) + right_distribution.get("GOOD", 0)
        left_pct = round(left_golden_good * 100 / left_total)
        right_pct = round(right_golden_good * 100 / right_total)
        summary["left_success_pct"] = left_pct
        summary["right_success_pct"] = right_pct
        summary["success_pct_delta"] = right_pct - left_pct

    # Root cause divergence
    rc_shifts = {
        cause: delta for cause, delta in root_cause_deltas.items() if delta != 0
    }
    summary["has_root_cause_divergence"] = bool(rc_shifts)
    summary["root_cause_shifts"] = rc_shifts

    return summary


def _root_cause_distribution(
    tasks: list[AgentTask],
    grades: dict[str, Any],
) -> dict[str, int]:
    """Compute root cause counts for BROKEN/WEAK tasks."""
    counts: dict[str, int] = Counter()
    for task in tasks:
        grade = grades.get(task.task_id)
        if grade is None:
            continue
        result = classify_root_cause(task, grade, analysis=analyze_task(task))
        if result is not None:
            counts[result.root_cause] += 1
    return dict(counts)


def _cost_summary(tasks: list[AgentTask], label: str) -> ModelCostSummary:
    total_cost = sum(_task_cost(task) for task in tasks)
    return ModelCostSummary(
        label=label,
        avg_cost_per_task=(total_cost / len(tasks)) if tasks else 0.0,
        total_cost=total_cost,
        task_count=len(tasks),
    )


def _surface_note(left_step: dict[str, Any], right_step: dict[str, Any]) -> tuple[bool, str]:
    same_tools = left_step.get("tools_available_names") == right_step.get("tools_available_names")
    same_history = left_step.get("conversation_history") == right_step.get("conversation_history")
    same_url = left_step.get("page_url") == right_step.get("page_url")
    identical = same_tools and same_history and same_url
    if identical:
        return True, "Decision surface was identical; same tools, same history, same URL"
    differences: list[str] = []
    if not same_tools:
        differences.append("tool set")
    if not same_history:
        differences.append("conversation history")
    if not same_url:
        differences.append("page URL")
    return False, "Decision surface differed in " + ", ".join(differences)


def _find_divergence(
    left_task: AgentTask,
    right_task: AgentTask,
    left_grade: str,
    right_grade: str,
    left_label: str,
    right_label: str,
) -> DivergencePoint | None:
    left_surface = surface_for_task(left_task)
    right_surface = surface_for_task(right_task)
    left_steps = left_surface["steps"]
    right_steps = right_surface["steps"]
    limit = min(len(left_steps), len(right_steps))
    for index in range(limit):
        left_step = left_steps[index]
        right_step = right_steps[index]
        if (
            left_step["tool_name"],
            left_step["tool_input"],
        ) == (
            right_step["tool_name"],
            right_step["tool_input"],
        ):
            continue
        decision_surface_identical, surface_note = _surface_note(left_step, right_step)
        left_rank = GRADE_ORDER.get(left_grade, 0)
        right_rank = GRADE_ORDER.get(right_grade, 0)
        if right_rank > left_rank:
            better_model = right_label
            better_grade = right_grade
        elif left_rank > right_rank:
            better_model = left_label
            better_grade = left_grade
        else:
            better_model = None
            better_grade = None
        return DivergencePoint(
            task_id=left_task.task_id,
            step=left_step["step"],
            left_tool=left_step["tool_name"],
            right_tool=right_step["tool_name"],
            decision_surface_identical=decision_surface_identical,
            surface_note=surface_note,
            better_model=better_model,
            better_grade=better_grade,
        )
    return None


def compare_model_runs(
    left_log_dir: str | Path,
    right_log_dir: str | Path,
    *,
    rules_path: str | Path | None = None,
) -> ModelComparisonResult:
    rules = load_rules(rules_path)
    left_tasks = load_tasks(left_log_dir)
    right_tasks = load_tasks(right_log_dir)
    left_label = _infer_label(left_tasks, Path(left_log_dir).name or "left")
    right_label = _infer_label(right_tasks, Path(right_log_dir).name or "right")

    left_grades = {
        task.task_id: grade_task(task, rules, analysis=analyze_task(task)) for task in left_tasks
    }
    right_grades = {
        task.task_id: grade_task(task, rules, analysis=analyze_task(task)) for task in right_tasks
    }

    left_distribution = _distribution([grade.grade for grade in left_grades.values()])
    right_distribution = _distribution([grade.grade for grade in right_grades.values()])
    grade_deltas = {
        name: right_distribution.get(name, 0) - left_distribution.get(name, 0)
        for name in sorted(
            set(left_distribution) | set(right_distribution),
            key=lambda grade: GRADE_ORDER.get(grade, -1),
            reverse=True,
        )
    }

    left_by_id = {task.task_id: task for task in left_tasks}
    right_by_id = {task.task_id: task for task in right_tasks}
    divergences: list[DivergencePoint] = []
    for task_id in sorted(set(left_by_id) & set(right_by_id)):
        divergence = _find_divergence(
            left_by_id[task_id],
            right_by_id[task_id],
            left_grades[task_id].grade,
            right_grades[task_id].grade,
            left_label,
            right_label,
        )
        if divergence is not None:
            divergences.append(divergence)

    # BUG #9: Auto-detect comparison type from directory names
    comparison_type, comparison_header = _detect_comparison_type(
        left_log_dir, right_log_dir, left_label, right_label,
    )

    # BUG #5: Compute root cause distributions and divergence summary
    left_rc_dist = _root_cause_distribution(left_tasks, left_grades)
    right_rc_dist = _root_cause_distribution(right_tasks, right_grades)
    all_causes = sorted(set(left_rc_dist) | set(right_rc_dist))
    rc_deltas = {
        cause: right_rc_dist.get(cause, 0) - left_rc_dist.get(cause, 0)
        for cause in all_causes
    }

    divergence_summary = _build_divergence_summary(
        left_distribution,
        right_distribution,
        grade_deltas,
        left_rc_dist,
        right_rc_dist,
        rc_deltas,
    )

    return ModelComparisonResult(
        left_label=left_label,
        right_label=right_label,
        left_grade_distribution=left_distribution,
        right_grade_distribution=right_distribution,
        grade_deltas=grade_deltas,
        divergences=divergences,
        left_cost=_cost_summary(left_tasks, left_label),
        right_cost=_cost_summary(right_tasks, right_label),
        matched_tasks=len(set(left_by_id) & set(right_by_id)),
        rules_name=rules.name,
        comparison_type=comparison_type,
        comparison_header=comparison_header,
        divergence_summary=divergence_summary,
        left_root_cause_distribution=left_rc_dist,
        right_root_cause_distribution=right_rc_dist,
        root_cause_deltas=rc_deltas,
    )


def format_model_comparison(result: ModelComparisonResult) -> str:
    # BUG #9: Use auto-detected header instead of always "Model Comparison"
    header = result.comparison_header or f"Model Comparison: {result.left_label} vs {result.right_label}"
    lines = [
        header,
        "",
        "Grade Distribution:",
    ]
    grade_names = ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
    for grade_name in grade_names:
        if (
            grade_name not in result.left_grade_distribution
            and grade_name not in result.right_grade_distribution
        ):
            continue
        delta = result.grade_deltas.get(grade_name, 0)
        sign = "+" if delta >= 0 else ""
        lines.append(
            f"  {grade_name:<7} {result.left_grade_distribution.get(grade_name, 0)} "
            f"-> {result.right_grade_distribution.get(grade_name, 0)} ({sign}{delta})"
        )

    # BUG #5: Show divergence summary with delta values
    ds = result.divergence_summary
    if ds:
        lines.extend(["", "Divergence Summary:"])
        if ds.get("has_grade_divergence"):
            shifts = ds.get("grade_shifts", {})
            shift_parts = []
            for grade, delta in shifts.items():
                sign = "+" if delta > 0 else ""
                shift_parts.append(f"{grade} {sign}{delta}")
            lines.append(f"  Grade shifts: {', '.join(shift_parts)}")
        else:
            lines.append("  Grade distributions are identical.")
        if "success_pct_delta" in ds:
            delta_val = ds["success_pct_delta"]
            sign = "+" if delta_val >= 0 else ""
            lines.append(
                f"  Success rate (GOLDEN+GOOD): "
                f"{ds['left_success_pct']}% -> {ds['right_success_pct']}% ({sign}{delta_val}pp)"
            )
        if ds.get("has_root_cause_divergence"):
            rc_shifts = ds.get("root_cause_shifts", {})
            rc_parts = []
            for cause, delta in rc_shifts.items():
                sign = "+" if delta > 0 else ""
                rc_parts.append(f"{cause} {sign}{delta}")
            lines.append(f"  Root cause shifts: {', '.join(rc_parts)}")

    # Root cause distribution section
    if result.left_root_cause_distribution or result.right_root_cause_distribution:
        lines.extend(["", "Root Cause Distribution:"])
        all_causes = sorted(
            set(result.left_root_cause_distribution) | set(result.right_root_cause_distribution)
        )
        for cause in all_causes:
            left_count = result.left_root_cause_distribution.get(cause, 0)
            right_count = result.right_root_cause_distribution.get(cause, 0)
            delta = result.root_cause_deltas.get(cause, 0)
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {cause:<25} {left_count} -> {right_count} ({sign}{delta})")

    lines.extend(["", "Divergence Points:"])
    if not result.divergences:
        lines.append("  No task-level decision divergences found in the matched set.")
    else:
        for divergence in result.divergences[:10]:
            lines.append(
                f"  {divergence.task_id} step {divergence.step}: "
                f"{result.left_label} chose {divergence.left_tool}, "
                f"{result.right_label} chose {divergence.right_tool}"
            )
            lines.append(f"    {divergence.surface_note}")
            if divergence.better_model and divergence.better_grade:
                lines.append(
                    f"    {divergence.better_model} made the better choice "
                    f"(graded {divergence.better_grade})"
                )

    left_cost = result.left_cost
    right_cost = result.right_cost
    lines.extend(["", "Cost Comparison:"])
    lines.append(
        f"  {left_cost.label}: avg ${left_cost.avg_cost_per_task:.4f}/task, "
        f"total ${left_cost.total_cost:.4f}"
    )
    lines.append(
        f"  {right_cost.label}: avg ${right_cost.avg_cost_per_task:.4f}/task, "
        f"total ${right_cost.total_cost:.4f}"
    )
    if left_cost.total_cost > 0 and right_cost.total_cost > 0:
        delta_pct = ((left_cost.total_cost - right_cost.total_cost) / left_cost.total_cost) * 100
        if delta_pct > 0:
            lines.append(
                f"  {right_cost.label} is {delta_pct:.1f}% cheaper across the compared runs"
            )
        elif delta_pct < 0:
            lines.append(
                f"  {right_cost.label} is {abs(delta_pct):.1f}% more expensive across the compared runs"
            )
    return "\n".join(lines)
