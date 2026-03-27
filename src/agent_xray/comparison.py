from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .analyzer import analyze_task, load_tasks
from .grader import grade_task, load_rules
from .schema import AgentTask
from .surface import surface_for_task

GRADE_ORDER = {
    "BROKEN": 0,
    "WEAK": 1,
    "OK": 2,
    "GOOD": 3,
    "GOLDEN": 4,
}


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
    )


def format_model_comparison(result: ModelComparisonResult) -> str:
    lines = [
        f"Model Comparison: {result.left_label} vs {result.right_label}",
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
