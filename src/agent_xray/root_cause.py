from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .analyzer import TaskAnalysis, analyze_task
from .grader import GradeResult
from .schema import AgentTask

ROOT_CAUSES = {
    "routing_bug": {
        "label": "Routing Bug",
        "description": "The task had no tools available or the needed tool was never exposed.",
        "fix_hint": "Review how your tool registry scopes tools for this task type.",
    },
    "approval_block": {
        "label": "Approval Block",
        "description": (
            "The task failed because tool execution was blocked by policy or confirmation gates."
        ),
        "fix_hint": "Review approval policies and tool risk configuration.",
    },
    "spin": {
        "label": "Spin",
        "description": "The agent repeated the same action without progressing.",
        "fix_hint": "Tighten loop detection and add stronger recovery instructions.",
    },
    "environment_drift": {
        "label": "Environment Drift",
        "description": (
            "The environment or target page changed in ways the runner could not handle."
        ),
        "fix_hint": "Inspect selectors, retries, and timeout handling in the runner.",
    },
    "tool_bug": {
        "label": "Tool Bug",
        "description": "The right tool was called, but it failed or returned unusable data.",
        "fix_hint": "Debug the tool implementation and improve tool result clarity.",
    },
    "tool_selection_bug": {
        "label": "Tool Selection Bug",
        "description": "The agent avoided the right tool despite having it.",
        "fix_hint": "Clarify tool descriptions and tool-choice examples.",
    },
    "early_abort": {
        "label": "Early Abort",
        "description": "The task stopped before a reasonable attempt was made.",
        "fix_hint": "Review stopping criteria and premature success/failure logic.",
    },
    "stuck_loop": {
        "label": "Stuck Loop",
        "description": "The task stayed on one page or state while continuing to act.",
        "fix_hint": "Improve progress checks and reassessment prompts.",
    },
    "reasoning_bug": {
        "label": "Reasoning Bug",
        "description": "The task progressed, but the strategy or next action was still wrong.",
        "fix_hint": "Add a concrete example to the prompt or examples corpus.",
    },
    "prompt_bug": {
        "label": "Prompt Bug",
        "description": "The most likely remaining issue is misleading or incomplete instructions.",
        "fix_hint": "Inspect prompt guidance around the failure step.",
    },
    "model_limit": {
        "label": "Model Limit",
        "description": "The task may exceed the model's current planning or perception ability.",
        "fix_hint": "Decompose the task or use a stronger model.",
    },
}


@dataclass(slots=True)
class RootCauseResult:
    task_id: str
    root_cause: str
    grade: str
    score: int
    confidence: str
    evidence: list[str] = field(default_factory=list)
    site_name: str = ""
    error_kinds: dict[str, int] = field(default_factory=dict)


def classify_task(
    task: AgentTask,
    grade: GradeResult,
    analysis: TaskAnalysis | None = None,
) -> RootCauseResult:
    analysis = analysis or analyze_task(task)
    evidence: list[str] = []
    result = RootCauseResult(
        task_id=task.task_id,
        root_cause="prompt_bug",
        grade=grade.grade,
        score=grade.score,
        confidence="medium",
        evidence=evidence,
        site_name=analysis.site_name,
        error_kinds=analysis.error_kinds,
    )
    if analysis.no_tools_steps:
        result.root_cause = "routing_bug"
        result.confidence = "high"
        evidence.append(f"{analysis.no_tools_steps} step(s) exposed zero tools")
        return result
    if analysis.error_kinds.get("approval_block"):
        result.root_cause = "approval_block"
        result.confidence = "high"
        evidence.append(f"{analysis.error_kinds['approval_block']} approval-blocked step(s)")
        return result
    if analysis.max_repeat_count >= 5:
        result.root_cause = "spin"
        result.confidence = "high"
        evidence.append(f"{analysis.max_repeat_tool} repeated {analysis.max_repeat_count} times")
        return result
    if analysis.error_rate > 0.5 and analysis.errors > 1:
        env_errors = sum(
            analysis.error_kinds.get(key, 0) for key in ("timeout", "click_fail", "not_found")
        )
        tool_errors = sum(
            analysis.error_kinds.get(key, 0) for key in ("unknown_tool", "validation", "other")
        )
        if env_errors >= tool_errors:
            result.root_cause = "environment_drift"
            result.confidence = "medium" if env_errors else "low"
            evidence.append(f"environment-style errors dominate: {analysis.error_kinds}")
        else:
            result.root_cause = "tool_bug"
            result.confidence = "medium"
            evidence.append(f"tool errors dominate: {analysis.error_kinds}")
        return result
    if analysis.step_count > 50 and len(analysis.unique_urls) < 2:
        result.root_cause = "model_limit"
        result.confidence = "medium"
        evidence.append(f"{analysis.step_count} steps with only {len(analysis.unique_urls)} URL(s)")
        return result
    if len(analysis.unique_urls) <= 1 and analysis.step_count >= 5:
        result.root_cause = "stuck_loop"
        evidence.append("stayed on one page while continuing to act")
        return result
    if analysis.step_count < 3:
        result.root_cause = "early_abort"
        evidence.append("task ended too early to gather evidence")
        return result
    if analysis.errors == 0 and len(analysis.unique_tools) > 1 and len(analysis.unique_urls) > 1:
        result.root_cause = "reasoning_bug"
        evidence.append("task progressed but still graded poorly without hard tool failures")
        return result
    if len(analysis.unique_tools) <= 1 and len(analysis.unique_urls) > 1:
        result.root_cause = "tool_selection_bug"
        evidence.append("low tool diversity despite navigation progress")
        return result
    evidence.append("fallback classification after excluding stronger operational causes")
    return result


def classify_failures(
    tasks: list[AgentTask],
    grades: list[GradeResult],
) -> list[RootCauseResult]:
    grade_by_task = {grade.task_id: grade for grade in grades}
    failures: list[RootCauseResult] = []
    for task in tasks:
        grade = grade_by_task[task.task_id]
        if grade.grade in {"BROKEN", "WEAK"}:
            failures.append(classify_task(task, grade))
    return failures


def summarize_root_causes(results: list[RootCauseResult]) -> dict[str, Any]:
    grouped: dict[str, list[RootCauseResult]] = defaultdict(list)
    for result in results:
        grouped[result.root_cause].append(result)
    summary: dict[str, Any] = {}
    total = max(1, len(results))
    for cause, items in sorted(grouped.items(), key=lambda item: -len(item[1])):
        summary[cause] = {
            "label": ROOT_CAUSES[cause]["label"],
            "count": len(items),
            "percentage": round(len(items) * 100 / total, 1),
            "fix_hint": ROOT_CAUSES[cause]["fix_hint"],
            "sample_task_ids": [item.task_id for item in items[:5]],
        }
    return summary


def format_root_causes_text(results: list[RootCauseResult]) -> str:
    if not results:
        return "No BROKEN or WEAK tasks found."
    lines = ["ROOT CAUSE ANALYSIS", "=" * 60, ""]
    for cause, meta in summarize_root_causes(results).items():
        lines.append(f"{cause}: {meta['count']} task(s) [{meta['percentage']}%]")
        lines.append(f"  {ROOT_CAUSES[cause]['description']}")
        lines.append(f"  Fix hint: {meta['fix_hint']}")
        lines.append("")
    worst = sorted(results, key=lambda item: item.score)
    lines.append("Worst tasks:")
    for item in worst[:10]:
        lines.append(
            f"  {item.task_id} [{item.grade}] {item.root_cause} score={item.score} "
            f"evidence={'; '.join(item.evidence[:2])}"
        )
    return "\n".join(lines)
