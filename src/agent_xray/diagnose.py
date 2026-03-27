from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .root_cause import ROOT_CAUSES, RootCauseResult

FIX_TARGETS = {
    "routing_bug": ["tool registry", "task-to-tool routing rules"],
    "approval_block": ["approval policy", "tool risk map"],
    "spin": ["loop detector", "recovery instructions"],
    "environment_drift": ["runner retries", "selector strategy", "timeouts"],
    "tool_bug": ["tool implementation", "tool result formatting"],
    "tool_selection_bug": ["tool descriptions", "tool-choice examples"],
    "early_abort": ["stop conditions", "success criteria"],
    "stuck_loop": ["progress checks", "navigation guidance"],
    "reasoning_bug": ["prompt examples", "decision heuristics"],
    "prompt_bug": ["prompt builder", "task-specific examples"],
    "model_limit": ["task decomposition", "model choice"],
}


@dataclass(slots=True)
class FixPlanEntry:
    priority: int
    root_cause: str
    count: int
    impact: int
    investigate_task: str
    targets: list[str]
    fix_hint: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority": self.priority,
            "root_cause": self.root_cause,
            "count": self.count,
            "impact": self.impact,
            "investigate_task": self.investigate_task,
            "targets": self.targets,
            "fix_hint": self.fix_hint,
            "evidence": self.evidence,
        }


def build_fix_plan(results: list[RootCauseResult]) -> list[FixPlanEntry]:
    grouped: dict[str, list[RootCauseResult]] = defaultdict(list)
    for result in results:
        grouped[result.root_cause].append(result)
    plan: list[FixPlanEntry] = []
    for cause, items in sorted(grouped.items(), key=lambda item: -len(item[1])):
        worst = min(items, key=lambda item: item.score)
        impact = len(items) * max(1, abs(worst.score))
        plan.append(
            FixPlanEntry(
                priority=0,
                root_cause=cause,
                count=len(items),
                impact=impact,
                investigate_task=worst.task_id,
                targets=FIX_TARGETS.get(cause, ["prompt builder"]),
                fix_hint=ROOT_CAUSES[cause]["fix_hint"],
                evidence=worst.evidence[:3],
            )
        )
    plan.sort(key=lambda entry: -entry.impact)
    for index, entry in enumerate(plan, start=1):
        entry.priority = index
    return plan


def format_fix_plan_text(plan: list[FixPlanEntry]) -> str:
    if not plan:
        return "No BROKEN or WEAK tasks found. Nothing to diagnose."
    lines = ["FIX PLAN", "=" * 60, ""]
    for entry in plan:
        lines.append(
            f"Priority #{entry.priority}: {entry.root_cause} "
            f"({entry.count} task(s), impact={entry.impact})"
        )
        lines.append(f"  Targets: {', '.join(entry.targets)}")
        lines.append(f"  Fix hint: {entry.fix_hint}")
        lines.append(f"  Investigate task: {entry.investigate_task}")
        if entry.evidence:
            lines.append(f"  Evidence: {'; '.join(entry.evidence)}")
        lines.append("")
    return "\n".join(lines)
