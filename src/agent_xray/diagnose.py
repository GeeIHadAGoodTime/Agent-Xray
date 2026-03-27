from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .root_cause import ROOT_CAUSES, RootCauseResult

FIX_TARGETS = {
    "routing_bug": [
        "tool registry / tool-routing rules",
        "approval or permission policy (ensure tool is authorized)",
        "tool server registration (ensure tool handler is wired)",
    ],
    "approval_block": [
        "approval policy / tool permission configuration",
        "tool risk level or trust classification",
        "auto-approval rules for common low-risk tools",
    ],
    "spin": [
        "loop/spin detector thresholds (max_retries, max_consecutive)",
        "recovery instructions in system prompt",
        "intervention message content after spin detected",
    ],
    "environment_drift": [
        "browser runner timeouts and retry logic",
        "CSS/ARIA selector strategy (selectors may have changed)",
        "page load wait conditions",
    ],
    "tool_bug": [
        "tool handler implementation (the tool itself is broken)",
        "tool result formatting (result may be unparseable by LLM)",
        "tool input validation (bad inputs not caught early)",
    ],
    "tool_selection_bug": [
        "tool descriptions (make the right tools more prominent)",
        "tool-choice examples in prompt",
        "tool set scoping (wrong tool set exposed for this task type)",
    ],
    "early_abort": [
        "stop conditions / success criteria",
        "minimum step count before allowing completion",
        "continuation_nudge logic",
    ],
    "stuck_loop": [
        "progress detection in browser prompt section",
        "reassessment prompts after N steps on same page",
        "navigation guidance (when to try a different approach)",
    ],
    "reasoning_bug": [
        "prompt examples for this task type",
        "decision heuristics and strategy guidance",
        "few-shot examples corpus",
    ],
    "prompt_bug": [
        "prompt builder / task-specific prompt sections",
        "see prompt_section field for which section to edit",
    ],
    "model_limit": [
        "task decomposition into subtasks",
        "model choice (try a more capable model)",
        "context window management",
    ],
}

# Maps prompt_section values to specific file/component targets
PROMPT_SECTION_TARGETS = {
    "research": ["research prompt section", "search vs browse tool priority"],
    "tools": ["tool descriptions and autonomy guardrails", "TOOL_PRIORITY ordering"],
    "browser": ["browser navigation prompt section", "progress detection instructions"],
    "payment": ["payment fill instructions", "form field identification guidance"],
    "planning": ["task planning prompt section", "multi-step strategy guidance"],
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
        targets = list(FIX_TARGETS.get(cause, ["prompt builder"]))
        # Enrich prompt_bug with section-specific targets
        if cause == "prompt_bug" and worst.prompt_section:
            section_targets = PROMPT_SECTION_TARGETS.get(worst.prompt_section, [])
            if section_targets:
                targets = section_targets + targets
        plan.append(
            FixPlanEntry(
                priority=0,
                root_cause=cause,
                count=len(items),
                impact=impact,
                investigate_task=worst.task_id,
                targets=targets,
                fix_hint=worst.prompt_fix_hint or ROOT_CAUSES[cause]["fix_hint"],
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
