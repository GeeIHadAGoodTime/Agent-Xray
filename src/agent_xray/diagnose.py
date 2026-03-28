from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from .root_cause import ROOT_CAUSES, RootCauseResult

_UNKNOWN_ROOT_CAUSE = {
    "label": "Unknown",
    "description": "Custom root cause.",
    "fix_hint": "Investigate manually.",
}

FIX_TARGETS: dict[str, list[str]] = {
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
    "memory_overload": [
        "context window management / compaction strategy",
        "long-context task decomposition",
        "summarization before context fills",
    ],
    "delegation_failure": [
        "delegation routing rules (which tasks get delegated)",
        "sub-agent tool availability and permissions",
        "delegation response handling",
    ],
    "test_failure_loop": [
        "test runner exit conditions",
        "max retry limits for failing tests",
        "error pattern detection to abort early",
    ],
    "tool_rejection_mismatch": [
        "tool approval / permission policy",
        "tool risk classification (rejected tools may be safe)",
        "focused tool set routing (ensure required tools are available)",
    ],
    "insufficient_sources": [
        "search tool configuration and availability",
        "minimum source count requirements",
        "source diversity guidance in prompt",
    ],
    "valid_alternative_path": [
        "task expectation configuration (accept non-browser paths)",
        "grading rules for alternative completion paths",
    ],
    "consultative_success": [
        "task expectation configuration (accept consultative answers)",
        "grading rules for consultative completions",
    ],
}
"""Default investigation targets keyed by root cause for the built-in target resolver."""

# Maps prompt_section values to specific file/component targets
PROMPT_SECTION_TARGETS = {
    "research": ["research prompt section", "search vs browse tool priority"],
    "tools": ["tool descriptions and autonomy guardrails", "TOOL_PRIORITY ordering"],
    "browser": ["browser navigation prompt section", "progress detection instructions"],
    "payment": ["payment fill instructions", "form field identification guidance"],
    "planning": ["task planning prompt section", "multi-step strategy guidance"],
}


SEVERITY_BY_ROOT_CAUSE = {
    "approval_block": 5,
    "environment_drift": 4,
    "routing_bug": 4,
    "tool_bug": 4,
    "spin": 3,
    "stuck_loop": 3,
    "tool_selection_bug": 3,
    "early_abort": 2,
    "model_limit": 2,
    "prompt_bug": 2,
    "reasoning_bug": 1,
    "memory_overload": 4,
    "delegation_failure": 3,
    "test_failure_loop": 3,
    "tool_rejection_mismatch": 5,
    "insufficient_sources": 2,
    "valid_alternative_path": 0,
    "consultative_success": 0,
}


class TargetResolver(Protocol):
    """Resolve likely fix targets for a root cause from its summarized evidence."""

    def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
        """Return investigation targets for the given root cause and evidence."""


def _extract_prompt_sections(evidence: list[str]) -> list[str]:
    sections: list[str] = []
    for item in evidence:
        if not item.startswith("prompt_section="):
            continue
        section = item.split("=", 1)[1].split(":", 1)[0].strip()
        if section and section not in sections:
            sections.append(section)
    return sections


class DefaultTargetResolver:
    """Resolve fix targets from the built-in root-cause and prompt-section mappings."""

    def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
        """Return default targets, with prompt-section enrichment for prompt bugs."""
        targets = list(FIX_TARGETS.get(root_cause, ["prompt builder"]))
        if root_cause != "prompt_bug":
            return targets
        section_targets: list[str] = []
        for section in _extract_prompt_sections(evidence):
            for target in PROMPT_SECTION_TARGETS.get(section, []):
                if target not in section_targets:
                    section_targets.append(target)
        return section_targets + targets


_TARGET_RESOLVERS: dict[str, TargetResolver] = {"default": DefaultTargetResolver()}
_ACTIVE_TARGET_RESOLVER = "default"


def register_target_resolver(
    name: str,
    resolver: TargetResolver,
    *,
    make_default: bool = False,
) -> None:
    """Register a named target resolver and optionally make it the active default."""
    global _ACTIVE_TARGET_RESOLVER
    _TARGET_RESOLVERS[name] = resolver
    if make_default:
        _ACTIVE_TARGET_RESOLVER = name


def get_target_resolver(name: str | None = None) -> TargetResolver:
    """Return a registered target resolver by name or the active default resolver."""
    resolver_name = _ACTIVE_TARGET_RESOLVER if name is None else name
    try:
        return _TARGET_RESOLVERS[resolver_name]
    except KeyError as exc:
        raise KeyError(f"Unknown target resolver: {resolver_name}") from exc


def _resolve_targets(
    root_cause: str,
    evidence: list[str],
    target_resolver: str | TargetResolver | None,
) -> list[str]:
    if target_resolver is None:
        resolver = get_target_resolver()
    elif isinstance(target_resolver, str):
        resolver = get_target_resolver(target_resolver)
    else:
        resolver = target_resolver
    return resolver.resolve(root_cause, evidence)


def _severity_for_root_cause(root_cause: str) -> int:
    return SEVERITY_BY_ROOT_CAUSE.get(root_cause, 2)


def _verify_command_for(root_cause: str, task_id: str) -> str:
    commands = {
        "approval_block": f"agent-xray surface {task_id} | grep approval",
        "environment_drift": f"agent-xray surface {task_id} | grep page_url",
        "routing_bug": f"agent-xray surface {task_id} | grep tools_available",
        "tool_bug": f"agent-xray surface {task_id} | grep tools_available",
        "spin": "agent-xray grade <dir> --rules default | grep SPIN",
        "stuck_loop": f"agent-xray surface {task_id} | grep page_url",
        "tool_selection_bug": f"agent-xray surface {task_id} | grep tool_name",
        "early_abort": f"agent-xray reasoning {task_id} | grep -i success",
        "model_limit": f"agent-xray reasoning {task_id} | grep -i context",
        "prompt_bug": f"agent-xray reasoning {task_id} | grep -i prompt",
        "reasoning_bug": f"agent-xray reasoning {task_id}",
        "memory_overload": f"agent-xray surface {task_id} | grep context",
        "delegation_failure": f"agent-xray surface {task_id} | grep delegat",
        "test_failure_loop": f"agent-xray surface {task_id} | grep test",
        "tool_rejection_mismatch": f"agent-xray surface {task_id} | grep rejected",
        "insufficient_sources": f"agent-xray surface {task_id} | grep search",
        "valid_alternative_path": f"agent-xray surface {task_id} | grep tool_name",
        "consultative_success": f"agent-xray reasoning {task_id}",
    }
    return commands.get(root_cause, f"agent-xray reasoning {task_id}")


@dataclass(slots=True)
class FixPlanEntry:
    """A prioritized fix-plan item summarizing what to inspect and verify.

    Attributes:
        priority: Rank in the generated fix plan.
        root_cause: Root-cause label represented by this plan entry.
        count: Number of failing tasks grouped into the entry.
        impact: Heuristic impact score used for ranking.
        severity: Root-cause severity score from the built-in heuristics.
        investigate_task: Representative task id to inspect first.
        targets: Suggested systems, prompt sections, or components to review.
        fix_hint: Human-readable fix hint for the grouped failures.
        verify_command: Suggested CLI command for validating the fix.
        evidence: Sample evidence strings taken from the worst task.
        low_confidence: Whether the sample size is below the minimum threshold.
    """

    priority: int
    root_cause: str
    count: int
    impact: int
    severity: int
    investigate_task: str
    targets: list[str]
    fix_hint: str
    verify_command: str
    evidence: list[str]
    low_confidence: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fix-plan entry into JSON-friendly data."""
        return {
            "priority": self.priority,
            "root_cause": self.root_cause,
            "count": self.count,
            "impact": self.impact,
            "severity": self.severity,
            "investigate_task": self.investigate_task,
            "targets": self.targets,
            "fix_hint": self.fix_hint,
            "verify_command": self.verify_command,
            "evidence": self.evidence,
            "low_confidence": self.low_confidence,
        }


def build_fix_plan(
    results: list[RootCauseResult],
    *,
    target_resolver: str | TargetResolver | None = None,
    min_sample_size: int = 3,
) -> list[FixPlanEntry]:
    """Build a prioritized fix plan from grouped root-cause classifications.

    Args:
        results: Root-cause results to group into plan entries.
        target_resolver: Optional resolver name or resolver instance used to map
            root causes to investigation targets.
        min_sample_size: Minimum number of tasks for a root cause to be
            considered high confidence.  Groups below this threshold receive
            a ``low_confidence`` flag and a warning in their evidence.

    Returns:
        list[FixPlanEntry]: Ranked fix-plan entries sorted by impact and
        severity.
    """
    grouped: dict[str, list[RootCauseResult]] = defaultdict(list)
    for result in results:
        grouped[result.root_cause].append(result)
    plan: list[FixPlanEntry] = []
    for cause, items in sorted(grouped.items(), key=lambda item: -len(item[1])):
        worst = min(items, key=lambda item: item.score)
        impact = len(items) * max(1, abs(worst.score))
        resolver_evidence = list(worst.evidence)
        if worst.prompt_section:
            resolver_evidence.append(f"prompt_section={worst.prompt_section}")
        targets = _resolve_targets(cause, resolver_evidence, target_resolver)
        is_low_confidence = len(items) < min_sample_size
        evidence = list(worst.evidence[:3])
        if is_low_confidence:
            evidence.append(
                f"LOW_CONFIDENCE: only {len(items)} task(s) "
                f"— verify on more examples before acting"
            )
        plan.append(
            FixPlanEntry(
                priority=0,
                root_cause=cause,
                count=len(items),
                impact=impact,
                severity=_severity_for_root_cause(cause),
                investigate_task=worst.task_id,
                targets=targets,
                fix_hint=worst.prompt_fix_hint
                or ROOT_CAUSES.get(cause, _UNKNOWN_ROOT_CAUSE)["fix_hint"],
                verify_command=_verify_command_for(cause, worst.task_id),
                evidence=evidence,
                low_confidence=is_low_confidence,
            )
        )
    plan.sort(key=lambda entry: (-entry.impact, -entry.severity))
    for index, entry in enumerate(plan, start=1):
        entry.priority = index
    return plan


def format_fix_plan_text(plan: list[FixPlanEntry]) -> str:
    """Render a fix plan as concise terminal-friendly text.

    Args:
        plan: Fix-plan entries to format.

    Returns:
        str: Multi-line text report suitable for CLI output.
    """
    if not plan:
        return "No BROKEN or WEAK tasks found. Nothing to diagnose."
    lines = ["FIX PLAN", "=" * 60, ""]
    for entry in plan:
        confidence_tag = " [LOW CONFIDENCE]" if entry.low_confidence else ""
        lines.append(
            f"Priority #{entry.priority}: {entry.root_cause} "
            f"(severity={entry.severity}/5, {entry.count} task(s), impact={entry.impact})"
            f"{confidence_tag}"
        )
        lines.append(f"  Targets: {', '.join(entry.targets)}")
        lines.append(f"  Fix hint: {entry.fix_hint}")
        lines.append(f"  Investigate task: {entry.investigate_task}")
        lines.append(f"  Verify: {entry.verify_command}")
        if entry.evidence:
            lines.append(f"  Evidence: {'; '.join(entry.evidence)}")
        lines.append("")
    return "\n".join(lines)
