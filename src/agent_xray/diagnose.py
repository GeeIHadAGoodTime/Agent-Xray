from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .root_cause import ROOT_CAUSES, RootCauseResult

_UNKNOWN_ROOT_CAUSE = {
    "label": "Unknown",
    "description": "Custom root cause.",
    "fix_hint": "Investigate manually.",
}

INVESTIGATION_HINTS: dict[str, list[str]] = {
    "routing_bug": [
        "tool registration or tool-routing logic",
        "tool authorization or permission gate that controls tool visibility",
        "handler wiring that connects tool names to implementations",
    ],
    "approval_block": [
        "tool approval policy or confirmation gate configuration",
        "tool risk classification or trust level assignment",
        "auto-approval rules for low-risk tool categories",
    ],
    "spin": [
        "spin detection threshold or max consecutive retry count",
        "recovery logic after repeated tool failures",
        "intervention or circuit-breaker when same tool fails repeatedly",
    ],
    "environment_drift": [
        "timeout and retry logic for external interactions",
        "selector strategy or element identification approach",
        "wait conditions for page or environment readiness",
    ],
    "tool_bug": [
        "tool handler implementation where the tool produces errors",
        "tool result format that the LLM must parse",
        "tool input validation that should reject malformed requests",
    ],
    "tool_selection_bug": [
        "tool descriptions that guide the LLM toward correct tool choice",
        "tool-choice examples or priority ordering in the prompt",
        "tool set scoping that controls which tools are exposed per task type",
    ],
    "early_abort": [
        "stop conditions or success criteria that trigger task completion",
        "minimum step count before the agent is allowed to finish",
        "continuation logic that nudges the agent to keep going",
    ],
    "stuck_loop": [
        "progress detection that identifies when the agent is stuck",
        "reassessment prompt triggered after repeated actions on the same state",
        "guidance for when to abandon the current approach and try another",
    ],
    "reasoning_bug": [
        "prompt examples or few-shot demonstrations for this task type",
        "decision heuristics or strategy guidance in the system prompt",
        "examples corpus that teaches correct reasoning patterns",
    ],
    "prompt_bug": [
        "prompt builder or task-specific prompt sections",
        "prompt section referenced in the prompt_section evidence field",
    ],
    "model_limit": [
        "task decomposition logic that breaks complex tasks into subtasks",
        "model selection or routing to a more capable model",
        "context window management or token budget allocation",
    ],
    "memory_overload": [
        "context compaction or summarization strategy",
        "task decomposition for long-running interactions",
        "message trimming or eviction policy before context fills",
    ],
    "delegation_failure": [
        "delegation routing that decides which tasks get delegated",
        "sub-agent tool availability and permission configuration",
        "delegation error handling and response parsing",
    ],
    "test_failure_loop": [
        "test runner exit conditions or max retry limits",
        "failure triage logic that decides whether to retry or change approach",
        "error pattern detection that aborts repeated identical failures",
    ],
    "tool_rejection_mismatch": [
        "tool approval or rejection policy configuration",
        "tool risk classification where rejected tools may actually be safe",
        "focused tool set routing that ensures required tools are available",
    ],
    "insufficient_sources": [
        "search tool configuration and availability for research tasks",
        "minimum source count or diversity requirements before synthesis",
        "source gathering guidance in the research prompt",
    ],
    "valid_alternative_path": [
        "task expectation or grading configuration that accepts non-browser paths",
        "alternative completion path recognition in the evaluation rules",
    ],
    "consultative_success": [
        "task expectation or grading configuration that accepts consultative answers",
        "consultative completion recognition in the evaluation rules",
    ],
    "unclassified": [
        "task surface and reasoning chain for manual pattern identification",
        "step log analysis for unusual sequences or error patterns",
    ],
}
"""Default investigation hints keyed by root cause for the built-in resolver.

These are CONCEPTS to search for in your codebase, not file paths.  Agent-xray
is a trace analyzer -- it knows WHAT happened and WHY, but it has never read
your source code and cannot reliably tell you WHERE to fix it.
"""

# Backward-compatible alias
FIX_TARGETS = INVESTIGATION_HINTS

# Maps prompt_section values to conceptual investigation areas
PROMPT_SECTION_TARGETS = {
    "research": ["research prompt section or search-vs-browse priority", "source gathering and synthesis instructions"],
    "tools": ["tool descriptions and autonomy guardrails", "tool priority ordering or selection guidance"],
    "browser": ["browser navigation prompt section", "progress detection or stuck-page instructions"],
    "payment": ["payment fill instructions or form field identification guidance", "payment gate or checkout handling logic"],
    "planning": ["task planning prompt section", "multi-step strategy or decomposition guidance"],
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
    "unclassified": 1,
}


class TargetResolver(Protocol):
    """Resolve investigation hints for a root cause from its summarized evidence.

    The default resolver returns conceptual search terms (not file paths).
    Plugin implementations MAY return file paths but should validate them
    against the target project, since paths go stale when codebases refactor.
    """

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
    """Resolve investigation hints from the built-in root-cause and prompt-section mappings.

    Returns conceptual search terms derived from trace evidence, not file paths.
    These are things to search for in your codebase, not files to open.
    """

    def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
        """Return investigation concepts, with prompt-section enrichment for prompt bugs."""
        targets = list(INVESTIGATION_HINTS.get(root_cause, ["prompt builder"]))
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


def _verify_command_for(root_cause: str, task_id: str, log_dir: str | None = None) -> str:
    """Return a suggested verification command for a root cause.

    These are starting points for investigation — adapt based on your
    findings.  They are not prescriptions.
    """
    dir_placeholder = log_dir if log_dir else "<dir>"
    commands = {
        "approval_block": f"agent-xray surface {task_id} | grep approval",
        "environment_drift": f"agent-xray surface {task_id} | grep page_url",
        "routing_bug": f"agent-xray surface {task_id} | grep tools_available",
        "tool_bug": f"agent-xray surface {task_id} | grep tools_available",
        "spin": f"agent-xray grade {dir_placeholder} --rules default | grep SPIN",
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
        "unclassified": f"agent-xray surface {task_id}",
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
        targets: Conceptual search terms for what to investigate in your codebase.
            These are concepts derived from trace evidence, not file paths.
            Plugin resolvers may return file paths, but the default resolver
            returns generic concepts that apply to any codebase.
        fix_hint: Human-readable fix hint for the grouped failures.
        verify_command: Suggested starting-point CLI command for investigation
            (adapt based on your findings).
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
            "investigate": self.targets,
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
    log_dir: str | None = None,
) -> list[FixPlanEntry]:
    """Build a prioritized fix plan from grouped root-cause classifications.

    Args:
        results: Root-cause results to group into plan entries.
        target_resolver: Optional resolver name or resolver instance used to map
            root causes to investigation targets.
        min_sample_size: Minimum number of tasks for a root cause to be
            considered high confidence.  Groups below this threshold receive
            a ``low_confidence`` flag and a warning in their evidence.
        log_dir: Trace directory path.  When provided, replaces ``<dir>``
            placeholders in generated verify commands with the actual path.

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
                verify_command=_verify_command_for(cause, worst.task_id, log_dir=log_dir),
                evidence=evidence,
                low_confidence=is_low_confidence,
            )
        )
    plan.sort(key=lambda entry: (-entry.impact, -entry.severity))
    for index, entry in enumerate(plan, start=1):
        entry.priority = index
    return plan


CODE_EXTENSIONS = {".py", ".js", ".ts", ".json", ".yaml", ".yml", ".toml", ".cfg", ".md"}
"""File extensions recognized as code paths during target validation."""


def validate_fix_targets(
    plan: list[FixPlanEntry],
    project_root: str | Path | None = None,
) -> list[FixPlanEntry]:
    """Check that file path targets in the fix plan actually exist on disk.

    For each target that looks like a file path (contains ``/`` and ends with
    a code extension like ``.py``, ``.json``, ``.js``, ``.ts``), check if it
    exists relative to *project_root*.  If not found, add a warning to the
    entry's evidence.

    Args:
        plan: Fix plan entries to validate.
        project_root: Root directory of the target project.  If ``None``,
            skip validation.

    Returns:
        The same plan entries, with stale path warnings added to evidence.
    """
    if project_root is None:
        return plan
    root = Path(project_root)
    if not root.is_dir():
        return plan

    for entry in plan:
        stale_targets: list[str] = []
        for target in entry.targets:
            # Only validate targets that look like file paths
            if "/" not in target:
                continue
            # Check if it ends with a code extension
            suffix = Path(target).suffix
            if suffix not in CODE_EXTENSIONS:
                continue
            full_path = root / target
            if not full_path.exists():
                stale_targets.append(target)

        if stale_targets:
            for target in stale_targets:
                entry.evidence.append(
                    f"STALE_TARGET: '{target}' not found on disk"
                    " \u2014 file may have been renamed or removed"
                )
    return plan


def list_all_targets(resolver: TargetResolver | None = None) -> dict[str, list[str]]:
    """Get all targets for all known root causes from a resolver.

    Args:
        resolver: Resolver to query.  Uses the active default when ``None``.

    Returns:
        Mapping of root cause label to its resolved target list.
    """
    if resolver is None:
        resolver = get_target_resolver()
    result: dict[str, list[str]] = {}
    for cause in ROOT_CAUSES:
        targets = resolver.resolve(cause, [])
        if targets:
            result[cause] = targets
    return result


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
        cause_meta = ROOT_CAUSES.get(entry.root_cause, _UNKNOWN_ROOT_CAUSE)
        lines.append(
            f"Priority #{entry.priority}: {entry.root_cause} "
            f"(severity={entry.severity}/5, {entry.count} task(s), impact={entry.impact})"
            f"{confidence_tag}"
        )
        # Show what happened from evidence
        non_stale = [e for e in entry.evidence if not e.startswith(("STALE_TARGET:", "LOW_CONFIDENCE:"))]
        if non_stale:
            lines.append(f"  What happened: {non_stale[0]}")
        # Show root cause description
        lines.append(f"  Root cause: {cause_meta['description']}")
        # Show investigation hints as search terms
        lines.append(f"  Search your codebase for: {', '.join(entry.targets)}")
        # Surface stale target warnings prominently
        stale_warnings = [e for e in entry.evidence if e.startswith("STALE_TARGET:")]
        for warning in stale_warnings:
            # Extract the path from the STALE_TARGET message
            path_part = warning.split("'")[1] if "'" in warning else warning
            lines.append(
                f"  \u26a0 STALE TARGET: {path_part} not found"
                " \u2014 update your resolver"
            )
        lines.append(f"  Investigate task: {entry.investigate_task}")
        lines.append(f"  Suggested verification: {entry.verify_command}")
        lines.append("")
    return "\n".join(lines)
