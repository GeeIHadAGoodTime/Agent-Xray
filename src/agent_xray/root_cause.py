from __future__ import annotations

import re
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
    prompt_section: str | None = None
    prompt_fix_hint: str | None = None


PROMPT_CONFUSION_RE = re.compile(
    r"\b(?:confused|unclear|not sure|unsure|don't know|cannot tell|can't tell|ambiguous)\b",
    re.IGNORECASE,
)
PROMPT_UNCERTAINTY_RE = re.compile(
    r"\b(?:should I|what if|which tool|not clear|unclear whether|don't know if|"
    r"maybe I should|I'm not certain|could try either)\b",
    re.IGNORECASE,
)
SEARCH_TOOL_MARKERS = ("search", "read", "fetch", "scrape")

# Maps evidence patterns to the prompt section/component that needs fixing.
# Each entry: (regex_on_evidence, section_name, fix_description)
PROMPT_BUG_PATTERNS = [
    (r"web_search.*browser|search.*instead of browse", "research", "LLM chose web_search when browser was open — clarify tool priority in research/search prompt section"),
    (r"hallucinated|unknown.tool", "tools", "LLM called nonexistent tools — add explicit tool availability guardrails"),
    (r"stuck on one page|same page.*\d+ steps", "browser", "LLM repeated actions on same page — strengthen progress detection in browser prompt section"),
    (r"only \d+ unique tool", "tools", "Low tool diversity — sharpen tool priority ordering in tool descriptions"),
    (r"payment|checkout.*never filled|reached checkout.*no fill", "payment", "Reached checkout but didn't fill payment — clarify payment fill instructions"),
    (r"tried.*different.*approach|backtrack|going back", "planning", "LLM tried to backtrack — strengthen planning strategy guidance"),
    (r"no.*result|empty.*response|returned nothing", "tools", "Tool returned empty — add result validation and retry guidance"),
    (r"already.*tried|tried.*before|same.*action.*again", "browser", "LLM repeated failed approach — add progress memory instructions"),
    (r"too many.*steps|running out|context.*full", "planning", "Agent aware of resource limits — improve task decomposition guidance"),
    (r"not sure which|multiple.*options|could.*either", "tools", "Tool selection uncertainty — sharpen tool descriptions and priority"),
    (r"page.*changed|unexpected.*layout|different.*from", "browser", "Page layout mismatch — update selector strategies"),
]


def _available_tools(task: AgentTask) -> set[str]:
    names: set[str] = set()
    for step in task.steps:
        names.update(step.tools_available_names or [])
    return names


def _browser_tool_available(task: AgentTask) -> bool:
    return any(name.startswith(("browser_", "desktop_")) for name in _available_tools(task))


def _used_browser_tool(task: AgentTask) -> bool:
    return any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.steps)


def _used_only_search_like_tools(task: AgentTask) -> bool:
    if not task.steps:
        return False
    return all(any(marker in step.tool_name.lower() for marker in SEARCH_TOOL_MARKERS) for step in task.steps)


def _has_prompt_confusion(task: AgentTask) -> bool:
    return any(
        bool(
            step.llm_reasoning
            and (
                PROMPT_CONFUSION_RE.search(step.llm_reasoning)
                or PROMPT_UNCERTAINTY_RE.search(step.llm_reasoning)
            )
        )
        for step in task.steps
    )


@dataclass(slots=True)
class ClassificationConfig:
    """Configurable thresholds for root cause classification."""

    spin_threshold: int = 5
    high_error_rate: float = 0.5
    model_limit_steps: int = 50
    stuck_loop_min_steps: int = 5
    early_abort_max_steps: int = 3


_DEFAULT_CONFIG = ClassificationConfig()


def classify_task(
    task: AgentTask,
    grade: GradeResult,
    analysis: TaskAnalysis | None = None,
    config: ClassificationConfig | None = None,
) -> RootCauseResult | None:
    if grade.grade not in {"BROKEN", "WEAK"}:
        return None
    cfg = config or _DEFAULT_CONFIG
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
    if analysis.max_repeat_count >= cfg.spin_threshold:
        result.root_cause = "spin"
        result.confidence = "high"
        evidence.append(f"{analysis.max_repeat_tool} repeated {analysis.max_repeat_count} times")
        return result
    if analysis.error_rate > cfg.high_error_rate and analysis.errors > 1:
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
    if _browser_tool_available(task) and not _used_browser_tool(task) and _used_only_search_like_tools(task):
        result.root_cause = "tool_selection_bug"
        result.confidence = "high"
        evidence.append("browser tools were available but the agent stayed on search/read tools")
        return result
    if _has_prompt_confusion(task):
        result.root_cause = "prompt_bug"
        result.confidence = "high"
        evidence.append("reasoning shows confusion about what to do next")
        return result
    if analysis.step_count > cfg.model_limit_steps and len(analysis.unique_urls) < 2:
        result.root_cause = "model_limit"
        result.confidence = "medium"
        evidence.append(f"{analysis.step_count} steps with only {len(analysis.unique_urls)} URL(s)")
        return result
    if len(analysis.unique_urls) <= 1 and analysis.step_count >= cfg.stuck_loop_min_steps:
        result.root_cause = "stuck_loop"
        evidence.append("stayed on one page while continuing to act")
        return result
    if analysis.step_count < cfg.early_abort_max_steps:
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
    # Prompt bug sub-classification: detect which section needs fixing
    _enrich_prompt_bug(result, task, analysis)
    evidence.append("fallback classification after excluding stronger operational causes")
    return result


def _enrich_prompt_bug(
    result: RootCauseResult,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> None:
    """When root_cause is prompt_bug, try to identify the specific prompt section."""
    if result.root_cause != "prompt_bug":
        return
    all_evidence = " ".join(result.evidence)
    # Check search-when-browser-open
    has_browser = any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.steps)
    has_search = any("search" in step.tool_name.lower() for step in task.steps)
    if has_browser and has_search:
        all_evidence += " web_search with browser open"
    # Check hallucinated tools
    if analysis.hallucinated_tools:
        all_evidence += " hallucinated unknown tool"
    # Check stuck on page
    if len(analysis.unique_urls) <= 1 and analysis.step_count >= 5:
        all_evidence += f" stuck on one page for {analysis.step_count} steps"
    # Check low tool diversity
    if len(analysis.unique_tools) <= 2 and analysis.step_count >= 8:
        all_evidence += f" only {len(analysis.unique_tools)} unique tools"
    # Check commerce-specific patterns
    commerce = analysis.signal_metrics.get("commerce", {})
    if commerce.get("reached_checkout") and not commerce.get("reached_payment") and analysis.step_count >= 10:
        all_evidence += " reached checkout never filled payment"

    for pattern, section, fix_desc in PROMPT_BUG_PATTERNS:
        if re.search(pattern, all_evidence, re.IGNORECASE):
            result.prompt_section = section
            result.prompt_fix_hint = fix_desc
            result.evidence.append(f"prompt_section={section}: {fix_desc}")
            return


def classify_failures(
    tasks: list[AgentTask],
    grades: list[GradeResult],
) -> list[RootCauseResult]:
    grade_by_task = {grade.task_id: grade for grade in grades}
    failures: list[RootCauseResult] = []
    for task in tasks:
        grade = grade_by_task[task.task_id]
        if grade.grade in {"BROKEN", "WEAK"}:
            classification = classify_task(task, grade)
            if classification is not None:
                failures.append(classification)
    return failures


def summarize_root_causes(results: list[RootCauseResult]) -> dict[str, Any]:
    grouped: dict[str, list[RootCauseResult]] = defaultdict(list)
    for result in results:
        grouped[result.root_cause].append(result)
    summary: dict[str, Any] = {}
    total = max(1, len(results))
    _unknown = {"label": "Unknown", "description": "Custom root cause.", "fix_hint": "Investigate manually."}
    for cause, items in sorted(grouped.items(), key=lambda item: -len(item[1])):
        cause_meta = ROOT_CAUSES.get(cause, _unknown)
        summary[cause] = {
            "label": cause_meta["label"],
            "count": len(items),
            "percentage": round(len(items) * 100 / total, 1),
            "fix_hint": cause_meta["fix_hint"],
            "sample_task_ids": [item.task_id for item in items[:5]],
        }
    return summary


def format_root_causes_text(results: list[RootCauseResult]) -> str:
    if not results:
        return "No BROKEN or WEAK tasks found."
    _unknown = {"label": "Unknown", "description": "Custom root cause.", "fix_hint": "Investigate manually."}
    lines = ["ROOT CAUSE ANALYSIS", "=" * 60, ""]
    for cause, meta in summarize_root_causes(results).items():
        cause_meta = ROOT_CAUSES.get(cause, _unknown)
        lines.append(f"{cause}: {meta['count']} task(s) [{meta['percentage']}%]")
        lines.append(f"  {cause_meta['description']}")
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
