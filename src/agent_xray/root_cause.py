from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .analyzer import analyze_task
from .grader import GradeResult
from .schema import AgentStep, AgentTask

if TYPE_CHECKING:
    from .analyzer import TaskAnalysis

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
    "delegation_failure": {
        "label": "Delegation Failure",
        "description": "A multi-agent workflow failed because delegation tools returned errors.",
        "fix_hint": "Inspect delegation tool reliability and sub-agent orchestration error handling.",
    },
    "test_failure_loop": {
        "label": "Test Failure Loop",
        "description": "A coding agent kept rerunning the same failing tests without changing course.",
        "fix_hint": "Add stronger failure triage before rerunning tests and require a new fix attempt.",
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
    "insufficient_sources": {
        "label": "Insufficient Sources",
        "description": "A research task answered before gathering enough independent evidence.",
        "fix_hint": "Require more source gathering before synthesis and enforce source diversity.",
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
    "memory_overload": {
        "label": "Memory Overload",
        "description": "Context pressure likely degraded the agent's later reasoning or output quality.",
        "fix_hint": "Trim context earlier, compact more selectively, or decompose the task.",
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

BASELINE_CONFIDENCE_SCORES = {"high": 0.9, "medium": 0.6, "low": 0.3}

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
TEST_TOOL_MARKERS = ("pytest", "test", "unit_test", "integration_test")
FILE_OP_MARKERS = ("file", "write", "read", "edit", "patch")
DELEGATION_TOOL_MARKERS = (
    "spawn_agent",
    "delegate",
    "delegation",
    "subagent",
    "sub_agent",
    "worker",
    "explorer",
    "wait_agent",
    "send_input",
)
TEST_FAILURE_RE = re.compile(
    r"\b(?:\d+\s+failed\b|failed\b|failure\b|traceback\b|assert(?:ion)?error\b|error:)\b",
    re.IGNORECASE,
)
TEST_SUCCESS_RE = re.compile(
    r"\b(?:all tests passed|0 failed|no tests failed)\b",
    re.IGNORECASE,
)
DELEGATION_ERROR_RE = re.compile(
    r"\b(?:error|failed|failure|unable|timeout|timed out|rejected)\b",
    re.IGNORECASE,
)
CONTEXT_PRESSURE_RE = re.compile(
    r"\b(?:context(?: window)?(?: is| was)? full|running out of context|out of context|"
    r"lost track|losing track|forget(?:ting)? earlier|truncated|compacted|evicted|"
    r"too much context)\b",
    re.IGNORECASE,
)

PROMPT_BUG_PATTERNS = [
    (
        r"web_search.*browser|search.*instead of browse",
        "research",
        "LLM chose web_search when browser was open - clarify tool priority in research/search prompt section",
    ),
    (
        r"hallucinated|unknown.tool",
        "tools",
        "LLM called nonexistent tools - add explicit tool availability guardrails",
    ),
    (
        r"stuck on one page|same page.*\d+ steps",
        "browser",
        "LLM repeated actions on same page - strengthen progress detection in browser prompt section",
    ),
    (
        r"only \d+ unique tool",
        "tools",
        "Low tool diversity - sharpen tool priority ordering in tool descriptions",
    ),
    (
        r"payment|checkout.*never filled|reached checkout.*no fill",
        "payment",
        "Reached checkout but did not fill payment - clarify payment fill instructions",
    ),
    (
        r"tried.*different.*approach|backtrack|going back",
        "planning",
        "LLM tried to backtrack - strengthen planning strategy guidance",
    ),
    (
        r"no.*result|empty.*response|returned nothing",
        "tools",
        "Tool returned empty - add result validation and retry guidance",
    ),
    (
        r"already.*tried|tried.*before|same.*action.*again",
        "browser",
        "LLM repeated failed approach - add progress memory instructions",
    ),
    (
        r"too many.*steps|running out|context.*full",
        "planning",
        "Agent aware of resource limits - improve task decomposition guidance",
    ),
    (
        r"not sure which|multiple.*options|could.*either",
        "tools",
        "Tool selection uncertainty - sharpen tool descriptions and priority",
    ),
    (
        r"page.*changed|unexpected.*layout|different.*from",
        "browser",
        "Page layout mismatch - update selector strategies",
    ),
]

ClassificationDecision = tuple[str, str, list[str]]


def _clamp_confidence_score(value: float) -> float:
    """Clamp a numeric confidence score into the supported 0.0-1.0 range."""

    return max(0.0, min(1.0, float(value)))


def _baseline_confidence_score(label: str) -> float:
    """Return the baseline numeric confidence for a compatibility label."""

    return BASELINE_CONFIDENCE_SCORES.get(label.lower(), BASELINE_CONFIDENCE_SCORES["medium"])


def _confidence_label_from_score(score: float) -> str:
    """Map a numeric confidence score back to the legacy string buckets."""

    normalized = _clamp_confidence_score(score)
    if normalized >= 0.8:
        return "high"
    if normalized >= 0.5:
        return "medium"
    return "low"


def _score_confidence(label: str, evidence: list[str]) -> float:
    """Derive a numeric confidence score from a baseline label and evidence count."""

    baseline = _baseline_confidence_score(label)
    bonus = min(max(0, len(evidence) - 1) * 0.05, 0.1)
    return _clamp_confidence_score(baseline + bonus)


def _normalize_text(value: str | None) -> str:
    """Normalize free-form text for signature matching."""

    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _normalize_usage_pct(value: float | None) -> float:
    """Normalize context usage metrics to percentage units."""

    if value is None:
        return 0.0
    usage = float(value)
    if usage <= 1.0:
        usage *= 100.0
    return usage


@dataclass(slots=True)
class RootCauseResult:
    """Structured root-cause classification for a single graded task."""

    task_id: str
    root_cause: str
    grade: str
    score: int
    confidence: str | float
    confidence_score: float = -1.0
    evidence: list[str] = field(default_factory=list)
    site_name: str = ""
    error_kinds: dict[str, int] = field(default_factory=dict)
    prompt_section: str | None = None
    prompt_fix_hint: str | None = None

    def __post_init__(self) -> None:
        """Normalize legacy and numeric confidence fields into a consistent shape."""

        if self.confidence_score >= 0:
            resolved_score = _clamp_confidence_score(self.confidence_score)
        elif isinstance(self.confidence, (int, float)) and not isinstance(self.confidence, bool):
            resolved_score = _clamp_confidence_score(float(self.confidence))
        else:
            resolved_score = _baseline_confidence_score(str(self.confidence))
        self.confidence_score = resolved_score
        self.confidence = _confidence_label_from_score(resolved_score)


def _available_tools(task: AgentTask) -> set[str]:
    """Return every tool name exposed across the task steps."""

    names: set[str] = set()
    for step in task.steps:
        names.update(step.tools_available_names or [])
    return names


def _browser_tool_available(task: AgentTask) -> bool:
    """Return whether any browser-style tool was exposed to the agent."""

    return any(name.startswith(("browser_", "desktop_")) for name in _available_tools(task))


def _used_browser_tool(task: AgentTask) -> bool:
    """Return whether the agent actually used a browser-style tool."""

    return any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.steps)


def _used_only_search_like_tools(task: AgentTask) -> bool:
    """Return whether the task exclusively used search or read style tools."""

    if not task.steps:
        return False
    return all(
        any(marker in step.tool_name.lower() for marker in SEARCH_TOOL_MARKERS)
        for step in task.steps
    )


def _has_prompt_confusion(task: AgentTask) -> bool:
    """Return whether reasoning text explicitly signals confusion or uncertainty."""

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


def _is_test_tool(step: AgentStep) -> bool:
    """Return whether a step is a test execution tool call."""

    tool = step.tool_name.lower()
    if any(marker in tool for marker in TEST_TOOL_MARKERS):
        return True
    command = _normalize_text(str(step.tool_input.get("command", "")))
    return bool(command and any(marker in command for marker in ("pytest", "unittest", "nosetests")))


def _is_file_operation(step: AgentStep) -> bool:
    """Return whether a step likely changed or inspected project files."""

    tool = step.tool_name.lower()
    return any(marker in tool for marker in FILE_OP_MARKERS)


def _looks_like_failed_test(step: AgentStep) -> bool:
    """Return whether a test execution step appears to have failed."""

    text = _normalize_text(step.error or step.tool_result)
    return bool(text and TEST_FAILURE_RE.search(text) and not TEST_SUCCESS_RE.search(text))


def _test_failure_signature(step: AgentStep) -> str:
    """Build a normalized signature for comparing repeated failing test outputs."""

    text = _normalize_text(step.error or step.tool_result)
    text = re.sub(r"\b\d+\s+passed\b", "", text)
    text = re.sub(r"\b\d+\s+warnings?\b", "", text)
    text = re.sub(r"\bcollected\s+\d+\s+items?\b", "", text)
    return text[:160]


def _is_delegation_tool(step: AgentStep) -> bool:
    """Return whether a step appears to be part of a multi-agent delegation workflow."""

    tool = step.tool_name.lower()
    return any(marker in tool for marker in DELEGATION_TOOL_MARKERS)


def _looks_like_delegation_failure(step: AgentStep) -> bool:
    """Return whether a delegation tool call returned an error-like outcome."""

    if step.error:
        return True
    return bool(DELEGATION_ERROR_RE.search(step.tool_result or ""))


def _final_output_is_short(task: AgentTask, *, short_limit: int) -> bool:
    """Return whether the final answer-like step is unusually terse."""

    answer_steps = [
        step
        for step in task.sorted_steps
        if step.tool_name.lower() in {"respond", "answer", "summarize", "write"}
        and step.tool_result
    ]
    if not answer_steps:
        return False
    final_text = (answer_steps[-1].tool_result or "").strip()
    if len(final_text) > short_limit:
        return False
    previous_lengths = [
        len((step.tool_result or "").strip()) for step in answer_steps[:-1] if step.tool_result
    ]
    return not previous_lengths or max(previous_lengths) >= len(final_text) * 2


def _memory_quality_drop_evidence(
    task: AgentTask,
    *,
    high_usage_steps: list[AgentStep],
    short_output_chars: int,
) -> list[str]:
    """Return evidence that later task quality degraded under context pressure."""

    evidence: list[str] = []
    sorted_steps = task.sorted_steps
    halfway = len(sorted_steps) // 2
    late_steps = sorted_steps[halfway:] if halfway else sorted_steps
    early_steps = sorted_steps[:halfway]
    late_errors = sum(1 for step in late_steps if step.error)
    early_errors = sum(1 for step in early_steps if step.error)

    if any(
        CONTEXT_PRESSURE_RE.search(
            " ".join(filter(None, [step.llm_reasoning, step.error, step.tool_result]))
        )
        for step in high_usage_steps
    ):
        evidence.append("reasoning or results explicitly mention context pressure")
    if any(_has_prompt_confusion(AgentTask(task_id=task.task_id, steps=[step])) for step in late_steps):
        evidence.append("late-step reasoning shows confusion after context usage spiked")
    if late_errors > early_errors and late_errors > 0:
        evidence.append(f"late-stage errors increased from {early_errors} to {late_errors}")
    if any(
        step.model
        and (
            (step.model.compaction_count or 0) > 0
            or (step.model.trimmed_messages or 0) > 0
            or (step.model.fifo_evicted_messages or 0) > 0
            or (step.model.screenshots_evicted or 0) > 0
        )
        for step in high_usage_steps
    ):
        evidence.append("high-context steps also triggered compaction or message eviction")
    if _final_output_is_short(task, short_limit=short_output_chars):
        evidence.append("final answer became unusually short under pressure")
    return evidence


def _apply_classification(
    result: RootCauseResult,
    *,
    root_cause: str,
    confidence: str,
    evidence: list[str],
) -> RootCauseResult:
    """Populate a result object with the chosen root cause and confidence values."""

    result.root_cause = root_cause
    result.evidence.extend(evidence)
    result.confidence_score = _score_confidence(confidence, result.evidence)
    result.confidence = _confidence_label_from_score(result.confidence_score)
    return result

@dataclass(slots=True)
class ClassificationConfig:
    """Configurable thresholds for root cause classification heuristics."""

    spin_threshold: int = 5
    high_error_rate: float = 0.5
    model_limit_steps: int = 50
    stuck_loop_min_steps: int = 5
    early_abort_max_steps: int = 3
    test_failure_loop_min_runs: int = 2
    insufficient_sources_min_searches: int = 2
    low_source_diversity_threshold: int = 2
    memory_overload_usage_pct: float = 85.0
    memory_overload_short_output_chars: int = 80


_DEFAULT_CONFIG = ClassificationConfig()


def _classify_routing_bug(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify tasks that were never given the tools they needed."""

    del task, config
    if analysis.no_tools_steps:
        return ("routing_bug", "high", [f"{analysis.no_tools_steps} step(s) exposed zero tools"])
    return None


def _classify_approval_block(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify failures caused by approval or permission gates."""

    del task, config
    if analysis.error_kinds.get("approval_block"):
        return (
            "approval_block",
            "high",
            [f"{analysis.error_kinds['approval_block']} approval-blocked step(s)"],
        )
    return None


def _classify_delegation_failure(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify multi-agent workflows whose delegation tools failed."""

    del analysis, config
    delegation_steps = [step for step in task.sorted_steps if _is_delegation_tool(step)]
    if not delegation_steps:
        return None
    failed_steps = [step for step in delegation_steps if _looks_like_delegation_failure(step)]
    if not failed_steps:
        return None
    evidence = [
        f"{len(failed_steps)} delegation step(s) returned errors",
        f"delegation tools seen: {', '.join(sorted({step.tool_name for step in delegation_steps}))}",
    ]
    if failed_steps[0].error:
        evidence.append(f"sample delegation error: {_normalize_text(failed_steps[0].error)[:120]}")
    return ("delegation_failure", "high", evidence)


def _classify_test_failure_loop(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify coding traces that keep rerunning the same failing tests."""

    coding = analysis.signal_metrics.get("coding", {})
    if int(coding.get("test_runs", 0)) < config.test_failure_loop_min_runs:
        return None

    test_steps = [
        step for step in task.sorted_steps if _is_test_tool(step) and _looks_like_failed_test(step)
    ]
    if len(test_steps) < config.test_failure_loop_min_runs:
        return None

    by_signature: Counter[str] = Counter()
    repeated_without_edit = False
    previous_for_signature: dict[str, int] = {}
    sorted_steps = task.sorted_steps
    step_positions = {id(step): index for index, step in enumerate(sorted_steps)}

    for step in test_steps:
        signature = _test_failure_signature(step)
        if not signature:
            continue
        by_signature[signature] += 1
        current_index = step_positions[id(step)]
        if signature in previous_for_signature:
            prior_index = previous_for_signature[signature]
            between = sorted_steps[prior_index + 1 : current_index]
            if not any(_is_file_operation(candidate) for candidate in between):
                repeated_without_edit = True
        previous_for_signature[signature] = current_index

    if not by_signature:
        return None
    signature, count = by_signature.most_common(1)[0]
    if count < config.test_failure_loop_min_runs:
        return None

    evidence = [f"same failing test signature repeated {count} times"]
    if repeated_without_edit:
        evidence.append("reran failing tests without an intervening file edit")
    ratio = float(coding.get("test_to_edit_ratio", 0.0))
    if ratio >= 1.5:
        evidence.append(f"test-to-edit ratio stayed high at {ratio:.1f}")
    if signature:
        evidence.append(f"failure signature: {signature[:100]}")
    return ("test_failure_loop", "high", evidence)


def _classify_spin(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify repeated actions that exceed the generic spin threshold."""

    del task
    if analysis.max_repeat_count >= config.spin_threshold:
        return (
            "spin",
            "high",
            [f"{analysis.max_repeat_tool} repeated {analysis.max_repeat_count} times"],
        )
    return None


def _classify_error_dominance(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify high-error traces as either environment drift or tool bugs."""

    del task
    if analysis.error_rate <= config.high_error_rate or analysis.errors <= 1:
        return None
    env_errors = sum(
        analysis.error_kinds.get(key, 0) for key in ("timeout", "click_fail", "not_found")
    )
    tool_errors = sum(
        analysis.error_kinds.get(key, 0) for key in ("unknown_tool", "validation", "other")
    )
    if env_errors >= tool_errors:
        confidence = "medium" if env_errors else "low"
        return (
            "environment_drift",
            confidence,
            [f"environment-style errors dominate: {analysis.error_kinds}"],
        )
    return ("tool_bug", "medium", [f"tool errors dominate: {analysis.error_kinds}"])


def _classify_insufficient_sources(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify research traces that answered without enough independent sourcing."""

    research = analysis.signal_metrics.get("research", {})
    search_count = int(research.get("search_count", 0))
    source_diversity = int(research.get("source_diversity", 0))
    looks_like_research = bool(
        search_count
        or (task.task_category or "").lower() == "research"
        or (research.get("has_synthesis_step") and int(research.get("read_count", 0)) > 0)
    )
    if not looks_like_research:
        return None
    if search_count >= config.insufficient_sources_min_searches:
        return None
    if source_diversity >= config.low_source_diversity_threshold:
        return None
    evidence = [
        f"search_count={search_count} below {config.insufficient_sources_min_searches}",
        f"source_diversity={source_diversity} below {config.low_source_diversity_threshold}",
    ]
    if research.get("has_synthesis_step"):
        evidence.append("agent synthesized an answer before gathering enough sources")
    return ("insufficient_sources", "high", evidence)


def _classify_tool_selection_from_search_bias(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify cases where the agent ignored available browser tools."""

    del analysis, config
    if _browser_tool_available(task) and not _used_browser_tool(task) and _used_only_search_like_tools(task):
        return (
            "tool_selection_bug",
            "high",
            ["browser tools were available but the agent stayed on search/read tools"],
        )
    return None


def _classify_memory_overload(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify failures that correlate with high context usage and later degradation."""

    del analysis
    high_usage_steps = [
        step
        for step in task.sorted_steps
        if _normalize_usage_pct(step.context_usage_pct) >= config.memory_overload_usage_pct
    ]
    if not high_usage_steps:
        return None
    evidence = _memory_quality_drop_evidence(
        task,
        high_usage_steps=high_usage_steps,
        short_output_chars=config.memory_overload_short_output_chars,
    )
    if not evidence:
        return None
    peak_usage = max(_normalize_usage_pct(step.context_usage_pct) for step in high_usage_steps)
    return (
        "memory_overload",
        "medium",
        [f"context usage peaked at {peak_usage:.1f}%"] + evidence,
    )


def _classify_prompt_confusion(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify tasks whose reasoning explicitly shows prompt confusion."""

    del analysis, config
    if _has_prompt_confusion(task):
        return ("prompt_bug", "high", ["reasoning shows confusion about what to do next"])
    return None


def _classify_model_limit(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify traces that are long and stagnant enough to suggest model limits."""

    del task
    if analysis.step_count > config.model_limit_steps and len(analysis.unique_urls) < 2:
        return (
            "model_limit",
            "medium",
            [f"{analysis.step_count} steps with only {len(analysis.unique_urls)} URL(s)"],
        )
    return None


def _classify_stuck_loop(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify traces that remain on one page or state while continuing to act."""

    del task
    if len(analysis.unique_urls) <= 1 and analysis.step_count >= config.stuck_loop_min_steps:
        return ("stuck_loop", "medium", ["stayed on one page while continuing to act"])
    return None


def _classify_early_abort(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify tasks that ended before collecting enough evidence."""

    del task
    if analysis.step_count < config.early_abort_max_steps:
        return ("early_abort", "medium", ["task ended too early to gather evidence"])
    return None


def _classify_reasoning_bug(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify progressing traces that still fail without obvious tool faults."""

    del task, config
    if analysis.errors == 0 and len(analysis.unique_tools) > 1 and len(analysis.unique_urls) > 1:
        return (
            "reasoning_bug",
            "medium",
            ["task progressed but still graded poorly without hard tool failures"],
        )
    return None


def _classify_tool_selection_from_low_diversity(
    task: AgentTask,
    analysis: TaskAnalysis,
    config: ClassificationConfig,
) -> ClassificationDecision | None:
    """Classify navigation-heavy traces that never broadened tool usage."""

    del task, config
    if len(analysis.unique_tools) <= 1 and len(analysis.unique_urls) > 1:
        return ("tool_selection_bug", "medium", ["low tool diversity despite navigation progress"])
    return None


def classify_task(
    task: AgentTask,
    grade: GradeResult,
    analysis: TaskAnalysis | None = None,
    config: ClassificationConfig | None = None,
) -> RootCauseResult | None:
    """Classify a single BROKEN or WEAK task into its most likely root cause."""

    if grade.grade not in {"BROKEN", "WEAK"}:
        return None
    cfg = config or _DEFAULT_CONFIG
    analysis = analysis or analyze_task(task)
    result = RootCauseResult(
        task_id=task.task_id,
        root_cause="prompt_bug",
        grade=grade.grade,
        score=grade.score,
        confidence="medium",
        evidence=[],
        site_name=analysis.site_name,
        error_kinds=analysis.error_kinds,
    )
    for classifier in (
        _classify_routing_bug,
        _classify_approval_block,
        _classify_delegation_failure,
        _classify_test_failure_loop,
        _classify_spin,
        _classify_error_dominance,
        _classify_insufficient_sources,
        _classify_tool_selection_from_search_bias,
        _classify_memory_overload,
        _classify_prompt_confusion,
        _classify_model_limit,
        _classify_stuck_loop,
        _classify_early_abort,
        _classify_reasoning_bug,
        _classify_tool_selection_from_low_diversity,
    ):
        decision = classifier(task, analysis, cfg)
        if decision is not None:
            root_cause, confidence, evidence = decision
            return _apply_classification(
                result,
                root_cause=root_cause,
                confidence=confidence,
                evidence=evidence,
            )

    _enrich_prompt_bug(result, task, analysis)
    result.evidence.append("fallback classification after excluding stronger operational causes")
    result.confidence_score = _score_confidence("medium", result.evidence)
    result.confidence = _confidence_label_from_score(result.confidence_score)
    return result


def _enrich_prompt_bug(
    result: RootCauseResult,
    task: AgentTask,
    analysis: TaskAnalysis,
) -> None:
    """Attach prompt-section hints to prompt-bug classifications when possible."""

    if result.root_cause != "prompt_bug":
        return
    all_evidence = " ".join(result.evidence)
    has_browser = any(step.tool_name.startswith(("browser_", "desktop_")) for step in task.steps)
    has_search = any("search" in step.tool_name.lower() for step in task.steps)
    if has_browser and has_search:
        all_evidence += " web_search with browser open"
    if analysis.hallucinated_tools:
        all_evidence += " hallucinated unknown tool"
    if len(analysis.unique_urls) <= 1 and analysis.step_count >= 5:
        all_evidence += f" stuck on one page for {analysis.step_count} steps"
    if len(analysis.unique_tools) <= 2 and analysis.step_count >= 8:
        all_evidence += f" only {len(analysis.unique_tools)} unique tools"
    commerce = analysis.signal_metrics.get("commerce", {})
    if (
        commerce.get("reached_checkout")
        and not commerce.get("reached_payment")
        and analysis.step_count >= 10
    ):
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
    """Classify every BROKEN or WEAK task in a batch."""

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
    """Aggregate classified failures into counts and percentages by cause."""

    grouped: dict[str, list[RootCauseResult]] = defaultdict(list)
    for result in results:
        grouped[result.root_cause].append(result)
    summary: dict[str, Any] = {}
    total = max(1, len(results))
    unknown = {
        "label": "Unknown",
        "description": "Custom root cause.",
        "fix_hint": "Investigate manually.",
    }
    for cause, items in sorted(grouped.items(), key=lambda item: -len(item[1])):
        cause_meta = ROOT_CAUSES.get(cause, unknown)
        summary[cause] = {
            "label": cause_meta["label"],
            "count": len(items),
            "percentage": round(len(items) * 100 / total, 1),
            "fix_hint": cause_meta["fix_hint"],
            "sample_task_ids": [item.task_id for item in items[:5]],
        }
    return summary


def format_root_causes_text(results: list[RootCauseResult]) -> str:
    """Render grouped root-cause results as plain text."""

    if not results:
        return "No BROKEN or WEAK tasks found."
    unknown = {
        "label": "Unknown",
        "description": "Custom root cause.",
        "fix_hint": "Investigate manually.",
    }
    lines = ["ROOT CAUSE ANALYSIS", "=" * 60, ""]
    for cause, meta in summarize_root_causes(results).items():
        cause_meta = ROOT_CAUSES.get(cause, unknown)
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



