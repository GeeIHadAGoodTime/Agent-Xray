"""Data completeness analysis for agent traces.

Detects missing observability dimensions and warns about blind spots
that could cause false confidence in grading or root-cause classification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schema import AgentTask


@dataclass(slots=True)
class CompletenessWarning:
    """A single data completeness issue."""

    dimension: str
    severity: str  # "critical", "high", "medium", "low"
    message: str
    affected_pct: float  # 0.0-100.0
    fix_hint: str


@dataclass(slots=True)
class CompletenessReport:
    """Full completeness assessment for a set of tasks."""

    warnings: list[CompletenessWarning] = field(default_factory=list)
    dimensions_checked: int = 0
    dimensions_ok: int = 0

    @property
    def score(self) -> float:
        """0.0-1.0 completeness score."""
        if self.dimensions_checked == 0:
            return 0.0
        return self.dimensions_ok / self.dimensions_checked

    @property
    def score_pct(self) -> int:
        return int(self.score * 100)

    def has_critical(self) -> bool:
        return any(w.severity == "critical" for w in self.warnings)

    def format_text(self) -> str:
        if not self.warnings:
            return f"Data completeness: {self.score_pct}% ({self.dimensions_ok}/{self.dimensions_checked} dimensions)"
        lines = [
            f"Data completeness: {self.score_pct}% ({self.dimensions_ok}/{self.dimensions_checked} dimensions)",
            "",
        ]
        by_severity = {"critical": [], "high": [], "medium": [], "low": []}
        for w in self.warnings:
            by_severity.get(w.severity, by_severity["low"]).append(w)
        for severity in ("critical", "high", "medium", "low"):
            for w in by_severity[severity]:
                tag = severity.upper()
                lines.append(f"  [{tag}] {w.dimension}: {w.message}")
                lines.append(f"         Fix: {w.fix_hint}")
        return "\n".join(lines)


def check_completeness(tasks: list[AgentTask]) -> CompletenessReport:
    """Analyze tasks for missing observability dimensions.

    Returns a report with warnings about data gaps that could cause
    false confidence in grading or root-cause classification.
    """
    if not tasks:
        return CompletenessReport(dimensions_checked=0, dimensions_ok=0)

    warnings: list[CompletenessWarning] = []
    total_dims = 0
    ok_dims = 0

    # 1. Outcome records
    total_dims += 1
    tasks_with_outcome = sum(1 for t in tasks if t.outcome is not None)
    outcome_pct = tasks_with_outcome / len(tasks) * 100
    if outcome_pct < 50:
        warnings.append(CompletenessWarning(
            dimension="outcome_records",
            severity="critical",
            message=f"{100 - outcome_pct:.0f}% of tasks have no outcome record. "
                    f"Grades rely on self-reported status which is missing for most tasks.",
            affected_pct=100 - outcome_pct,
            fix_hint="Emit a task_complete event with outcome/status at the end of each task.",
        ))
    else:
        ok_dims += 1

    # 2. Tool schemas
    total_dims += 1
    has_tool_schemas = False
    for t in tasks:
        for step in t.steps[:3]:
            if step.extensions.get("tool_schemas"):
                has_tool_schemas = True
                break
        if has_tool_schemas:
            break
    if not has_tool_schemas:
        warnings.append(CompletenessWarning(
            dimension="tool_schemas",
            severity="critical",
            message="No tool schemas recorded. Tool description bugs, injected parameters, "
                    "and schema-level issues are completely invisible.",
            affected_pct=100.0,
            fix_hint="Log tool_schemas (or a hash + first-step dump) in your step logger.",
        ))
    else:
        ok_dims += 1

    # 3. Model name
    total_dims += 1
    steps_with_model = 0
    steps_with_real_model = 0
    for t in tasks:
        for step in t.steps:
            if step.model:
                steps_with_model += 1
                if step.model.model_name and step.model.model_name != "unknown":
                    steps_with_real_model += 1
    if steps_with_model > 0:
        unknown_pct = (1 - steps_with_real_model / steps_with_model) * 100
        if unknown_pct > 80:
            warnings.append(CompletenessWarning(
                dimension="model_name",
                severity="medium",
                message=f"{unknown_pct:.0f}% of steps have model_name='unknown'. "
                        f"Model comparison and A/B testing features are inoperable.",
                affected_pct=unknown_pct,
                fix_hint="Propagate the actual model identifier to the step logger.",
            ))
        else:
            ok_dims += 1
    else:
        ok_dims += 1

    # 4. Cache tokens
    total_dims += 1
    has_cache_tokens = any(
        step.model and (step.model.cache_read_tokens is not None or step.model.cache_creation_tokens is not None)
        for t in tasks
        for step in t.steps
    )
    if not has_cache_tokens:
        has_any_tokens = any(
            step.model and (step.model.input_tokens or step.model.output_tokens)
            for t in tasks
            for step in t.steps
        )
        if has_any_tokens:
            warnings.append(CompletenessWarning(
                dimension="cache_tokens",
                severity="high",
                message="Token counts present but cache_read_tokens/cache_creation_tokens missing. "
                        "Cost tracking cannot distinguish cache-warm from cold steps.",
                affected_pct=100.0,
                fix_hint="Include cache_read_tokens and cache_creation_tokens in llm_usage.",
            ))
        else:
            ok_dims += 1  # No token data at all — not a cache-specific gap
    else:
        ok_dims += 1

    # 5. Final answer
    total_dims += 1
    tasks_with_answer = sum(
        1 for t in tasks
        if t.outcome and t.outcome.final_answer and len(t.outcome.final_answer.strip()) > 0
    )
    if tasks_with_outcome > 0:
        answer_pct = tasks_with_answer / tasks_with_outcome * 100
        if answer_pct < 30:
            warnings.append(CompletenessWarning(
                dimension="final_answer",
                severity="high",
                message=f"Only {answer_pct:.0f}% of completed tasks have a final_answer. "
                        f"Outcome verification and answer quality analysis are impossible.",
                affected_pct=100 - answer_pct,
                fix_hint="Include final_answer text in the task_complete event.",
            ))
        else:
            ok_dims += 1
    else:
        ok_dims += 1  # Can't assess without outcomes

    # 6. System prompt visibility
    total_dims += 1
    has_prompt = any(
        "system_prompt_text" in t.metadata
        for t in tasks
    )
    if not has_prompt:
        warnings.append(CompletenessWarning(
            dimension="system_prompt",
            severity="medium",
            message="No system_prompt_text recorded. Prompt-level bugs require "
                    "manual inspection of source code.",
            affected_pct=100.0,
            fix_hint="Log system_prompt_text (or hash + full text on step 1) in step events.",
        ))
    else:
        ok_dims += 1

    # 7. Rejected tools usage
    total_dims += 1
    has_rejected_data = any(
        step.tools and step.tools.rejected_tools
        for t in tasks
        for step in t.steps
    )
    if not has_rejected_data:
        warnings.append(CompletenessWarning(
            dimension="rejected_tools",
            severity="low",
            message="No rejected_tools data. Cannot detect tool-rejection-caused failures.",
            affected_pct=100.0,
            fix_hint="Log rejected_tools when tools are filtered by policy or routing.",
        ))
    else:
        ok_dims += 1

    # 8. Approval path
    total_dims += 1
    has_approval = any(
        step.reasoning and step.reasoning.approval_path
        for t in tasks
        for step in t.steps
    )
    if not has_approval:
        warnings.append(CompletenessWarning(
            dimension="approval_path",
            severity="low",
            message="No approval_path data. Cannot distinguish risk_safe from risk_blocked steps.",
            affected_pct=100.0,
            fix_hint="Log approval_path for each step's risk classification.",
        ))
    else:
        ok_dims += 1

    # 9. Conversation history
    total_dims += 1
    tasks_with_turns_no_summary = sum(
        1 for t in tasks
        if t.metadata.get("prior_conversation_turns", 0) > 0
        and not t.metadata.get("prior_conversation_summary")
    )
    if tasks_with_turns_no_summary > 0:
        warnings.append(CompletenessWarning(
            dimension="conversation_history",
            severity="high",
            message="Tasks have prior conversation turns but no conversation summary. "
                    "Decision-surface replay for multi-turn conversations is incomplete.",
            affected_pct=tasks_with_turns_no_summary / len(tasks) * 100,
            fix_hint="Log prior_conversation_summary alongside prior_conversation_turns.",
        ))
    else:
        ok_dims += 1

    # 10. Temporal data (step durations)
    total_dims += 1
    total_steps = sum(len(t.steps) for t in tasks)
    steps_with_duration = sum(
        1 for t in tasks
        for step in t.steps
        if step.duration_ms is not None
    )
    if total_steps > 0:
        duration_pct = steps_with_duration / total_steps * 100
        if duration_pct < 50:
            warnings.append(CompletenessWarning(
                dimension="step_durations",
                severity="medium",
                message=f"{100 - duration_pct:.0f}% of steps lack duration_ms. "
                        f"Temporal analysis (bottleneck detection, rate limiting patterns) is limited.",
                affected_pct=100 - duration_pct,
                fix_hint="Record duration_ms on every step event.",
            ))
        else:
            ok_dims += 1
    else:
        ok_dims += 1

    # 11. System context components
    total_dims += 1
    has_system_context = any(
        "system_context_components" in t.metadata
        for t in tasks
    )
    if not has_system_context:
        warnings.append(CompletenessWarning(
            dimension="system_context",
            severity="medium",
            message="No system_context_components recorded. Frustration-correlated failures, "
                    "user-model mismatches, and context injection bugs are invisible.",
            affected_pct=100.0,
            fix_hint="Log system_context_components with subfields "
                     "(playback, memory, frustration, user_model, delivery_address) on step 1.",
        ))
    else:
        ok_dims += 1

    # 12. LLM reasoning trace
    total_dims += 1
    steps_with_reasoning = sum(
        1 for t in tasks
        for step in t.steps
        if step.reasoning and step.reasoning.llm_reasoning
    )
    if total_steps > 0:
        reasoning_pct = steps_with_reasoning / total_steps * 100
        if reasoning_pct < 30:
            warnings.append(CompletenessWarning(
                dimension="llm_reasoning",
                severity="medium",
                message=f"{100 - reasoning_pct:.0f}% of steps have no reasoning trace. "
                        f"Cannot diagnose why the model made specific tool choices.",
                affected_pct=100 - reasoning_pct,
                fix_hint="Capture llm_reasoning (the model's chain-of-thought) in step events.",
            ))
        else:
            ok_dims += 1
    else:
        ok_dims += 1

    # 13. Step count consistency (outcome.total_steps vs actual steps)
    total_dims += 1
    ghost_tasks = sum(
        1 for t in tasks
        if t.outcome and getattr(t.outcome, "total_steps", None)
        and t.outcome.total_steps > 0 and len(t.steps) == 0
    )
    if ghost_tasks > 0:
        warnings.append(CompletenessWarning(
            dimension="step_data_loss",
            severity="critical",
            message=f"{ghost_tasks} task(s) have outcome.total_steps > 0 but zero step records. "
                    f"Step events were lost or written to a different file.",
            affected_pct=ghost_tasks / len(tasks) * 100,
            fix_hint="Ensure step events and outcome events are written to the same log file.",
        ))
    else:
        ok_dims += 1

    return CompletenessReport(
        warnings=warnings,
        dimensions_checked=total_dims,
        dimensions_ok=ok_dims,
    )
