from __future__ import annotations

import difflib
import json
from dataclasses import asdict
from typing import Any

from .analyzer import build_task_tree, summarize_tool_result
from .protocols import PromptBuilder, ToolRegistry
from .schema import AgentTask

_SURFACE_COMPLETENESS_FIELDS = (
    "llm_reasoning",
    "model_name",
    "temperature",
    "tool_choice",
    "prompt_variant",
    "tools_available_names",
    "tools_available_count",
    "rejected_tools",
    "focused_set",
    "message_count",
    "conversation_turn_count",
    "system_prompt_hash",
    "context_usage_pct",
    "context_window",
    "compaction",
    "correction_messages",
    "intervention_signals",
    "approval_path",
    "page_url",
    "screenshot_state",
    "snapshot_compression",
    "memory",
    "rag",
)


def _prompt_text(task: AgentTask, prompt_builder: PromptBuilder | None) -> str | None:
    if isinstance(task.metadata.get("system_prompt_text"), str):
        return str(task.metadata["system_prompt_text"])
    if task.steps:
        first = task.sorted_steps[0]
        ext_prompt = first.extensions.get("system_prompt_text")
        if isinstance(ext_prompt, str):
            return ext_prompt
    if prompt_builder is not None:
        return prompt_builder.build_prompt(task)
    return None


def _system_components(task: AgentTask) -> dict[str, Any] | None:
    if isinstance(task.metadata.get("system_context_components"), dict):
        return task.metadata["system_context_components"]  # type: ignore[no-any-return]
    if task.steps:
        first = task.sorted_steps[0]
        comp = first.extensions.get("system_context_components")
        if isinstance(comp, dict):
            return comp
    return None


def _window_history(
    history: list[dict[str, str]],
    *,
    max_history_steps: int,
) -> list[dict[str, str]]:
    if max_history_steps <= 0:
        return []
    if len(history) <= max_history_steps:
        return list(history)
    if max_history_steps < 4:
        return list(history[-max_history_steps:])
    omitted = len(history) - max_history_steps
    marker = {"role": "history_window", "content": f"[...{omitted} steps omitted...]"}
    return [
        *history[:3],
        marker,
        *history[-(max_history_steps - 3) :],
    ]


def _step_extension_value(
    step: Any,
    field_name: str,
    *,
    nested_container: str | None = None,
    nested_aliases: tuple[str, ...] = (),
) -> Any:
    value = getattr(step, field_name, None)
    if value is not None:
        return value
    if field_name in step.extensions:
        return step.extensions[field_name]
    if nested_container is None:
        return None
    container = step.extensions.get(nested_container)
    if not isinstance(container, dict):
        return None
    if field_name in container:
        return container[field_name]
    for alias in nested_aliases:
        if alias in container:
            return container[alias]
    return None


def _memory_rag_fields(step: Any) -> dict[str, Any]:
    return {
        "memory_query": _step_extension_value(
            step,
            "memory_query",
            nested_container="memory",
            nested_aliases=("query",),
        ),
        "memory_results": _step_extension_value(
            step,
            "memory_results",
            nested_container="memory",
            nested_aliases=("results",),
        ),
        "memory_store_key": _step_extension_value(
            step,
            "memory_store_key",
            nested_container="memory",
            nested_aliases=("store_key", "key"),
        ),
        "rag_query": _step_extension_value(
            step,
            "rag_query",
            nested_container="rag",
            nested_aliases=("query",),
        ),
        "rag_documents_count": _step_extension_value(
            step,
            "rag_documents_count",
            nested_container="rag",
            nested_aliases=("documents_count", "document_count"),
        ),
        "rag_relevance_scores": _step_extension_value(
            step,
            "rag_relevance_scores",
            nested_container="rag",
            nested_aliases=("relevance_scores", "scores"),
        ),
    }


def _surface_presence(
    *,
    reasoning: Any,
    model: Any,
    tools: Any,
    browser: Any,
    has_tool_surface: bool,
    memory_rag: dict[str, Any],
) -> dict[str, bool]:
    compaction_present = bool(
        model
        and (
            model.compaction_count is not None
            or model.compaction_method is not None
            or model.compaction_messages_before is not None
            or model.compaction_messages_after is not None
            or model.compaction_summary_preview is not None
            or model.trimmed_messages is not None
            or model.fifo_evicted_messages is not None
            or model.screenshots_evicted is not None
        )
    )
    interventions_present = bool(
        reasoning
        and (
            reasoning.spin_intervention is not None
            or reasoning.error_registry_context is not None
            or reasoning.continuation_nudge is not None
            or reasoning.force_termination is not None
            or reasoning.hard_loop_breaker is not None
            or reasoning.consecutive_failure_warning is not None
        )
    )
    screenshot_state_present = bool(
        browser and (browser.had_screenshot is not None or browser.had_screenshot_image is not None)
    )
    snapshot_compression_present = bool(
        browser
        and (
            browser.snapshot_compressed is not None or browser.snapshot_pre_compress_len is not None
        )
    )
    memory_present = any(
        value is not None for key, value in memory_rag.items() if key.startswith("memory_")
    )
    rag_present = any(
        value is not None for key, value in memory_rag.items() if key.startswith("rag_")
    )
    return {
        "llm_reasoning": bool(reasoning and reasoning.llm_reasoning),
        "model_name": bool(model and model.model_name is not None),
        "temperature": bool(model and model.temperature is not None),
        "tool_choice": bool(model and model.tool_choice is not None),
        "prompt_variant": bool(model and model.prompt_variant is not None),
        "tools_available_names": has_tool_surface,
        "tools_available_count": has_tool_surface,
        "rejected_tools": bool(tools and tools.rejected_tools is not None),
        "focused_set": bool(tools and tools.focused_set is not None),
        "message_count": bool(tools and tools.message_count is not None),
        "conversation_turn_count": bool(tools and tools.conversation_turn_count is not None),
        "system_prompt_hash": bool(tools and tools.system_prompt_hash is not None),
        "context_usage_pct": bool(model and model.context_usage_pct is not None),
        "context_window": bool(model and model.context_window is not None),
        "compaction": compaction_present,
        "correction_messages": bool(reasoning and reasoning.correction_messages is not None),
        "intervention_signals": interventions_present,
        "approval_path": bool(reasoning and reasoning.approval_path is not None),
        "page_url": bool(browser and browser.page_url is not None),
        "screenshot_state": screenshot_state_present,
        "snapshot_compression": snapshot_compression_present,
        "memory": memory_present,
        "rag": rag_present,
    }


def _completeness_metadata(
    *,
    reasoning: Any,
    model: Any,
    tools: Any,
    browser: Any,
    has_tool_surface: bool,
    memory_rag: dict[str, Any],
) -> tuple[float, list[str]]:
    present = _surface_presence(
        reasoning=reasoning,
        model=model,
        tools=tools,
        browser=browser,
        has_tool_surface=has_tool_surface,
        memory_rag=memory_rag,
    )
    missing_surfaces = [
        field_name for field_name in _SURFACE_COMPLETENESS_FIELDS if not present[field_name]
    ]
    completeness = 0.5 + 0.5 * (
        (len(_SURFACE_COMPLETENESS_FIELDS) - len(missing_surfaces))
        / len(_SURFACE_COMPLETENESS_FIELDS)
    )
    return round(completeness, 3), missing_surfaces


def _step_signature(step: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    return (
        step.get("tool_name"),
        step.get("tool_input"),
        step.get("page_url"),
        step.get("error"),
    )


def _alignment_entry(
    status: str,
    left_step: dict[str, Any] | None,
    right_step: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "status": status,
        "left_step": left_step.get("step") if left_step else None,
        "right_step": right_step.get("step") if right_step else None,
        "left_tool_name": left_step.get("tool_name") if left_step else None,
        "right_tool_name": right_step.get("tool_name") if right_step else None,
        "left_tool_input": left_step.get("tool_input") if left_step else None,
        "right_tool_input": right_step.get("tool_input") if right_step else None,
    }


def _aligned_steps(
    left_steps: list[dict[str, Any]],
    right_steps: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, float]:
    left_tool_names = [str(step.get("tool_name") or "") for step in left_steps]
    right_tool_names = [str(step.get("tool_name") or "") for step in right_steps]
    matcher = difflib.SequenceMatcher(a=left_tool_names, b=right_tool_names, autojunk=False)
    alignment: list[dict[str, Any]] = []
    divergence_point: dict[str, Any] | None = None
    exact_signature_matches = 0

    for tag, left_start, left_end, right_start, right_end in matcher.get_opcodes():
        if tag == "equal":
            for left_step, right_step in zip(
                left_steps[left_start:left_end],
                right_steps[right_start:right_end],
                strict=True,
            ):
                status = "match"
                if _step_signature(left_step) != _step_signature(right_step):
                    status = "changed"
                    if divergence_point is None:
                        divergence_point = _alignment_entry(status, left_step, right_step)
                else:
                    exact_signature_matches += 1
                alignment.append(_alignment_entry(status, left_step, right_step))
            continue

        left_block = left_steps[left_start:left_end]
        right_block = right_steps[right_start:right_end]
        block_size = max(len(left_block), len(right_block))
        for index in range(block_size):
            left_entry: dict[str, Any] | None = (
                left_block[index] if index < len(left_block) else None
            )
            right_entry: dict[str, Any] | None = (
                right_block[index] if index < len(right_block) else None
            )
            status = "replace"
            if left_entry is None:
                status = "insert"
            elif right_entry is None:
                status = "delete"
            entry = _alignment_entry(status, left_entry, right_entry)
            if divergence_point is None:
                divergence_point = entry
            alignment.append(entry)

    max_steps = max(len(left_steps), len(right_steps))
    exact_similarity = (exact_signature_matches / max_steps) if max_steps else 1.0
    similarity_score = round((matcher.ratio() + exact_similarity) / 2, 3)
    return alignment, divergence_point, similarity_score


def surface_for_task(
    task: AgentTask,
    *,
    prompt_builder: PromptBuilder | None = None,
    tool_registry: ToolRegistry | None = None,
    max_history_steps: int = 20,
) -> dict[str, Any]:
    """Reconstruct the per-step decision surface for a single task.

    Args:
        task: Task to inspect.
        prompt_builder: Optional fallback prompt builder when the trace did not
            capture system prompt text.
        tool_registry: Optional fallback registry used when steps do not record
            their tool surface explicitly.
        max_history_steps: Maximum history entries retained per step.

    Returns:
        dict[str, Any]: JSON-friendly surface data including prompt context,
        per-step tool visibility, browser state, and rolling history.
    """
    history: list[dict[str, str]] = (
        [{"role": "user", "content": task.task_text}] if task.task_text else []
    )
    steps: list[dict[str, Any]] = []
    for step in task.sorted_steps:
        model = step.model
        tools = step.tools
        reasoning = step.reasoning
        browser = step.browser
        tools_available = tools.tools_available if tools else None
        has_tool_metadata = tools_available is not None
        tool_names = list(tools_available) if tools_available is not None else []
        if not has_tool_metadata and tool_registry is not None:
            tool_names = tool_registry.tool_names(task=task, step=step)
        memory_rag = _memory_rag_fields(step)
        has_tool_surface = has_tool_metadata or tool_registry is not None
        completeness, missing_surfaces = _completeness_metadata(
            reasoning=reasoning,
            model=model,
            tools=tools,
            browser=browser,
            has_tool_surface=has_tool_surface,
            memory_rag=memory_rag,
        )
        entry = {
            # Core
            "step": step.step,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_result_summary": summarize_tool_result(step),
            "error": step.error,
            "timestamp": step.timestamp,
            "duration_ms": step.duration_ms,
            # LLM decision context
            "llm_reasoning": reasoning.llm_reasoning
            if reasoning and reasoning.llm_reasoning
            else "",
            "model_name": model.model_name if model else None,
            "temperature": model.temperature if model else None,
            "tool_choice": model.tool_choice if model else None,
            "model": asdict(model) if model else None,
            "prompt_variant": model.prompt_variant if model else None,
            # Tool availability — THE decision surface
            "tools_available_names": tool_names,
            "tools_available_count": (
                tools.tools_available_count
                if tools and tools.tools_available_count is not None
                else len(tool_names)
            ),
            "rejected_tools": tools.rejected_tools if tools else None,
            "focused_set": tools.focused_set if tools else None,
            # Context pressure
            "message_count": tools.message_count if tools else None,
            "conversation_turn_count": tools.conversation_turn_count if tools else None,
            "system_prompt_hash": tools.system_prompt_hash if tools else None,
            "context_usage_pct": model.context_usage_pct if model else None,
            "context_window": model.context_window if model else None,
            # Compaction & trimming
            "compaction_count": model.compaction_count if model else None,
            "compaction_method": model.compaction_method if model else None,
            "compaction_messages_before": model.compaction_messages_before if model else None,
            "compaction_messages_after": model.compaction_messages_after if model else None,
            "compaction_summary_preview": model.compaction_summary_preview if model else None,
            "trimmed_messages": model.trimmed_messages if model else None,
            "fifo_evicted_messages": model.fifo_evicted_messages if model else None,
            "screenshots_evicted": model.screenshots_evicted if model else None,
            # Dynamic injections
            "correction_messages": (
                reasoning.correction_messages if reasoning and reasoning.correction_messages else []
            ),
            "spin_intervention": reasoning.spin_intervention if reasoning else None,
            "error_registry_context": reasoning.error_registry_context if reasoning else None,
            "continuation_nudge": reasoning.continuation_nudge if reasoning else None,
            "force_termination": reasoning.force_termination if reasoning else None,
            "hard_loop_breaker": reasoning.hard_loop_breaker if reasoning else None,
            "consecutive_failure_warning": reasoning.consecutive_failure_warning
            if reasoning
            else None,
            "approval_path": reasoning.approval_path if reasoning else None,
            "llm_decision": reasoning.llm_decision if reasoning else None,
            # Browser state
            "page_url": browser.page_url if browser else None,
            "snapshot_compressed": browser.snapshot_compressed if browser else None,
            "had_screenshot": browser.had_screenshot if browser else None,
            "had_screenshot_image": browser.had_screenshot_image if browser else None,
            "snapshot_pre_compress_len": browser.snapshot_pre_compress_len if browser else None,
            "browser_tiers_used": browser.browser_tiers_used if browser else None,
            # Memory & retrieval
            "memory_query": memory_rag["memory_query"],
            "memory_results": memory_rag["memory_results"],
            "memory_store_key": memory_rag["memory_store_key"],
            "rag_query": memory_rag["rag_query"],
            "rag_documents_count": memory_rag["rag_documents_count"],
            "rag_relevance_scores": memory_rag["rag_relevance_scores"],
            # Instrumentation quality
            "completeness": completeness,
            "missing_surfaces": missing_surfaces,
            # Accumulated conversation
            "conversation_history": _window_history(
                history,
                max_history_steps=max_history_steps,
            ),
        }
        steps.append(entry)
        if reasoning and reasoning.llm_reasoning:
            history.append({"role": "assistant_reasoning", "content": reasoning.llm_reasoning})
        history.append(
            {
                "role": "tool_call",
                "content": (
                    f"{step.tool_name} "
                    f"{json.dumps(step.tool_input, sort_keys=True, ensure_ascii=True)}"
                ),
            }
        )
        if step.error:
            history.append({"role": "tool_error", "content": step.error})
        elif step.tool_result:
            history.append(
                {"role": "tool_result", "content": summarize_tool_result(step, limit=400)}
            )
    return {
        "task_id": task.task_id,
        "task_text": task.task_text,
        "task_category": task.task_category,
        "outcome": task.outcome.to_dict() if task.outcome else None,
        "prompt_text": _prompt_text(task, prompt_builder),
        "system_context_components": _system_components(task),
        "prior_conversation_summary": task.metadata.get("prior_conversation_summary"),
        "metadata": task.metadata,
        "steps": steps,
    }


def reasoning_for_task(
    task: AgentTask,
    *,
    prompt_builder: PromptBuilder | None = None,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Extract a reasoning-centric view from a task surface.

    Args:
        task: Task to inspect.
        prompt_builder: Optional fallback prompt builder.
        tool_registry: Optional fallback tool registry.

    Returns:
        dict[str, Any]: Simplified reasoning chain with decisions, results, and
        interventions for each step.
    """
    surface = surface_for_task(task, prompt_builder=prompt_builder, tool_registry=tool_registry)
    return {
        "task_id": surface["task_id"],
        "task_text": surface["task_text"],
        "outcome": surface["outcome"],
        "reasoning_chain": [
            {
                "step": step["step"],
                "reasoning": step["llm_reasoning"],
                "decision": {
                    "tool_name": step["tool_name"],
                    "tool_input": step["tool_input"],
                },
                "result_summary": step["tool_result_summary"],
                "error": step["error"],
                "spin_intervention": step["spin_intervention"],
                "correction_messages": step["correction_messages"],
            }
            for step in surface["steps"]
        ],
    }


def diff_tasks(
    left: AgentTask,
    right: AgentTask,
    *,
    prompt_builder: PromptBuilder | None = None,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Compare two tasks step by step and identify their divergence.

    Args:
        left: Baseline or left-hand task.
        right: Comparison or right-hand task.
        prompt_builder: Optional fallback prompt builder.
        tool_registry: Optional fallback tool registry.

    Returns:
        dict[str, Any]: Surface snapshots for both tasks plus divergence,
        alignment, and prompt-diff metadata.
    """
    left_surface = surface_for_task(
        left, prompt_builder=prompt_builder, tool_registry=tool_registry
    )
    right_surface = surface_for_task(
        right, prompt_builder=prompt_builder, tool_registry=tool_registry
    )
    step_alignment, divergence_point, similarity_score = _aligned_steps(
        left_surface["steps"],
        right_surface["steps"],
    )
    diverged_at = (
        None
        if divergence_point is None
        else (divergence_point["left_step"] or divergence_point["right_step"])
    )
    prompt_diff = list(
        difflib.unified_diff(
            (left_surface.get("prompt_text") or "").splitlines(),
            (right_surface.get("prompt_text") or "").splitlines(),
            fromfile=left.task_id,
            tofile=right.task_id,
            lineterm="",
        )
    )
    return {
        "left": left_surface,
        "right": right_surface,
        "diverged_at_step": diverged_at,
        "step_alignment": step_alignment,
        "similarity_score": similarity_score,
        "divergence_point": divergence_point,
        "prompt_diff": prompt_diff,
    }


def tree_for_tasks(tasks: list[AgentTask]) -> dict[str, dict[str, list[str]]]:
    """Group tasks into a ``day -> site -> task_ids`` tree.

    Args:
        tasks: Tasks to group.

    Returns:
        dict[str, dict[str, list[str]]]: Nested mapping grouped by day and then
        inferred site name.
    """
    return build_task_tree(tasks)


def format_surface_text(surface: dict[str, Any]) -> str:
    """Render a human-readable task surface view."""
    lines = [
        "=" * 72,
        f"AGENT XRAY SURFACE: {surface['task_id']}",
        "=" * 72,
    ]
    if surface.get("task_text"):
        lines.append(f"user: {surface['task_text']}")
    if surface.get("prompt_text"):
        prompt = str(surface["prompt_text"]).strip()
        lines.extend(["", "SYSTEM PROMPT:", prompt[:2000], ""])
    if surface.get("system_context_components"):
        lines.append("PROMPT COMPONENTS:")
        for key, value in surface["system_context_components"].items():
            lines.append(f"  {key}: {str(value)[:200]}")
        lines.append("")
    if surface.get("prior_conversation_summary"):
        lines.extend(["PRIOR CONTEXT:", surface["prior_conversation_summary"], ""])
    # Compute the common missing surfaces across all steps (show once, not per-step)
    _all_missing: list[set[str]] = [
        set(step.get("missing_surfaces") or []) for step in surface["steps"]
    ]
    _common_missing = set.intersection(*_all_missing) if _all_missing else set()
    if _common_missing:
        lines.append(
            f"Note: {len(_common_missing)} surfaces not recorded in any step: "
            + ", ".join(sorted(_common_missing))
        )
        lines.append("")
    for step in surface["steps"]:
        lines.extend(
            [
                "-" * 72,
                f"STEP {step['step']} [{step.get('timestamp') or ''}]",
            ]
        )
        model_parts = [
            f"model={step.get('model_name') or '?'}",
            f"temp={step.get('temperature')}",
            f"tool_choice={step.get('tool_choice') or '?'}",
        ]
        if step.get("prompt_variant"):
            model_parts.append(f"prompt={step['prompt_variant']}")
        lines.append(f"model: {' '.join(model_parts)}")
        tool_count = step.get("tools_available_count") or len(
            step.get("tools_available_names") or []
        )
        lines.append(f"tools: {tool_count} available")
        if step.get("rejected_tools"):
            lines.append(f"  rejected: {', '.join(step['rejected_tools'])}")
        if step.get("focused_set"):
            lines.append(f"  focused_set: {step['focused_set']}")
        ctx_parts: list[str] = []
        if step.get("context_usage_pct") is not None:
            ctx_parts.append(f"{step['context_usage_pct']:.0f}% context used")
        if step.get("message_count") is not None:
            ctx_parts.append(f"{step['message_count']} messages")
        if step.get("compaction_count"):
            ctx_parts.append(f"{step['compaction_count']} compactions")
        if ctx_parts:
            lines.append(f"context: {', '.join(ctx_parts)}")
        if step.get("compaction_method"):
            lines.append(
                f"  compacted: {step['compaction_method']} "
                f"({step.get('compaction_messages_before')} -> "
                f"{step.get('compaction_messages_after')} messages)"
            )
        if step.get("trimmed_messages"):
            lines.append(f"  trimmed: {step['trimmed_messages']} messages")
        lines.append(
            f"surface: completeness={step.get('completeness', 0.0):.3f} "
            f"missing={len(step.get('missing_surfaces') or [])}"
        )
        step_missing = set(step.get("missing_surfaces") or [])
        unique_missing = step_missing - _common_missing
        if unique_missing:
            lines.append(f"  missing_surfaces: {', '.join(sorted(unique_missing))}")
        injections: list[str] = []
        if step.get("spin_intervention"):
            injections.append(f"SPIN: {step['spin_intervention']}")
        if step.get("correction_messages"):
            for msg in step["correction_messages"]:
                injections.append(f"CORRECTION: {msg}")
        if step.get("error_registry_context"):
            injections.append(f"ERROR_CONTEXT: {step['error_registry_context'][:200]}")
        if step.get("hard_loop_breaker"):
            injections.append(f"LOOP_BREAK: {step['hard_loop_breaker']}")
        if step.get("force_termination"):
            injections.append(f"FORCED_STOP: {step['force_termination']}")
        if step.get("continuation_nudge"):
            injections.append(f"NUDGE: {step['continuation_nudge']}")
        if step.get("approval_path"):
            injections.append(f"APPROVAL: {step['approval_path']}")
        if injections:
            lines.append("injections:")
            for inj in injections:
                lines.append(f"  {inj}")
        if (
            step.get("memory_query") is not None
            or step.get("memory_results") is not None
            or step.get("memory_store_key") is not None
        ):
            lines.append(
                f"memory: query={step.get('memory_query')!r} "
                f"results={step.get('memory_results')!r} "
                f"store_key={step.get('memory_store_key')!r}"
            )
        if (
            step.get("rag_query") is not None
            or step.get("rag_documents_count") is not None
            or step.get("rag_relevance_scores") is not None
        ):
            lines.append(
                f"rag: query={step.get('rag_query')!r} "
                f"documents={step.get('rag_documents_count')!r} "
                f"scores={step.get('rag_relevance_scores')!r}"
            )
        lines.extend(
            [
                f"decision: {step['tool_name']} "
                f"{json.dumps(step['tool_input'], sort_keys=True, ensure_ascii=True)}",
                f"reasoning: {step['llm_reasoning'] or '(none)'}",
                f"result: {step['tool_result_summary'] or '(empty)'}",
            ]
        )
        if step.get("error"):
            lines.append(f"error: {step['error']}")
    return "\n".join(lines)


def format_reasoning_text(reasoning: dict[str, Any]) -> str:
    """Render a compact text view of a task reasoning chain."""
    lines = [f"REASONING CHAIN: {reasoning['task_id']}"]
    if reasoning.get("task_text"):
        lines.append(f"user: {reasoning['task_text']}")
    for step in reasoning["reasoning_chain"]:
        lines.append(f"\nStep {step['step']}")
        lines.append(step["reasoning"] or "(no reasoning)")
        lines.append(
            f"-> {step['decision']['tool_name']} "
            f"{json.dumps(step['decision']['tool_input'], sort_keys=True, ensure_ascii=True)}"
        )
        if step.get("error"):
            lines.append(f"!! {step['error']}")
        elif step.get("result_summary"):
            lines.append(f"<- {step['result_summary']}")
    return "\n".join(lines)


def format_tree_text(tree: dict[str, dict[str, list[str]]]) -> str:
    """Render a day/site/task tree as plain text."""
    lines = ["TASK TREE"]
    for day, sites in tree.items():
        lines.append(day)
        for site, task_ids in sites.items():
            lines.append(f"  {site}")
            for task_id in task_ids:
                lines.append(f"    {task_id}")
    return "\n".join(lines)
