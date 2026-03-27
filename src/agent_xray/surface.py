from __future__ import annotations

import difflib
import json
from dataclasses import asdict
from typing import Any

from .analyzer import build_task_tree, resolve_task, summarize_tool_result
from .protocols import PromptBuilder, ToolRegistry
from .schema import AgentTask


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
        return task.metadata["system_context_components"]
    if task.steps:
        first = task.sorted_steps[0]
        comp = first.extensions.get("system_context_components")
        if isinstance(comp, dict):
            return comp
    return None


def surface_for_task(
    task: AgentTask,
    *,
    prompt_builder: PromptBuilder | None = None,
    tool_registry: ToolRegistry | None = None,
) -> dict[str, Any]:
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
                tools.tools_available_count if tools and tools.tools_available_count is not None
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
            "consecutive_failure_warning": reasoning.consecutive_failure_warning if reasoning else None,
            "approval_path": reasoning.approval_path if reasoning else None,
            # Browser state
            "page_url": browser.page_url if browser else None,
            "snapshot_compressed": browser.snapshot_compressed if browser else None,
            "had_screenshot": browser.had_screenshot if browser else None,
            "had_screenshot_image": browser.had_screenshot_image if browser else None,
            "snapshot_pre_compress_len": browser.snapshot_pre_compress_len if browser else None,
            # Accumulated conversation
            "conversation_history": list(history),
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
    left_surface = surface_for_task(
        left, prompt_builder=prompt_builder, tool_registry=tool_registry
    )
    right_surface = surface_for_task(
        right, prompt_builder=prompt_builder, tool_registry=tool_registry
    )
    diverged_at = None
    for index, pair in enumerate(
        zip(left_surface["steps"], right_surface["steps"], strict=False),
        start=1,
    ):
        left_step, right_step = pair
        if (
            left_step["tool_name"],
            left_step["tool_input"],
            left_step["page_url"],
            left_step["error"],
        ) != (
            right_step["tool_name"],
            right_step["tool_input"],
            right_step["page_url"],
            right_step["error"],
        ):
            diverged_at = index
            break
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
        "prompt_diff": prompt_diff,
    }


def tree_for_tasks(tasks: list[AgentTask]) -> dict[str, dict[str, list[str]]]:
    return build_task_tree(tasks)


def format_surface_text(surface: dict[str, Any]) -> str:
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
    lines = ["TASK TREE"]
    for day, sites in tree.items():
        lines.append(day)
        for site, task_ids in sites.items():
            lines.append(f"  {site}")
            for task_id in task_ids:
                lines.append(f"    {task_id}")
    return "\n".join(lines)


def resolve_surface_task(tasks: list[AgentTask], query: str) -> AgentTask:
    return resolve_task(tasks, query)
