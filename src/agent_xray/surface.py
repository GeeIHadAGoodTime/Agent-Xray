from __future__ import annotations

import difflib
import json
from typing import Any

from .adapter import PromptBuilder, ToolRegistry
from .analyzer import build_task_tree, resolve_task, summarize_tool_result
from .schema import AgentTask


def _prompt_text(task: AgentTask, prompt_builder: PromptBuilder | None) -> str | None:
    if isinstance(task.metadata.get("system_prompt_text"), str):
        return str(task.metadata["system_prompt_text"])
    if prompt_builder is None:
        return None
    return prompt_builder.build_prompt(task)


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
        tool_names = list(step.tools_available or [])
        if not tool_names and tool_registry is not None:
            tool_names = tool_registry.tool_names(task=task, step=step)
        entry = {
            "step": step.step,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_result_summary": summarize_tool_result(step),
            "error": step.error,
            "llm_reasoning": step.llm_reasoning or "",
            "timestamp": step.timestamp,
            "model_name": step.model_name,
            "temperature": step.temperature,
            "tool_choice": step.tool_choice,
            "message_count": step.message_count,
            "tools_available_names": tool_names,
            "page_url": step.page_url,
            "system_prompt_hash": step.system_prompt_hash,
            "context_usage_pct": step.context_usage_pct,
            "context_window": step.context_window,
            "compaction_count": step.compaction_count,
            "snapshot_compressed": step.snapshot_compressed,
            "had_screenshot": step.had_screenshot,
            "correction_messages": step.correction_messages or [],
            "spin_intervention": step.spin_intervention,
            "conversation_history": list(history),
        }
        steps.append(entry)
        if step.llm_reasoning:
            history.append({"role": "assistant_reasoning", "content": step.llm_reasoning})
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
        lines.extend(["", "PROMPT:", prompt, ""])
    for step in surface["steps"]:
        lines.extend(
            [
                "-" * 72,
                f"STEP {step['step']} [{step.get('timestamp') or ''}]",
                f"context: model={step.get('model_name') or '?'} temp={step.get('temperature')} "
                f"tool_choice={step.get('tool_choice') or '?'} "
                f"tools={len(step.get('tools_available_names') or [])}",
                f"decision: {step['tool_name']} "
                f"{json.dumps(step['tool_input'], sort_keys=True, ensure_ascii=True)}",
                f"reasoning: {step['llm_reasoning'] or '(none)'}",
                f"result: {step['tool_result_summary'] or '(empty)'}",
            ]
        )
        if step.get("error"):
            lines.append(f"error: {step['error']}")
        if step.get("spin_intervention"):
            lines.append(f"spin: {step['spin_intervention']}")
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
