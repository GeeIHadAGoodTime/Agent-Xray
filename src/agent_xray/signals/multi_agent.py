"""Signals for multi-agent orchestration and delegation behavior."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..schema import AgentStep, AgentTask


class MultiAgentDetector:
    """Detect delegation, handoffs, and sub-agent orchestration patterns."""

    name = "multi_agent"

    DELEGATION_TOOLS = {
        "assign_agent",
        "delegate",
        "delegate_task",
        "handoff",
        "handoff_to_agent",
        "route_task",
        "run_sub_agent",
        "send_input",
        "spawn_agent",
        "sub_agent",
        "transfer_to_agent",
    }
    HANDOFF_TOOLS = {"handoff", "handoff_to_agent", "route_task", "transfer_to_agent"}
    SUB_AGENT_TOOLS = {"run_sub_agent", "send_input", "spawn_agent", "sub_agent"}
    SUCCESS_PATTERNS = (
        "accepted",
        "assigned",
        "completed",
        "done",
        "finished",
        "queued",
        "resolved",
        "success",
    )
    TARGET_KEYS = (
        "agent",
        "agent_id",
        "agent_name",
        "agent_role",
        "delegate",
        "delegate_to",
        "delegation_target",
        "handoff_to",
        "recipient",
        "role",
        "target",
        "target_agent",
        "worker",
    )
    DELEGATION_PATTERNS = (
        re.compile(
            r"\b(?:delegate|assign|route)\s+(?:this\s+)?(?:task|work)?\s*(?:to)?\s+([a-z0-9_-]+)",
            re.IGNORECASE,
        ),
        re.compile(r"\bhandoff\s+(?:to\s+)?([a-z0-9_-]+)", re.IGNORECASE),
        re.compile(
            r"\bspawn(?:ing)?\s+(?:a\s+)?(?:sub[- ]?agent|worker)\s+([a-z0-9_-]+)?",
            re.IGNORECASE,
        ),
    )
    DEPTH_KEYS = ("depth", "delegation_depth", "agent_depth", "level", "nesting_level")

    def detect_step(self, step: AgentStep) -> dict[str, bool | str | None]:
        """Analyze one step for multi-agent orchestration signals."""

        tool = step.tool_name.lower()
        combined_text = " ".join(self._iter_step_strings(step))
        target = self._extract_target(step, combined_text)
        is_sub_agent_call = tool in self.SUB_AGENT_TOOLS or "spawn" in tool or "sub_agent" in tool
        is_handoff = tool in self.HANDOFF_TOOLS or "handoff" in combined_text.lower()
        is_delegation = (
            tool in self.DELEGATION_TOOLS
            or "delegate" in tool
            or is_handoff
            or is_sub_agent_call
            or any(pattern.search(combined_text) for pattern in self.DELEGATION_PATTERNS)
        )
        return {
            "is_delegation": is_delegation,
            "is_handoff": is_handoff,
            "is_sub_agent_call": is_sub_agent_call,
            "delegation_target": target,
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool | str | None]]
    ) -> dict[str, int | float]:
        """Summarize delegation activity across a task."""

        delegation_steps = [
            index for index, signals in enumerate(step_signals) if signals["is_delegation"]
        ]
        targets = {
            str(signals["delegation_target"]).strip()
            for signals in step_signals
            if isinstance(signals.get("delegation_target"), str)
            and str(signals["delegation_target"]).strip()
        }
        agents = targets | self._collect_agents(task)
        successes = 0
        for index in delegation_steps:
            if self._delegation_succeeded(task, step_signals, index):
                successes += 1
        return {
            "delegation_count": len(delegation_steps),
            "unique_agents": len(agents),
            "delegation_success_rate": successes / max(len(delegation_steps), 1),
            "max_delegation_depth": self._max_delegation_depth(task, step_signals),
        }

    def _collect_agents(self, task: AgentTask) -> set[str]:
        agents: set[str] = set()
        for step in task.sorted_steps:
            for key in ("agent_role", "agent", "role", "worker"):
                value = step.tool_input.get(key)
                if isinstance(value, str) and value.strip():
                    agents.add(value.strip())
            for key in ("agent_role", "agent", "role", "worker"):
                value = step.extensions.get(key)
                if isinstance(value, str) and value.strip():
                    agents.add(value.strip())
        return agents

    def _delegation_succeeded(
        self,
        task: AgentTask,
        step_signals: list[dict[str, bool | str | None]],
        index: int,
    ) -> bool:
        step = task.sorted_steps[index]
        target = step_signals[index].get("delegation_target")
        result = (step.tool_result or "").lower()
        if step.error:
            return False
        if isinstance(target, str) and target:
            target_lower = target.lower()
            for later_step in task.sorted_steps[index + 1 :]:
                role_candidates = {
                    str(value).strip().lower()
                    for value in (
                        later_step.tool_input.get("agent_role"),
                        later_step.tool_input.get("agent"),
                        later_step.tool_input.get("role"),
                        later_step.extensions.get("agent_role"),
                        later_step.extensions.get("agent"),
                        later_step.extensions.get("role"),
                    )
                    if isinstance(value, str) and str(value).strip()
                }
                if target_lower in role_candidates:
                    return True
        return any(pattern in result for pattern in self.SUCCESS_PATTERNS)

    def _max_delegation_depth(
        self,
        task: AgentTask,
        step_signals: list[dict[str, bool | str | None]],
    ) -> int:
        max_depth = 0
        running_depth = 0
        for step, signals in zip(task.sorted_steps, step_signals, strict=False):
            explicit_depth = self._extract_depth(step)
            if explicit_depth is not None:
                max_depth = max(max_depth, explicit_depth)
            if signals["is_sub_agent_call"]:
                running_depth += 1
            elif signals["is_delegation"]:
                running_depth = max(running_depth, 1)
            else:
                running_depth = 0
            max_depth = max(max_depth, running_depth)
        return max_depth

    def _extract_depth(self, step: AgentStep) -> int | None:
        for key in self.DEPTH_KEYS:
            value = step.tool_input.get(key)
            if value is None:
                value = step.extensions.get(key)
            try:
                depth = int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
            if depth > 0:
                return depth
        return None

    def _extract_target(self, step: AgentStep, combined_text: str) -> str | None:
        for key in self.TARGET_KEYS:
            value = step.tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        for pattern in self.DELEGATION_PATTERNS:
            match = pattern.search(combined_text)
            if match and match.group(1):
                return match.group(1).strip()
        return None

    def _iter_step_strings(self, step: AgentStep) -> Iterable[str]:
        yield step.tool_name
        if step.tool_result:
            yield step.tool_result
        if step.error:
            yield step.error
        yield from self._iter_strings(step.tool_input)
        yield from self._iter_strings(step.extensions)

    def _iter_strings(self, value: Any) -> Iterable[str]:
        if isinstance(value, str):
            yield value
            return
        if isinstance(value, dict):
            for item in value.values():
                yield from self._iter_strings(item)
            return
        if isinstance(value, list):
            for item in value:
                yield from self._iter_strings(item)
