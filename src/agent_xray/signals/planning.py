"""Signals for explicit planning and plan execution behavior."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from ..schema import AgentStep, AgentTask


class PlanningDetector:
    """Detect plan creation, execution, revision, and goal tracking patterns."""

    name = "planning"

    CREATION_TOOLS = {"create_plan", "make_plan", "plan", "plan_task"}
    REVISION_TOOLS = {"adjust_plan", "replan", "revise_plan", "update_plan"}
    STEP_TOOLS = {"complete_plan_step", "execute_plan_step", "mark_plan_step", "update_plan_step"}
    PLAN_ID_KEYS = ("plan_id", "id")
    PLAN_ID_RE = re.compile(r"\bplan[_ -]?([a-z0-9_-]+)\b", re.IGNORECASE)
    COMPLETION_PATTERNS = ("complete", "completed", "done", "finished", "checked off")

    def detect_step(self, step: AgentStep) -> dict[str, bool | str | None]:
        """Analyze one step for planning signals."""

        tool = step.tool_name.lower()
        combined_text = " ".join(self._iter_step_strings(step)).lower()
        is_plan_creation = (
            tool in self.CREATION_TOOLS
            or "create plan" in combined_text
            or ("plan" in tool and isinstance(step.tool_input.get("steps"), list))
        )
        is_plan_revision = (
            tool in self.REVISION_TOOLS
            or "replan" in tool
            or "revise plan" in combined_text
            or "update plan" in combined_text
        )
        is_plan_step = (
            tool in self.STEP_TOOLS
            or "plan_step" in tool
            or any(
                key in step.tool_input for key in ("step", "step_id", "current_step", "step_index")
            )
            or any(
                marker in combined_text for marker in ("plan step", "step 1", "step 2", "step 3")
            )
        )
        return {
            "is_plan_step": is_plan_step,
            "is_plan_creation": is_plan_creation,
            "is_plan_revision": is_plan_revision,
            "is_goal_tracking": "goal" in combined_text
            or "objective" in combined_text
            or "success criteria" in combined_text,
            "plan_id": self._extract_plan_id(step, combined_text),
        }

    def summarize(
        self, task: AgentTask, step_signals: list[dict[str, bool | str | None]]
    ) -> dict[str, int | float]:
        """Summarize planning activity across a task."""

        total_expected_steps = self._expected_step_count(task)
        completed_steps = 0
        executed_steps = 0
        for step, signals in zip(task.sorted_steps, step_signals, strict=False):
            if signals["is_plan_step"]:
                executed_steps += 1
                if self._step_completed(step):
                    completed_steps += 1
        return {
            "plans_created": sum(1 for signals in step_signals if signals["is_plan_creation"]),
            "plan_steps_executed": executed_steps,
            "plan_revisions": sum(1 for signals in step_signals if signals["is_plan_revision"]),
            "plan_completion_rate": completed_steps / max(total_expected_steps, 1),
        }

    def _expected_step_count(self, task: AgentTask) -> int:
        expected = 0
        for step in task.sorted_steps:
            steps_value = step.tool_input.get("steps")
            if isinstance(steps_value, list):
                expected = max(expected, len(steps_value))
            for key in ("planned_steps", "total_steps"):
                value = step.tool_input.get(key)
                try:
                    expected = max(expected, int(value))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        if expected > 0:
            return expected
        observed = 0
        for step in task.sorted_steps:
            for key in ("step", "step_id", "current_step", "step_index"):
                try:
                    observed = max(observed, int(step.tool_input.get(key)))  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    continue
        return observed

    def _extract_plan_id(self, step: AgentStep, combined_text: str) -> str | None:
        for key in self.PLAN_ID_KEYS:
            value = step.tool_input.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        match = self.PLAN_ID_RE.search(combined_text)
        if match:
            return match.group(1).strip()
        return None

    def _step_completed(self, step: AgentStep) -> bool:
        status = step.tool_input.get("status")
        if isinstance(status, str) and status.strip().lower() in {
            "complete",
            "completed",
            "done",
            "finished",
        }:
            return True
        result = (step.tool_result or "").lower()
        return any(pattern in result for pattern in self.COMPLETION_PATTERNS)

    def _iter_step_strings(self, step: AgentStep) -> Iterable[str]:
        yield step.tool_name
        if step.tool_result:
            yield step.tool_result
        if step.llm_reasoning:
            yield step.llm_reasoning
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
