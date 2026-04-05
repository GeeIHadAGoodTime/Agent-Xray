from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Header, ListItem, ListView, RichLog, Static

from ..analyzer import analyze_task, load_tasks, resolve_task
from ..grader import grade_task, load_rules
from ..root_cause import classify_task
from ..surface import surface_for_task


def _step_signature(step: dict[str, Any]) -> str:
    return json.dumps(
        {
            "tool_name": step["tool_name"],
            "tool_input": step["tool_input"],
        },
        sort_keys=True,
        ensure_ascii=True,
    )


def _format_money(value: float) -> str:
    return f"${value:.4f}" if value else "$0.0000"


class AgentXrayApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
    }

    #summary {
        height: 1;
        padding: 0 1;
    }

    #body {
        height: 1fr;
    }

    #steps-pane {
        width: 32;
        border: round $accent;
    }

    #detail-pane {
        width: 1fr;
        border: round $accent;
    }

    #steps-title, #detail-title {
        padding: 0 1;
        text-style: bold;
    }

    #steps {
        height: 1fr;
    }

    #detail {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("up", "move_up", "Up"),
        ("down", "move_down", "Down"),
        ("d", "show_diff", "Diff"),
        ("r", "show_root_cause", "Root Cause"),
        ("g", "show_grade", "Grade"),
        ("s", "show_surface", "Surface"),
        ("q", "quit", "Quit"),
    ]

    def __init__(self, *, log_dir: str | Path, task_id: str | None = None) -> None:
        super().__init__()
        self.log_dir = Path(log_dir)
        self.tasks = load_tasks(self.log_dir)
        if not self.tasks:
            raise ValueError(f"No tasks found under {self.log_dir}")
        self.agent_task = resolve_task(self.tasks, task_id) if task_id else self.tasks[-1]
        self.rules = load_rules()
        self.analysis = analyze_task(self.agent_task)
        self.grade = grade_task(self.agent_task, self.rules, analysis=self.analysis)
        self.root_cause = (
            classify_task(self.agent_task, self.grade, self.analysis)
            if self.grade.grade in {"WEAK", "BROKEN"}
            else None
        )
        self.surface = surface_for_task(self.agent_task)
        self.mode = "surface"
        self.selected_index = 0
        self.repeat_signatures = self._build_repeat_signatures()
        self.last_successful_step = max(
            (index for index, step in enumerate(self.agent_task.sorted_steps) if not step.error),
            default=None,
        )

    @property
    def task(self):  # type: ignore[override]
        return self.agent_task

    def compose(self) -> ComposeResult:
        yield Header()
        yield Static(self._summary_text(), id="summary")
        yield Horizontal(
            Vertical(
                Static("Steps", id="steps-title"),
                ListView(*self._build_step_items(), id="steps"),
                id="steps-pane",
            ),
            Vertical(
                Static("Decision Surface", id="detail-title"),
                RichLog(id="detail", wrap=True, highlight=False, markup=False),
                id="detail-pane",
            ),
            id="body",
        )
        yield Footer()

    def on_mount(self) -> None:
        list_view = self.query_one("#steps", ListView)
        if list_view.children:
            list_view.index = 0
        self._render_detail()

    def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        if event.list_view.id != "steps":
            return
        self.selected_index = max(0, event.list_view.index or 0)
        if self.mode == "surface":
            self._render_detail()

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.list_view.id != "steps":
            return
        self.selected_index = max(0, event.list_view.index or 0)
        self._render_detail()

    def action_move_up(self) -> None:
        if not self.surface["steps"]:
            return
        list_view = self.query_one("#steps", ListView)
        current = list_view.index or 0
        list_view.index = max(0, current - 1)
        self.selected_index = list_view.index
        self._render_detail()

    def action_move_down(self) -> None:
        if not self.surface["steps"]:
            return
        list_view = self.query_one("#steps", ListView)
        current = list_view.index or 0
        list_view.index = min(len(self.surface["steps"]) - 1, current + 1)
        self.selected_index = list_view.index
        self._render_detail()

    def action_show_surface(self) -> None:
        self.mode = "surface"
        self._render_detail()

    def action_show_diff(self) -> None:
        self.mode = "diff"
        self._render_detail()

    def action_show_root_cause(self) -> None:
        self.mode = "root"
        self._render_detail()

    def action_show_grade(self) -> None:
        self.mode = "grade"
        self._render_detail()

    def _summary_text(self) -> str:
        return (
            f"Task: {self.agent_task.task_id}  Grade: {self.grade.grade} ({self.grade.score}pts)  "
            f"Steps: {len(self.agent_task.steps)}  Cost: {_format_money(self.analysis.total_cost_usd)}"
        )

    def _build_repeat_signatures(self) -> set[int]:
        seen: set[str] = set()
        repeats: set[int] = set()
        for index, step in enumerate(self.surface["steps"]):
            signature = _step_signature(step)
            if signature in seen:
                repeats.add(index)
            seen.add(signature)
        return repeats

    def _indicator_for(self, index: int, step: dict[str, Any]) -> tuple[str, str]:
        if self.last_successful_step == index:
            return "✓", "green"
        if step.get("error") or self.analysis.is_spin:
            return "⚠", "red"
        if index in self.repeat_signatures:
            return "🔄", "yellow"
        previous_url = self.surface["steps"][index - 1]["page_url"] if index > 0 else None
        if step.get("page_url") and step.get("page_url") != previous_url:
            return "▲", "cyan"
        return "•", "white"

    def _build_step_items(self) -> list[ListItem]:
        items: list[ListItem] = []
        for index, step in enumerate(self.surface["steps"]):
            indicator, color = self._indicator_for(index, step)
            label = f"[{color}]{step['step']:>2}. {step['tool_name']} {indicator}[/{color}]"
            items.append(ListItem(Static(label, markup=True)))
        return items

    def _current_step(self) -> dict[str, Any]:
        return self.surface["steps"][self.selected_index]

    def _write_detail(self, text: str) -> None:
        detail = self.query_one("#detail", RichLog)
        detail.clear()
        for line in text.splitlines() or [""]:
            detail.write(line)

    def _render_detail(self) -> None:
        if not self.surface["steps"]:
            self._write_detail("This task has no recorded steps.")
            return
        if self.mode == "diff":
            self._write_detail(self._format_diff())
            return
        if self.mode == "root":
            self._write_detail(self._format_root_cause())
            return
        if self.mode == "grade":
            self._write_detail(self._format_grade())
            return
        self._write_detail(self._format_surface())

    def _format_surface(self) -> str:
        step = self._current_step()
        history = step.get("conversation_history") or []
        recent_history = history[-3:]
        tools = ", ".join(step.get("tools_available_names") or []) or "(unknown)"
        lines = [
            f"Step {step['step']}: {step['tool_name']}",
            "",
            f"Tools Available ({len(step.get('tools_available_names') or [])}):",
            f"  {tools}",
            "",
            f"Conversation History ({len(recent_history)} messages shown):",
        ]
        if recent_history:
            for item in recent_history:
                content = str(item.get("content") or "").replace("\n", " ").strip()
                lines.append(f"  [{item.get('role')}] {content[:140]}")
        else:
            lines.append("  (none)")
        lines.extend(
            [
                "",
                "Reasoning:",
                f"  {step.get('llm_reasoning') or '(none)'}",
                "",
                "Result:",
                f"  {step.get('tool_result_summary') or '(empty)'}",
                "",
                f"Context: {self._format_context(step)}",
                f"Corrections: {self._format_corrections(step)}",
                f"Duration: {step.get('duration_ms') or 0}ms",
            ]
        )
        if step.get("page_url"):
            lines.append(f"URL: {step['page_url']}")
        if step.get("error"):
            lines.append(f"Error: {step['error']}")
        if step.get("model"):
            cost = step["model"].get("cost_usd")
            if cost is not None:
                lines.append(f"Step Cost: {_format_money(float(cost))}")
        return "\n".join(lines)

    def _format_context(self, step: dict[str, Any]) -> str:
        usage = step.get("context_usage_pct")
        window = step.get("context_window")
        if usage is None and window is None:
            return "unknown"
        if usage is None:
            return f"window={window}"
        if usage <= 1:
            pct_text = f"{usage * 100:.0f}%"
        else:
            pct_text = f"{usage:.0f}%"
        return f"{pct_text} used ({window or '?'} tokens)"

    def _format_corrections(self, step: dict[str, Any]) -> str:
        corrections = step.get("correction_messages") or []
        if not corrections:
            return "None"
        return "; ".join(str(item) for item in corrections[:3])

    def _surface_snapshot(self, step: dict[str, Any]) -> list[str]:
        return [
            f"tool={step['tool_name']}",
            f"input={json.dumps(step['tool_input'], sort_keys=True, ensure_ascii=True)}",
            f"tools={','.join(step.get('tools_available_names') or [])}",
            f"url={step.get('page_url') or ''}",
            f"history={json.dumps(step.get('conversation_history') or [], sort_keys=True, ensure_ascii=True)}",
            f"reasoning={step.get('llm_reasoning') or ''}",
            f"result={step.get('tool_result_summary') or ''}",
        ]

    def _format_diff(self) -> str:
        if self.selected_index == 0:
            return "No previous step to diff against."
        current = self._current_step()
        previous = self.surface["steps"][self.selected_index - 1]
        diff = difflib.unified_diff(
            self._surface_snapshot(previous),
            self._surface_snapshot(current),
            fromfile=f"step-{previous['step']}",
            tofile=f"step-{current['step']}",
            lineterm="",
        )
        lines = list(diff)
        return "\n".join(lines) if lines else "No decision-surface changes detected."

    def _format_root_cause(self) -> str:
        if self.root_cause is None:
            return "Task is not graded as WEAK or BROKEN; no root cause classified."
        lines = [
            f"Root Cause: {self.root_cause.root_cause}",
            f"Confidence: {self.root_cause.confidence}",
            f"Site: {self.root_cause.site_name or self.analysis.site_name}",
        ]
        if self.root_cause.evidence:
            lines.append("")
            lines.append("Evidence:")
            for item in self.root_cause.evidence:
                lines.append(f"  - {item}")
        return "\n".join(lines)

    def _format_grade(self) -> str:
        lines = [
            f"Grade: {self.grade.grade}",
            f"Score: {self.grade.score}",
            "",
            "Signals:",
        ]
        for signal in self.grade.signals:
            status = "PASS" if signal.passed else "FAIL"
            lines.append(
                f"  {status:<4} {signal.name}: {signal.points:+d} "
                f"(actual={signal.actual}) {signal.reason}"
            )
        lines.extend(
            [
                "",
                "Task Metrics:",
                f"  unique_tools={len(self.analysis.unique_tools)}",
                f"  unique_urls={len(self.analysis.unique_urls)}",
                f"  errors={self.analysis.errors}",
                f"  error_rate={self.analysis.error_rate:.2f}",
                f"  total_cost={_format_money(self.analysis.total_cost_usd)}",
            ]
        )
        return "\n".join(lines)
