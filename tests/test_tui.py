from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_xray.schema import AgentStep, AgentTask, TaskOutcome


def _step(
    step: int,
    tool_name: str,
    tool_input: dict[str, object] | None = None,
    *,
    tool_result: str | None = None,
    error: str | None = None,
    page_url: str | None = None,
    tools_available: list[str] | None = None,
    llm_reasoning: str | None = None,
    correction_messages: list[str] | None = None,
    cost_usd: float | None = None,
    context_usage_pct: float | None = None,
    context_window: int | None = None,
) -> AgentStep:
    return AgentStep(
        task_id="task-2",
        step=step,
        tool_name=tool_name,
        tool_input=tool_input or {},
        tool_result=tool_result,
        error=error,
        page_url=page_url,
        tools_available=tools_available,
        llm_reasoning=llm_reasoning,
        correction_messages=correction_messages,
        cost_usd=cost_usd,
        context_usage_pct=context_usage_pct,
        context_window=context_window,
    )


def _task() -> AgentTask:
    steps = [
        _step(
            1,
            "browser_navigate",
            {"url": "https://shop.example.test/cart"},
            tool_result="Loaded cart page.",
            page_url="https://shop.example.test/cart",
            tools_available=["browser_navigate", "browser_click"],
            llm_reasoning="Open the cart first.",
            correction_messages=["Avoid duplicate clicks."],
            cost_usd=0.1201,
            context_usage_pct=0.42,
            context_window=128000,
        ),
        _step(
            2,
            "browser_click",
            {"ref": "checkout"},
            tool_result="Checkout page loaded.",
            page_url="https://shop.example.test/checkout",
            tools_available=["browser_click", "browser_fill_ref"],
            llm_reasoning="Proceed to checkout.",
            cost_usd=0.2202,
            context_usage_pct=0.51,
            context_window=128000,
        ),
        _step(
            3,
            "browser_click",
            {"ref": "checkout"},
            tool_result="Checkout page loaded again.",
            error="Timed out waiting for change.",
            page_url="https://shop.example.test/checkout",
            tools_available=["browser_click", "browser_fill_ref"],
            llm_reasoning="The checkout action repeated.",
            cost_usd=0.3303,
            context_usage_pct=0.64,
            context_window=128000,
        ),
    ]
    return AgentTask(
        task_id="task-2",
        task_text="Buy the mug and inspect the checkout flow.",
        task_category="commerce",
        steps=steps,
        outcome=TaskOutcome(
            task_id="task-2",
            status="failed",
            total_steps=len(steps),
            total_duration_s=3.0,
            final_answer="The flow repeated at checkout.",
            timestamp="2026-04-05T12:00:00Z",
        ),
    )


def _surface(task: AgentTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "steps": [
            {
                "step": 1,
                "tool_name": "browser_navigate",
                "tool_input": {"url": "https://shop.example.test/cart"},
                "tools_available_names": ["browser_navigate", "browser_click"],
                "conversation_history": [{"role": "user", "content": task.task_text}],
                "llm_reasoning": "Open the cart first.",
                "tool_result_summary": "Loaded cart page.",
                "duration_ms": 120,
                "page_url": "https://shop.example.test/cart",
                "error": None,
                "model": {"cost_usd": 0.1201},
                "context_usage_pct": 0.42,
                "context_window": 128000,
                "correction_messages": ["Avoid duplicate clicks."],
            },
            {
                "step": 2,
                "tool_name": "browser_click",
                "tool_input": {"ref": "checkout"},
                "tools_available_names": ["browser_click", "browser_fill_ref"],
                "conversation_history": [
                    {"role": "user", "content": task.task_text},
                    {"role": "assistant_reasoning", "content": "Open the cart first."},
                ],
                "llm_reasoning": "Proceed to checkout.",
                "tool_result_summary": "Checkout page loaded.",
                "duration_ms": 220,
                "page_url": "https://shop.example.test/checkout",
                "error": None,
                "model": {"cost_usd": 0.2202},
                "context_usage_pct": 0.51,
                "context_window": 128000,
                "correction_messages": [],
            },
            {
                "step": 3,
                "tool_name": "browser_click",
                "tool_input": {"ref": "checkout"},
                "tools_available_names": ["browser_click", "browser_fill_ref"],
                "conversation_history": [
                    {"role": "user", "content": task.task_text},
                    {"role": "assistant_reasoning", "content": "Proceed to checkout."},
                ],
                "llm_reasoning": "The checkout action repeated.",
                "tool_result_summary": "Checkout page loaded again.",
                "duration_ms": 330,
                "page_url": "https://shop.example.test/checkout",
                "error": "Timed out waiting for change.",
                "model": {"cost_usd": 0.3303},
                "context_usage_pct": 0.64,
                "context_window": 128000,
                "correction_messages": ["Try a different selector."],
            },
        ],
    }


def _install_fake_textual(monkeypatch: pytest.MonkeyPatch):
    textual_mod = types.ModuleType("textual")
    textual_mod.__spec__ = ModuleSpec("textual", loader=None)
    app_mod = types.ModuleType("textual.app")
    containers_mod = types.ModuleType("textual.containers")
    widgets_mod = types.ModuleType("textual.widgets")

    class _Widget:
        def __init__(self, *children, id: str | None = None, **kwargs) -> None:
            self.children = list(children)
            self.id = id
            for key, value in kwargs.items():
                setattr(self, key, value)

    class App:
        def __class_getitem__(cls, _item):
            return cls

        def __init__(self, *args, **kwargs) -> None:
            self._queries: dict[str, object] = {}

        def query_one(self, selector: str, _kind=None):
            return self._queries[selector]

    class Header(_Widget):
        pass

    class Footer(_Widget):
        pass

    class Static(_Widget):
        def __init__(self, text: str = "", *children, id: str | None = None, **kwargs) -> None:
            super().__init__(*children, id=id, **kwargs)
            self.text = text

    class ListItem(_Widget):
        pass

    class ListView(_Widget):
        class Highlighted:
            def __init__(self, list_view) -> None:
                self.list_view = list_view

        class Selected:
            def __init__(self, list_view) -> None:
                self.list_view = list_view

        def __init__(self, *children, id: str | None = None, **kwargs) -> None:
            super().__init__(*children, id=id, **kwargs)
            self.index: int | None = None

    class RichLog(_Widget):
        def __init__(self, *children, id: str | None = None, **kwargs) -> None:
            super().__init__(*children, id=id, **kwargs)
            self.lines: list[str] = []

        def clear(self) -> None:
            self.lines.clear()

        def write(self, line: str) -> None:
            self.lines.append(line)

    class Horizontal(_Widget):
        pass

    class Vertical(_Widget):
        pass

    app_mod.App = App
    app_mod.ComposeResult = list
    containers_mod.Horizontal = Horizontal
    containers_mod.Vertical = Vertical
    widgets_mod.Footer = Footer
    widgets_mod.Header = Header
    widgets_mod.ListItem = ListItem
    widgets_mod.ListView = ListView
    widgets_mod.RichLog = RichLog
    widgets_mod.Static = Static

    monkeypatch.setitem(sys.modules, "textual", textual_mod)
    monkeypatch.setitem(sys.modules, "textual.app", app_mod)
    monkeypatch.setitem(sys.modules, "textual.containers", containers_mod)
    monkeypatch.setitem(sys.modules, "textual.widgets", widgets_mod)

    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "textual":
            return ModuleSpec("textual", loader=None)
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    monkeypatch.delitem(sys.modules, "agent_xray.tui", raising=False)
    monkeypatch.delitem(sys.modules, "agent_xray.tui.app", raising=False)
    return importlib.import_module("agent_xray.tui.app")


def _build_app(
    monkeypatch: pytest.MonkeyPatch,
    *,
    grade_name: str = "WEAK",
    task_id: str | None = None,
):
    module = _install_fake_textual(monkeypatch)
    task = _task()
    other_task = AgentTask(task_id="task-1", steps=[], task_text="Earlier task")
    analysis = SimpleNamespace(
        total_cost_usd=0.6706,
        is_spin=False,
        site_name="shop.example.test",
        unique_tools=["browser_navigate", "browser_click"],
        unique_urls=["https://shop.example.test/cart", "https://shop.example.test/checkout"],
        errors=1,
        error_rate=1 / 3,
    )
    grade = SimpleNamespace(
        grade=grade_name,
        score=7,
        signals=[
            SimpleNamespace(
                passed=False,
                name="spin",
                points=-3,
                actual=3,
                reason="Repeated checkout action detected.",
            )
        ],
    )
    root_cause = SimpleNamespace(
        root_cause="stuck_loop",
        confidence=0.82,
        site_name="shop.example.test",
        evidence=["Repeated checkout action detected."],
    )
    surface = _surface(task)

    monkeypatch.setattr(module, "load_tasks", lambda log_dir: [other_task, task])
    monkeypatch.setattr(module, "resolve_task", lambda tasks, requested_task_id: task if requested_task_id == task.task_id else tasks[0])
    monkeypatch.setattr(module, "load_rules", lambda: SimpleNamespace(name="fake"))
    monkeypatch.setattr(module, "analyze_task", lambda loaded_task: analysis)
    monkeypatch.setattr(module, "grade_task", lambda loaded_task, rules, analysis=None: grade)
    monkeypatch.setattr(module, "classify_task", lambda loaded_task, grade_result, loaded_analysis: root_cause)
    monkeypatch.setattr(module, "surface_for_task", lambda loaded_task: surface)

    app = module.AgentXrayApp(log_dir=Path("."), task_id=task_id)
    return module, app


def _wire_queries(app_module, app):
    list_view = app_module.ListView(*app._build_step_items(), id="steps")
    detail = app_module.RichLog(id="detail", wrap=True, highlight=False, markup=False)

    def query_one(selector: str, _kind=None):
        mapping = {
            "#steps": list_view,
            "#detail": detail,
        }
        return mapping[selector]

    app.query_one = query_one
    return list_view, detail


def test_tui_initializes_state_from_task_and_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    _module, app = _build_app(monkeypatch)

    assert app.task.task_id == "task-2"
    assert app.mode == "surface"
    assert app.selected_index == 0
    assert app.repeat_signatures == {2}
    assert app.last_successful_step == 1
    assert "Task: task-2" in app._summary_text()
    assert "Grade: WEAK (7pts)" in app._summary_text()
    assert "Cost: $0.6706" in app._summary_text()


def test_tui_compose_builds_summary_steps_and_detail_panes(monkeypatch: pytest.MonkeyPatch) -> None:
    module, app = _build_app(monkeypatch, task_id="task-2")

    components = list(app.compose())

    assert isinstance(components[0], module.Header)
    assert isinstance(components[-1], module.Footer)
    assert components[1].id == "summary"
    assert "task-2" in components[1].text

    body = components[2]
    assert body.id == "body"
    steps_pane = body.children[0]
    detail_pane = body.children[1]
    assert steps_pane.id == "steps-pane"
    assert detail_pane.id == "detail-pane"
    assert steps_pane.children[0].text == "Steps"
    assert detail_pane.children[0].text == "Decision Surface"
    step_list = steps_pane.children[1]
    assert step_list.id == "steps"
    assert len(step_list.children) == 3
    assert "1. browser_navigate" in step_list.children[0].children[0].text


def test_tui_on_mount_renders_surface_without_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    module, app = _build_app(monkeypatch)
    list_view, detail = _wire_queries(module, app)

    app.on_mount()

    rendered = "\n".join(detail.lines)
    assert list_view.index == 0
    assert "Step 1: browser_navigate" in rendered
    assert "Tools Available (2):" in rendered
    assert "Open the cart first." in rendered
    assert "Step Cost: $0.1201" in rendered


def test_tui_navigation_actions_update_selection_and_render(monkeypatch: pytest.MonkeyPatch) -> None:
    module, app = _build_app(monkeypatch)
    list_view, detail = _wire_queries(module, app)
    app.on_mount()

    app.action_move_down()
    assert list_view.index == 1
    assert app.selected_index == 1
    assert "Step 2: browser_click" in "\n".join(detail.lines)

    app.action_move_down()
    assert list_view.index == 2
    assert app.selected_index == 2

    app.action_move_down()
    assert list_view.index == 2

    app.action_move_up()
    assert list_view.index == 1
    assert app.selected_index == 1


def test_tui_keybindings_and_mode_switches_render_expected_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, app = _build_app(monkeypatch)
    _list_view, detail = _wire_queries(module, app)
    app.selected_index = 1

    assert ("up", "move_up", "Up") in app.BINDINGS
    assert ("d", "show_diff", "Diff") in app.BINDINGS
    assert ("r", "show_root_cause", "Root Cause") in app.BINDINGS
    assert ("g", "show_grade", "Grade") in app.BINDINGS
    assert ("s", "show_surface", "Surface") in app.BINDINGS

    app.action_show_diff()
    assert "--- step-1" in "\n".join(detail.lines)

    app.action_show_root_cause()
    root_rendered = "\n".join(detail.lines)
    assert "Root Cause: stuck_loop" in root_rendered
    assert "Confidence: 0.82" in root_rendered

    app.action_show_grade()
    grade_rendered = "\n".join(detail.lines)
    assert "Grade: WEAK" in grade_rendered
    assert "Signals:" in grade_rendered
    assert "FAIL spin: -3" in grade_rendered


def test_tui_list_view_events_ignore_other_lists_and_refresh_steps_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, app = _build_app(monkeypatch)
    list_view, detail = _wire_queries(module, app)
    other_list = module.ListView(id="other")
    app.on_mount()

    app.on_list_view_highlighted(module.ListView.Highlighted(other_list))
    assert app.selected_index == 0

    list_view.index = 2
    app.on_list_view_highlighted(module.ListView.Highlighted(list_view))
    assert app.selected_index == 2
    assert "Step 3: browser_click" in "\n".join(detail.lines)

    list_view.index = 1
    app.on_list_view_selected(module.ListView.Selected(list_view))
    assert app.selected_index == 1
    assert "Step 2: browser_click" in "\n".join(detail.lines)
