from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys

import pytest

from agent_xray.cli import build_parser


def _import_tui_without_textual(monkeypatch: pytest.MonkeyPatch):
    original_find_spec = importlib.util.find_spec

    def fake_find_spec(name: str, *args, **kwargs):
        if name == "textual":
            return None
        return original_find_spec(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)
    sys.modules.pop("agent_xray.tui", None)
    sys.modules.pop("agent_xray.tui.app", None)
    return importlib.import_module("agent_xray.tui")


def test_tui_module_importable_without_textual(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_tui_without_textual(monkeypatch)
    assert hasattr(module, "AgentXrayApp")


def test_tui_app_requires_textual(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _import_tui_without_textual(monkeypatch)
    with pytest.raises(ImportError, match="Textual UI requires textual"):
        module.AgentXrayApp(log_dir=".")


def test_cli_tui_command_registered() -> None:
    parser = build_parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    assert "tui" in subparsers.choices
