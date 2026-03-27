from __future__ import annotations

import asyncio

import pytest

import agent_xray.runner as runner_module


class _ProtocolRunner:
    async def send(self, task_text: str) -> str:
        return task_text

    async def get_status(self, task_id: str) -> str:
        return task_id


def test_task_runner_protocol_interface() -> None:
    assert isinstance(_ProtocolRunner(), runner_module.TaskRunner)


def test_generic_http_runner_requires_httpx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "httpx", None)
    with pytest.raises(ImportError, match="httpx is required for GenericHTTPRunner"):
        runner_module.GenericHTTPRunner("https://api.example.test")


def test_runner_protocol_is_runtime_checkable() -> None:
    assert isinstance(runner_module.StaticRunner(), runner_module.TaskRunner)


def test_static_runner_interface() -> None:
    runner = runner_module.StaticRunner(task_id="task-123", status="finished")
    assert asyncio.run(runner.send("hello")) == "task-123"
    assert asyncio.run(runner.get_status("task-123")) == "finished"


def test_runner_base_url_stored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner_module, "httpx", object())
    runner = runner_module.GenericHTTPRunner("https://api.example.test")
    assert runner.base_url == "https://api.example.test"
