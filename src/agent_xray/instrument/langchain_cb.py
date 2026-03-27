"""LangChain callback handler that emits agent-xray JSONL traces.

Provides :class:`XRayCallbackHandler`, which implements the LangChain callback
protocol and can be passed to any LangChain agent or chain.

Usage::

    from agent_xray.instrument import XRayCallbackHandler

    handler = XRayCallbackHandler(output_dir="./traces")
    agent.run("Buy headphones", callbacks=[handler])
    handler.close()
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from .base import StepRecorder

_DEFAULT_OUTPUT_DIR = "./traces"


class XRayCallbackHandler:
    """LangChain-compatible callback handler that writes agent-xray JSONL.

    This class implements the LangChain ``BaseCallbackHandler`` interface
    without requiring the ``langchain`` package at import time.  It can be
    passed directly as a callback to any LangChain agent, chain, or tool.

    Args:
        output_dir: Trace output directory.
        task_id: Default task identifier.

    Example:
        >>> handler = XRayCallbackHandler(output_dir="./traces")
        >>> # agent.run("task", callbacks=[handler])
        >>> handler.close()
    """

    def __init__(
        self,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        *,
        task_id: str | None = None,
    ) -> None:
        self._recorder = StepRecorder(output_dir, task_id=task_id)
        self._task_id = task_id or f"lc-{uuid.uuid4().hex[:8]}"
        self._pending_tools: dict[str, dict[str, Any]] = {}
        self._pending_llm_start: float | None = None
        self._last_model: str | None = None
        self._last_input_tokens: int | None = None
        self._last_output_tokens: int | None = None
        self._task_started = False

    @property
    def recorder(self) -> StepRecorder:
        """Access the underlying step recorder."""
        return self._recorder

    def _ensure_task_started(self) -> None:
        if not self._task_started:
            self._recorder.start_task(self._task_id)
            self._task_started = True

    # -- LangChain callback protocol methods --

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call starts."""
        self._ensure_task_started()
        self._pending_llm_start = time.monotonic()
        invocation = kwargs.get("invocation_params") or {}
        self._last_model = (
            invocation.get("model_name")
            or invocation.get("model")
            or serialized.get("kwargs", {}).get("model_name")
        )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call completes."""
        self._pending_llm_start = None
        token_usage = getattr(response, "llm_output", None)
        if isinstance(token_usage, dict):
            usage = token_usage.get("token_usage") or token_usage
            self._last_input_tokens = usage.get("prompt_tokens")
            self._last_output_tokens = usage.get("completion_tokens")

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation starts."""
        self._ensure_task_started()
        key = str(run_id or uuid.uuid4().hex)
        tool_name = serialized.get("name") or kwargs.get("name") or ""
        tool_input: dict[str, Any]
        if isinstance(input_str, dict):
            tool_input = dict(input_str)
        elif isinstance(input_str, str):
            tool_input = {"input": input_str}
        else:
            tool_input = {"input": str(input_str)}
        self._pending_tools[key] = {
            "tool_name": str(tool_name),
            "tool_input": tool_input,
            "start": time.monotonic(),
        }

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation completes."""
        key = str(run_id or "")
        pending = self._pending_tools.pop(key, None)
        if pending is None:
            if self._pending_tools:
                key = next(iter(self._pending_tools))
                pending = self._pending_tools.pop(key)
            else:
                return

        elapsed_ms = int((time.monotonic() - pending["start"]) * 1000)
        self._recorder.record_step(
            task_id=self._task_id,
            tool_name=pending["tool_name"],
            tool_input=pending["tool_input"],
            tool_result=str(output) if output is not None else None,
            duration_ms=elapsed_ms,
            model_name=self._last_model,
            input_tokens=self._last_input_tokens,
            output_tokens=self._last_output_tokens,
        )
        self._last_input_tokens = None
        self._last_output_tokens = None

    def on_tool_error(
        self,
        error: BaseException | str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a tool invocation fails."""
        key = str(run_id or "")
        pending = self._pending_tools.pop(key, None)
        if pending is None:
            if self._pending_tools:
                key = next(iter(self._pending_tools))
                pending = self._pending_tools.pop(key)
            else:
                return

        elapsed_ms = int((time.monotonic() - pending["start"]) * 1000)
        self._recorder.record_step(
            task_id=self._task_id,
            tool_name=pending["tool_name"],
            tool_input=pending["tool_input"],
            error=str(error),
            duration_ms=elapsed_ms,
            model_name=self._last_model,
        )

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain starts. Used to capture task text."""
        self._ensure_task_started()

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain completes."""

    def on_chain_error(
        self,
        error: BaseException | str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when a chain fails."""

    def on_llm_error(
        self,
        error: BaseException | str,
        *,
        run_id: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Called when an LLM call fails."""
        self._pending_llm_start = None

    def close(self) -> None:
        """End the task and close the recorder."""
        if self._task_started:
            self._recorder.end_task(self._task_id, "success")
        self._recorder.close()


__all__ = [
    "XRayCallbackHandler",
]
