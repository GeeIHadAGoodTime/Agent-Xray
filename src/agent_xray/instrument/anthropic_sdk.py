"""Auto-instrumentation for the Anthropic Python SDK.

Wraps the ``anthropic.Anthropic`` and ``anthropic.AsyncAnthropic`` clients so
that every ``messages.create`` call has its tool-use blocks logged as agent-xray
steps automatically.

Usage::

    from agent_xray.instrument import AnthropicInstrumentor

    AnthropicInstrumentor(output_dir="./traces").instrument()

    # Now all Anthropic client calls are traced.
    client = anthropic.Anthropic()
    response = client.messages.create(model="claude-sonnet-4-20250514", ...)

Or as a decorator::

    from agent_xray.instrument import xray_trace

    @xray_trace(output_dir="./traces")
    def my_agent(task_text: str) -> str:
        ...
"""

from __future__ import annotations

import functools
import time
import uuid
from typing import Any

from .base import StepRecorder

_DEFAULT_OUTPUT_DIR = "./traces"


def _extract_tool_uses(response: Any) -> list[dict[str, Any]]:
    """Pull tool_use blocks from an Anthropic response."""
    blocks: list[dict[str, Any]] = []
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return blocks
    for block in content:
        block_type = getattr(block, "type", None)
        if block_type != "tool_use":
            continue
        blocks.append(
            {
                "id": getattr(block, "id", None),
                "name": getattr(block, "name", ""),
                "input": getattr(block, "input", {}),
            }
        )
    return blocks


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    """Extract input/output token counts from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return (None, None)
    return (
        getattr(usage, "input_tokens", None),
        getattr(usage, "output_tokens", None),
    )


def _extract_model(response: Any) -> str | None:
    return getattr(response, "model", None)


class AnthropicInstrumentor:
    """Monkey-patch instrumentor for the Anthropic Python SDK.

    Args:
        output_dir: Trace output directory.
        task_id: Default task identifier for all recorded steps.

    Example:
        >>> instrumentor = AnthropicInstrumentor(output_dir="./traces")
        >>> instrumentor.instrument()
    """

    def __init__(
        self,
        output_dir: str = _DEFAULT_OUTPUT_DIR,
        *,
        task_id: str | None = None,
    ) -> None:
        self._recorder = StepRecorder(output_dir, task_id=task_id)
        self._patched = False
        self._originals: dict[str, Any] = {}

    @property
    def recorder(self) -> StepRecorder:
        """Access the underlying step recorder."""
        return self._recorder

    def instrument(self) -> None:
        """Apply monkey-patches to the Anthropic SDK.

        After calling this, every ``messages.create`` invocation on any
        ``Anthropic`` or ``AsyncAnthropic`` client will emit trace steps.

        Raises:
            ImportError: If the ``anthropic`` package is not installed.
        """
        if self._patched:
            return
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "anthropic package required. Install with: pip install agent-xray[anthropic]"
            ) from exc

        messages_cls = anthropic.resources.Messages
        async_messages_cls = anthropic.resources.AsyncMessages

        self._originals["sync_create"] = messages_cls.create
        self._originals["async_create"] = async_messages_cls.create

        recorder = self._recorder

        original_sync = self._originals["sync_create"]

        @functools.wraps(original_sync)
        def traced_create(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            response = original_sync(self_inner, *args, **kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            _log_response(recorder, response, elapsed_ms, kwargs)
            return response

        original_async = self._originals["async_create"]

        @functools.wraps(original_async)
        async def traced_create_async(self_inner: Any, *args: Any, **kwargs: Any) -> Any:
            start = time.monotonic()
            response = await original_async(self_inner, *args, **kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            _log_response(recorder, response, elapsed_ms, kwargs)
            return response

        messages_cls.create = traced_create  # type: ignore[assignment]
        async_messages_cls.create = traced_create_async  # type: ignore[assignment]
        self._patched = True

    def uninstrument(self) -> None:
        """Remove monkey-patches and close the recorder."""
        if not self._patched:
            return
        try:
            import anthropic  # type: ignore[import-untyped]
        except ImportError:
            return

        if "sync_create" in self._originals:
            anthropic.resources.Messages.create = self._originals["sync_create"]
        if "async_create" in self._originals:
            anthropic.resources.AsyncMessages.create = self._originals["async_create"]
        self._originals.clear()
        self._recorder.close()
        self._patched = False


def _log_response(
    recorder: StepRecorder,
    response: Any,
    elapsed_ms: int,
    kwargs: dict[str, Any],
) -> None:
    tool_uses = _extract_tool_uses(response)
    if not tool_uses:
        return
    model_name = _extract_model(response)
    input_tokens, output_tokens = _extract_usage(response)
    tools_kwarg = kwargs.get("tools")
    tools_available: list[str] | None = None
    if isinstance(tools_kwarg, list):
        names = []
        for tool_def in tools_kwarg:
            if isinstance(tool_def, dict):
                name = tool_def.get("name")
                if name:
                    names.append(str(name))
        if names:
            tools_available = names

    per_tool_ms = elapsed_ms // max(1, len(tool_uses))
    for tool_use in tool_uses:
        tool_input = tool_use.get("input")
        if not isinstance(tool_input, dict):
            tool_input = {}
        recorder.record_step(
            tool_name=str(tool_use.get("name", "")),
            tool_input=tool_input,
            duration_ms=per_tool_ms,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tools_available=tools_available,
        )


def xray_trace(
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    *,
    task_id: str | None = None,
) -> Any:
    """Decorator that instruments an agent function with Anthropic tracing.

    The decorated function gets a ``StepRecorder`` injected as a keyword
    argument ``recorder`` (unless it already provides one).

    Args:
        output_dir: Trace output directory.
        task_id: Default task identifier.

    Example:
        >>> @xray_trace(output_dir="./traces")
        ... def my_agent(prompt: str) -> str:
        ...     return "done"
    """

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            resolved_task_id = task_id or f"task-{uuid.uuid4().hex[:8]}"
            instrumentor = AnthropicInstrumentor(output_dir=output_dir, task_id=resolved_task_id)
            instrumentor.instrument()
            recorder = instrumentor.recorder
            recorder.start_task(resolved_task_id)
            start = time.monotonic()
            error_msg: str | None = None
            result = None
            try:
                result = fn(*args, **kwargs)
            except Exception as exc:
                error_msg = str(exc)
                raise
            finally:
                elapsed = time.monotonic() - start
                status = "failed" if error_msg else "success"
                recorder.end_task(
                    resolved_task_id,
                    status,
                    total_duration_s=elapsed,
                )
                instrumentor.uninstrument()
            return result

        return wrapper

    return decorator


__all__ = [
    "AnthropicInstrumentor",
    "xray_trace",
]
