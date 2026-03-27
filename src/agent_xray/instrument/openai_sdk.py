"""Auto-instrumentation for the OpenAI Python SDK.

Wraps ``openai.OpenAI`` and ``openai.AsyncOpenAI`` chat completion calls so
that every ``tool_calls`` block in a response is logged as an agent-xray step.

Usage::

    from agent_xray.instrument import OpenAIInstrumentor

    OpenAIInstrumentor(output_dir="./traces").instrument()

    client = openai.OpenAI()
    response = client.chat.completions.create(model="gpt-4o", ...)
"""

from __future__ import annotations

import functools
import time
from typing import Any

from .base import StepRecorder

_DEFAULT_OUTPUT_DIR = "./traces"


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Pull tool_calls from an OpenAI chat completion response."""
    calls: list[dict[str, Any]] = []
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return calls
    message = getattr(choices[0], "message", None)
    if message is None:
        return calls
    tool_calls = getattr(message, "tool_calls", None)
    if not isinstance(tool_calls, list):
        return calls
    for tc in tool_calls:
        function = getattr(tc, "function", None)
        if function is None:
            continue
        import json

        arguments = getattr(function, "arguments", "{}")
        try:
            parsed_args = json.loads(arguments) if isinstance(arguments, str) else arguments
        except (json.JSONDecodeError, TypeError):
            parsed_args = {"raw": arguments}
        if not isinstance(parsed_args, dict):
            parsed_args = {"value": parsed_args}
        calls.append(
            {
                "id": getattr(tc, "id", None),
                "name": getattr(function, "name", ""),
                "arguments": parsed_args,
            }
        )
    return calls


def _extract_usage(response: Any) -> tuple[int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return (None, None)
    return (
        getattr(usage, "prompt_tokens", None),
        getattr(usage, "completion_tokens", None),
    )


def _extract_model(response: Any) -> str | None:
    return getattr(response, "model", None)


class OpenAIInstrumentor:
    """Monkey-patch instrumentor for the OpenAI Python SDK.

    Args:
        output_dir: Trace output directory.
        task_id: Default task identifier for all recorded steps.

    Example:
        >>> instrumentor = OpenAIInstrumentor(output_dir="./traces")
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
        """Apply monkey-patches to the OpenAI SDK.

        Raises:
            ImportError: If the ``openai`` package is not installed.
        """
        if self._patched:
            return
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "openai package required. Install with: pip install agent-xray[openai]"
            ) from exc

        completions_cls = openai.resources.chat.Completions
        async_completions_cls = openai.resources.chat.AsyncCompletions

        self._originals["sync_create"] = completions_cls.create
        self._originals["async_create"] = async_completions_cls.create

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

        completions_cls.create = traced_create  # type: ignore[assignment]
        async_completions_cls.create = traced_create_async  # type: ignore[assignment]
        self._patched = True

    def uninstrument(self) -> None:
        """Remove monkey-patches and close the recorder."""
        if not self._patched:
            return
        try:
            import openai  # type: ignore[import-untyped]
        except ImportError:
            return

        if "sync_create" in self._originals:
            openai.resources.chat.Completions.create = self._originals["sync_create"]
        if "async_create" in self._originals:
            openai.resources.chat.AsyncCompletions.create = self._originals["async_create"]
        self._originals.clear()
        self._recorder.close()
        self._patched = False


def _log_response(
    recorder: StepRecorder,
    response: Any,
    elapsed_ms: int,
    kwargs: dict[str, Any],
) -> None:
    tool_calls = _extract_tool_calls(response)
    if not tool_calls:
        return
    model_name = _extract_model(response)
    input_tokens, output_tokens = _extract_usage(response)
    tools_kwarg = kwargs.get("tools")
    tools_available: list[str] | None = None
    if isinstance(tools_kwarg, list):
        names = []
        for tool_def in tools_kwarg:
            if isinstance(tool_def, dict):
                fn = tool_def.get("function")
                if isinstance(fn, dict):
                    name = fn.get("name")
                    if name:
                        names.append(str(name))
        if names:
            tools_available = names

    per_tool_ms = elapsed_ms // max(1, len(tool_calls))
    for tc in tool_calls:
        recorder.record_step(
            tool_name=str(tc.get("name", "")),
            tool_input=tc.get("arguments", {}),
            duration_ms=per_tool_ms,
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tools_available=tools_available,
        )


__all__ = [
    "OpenAIInstrumentor",
]
