"""Auto-instrumentation callbacks for agent-xray.

This package provides SDK-specific instrumentors that capture agent tool calls
as JSONL step logs compatible with ``agent-xray`` analysis, grading, and
reporting.

Quick start::

    from agent_xray.instrument import auto_instrument

    auto_instrument(output_dir="./traces")

This patches all detected SDKs (Anthropic, OpenAI) so that tool calls are
logged automatically.  For framework-specific usage, import the instrumentor
directly::

    from agent_xray.instrument import AnthropicInstrumentor
    from agent_xray.instrument import OpenAIInstrumentor
    from agent_xray.instrument import XRayCallbackHandler
    from agent_xray.instrument import XRayMCPProxy
"""

from __future__ import annotations

from typing import Any

from .base import StepRecorder

_DEFAULT_OUTPUT_DIR = "./traces"


def auto_instrument(
    output_dir: str = _DEFAULT_OUTPUT_DIR,
    *,
    task_id: str | None = None,
) -> list[Any]:
    """Detect installed SDKs and instrument them all.

    Tries to patch each supported SDK (Anthropic, OpenAI).  SDKs that are not
    installed are silently skipped.

    Args:
        output_dir: Directory where ``.jsonl`` trace files are written.
        task_id: Default task identifier for recorded steps.

    Returns:
        A list of active instrumentor instances (for later ``uninstrument()``).

    Example:
        >>> instrumentors = auto_instrument(output_dir="./traces")
        >>> # ... run your agent ...
        >>> for inst in instrumentors:
        ...     inst.uninstrument()
    """
    active: list[Any] = []

    try:
        from .anthropic_sdk import AnthropicInstrumentor

        inst = AnthropicInstrumentor(output_dir, task_id=task_id)
        inst.instrument()
        active.append(inst)
    except ImportError:
        pass

    try:
        from .openai_sdk import OpenAIInstrumentor

        inst = OpenAIInstrumentor(output_dir, task_id=task_id)
        inst.instrument()
        active.append(inst)
    except ImportError:
        pass

    return active


# Lazy imports to avoid requiring SDK packages at import time
def __getattr__(name: str) -> Any:
    if name == "AnthropicInstrumentor":
        from .anthropic_sdk import AnthropicInstrumentor

        return AnthropicInstrumentor
    if name == "xray_trace":
        from .anthropic_sdk import xray_trace

        return xray_trace
    if name == "OpenAIInstrumentor":
        from .openai_sdk import OpenAIInstrumentor

        return OpenAIInstrumentor
    if name == "XRayCallbackHandler":
        from .langchain_cb import XRayCallbackHandler

        return XRayCallbackHandler
    if name == "XRayMCPProxy":
        from .mcp_proxy import XRayMCPProxy

        return XRayMCPProxy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnthropicInstrumentor",
    "OpenAIInstrumentor",
    "StepRecorder",
    "XRayCallbackHandler",
    "XRayMCPProxy",
    "auto_instrument",
    "xray_trace",
]
