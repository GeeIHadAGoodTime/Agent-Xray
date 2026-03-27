"""Interactive Textual UI for agent-xray."""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

if find_spec("textual") is not None:
    from .app import AgentXrayApp
else:

    class AgentXrayApp:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise ImportError(
                "Textual UI requires textual. Install with: pip install agent-xray[tui]"
            )


__all__ = ["AgentXrayApp"]
