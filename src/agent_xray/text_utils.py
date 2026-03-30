from __future__ import annotations

import json
from typing import Any


def tool_result_text(tool_result: Any) -> str:
    """Normalize tool_result values into searchable text.

    Some traces store tool output as a plain string while others return a
    structured payload with a top-level ``data`` field. This helper extracts the
    useful text path without losing fallback visibility into the full payload.
    """
    if tool_result is None:
        return ""
    if isinstance(tool_result, str):
        return tool_result
    if isinstance(tool_result, dict):
        data = tool_result.get("data")
        if data is None:
            return json.dumps(tool_result, ensure_ascii=False, default=str)
        if isinstance(data, str):
            return data
        return json.dumps(data, ensure_ascii=False, default=str)
    return str(tool_result)
