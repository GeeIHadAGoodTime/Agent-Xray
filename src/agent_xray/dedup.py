from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


def _normalize_task_text(task: Any) -> str:
    text = str(getattr(task, "task_text", None) or "").strip().lower()
    normalized = re.sub(r"\s+", " ", text)
    return normalized or str(getattr(task, "task_id", ""))


def _timestamp_sort_key(value: Any) -> tuple[int, str]:
    raw = str(value or "").strip()
    if not raw:
        return (0, "")
    try:
        return (1, str(float(raw)))
    except (TypeError, ValueError):
        pass
    try:
        normalized = raw.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return (2, parsed.astimezone(timezone.utc).isoformat())
    except ValueError:
        return (1, raw)


def _task_recency_key(task: Any) -> tuple[tuple[int, str], int, str]:
    latest_timestamp = (0, "")
    for step in getattr(task, "steps", []) or []:
        timestamp = _timestamp_sort_key(getattr(step, "timestamp", ""))
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
    outcome = getattr(task, "outcome", None)
    outcome_timestamp = _timestamp_sort_key(getattr(outcome, "timestamp", ""))
    if outcome_timestamp > latest_timestamp:
        latest_timestamp = outcome_timestamp
    return (
        latest_timestamp,
        len(getattr(task, "steps", []) or []),
        str(getattr(task, "task_id", "")),
    )


def _dedupe_tasks(tasks: list[Any]) -> list[Any]:
    """Keep only the latest trace per normalized task text."""
    deduped: dict[str, tuple[tuple[str, int, str], int, Any]] = {}
    for index, task in enumerate(tasks):
        normalized = _normalize_task_text(task)
        recency = _task_recency_key(task)
        existing = deduped.get(normalized)
        if existing is None or (recency, index) >= (existing[0], existing[1]):
            deduped[normalized] = (recency, index, task)
    keep_ids = {id(entry[2]) for entry in deduped.values()}
    return [task for task in tasks if id(task) in keep_ids]
