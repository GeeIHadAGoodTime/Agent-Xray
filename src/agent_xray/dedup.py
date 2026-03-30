from __future__ import annotations

import re
from typing import Any


def _normalize_task_text(task: Any) -> str:
    text = str(getattr(task, "task_text", None) or "").strip().lower()
    normalized = re.sub(r"\s+", " ", text)
    return normalized or str(getattr(task, "task_id", ""))


def _task_recency_key(task: Any) -> tuple[str, int, str]:
    latest_timestamp = ""
    for step in getattr(task, "steps", []) or []:
        timestamp = str(getattr(step, "timestamp", "") or "")
        if timestamp > latest_timestamp:
            latest_timestamp = timestamp
    outcome = getattr(task, "outcome", None)
    outcome_timestamp = str(getattr(outcome, "timestamp", "") or "")
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
