"""Live watch mode: tail a JSONL log file and grade tasks as they complete."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

from .analyzer import analyze_task
from .grader import GradeResult, grade_task, load_rules
from .schema import AgentStep, AgentTask, TaskOutcome

GRADE_LABELS = ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"]
ANSI_RESET = "\033[0m"
GRADE_COLORS = {
    "GOLDEN": "\033[32m",
    "GOOD": "\033[36m",
    "OK": "\033[37m",
    "WEAK": "\033[33m",
    "BROKEN": "\033[31m",
}


def _colorize(text: str, grade: str, *, color: bool = True) -> str:
    if not color:
        return text
    code = GRADE_COLORS.get(grade)
    if code is None:
        return text
    return f"{code}{text}{ANSI_RESET}"


def _truncate(text: str, max_len: int = 50) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _extract_timestamp_time(task: AgentTask) -> str:
    """Extract HH:MM:SS from the outcome or last step timestamp."""
    ts: str | None = None
    if task.outcome and task.outcome.timestamp:
        ts = task.outcome.timestamp
    elif task.steps:
        last = task.sorted_steps[-1]
        ts = last.timestamp
    if ts and len(ts) >= 19:
        # ISO-8601: 2026-03-26T12:03:30Z -> 12:03:30
        try:
            return ts[11:19]
        except (IndexError, ValueError):
            pass
    return "??:??:??"


def _format_line(
    task: AgentTask,
    grade: GradeResult,
    *,
    color: bool = True,
) -> str:
    ts = _extract_timestamp_time(task)
    task_id = task.task_id[:12]
    grade_label = grade.grade
    score_sign = "+" if grade.score >= 0 else ""
    step_count = len(task.steps)
    outcome_status = task.outcome.status if task.outcome else "unknown"
    task_text = _truncate(task.task_text or "", 50)

    colored_grade = _colorize(f"{grade_label:6s} ({score_sign}{grade.score})", grade_label, color=color)
    return f"{ts}  {task_id:<12s}  {colored_grade}  {step_count:3d} steps  {outcome_status:<18s}  \"{task_text}\""


def _format_tally(counts: dict[str, int], *, color: bool = True) -> str:
    total = sum(counts.values())
    parts = [f"Total: {total}"]
    for label in GRADE_LABELS:
        count = counts.get(label, 0)
        parts.append(f"{label}: {count}")
    inner = " | ".join(parts)
    return f"[{inner}]"


def _build_task_from_accumulated(
    task_id: str,
    steps: list[dict[str, Any]],
    outcome: dict[str, Any] | None,
) -> AgentTask:
    """Build an AgentTask from accumulated raw step dicts."""
    parsed_steps = []
    task_text: str | None = None
    task_category: str | None = None
    for raw in steps:
        if raw.get("user_text") and not task_text:
            task_text = str(raw["user_text"])
        if raw.get("task_category") and not task_category:
            task_category = str(raw["task_category"])
        parsed_steps.append(AgentStep.from_dict(raw))

    parsed_outcome: TaskOutcome | None = None
    if outcome is not None:
        parsed_outcome = TaskOutcome.from_dict(outcome)
        if outcome.get("user_text") and not task_text:
            task_text = str(outcome["user_text"])
        if outcome.get("task_category") and not task_category:
            task_category = str(outcome["task_category"])

    return AgentTask(
        task_id=task_id,
        steps=parsed_steps,
        task_text=task_text,
        task_category=task_category,
        outcome=parsed_outcome,
    )


def watch_file(
    path: str | Path,
    *,
    rules_path: str | None = None,
    poll_interval: float = 2.0,
    json_output: bool = False,
    color: bool = True,
) -> None:
    """Tail a JSONL log file and print grades as tasks complete.

    Args:
        path: Path to the JSONL log file.
        rules_path: Optional ruleset name or path.
        poll_interval: Seconds between polls (default 2).
        json_output: Emit one JSON object per completed task.
        color: Use ANSI colors in text output.
    """
    file_path = Path(path)
    if not file_path.exists():
        print(f"File not found: {file_path}", file=sys.stderr)
        return

    rules = load_rules(rules_path)

    # Accumulated state: task_id -> list of raw step dicts
    task_steps: dict[str, list[dict[str, Any]]] = {}
    task_outcomes: dict[str, dict[str, Any]] = {}
    graded_tasks: set[str] = set()
    grade_counts: dict[str, int] = {label: 0 for label in GRADE_LABELS}
    file_position = 0

    if not json_output:
        print(f"Watching {file_path} (rules: {rules.name}, poll: {poll_interval}s)")
        print(f"Press Ctrl+C to stop.\n")

    try:
        while True:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as fh:
                    fh.seek(file_position)
                    new_lines = fh.readlines()
                    file_position = fh.tell()
            except OSError as exc:
                if not json_output:
                    print(f"Read error: {exc}", file=sys.stderr)
                time.sleep(poll_interval)
                continue

            new_completed: list[str] = []

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                task_id = str(payload.get("task_id") or "")
                if not task_id:
                    continue

                event = payload.get("event")
                is_outcome = event == "task_complete" or (
                    payload.get("outcome") is not None
                    and payload.get("tool_name") in (None, "")
                )

                if is_outcome:
                    task_outcomes[task_id] = payload
                    if task_id not in graded_tasks:
                        new_completed.append(task_id)
                elif payload.get("tool_name"):
                    task_steps.setdefault(task_id, []).append(payload)

            # Grade newly completed tasks
            for task_id in new_completed:
                if task_id in graded_tasks:
                    continue
                graded_tasks.add(task_id)

                steps = task_steps.get(task_id, [])
                outcome = task_outcomes.get(task_id)
                task = _build_task_from_accumulated(task_id, steps, outcome)

                if not task.steps:
                    continue

                grade = grade_task(task, rules)
                grade_counts[grade.grade] = grade_counts.get(grade.grade, 0) + 1

                if json_output:
                    obj = {
                        "task_id": task.task_id,
                        "grade": grade.grade,
                        "score": grade.score,
                        "step_count": len(task.steps),
                        "outcome": task.outcome.status if task.outcome else "unknown",
                        "task_text": task.task_text or "",
                        "timestamp": _extract_timestamp_time(task),
                    }
                    print(json.dumps(obj), flush=True)
                else:
                    print(_format_line(task, grade, color=color), flush=True)

            # Print running tally if we emitted any results and not in json mode
            if new_completed and not json_output:
                print(_format_tally(grade_counts, color=color), flush=True)

            time.sleep(poll_interval)

    except KeyboardInterrupt:
        if not json_output:
            print(f"\n\nFinal tally:")
            print(_format_tally(grade_counts, color=color))
