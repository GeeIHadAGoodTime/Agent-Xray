from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .analyzer import analyze_task, resolve_task
from .capture import detect_milestone
from .schema import AgentTask

MILESTONE_ORDER = ["CART", "FORM_FILL", "CHECKOUT", "PAYMENT"]
MILESTONE_RANK = {name: index for index, name in enumerate(MILESTONE_ORDER)}


def load_fixture(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture must contain a JSON object")
    return cast(dict[str, Any], payload)


def text_similarity(left: str, right: str) -> float:
    left_words = set(left.lower().split())
    right_words = set(right.lower().split())
    if not left_words or not right_words:
        return 0.0
    return len(left_words & right_words) / len(left_words | right_words)


def task_milestones(task: AgentTask) -> list[str]:
    milestones: list[str] = []
    for step in task.sorted_steps:
        milestone = detect_milestone(step)
        if milestone and milestone not in milestones:
            milestones.append(milestone)
    return milestones


def _max_rank(milestones: list[str]) -> int:
    if not milestones:
        return -1
    return max(MILESTONE_RANK.get(milestone, -1) for milestone in milestones)


def find_best_match(fixture: dict[str, Any], tasks: list[AgentTask]) -> AgentTask | None:
    fixture_id = fixture.get("task_id")
    if fixture_id:
        try:
            return resolve_task(tasks, str(fixture_id))
        except KeyError:
            pass
    fixture_text = str(fixture.get("user_text") or "")
    fixture_site = str(fixture.get("site") or "")
    best_match = None
    best_score = 0.0
    for task in tasks:
        analysis = analyze_task(task)
        if fixture_site and analysis.site_name != fixture_site:
            continue
        score = text_similarity(fixture_text, task.task_text or "")
        if score > best_score:
            best_score = score
            best_match = task
    return best_match if best_score >= 0.4 else None


def _check_integrity_hashes(integrity_hashes: dict[str, str] | None) -> list[str]:
    """Verify integrity hashes and return list of drifted file descriptions."""
    if not integrity_hashes:
        return []
    from .flywheel import IntegrityLock, check_integrity

    locks = [
        IntegrityLock(file_path=path, expected_hash=h, actual_hash=h)
        for path, h in integrity_hashes.items()
    ]
    violated = check_integrity(locks)
    return [lock.file_path for lock in violated]


def compare_fixture_to_task(
    fixture: dict[str, Any],
    task: AgentTask,
    *,
    integrity_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    # Check for evaluation drift before comparing
    drifted = _check_integrity_hashes(integrity_hashes)
    if drifted:
        return {
            "fixture_task_id": fixture.get("task_id"),
            "current_task_id": task.task_id,
            "verdict": "EVALUATION_DRIFT",
            "detail": f"evaluator files changed during run: {', '.join(drifted)}",
        }

    analysis = analyze_task(task)
    current_milestones = task_milestones(task)
    golden_milestones = [str(item) for item in fixture.get("milestones_reached", [])]
    current_rank = _max_rank(current_milestones)
    golden_rank = _max_rank(golden_milestones)
    step_delta = len(task.steps) - int(fixture.get("total_steps", 0) or 0)
    if current_rank > golden_rank:
        verdict = "IMPROVED"
        detail = "reached a later milestone"
    elif current_rank < golden_rank:
        verdict = "REGRESSION"
        detail = "failed to reach the golden milestone depth"
    elif step_delta >= 5:
        verdict = "REGRESSION"
        detail = f"same milestone depth but {step_delta} more steps"
    elif step_delta <= -2:
        verdict = "IMPROVED"
        detail = f"same milestone depth with {abs(step_delta)} fewer steps"
    else:
        verdict = "STABLE"
        detail = "milestones and step count are similar"
    return {
        "fixture_task_id": fixture.get("task_id"),
        "current_task_id": task.task_id,
        "site": analysis.site_name,
        "golden_milestones": golden_milestones,
        "current_milestones": current_milestones,
        "golden_steps": int(fixture.get("total_steps", 0) or 0),
        "current_steps": len(task.steps),
        "step_delta": step_delta,
        "verdict": verdict,
        "detail": detail,
    }


def replay_fixture(
    fixture_path: str | Path,
    tasks: list[AgentTask],
    *,
    integrity_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    fixture = load_fixture(fixture_path)
    task = find_best_match(fixture, tasks)
    if task is None:
        return {
            "fixture_task_id": fixture.get("task_id"),
            "verdict": "UNMATCHED",
            "detail": "no suitable task found in the current log selection",
        }
    return compare_fixture_to_task(fixture, task, integrity_hashes=integrity_hashes)


def format_replay_text(result: dict[str, Any]) -> str:
    return (
        f"REPLAY {result.get('fixture_task_id')} -> {result.get('current_task_id', '(none)')}\n"
        f"verdict: {result['verdict']}\n"
        f"detail: {result['detail']}"
    )
