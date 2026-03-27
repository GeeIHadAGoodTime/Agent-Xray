from __future__ import annotations

from pathlib import Path
from typing import Any

from .analyzer import load_tasks
from .diagnose import build_fix_plan
from .grader import grade_tasks, load_rules
from .replay import replay_fixture
from .root_cause import classify_failures, summarize_root_causes


def run_flywheel(
    log_dir: str | Path,
    *,
    rules_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
    days: int | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(log_dir, days=days)
    rules = load_rules(rules_path)
    grades = grade_tasks(tasks, rules)
    root_causes = classify_failures(tasks, grades)
    result: dict[str, Any] = {
        "summary": {
            "tasks": len(tasks),
            "rules": rules.name,
            "grades": {
                grade.task_id: {"grade": grade.grade, "score": grade.score} for grade in grades
            },
        },
        "root_causes": summarize_root_causes(root_causes),
        "fix_plan": [entry.to_dict() for entry in build_fix_plan(root_causes)],
    }
    if fixture_path:
        result["replay"] = replay_fixture(fixture_path, tasks)
    return result
