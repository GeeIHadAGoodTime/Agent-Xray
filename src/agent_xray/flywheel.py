from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from .analyzer import analyze_task, load_tasks
from .diagnose import build_fix_plan
from .grader import GradeResult, grade_task, load_rules
from .replay import replay_fixture
from .root_cause import RootCauseResult, classify_task
from .signals import SignalDetector

GRADE_ORDER = {
    "BROKEN": 0,
    "WEAK": 1,
    "OK": 2,
    "GOOD": 3,
    "GOLDEN": 4,
}
PASSING_GRADES = {"GOLDEN", "GOOD"}
FAILING_GRADES = {"WEAK", "BROKEN"}
DetectorHook = Callable[..., RootCauseResult | None]


@dataclass(slots=True)
class FlywheelResult:
    """Complete flywheel analysis result with optional baseline comparison."""

    grade_distribution: dict[str, int]
    root_cause_distribution: dict[str, int]
    total_tasks: int
    passing_tasks: int
    failing_tasks: int
    fix_plan: list[dict[str, Any]]
    baseline_grade_distribution: dict[str, int] | None = None
    grade_deltas: dict[str, int] | None = None
    regressions: list[str] | None = None
    improvements: list[str] | None = None
    trend: str | None = None
    task_grades: dict[str, str] = field(default_factory=dict)
    task_scores: dict[str, int] = field(default_factory=dict)
    rules_name: str | None = None
    fixture_replays: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _distribution(values: list[str], order: list[str]) -> dict[str, int]:
    counts = Counter(values)
    return {name: counts.get(name, 0) for name in order if counts.get(name, 0) or name in counts}


def _load_baseline(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("baseline file must contain a JSON object")
    return cast(dict[str, Any], payload)


def _extract_baseline_task_grades(payload: dict[str, Any]) -> dict[str, str]:
    if isinstance(payload.get("task_grades"), dict):
        return {
            str(task_id): str(grade)
            for task_id, grade in payload["task_grades"].items()
            if isinstance(grade, str)
        }
    summary_grades = payload.get("summary", {}).get("grades")
    if isinstance(summary_grades, dict):
        return {
            str(task_id): str(meta.get("grade"))
            for task_id, meta in summary_grades.items()
            if isinstance(meta, dict) and meta.get("grade")
        }
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return {
            str(item.get("task_id")): str(item.get("grade"))
            for item in tasks
            if isinstance(item, dict) and item.get("task_id") and item.get("grade")
        }
    return {}


def _extract_baseline_distribution(payload: dict[str, Any]) -> dict[str, int] | None:
    distribution = payload.get("baseline_grade_distribution") or payload.get("grade_distribution")
    if isinstance(distribution, dict):
        return {str(key): int(value) for key, value in distribution.items()}
    summary = payload.get("summary", {})
    for key in ("grade_distribution", "distribution"):
        if isinstance(summary.get(key), dict):
            return {str(name): int(count) for name, count in summary[key].items()}
    return None


def _compare_baseline(
    current_grades: dict[str, str],
    current_distribution: dict[str, int],
    baseline_payload: dict[str, Any],
) -> dict[str, Any]:
    baseline_distribution = _extract_baseline_distribution(baseline_payload) or {}
    baseline_task_grades = _extract_baseline_task_grades(baseline_payload)
    all_grade_names = sorted(
        set(current_distribution) | set(baseline_distribution),
        key=lambda name: GRADE_ORDER.get(name, -1),
        reverse=True,
    )
    grade_deltas = {
        name: current_distribution.get(name, 0) - baseline_distribution.get(name, 0)
        for name in all_grade_names
    }
    regressions: list[str] = []
    improvements: list[str] = []
    total_delta = 0
    for task_id, current_grade in current_grades.items():
        baseline_grade = baseline_task_grades.get(task_id)
        if baseline_grade is None:
            continue
        delta = GRADE_ORDER.get(current_grade, 0) - GRADE_ORDER.get(baseline_grade, 0)
        total_delta += delta
        if delta < 0:
            regressions.append(task_id)
        elif delta > 0:
            improvements.append(task_id)
    if total_delta > 0:
        trend = "improving"
    elif total_delta < 0:
        trend = "degrading"
    else:
        trend = "stable"
    return {
        "baseline_grade_distribution": baseline_distribution or None,
        "grade_deltas": grade_deltas or None,
        "regressions": sorted(regressions) or None,
        "improvements": sorted(improvements) or None,
        "trend": trend,
    }


def _classify_failures_with_hooks(
    tasks: list[Any],
    grades: list[GradeResult],
    analyses: dict[str, Any],
    detectors: list[SignalDetector] | list[DetectorHook] | None,
) -> list[RootCauseResult]:
    failures: list[RootCauseResult] = []
    hooks = [
        hook
        for hook in (detectors or [])
        if callable(hook) and not isinstance(hook, SignalDetector)
    ]
    grade_by_task = {grade.task_id: grade for grade in grades}
    for task in tasks:
        grade = grade_by_task[task.task_id]
        if grade.grade not in FAILING_GRADES:
            continue
        default = classify_task(task, grade, analyses[task.task_id])
        chosen = default
        for hook in hooks:
            try:
                candidate = hook(task, grade, analyses[task.task_id], chosen)
            except TypeError:
                candidate = hook(task, grade, analyses[task.task_id])
            if isinstance(candidate, RootCauseResult):
                chosen = candidate
                break
        failures.append(chosen)
    return failures


def _replay_fixtures(fixture_dir: Path | None, tasks: list[Any]) -> list[dict[str, Any]]:
    if fixture_dir is None or not fixture_dir.exists():
        return []
    if fixture_dir.is_file():
        return [replay_fixture(fixture_dir, tasks)]
    results: list[dict[str, Any]] = []
    for path in sorted(fixture_dir.glob("*.json")):
        results.append(replay_fixture(path, tasks))
    return results


def run_flywheel(
    log_dir: Path | str,
    *,
    rules_path: Path | str | None = None,
    fixture_dir: Path | str | None = None,
    baseline_path: Path | str | None = None,
    output_path: Path | str | None = None,
    detectors: list[SignalDetector] | list[DetectorHook] | None = None,
) -> FlywheelResult:
    tasks = load_tasks(log_dir)
    rules = load_rules(rules_path)
    signal_detectors = [
        detector for detector in (detectors or []) if isinstance(detector, SignalDetector)
    ]
    analyses = {
        task.task_id: analyze_task(task, detectors=signal_detectors or None) for task in tasks
    }
    grades = [grade_task(task, rules, analysis=analyses[task.task_id]) for task in tasks]
    root_causes = _classify_failures_with_hooks(tasks, grades, analyses, detectors)

    current_distribution = _distribution(
        [grade.grade for grade in grades],
        ["GOLDEN", "GOOD", "OK", "WEAK", "BROKEN"],
    )
    root_cause_distribution = _distribution(
        [result.root_cause for result in root_causes],
        sorted({result.root_cause for result in root_causes}),
    )
    result = FlywheelResult(
        grade_distribution=current_distribution,
        root_cause_distribution=root_cause_distribution,
        total_tasks=len(tasks),
        passing_tasks=sum(1 for grade in grades if grade.grade in PASSING_GRADES),
        failing_tasks=sum(1 for grade in grades if grade.grade in FAILING_GRADES),
        fix_plan=[entry.to_dict() for entry in build_fix_plan(root_causes)],
        task_grades={grade.task_id: grade.grade for grade in grades},
        task_scores={grade.task_id: grade.score for grade in grades},
        rules_name=rules.name,
        fixture_replays=_replay_fixtures(Path(fixture_dir) if fixture_dir else None, tasks),
    )

    if baseline_path:
        baseline_payload = _load_baseline(Path(baseline_path))
        comparison = _compare_baseline(result.task_grades, current_distribution, baseline_payload)
        result.baseline_grade_distribution = comparison["baseline_grade_distribution"]
        result.grade_deltas = comparison["grade_deltas"]
        result.regressions = comparison["regressions"]
        result.improvements = comparison["improvements"]
        result.trend = comparison["trend"]

    if output_path:
        Path(output_path).write_text(
            json.dumps(result.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    return result
