from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any

from .analyzer import TaskAnalysis, analyze_task
from .schema import AgentTask


@dataclass(slots=True)
class RuleSet:
    name: str
    description: str
    signals: list[dict[str, Any]]
    grade_thresholds: dict[str, int]
    golden_requirements: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SignalResult:
    name: str
    passed: bool
    points: int
    actual: Any
    reason: str


@dataclass(slots=True)
class GradeResult:
    task_id: str
    grade: str
    score: int
    reasons: list[str]
    metrics: dict[str, Any]
    signals: list[SignalResult]


def _default_rules_path() -> Path:
    return Path(str(files("agent_xray.rules").joinpath("default.json")))


def load_rules(path: str | Path | None = None) -> RuleSet:
    rule_path = Path(path) if path else _default_rules_path()
    payload = json.loads(rule_path.read_text(encoding="utf-8"))
    return RuleSet(
        name=payload["name"],
        description=payload.get("description", ""),
        signals=list(payload.get("signals", [])),
        grade_thresholds=dict(payload.get("grade_thresholds", {})),
        golden_requirements=list(payload.get("golden_requirements", [])),
    )


def _compare(actual: Any, rule: dict[str, Any]) -> bool:
    if "gte" in rule:
        return actual >= rule["gte"]
    if "gt" in rule:
        return actual > rule["gt"]
    if "lte" in rule:
        return actual <= rule["lte"]
    if "lt" in rule:
        return actual < rule["lt"]
    if "equals" in rule:
        return actual == rule["equals"]
    if "in" in rule:
        return actual in rule["in"]
    if "contains_any" in rule:
        values = actual if isinstance(actual, list) else [actual]
        return any(item in values for item in rule["contains_any"])
    raise ValueError(f"rule '{rule.get('name', '<unknown>')}' is missing a comparator")


def _reason(rule: dict[str, Any], passed: bool, actual: Any) -> str:
    template = rule["reason"] if passed else rule.get("else_reason", "")
    return template.format(actual=actual, expected=rule.get("gte") or rule.get("equals"))


def grade_task(
    task: AgentTask, rules: RuleSet, analysis: TaskAnalysis | None = None
) -> GradeResult:
    analysis = analysis or analyze_task(task)
    metrics = analysis.metrics()
    score = 0
    reasons: list[str] = []
    signal_results: list[SignalResult] = []
    for rule in rules.signals:
        actual = metrics.get(rule["metric"])
        passed = _compare(actual, rule)
        points = int(rule["points"]) if passed else int(rule.get("else_points", 0))
        score += points
        reason = _reason(rule, passed, actual)
        if reason:
            reasons.append(reason)
        signal_results.append(
            SignalResult(
                name=rule["name"],
                passed=passed,
                points=points,
                actual=actual,
                reason=reason,
            )
        )
    thresholds = rules.grade_thresholds
    golden_floor = int(thresholds.get("GOLDEN", 8))
    good_floor = int(thresholds.get("GOOD", 5))
    ok_floor = int(thresholds.get("OK", 2))
    weak_floor = int(thresholds.get("WEAK", 0))
    if score >= golden_floor:
        grade = "GOLDEN"
    elif score >= good_floor:
        grade = "GOOD"
    elif score >= ok_floor:
        grade = "OK"
    elif score >= weak_floor:
        grade = "WEAK"
    else:
        grade = "BROKEN"
    if grade == "GOLDEN" and rules.golden_requirements:
        unmet = []
        for requirement in rules.golden_requirements:
            actual = metrics.get(requirement["metric"])
            if not _compare(actual, requirement):
                unmet.append(
                    requirement.get("reason") or f"{requirement['metric']} requirement not met"
                )
        if unmet:
            grade = "GOOD"
            reasons.extend(unmet)
    return GradeResult(
        task_id=task.task_id,
        grade=grade,
        score=score,
        reasons=reasons,
        metrics=metrics,
        signals=signal_results,
    )


def grade_tasks(tasks: list[AgentTask], rules: RuleSet) -> list[GradeResult]:
    analyses = {task.task_id: analyze_task(task) for task in tasks}
    return [grade_task(task, rules, analysis=analyses[task.task_id]) for task in tasks]
