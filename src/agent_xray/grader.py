from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from itertools import combinations
from pathlib import Path
from typing import Any

from .analyzer import TaskAnalysis, analyze_task
from .schema import AgentTask

SUPPORTED_OPERATORS = ("gte", "gt", "lte", "lt", "equals", "in", "contains_any", "ne", "not_in")
REQUIRED_GRADE_THRESHOLDS = ("GOLDEN", "GOOD", "OK", "WEAK")


@dataclass(slots=True)
class RuleSet:
    """Grading configuration loaded from a rules JSON file.

    Attributes:
        name: Ruleset name.
        description: Human-readable summary of what the ruleset scores.
        signals: Rule objects evaluated against analysis metrics.
        grade_thresholds: Raw-score cutoffs for each grade bucket.
        golden_requirements: Extra requirements that must pass for ``GOLDEN``.
    """

    name: str
    description: str
    signals: list[dict[str, Any]]
    grade_thresholds: dict[str, int]
    golden_requirements: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class SignalResult:
    name: str
    passed: bool
    points: int
    actual: Any
    reason: str


@dataclass(slots=True)
class GradeResult:
    """Per-task grading outcome with raw and normalized scores.

    Attributes:
        task_id: Identifier of the graded task.
        grade: Assigned grade label.
        score: Raw integer score accumulated from the rules.
        reasons: Human-readable reasons emitted by matching rules.
        metrics: Flattened metrics map used during grading.
        signals: Per-rule pass/fail results.
        normalized_score: Score normalized to the ``0.0`` to ``1.0`` range.
    """

    task_id: str
    grade: str
    score: int
    reasons: list[str]
    metrics: dict[str, Any]
    signals: list[SignalResult]
    normalized_score: float = 0.0


def _default_rules_path() -> Path:
    return Path(str(files("agent_xray.rules").joinpath("default.json")))


def _resolve_rules_path(path: str | Path | None) -> Path:
    if path is None:
        return _default_rules_path()
    candidate = Path(path)
    if candidate.exists():
        return candidate
    if candidate.suffix != ".json":
        bundled = Path(str(files("agent_xray.rules").joinpath(f"{candidate.name}.json")))
        if bundled.exists():
            return bundled
    bundled = Path(str(files("agent_xray.rules").joinpath(candidate.name)))
    if bundled.exists():
        return bundled
    return candidate


def load_rules(path: str | Path | None = None) -> RuleSet:
    """Load a ruleset from a user path or bundled rules name.

    Args:
        path: File path, bundled rules basename, or ``None`` for the default
            bundled ruleset.

    Returns:
        RuleSet: Parsed grading configuration.

    Example:
        >>> rules = load_rules("default")
        >>> rules.name
        'default'
    """

    rule_path = _resolve_rules_path(path)
    raw_payload = json.loads(rule_path.read_text(encoding="utf-8"))
    if not isinstance(raw_payload, dict):
        raise ValueError(f"rules file did not contain a JSON object: {rule_path}")
    payload = {str(key): value for key, value in raw_payload.items()}
    # Handle ruleset inheritance
    if "extends" in payload:
        base_path = _resolve_rules_path(payload["extends"])
        base_raw = json.loads(base_path.read_text(encoding="utf-8"))
        if not isinstance(base_raw, dict):
            raise ValueError(f"base rules file is not a JSON object: {base_path}")
        base = {str(k): v for k, v in base_raw.items()}
        # Merge signals: base first, child overrides by name
        base_signals = {s.get("name"): s for s in base.get("signals", []) if isinstance(s, dict)}
        for sig in payload.get("signals", []):
            if isinstance(sig, dict) and sig.get("name"):
                base_signals[sig["name"]] = sig
            elif isinstance(sig, dict):
                base_signals[id(sig)] = sig
        payload.setdefault("signals", list(base_signals.values()))
        # Merge thresholds: base, then child overrides
        merged_thresholds = dict(base.get("grade_thresholds", base.get("thresholds", {})))
        merged_thresholds.update(payload.get("grade_thresholds", payload.get("thresholds", {})))
        payload["grade_thresholds"] = merged_thresholds
        # Merge golden_requirements: base + child
        base_golden = [r for r in base.get("golden_requirements", []) if isinstance(r, (str, dict))]
        child_golden = [r for r in payload.get("golden_requirements", []) if isinstance(r, (str, dict))]
        payload["golden_requirements"] = base_golden + child_golden
    rules = RuleSet(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        signals=[
            {str(key): value for key, value in rule.items()}
            for rule in payload.get("signals", [])
            if isinstance(rule, dict)
        ],
        grade_thresholds=dict(payload.get("grade_thresholds", payload.get("thresholds", {}))),
        golden_requirements=[
            item for item in payload.get("golden_requirements", []) if isinstance(item, (str, dict))
        ],
    )
    for warning in validate_rules(rules):
        print(f"[agent_xray] rules warning ({rule_path}): {warning}", file=sys.stderr)
    return rules


def _normalize_rule(rule: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(rule)
    if "op" in normalized:
        normalized["op"] = str(normalized["op"])
        normalized["value"] = normalized.get("value")
        return normalized
    for operator in SUPPORTED_OPERATORS:
        if operator in normalized:
            normalized["op"] = operator
            normalized["value"] = normalized[operator]
            return normalized
    raise ValueError(f"rule '{_rule_name(rule)}' is missing a comparator")


def _compare(actual: Any, rule: dict[str, Any]) -> bool:
    """Evaluate a rule against a metric value.

    Both legacy shorthand comparators such as ``{"gte": 3}`` and normalized
    ``{"op": "gte", "value": 3}`` forms are accepted.
    """

    normalized = _normalize_rule(rule)
    try:
        op = normalized["op"]
        expected = normalized.get("value")
        if op == "gte":
            return bool(actual >= expected)
        if op == "gt":
            return bool(actual > expected)
        if op == "lte":
            return bool(actual <= expected)
        if op == "lt":
            return bool(actual < expected)
        if op == "equals":
            return bool(actual == expected)
        if op == "in":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return bool(actual in expected)
        if op == "contains_any":
            if not isinstance(expected, (list, tuple, set)):
                return False
            values = actual if isinstance(actual, list) else [actual]
            return any(item in values for item in expected)
        if op == "ne":
            return bool(actual != expected)
        if op == "not_in":
            if not isinstance(expected, (list, tuple, set)):
                return False
            return bool(actual not in expected)
    except TypeError as exc:
        warnings.warn(
            f"rule comparison raised TypeError for {_rule_name(rule)}: {exc}",
            stacklevel=2,
        )
        return False
    raise ValueError(f"rule '{_rule_name(rule)}' uses unknown operator '{op}'")


def _rule_field(rule: dict[str, Any]) -> str:
    return str(rule.get("field") or rule.get("metric") or "")


def _rule_name(rule: dict[str, Any]) -> str:
    return str(rule.get("name") or rule.get("label") or _rule_field(rule) or "<unknown>")


def _expected_value(rule: dict[str, Any]) -> Any:
    try:
        return _normalize_rule(rule).get("value")
    except ValueError:
        return None


def _resolve_metric(metrics: dict[str, Any], field_name: str) -> Any:
    """Resolve a metric by dotted path from the flattened analysis metrics map."""

    if not field_name:
        return None
    current: Any = metrics
    for part in field_name.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _reason(rule: dict[str, Any], passed: bool, actual: Any) -> str:
    template = str(rule.get("reason", "") if passed else rule.get("else_reason", ""))
    if not template:
        return ""
    return template.format(actual=actual, expected=_expected_value(rule))


def normalize_score(raw_score: int, rule_set: RuleSet) -> float:
    """Normalize a raw score into the ``0.0`` to ``1.0`` range for a ruleset."""

    minimum = 0
    maximum = 0
    for rule in rule_set.signals:
        passed_points = int(rule.get("points", 0))
        failed_points = int(rule.get("else_points", 0))
        minimum += min(passed_points, failed_points)
        maximum += max(passed_points, failed_points)
    if maximum == minimum:
        return 1.0
    bounded = min(max(raw_score, minimum), maximum)
    return (bounded - minimum) / (maximum - minimum)


@lru_cache(maxsize=1)
def _available_metric_paths() -> frozenset[str]:
    metrics = analyze_task(AgentTask(task_id="validation-probe", steps=[])).metrics()
    paths: set[str] = set()

    def visit(prefix: str, value: Any) -> None:
        if prefix:
            paths.add(prefix)
        if isinstance(value, dict):
            for key, nested in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                visit(child, nested)

    for key, value in metrics.items():
        visit(str(key), value)
    return frozenset(paths)


def _rule_has_comparison_value(rule: dict[str, Any], normalized: dict[str, Any]) -> bool:
    if "value" in rule:
        return True
    return str(normalized["op"]) in rule


def _points_polarity(rule: dict[str, Any]) -> set[int]:
    polarity: set[int] = set()
    for key in ("points", "else_points"):
        value = int(rule.get(key, 0))
        if value > 0:
            polarity.add(1)
        elif value < 0:
            polarity.add(-1)
    return polarity


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _range_bounds(rule: dict[str, Any]) -> tuple[float | None, bool, float | None, bool] | None:
    op = str(rule.get("op"))
    expected = rule.get("value")
    if not _is_numeric(expected):
        return None
    val = float(expected)  # type: ignore[arg-type]  # guarded by _is_numeric
    if op == "gt":
        return (val, False, None, False)
    if op == "gte":
        return (val, True, None, False)
    if op == "lt":
        return (None, False, val, False)
    if op == "lte":
        return (None, False, val, True)
    return None


def _ranges_overlap(
    left: tuple[float | None, bool, float | None, bool],
    right: tuple[float | None, bool, float | None, bool],
) -> bool:
    lower: float | None = None
    lower_inclusive = False
    for candidate, inclusive in ((left[0], left[1]), (right[0], right[1])):
        if candidate is None:
            continue
        if lower is None or candidate > lower:
            lower = candidate
            lower_inclusive = inclusive
        elif candidate == lower:
            lower_inclusive = lower_inclusive and inclusive

    upper: float | None = None
    upper_inclusive = False
    for candidate, inclusive in ((left[2], left[3]), (right[2], right[3])):
        if candidate is None:
            continue
        if upper is None or candidate < upper:
            upper = candidate
            upper_inclusive = inclusive
        elif candidate == upper:
            upper_inclusive = upper_inclusive and inclusive

    if lower is None or upper is None:
        return True
    if lower < upper:
        return True
    return lower == upper and lower_inclusive and upper_inclusive


def _rule_overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if _rule_field(left) != _rule_field(right):
        return False
    if left["op"] == right["op"] and left.get("value") == right.get("value"):
        return True
    left_range = _range_bounds(left)
    right_range = _range_bounds(right)
    if left_range and right_range:
        return _ranges_overlap(left_range, right_range)
    for range_rule, scalar_rule in ((left, right), (right, left)):
        bounds = _range_bounds(range_rule)
        if (
            bounds is None
            or scalar_rule["op"] != "equals"
            or not _is_numeric(scalar_rule.get("value"))
        ):
            continue
        value = float(scalar_rule["value"])
        lower, lower_inclusive, upper, upper_inclusive = bounds
        lower_ok = lower is None or value > lower or (lower_inclusive and value == lower)
        upper_ok = upper is None or value < upper or (upper_inclusive and value == upper)
        if lower_ok and upper_ok:
            return True
    if left["op"] in {"in", "not_in"} and right["op"] in {"in", "not_in"}:
        left_values = left.get("value")
        right_values = right.get("value")
        if isinstance(left_values, (list, tuple, set)) and isinstance(
            right_values, (list, tuple, set)
        ):
            return bool(set(left_values) & set(right_values))
    return False


def validate_rules(rules: RuleSet) -> list[str]:
    """Validate a ruleset and return human-readable warnings."""

    warnings_list: list[str] = []
    available_fields = _available_metric_paths()
    normalized_rules: list[dict[str, Any]] = []

    for threshold in REQUIRED_GRADE_THRESHOLDS:
        if threshold not in rules.grade_thresholds:
            warnings_list.append(f"missing grade threshold '{threshold}'")

    for index, raw_rule in enumerate(rules.signals, start=1):
        field_name = _rule_field(raw_rule)
        rule_name = _rule_name(raw_rule)
        normalized: dict[str, Any] | None
        try:
            normalized = _normalize_rule(raw_rule)
        except ValueError as exc:
            warnings_list.append(str(exc))
            normalized = None

        if field_name and field_name not in available_fields:
            warnings_list.append(
                f"rule {index} '{rule_name}' references unknown field '{field_name}'"
            )

        if normalized is not None:
            op = normalized["op"]
            if op not in SUPPORTED_OPERATORS:
                warnings_list.append(f"rule {index} '{rule_name}' uses unknown operator '{op}'")
            elif not _rule_has_comparison_value(raw_rule, normalized):
                warnings_list.append(
                    f"rule {index} '{rule_name}' is missing a comparison value for '{op}'"
                )
            normalized_rules.append({"index": index, **normalized})

    for left, right in combinations(normalized_rules, 2):
        if not _rule_overlaps(left, right):
            continue
        left_name = _rule_name(left)
        right_name = _rule_name(right)
        left_polarity = _points_polarity(left)
        right_polarity = _points_polarity(right)
        if left_polarity and right_polarity and left_polarity != right_polarity:
            warnings_list.append(
                f"rules {left['index']} '{left_name}' and {right['index']} '{right_name}' overlap with contradictory scoring"
            )
        elif left["op"] == right["op"] and left.get("value") == right.get("value"):
            warnings_list.append(
                f"rules {left['index']} '{left_name}' and {right['index']} '{right_name}' overlap on the same field"
            )

    return warnings_list


def grade_task(
    task: AgentTask,
    rules: RuleSet | None = None,
    *,
    rules_path: str | Path | None = None,
    analysis: TaskAnalysis | None = None,
) -> GradeResult:
    """Grade a single task against a ruleset.

    Args:
        task: Task to grade.
        rules: Optional already-loaded ruleset.
        rules_path: Optional path or bundled rules name to load when
            ``rules`` is not supplied.
        analysis: Optional precomputed task analysis to reuse.

    Returns:
        GradeResult: Grade, score, reasons, and per-rule signal outcomes.
    """

    rules = rules or load_rules(rules_path)
    analysis = analysis or analyze_task(task)
    metrics = analysis.metrics()
    score = 0
    reasons: list[str] = []
    signal_results: list[SignalResult] = []
    for rule in rules.signals:
        actual = _resolve_metric(metrics, _rule_field(rule))
        passed = _compare(actual, rule)
        points = int(rule["points"]) if passed else int(rule.get("else_points", 0))
        score += points
        reason = _reason(rule, passed, actual)
        if reason:
            reasons.append(reason)
        signal_results.append(
            SignalResult(
                name=_rule_name(rule),
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
        signal_lookup = {signal.name: signal for signal in signal_results}
        for requirement in rules.golden_requirements:
            if isinstance(requirement, str):
                signal = signal_lookup.get(requirement)
                if signal is None or not signal.passed:
                    unmet.append(f"{requirement} requirement not met")
                continue
            if not isinstance(requirement, dict):
                continue
            actual = _resolve_metric(metrics, _rule_field(requirement))
            if not _compare(actual, requirement):
                unmet.append(
                    str(
                        requirement.get("reason")
                        or f"{_rule_field(requirement)} requirement not met"
                    )
                )
        if unmet:
            grade = "GOOD"
            reasons.extend(unmet)
    return GradeResult(
        task_id=task.task_id,
        grade=grade,
        score=score,
        normalized_score=normalize_score(score, rules),
        reasons=reasons,
        metrics=metrics,
        signals=signal_results,
    )


def grade_tasks(tasks: list[AgentTask], rules: RuleSet) -> list[GradeResult]:
    """Grade a list of tasks with one shared ruleset.

    Args:
        tasks: Tasks to grade.
        rules: Ruleset to evaluate against each task.

    Returns:
        list[GradeResult]: Grade results in the same order as ``tasks``.
    """

    analyses = {task.task_id: analyze_task(task) for task in tasks}
    return [grade_task(task, rules, analysis=analyses[task.task_id]) for task in tasks]
