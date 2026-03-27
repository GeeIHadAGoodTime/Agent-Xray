"""pytest plugin for agent testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from .analyzer import TaskAnalysis, analyze_task
from .grader import grade_task
from .root_cause import classify_task
from .schema import AgentStep, AgentTask


@dataclass(slots=True)
class XrayReport:
    grade: str
    score: int
    reasons: list[str]
    root_cause: str | None
    analysis: TaskAnalysis

    @property
    def unique_tools(self) -> int:
        return len(self.analysis.unique_tools)

    @property
    def error_rate(self) -> float:
        return float(self.analysis.error_rate)


class XrayFixture:
    def __init__(self, rules_path: str | None = None) -> None:
        self.rules_path = rules_path

    def analyze(self, steps: list[AgentStep] | list[dict[str, Any]]) -> XrayReport:
        typed_steps: list[AgentStep] = []
        for step in steps:
            if isinstance(step, dict):
                typed_steps.append(AgentStep.from_dict(step))
            else:
                typed_steps.append(step)
        task = AgentTask.from_steps(typed_steps)
        analysis = analyze_task(task)
        grade = grade_task(task, rules_path=self.rules_path, analysis=analysis)
        cause = classify_task(task, grade, analysis) if grade.grade in {"WEAK", "BROKEN"} else None
        return XrayReport(
            grade=grade.grade,
            score=grade.score,
            reasons=grade.reasons,
            root_cause=cause.root_cause if cause else None,
            analysis=analysis,
        )


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agent-xray")
    group.addoption(
        "--xray-rules",
        action="store",
        default=None,
        help="Rules file path or bundled rules name used by the xray fixture.",
    )


@pytest.fixture
def xray(request: pytest.FixtureRequest) -> XrayFixture:
    """Fixture for analyzing agent steps in tests."""

    rules_path = request.config.getoption("--xray-rules")
    return XrayFixture(rules_path=rules_path)
