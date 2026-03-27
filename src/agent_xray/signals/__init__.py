"""Pluggable signal detection for agent task analysis."""

from __future__ import annotations

import importlib.metadata
import warnings
from typing import Any, Protocol, runtime_checkable

from ..schema import AgentStep, AgentTask


@runtime_checkable
class SignalDetector(Protocol):
    """Protocol for signal detection plugins."""

    name: str

    def detect_step(self, step: AgentStep) -> dict[str, bool]:
        """Analyze a single step and return boolean signals."""
        ...

    def summarize(self, task: AgentTask, step_signals: list[dict[str, bool]]) -> dict[str, Any]:
        """Summarize step-level signals into task-level metrics."""
        ...


def _iter_entry_points() -> list[importlib.metadata.EntryPoint]:
    try:
        entry_points = importlib.metadata.entry_points()
    except Exception as exc:
        warnings.warn(f"Failed to enumerate signal plugins: {exc}", stacklevel=2)
        return []
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group="agent_xray.signals"))
    return list(entry_points.get("agent_xray.signals", []))


def discover_detectors() -> list[SignalDetector]:
    """Discover installed signal detector plugins via entry points."""

    from .coding import CodingDetector
    from .commerce import CommerceDetector
    from .research import ResearchDetector

    detectors: list[SignalDetector] = [
        CommerceDetector(),
        CodingDetector(),
        ResearchDetector(),
    ]

    for entry_point in _iter_entry_points():
        try:
            loaded = entry_point.load()
            instance = loaded() if isinstance(loaded, type) else loaded
            if isinstance(instance, SignalDetector):
                detectors.append(instance)
        except Exception as exc:
            warnings.warn(f"Failed to load plugin {entry_point.name}: {exc}", stacklevel=2)
    return detectors


def run_detection(
    task: AgentTask,
    detectors: list[SignalDetector] | None = None,
) -> dict[str, dict[str, Any]]:
    """Run all detectors on a task and return combined metrics."""

    active_detectors = detectors or discover_detectors()
    results: dict[str, dict[str, Any]] = {}
    for detector in active_detectors:
        step_signals = [detector.detect_step(step) for step in task.sorted_steps]
        results[detector.name] = detector.summarize(task, step_signals)
    return results
