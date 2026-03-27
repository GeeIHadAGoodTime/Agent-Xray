"""Pluggable signal detection for agent task analysis."""

from __future__ import annotations

import importlib.metadata
import warnings
from importlib import import_module
from typing import Any, Protocol, runtime_checkable

from ..schema import AgentStep, AgentTask

StepSignals = dict[str, Any]
SummaryMetrics = dict[str, Any]


@runtime_checkable
class SignalDetector(Protocol):
    """Protocol for signal detection plugins."""

    name: str

    def detect_step(self, step: AgentStep) -> StepSignals:
        """Analyze a single step and return detector-specific step signals."""
        ...

    def summarize(self, task: AgentTask, step_signals: list[StepSignals]) -> SummaryMetrics:
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
    """Discover built-in and plugin signal detectors.

    Returns:
        list[SignalDetector]: Built-in detector instances plus any valid
        third-party detectors registered under the ``agent_xray.signals``
        entry-point group.
    """

    detectors: list[SignalDetector] = [detector_cls() for detector_cls in BUILTIN_DETECTORS]

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
    """Run all detectors on a task and collect their metrics.

    Args:
        task: Task to analyze with detectors.
        detectors: Optional explicit detector instances. When omitted,
            :func:`discover_detectors` is used.

    Returns:
        dict[str, dict[str, Any]]: Detector metrics keyed by detector name.
    """

    active_detectors = detectors if detectors is not None else discover_detectors()
    results: dict[str, dict[str, Any]] = {}
    for detector in active_detectors:
        step_signals = [detector.detect_step(step) for step in task.sorted_steps]
        results[detector.name] = detector.summarize(task, step_signals)
    return results


def _load_builtin_detector(module_name: str, class_name: str) -> type[SignalDetector] | None:
    try:
        module = import_module(f"{__name__}.{module_name}")
    except ModuleNotFoundError as exc:
        warnings.warn(f"Skipping built-in detector {module_name}: {exc}", stacklevel=2)
        return None
    detector = getattr(module, class_name, None)
    if detector is None:
        warnings.warn(
            f"Skipping built-in detector {module_name}: missing class {class_name}",
            stacklevel=2,
        )
        return None
    return detector  # type: ignore[no-any-return]


BUILTIN_DETECTORS: tuple[type[SignalDetector], ...] = tuple(
    detector
    for detector in (
        _load_builtin_detector("commerce", "CommerceDetector"),
        _load_builtin_detector("coding", "CodingDetector"),
        _load_builtin_detector("research", "ResearchDetector"),
        _load_builtin_detector("multi_agent", "MultiAgentDetector"),
        _load_builtin_detector("memory", "MemoryDetector"),
        _load_builtin_detector("planning", "PlanningDetector"),
    )
    if detector is not None
)

__all__ = [
    "BUILTIN_DETECTORS",
    "SignalDetector",
    "StepSignals",
    "SummaryMetrics",
    "discover_detectors",
    "run_detection",
]
