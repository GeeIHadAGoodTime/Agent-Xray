from __future__ import annotations

from .analyzer import TaskAnalysis, analyze_task, analyze_tasks, load_tasks
from .grader import (
    GradeResult,
    RuleSet,
    grade_task,
    grade_tasks,
    load_rules,
    normalize_score,
    validate_rules,
)
from .protocols import PromptBuilder, StaticPromptBuilder, StaticToolRegistry, ToolRegistry
from .root_cause import ClassificationConfig, RootCauseResult, classify_failures, classify_task
from .schema import (
    AgentStep,
    AgentTask,
    BrowserContext,
    ModelContext,
    ReasoningContext,
    TaskOutcome,
    ToolContext,
)
from .signals import SignalDetector, discover_detectors, run_detection
from .surface import reasoning_for_task, surface_for_task

__all__ = [
    "AgentStep",
    "AgentTask",
    "BrowserContext",
    "ModelContext",
    "ReasoningContext",
    "TaskOutcome",
    "ToolContext",
    "ToolRegistry",
    "StaticToolRegistry",
    "PromptBuilder",
    "StaticPromptBuilder",
    "TaskAnalysis",
    "RuleSet",
    "GradeResult",
    "ClassificationConfig",
    "RootCauseResult",
    "SignalDetector",
    "analyze_task",
    "analyze_tasks",
    "load_tasks",
    "grade_task",
    "grade_tasks",
    "load_rules",
    "normalize_score",
    "validate_rules",
    "classify_task",
    "classify_failures",
    "surface_for_task",
    "reasoning_for_task",
    "discover_detectors",
    "run_detection",
]

__version__ = "1.0.4"
