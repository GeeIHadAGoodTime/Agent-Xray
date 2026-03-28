from __future__ import annotations

from .analyzer import TaskAnalysis, analyze_task, analyze_tasks, load_tasks
from .completeness import CompletenessReport, CompletenessWarning, check_completeness
from .comparison import ModelComparisonResult, compare_model_runs, format_model_comparison
from .diagnose import (
    DefaultTargetResolver,
    FixPlanEntry,
    TargetResolver,
    build_fix_plan,
    format_fix_plan_text,
    get_target_resolver,
    register_target_resolver,
)
from .grader import (
    GradeResult,
    RuleSet,
    SignalResult,
    grade_task,
    grade_tasks,
    load_rules,
    normalize_score,
    validate_rules,
)
from .protocols import PromptBuilder, StaticPromptBuilder, StaticToolRegistry, ToolRegistry
from .replay import format_replay_text, replay_fixture
from .root_cause import (
    ROOT_CAUSES,
    ClassificationConfig,
    RootCauseResult,
    classify_failures,
    classify_task,
    summarize_root_causes,
)
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
from .surface import (
    diff_tasks,
    format_reasoning_text,
    format_surface_text,
    format_tree_text,
    reasoning_for_task,
    surface_for_task,
    tree_for_tasks,
)

__all__ = [
    # Schema
    "AgentStep",
    "AgentTask",
    "BrowserContext",
    "ModelContext",
    "ReasoningContext",
    "TaskOutcome",
    "ToolContext",
    # Protocols
    "ToolRegistry",
    "StaticToolRegistry",
    "PromptBuilder",
    "StaticPromptBuilder",
    # Analysis
    "TaskAnalysis",
    "analyze_task",
    "analyze_tasks",
    "load_tasks",
    # Grading
    "RuleSet",
    "GradeResult",
    "SignalResult",
    "grade_task",
    "grade_tasks",
    "load_rules",
    "normalize_score",
    "validate_rules",
    # Completeness
    "CompletenessReport",
    "CompletenessWarning",
    "check_completeness",
    # Root cause
    "ROOT_CAUSES",
    "ClassificationConfig",
    "RootCauseResult",
    "classify_task",
    "classify_failures",
    "summarize_root_causes",
    # Diagnosis / fix plan
    "DefaultTargetResolver",
    "FixPlanEntry",
    "TargetResolver",
    "build_fix_plan",
    "format_fix_plan_text",
    "get_target_resolver",
    "register_target_resolver",
    # Surface / diff / tree
    "surface_for_task",
    "reasoning_for_task",
    "diff_tasks",
    "format_surface_text",
    "format_reasoning_text",
    "format_tree_text",
    "tree_for_tasks",
    # Comparison
    "ModelComparisonResult",
    "compare_model_runs",
    "format_model_comparison",
    # Replay
    "replay_fixture",
    "format_replay_text",
    # Signals
    "SignalDetector",
    "discover_detectors",
    "run_detection",
]

__version__ = "1.2.5"
