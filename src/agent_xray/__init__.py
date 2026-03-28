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
from .baseline import (
    Baseline,
    OverheadResult,
    PromptHashGroup,
    build_baseline,
    format_overhead_report,
    format_prompt_impact_report,
    generate_naked_prompt,
    group_by_prompt_hash,
    measure_overhead,
)
from .golden import (
    OPTIMIZATION_PROFILES,
    GoldenRank,
    capture_exemplar,
    explain_efficiency_gap,
    find_exemplars,
    format_golden_ranking,
    rank_golden_runs,
)
from .signals import SignalDetector, discover_detectors, run_detection
from .surface import (
    diff_tasks,
    enriched_tree_for_tasks,
    format_diff_summary,
    format_enriched_tree_text,
    format_prompt_diff,
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
    "format_diff_summary",
    "format_prompt_diff",
    "format_enriched_tree_text",
    "tree_for_tasks",
    "enriched_tree_for_tasks",
    # Comparison
    "ModelComparisonResult",
    "compare_model_runs",
    "format_model_comparison",
    # Replay
    "replay_fixture",
    "format_replay_text",
    # Golden ranking
    "GoldenRank",
    "OPTIMIZATION_PROFILES",
    "rank_golden_runs",
    "find_exemplars",
    "explain_efficiency_gap",
    "format_golden_ranking",
    "capture_exemplar",
    # Baseline / overhead
    "Baseline",
    "OverheadResult",
    "PromptHashGroup",
    "generate_naked_prompt",
    "build_baseline",
    "measure_overhead",
    "group_by_prompt_hash",
    "format_overhead_report",
    "format_prompt_impact_report",
    # Signals
    "SignalDetector",
    "discover_detectors",
    "run_detection",
]

__version__ = "1.2.6"
