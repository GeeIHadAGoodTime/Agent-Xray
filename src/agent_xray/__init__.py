from __future__ import annotations

from .analyzer import TaskAnalysis, analyze_task, analyze_tasks, load_tasks
from .completeness import CompletenessReport, CompletenessWarning, check_completeness
from .comparison import ModelComparisonResult, compare_model_runs, format_model_comparison
from .diagnose import (
    DefaultTargetResolver,
    FIX_TARGETS,
    FixPlanEntry,
    INVESTIGATION_HINTS,
    TargetResolver,
    build_fix_plan,
    format_fix_plan_text,
    get_target_resolver,
    list_all_targets,
    register_target_resolver,
    validate_fix_targets,
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
from .enforce import (
    ChangeRecord,
    ChallengeResult,
    DiffHunk,
    EnforceConfig,
    EnforceReport,
    TestResult,
    build_enforce_report,
    compare_test_results,
    enforce_auto,
    enforce_challenge,
    enforce_check,
    enforce_guard,
    enforce_init,
    enforce_plan,
    enforce_reset,
    enforce_status,
    parse_test_output,
    run_tests,
)
from .enforce_audit import (
    GamingSignal,
    analyze_successful_changes,
    audit_change,
    challenge_iterations,
    classify_change_quality,
    classify_diff_quality,
    detect_assertion_weakening,
    detect_early_return,
    detect_exception_swallowing,
    detect_hardcoded_values,
    detect_import_removal,
    detect_mock_insertion,
    detect_rule_violations,
    detect_special_case_branching,
    detect_test_file_modification,
    quality_distribution,
)
from .enforce_report import (
    format_enforce_json,
    format_enforce_markdown,
    format_enforce_text,
    generate_report,
)
try:
    from .mcp_server import main as mcp_main
    from .mcp_server import server as mcp_server
except ImportError:
    mcp_main = None
    mcp_server = None
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
    "FIX_TARGETS",
    "FixPlanEntry",
    "INVESTIGATION_HINTS",
    "TargetResolver",
    "build_fix_plan",
    "format_fix_plan_text",
    "get_target_resolver",
    "list_all_targets",
    "register_target_resolver",
    "validate_fix_targets",
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
    # Enforce
    "DiffHunk",
    "EnforceConfig",
    "TestResult",
    "ChangeRecord",
    "ChallengeResult",
    "EnforceReport",
    "enforce_init",
    "enforce_check",
    "enforce_status",
    "enforce_challenge",
    "enforce_reset",
    "enforce_plan",
    "enforce_guard",
    "enforce_auto",
    "build_enforce_report",
    "run_tests",
    "parse_test_output",
    "compare_test_results",
    # Enforce audit
    "GamingSignal",
    "audit_change",
    "challenge_iterations",
    "analyze_successful_changes",
    "classify_change_quality",
    "classify_diff_quality",
    "detect_rule_violations",
    "quality_distribution",
    "detect_test_file_modification",
    "detect_hardcoded_values",
    "detect_special_case_branching",
    "detect_mock_insertion",
    "detect_assertion_weakening",
    "detect_exception_swallowing",
    "detect_early_return",
    "detect_import_removal",
    # Enforce report
    "format_enforce_text",
    "format_enforce_json",
    "format_enforce_markdown",
    "generate_report",
    "mcp_server",
    "mcp_main",
]

__version__ = "1.5.0"
