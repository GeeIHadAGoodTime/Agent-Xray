"""Enforce the principle: the project IS the documentation.

Every user-visible feature -- MCP tools, CLI commands, signal detectors,
public API exports, and entry points -- must have corresponding documentation
in README.md.  An undocumented feature is not a feature; it is an easter egg
that nobody finds.

This test suite reads the source of truth (source files, pyproject.toml) and
checks that README.md mentions each discoverable feature at least once.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths (relative to *this* test file, so the suite works from any cwd)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_README = _PROJECT_ROOT / "README.md"
_SRC = _PROJECT_ROOT / "src" / "agent_xray"
_MCP_SERVER = _SRC / "mcp_server.py"
_CLI = _SRC / "cli.py"
_INIT = _SRC / "__init__.py"
_PYPROJECT = _PROJECT_ROOT / "pyproject.toml"


@pytest.fixture(scope="module")
def readme_text() -> str:
    return _README.read_text(encoding="utf-8")


# ===================================================================
# 1. MCP tools
# ===================================================================

MCP_TOOLS = [
    "enforce_init",
    "enforce_check",
    "enforce_diff",
    "enforce_plan",
    "enforce_guard",
    "enforce_status",
    "enforce_challenge",
    "enforce_reset",
    "enforce_report",
    "analyze",
    "grade",
    "root_cause",
    "completeness",
    "surface_task",
    "search_tasks",
    "diagnose",
    "compare_runs",
    "report",
    "diff_tasks",
    "reasoning",
    "tree",
    "golden_rank",
    "golden_compare",
    "task_bank_validate",
    "task_bank_list",
    "flywheel",
    "capture_task",
    "pricing_show",
    "replay",
    "validate_targets",
    "rules_list",
    "rules_show",
    "rules_init",
    "baseline_capture",
    "baseline_list",
    "golden_best",
    "golden_profiles",
    "pricing_list",
    "baseline_generate",
    "task_bank_show",
    "format_detect",
    "triage",
]


@pytest.mark.parametrize("tool_name", MCP_TOOLS)
def test_mcp_tool_documented(tool_name: str, readme_text: str) -> None:
    """Every @server.tool() function in mcp_server.py must appear in README.md."""
    # MCP tool names use underscores; the README table lists them literally.
    assert tool_name in readme_text, (
        f"MCP tool '{tool_name}' is not documented in README.md "
        "-- every user-visible feature must be documented"
    )


def test_mcp_tools_exhaustive() -> None:
    """MCP_TOOLS list must match @server.tool() definitions in mcp_server.py."""
    source = _MCP_SERVER.read_text(encoding="utf-8")
    # Find all function names decorated with @server.tool()
    pattern = re.compile(r"@server\.tool\(\)\s+def\s+(\w+)\(")
    found = set(pattern.findall(source))
    expected = set(MCP_TOOLS)
    assert found == expected, (
        f"MCP_TOOLS list is out of sync with mcp_server.py. "
        f"Missing from test: {found - expected}. "
        f"Extra in test: {expected - found}."
    )


# ===================================================================
# 2. CLI top-level commands
# ===================================================================

CLI_TOP_LEVEL_COMMANDS = [
    "analyze",
    "surface",
    "reasoning",
    "diff",
    "grade",
    "root-cause",
    "tree",
    "search",
    "compare",
    "completeness",
    "diagnose",
    "validate-targets",
    "report",
    "watch",
    "quickstart",
    "tui",
    "capture",
    "replay",
    "flywheel",
    "record",
]


@pytest.mark.parametrize("cmd", CLI_TOP_LEVEL_COMMANDS)
def test_cli_command_documented(cmd: str, readme_text: str) -> None:
    """Every top-level CLI command must appear in README.md."""
    # Commands like "root-cause" appear as "agent-xray root-cause" in the README.
    search = f"agent-xray {cmd}"
    assert search in readme_text, (
        f"CLI command '{cmd}' (as '{search}') is not documented in README.md "
        "-- every user-visible feature must be documented"
    )


def test_cli_top_level_commands_exhaustive() -> None:
    """CLI_TOP_LEVEL_COMMANDS must match _add_subparser calls in cli.py."""
    source = _CLI.read_text(encoding="utf-8")
    # Match _add_subparser(sub, "name", ...) -- the second argument is the
    # command name as a quoted string.  Also handle the loop variant that
    # registers surface/reasoning.
    pattern = re.compile(
        r'_add_subparser\(\s*sub,\s*"([^"]+)"',
    )
    found = set(pattern.findall(source))
    # The loop at ~line 2463 registers "surface" and "reasoning" via a
    # tuple, not a direct _add_subparser(sub, ...) with a literal.  The
    # regex may or may not catch those (the variable `name` is used).
    # Explicitly confirm they are included by also scanning for the tuple.
    loop_pattern = re.compile(
        r'\(\s*"(\w[\w-]*)".*cmd_\w+.*agent-xray',
    )
    found |= set(loop_pattern.findall(source))

    # Subcommand groups (enforce, rules, pricing, golden, baseline,
    # task-bank) are registered the same way but are NOT top-level --
    # they are groups with their own subcommands.
    subcommand_groups = {
        "enforce",
        "rules",
        "pricing",
        "golden",
        "baseline",
        "task-bank",
    }
    top_level_from_source = found - subcommand_groups
    expected = set(CLI_TOP_LEVEL_COMMANDS)

    assert top_level_from_source == expected, (
        f"CLI_TOP_LEVEL_COMMANDS is out of sync with cli.py. "
        f"Missing from test: {top_level_from_source - expected}. "
        f"Extra in test: {expected - top_level_from_source}."
    )


# ===================================================================
# 3. CLI subcommand groups
# ===================================================================

CLI_SUBCOMMAND_GROUPS = [
    "enforce",
    "rules",
    "pricing",
    "golden",
    "baseline",
    "task-bank",
]


@pytest.mark.parametrize("group", CLI_SUBCOMMAND_GROUPS)
def test_cli_subcommand_group_documented(group: str, readme_text: str) -> None:
    """Every CLI subcommand group must be mentioned in README.md."""
    # Groups appear as "agent-xray <group>" in CLI reference tables.
    search = f"agent-xray {group}"
    assert search in readme_text, (
        f"CLI subcommand group '{group}' (as '{search}') is not documented "
        "in README.md -- every user-visible feature must be documented"
    )


# ===================================================================
# 4. Signal detectors
# ===================================================================

SIGNAL_DETECTORS = [
    "CommerceDetector",
    "CodingDetector",
    "ResearchDetector",
    "PlanningDetector",
    "MemoryDetector",
    "MultiAgentDetector",
]


@pytest.mark.parametrize("detector", SIGNAL_DETECTORS)
def test_signal_detector_documented(detector: str, readme_text: str) -> None:
    """Every built-in signal detector must appear in README.md."""
    assert detector in readme_text, (
        f"Signal detector '{detector}' is not documented in README.md "
        "-- every user-visible feature must be documented"
    )


def test_signal_detectors_exhaustive() -> None:
    """SIGNAL_DETECTORS must match the detector classes in signals/."""
    signals_dir = _SRC / "signals"
    # Scan all .py files in signals/ for "class XxxDetector:" patterns,
    # excluding the SignalDetector protocol itself.
    found: set[str] = set()
    for py_file in signals_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        for match in re.finditer(r"^class\s+(\w+Detector)\b", source, re.MULTILINE):
            name = match.group(1)
            if name != "SignalDetector":
                found.add(name)
    expected = set(SIGNAL_DETECTORS)
    assert found == expected, (
        f"SIGNAL_DETECTORS is out of sync with signals/. "
        f"Missing from test: {found - expected}. "
        f"Extra in test: {expected - found}."
    )


# ===================================================================
# 5. Public API exports (__all__)
# ===================================================================

# Internal helpers that are intentionally not documented as user-facing
# features in README -- they exist in __all__ for programmatic import
# convenience but are not independent "features" a user would look up.
_INTERNAL_EXPORTS = {
    "mcp_server",
    "mcp_main",
}

# Implementation-detail types, low-level helpers, and aliases that the
# README documents indirectly (e.g. via the module they belong to, or via
# their parent concept) rather than by exact symbol name.  These are still
# public API, but the README covers them at the conceptual level.
_IMPLICITLY_DOCUMENTED = {
    # Types documented via their parent concepts (e.g. "Root Cause
    # Classification" section, "Grading System" section, etc.)
    "ClassificationConfig",
    "RootCauseResult",
    "ROOT_CAUSES",
    "SignalResult",
    "RuleSet",
    "GradeResult",
    "CompletenessReport",
    "CompletenessWarning",
    "ModelComparisonResult",
    "GoldenRank",
    "OPTIMIZATION_PROFILES",
    "OverheadResult",
    "PromptHashGroup",
    "Baseline",
    "TaskBankEntry",
    "TaskBank",
    "DiffHunk",
    "TestResult",
    "ChangeRecord",
    "ChallengeResult",
    "EnforceReport",
    "GamingSignal",
    "FixPlanEntry",
    "FIX_TARGETS",
    "INVESTIGATION_HINTS",
    "TargetResolver",
    # Types documented via their parent diff/comparison concepts
    "SimilarityBreakdown",
    # Schema types documented via AgentStep/AgentTask sections
    "BrowserContext",
    "ModelContext",
    "ReasoningContext",
    "TaskOutcome",
    "ToolContext",
    # Lower-level functions documented via their parent module/concept
    "analyze_task",
    "analyze_tasks",
    "load_tasks",
    "grade_tasks",  # plural form; README shows grade_task (singular) in example
    "classify_failures",
    "summarize_root_causes",
    "normalize_score",
    "validate_rules",
    "get_target_resolver",
    "reasoning_for_task",
    "diff_tasks",
    "tree_for_tasks",
    "enriched_tree_for_tasks",
    "compare_model_runs",
    "format_model_comparison",
    "replay_fixture",
    "format_replay_text",
    "format_golden_ranking",
    "format_overhead_report",
    "format_prompt_impact_report",
    "format_surface_text",
    "format_reasoning_text",
    "format_tree_text",
    "format_diff_summary",
    "format_prompt_diff",
    "format_enriched_tree_text",
    "run_detection",
    "run_tests",
    "parse_test_output",
    "compare_test_results",
    "enforce_auto",
    "enforce_diff",
    "enforce_status",
    "audit_change",
    "challenge_iterations",
    "analyze_successful_changes",
    "classify_change_quality",
    "classify_diff_quality",
    "detect_rule_violations",
    "quality_distribution",
    "format_enforce_json",
    "format_enforce_markdown",
    "generate_report",
}


def _parse_all_exports() -> list[str]:
    """Parse __all__ from __init__.py without importing the package."""
    source = _INIT.read_text(encoding="utf-8")
    # Find the __all__ = [...] block
    match = re.search(
        r"__all__\s*=\s*\[([^\]]+)\]",
        source,
        re.DOTALL,
    )
    assert match, "__all__ not found in __init__.py"
    block = match.group(1)
    return re.findall(r'"(\w+)"', block)


# Exported names that must appear literally in README.  This is the
# subset of __all__ that represents a user-facing concept worth calling
# out individually.
_FEATURED_EXPORTS: list[str] = [
    name
    for name in _parse_all_exports()
    if name not in _INTERNAL_EXPORTS and name not in _IMPLICITLY_DOCUMENTED
]


@pytest.mark.parametrize("name", _FEATURED_EXPORTS)
def test_public_api_export_documented(name: str, readme_text: str) -> None:
    """Every featured public API export must appear at least once in README.md.

    Names in _INTERNAL_EXPORTS and _IMPLICITLY_DOCUMENTED are excluded --
    the former are wiring helpers, the latter are covered by their parent
    concept's documentation.
    """
    assert name in readme_text, (
        f"Public API export '{name}' (from __all__ in __init__.py) is not "
        "documented in README.md -- every user-visible feature must be documented"
    )


def test_no_unexpected_undocumented_exports(readme_text: str) -> None:
    """All __all__ exports must be either in README or in an exclusion list.

    This catches new exports that are added to __all__ without updating
    either README.md or the exclusion lists in this test.
    """
    all_exports = set(_parse_all_exports())
    accounted_for = _INTERNAL_EXPORTS | _IMPLICITLY_DOCUMENTED
    for name in all_exports:
        if name in accounted_for:
            continue
        assert name in readme_text, (
            f"Public API export '{name}' is in __all__ but appears in neither "
            "README.md nor the _IMPLICITLY_DOCUMENTED exclusion list in "
            "test_readme_coverage.py.  Either document it in README.md or "
            "add it to _IMPLICITLY_DOCUMENTED with a justification comment."
        )


# ===================================================================
# 6. Entry points
# ===================================================================

def test_entry_point_agent_xray_documented(readme_text: str) -> None:
    """The 'agent-xray' CLI entry point must be documented in README.md."""
    assert "agent-xray" in readme_text, (
        "Entry point 'agent-xray' is not documented in README.md "
        "-- every user-visible feature must be documented"
    )


def test_entry_point_agent_xray_mcp_documented(readme_text: str) -> None:
    """The MCP server entry point must be documented in README.md.

    The entry point can be invoked as either 'agent-xray-mcp' (the
    pyproject.toml script name) or 'python -m agent_xray.mcp_server'.
    We accept either form.
    """
    has_script_name = "agent-xray-mcp" in readme_text
    has_module_invocation = "agent_xray.mcp_server" in readme_text
    assert has_script_name or has_module_invocation, (
        "MCP server entry point is not documented in README.md -- "
        "expected either 'agent-xray-mcp' or 'python -m agent_xray.mcp_server'"
    )


def test_entry_points_match_pyproject() -> None:
    """Entry points in pyproject.toml must be covered by our test assertions."""
    source = _PYPROJECT.read_text(encoding="utf-8")
    # Look for [project.scripts] entries
    in_scripts = False
    entry_points: list[str] = []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped == "[project.scripts]":
            in_scripts = True
            continue
        if in_scripts:
            if stripped.startswith("["):
                break
            if "=" in stripped:
                name = stripped.split("=")[0].strip()
                entry_points.append(name)

    expected = {"agent-xray", "agent-xray-mcp"}
    found = set(entry_points)
    assert found == expected, (
        f"pyproject.toml entry points changed. Found: {found}, expected: {expected}. "
        "Update the entry-point tests in test_readme_coverage.py."
    )
