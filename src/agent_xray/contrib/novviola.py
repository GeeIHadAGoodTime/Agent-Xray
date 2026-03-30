"""NOVVIOLA-specific TargetResolver plugin for agent-xray.

WARNING: MANUAL MAINTENANCE REQUIRED
-------------------------------------
This plugin returns PROJECT-SPECIFIC FILE PATHS, not generic concepts.  Unlike
the default resolver (which returns codebase-agnostic search terms), this
resolver maps root causes to exact NOVVIOLA files.  These paths go stale when
the codebase refactors.  The NOVVIOLA team must keep NOVVIOLA_FIX_TARGETS and
PROMPT_BUG_PATTERNS up to date manually, or run ``agent-xray validate-targets``
to detect drift.

This is opt-in project-specific knowledge.  The default agent-xray resolver
returns conceptual investigation hints that work for any codebase without
maintenance.  Only register this plugin if you want NOVVIOLA file paths in
your diagnose output and accept the maintenance burden.

Usage:
    from agent_xray.contrib.novviola import register
    register()  # Makes NovviolaTargetResolver the active default

Or import the resolver directly:
    from agent_xray.contrib.novviola import NovviolaTargetResolver
    resolver = NovviolaTargetResolver()
    targets = resolver.resolve("routing_bug", ["3 step(s) exposed zero tools"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agent_xray.diagnose import register_target_resolver

# ======================================================================
# ROOT CAUSE -> NOVVIOLA FILE PATH MAPPING
# ======================================================================
# Every root cause from agent-xray's ROOT_CAUSES maps to concrete files
# in the NOVVIOLA codebase.  These paths require manual maintenance --
# run ``agent-xray validate-targets --project-root .`` to check for drift.

NOVVIOLA_FIX_TARGETS: dict[str, list[str]] = {
    "routing_bug": [
        "intent/pipeline.py",
        "intent/ai_controller.py",
        "mcp_hub/approval_bridge.py",
        "mcp_hub/client_hub.py",
    ],
    "approval_block": [
        "mcp_hub/approval_bridge.py",
    ],
    "spin": [
        "intent/spin_detector.py",
    ],
    "environment_drift": [
        "mcp_servers/browser/server.py",
    ],
    "tool_bug": [
        "mcp_servers/browser/server.py",
        "mcp_servers/core_tools/server.py",
        "intent/tools/",
    ],
    "tool_selection_bug": [
        "services/llm/prompts/builder.py",
        "services/llm/prompts/sections/minimal_core.py",
        "intent/ai_controller.py",
        "mcp_hub/client_hub.py",
    ],
    "early_abort": [
        "intent/agent_executor.py",
    ],
    "stuck_loop": [
        "services/llm/prompts/sections/minimal_core.py",
        "services/llm/prompts/builder.py",
        "mcp_servers/browser/server.py",
    ],
    "reasoning_bug": [
        "services/llm/prompts/sections/minimal_core.py",
        "services/llm/prompts/builder.py",
    ],
    "prompt_bug": [
        # Populated dynamically via PROMPT_BUG_PATTERNS; fallback is builder.py
        "services/llm/prompts/builder.py",
    ],
    "model_limit": [
        # No direct code fix -- task decomposition or model upgrade
    ],
    "memory_overload": [
        "intent/agent_executor.py",
        "services/llm/prompts/builder.py",
    ],
    "delegation_failure": [
        "intent/agent_executor.py",
        "mcp_servers/core_tools/server.py",
    ],
    "test_failure_loop": [
        "intent/agent_executor.py",
    ],
    "tool_rejection_mismatch": [
        "mcp_hub/approval_bridge.py",
        "mcp_hub/client_hub.py",
    ],
    "insufficient_sources": [
        "services/llm/prompts/sections/minimal_core.py",
        "mcp_servers/core_tools/server.py",
    ],
}

# ======================================================================
# PROMPT BUG PATTERN -> SPECIFIC PROMPT SECTION FILE
# ======================================================================
# Maps evidence strings to the exact prompt section file that needs editing.
# Sourced from diagnose_and_fix.py PROMPT_BUG_PATTERNS.

PROMPT_BUG_PATTERNS: list[tuple[str, str, str]] = [
    # (regex_pattern, novviola_file_path, fix_description)
    (
        r"web_search.*browser|search.*instead of browse",
        "services/llm/prompts/sections/minimal_core.py",
        "Add rule: if a live browser page already exists, inspect it with "
        "browser tools before starting a fresh web_search.",
    ),
    (
        r"hallucinated|unknown.tool",
        "mcp_hub/client_hub.py",
        "Make the tool surface reflect the actual runtime allowlist and "
        "defer/discover flow instead of advertising everything.",
    ),
    (
        r"stuck on one page|same page.*\d+ steps",
        "mcp_servers/browser/server.py",
        "Make duplicate-page hints specific to the current page and use "
        "the actual tool names the model can call.",
    ),
    (
        r"only \d+ unique tool",
        "intent/ai_controller.py",
        "Reduce over-constrained tool surfaces and preserve the real "
        "in-scope tool diversity for the task.",
    ),
    (
        r"payment|checkout.*never filled|reached checkout.*no fill",
        "services/llm/prompts/sections/minimal_core.py",
        "Clarify that the agent stops at visible card fields and emits "
        "PAYMENT_GATE instead of filling sensitive payment data.",
    ),
    (
        r"ignored payment gate",
        "services/llm/prompts/sections/minimal_core.py",
        "Keep the payment gate close to end of core prompt and ensure "
        "runtime only triggers it on visible card-entry fields.",
    ),
    (
        r"tried.*different.*approach|backtrack|going back",
        "services/llm/prompts/sections/minimal_core.py",
        "Strengthen planning strategy guidance to avoid flip-flopping.",
    ),
    (
        r"no.*result|empty.*response|returned nothing",
        "mcp_servers/core_tools/server.py",
        "Add result validation and retry guidance for empty tool returns.",
    ),
    (
        r"already.*tried|tried.*before|same.*action.*again",
        "mcp_servers/browser/server.py",
        "Add progress memory instructions so the agent does not repeat "
        "failed approaches.",
    ),
    (
        r"too many.*steps|running out|context.*full",
        "services/llm/prompts/sections/minimal_core.py",
        "Improve task decomposition guidance when resource limits loom.",
    ),
    (
        r"not sure which|multiple.*options|could.*either",
        "mcp_servers/core_tools/server.py",
        "Sharpen tool descriptions and priority ordering to reduce "
        "selection uncertainty.",
    ),
    (
        r"page.*changed|unexpected.*layout|different.*from",
        "mcp_servers/browser/server.py",
        "Update selector strategies for layout mismatches.",
    ),
    (
        r"frustration context injected",
        "services/llm/prompts/builder.py",
        "Review frustration context injection path -- may alter model behavior.",
    ),
    (
        r"delivery address.*fill failed",
        "mcp_servers/browser/server.py",
        "Ensure delivery address context is used by browser fill logic.",
    ),
]

# Prompt section name -> NOVVIOLA file mapping (for prompt_section= evidence)
# After prompt consolidation (2026-03-27), most sections were merged into
# minimal_core.py. Only response_format.py remains as a separate file.
PROMPT_SECTION_FILE_MAP: dict[str, str] = {
    "research": "services/llm/prompts/sections/minimal_core.py",
    "tools": "mcp_servers/core_tools/server.py",
    "browser": "mcp_servers/browser/server.py",
    "payment": "services/llm/prompts/sections/minimal_core.py",
    "planning": "services/llm/prompts/sections/minimal_core.py",
    "delegation": "services/llm/prompts/sections/minimal_core.py",
    "response_format": "services/llm/prompts/sections/response_format.py",
    "proactive": "services/llm/prompts/sections/minimal_core.py",
    "minimal_core": "services/llm/prompts/sections/minimal_core.py",
    "music": "services/llm/prompts/sections/minimal_core.py",
}


def _match_prompt_bug_pattern(evidence: list[str]) -> tuple[str, str] | None:
    """Match evidence text against PROMPT_BUG_PATTERNS.

    Returns (file_path, fix_description) for the first matching pattern,
    or None if no pattern matches.
    """
    evidence_text = " ".join(evidence).lower()
    for pattern, file_path, fix_desc in PROMPT_BUG_PATTERNS:
        if re.search(pattern, evidence_text, re.IGNORECASE):
            return file_path, fix_desc
    return None


def _extract_prompt_section_files(evidence: list[str]) -> list[str]:
    """Extract NOVVIOLA file paths from prompt_section= evidence entries."""
    files: list[str] = []
    for item in evidence:
        if not item.startswith("prompt_section="):
            continue
        section = item.split("=", 1)[1].split(":", 1)[0].strip()
        file_path = PROMPT_SECTION_FILE_MAP.get(section)
        if file_path and file_path not in files:
            files.append(file_path)
    return files


class NovviolaTargetResolver:
    """Resolve agent-xray root causes to NOVVIOLA-specific file paths.

    .. warning::

       This resolver returns **file paths** that require manual maintenance.
       Unlike the default resolver (which returns codebase-agnostic concepts),
       these paths go stale when the NOVVIOLA codebase refactors.  Run
       ``agent-xray validate-targets --project-root .`` regularly to detect
       drift.

    Implements the ``TargetResolver`` protocol from ``agent_xray.diagnose``.
    Maps every root cause to exact file paths in the NOVVIOLA codebase,
    with pattern-specific targeting for prompt_bug classifications.
    """

    def resolve(self, root_cause: str, evidence: list[str]) -> list[str]:
        """Return NOVVIOLA file paths for the given root cause and evidence.

        For prompt_bug, this does two levels of refinement:
        1. Checks prompt_section= evidence entries to map to specific section files.
        2. Matches evidence text against PROMPT_BUG_PATTERNS for pattern-specific files.
        Falls back to the generic file list from NOVVIOLA_FIX_TARGETS.
        """
        targets = list(NOVVIOLA_FIX_TARGETS.get(root_cause, []))

        if root_cause != "prompt_bug":
            return targets if targets else ["(no direct code fix)"]

        # For prompt_bug, try to narrow to specific files
        refined: list[str] = []

        # Level 1: prompt_section= evidence -> specific section file
        section_files = _extract_prompt_section_files(evidence)
        for f in section_files:
            if f not in refined:
                refined.append(f)

        # Level 2: pattern matching against evidence text
        match = _match_prompt_bug_pattern(evidence)
        if match:
            file_path, _fix_desc = match
            if file_path not in refined:
                refined.append(file_path)

        # Always include the fallback builder.py at the end
        if refined:
            for f in targets:
                if f not in refined:
                    refined.append(f)
            return refined

        return targets


# ======================================================================
# NOVVIOLA-SPECIFIC VERIFY COMMANDS
# ======================================================================

NOVVIOLA_VERIFY_COMMANDS: dict[str, str] = {
    "routing_bug": (
        'grep "tools_available_count" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool | grep -E 'tools_available_count|task_id'"
    ),
    "approval_block": (
        "grep '\"error\".*not approved' logs/structured/agent-steps-*.jsonl"
    ),
    "spin": (
        'grep "spin_intervention" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool"
    ),
    "environment_drift": (
        "grep -E '\"error\".*timeout|click_fail|not_found' "
        "logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl | tail -20"
    ),
    "tool_bug": (
        'grep "\"error\"" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool | grep -E 'tool_name|error'"
    ),
    "tool_selection_bug": (
        'grep "tool_name" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool | grep -E 'tool_name|tools_available_count'"
    ),
    "early_abort": (
        "curl -s http://localhost:8756/api/status | python -m json.tool"
    ),
    "stuck_loop": (
        'grep "page_url" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool | grep page_url | sort | uniq -c | sort -rn | head -10"
    ),
    "reasoning_bug": (
        'grep "llm_reasoning" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| tail -5 | python -m json.tool"
    ),
    "prompt_bug": (
        'grep "llm_reasoning" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| tail -10 | python -m json.tool"
    ),
    "model_limit": (
        "agent-xray analyze logs/structured/ --format generic --json"
    ),
    "memory_overload": (
        'grep "context_usage_pct" logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl '
        "| python -m json.tool | grep context_usage_pct | sort -t: -k2 -rn | head -10"
    ),
    "delegation_failure": (
        "grep -E 'spawn_subtask|delegate' "
        "logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl "
        "| python -m json.tool | grep -E 'tool_name|error'"
    ),
    "test_failure_loop": (
        "grep -E 'pytest|test_run' "
        "logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl "
        "| python -m json.tool | grep -E 'tool_name|error|tool_result' | tail -20"
    ),
    "tool_rejection_mismatch": (
        "grep -E 'rejected|focused_set' "
        "logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl "
        "| python -m json.tool | grep -E 'rejected|focused_set' | head -20"
    ),
    "insufficient_sources": (
        "grep -E 'web_search|browser_navigate' "
        "logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl "
        "| python -m json.tool | grep tool_name | sort | uniq -c | sort -rn"
    ),
}


class NovviolaVerifyCommands:
    """Provide NOVVIOLA-specific verification commands for each root cause.

    These commands reference NOVVIOLA step logs, curl endpoints, and analysis
    scripts -- the actual diagnostic tools used in production.
    """

    def get(self, root_cause: str, task_id: str = "") -> str:
        """Return a verify command for the given root cause.

        Args:
            root_cause: The root-cause label (e.g., "routing_bug").
            task_id: Optional task id for task-specific commands.

        Returns:
            A shell command string for verifying the root cause in NOVVIOLA.
        """
        base = NOVVIOLA_VERIFY_COMMANDS.get(root_cause, "")
        if not base:
            return (
                "tail -20 logs/structured/agent-steps-$(date -u +%Y%m%d).jsonl "
                "| python -m json.tool"
            )
        if task_id:
            return f'grep "{task_id}" logs/structured/agent-steps-*.jsonl | python -m json.tool'
        return base


def register() -> None:
    """Register the NOVVIOLA target resolver as the active default.

    Call this once at startup (e.g., in a conftest or agent bootstrap) to
    make all ``build_fix_plan`` calls use NOVVIOLA file paths automatically.
    """
    register_target_resolver(
        "novviola",
        NovviolaTargetResolver(),
        make_default=True,
    )
